"""State custody and explicit-override precedence for component references."""

from dataclasses import replace
from hashlib import sha256

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import (
    build_default_component_library,
    build_default_restraint_library,
)
from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.idealized import IdealizedComponent
from protrepair.chemistry.restraint.library import RestraintLibrary
from protrepair.chemistry.restraint.monomer import component_reference_from_monomer
from protrepair.chemistry.restraint.reference import ComponentRestraintReference
from protrepair.chemistry.restraint.structure import StructureRestraints
from protrepair.chemistry.restraint.template import BondRestraintTarget
from protrepair.diagnostics.geometry import (
    HeavyGeometryPolicy,
    bond_length_tolerance_angstrom,
    detect_heavy_geometry,
)
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    AtomTopology,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.continuous import rdkit as native
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationConfig,
    ContinuousRelaxationForceField,
)
from protrepair.transformer.local import (
    DirectRegionTransformationSpec,
    LocalScopeSpec,
    profile_transform_local_region,
)

RESIDUE = ResidueId("A", 5)
BACKBONE = frozenset({"N", "C", "O", "OXT"})


def _component(component_id: str = "MSE") -> IdealizedComponent:
    component = (
        build_default_component_library().require(component_id).reference_component
    )
    assert component is not None
    return component


def _reference(component_id: str = "MSE") -> ComponentRestraintReference:
    component = _component(component_id)
    template = build_default_restraint_library().require(component_id)
    old = template.bond_target("CB", "CG")
    assert old is not None
    return ComponentRestraintReference(
        component=component,
        atom_names=frozenset(component.heavy_atom_names()) - BACKBONE,
        boundary_atom_names=frozenset({"CA"}),
        bond_targets=(
            replace(old, target_distance_angstrom=old.target_distance_angstrom + 0.02),
        ),
        angle_targets=(),
        source="test:synthetic-target-on-bundled-graph",
    )


def _structure(
    *,
    retained: bool = False,
    isotope: str = "H",
    omit: tuple[str, ...] = (),
    renamed_h: bool = False,
    component_id: str = "MSE",
) -> ProteinStructure:
    component = _component(component_id)
    present = [
        atom
        for atom in component.atoms
        if atom.atom_name not in omit
        and (
            retained
            or atom.atom_name != "OXT"
            and not (
                atom.is_hydrogen()
                and component.hydrogen_anchor_atom_name(atom.atom_name) in {"N", "OXT"}
            )
        )
    ]
    names = {
        atom.atom_name: f"Q{index}"
        if atom.is_hydrogen() and renamed_h
        else atom.atom_name
        for index, atom in enumerate(present)
    }
    residue = residue_payload(
        component_id=component_id,
        residue_id=RESIDUE,
        atoms=tuple(
            atom_payload(
                names[atom.atom_name],
                isotope if atom.is_hydrogen() else atom.element,
                Vec3(*(atom.ideal_position or (0, 0, 0))),
                formal_charge=atom.formal_charge,
            )
            for atom in present
        ),
    )
    structure = build_structure(
        chains=() if retained else (chain_payload("A", (residue,)),),
        ligands=(residue,) if retained else (),
        source_format=FileFormat.MMCIF,
    )
    constitution = structure.constitution
    bonds = tuple(
        TopologyBond(
            constitution.atom_index(AtomRef(RESIDUE, names[b.atom_name_1])),
            constitution.atom_index(AtomRef(RESIDUE, names[b.atom_name_2])),
            b.order,
            b.aromatic,
        )
        for b in component.bonds
        if b.atom_name_1 in names and b.atom_name_2 in names
    )
    return ProteinStructure.from_payload(
        constitution=constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=bonds,
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def _bound(structure: ProteinStructure, library: RestraintLibrary):
    result = StructureRestraints(
        structure, build_default_component_library(), library
    ).for_residue(RESIDUE)
    assert result is not None
    return result


@pytest.mark.parametrize("retained", (False, True))
@pytest.mark.parametrize("isotope", ("H", "D", "T"))
@pytest.mark.parametrize("renamed_h", (False, True))
def test_matching_current_state_selects_reference_without_mutation(
    retained, isotope, renamed_h
):
    structure = _structure(retained=retained, isotope=isotope, renamed_h=renamed_h)
    reference = _reference()
    library = build_default_restraint_library().with_component_reference(reference)
    before = structure
    result = _bound(structure, library)
    assert result.bond_target("CB", "CG") == reference.bond_targets[0]
    assert structure is before
    fallback = library.require("MSE")
    assert len(result.bond_targets) == len(fallback.bond_targets)
    assert result.angle_targets == fallback.angle_targets
    assert result.chirality_targets == fallback.chirality_targets
    assert result.plane_targets == fallback.plane_targets


@pytest.mark.parametrize("component_id", ("MSE", "NLE"))
def test_reference_rule_is_not_a_phosphate_component_recipe(component_id):
    reference = _reference(component_id)
    library = build_default_restraint_library().with_component_reference(reference)
    assert (
        _bound(_structure(component_id=component_id), library).bond_target("CB", "CG")
        == reference.bond_targets[0]
    )


@pytest.mark.parametrize(
    "omit",
    (("HB2",), ("CG",), tuple(a.atom_name for a in _component().hydrogen_atoms())),
)
def test_missing_h_or_scaffold_keeps_static_targets(omit):
    reference = _reference()
    library = build_default_restraint_library().with_component_reference(reference)
    assert _bound(_structure(omit=omit), library) is library.require("MSE")


@pytest.mark.parametrize(
    "order,relationship",
    (
        (None, BondRelationshipType.COVALENT),
        (1, BondRelationshipType.UNKNOWN),
        (2, BondRelationshipType.COVALENT),
    ),
)
def test_unresolved_or_incompatible_heavy_bond_keeps_static_targets(
    order, relationship
):
    structure = _structure()
    constitution = structure.constitution
    pair = {constitution.atom_index(AtomRef(RESIDUE, n)) for n in ("CB", "CG")}
    structure = ProteinStructure.from_payload(
        constitution=constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=tuple(
                replace(b, order=order, relationship_type=relationship)
                if set(b.endpoint_pair()) == pair
                else b
                for b in structure.topology.bonds
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )
    library = build_default_restraint_library().with_component_reference(_reference())
    assert _bound(structure, library) is library.require("MSE")


def test_explicit_template_replacement_and_merge_remain_authoritative():
    reference = _reference()
    library = build_default_restraint_library().with_component_reference(reference)
    custom = replace(
        library.require("MSE"), bond_targets=(BondRestraintTarget("CB", "CG", 1.7),)
    )
    direct = library.with_template(custom)
    merged = library.merged_with(RestraintLibrary(templates={"MSE": custom}))
    for result in (direct, merged):
        assert _bound(_structure(), result) is custom
        assert "MSE" not in result.component_references
    assert (
        _bound(_structure(), RestraintLibrary().merged_with(library)).bond_target(
            "CB", "CG"
        )
        == reference.bond_targets[0]
    )


def test_reference_cannot_add_or_duplicate_state_and_target_identities():
    reference = _reference()
    library = build_default_restraint_library().with_component_reference(reference)
    with pytest.raises(ValueError, match="must not repeat"):
        library.with_component_reference(reference)
    with pytest.raises(ValueError, match="connectivity"):
        library.with_component_reference(
            replace(reference, bond_targets=(BondRestraintTarget("CB", "CE", 1.5),))
        )
    with pytest.raises(ValueError, match="writable"):
        replace(reference, bond_targets=(BondRestraintTarget("CA", "CB", 1.5),))


def test_atom_and_bond_order_do_not_change_state_identity():
    reference = _reference()
    reordered = replace(
        reference.component,
        atoms=tuple(reversed(reference.component.atoms)),
        bonds=tuple(
            BondDefinition(b.atom_name_2, b.atom_name_1, b.order, b.aromatic)
            for b in reversed(reference.component.bonds)
        ),
    )
    assert reference.matches(reordered)
    assert not reference.matches(
        replace(
            reordered,
            atoms=tuple(
                replace(a, formal_charge=1) if a.atom_name == "CG" else a
                for a in reordered.atoms
            ),
        )
    )


def test_coordinates_and_outside_terminal_state_do_not_select_body_state():
    reference = _reference()
    changed = replace(
        reference.component,
        atoms=tuple(
            replace(
                a,
                ideal_position=(99, 99, 99),
                formal_charge=1 if a.atom_name == "N" else a.formal_charge,
            )
            for a in reference.component.atoms
        ),
    )
    assert reference.matches(changed)


def test_unparented_hydrogen_does_not_match_as_zero():
    reference = _reference()
    broken = replace(
        reference.component,
        bonds=tuple(
            b
            for b in reference.component.bonds
            if "HB2" not in {b.atom_name_1, b.atom_name_2}
        ),
    )
    assert not reference.matches(broken)


def _monomer_text(component: IdealizedComponent) -> str:
    template = build_default_restraint_library().require(component.component_id)
    rows = [
        "data_comp_" + component.component_id,
        "loop_",
        "_chem_comp_atom.atom_id",
        "_chem_comp_atom.type_symbol",
        "_chem_comp_atom.charge",
    ]
    rows += [f"{a.atom_name} {a.element} {a.formal_charge}" for a in component.atoms]
    rows += [
        "loop_",
        "_chem_comp_bond.atom_id_1",
        "_chem_comp_bond.atom_id_2",
        "_chem_comp_bond.type",
        "_chem_comp_bond.aromatic",
        "_chem_comp_bond.value_dist",
        "_chem_comp_bond.value_dist_esd",
    ]
    orders = {1: "SINGLE", 2: "DOUBLE", 3: "TRIPLE"}
    for bond in component.bonds:
        target = template.bond_target(bond.atom_name_1, bond.atom_name_2)
        distance = 1.0 if target is None else target.target_distance_angstrom + 0.01
        rows.append(
            f"{bond.atom_name_1} {bond.atom_name_2} {orders[bond.order]} {'y' if bond.aromatic else 'n'} {distance} 0.02"
        )
    rows += [
        "loop_",
        "_chem_comp_angle.atom_id_1",
        "_chem_comp_angle.atom_id_2",
        "_chem_comp_angle.atom_id_3",
        "_chem_comp_angle.value_angle",
        "_chem_comp_angle.value_angle_esd",
    ]
    rows += [
        f"{t.atom_name_1} {t.center_atom_name} {t.atom_name_2} {t.target_angle_degrees + 0.5} 1.5"
        for t in template.angle_targets
    ]
    return "\n".join(rows) + "\n"


def test_monomer_adapter_reads_explicit_values_without_using_coordinates():
    reference = _reference()
    fallback = build_default_restraint_library().require("MSE")
    loaded = component_reference_from_monomer(
        _monomer_text(reference.component),
        fallback=fallback,
        atom_names=reference.atom_names,
        boundary_atom_names=reference.boundary_atom_names,
        source="test:generated-monomer-record",
    )
    assert loaded.matches(reference.component)
    expected = fallback.bond_target("CB", "CG")
    assert expected is not None
    actual = loaded.apply_to(fallback).bond_target("CB", "CG")
    assert actual is not None
    assert actual.target_distance_angstrom == pytest.approx(
        expected.target_distance_angstrom + 0.01
    )
    assert actual.esd_angstrom == 0.02
    library = build_default_restraint_library().with_component_reference(loaded)
    assert _bound(_structure(renamed_h=True), library).bond_target("CB", "CG") == actual


@pytest.mark.parametrize("value", ("nan", "inf", "-1", "0"))
def test_invalid_reference_numbers_are_configuration_errors(value):
    reference = _reference()
    text = _monomer_text(reference.component).replace(" 0.02\n", f" {value}\n")
    with pytest.raises(ValueError, match="finite and positive"):
        component_reference_from_monomer(
            text,
            fallback=build_default_restraint_library().require("MSE"),
            atom_names=reference.atom_names,
            boundary_atom_names=reference.boundary_atom_names,
            source="test:invalid-record",
        )


def test_missing_angle_preserves_the_whole_affected_center():
    reference = _reference()
    fallback = build_default_restraint_library().require("MSE")
    text = _monomer_text(reference.component)
    omitted = next(t for t in fallback.angle_targets if t.center_atom_name == "CG")
    line = f"{omitted.atom_name_1} {omitted.center_atom_name} {omitted.atom_name_2} {omitted.target_angle_degrees + 0.5} 1.5\n"
    assert line in text
    loaded = component_reference_from_monomer(
        text.replace(line, ""),
        fallback=fallback,
        atom_names=reference.atom_names,
        boundary_atom_names=reference.boundary_atom_names,
        source="test:partial-angles",
    )
    assert loaded.apply_to(fallback).angle_targets_for_center(
        "CG"
    ) == fallback.angle_targets_for_center("CG")


def _with_topology(
    structure: ProteinStructure, topology: StructureTopology
) -> ProteinStructure:
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=topology,
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def test_unspecified_default_charges_use_the_active_component():
    structure = _structure()
    structure = _with_topology(
        structure,
        StructureTopology(
            constitution=structure.constitution,
            atom_topologies=tuple(
                AtomTopology() for _ in structure.constitution.atom_slots
            ),
            bonds=structure.topology.bonds,
        ),
    )
    reference = _reference()
    library = build_default_restraint_library().with_component_reference(reference)
    assert (
        _bound(structure, library).bond_target("CB", "CG") == reference.bond_targets[0]
    )


@pytest.mark.parametrize("declared", (False, True))
def test_nondefault_state_requires_its_charge_facts(declared):
    reference = _reference("PTR")
    graph = reference.component.without_hydrogens_anchored_to({"O2P", "O3P"})
    graph = replace(
        graph,
        atoms=tuple(
            replace(a, formal_charge=-1) if a.atom_name in {"O2P", "O3P"} else a
            for a in graph.atoms
        ),
    )
    reference = replace(reference, component=graph)
    source = _structure(retained=True, component_id="PTR", omit=("HO2P", "HO3P"))
    topologies = tuple(
        AtomTopology(
            formal_charge=-1 if declared and atom.name in {"O2P", "O3P"} else None
        )
        for atom in source.constitution.atom_slots
    )
    source = _with_topology(
        source,
        StructureTopology(
            constitution=source.constitution,
            atom_topologies=topologies,
            bonds=source.topology.bonds,
        ),
    )
    library = build_default_restraint_library().with_component_reference(reference)
    expected = (
        reference.bond_targets[0]
        if declared
        else library.require("PTR").bond_target("CB", "CG")
    )
    assert _bound(source, library).bond_target("CB", "CG") == expected


def test_missing_polar_hydrogen_is_not_evidence_for_reference_state():
    library = build_default_restraint_library().with_component_reference(
        _reference("PTR")
    )
    assert _bound(
        _structure(retained=True, component_id="PTR", omit=("HO2P",)), library
    ) is library.require("PTR")


@pytest.mark.parametrize("atom_name", ("CG", "CA"))
@pytest.mark.parametrize(
    "relationship",
    (
        BondRelationshipType.COVALENT,
        BondRelationshipType.METAL_COORDINATION,
        BondRelationshipType.HYDROGEN_BOND,
    ),
)
def test_external_relationship_withdraws_isolated_reference(atom_name, relationship):
    source = _structure()
    neighbor_id = ResidueId("Z", 1)
    neighbor, geometry, charges = residue_payload(
        component_id="ZN",
        residue_id=neighbor_id,
        atoms=(atom_payload("ZN", "ZN", Vec3(5, 5, 5), formal_charge=2),),
    )
    source = source.with_ligand_facets(
        ligand_sites=(neighbor,),
        ligand_geometries=(geometry,),
        ligand_formal_charge_payloads=(charges,),
    )
    reference = _reference()
    library = build_default_restraint_library().with_component_reference(reference)
    assert _bound(source, library).bond_target("CB", "CG") == reference.bond_targets[0]
    bond = TopologyBond(
        source.constitution.atom_index(AtomRef(RESIDUE, atom_name)),
        source.constitution.atom_index(AtomRef(neighbor_id, "ZN")),
        relationship_type=relationship,
    )
    source = _with_topology(
        source,
        StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(*source.topology.bonds, bond),
        ),
    )
    assert _bound(source, library) is library.require("MSE")


def test_monomer_adapter_decodes_quoted_tokens_and_binds_content_identity():
    reference = _reference()
    text = "\n".join(
        line
        if line.startswith(("data_", "loop_", "_"))
        else " ".join('"' + token + '"' for token in line.split())
        for line in _monomer_text(reference.component).splitlines()
    )
    loaded = component_reference_from_monomer(
        text,
        fallback=build_default_restraint_library().require("MSE"),
        atom_names=reference.atom_names,
        boundary_atom_names=reference.boundary_atom_names,
        source="test:quoted-record",
    )
    assert loaded.matches(reference.component)
    assert (
        loaded.source
        == "test:quoted-record#text-sha256=" + sha256(text.encode("utf-8")).hexdigest()
    )


def test_source_identity_is_required_before_adding_content_digest():
    reference = _reference()
    with pytest.raises(ValueError, match="source identity"):
        component_reference_from_monomer(
            _monomer_text(reference.component),
            fallback=build_default_restraint_library().require("MSE"),
            atom_names=reference.atom_names,
            boundary_atom_names=reference.boundary_atom_names,
            source="   ",
        )


def test_missing_target_loop_is_a_reference_configuration_error():
    reference = _reference()
    text = _monomer_text(reference.component).split("loop_\n_chem_comp_angle", 1)[0]
    with pytest.raises(ValueError, match="angle-target loop"):
        component_reference_from_monomer(
            text,
            fallback=build_default_restraint_library().require("MSE"),
            atom_names=reference.atom_names,
            boundary_atom_names=reference.boundary_atom_names,
            source="test:missing-angles",
        )


def test_handbuilt_reference_cannot_partially_replace_an_angle_center():
    reference = _reference("PTR")
    fallback = build_default_restraint_library().require("PTR")
    target = next(t for t in fallback.angle_targets if t.center_atom_name == "CG")
    with pytest.raises(ValueError, match="complete angle centers"):
        build_default_restraint_library().with_component_reference(
            replace(reference, angle_targets=(target,))
        )
    component = _component("TPO")
    target = next(
        t
        for t in build_default_restraint_library().require("TPO").angle_targets
        if t.center_atom_name == "CB" and "CA" not in {t.atom_name_1, t.atom_name_2}
    )
    with pytest.raises(ValueError, match="boundary"):
        ComponentRestraintReference(
            component,
            frozenset(component.heavy_atom_names()) - BACKBONE,
            frozenset({"CA"}),
            (),
            (target,),
            "test:boundary-center",
        )


def test_diagnostics_use_the_matched_target_not_the_static_fallback():
    reference = _reference()
    target = reference.bond_targets[0]
    target = replace(
        target, target_distance_angstrom=target.target_distance_angstrom + 0.05
    )
    reference = replace(reference, bond_targets=(target,))
    library = build_default_restraint_library().with_component_reference(reference)
    report = detect_heavy_geometry(
        _structure(),
        component_library=build_default_component_library(),
        restraint_library=library,
        residue_ids={RESIDUE},
    )
    finding = next(
        item
        for item in report.bond_length_outliers
        if {item.atom_name_1, item.atom_name_2} == {"CB", "CG"}
    )
    assert finding.expected_distance_angstrom == target.target_distance_angstrom
    assert finding.restraint_backed


@pytest.mark.parametrize("retained", (False, True))
@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_real_ingress_preserves_reference_matching_in_both_roles(retained, file_format):
    source = read_structure_string(
        write_structure_string(_structure(retained=retained), file_format), file_format
    )
    reference = _reference()
    library = build_default_restraint_library().with_component_reference(reference)
    assert _bound(source, library).bond_target("CB", "CG") == reference.bond_targets[0]


def test_real_ingress_and_local_runtime_deliver_matched_native_constraint(monkeypatch):
    source = read_structure_string(
        write_structure_string(_structure(retained=True), FileFormat.MMCIF),
        FileFormat.MMCIF,
    )
    reference = _reference()
    library = build_default_restraint_library().with_component_reference(reference)
    target = reference.bond_targets[0]
    pair: list[frozenset[int]] = []
    calls = []
    original_build = native.build_rdkit_force_field
    original_constraint = native.add_rdkit_distance_constraint

    def build(
        molecule,
        *,
        problem,
        restraint_library,
        rdkit_atom_index_by_structure_atom_index,
    ):
        assert restraint_library is library
        observed = StructureRestraints(
            problem.region.snapshot.structure,
            problem.component_library,
            restraint_library,
        ).for_residue(RESIDUE)
        assert observed is not None and observed.bond_target("CB", "CG") == target
        constitution = problem.region.snapshot.structure.constitution
        pair.append(
            frozenset(
                rdkit_atom_index_by_structure_atom_index[
                    constitution.atom_index(AtomRef(RESIDUE, name))
                ]
                for name in ("CB", "CG")
            )
        )
        return original_build(
            molecule,
            problem=problem,
            restraint_library=restraint_library,
            rdkit_atom_index_by_structure_atom_index=rdkit_atom_index_by_structure_atom_index,
        )

    def constraint(force_field, **kwargs):
        if (
            pair
            and frozenset((kwargs["atom_index_1"], kwargs["atom_index_2"])) == pair[-1]
        ):
            tolerance = bond_length_tolerance_angstrom(
                policy=HeavyGeometryPolicy(), target=target
            )
            assert (
                kwargs["minimum_distance"]
                == target.target_distance_angstrom - tolerance
            )
            assert (
                kwargs["maximum_distance"]
                == target.target_distance_angstrom + tolerance
            )
            calls.append(kwargs)
        return original_constraint(force_field, **kwargs)

    monkeypatch.setattr(native, "build_rdkit_force_field", build)
    monkeypatch.setattr(native, "add_rdkit_distance_constraint", constraint)
    result, _ = profile_transform_local_region(
        source,
        DirectRegionTransformationSpec(
            LocalScopeSpec.from_residues((RESIDUE,)),
            ContinuousRelaxationForceField.UFF,
            config=ContinuousRelaxationConfig(max_iterations=1),
        ),
        component_library=build_default_component_library(),
        restraint_library=library,
    )
    assert calls and result.relaxation_executions

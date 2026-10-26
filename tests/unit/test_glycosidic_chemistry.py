"""Deposited glycosidic links determine both partners' hydrogen inventories."""

from collections import Counter
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    residue_payload,
)
from tests.support.component_reference import assert_component_reference_matches

from protrepair.chemistry import (
    HydrogenSemantics,
    IdealGeometryHeavyAtomMode,
    IdealGeometryHeavyAtomSemantics,
    IdealGeometryHydrogenSemantics,
    build_default_component_library,
)
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.preparation import AminoAcidPreparation
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.chemistry.retained_non_polymer.glycosidic import (
    GlycosidicTemplateBinding,
)
from protrepair.chemistry.retained_non_polymer.registry import (
    build_bundled_retained_non_polymer_registry,
)
from protrepair.geometry import Vec3
from protrepair.io import read_structure, read_structure_string, write_structure_string
from protrepair.io.nonstandard_component_ingress import (
    read_nonstandard_component_registry,
)
from protrepair.state import HydrogenCoverageState
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.topology import TopologyAvailabilityState
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import AtomSite
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.hydrogen.core import materialize_hydrogens_core
from protrepair.transformer.completion.hydrogen.topology import (
    resolve_fixed_hydrogen_topology,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen import (
    repair as retained_repair,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
    resolve_retained_non_polymer_topology,
)
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch

SOURCE = Path(__file__).parents[1] / "fixtures/corpus/pdb2z62.ent"
ASN_BY_CHAIN = {"B": 173, "C": 35, "D": 205}
EXPECTED_H = {
    "B": (12, 13, 11, 11),
    "C": (12, 14, 11),
    "D": (13, 14),
}


def _source(chain="D"):
    asn = ASN_BY_CHAIN[chain]
    records = []
    for line in SOURCE.read_text().splitlines():
        if line.startswith("ATOM  ") and line[21] == "A":
            if asn - 1 <= int(line[22:26]) <= asn + 1:
                records.append(line)
        elif (
            line.startswith("HETATM")
            and line[21] == chain
            or line.startswith("LINK  ")
            and chain in (line[21], line[51])
        ):
            records.append(line)
    return read_structure_string("\n".join(records) + "\nEND\n", FileFormat.PDB)


@pytest.mark.parametrize("chain", ("B", "C", "D"))
def test_glycan_inventory_and_source_link_orders_are_realized(chain):
    source = _source(chain)
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues
    assert (
        tuple(
            sum(atom.is_hydrogen() for atom in residue.atom_sites)
            for residue in result.structure.constitution.ligands
        )
        == EXPECTED_H[chain]
    )

    facts = (
        StructureChemistryObservation(
            result.structure, component_library=build_default_component_library()
        )
        .whole_structure_facts()[1]
        .retained_non_polymer_facts
    )
    assert all(
        fact.heavy_atom_topology_availability_state is TopologyAvailabilityState.PRESENT
        and fact.hydrogen_topology_availability_state
        is TopologyAvailabilityState.PRESENT
        for fact in facts
    )


def test_glycosylated_asn_has_one_secondary_amide_hydrogen():
    source = _source()
    retained = add_retained_non_polymer_hydrogens(source).structure
    result = materialize_hydrogens_core(
        retained, target_residue_ids=frozenset((ResidueId("A", 205),))
    )
    index = result.structure.constitution.residue_index(ResidueId("A", 205))
    anchors = result.structure.topology.residue_hydrogen_anchor_by_name(
        constitution=result.structure.constitution, residue_index=index
    )
    assert sum(parent == "ND2" for parent in anchors.values()) == 1
    residue_id = ResidueId("A", 205)

    def position(name):
        return result.structure.geometry.position(
            result.structure.constitution.atom_index(AtomRef(residue_id, name))
        ).to_array()

    center = position("ND2")
    c = position("CG") - center
    partner = (
        result.structure.geometry.position(
            result.structure.constitution.atom_index(AtomRef(ResidueId("D", 1), "C1"))
        ).to_array()
        - center
    )
    hydrogen = next(name for name, anchor in anchors.items() if anchor == "ND2")
    h = position(hydrogen) - center
    assert np.linalg.norm(h) == pytest.approx(1.01)
    assert np.dot(h, np.cross(c, partner)) == pytest.approx(0, abs=1e-10)
    assert np.dot(h, c) / np.linalg.norm(c) == pytest.approx(
        np.dot(h, partner) / np.linalg.norm(partner)
    )


@pytest.mark.parametrize("without_backbone", (False, True))
def test_grouped_asn_outputs_keep_one_linked_amide_h(without_backbone):
    source = _source()
    library = build_default_component_library()
    template = library.require("ASN")
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, HydrogenSemantics)
    plan = tuple(
        operation
        for operation in semantics.plan_with_backbone or ()
        if not {"1HD2", "2HD2"}.intersection(operation[0])
    ) + ((("1HD2", "2HD2"), "class2", ("CB", "CG", "ND2", 1.01)),)
    template = replace(
        template,
        hydrogen_semantics=replace(
            semantics,
            plan_with_backbone=plan,
            plan_without_backbone=plan if without_backbone else None,
        ),
    )
    library = library.with_template(template)
    residue_id = ResidueId("A", 205)
    residue = source.constitution.residue_or_ligand(residue_id)
    assert residue is not None
    binding = GlycosidicTemplateBinding(source, residue, template, library)
    assert tuple(
        name
        for name, parent in binding.template.template_hydrogen_anchor_by_name(
            binding.template.expected_hydrogen_atom_names()
        ).items()
        if parent == "ND2"
    ) == ("1HD2",)
    result = materialize_hydrogens_core(
        source, library, target_residue_ids=frozenset((residue_id,))
    )
    # Custom fixed-H plans do not authorize the separate backbone microstate sites.
    assert {issue.atom_names for issue in result.issues} == {("N",), ("C", "O", "OXT")}
    assert Counter(_hydrogen_parents(result.structure, residue_id).values())["ND2"] == 1
    _assert_source_preserved(source, result.structure)
    fact = (
        StructureChemistryObservation(result.structure, component_library=library)
        .residue_completion_facts(residue_id)
        .chemistry_readiness_facts
    )
    assert fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE


def _assert_source_preserved(source, structure):
    for old, _ in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(old)
        current = structure.constitution.atom_index(ref)
        assert structure.constitution.atom_site_at(
            current
        ) == source.constitution.atom_site_at(old)
        assert structure.geometry.atom_geometry(
            current
        ) == source.geometry.atom_geometry(old)
        charge = source.topology.formal_charge(old)
        if charge is not None:
            assert structure.topology.formal_charge(current) == charge
    for bond in source.topology.bonds:
        refs = tuple(source.constitution.atom_ref_at(i) for i in bond.endpoint_pair())
        current = structure.topology.bond_between(
            *(structure.constitution.atom_index(ref) for ref in refs)
        )
        assert current is not None
        assert current.provenance == bond.provenance
        assert current.source_metadata == bond.source_metadata


@pytest.mark.parametrize("chain", ("B", "C", "D"))
def test_glycan_graph_edits_preserve_original_facets_metadata_and_remap(chain):
    source = _source(chain)
    result = add_retained_non_polymer_hydrogens(source)
    _assert_source_preserved(source, result.structure)
    for bond in result.structure.topology.bonds:
        first, second = (
            result.structure.constitution.atom_ref_at(i) for i in bond.endpoint_pair()
        )
        if first.residue_id != second.residue_id:
            assert (
                bond.relationship_type is BondRelationshipType.COVALENT
                and bond.order == 1
            )
    repeated = add_retained_non_polymer_hydrogens(result.structure)
    assert (
        repeated.structure is result.structure
        and not repeated.repairs
        and not repeated.issues
    )


@pytest.mark.parametrize("name,expected_h", (("NAG", 15), ("BMA", 12), ("FUL", 12)))
def test_frozen_isolated_ccd_inventory_is_unchanged(name, expected_h):
    library = build_default_component_library()
    record = read_nonstandard_component_registry(
        SOURCE.parents[1] / "chemistry/retained" / f"{name}.cif"
    ).get(name)
    assert record is not None
    bundled = build_bundled_retained_non_polymer_registry().get(name)
    assert bundled is not None
    assert_component_reference_matches(bundled, record)
    component = record.to_idealized_component()
    atoms = []
    for atom in component.heavy_atoms():
        assert atom.ideal_position is not None
        atoms.append(
            atom_payload(
                atom.atom_name, atom.element, Vec3.from_iterable(atom.ideal_position)
            )
        )
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id=name,
                residue_id=ResidueId("L", 1),
                is_hetero=True,
                atoms=tuple(atoms),
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues
    assert (
        sum(
            atom.is_hydrogen()
            for atom in result.structure.constitution.ligands[0].atom_sites
        )
        == expected_h
    )
    _assert_source_preserved(source, result.structure)
    assert (
        library.require(name).expected_heavy_atom_names()
        == component.heavy_atom_names()
    )


def _with_bonds(source, bonds):
    return ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=bonds,
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )


def _change_link(
    source,
    *,
    relationship=BondRelationshipType.COVALENT,
    order=None,
    anchor=None,
    aromatic=False,
):
    donor = AtomRef(ResidueId("D", 1), "C1")
    acceptor = AtomRef(ResidueId("A", 205), "ND2")
    old = source.topology.bond_between(
        source.constitution.atom_index(donor), source.constitution.atom_index(acceptor)
    )
    assert old is not None
    bond = replace(
        old,
        relationship_type=relationship,
        order=order,
        aromatic=aromatic,
        source_metadata=None,
    )
    if anchor is not None:
        bond = replace(
            bond,
            atom_index_1=source.constitution.atom_index(donor),
            atom_index_2=source.constitution.atom_index(
                AtomRef(ResidueId("A", 205), anchor)
            ),
        )
    return _with_bonds(
        source, tuple(bond if b == old else b for b in source.topology.bonds)
    )


@pytest.mark.parametrize(
    "kwargs",
    (
        {"relationship": BondRelationshipType.UNKNOWN},
        {"relationship": BondRelationshipType.DISULFIDE},
        {"order": 2},
        {"aromatic": True},
        {"anchor": "CB"},
        {"anchor": "N"},
    ),
)
def test_unsupported_glycosidic_link_is_not_admitted_or_modified(kwargs):
    source = _change_link(_source(), **kwargs)
    for transformer in (
        add_retained_non_polymer_hydrogens,
        resolve_retained_non_polymer_topology,
    ):
        result = transformer(
            source,
            build_default_component_library(),
            target_residue_ids=frozenset((ResidueId("D", 1),)),
        )
        assert result.structure is source and result.issues and not result.repairs
    binding_residue = source.constitution.ligands[0]
    with pytest.raises(ValueError):
        GlycosidicTemplateBinding(
            source,
            binding_residue,
            build_default_component_library().require("NAG"),
            build_default_component_library(),
        )


def test_duplicate_anomeric_attachment_is_not_arbitrarily_selected():
    source = _source()
    source = _with_bonds(
        source,
        (
            *source.topology.bonds,
            TopologyBond(
                source.constitution.atom_index(AtomRef(ResidueId("D", 1), "C1")),
                source.constitution.atom_index(AtomRef(ResidueId("A", 205), "OD1")),
            ),
        ),
    )
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((ResidueId("D", 1),))
    )
    assert result.structure is source and result.issues


@pytest.mark.parametrize("name", ("C1", "C3"))
@pytest.mark.parametrize("chain,sequence", (("D", 1), ("B", 3), ("B", 4)))
def test_actual_coordinate_stereo_is_checked_at_anomer_and_ring_center(
    name, chain, sequence
):
    source = _source(chain)
    residue_id = ResidueId(chain, sequence)
    residue = source.constitution.residue_or_ligand(residue_id)
    assert residue is not None
    library = build_default_component_library()
    template = library.require(residue.component_id)
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    component = semantics.component
    binding = GlycosidicTemplateBinding(source, residue, template, library)
    partner = next(
        outside
        for inside, outside in binding.external_bonds
        if inside.atom_name == "C1"
    )
    refs = tuple(
        partner if neighbor == "O1" else AtomRef(residue_id, neighbor)
        for neighbor in sorted(component.bonded_atom_names(name))
        if (atom := component.atom(neighbor)) is not None and not atom.is_hydrogen()
    )
    a, b, c = (
        source.geometry.position(source.constitution.atom_index(ref)).to_array()
        for ref in refs
    )
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal)
    geometry = source.residue_geometry(source.constitution.residue_index(residue_id))
    old = geometry.position(name).to_array()
    flipped = old - 2 * np.dot(old - a, normal) * normal
    source = source.with_updated_residue_facets(
        residue,
        residue_geometry=geometry.with_atom_geometry(
            name,
            geometry.atom_geometry(name).with_position(Vec3.from_iterable(flipped)),
        ),
    )
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((residue_id,))
    )
    assert result.structure is source and result.issues and not result.repairs
    assert any("stereo" in issue.message for issue in result.issues)


@pytest.mark.parametrize(
    "mutation", ("missing", "extra", "element", "donor_charge", "asn_charge", "no_link")
)
def test_incomplete_or_contradictory_application_keeps_source(mutation):
    source = _source()
    residue = source.constitution.ligands[0]
    index = source.constitution.residue_index(residue.residue_id)
    geometry = source.residue_geometry(index)
    if mutation == "missing":
        source = source.without_atom_refs((AtomRef(residue.residue_id, "C7"),))
    elif mutation == "extra":
        source = source.with_updated_residue_facets(
            residue.with_atom_site(AtomSite("O1", "O")),
            residue_geometry=geometry.with_atom_geometry(
                "O1", AtomGeometry(Vec3(1, 2, 3))
            ),
        )
    elif mutation == "element":
        source = source.with_updated_residue_facets(
            residue.with_atom_site(AtomSite("O7", "N")), residue_geometry=geometry
        )
    elif mutation in ("donor_charge", "asn_charge"):
        target = (
            residue
            if mutation == "donor_charge"
            else source.constitution.residue_or_ligand(ResidueId("A", 205))
        )
        assert target is not None
        name = "C1" if mutation == "donor_charge" else "ND2"
        source = source.with_updated_residue_facets(
            target,
            residue_geometry=source.residue_geometry(
                source.constitution.residue_index(target.residue_id)
            ),
            formal_charge_by_atom_name=((name, 1),),
        )
    else:
        first, second = (
            AtomRef(residue.residue_id, "C1"),
            AtomRef(ResidueId("A", 205), "ND2"),
        )
        bond = source.topology.bond_between(
            source.constitution.atom_index(first),
            source.constitution.atom_index(second),
        )
        source = _with_bonds(
            source, tuple(b for b in source.topology.bonds if b != bond)
        )
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((ResidueId("D", 1),))
    )
    assert result.structure is source and result.issues and not result.repairs


def _deposit_selected_hydrogens(
    *, isotope="H", names=("H1",), amide_h=0, hydroxyl=False
):
    source = _source()
    generated = add_retained_non_polymer_hydrogens(source).structure
    selected = {AtomRef(ResidueId("D", 1), name) for name in names}
    generated = generated.without_atom_refs(
        tuple(
            generated.constitution.atom_ref_at(i)
            for i, _ in generated.geometry.iter_positions()
            if generated.constitution.atom_site_at(i).is_hydrogen()
            and generated.constitution.atom_ref_at(i) not in selected
        )
    )
    residue_id = ResidueId("D", 1)
    residue = generated.constitution.residue_or_ligand(residue_id)
    assert residue is not None
    geometry = generated.residue_geometry(
        generated.constitution.residue_index(residue_id)
    )
    residue = residue.with_atom_sites(
        replace(atom, element=isotope)
        if AtomRef(residue_id, atom.name) in selected
        else atom
        for atom in residue.atom_sites
    )
    if hydroxyl:
        residue = residue.with_atom_site(AtomSite("HO4", isotope))
        geometry = geometry.with_atom_geometry(
            "HO4",
            AtomGeometry(
                Vec3.from_iterable(geometry.position("O4").to_array() + (0, 0, 1))
            ),
        )
    generated = generated.with_updated_residue_facets(
        residue, residue_geometry=geometry
    )
    if amide_h:
        r = generated.constitution.residue_or_ligand(ResidueId("A", 205))
        assert r is not None
        g = generated.residue_geometry(
            generated.constitution.residue_index(r.residue_id)
        )
        for number in range(amide_h):
            name = f"HD2{number + 1}"
            r = r.with_atom_site(AtomSite(name, isotope))
            g = g.with_atom_geometry(
                name,
                AtomGeometry(
                    Vec3.from_iterable(
                        g.position("ND2").to_array() + (0, 0, 1 + 0.1 * number)
                    )
                ),
            )
        generated = generated.with_updated_residue_facets(r, residue_geometry=g)
    return read_structure_string(
        write_structure_string(generated, FileFormat.MMCIF), FileFormat.MMCIF
    )


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_compatible_deposited_glycan_hydrogen_keeps_identity_geometry_and_scalars(
    isotope,
):
    source = _deposit_selected_hydrogens(isotope=isotope)
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues
    _assert_source_preserved(source, result.structure)
    assert (
        sum(
            atom.is_hydrogen()
            for atom in result.structure.constitution.ligands[0].atom_sites
        )
        == 13
    )


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
@pytest.mark.parametrize("kind", ("hydroxyl", "overfilled_amide"))
def test_original_h_cannot_be_removed_to_make_linked_valence_valid(isotope, kind):
    source = _deposit_selected_hydrogens(
        isotope=isotope,
        hydroxyl=kind == "hydroxyl",
        amide_h=2 if kind == "overfilled_amide" else 0,
    )
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((ResidueId("D", 1),))
    )
    assert result.structure is source and result.issues and not result.repairs
    result = materialize_hydrogens_core(
        source, target_residue_ids=frozenset((ResidueId("A", 205),))
    )
    _assert_source_preserved(source, result.structure)
    assert bool(result.issues) == (kind == "overfilled_amide")


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_complete_glycan_roundtrip_and_topology_only_resolution(file_format):
    source = add_retained_non_polymer_hydrogens(_source()).structure
    recovered = read_structure_string(
        write_structure_string(source, file_format), file_format
    )
    result = resolve_retained_non_polymer_topology(
        recovered, build_default_component_library()
    )
    assert not result.issues
    assert (
        result.structure.constitution is recovered.constitution
        and result.structure.geometry is recovered.geometry
    )
    _assert_source_preserved(recovered, result.structure)
    assert not resolve_retained_non_polymer_topology(
        result.structure, build_default_component_library()
    ).repairs


def test_topology_only_does_not_publish_a_half_resolved_link_without_h():
    source = _source()
    result = resolve_retained_non_polymer_topology(
        source, build_default_component_library()
    )
    assert result.structure is source and result.issues and not result.repairs


@pytest.mark.parametrize("names,order", (({"CG", "OD1"}, 1), ({"CG", "ND2"}, 2)))
@pytest.mark.parametrize("origin", ("template", "source"))
def test_linked_asn_requires_actual_amide_bond_orders(names, order, origin):
    source = _source()
    library = build_default_component_library()
    if origin == "template":
        template = library.require("ASN")
        library = library.with_template(
            replace(
                template,
                definition=replace(
                    template.definition,
                    bonds=tuple(
                        replace(bond, order=order)
                        if {bond.atom_name_1, bond.atom_name_2} == names
                        else bond
                        for bond in template.definition.bonds
                    ),
                ),
            )
        )
    else:
        indices = tuple(
            source.constitution.atom_index(AtomRef(ResidueId("A", 205), name))
            for name in sorted(names)
        )
        original_bond = source.topology.bond_between(*indices)
        source = _with_bonds(
            source,
            tuple(
                replace(bond, order=order) if bond == original_bond else bond
                for bond in source.topology.bonds
            ),
        )
    result = add_retained_non_polymer_hydrogens(
        source, library, target_residue_ids=frozenset((ResidueId("D", 1),))
    )
    assert result.structure is source and result.issues and not result.repairs
    result = materialize_hydrogens_core(
        source, library, target_residue_ids=frozenset((ResidueId("A", 205),))
    )
    _assert_source_preserved(source, result.structure)
    asn = result.structure.constitution.residue_or_ligand(ResidueId("A", 205))
    assert asn is not None and not asn.has_atom_site("1HD2")
    assert result.issues


def test_secondary_amide_does_not_evaluate_an_unused_primary_amide_direction():
    source = _source()
    residue = source.constitution.residue_or_ligand(ResidueId("A", 205))
    assert residue is not None
    geometry = source.residue_geometry(
        source.constitution.residue_index(residue.residue_id)
    )
    source = source.with_updated_residue_facets(
        residue,
        residue_geometry=geometry.with_atom_geometry(
            "CB",
            geometry.atom_geometry("CB").with_position(
                Vec3.from_iterable(
                    2 * geometry.position("CG").to_array()
                    - geometry.position("ND2").to_array()
                )
            ),
        ),
    )
    result = materialize_hydrogens_core(
        source, target_residue_ids=frozenset((residue.residue_id,))
    )
    anchors = result.structure.topology.residue_hydrogen_anchor_by_name(
        constitution=result.structure.constitution,
        residue_index=result.structure.constitution.residue_index(residue.residue_id),
    )
    assert sum(parent == "ND2" for parent in anchors.values()) == 1
    _assert_source_preserved(source, result.structure)


def _with_generated_occupied_hydroxyl(*, charged=False):
    source = _source()
    residue = source.constitution.ligands[0]
    geometry = source.residue_geometry(
        source.constitution.residue_index(residue.residue_id)
    )
    return source.with_updated_residue_facets(
        residue.with_atom_site(AtomSite("HO4", "H")),
        residue_geometry=geometry.with_atom_geometry(
            "HO4",
            AtomGeometry(
                Vec3.from_iterable(geometry.position("O4").to_array() + (0, 0, 1))
            ),
        ),
        formal_charge_by_atom_name=(("HO4", 1),) if charged else (),
    )


def test_only_generated_occupied_hydroxyl_h_can_be_pruned():
    source = _with_generated_occupied_hydroxyl()
    original = _source()
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues
    assert not result.structure.constitution.ligands[0].has_atom_site("HO4")
    _assert_source_preserved(original, result.structure)
    assert any(event.atom_names == ("HO4",) for event in result.repairs)
    repeated = add_retained_non_polymer_hydrogens(result.structure)
    assert (
        repeated.structure is result.structure
        and not repeated.repairs
        and not repeated.issues
    )


@pytest.mark.parametrize("mode", ("charged", "topology_only", "redeposited"))
def test_hydroxyl_pruning_cannot_bypass_charge_source_or_topology_only_contract(mode):
    source = _with_generated_occupied_hydroxyl(charged=mode == "charged")
    if mode == "redeposited":
        source = read_structure_string(
            write_structure_string(source, FileFormat.MMCIF), FileFormat.MMCIF
        )
    transformer = (
        resolve_retained_non_polymer_topology
        if mode == "topology_only"
        else add_retained_non_polymer_hydrogens
    )
    result = transformer(
        source,
        build_default_component_library(),
        target_residue_ids=frozenset((ResidueId("D", 1),)),
    )
    assert result.structure is source and result.issues and not result.repairs


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_source_secondary_amide_isotope_is_not_replaced_by_template_h(isotope):
    source = _deposit_selected_hydrogens(isotope=isotope, amide_h=1)
    result = materialize_hydrogens_core(
        source, target_residue_ids=frozenset((ResidueId("A", 205),))
    )
    _assert_source_preserved(source, result.structure)
    anchors = result.structure.topology.residue_hydrogen_anchor_by_name(
        constitution=result.structure.constitution,
        residue_index=result.structure.constitution.residue_index(ResidueId("A", 205)),
    )
    assert tuple(name for name, parent in anchors.items() if parent == "ND2") == (
        "HD21",
    )


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_original_anomeric_h_on_opposite_tetrahedron_is_preserved_and_refused(isotope):
    source = _deposit_selected_hydrogens(isotope=isotope)
    residue = source.constitution.ligands[0]
    geometry = source.residue_geometry(
        source.constitution.residue_index(residue.residue_id)
    )
    # Put H beyond all three heavy neighbors, leaving the heavy scaffold intact.
    positions = [
        source.geometry.position(source.constitution.atom_index(ref)).to_array()
        for ref in (
            AtomRef(residue.residue_id, "C2"),
            AtomRef(residue.residue_id, "O5"),
            AtomRef(ResidueId("A", 205), "ND2"),
        )
    ]
    h = np.mean(positions, axis=0) + 2 * (
        np.mean(positions, axis=0) - geometry.position("C1").to_array()
    )
    source = source.with_updated_residue_facets(
        residue,
        residue_geometry=geometry.with_atom_geometry(
            "H1", geometry.atom_geometry("H1").with_position(Vec3.from_iterable(h))
        ),
    )
    source = read_structure_string(
        write_structure_string(source, FileFormat.MMCIF), FileFormat.MMCIF
    )
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((ResidueId("D", 1),))
    )
    assert result.structure is source and result.issues and not result.repairs
    assert any("stereo" in issue.message for issue in result.issues)


def test_explicit_invalid_override_is_not_replaced_with_a_successful_glycan_template():
    source = _source()
    result = add_retained_non_polymer_hydrogens(
        source,
        target_residue_ids=frozenset((ResidueId("D", 1),)),
        chemistry_evidence=(
            RetainedNonPolymerChemistryEvidence(
                residue_id=ResidueId("D", 1),
                smiles="not-a-smiles",
                heavy_atom_names=source.constitution.ligands[0].atom_site_names(),
            ),
        ),
    )
    assert result.structure is source and result.issues and not result.repairs


def test_unlinked_asn_stays_a_primary_amide_after_context_rebinding():
    linked = _source()
    context = PolymerMicrostateContext(linked)
    ref = AtomRef(ResidueId("A", 205), "ND2")
    assert len(context.external_heavy_bonds(ref)) == 1
    unlinked = _with_bonds(
        linked,
        tuple(
            bond
            for bond in linked.topology.bonds
            if bond not in context.external_heavy_bonds(ref)
        ),
    )
    preparation = AminoAcidPreparation(
        PolymerMicrostateContext(unlinked), build_default_component_library()
    )
    assert (
        sum(
            parent == "ND2"
            for _, parent in preparation.fixed_hydrogen_atom_sites(ref.residue_id)
        )
        == 2
    )
    assert (
        sum(
            parent == "ND2"
            for _, parent in AminoAcidPreparation(
                context, build_default_component_library()
            ).fixed_hydrogen_atom_sites(ref.residue_id)
        )
        == 1
    )


@pytest.mark.parametrize(
    "residue_id,name", ((ResidueId("D", 1), "C1"), (ResidueId("A", 205), "ND2"))
)
def test_partner_pose_changes_recheck_sugar_stereo_not_asn_inventory(residue_id, name):
    source = _source()
    observation = StructureChemistryObservation(source)
    expectation = observation.hydrogen_expectation_model
    preparation = expectation.polymer_preparation
    asn_id = ResidueId("A", 205)
    assert (
        sum(
            parent == "ND2"
            for _, parent in preparation.fixed_hydrogen_atom_sites(asn_id)
        )
        == 1
    )
    geometry = source.residue_geometry(source.constitution.residue_index(residue_id))
    moved_ref = AtomRef(residue_id, name)
    a, b, c = (
        source.geometry.position(source.constitution.atom_index(ref)).to_array()
        for ref in (
            AtomRef(ResidueId("D", 1), "C1"),
            AtomRef(ResidueId("D", 1), "O5"),
            AtomRef(ResidueId("D", 1), "C2"),
            AtomRef(asn_id, "ND2"),
        )
        if ref != moved_ref
    )
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal)
    old = geometry.position(name).to_array()
    flipped = old - 2 * np.dot(old - a, normal) * normal
    moved = source.with_updated_residue_geometries(
        (
            (
                residue_id,
                geometry.with_atom_geometry(
                    name,
                    geometry.atom_geometry(name).with_position(
                        Vec3.from_iterable(flipped)
                    ),
                ),
            ),
        )
    )
    assert preparation.matches_chemistry(moved)
    rebound = observation.for_structure(moved).hydrogen_expectation_model
    fresh = StructureChemistryObservation(moved).hydrogen_expectation_model
    for candidate in (rebound.polymer_preparation, fresh.polymer_preparation):
        assert (
            sum(
                parent == "ND2"
                for _, parent in candidate.fixed_hydrogen_atom_sites(asn_id)
            )
            == 1
        )
    sugar_id = ResidueId("D", 1)
    assert rebound.resolution_for_retained_non_polymer(
        sugar_id
    ) == fresh.resolution_for_retained_non_polymer(sugar_id)
    assert (
        "stereo" in rebound.resolution_for_retained_non_polymer(sugar_id).failure_reason
    )


@pytest.mark.parametrize(
    "mutation", ("charged", "anomeric_double", "aromatic", "hydroxyl_double")
)
def test_component_label_does_not_override_contradictory_pyranose_graph(mutation):
    source = _source()
    library = build_default_component_library()
    template = library.require("NAG")
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    component = semantics.component
    if mutation == "charged":
        component = replace(
            component,
            atoms=tuple(
                replace(atom, formal_charge=1) if atom.atom_name == "C1" else atom
                for atom in component.atoms
            ),
        )
    else:
        names = {"O4", "HO4"} if mutation == "hydroxyl_double" else {"C1", "C2"}
        component = replace(
            component,
            bonds=tuple(
                replace(bond, order=2, aromatic=mutation == "aromatic")
                if {bond.atom_name_1, bond.atom_name_2} == names
                else bond
                for bond in component.bonds
            ),
        )
    library = library.with_template(
        replace(template, hydrogen_semantics=IdealGeometryHydrogenSemantics(component))
    )
    result = add_retained_non_polymer_hydrogens(source, library)
    assert result.structure is source and result.issues and not result.repairs


def test_failure_in_one_glycan_does_not_change_it_or_block_other_glycans():
    source = read_structure_string(SOURCE.read_text(), FileFormat.PDB)
    bad = source.constitution.residue_or_ligand(ResidueId("B", 1))
    assert bad is not None
    source = source.with_updated_residue_facets(
        bad,
        residue_geometry=source.residue_geometry(
            source.constitution.residue_index(bad.residue_id)
        ),
        formal_charge_by_atom_name=(("C1", 1),),
    )
    result = add_retained_non_polymer_hydrogens(source)
    _assert_source_preserved(source, result.structure)
    for ligand in result.structure.constitution.ligands:
        if ligand.component_id not in {"NAG", "BMA", "FUL"}:
            continue
        count = sum(atom.is_hydrogen() for atom in ligand.atom_sites)
        if ligand.residue_id == ResidueId("B", 1):
            assert count == 0
        else:
            assert (
                count
                == EXPECTED_H[ligand.residue_id.chain_id][ligand.residue_id.seq_num - 1]
            )
    assert result.issues and all(
        issue.residue_id is not None and issue.residue_id.chain_id == "B"
        for issue in result.issues
    )


def test_generated_anomeric_h_is_stereo_checked_before_atom_or_link_publication(
    monkeypatch,
):
    source = _source()
    generate = retained_repair.generate_component_hydrogen_patch
    residue_id = ResidueId("D", 1)
    c1 = source.geometry.position(
        source.constitution.atom_index(AtomRef(residue_id, "C1"))
    ).to_array()
    neighbors = [
        source.geometry.position(source.constitution.atom_index(ref)).to_array()
        for ref in (
            AtomRef(residue_id, "C2"),
            AtomRef(residue_id, "O5"),
            AtomRef(ResidueId("A", 205), "ND2"),
        )
    ]
    opposite = Vec3.from_iterable(
        np.mean(neighbors, axis=0) + 2 * (np.mean(neighbors, axis=0) - c1)
    )

    def wrong_h(
        *,
        residue,
        patch,
        semantics,
        selected_hydrogen_names=None,
        skip_hydrogens_with_absent_anchors=False,
    ):
        generated = generate(
            residue=residue,
            patch=patch,
            semantics=semantics,
            selected_hydrogen_names=selected_hydrogen_names,
            skip_hydrogens_with_absent_anchors=skip_hydrogens_with_absent_anchors,
        )
        assert generated is not None
        return OrderedAtomPatch(
            tuple(
                replace(placement, position=opposite)
                if placement.atom_name == "H1"
                else placement
                for placement in generated.placements
            )
        )

    monkeypatch.setattr(retained_repair, "generate_component_hydrogen_patch", wrong_h)
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((residue_id,))
    )
    assert result.structure is source and result.issues and not result.repairs
    assert any("stereo" in issue.message for issue in result.issues)


@pytest.mark.parametrize("isotope", ("D", "T"))
def test_explicitly_bonded_source_isotope_with_non_template_name_keeps_its_identity(
    isotope,
):
    source = _deposit_selected_hydrogens(isotope=isotope)
    records = []
    for line in write_structure_string(source, FileFormat.PDB).splitlines():
        if line.startswith("HETATM") and line[12:16].strip() == "H1":
            line = line[:12] + " Q1 " + line[16:]
        records.append(line)
    source = read_structure_string("\n".join(records) + "\n", FileFormat.PDB)
    result = add_retained_non_polymer_hydrogens(source)
    _assert_source_preserved(source, result.structure)
    assert not result.issues
    residue = result.structure.constitution.ligands[0]
    assert residue.atom_site("Q1").element == isotope and not residue.has_atom_site(
        "H1"
    )
    fact = (
        StructureChemistryObservation(
            result.structure, component_library=build_default_component_library()
        )
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert (
        fact.hydrogen_topology_availability_state is TopologyAvailabilityState.PRESENT
    )


def test_contextual_inventory_projects_custom_ideal_heavy_semantics_too():
    source = _source()
    library = build_default_component_library()
    template = library.require("NAG")
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    library = library.with_template(
        replace(
            template,
            heavy_atom_semantics=IdealGeometryHeavyAtomSemantics(
                semantics.component,
                IdealGeometryHeavyAtomMode.REFERENCE_OR_SALVAGE,
            ),
        )
    )
    result = add_retained_non_polymer_hydrogens(source, library)
    assert not result.issues
    assert (
        tuple(
            sum(atom.is_hydrogen() for atom in ligand.atom_sites)
            for ligand in result.structure.constitution.ligands
        )
        == EXPECTED_H["D"]
    )
    facts = (
        StructureChemistryObservation(result.structure, component_library=library)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts
    )
    assert all(
        fact.heavy_atom_topology_availability_state is TopologyAvailabilityState.PRESENT
        for fact in facts
    )


def _flip_center(source, residue_id, name):
    template = build_default_component_library().require(
        source.constitution.residue_or_ligand(residue_id).component_id
    )
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    component = semantics.component
    refs = tuple(
        AtomRef(residue_id, neighbor)
        for neighbor in component.bonded_atom_names(name)
        if (atom := component.atom(neighbor)) is not None and not atom.is_hydrogen()
    )
    a, b, c = (
        source.geometry.position(source.constitution.atom_index(ref)).to_array()
        for ref in refs
    )
    normal = np.cross(b - a, c - a)
    normal /= np.linalg.norm(normal)
    geometry = source.residue_geometry(source.constitution.residue_index(residue_id))
    old = geometry.position(name).to_array()
    return source.with_updated_residue_geometries(
        (
            (
                residue_id,
                geometry.with_atom_geometry(
                    name,
                    geometry.atom_geometry(name).with_position(
                        Vec3.from_iterable(old - 2 * np.dot(old - a, normal) * normal)
                    ),
                ),
            ),
        )
    )


def _hydrogen_parents(structure, residue_id):
    return structure.topology.residue_hydrogen_anchor_by_name(
        constitution=structure.constitution,
        residue_index=structure.constitution.residue_index(residue_id),
    )


def test_unknown_link_leaves_only_nd2_unresolved_and_prepares_independent_asn_h():
    source = _change_link(_source(), relationship=BondRelationshipType.UNKNOWN)
    residue_id = ResidueId("A", 205)
    result = materialize_hydrogens_core(
        source, target_residue_ids=frozenset((residue_id,))
    )
    parents = Counter(_hydrogen_parents(result.structure, residue_id).values())
    assert parents["CA"] == 1 and parents["CB"] == 2
    assert parents["ND2"] == 0
    assert any(issue.residue_id == residue_id for issue in result.issues)
    observation = StructureChemistryObservation(result.structure)
    fact = observation.residue_completion_facts(residue_id).chemistry_readiness_facts
    assert fact.hydrogen_coverage_state is HydrogenCoverageState.PARTIAL
    assert fact.hydrogen_topology_availability_state is TopologyAvailabilityState.ABSENT


@pytest.mark.parametrize("bad_seq", (1, 2))
def test_remote_stereo_does_not_veto_a_valid_neighbor_or_asn(bad_seq):
    source = _source()
    bad_id, good_id = ResidueId("D", bad_seq), ResidueId("D", 3 - bad_seq)
    source = _flip_center(source, bad_id, "C3")
    result = add_retained_non_polymer_hydrogens(source)
    assert {issue.residue_id for issue in result.issues} == {bad_id}
    good = result.structure.constitution.residue_or_ligand(good_id)
    bad = result.structure.constitution.residue_or_ligand(bad_id)
    assert good is not None and bad is not None
    assert any(atom.is_hydrogen() for atom in good.atom_sites)
    assert not any(atom.is_hydrogen() for atom in bad.atom_sites)
    asn_id = ResidueId("A", 205)
    polymer = materialize_hydrogens_core(
        result.structure, target_residue_ids=frozenset((asn_id,))
    )
    assert Counter(_hydrogen_parents(polymer.structure, asn_id).values())["ND2"] == 1


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_unresolved_amide_preserves_source_isotopes_and_known_parent_topology(isotope):
    source = _change_link(
        _deposit_selected_hydrogens(isotope=isotope, amide_h=1),
        relationship=BondRelationshipType.UNKNOWN,
    )
    residue_id = ResidueId("A", 205)
    result = materialize_hydrogens_core(
        source, target_residue_ids=frozenset((residue_id,))
    )
    _assert_source_preserved(source, result.structure)
    assert Counter(_hydrogen_parents(result.structure, residue_id).values())["CB"] == 2
    residue = result.structure.constitution.residue_or_ligand(residue_id)
    assert residue is not None and residue.atom_site("HD21").element == isotope

    stripped = _with_bonds(
        result.structure,
        tuple(
            bond
            for bond in result.structure.topology.bonds
            if not any(
                result.structure.constitution.atom_ref_at(i).residue_id == residue_id
                and result.structure.constitution.atom_site_at(i).is_hydrogen()
                for i in bond.endpoint_pair()
            )
        ),
    )
    topology_only = resolve_fixed_hydrogen_topology(
        stripped,
        build_default_component_library(),
        target_residue_ids=frozenset((residue_id,)),
    )
    assert topology_only.structure.constitution is stripped.constitution
    assert topology_only.structure.geometry is stripped.geometry
    assert any(issue.residue_id == residue_id for issue in topology_only.issues)
    parents = Counter(_hydrogen_parents(topology_only.structure, residue_id).values())
    assert parents["CA"] == 1 and parents["CB"] == 2 and parents["ND2"] == 0
    fact = (
        StructureChemistryObservation(topology_only.structure)
        .residue_completion_facts(residue_id)
        .chemistry_readiness_facts
    )
    assert fact.hydrogen_coverage_state is HydrogenCoverageState.PARTIAL


def test_1g82_rejected_attachment_does_not_remove_independent_asn_h():
    source = read_structure(Path(__file__).parents[1] / "fixtures/corpus/pdb1g82.ent")
    residue_ids = frozenset(ResidueId(chain, 79) for chain in "ABCD")
    result = materialize_hydrogens_core(source, target_residue_ids=residue_ids)
    _assert_source_preserved(source, result.structure)
    for residue_id in residue_ids:
        parents = Counter(_hydrogen_parents(result.structure, residue_id).values())
        assert parents["CA"] == 1 and parents["CB"] == 2
        assert parents["ND2"] == (0 if residue_id.chain_id == "A" else 1)
    assert {issue.residue_id for issue in result.issues} == {ResidueId("A", 79)}


def test_1vbo_only_opposite_stereo_residues_are_rejected():
    source = read_structure(Path(__file__).parents[1] / "fixtures/corpus/pdb1vbo.ent")
    selected = frozenset(
        ResidueId(chain, number) for chain in "IJK" for number in (1, 2, 3)
    )
    result = add_retained_non_polymer_hydrogens(source, target_residue_ids=selected)
    _assert_source_preserved(source, result.structure)
    assert {issue.residue_id for issue in result.issues} == {
        ResidueId(chain, 1) for chain in "IJK"
    }
    for residue_id in selected:
        residue = result.structure.constitution.residue_or_ligand(residue_id)
        assert residue is not None
        assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == (
            0 if residue_id.seq_num == 1 else 11
        )

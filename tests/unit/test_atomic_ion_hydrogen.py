"""Atomic inventory is not an organic implicit-hydrogen request."""

from dataclasses import replace

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    residue_payload,
)
from tests.support.request_builders import (
    ingress_options,
    whole_structure_requested_goals,
)

from protrepair.api import process_structure
from protrepair.chemistry import (
    ChemicalComponentDefinition,
    ComponentLibrary,
    HeavyAtomSemantics,
    IdealGeometryHydrogenSemantics,
    ResidueTemplate,
)
from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.idealized import (
    IdealizedComponent,
    IdealizedComponentAtom,
)
from protrepair.chemistry.inference.retained_non_polymer_fallback import (
    infer_retained_non_polymer_rdkit_fallback,
)
from protrepair.chemistry.internal_coordinates import InternalCoordinateProgram
from protrepair.diagnostics.source_microstate import (
    MicrostateStructuralRole,
    collect_microstate_evidence,
)
from protrepair.errors import UnknownElementRadiusError
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import HydrogenApplicabilityState, HydrogenCoverageState
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.retained_non_polymer_chemistry import (
    resolve_retained_non_polymer_chemistry,
)
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.provenance import FileFormat
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
    resolve_retained_non_polymer_topology,
)
from protrepair.transformer.continuous.support import (
    LocalBondPlanningSupportMode,
    resolve_local_bond_planning_support,
)
from protrepair.workflow.contracts import LigandPolicy

ION = ResidueId("Z", 71)


def _source(element="CA", charge=None, *, hydrogen=None):
    atoms = [
        atom_payload(
            "X1",
            element,
            Vec3(1, 2, 3),
            occupancy=0.6,
            b_factor=17.0,
            formal_charge=charge,
        )
    ]
    if hydrogen is not None:
        atoms.append(atom_payload("HX", hydrogen, Vec3(1, 2, 5.07)))
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK", residue_id=ION, is_hetero=True, atoms=tuple(atoms)
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    return structure.with_provenance(
        replace(
            structure.provenance,
            ingress=replace(
                structure.provenance.ingress,
                observation=StructureObservation.from_source_facets(
                    constitution=structure.constitution,
                    geometry=structure.geometry,
                    topology=structure.topology,
                ),
            ),
        )
    )


@pytest.mark.parametrize(
    "element", ("CA", "MG", "NA", "K", "ZN", "FE", "CU", "CL", "BR", "F", "I")
)
@pytest.mark.parametrize("charge", (None, 0, 2, -1))
@pytest.mark.parametrize("allow_fallback", (True, False))
def test_atomic_ion_has_zero_h_expectation_without_selecting_charge(
    element, charge, allow_fallback
):
    source = _source(element, charge)
    library = ComponentLibrary()
    resolution = resolve_retained_non_polymer_chemistry(
        source,
        source.constitution.ligands[0],
        component_library=library,
        allow_rdkit_fallback=allow_fallback,
    )
    assert resolution.source.is_resolved()
    assert resolution.expected_hydrogen_atom_names == ()
    assert resolution.formal_charge_requirements == ()
    _, chemistry = StructureChemistryObservation(
        source,
        component_library=library,
        allow_retained_non_polymer_rdkit_fallback=allow_fallback,
    ).whole_structure_facts()
    fact = chemistry.retained_non_polymer_facts[0]
    assert (
        fact.hydrogen_applicability_state is HydrogenApplicabilityState.NOT_APPLICABLE
    )
    assert fact.is_supported()
    assert not fact.requires_hydrogen_completion()
    assert not fact.requires_topology_resolution()
    assert (
        source.topology.formal_charge(
            source.constitution.atom_index(AtomRef(ION, "X1"))
        )
        == charge
    )


@pytest.mark.parametrize("element", ("CA", "MG", "NA", "CL"))
@pytest.mark.parametrize(
    "operation",
    (add_retained_non_polymer_hydrogens, resolve_retained_non_polymer_topology),
)
@pytest.mark.parametrize("allow_fallback", (True, False))
def test_atomic_ion_direct_transform_preserves_all_source_facets(
    element, operation, allow_fallback
):
    source = _source(element)
    result = operation(
        source,
        component_library=ComponentLibrary(),
        allow_retained_non_polymer_rdkit_fallback=allow_fallback,
    )
    assert result.structure is source
    assert result.repairs == ()
    assert result.issues == ()


@pytest.mark.parametrize("element", ("CA", "MG", "CL"))
def test_public_complete_h_goal_does_not_create_atomic_hydrides(element):
    source = _source(element)
    result = process_structure(
        source,
        ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert not result.has_errors()
    assert result.structure.constitution == source.constitution
    assert result.structure.geometry == source.geometry
    assert result.structure.topology == source.topology
    assert (
        result.structure.provenance.ingress.observation
        == source.provenance.ingress.observation
    )


@pytest.mark.parametrize("element", ("CA", "MG", "CL"))
def test_raw_rdkit_fallback_does_not_invent_atomic_hydrogens(element):
    source = _source(element)
    result = infer_retained_non_polymer_rdkit_fallback(
        source.constitution.ligands[0],
        source.residue_geometry(source.constitution.residue_index(ION)),
    )
    assert result.hydrogen_atom_names == ()
    assert result.hydrogen_bond_definitions == ()
    assert result.hydrogenated_molecule.GetNumAtoms() == 1
    assert result.hydrogenated_molecule.GetAtomWithIdx(0).GetNumImplicitHs() == 0


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
@pytest.mark.parametrize("removed", (False, True))
def test_atomic_rule_does_not_erase_or_reinterpret_source_hydrogen(isotope, removed):
    source = _source(hydrogen=isotope)
    current = source.without_atom_refs((AtomRef(ION, "HX"),)) if removed else source
    resolution = resolve_retained_non_polymer_chemistry(
        current,
        current.constitution.ligands[0],
        component_library=ComponentLibrary(),
    )
    assert not resolution.source.is_resolved()
    result = add_retained_non_polymer_hydrogens(
        current, component_library=ComponentLibrary()
    )
    assert result.structure is current
    assert result.repairs == ()
    assert result.issues
    assert (
        result.structure.provenance.ingress.observation
        == source.provenance.ingress.observation
    )


def test_explicit_molecular_hydride_override_precedes_atomic_rule():
    source = _source(charge=0)
    evidence = RetainedNonPolymerChemistryOverride(ION, "[CaH2]", ("X1",)).to_evidence()
    result = add_retained_non_polymer_hydrogens(
        source,
        component_library=ComponentLibrary(),
        chemistry_evidence=(evidence,),
    )
    residue = result.structure.constitution.ligands[0]
    assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == 2
    assert len(result.structure.topology.bonds) == 2


def test_carbon_named_like_calcium_remains_an_organic_fallback():
    source = _source(element="C")
    result = add_retained_non_polymer_hydrogens(
        source, component_library=ComponentLibrary()
    )
    assert (
        sum(
            atom.is_hydrogen()
            for atom in result.structure.constitution.ligands[0].atom_sites
        )
        == 4
    )


def _linked_source(
    relationship=BondRelationshipType.METAL_COORDINATION, *, reverse=False
):
    residues = (
        residue_payload(
            component_id="UNK",
            residue_id=ION,
            is_hetero=True,
            atoms=(atom_payload("X1", "CA", Vec3(0, 0, 0)),),
        ),
        residue_payload(
            component_id="UNK",
            residue_id=ResidueId("A", 1),
            is_hetero=True,
            atoms=(atom_payload("O1", "O", Vec3(2.4, 0, 0)),),
        ),
        residue_payload(
            component_id="UNK",
            residue_id=ResidueId("B", 1),
            is_hetero=True,
            atoms=(atom_payload("O1", "O", Vec3(-2.4, 0, 0)),),
        ),
    )
    structure = build_structure(
        chains=(),
        ligands=tuple(reversed(residues)) if reverse else residues,
        source_format=FileFormat.MMCIF,
    )
    bonds = tuple(
        TopologyBond(
            structure.constitution.atom_index(AtomRef(ION, "X1")),
            structure.constitution.atom_index(AtomRef(ResidueId(chain, 1), "O1")),
            order=None,
            relationship_type=relationship,
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=SourceBondMetadata(
                source_id=f"coord-{chain}",
                record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
                reported_relationship_type=relationship,
                reported_distance_angstrom=2.4,
            ),
        )
        for chain in ("A", "B")
    )
    topology = StructureTopology(
        constitution=structure.constitution,
        atom_topologies=structure.topology.atom_topologies,
        bonds=bonds,
    )
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=topology,
        polymer_blueprint=structure.polymer_blueprint,
        provenance=replace(
            structure.provenance,
            ingress=replace(
                structure.provenance.ingress,
                observation=StructureObservation.from_source_facets(
                    constitution=structure.constitution,
                    geometry=structure.geometry,
                    topology=topology,
                ),
            ),
        ),
    )


@pytest.mark.parametrize("reverse", (False, True))
@pytest.mark.parametrize(
    "relationship",
    (BondRelationshipType.METAL_COORDINATION, BondRelationshipType.HYDROGEN_BOND),
)
def test_atomic_admission_preserves_multiple_typed_cross_chain_connections(
    reverse, relationship
):
    source = _linked_source(relationship, reverse=reverse)
    result = add_retained_non_polymer_hydrogens(
        source,
        component_library=ComponentLibrary(),
        target_residue_ids=frozenset((ION,)),
    )
    assert result.structure is source
    assert result.issues == ()
    residue = source.constitution.residue_or_ligand(ION)
    assert residue is not None
    resolution = resolve_retained_non_polymer_chemistry(
        source, residue, component_library=ComponentLibrary()
    )
    assert resolution.source.is_resolved()
    assert resolution.expected_hydrogen_atom_names == ()


@pytest.mark.parametrize("removed", (False, True))
@pytest.mark.parametrize(
    "relationship", (BondRelationshipType.COVALENT, BondRelationshipType.UNKNOWN)
)
def test_atomic_admission_cannot_hide_source_molecular_connections(
    removed, relationship
):
    source = _linked_source(relationship)
    current = (
        source
        if not removed
        else ProteinStructure.from_payload(
            constitution=source.constitution,
            geometry=source.geometry,
            topology=StructureTopology(
                constitution=source.constitution,
                atom_topologies=source.topology.atom_topologies,
            ),
            polymer_blueprint=source.polymer_blueprint,
            provenance=source.provenance,
        )
    )
    residue = current.constitution.residue_or_ligand(ION)
    assert residue is not None
    resolution = resolve_retained_non_polymer_chemistry(
        current, residue, component_library=ComponentLibrary()
    )
    assert not resolution.source.is_resolved()
    result = add_retained_non_polymer_hydrogens(
        current,
        component_library=ComponentLibrary(),
        target_residue_ids=frozenset((ION,)),
    )
    assert result.structure is current
    assert result.issues


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_selected_hydride_keeps_source_isotope_and_can_add_missing_h(isotope):
    source = _source(charge=0, hydrogen=isotope)
    evidence = RetainedNonPolymerChemistryOverride(ION, "[CaH2]", ("X1",)).to_evidence()
    result = add_retained_non_polymer_hydrogens(
        source, component_library=ComponentLibrary(), chemistry_evidence=(evidence,)
    )
    residue = result.structure.constitution.ligands[0]
    assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == 2
    assert residue.atom_site("HX").element == isotope
    index = result.structure.constitution.atom_index(AtomRef(ION, "HX"))
    source_index = source.constitution.atom_index(AtomRef(ION, "HX"))
    assert result.structure.geometry.atom_geometry(
        index
    ) == source.geometry.atom_geometry(source_index)


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
@pytest.mark.parametrize("charge", (None, 2, -1))
def test_atomic_source_public_roundtrip_keeps_charge_scalars_and_inventory(
    file_format, charge
):
    source = _source(charge=charge)
    ingested = read_structure_string(
        write_structure_string(source, file_format=file_format), file_format=file_format
    )
    result = process_structure(
        ingested,
        ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    reread = read_structure_string(
        write_structure_string(result.structure, file_format=file_format),
        file_format=file_format,
    )
    assert reread.constitution == ingested.constitution
    assert reread.geometry == ingested.geometry
    assert reread.topology == ingested.topology


def test_unknown_element_is_an_unresolved_site_not_a_native_exception():
    source = _source(element="X")
    resolution = resolve_retained_non_polymer_chemistry(
        source, source.constitution.ligands[0], component_library=ComponentLibrary()
    )
    assert not resolution.source.is_resolved()
    result = add_retained_non_polymer_hydrogens(
        source, component_library=ComponentLibrary()
    )
    assert result.structure is source
    assert result.issues


def test_unknown_element_public_workflow_keeps_owned_radius_refusal():
    source = _source(element="X")
    with pytest.raises(UnknownElementRadiusError):
        process_structure(
            source,
            ingress=ingress_options(ligand_policy=LigandPolicy.KEEP),
            requested_goals=whole_structure_requested_goals(
                HydrogenCoverageState.COMPLETE
            ),
        )


@pytest.mark.parametrize(
    "element",
    (
        "AG",
        "AL",
        "BA",
        "CA",
        "CD",
        "CL",
        "CO",
        "CU",
        "FE",
        "HG",
        "K",
        "LI",
        "MG",
        "MN",
        "NA",
        "NI",
        "RB",
        "SR",
        "ZN",
        "AU",
        "BR",
        "I",
    ),
)
@pytest.mark.parametrize("movable", (True, False))
def test_atomic_classification_is_shared_without_authorizing_metal_motion(
    element, movable
):
    source = _source(element)
    index = source.constitution.residue_index(ION)
    residue = source.constitution.residue_site_at(index)
    library = ComponentLibrary()
    evidence = collect_microstate_evidence(
        residue,
        residue_geometry=source.residue_geometry(index),
        source_formal_charge_by_atom_name=dict(
            source.residue_formal_charge_by_atom_name(index)
        ),
        standard_component_library=library,
        component_library=library,
    )
    assert (
        evidence.classification.structural_role
        is MicrostateStructuralRole.SINGLE_ATOM_INORGANIC
    )
    support = resolve_local_bond_planning_support(
        ProteinStructureSnapshot.from_structure(source),
        index,
        residue,
        movable_atom_indices=(source.constitution.atom_index(AtomRef(ION, "X1")),)
        if movable
        else (),
        component_library=library,
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert support.mode is (
        LocalBondPlanningSupportMode.UNSUPPORTED
        if movable
        else LocalBondPlanningSupportMode.NONBONDING_PASSIVE_CONTEXT
    )
    assert support.fallback_bond_definitions == ()


@pytest.mark.parametrize("hydrogen_count", (0, 2))
def test_component_template_authority_is_not_replaced_by_atomic_rule(hydrogen_count):
    atoms = (
        IdealizedComponentAtom(
            "X1", "CA", 2 if not hydrogen_count else 0, ideal_position=(0, 0, 0)
        ),
    ) + tuple(
        IdealizedComponentAtom(
            f"H{i + 1}", "H", 0, ideal_position=(0, 0, 2.07 if i == 0 else -2.07)
        )
        for i in range(hydrogen_count)
    )
    bonds = tuple(BondDefinition("X1", atom.atom_name) for atom in atoms[1:])
    template = ResidueTemplate(
        definition=ChemicalComponentDefinition(
            "UNK",
            tuple(atom.atom_name for atom in atoms),
            bonds=bonds,
            formal_charges={atom.atom_name: atom.formal_charge for atom in atoms},
        ),
        heavy_atom_semantics=HeavyAtomSemantics(
            program=InternalCoordinateProgram.backbone_only(), atom_order=("X1",)
        ),
        hydrogen_semantics=IdealGeometryHydrogenSemantics(
            IdealizedComponent("UNK", None, atoms, bonds)
        ),
    )
    library = ComponentLibrary(templates={"UNK": template})
    source = _source()
    resolution = resolve_retained_non_polymer_chemistry(
        source, source.constitution.ligands[0], component_library=library
    )
    assert len(resolution.expected_hydrogen_atom_names) == hydrogen_count
    assert resolution.formal_charge_requirements == (
        ("X1", 2 if not hydrogen_count else 0),
    )
    result = add_retained_non_polymer_hydrogens(source, component_library=library)
    if hydrogen_count:
        # This ideal-coordinate template has no orienting heavy-atom frame.
        assert result.structure is source
        assert result.issues
        return
    assert not result.issues
    assert (
        sum(
            atom.is_hydrogen()
            for atom in result.structure.constitution.ligands[0].atom_sites
        )
        == hydrogen_count
    )
    assert result.structure.topology.formal_charge(
        result.structure.constitution.atom_index(AtomRef(ION, "X1"))
    ) == (2 if not hydrogen_count else None)


def test_generated_h_on_original_atomic_site_is_not_erased_or_extended():
    original = _source()
    molecular = _source(hydrogen="H")
    current = molecular.with_provenance(original.provenance)
    result = add_retained_non_polymer_hydrogens(
        current, component_library=ComponentLibrary()
    )
    assert result.structure is current
    assert result.issues
    resolution = resolve_retained_non_polymer_chemistry(
        current, current.constitution.ligands[0], component_library=ComponentLibrary()
    )
    assert not resolution.source.is_resolved()


@pytest.mark.parametrize("change", ("none", "remove", "rebind", "retype", "missing"))
@pytest.mark.parametrize("reverse", (False, True))
def test_atomic_inventory_requires_original_coordination_pairs(change, reverse):
    source = _linked_source(reverse=reverse)
    residue = source.constitution.residue_or_ligand(ION)
    assert residue is not None
    original = resolve_retained_non_polymer_chemistry(
        source, residue, component_library=ComponentLibrary()
    )
    first = source.topology.bonds[0]
    if change == "missing":
        source = source.without_atom_refs((AtomRef(ResidueId("A", 1), "O1"),))
    elif change != "none":
        if change == "rebind":
            spare = residue_payload(
                component_id="UNK",
                residue_id=ResidueId("C", 1),
                is_hetero=True,
                atoms=(atom_payload("O1", "O", Vec3(0, 2.4, 0)),),
            )
            source = source.with_ligand_facets(
                ligand_sites=(*source.constitution.ligands, spare[0]),
                ligand_geometries=(
                    *(
                        source.residue_geometry(
                            source.constitution.residue_index(site.residue_id)
                        )
                        for site in source.constitution.ligands
                    ),
                    spare[1],
                ),
            )
        replacement = (
            replace(
                first,
                atom_index_1=source.constitution.atom_index(AtomRef(ION, "X1")),
                atom_index_2=source.constitution.atom_index(
                    AtomRef(ResidueId("C", 1), "O1")
                ),
            )
            if change == "rebind"
            else replace(first, relationship_type=BondRelationshipType.HYDROGEN_BOND)
        )
        source = ProteinStructure.from_payload(
            constitution=source.constitution,
            geometry=source.geometry,
            topology=StructureTopology(
                constitution=source.constitution,
                atom_topologies=source.topology.atom_topologies,
                bonds=tuple(
                    replacement if bond == first else bond
                    for bond in source.topology.bonds
                    if change != "remove" or bond != first
                ),
            ),
            provenance=source.provenance,
        )
    residue = source.constitution.residue_or_ligand(ION)
    assert residue is not None
    resolution = resolve_retained_non_polymer_chemistry(
        source, residue, component_library=ComponentLibrary()
    )
    if change == "none":
        assert resolution.source.is_resolved()
        assert len(resolution.coordination_bonds) == 2
        assert resolution.has_coordination_topology(source)
    else:
        assert not resolution.source.is_resolved()
        assert not original.has_coordination_topology(source)
    for operation in (
        add_retained_non_polymer_hydrogens,
        resolve_retained_non_polymer_topology,
    ):
        result = operation(
            source, ComponentLibrary(), target_residue_ids=frozenset((ION,))
        )
        assert result.structure is source and not result.repairs
        assert bool(result.issues) == (change != "none")

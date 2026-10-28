"""Reference inventory verification must not grant preparation authority."""

from dataclasses import replace
from unittest.mock import Mock

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_bonded_structure,
    residue_payload,
)
from tests.support.request_builders import whole_structure_requested_goals
from tests.support.retained_reference_fixtures import LIGAND, reference_structure

import protrepair.state.reference_hydrogen as reference_chemistry
from protrepair.api import process_structure
from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.chemistry.inference.idealized_stereochemistry import (
    require_compatible_component_reference,
)
from protrepair.chemistry.nonstandard.registry import build_bundled_nonstandard_registry
from protrepair.chemistry.retained_non_polymer.hydrogen_correspondence import (
    source_hydrogen_bond_length_is_compatible,
)
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.errors import RdkitUnavailableError
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.state import HydrogenCoverageState
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.reference_hydrogen import ReferenceHydrogenParentInterpretation
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryFailureKind,
    RetainedNonPolymerChemistryResolution,
    RetainedReferenceHydrogenInventory,
)
from protrepair.structure import AtomRef, ProteinStructure, ResidueId
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.observation import StructureObservation
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
    resolve_retained_non_polymer_topology,
)
from protrepair.transformer.continuous.readiness import (
    ContinuousRelaxationReadinessPolicy,
)
from protrepair.workflow.contracts import LigandPolicy, StructureIngressOptions
from protrepair.workflow.engine.reporting import evaluate_requested_goal_report


@pytest.mark.parametrize("component_id", ("3CF", "NLE", "02L"))
def test_complete_reference_inventory_satisfies_only_h_coverage(
    component_id: str,
) -> None:
    source = reference_structure(component_id, with_h=True)
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )

    assert fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
    assert fact.reference_hydrogens_verified
    assert fact.has_complete_reference_hydrogens()
    assert not fact.hydrogen_generation_supported
    assert not fact.is_supported()
    assert not fact.requires_hydrogen_completion()
    assert not fact.requires_topology_resolution()

    for operation in (
        add_retained_non_polymer_hydrogens,
        resolve_retained_non_polymer_topology,
    ):
        result = operation(source, build_default_component_library())
        assert result.structure == source
        assert not result.repairs
        assert not result.issues

    result = process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert result.structure == source
    assert result.requested_goal_report is not None
    outcome = next(
        outcome
        for outcome in result.requested_goal_report.outcomes
        if isinstance(outcome.requested_goal.value, HydrogenCoverageState)
    )
    assert outcome.observed_state is HydrogenCoverageState.COMPLETE
    assert result.requested_goals_fully_satisfied()
    assert not any(
        issue.kind is ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN
        and issue.residue_id == LIGAND
        for issue in result.issues
    )


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
@pytest.mark.parametrize("component_id", ("3CF", "02L"))
def test_file_inputs_are_verified_without_new_connectivity(
    component_id: str,
    file_format: FileFormat,
) -> None:
    source = reference_structure(component_id, with_h=True)
    source = read_structure_string(
        write_structure_string(source, file_format), file_format
    )
    before = (source.constitution, source.geometry, source.topology, source.provenance)
    observation = StructureChemistryObservation(source)
    fact = observation.whole_structure_facts()[1].retained_non_polymer_facts[0]
    assert fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
    assert (
        source.constitution,
        source.geometry,
        source.topology,
        source.provenance,
    ) == before


def test_partial_inventory_is_known_but_cannot_be_generated() -> None:
    source = reference_structure("3CF", with_h=True)
    hydrogen = next(
        a.name for a in source.constitution.ligands[0].atom_sites if a.is_hydrogen()
    )
    source = source.without_atom_refs((AtomRef(LIGAND, hydrogen),))
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_coverage_state is HydrogenCoverageState.PARTIAL
    assert fact.reference_hydrogens_verified
    assert not fact.has_complete_reference_hydrogens()
    assert fact.supports_hydrogen_verification()
    assert not fact.requires_hydrogen_completion()
    result = add_retained_non_polymer_hydrogens(source)
    assert result.structure == source and not result.repairs
    assert len(result.issues) == 1


@pytest.mark.parametrize(
    "parents",
    (
        (
            ("HX", "N", ReferenceHydrogenParentInterpretation.REFERENCE_NAME),
            ("HX", "N", ReferenceHydrogenParentInterpretation.REFERENCE_NAME),
        ),
        (("HX", "UNKNOWN", ReferenceHydrogenParentInterpretation.REFERENCE_NAME),),
        (
            ("HX1", "CA", ReferenceHydrogenParentInterpretation.REFERENCE_NAME),
            ("HX2", "CA", ReferenceHydrogenParentInterpretation.REFERENCE_NAME),
        ),
        (("N", "N", ReferenceHydrogenParentInterpretation.REFERENCE_NAME),),
    ),
)
def test_inventory_rejects_duplicate_unknown_and_overfull_parent_groups(
    parents: tuple[
        tuple[str, str, BondProvenance | ReferenceHydrogenParentInterpretation], ...
    ],
) -> None:
    reference = build_default_component_library().require("NLE").reference_component
    assert reference is not None
    with pytest.raises(ValueError):
        RetainedReferenceHydrogenInventory(reference, LIGAND, parents)


@pytest.mark.parametrize("charges", ({"N": 1, "OXT": -1}, {"CE": 1}))
def test_alternative_or_incompatible_charge_is_not_certified(
    charges: dict[str, int],
) -> None:
    source = reference_structure("NLE", with_h=True)
    index = source.constitution.residue_index(LIGAND)
    current = dict(source.residue_formal_charge_by_atom_name(index))
    current.update(charges)
    source = source.with_updated_residue_facets(
        source.constitution.ligands[0],
        residue_geometry=source.residue_geometry(index),
        formal_charge_by_atom_name=tuple(current.items()),
    )
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_coverage_state is None
    assert fact.chemistry_failure is not None
    assert (
        fact.chemistry_failure.kind
        is RetainedNonPolymerChemistryFailureKind.REFERENCE_STATE_NOT_COVERED
    )


def _with_bonds(
    source: ProteinStructure, bonds: tuple[TopologyBond, ...]
) -> ProteinStructure:
    return ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=bonds,
        ),
        provenance=source.provenance,
    )


def _renamed_hydrogens(source: ProteinStructure, *, permute: bool) -> ProteinStructure:
    residue = source.constitution.ligands[0]
    old_names = tuple(atom.name for atom in residue.atom_sites if atom.is_hydrogen())
    new_names = (
        old_names[1:] + old_names[:1]
        if permute
        else tuple(f"Q{i:03d}" for i in range(len(old_names)))
    )
    names = dict(zip(old_names, new_names, strict=True))
    index = source.constitution.residue_index(residue.residue_id)
    geometry = source.residue_geometry(index)
    return source.with_updated_residue_facets(
        replace(
            residue,
            atom_sites=tuple(
                atom.with_name(names.get(atom.name, atom.name))
                for atom in residue.atom_sites
            ),
        ),
        residue_geometry=ResidueGeometry(
            {
                names.get(name, name): atom
                for name, atom in geometry.atoms_by_name.items()
            }
        ),
        formal_charge_by_atom_name=tuple(
            (names.get(name, name), charge)
            for name, charge in source.residue_formal_charge_by_atom_name(index)
        ),
    )


@pytest.mark.parametrize("component_id", ("3CF", "NLE", "02L"))
@pytest.mark.parametrize("permute", (False, True))
@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_same_state_h_names_do_not_block_read_only_verification(
    component_id: str, permute: bool, isotope: str
) -> None:
    source = _renamed_hydrogens(
        reference_structure(component_id, with_h=True, isotope=isotope),
        permute=permute,
    )
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.has_complete_reference_hydrogens()
    assert not fact.requires_hydrogen_completion()
    assert not fact.requires_topology_resolution()
    result = process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert result.structure == source
    assert result.requested_goals_fully_satisfied()
    assert not any(
        issue.kind is ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN
        for issue in result.issues
    )


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_arbitrary_source_h_names_survive_file_round_trips(
    file_format: FileFormat,
) -> None:
    source = _renamed_hydrogens(reference_structure("3CF", with_h=True), permute=False)
    source = read_structure_string(
        write_structure_string(source, file_format), file_format
    )
    original = source
    result = process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert result.structure == original
    assert result.requested_goals_fully_satisfied()


@pytest.mark.parametrize("rename", (False, True))
def test_reference_lengths_are_prepared_lazily_once(
    rename: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = reference_structure("3CF", with_h=True)
    if rename:
        source = _renamed_hydrogens(source, permute=False)
    prepare = Mock(
        side_effect=reference_chemistry.component_reference_hydrogen_bond_lengths
    )
    monkeypatch.setattr(
        reference_chemistry, "component_reference_hydrogen_bond_lengths", prepare
    )
    resolution = RetainedNonPolymerChemistryResolution.observe_unprepared_reference(
        build_default_component_library().require("3CF"),
        source,
        source.constitution.ligands[0],
    )
    assert resolution is not None and resolution.has_complete_reference_hydrogens()
    assert prepare.call_count == int(rename)
    inventory = resolution.reference_hydrogen_inventory
    assert inventory is not None
    assert {basis for _, _, basis in inventory.hydrogen_parents} == {
        ReferenceHydrogenParentInterpretation.COMPATIBLE_GEOMETRY
        if rename
        else ReferenceHydrogenParentInterpretation.REFERENCE_NAME
    }


@pytest.mark.parametrize("backend_failure", (False, True))
def test_failed_reference_correspondence_is_typed_and_preserves_source(
    backend_failure: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    source = _renamed_hydrogens(reference_structure("3CF", with_h=True), permute=False)
    monkeypatch.setattr(
        reference_chemistry,
        "component_reference_hydrogen_bond_lengths",
        Mock(
            side_effect=RdkitUnavailableError("unavailable")
            if backend_failure
            else ValueError("no temporary pose")
        ),
    )
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_coverage_state is None
    assert fact.chemistry_failure is not None
    assert fact.chemistry_failure.kind is (
        RetainedNonPolymerChemistryFailureKind.BACKEND_UNAVAILABLE
        if backend_failure
        else RetainedNonPolymerChemistryFailureKind.HYDROGEN_PARENT_UNRESOLVED
    )
    result = add_retained_non_polymer_hydrogens(source)
    assert result.structure == source
    assert not result.repairs
    assert len(result.issues) == 1


def test_arbitrary_h_names_do_not_make_remote_h_correspondence_valid() -> None:
    original = reference_structure("3CF", with_h=True)
    old_names = tuple(
        atom.name
        for atom in original.constitution.ligands[0].atom_sites
        if atom.is_hydrogen()
    )
    target = f"Q{old_names.index('HXT'):03d}"
    source = _renamed_hydrogens(original, permute=False)
    geometry = source.residue_geometry(source.constitution.residue_index(LIGAND))
    source = source.with_updated_residue_geometries(
        (
            (
                LIGAND,
                geometry.with_atom_geometry(
                    target,
                    geometry.atom_geometry(target).with_position(
                        geometry.position(target).with_offset(3, 0, 0)
                    ),
                ),
            ),
        )
    )
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_coverage_state is None
    assert fact.chemistry_failure is not None
    assert (
        fact.chemistry_failure.kind
        is RetainedNonPolymerChemistryFailureKind.HYDROGEN_PARENT_UNRESOLVED
    )


def test_geometric_correspondence_does_not_force_another_state_to_fit() -> None:
    source = reference_structure("3CF", with_h=True)
    reference = build_default_component_library().require("3CF").reference_component
    assert reference is not None
    on_nitrogen = next(
        atom.atom_name
        for atom in reference.hydrogen_atoms()
        if reference.hydrogen_anchor_atom_name(atom.atom_name) == "N"
    )
    index = source.constitution.residue_index(LIGAND)
    geometry = source.residue_geometry(index)
    source = source.with_updated_residue_facets(
        source.constitution.ligands[0],
        residue_geometry=geometry.with_atom_geometry(
            "HXT",
            geometry.atom_geometry("HXT").with_position(geometry.position(on_nitrogen)),
        ),
        formal_charge_by_atom_name=tuple(
            (atom.name, None) for atom in source.constitution.ligands[0].atom_sites
        ),
    )
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_coverage_state is None
    assert fact.chemistry_failure is not None
    assert (
        fact.chemistry_failure.kind
        is RetainedNonPolymerChemistryFailureKind.REFERENCE_STATE_NOT_COVERED
    )


@pytest.mark.parametrize(
    "source_distance, prepared_distance, expected",
    (
        (1.0, 1.2, True),
        (1.0, 1.5, False),
        (float("nan"), 1.0, False),
        (float("inf"), 1.0, False),
    ),
)
def test_shared_source_prepared_length_policy(
    source_distance: float, prepared_distance: float, expected: bool
) -> None:
    assert (
        source_hydrogen_bond_length_is_compatible(source_distance, prepared_distance)
        is expected
    )


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
@pytest.mark.parametrize("parent_basis", ("name", "current", "original"))
def test_name_parent_consistency_does_not_override_connections(
    isotope: str, parent_basis: str
) -> None:
    source = reference_structure("3CF", with_h=True, isotope=isotope)
    if parent_basis != "name":
        original = _with_bonds(
            source,
            source.topology.bonds
            + (
                TopologyBond(
                    source.constitution.atom_index(AtomRef(LIGAND, "HXT")),
                    source.constitution.atom_index(AtomRef(LIGAND, "OXT")),
                    provenance=BondProvenance.SOURCE_EXPLICIT
                    if parent_basis == "original"
                    else BondProvenance.TEMPLATE_RESOLVED,
                    relationship_type=BondRelationshipType.UNKNOWN
                    if parent_basis == "original"
                    else BondRelationshipType.COVALENT,
                    order=None if parent_basis == "original" else 1,
                ),
            ),
        )
        if parent_basis == "current":
            source = original
        else:
            source = source.with_provenance(
                replace(
                    source.provenance,
                    ingress=replace(
                        source.provenance.ingress,
                        observation=StructureObservation.from_source_facets(
                            constitution=original.constitution,
                            geometry=original.geometry,
                            topology=original.topology,
                        ),
                    ),
                )
            )

    index = source.constitution.residue_index(LIGAND)
    geometry = source.residue_geometry(index)
    source = source.with_updated_residue_geometries(
        (
            (
                LIGAND,
                geometry.with_atom_geometry(
                    "HXT",
                    geometry.atom_geometry("HXT").with_position(
                        geometry.position("N").with_offset(0, 0, 0.1)
                    ),
                ),
            ),
        )
    )
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    if parent_basis != "name":
        assert fact.has_complete_reference_hydrogens()
        assert not fact.hydrogen_generation_supported
        return

    assert fact.hydrogen_coverage_state is None
    assert not fact.reference_hydrogens_verified
    assert fact.chemistry_failure is not None
    assert (
        fact.chemistry_failure.kind
        is RetainedNonPolymerChemistryFailureKind.HYDROGEN_PARENT_UNRESOLVED
    )
    repaired = add_retained_non_polymer_hydrogens(source)
    assert repaired.structure == source
    assert not repaired.repairs
    assert len(repaired.issues) == 1
    result = process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert result.structure == source
    assert not result.requested_goals_fully_satisfied()
    assert result.requested_goal_report is not None
    outcome = next(
        outcome
        for outcome in result.requested_goal_report.outcomes
        if isinstance(outcome.requested_goal.value, HydrogenCoverageState)
    )
    assert outcome.observed_state is None
    assert outcome.blocking_scopes
    assert outcome.details is not None
    assert fact.chemistry_failure.detail in outcome.details


@pytest.mark.parametrize("reverse_atoms", (False, True))
def test_name_parent_ties_do_not_depend_on_atom_order(reverse_atoms: bool) -> None:
    source = reference_structure("NLE", with_h=True)
    residue = source.constitution.ligands[0]
    geometry = source.residue_geometry(source.constitution.residue_index(LIGAND))
    geometry = geometry.with_atom_geometries(
        (name, geometry.atom_geometry(name).with_position(geometry.position("C")))
        for name in ("HXT", "OXT")
    )
    assert geometry.position("HXT").distance_to(geometry.position("OXT")) == 0
    assert geometry.position("HXT").distance_to(geometry.position("C")) == 0
    if reverse_atoms:
        residue = replace(residue, atom_sites=tuple(reversed(residue.atom_sites)))
    source = source.with_updated_residue_facets(residue, residue_geometry=geometry)
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.has_complete_reference_hydrogens()


def test_nearest_name_parent_check_does_not_impose_a_distance_bound() -> None:
    source = reference_structure("3CF", with_h=True)
    geometry = source.residue_geometry(source.constitution.residue_index(LIGAND))
    position = geometry.position("HXT").with_offset(3, 0, 0)
    assert position.distance_to(geometry.position("OXT")) > 3
    source = source.with_updated_residue_geometries(
        (
            (
                LIGAND,
                geometry.with_atom_geometry(
                    "HXT", geometry.atom_geometry("HXT").with_position(position)
                ),
            ),
        )
    )
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.has_complete_reference_hydrogens()
    assert not fact.hydrogen_generation_supported


@pytest.mark.parametrize("x", (float("nan"), float("inf")))
def test_nonfinite_name_parent_distance_cannot_establish_coverage(x: float) -> None:
    source = reference_structure("3CF", with_h=True)
    geometry = source.residue_geometry(source.constitution.residue_index(LIGAND))
    source = source.with_updated_residue_geometries(
        (
            (
                LIGAND,
                geometry.with_atom_geometry(
                    "HXT",
                    geometry.atom_geometry("HXT").with_position(Vec3(x, 0, 0)),
                ),
            ),
        )
    )
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_coverage_state is None
    assert fact.chemistry_failure is not None
    assert (
        fact.chemistry_failure.kind
        is RetainedNonPolymerChemistryFailureKind.HYDROGEN_PARENT_UNRESOLVED
    )


@pytest.mark.parametrize("explicit", (False, True))
def test_unknown_h_name_can_use_geometry_without_publishing_bonds(
    explicit: bool,
) -> None:
    source = reference_structure("NLE", with_h=True)
    residue = source.constitution.ligands[0]
    old = next(atom for atom in residue.atom_sites if atom.is_hydrogen())
    template = build_default_component_library().require("NLE").reference_component
    assert template is not None
    parent = template.hydrogen_anchor_atom_name(old.name)
    assert parent is not None
    index = source.constitution.residue_index(LIGAND)
    geometry = source.residue_geometry(index)
    source = source.with_updated_residue_facets(
        replace(
            residue,
            atom_sites=tuple(
                atom.with_name("HX99") if atom.name == old.name else atom
                for atom in residue.atom_sites
            ),
        ),
        residue_geometry=ResidueGeometry(
            {
                "HX99" if name == old.name else name: atom
                for name, atom in geometry.atoms_by_name.items()
            }
        ),
        formal_charge_by_atom_name=tuple(
            ("HX99" if name == old.name else name, charge)
            for name, charge in source.residue_formal_charge_by_atom_name(index)
        ),
    )
    if explicit:
        source = _with_bonds(
            source,
            source.topology.bonds
            + (
                TopologyBond(
                    source.constitution.atom_index(AtomRef(LIGAND, "HX99")),
                    source.constitution.atom_index(AtomRef(LIGAND, parent)),
                    relationship_type=BondRelationshipType.UNKNOWN,
                    order=None,
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                ),
            ),
        )
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.has_complete_reference_hydrogens()
    assert fact.hydrogen_topology_availability_state.is_unavailable()
    assert not fact.requires_topology_resolution()


def test_multiple_parent_edges_are_not_replaced_by_reference_naming() -> None:
    source = reference_structure("NLE", with_h=True)
    hydrogen = next(
        a.name for a in source.constitution.ligands[0].atom_sites if a.is_hydrogen()
    )
    source = _with_bonds(
        source,
        source.topology.bonds
        + tuple(
            TopologyBond(
                source.constitution.atom_index(AtomRef(LIGAND, hydrogen)),
                source.constitution.atom_index(AtomRef(LIGAND, parent)),
                provenance=BondProvenance.SOURCE_EXPLICIT,
            )
            for parent in ("N", "CA")
        ),
    )
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_coverage_state is None
    assert fact.chemistry_failure is not None
    assert (
        fact.chemistry_failure.kind
        is RetainedNonPolymerChemistryFailureKind.HYDROGEN_PARENT_UNRESOLVED
    )


def test_original_parent_is_not_hidden_when_current_edge_is_removed() -> None:
    source = reference_structure("NLE", with_h=True)
    hydrogen = next(
        a.name for a in source.constitution.ligands[0].atom_sites if a.is_hydrogen()
    )
    original = _with_bonds(
        source,
        source.topology.bonds
        + (
            TopologyBond(
                source.constitution.atom_index(AtomRef(LIGAND, hydrogen)),
                source.constitution.atom_index(AtomRef(LIGAND, "CA")),
                provenance=BondProvenance.SOURCE_EXPLICIT,
            ),
        ),
    )
    observation = StructureObservation.from_source_facets(
        constitution=original.constitution,
        geometry=original.geometry,
        topology=original.topology,
    )
    source = source.with_provenance(
        replace(
            source.provenance,
            ingress=replace(source.provenance.ingress, observation=observation),
        )
    )
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_coverage_state is None


def test_pose_change_invalidates_previous_reference_assessment() -> None:
    source = reference_structure("NLE", with_h=True)
    observation = StructureChemistryObservation(source)
    assert (
        observation.whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
        .hydrogen_coverage_state
        is HydrogenCoverageState.COMPLETE
    )
    index = source.constitution.residue_index(LIGAND)
    geometry = source.residue_geometry(index)
    center = geometry.position("CA")
    reflected = ResidueGeometry(
        {
            name: replace(
                atom,
                position=Vec3(
                    2 * center.x - atom.position.x, atom.position.y, atom.position.z
                ),
            )
            for name, atom in geometry.atoms_by_name.items()
        }
    )
    changed = source.with_updated_residue_geometries(((LIGAND, reflected),))
    with pytest.raises(ValueError):
        observation.require_context(
            changed, component_library=build_default_component_library()
        )
    fact = (
        StructureChemistryObservation(changed)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_coverage_state is None


def test_unavailable_native_validation_is_a_reason_not_a_process_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "protrepair.chemistry.inference.idealized_stereochemistry.Chem", None
    )
    source = reference_structure("NLE", with_h=True)
    result = add_retained_non_polymer_hydrogens(source)
    assert result.structure == source and len(result.issues) == 1
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_coverage_state is None
    assert fact.chemistry_failure is not None
    assert (
        fact.chemistry_failure.kind
        is RetainedNonPolymerChemistryFailureKind.BACKEND_UNAVAILABLE
    )


def test_verified_h_does_not_admit_local_refinement(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ff = Mock(side_effect=AssertionError("read-only validation grants no FF rights"))
    monkeypatch.setattr(
        "protrepair.transformer.continuous.rdkit.build_rdkit_force_field", ff
    )
    source = reference_structure("NLE", with_h=True)
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
    assert ContinuousRelaxationReadinessPolicy().retained_non_polymer_readiness_blocker(
        (fact,)
    )
    ff.assert_not_called()


def test_unknown_charge_is_compatible_without_becoming_zero() -> None:
    source = reference_structure("NLE", with_h=True)
    index = source.constitution.residue_index(LIGAND)
    source = source.with_updated_residue_facets(
        source.constitution.ligands[0],
        residue_geometry=source.residue_geometry(index),
    )
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
    result = add_retained_non_polymer_hydrogens(source)
    assert result.structure == source
    assert all(
        value is None for _, value in source.residue_formal_charge_by_atom_name(index)
    )


def test_unreferenced_false_local_complete_does_not_bypass_goal_support(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        "protrepair.state.retained_non_polymer_chemistry.infer_retained_non_polymer_rdkit_fallback",
        Mock(side_effect=ValueError("unsupported unknown chemistry")),
    )
    source = reference_structure("02L", with_h=True)
    library = ComponentLibrary({})
    observation = StructureChemistryObservation(source, component_library=library)
    fact = observation.whole_structure_facts()[1].retained_non_polymer_facts[0]
    assert fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
    assert not fact.supports_hydrogen_verification()
    report = evaluate_requested_goal_report(
        source,
        component_library=library,
        chemistry_observation=observation,
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    outcome = next(
        o
        for o in report.outcomes
        if isinstance(o.requested_goal.value, HydrogenCoverageState)
    )
    assert outcome.observed_state is None and not outcome.is_satisfied()
    index = source.constitution.residue_index(LIGAND)
    unknown = source.with_updated_residue_facets(
        source.constitution.ligands[0].with_component_id("ZZQ"),
        residue_geometry=source.residue_geometry(index),
        formal_charge_by_atom_name=source.residue_formal_charge_by_atom_name(index),
    )
    public = process_structure(
        unknown,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert not public.requested_goals_fully_satisfied()
    assert public.structure == unknown


@pytest.mark.parametrize(
    "relationship",
    (
        BondRelationshipType.COVALENT,
        BondRelationshipType.DISULFIDE,
        BondRelationshipType.METAL_COORDINATION,
        BondRelationshipType.UNKNOWN,
    ),
)
@pytest.mark.parametrize("hide_current", (False, True))
def test_external_source_context_is_not_hidden_by_reference_validation(
    relationship: BondRelationshipType,
    hide_current: bool,
) -> None:
    original = reference_structure("NLE", with_h=True)
    residue = original.constitution.ligands[0]
    index = original.constitution.residue_index(LIGAND)
    partner = ResidueId("M", 1)
    original = build_bonded_structure(
        chains=(),
        ligands=(
            (
                residue,
                original.residue_geometry(index),
                original.residue_formal_charge_by_atom_name(index),
            ),
            residue_payload(
                component_id="ZN",
                residue_id=partner,
                atoms=(atom_payload("ZN", "ZN", Vec3(20, 0, 0), formal_charge=2),),
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    original = _with_bonds(
        original,
        original.topology.bonds
        + (
            TopologyBond(
                original.constitution.atom_index(AtomRef(LIGAND, "N")),
                original.constitution.atom_index(AtomRef(partner, "ZN")),
                relationship_type=relationship,
                order=None
                if relationship
                in {
                    BondRelationshipType.METAL_COORDINATION,
                    BondRelationshipType.UNKNOWN,
                }
                else 1,
                provenance=BondProvenance.SOURCE_EXPLICIT,
            ),
        ),
    )
    observation = StructureObservation.from_source_facets(
        constitution=original.constitution,
        geometry=original.geometry,
        topology=original.topology,
    )
    source = original.with_provenance(
        replace(
            original.provenance,
            ingress=replace(original.provenance.ingress, observation=observation),
        )
    )
    if hide_current:
        source = source.without_atom_refs((AtomRef(partner, "ZN"),))
    fact = (
        StructureChemistryObservation(source)
        .projected_facts((LIGAND,))[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_coverage_state is None
    assert not fact.reference_hydrogens_verified
    assert fact.chemistry_failure is not None
    assert not fact.supports_hydrogen_verification()


@pytest.mark.parametrize("with_h", (False, True))
@pytest.mark.parametrize("component_id, missing", (("NLE", "CE"), ("3CF", "N1")))
def test_complete_heavy_scaffold_is_required_for_reference_verification(
    with_h: bool,
    component_id: str,
    missing: str,
) -> None:
    source = reference_structure(component_id, with_h=with_h).without_atom_refs(
        (AtomRef(LIGAND, missing),)
    )
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_coverage_state is (
        None if with_h else HydrogenCoverageState.NONE
    )
    assert not fact.supports_hydrogen_verification()
    assert not fact.requires_hydrogen_completion()
    assert fact.chemistry_failure is not None
    assert fact.chemistry_failure.kind is (
        RetainedNonPolymerChemistryFailureKind.REFERENCE_STATE_NOT_COVERED
        if with_h
        else RetainedNonPolymerChemistryFailureKind.NO_ADMITTED_PREPARATION
    )

    result = process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert result.structure == source
    assert result.requested_goal_report is not None
    outcome = next(
        outcome
        for outcome in result.requested_goal_report.outcomes
        if isinstance(outcome.requested_goal.value, HydrogenCoverageState)
    )
    assert outcome.details is not None
    assert fact.chemistry_failure.detail in outcome.details
    warnings = tuple(
        issue
        for issue in result.issues
        if issue.kind is ValidationIssueKind.UNRESOLVED_RETAINED_HYDROGEN
    )
    assert len(warnings) == 1
    assert fact.chemistry_failure.detail in warnings[0].message


def test_reference_replacement_invalidates_validation_not_atom_payload() -> None:
    source = reference_structure("NLE", with_h=True)
    library = build_default_component_library()
    template = library.require("NLE")
    reference = template.reference_component
    assert reference is not None
    invalid = replace(
        reference,
        atoms=tuple(
            replace(atom, ideal_position=None) if atom.atom_name == "CA" else atom
            for atom in reference.atoms
        ),
    )
    replaced = library.with_template(replace(template, reference_component=invalid))
    fact = (
        StructureChemistryObservation(source, component_library=replaced)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_coverage_state is None
    assert fact.chemistry_failure is not None
    assert (
        fact.chemistry_failure.kind
        is RetainedNonPolymerChemistryFailureKind.REFERENCE_INVALID
    )
    assert add_retained_non_polymer_hydrogens(source, replaced).structure == source
    assert (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
        .hydrogen_coverage_state
        is HydrogenCoverageState.COMPLETE
    )


@pytest.mark.parametrize("mirror", (False, True))
@pytest.mark.parametrize("rename", (False, True))
def test_all_bundled_reference_self_inventories_keep_native_refusals(
    mirror: bool,
    rename: bool,
) -> None:
    examined = 0
    for record in build_bundled_nonstandard_registry().records.values():
        template = record.to_template()
        if not template.has_unprepared_hydrogen_reference():
            continue
        reference = record.to_idealized_component()
        source = build_bonded_structure(
            chains=(),
            ligands=(
                residue_payload(
                    component_id=record.component_id,
                    residue_id=LIGAND,
                    atoms=tuple(
                        atom_payload(
                            f"OBS_H_{index}"
                            if rename and atom.is_hydrogen()
                            else atom.atom_name,
                            atom.element,
                            Vec3(
                                (-1 if mirror else 1)
                                * (atom.ideal_position or (0.0, 0.0, 0.0))[0],
                                *(atom.ideal_position or (0.0, 0.0, 0.0))[1:],
                            ),
                            formal_charge=atom.formal_charge,
                        )
                        for index, atom in enumerate(record.atoms)
                    ),
                ),
            ),
            source_format=FileFormat.MMCIF,
        )
        resolution = RetainedNonPolymerChemistryResolution.observe_unprepared_reference(
            template, source, source.constitution.ligands[0]
        )
        assert resolution is not None and not resolution.can_generate_hydrogens()
        try:
            require_compatible_component_reference(reference)
        except ValueError:
            assert resolution.reference_hydrogen_inventory is None
            assert resolution.failure is not None
            assert (
                resolution.failure.kind
                is RetainedNonPolymerChemistryFailureKind.REFERENCE_INVALID
            )
        else:
            if mirror and any(atom.stereo_config for atom in record.atoms):
                assert resolution.reference_hydrogen_inventory is None, (
                    record.component_id
                )
                assert resolution.failure is not None
                assert (
                    resolution.failure.kind
                    is RetainedNonPolymerChemistryFailureKind.SOURCE_CONFLICT
                )
            else:
                assert resolution.reference_hydrogen_inventory is not None, (
                    record.component_id,
                    resolution.failure,
                )
                assert (
                    resolution.reference_hydrogen_inventory.coverage_state()
                    is HydrogenCoverageState.COMPLETE
                )
        examined += 1
    assert examined > 0


def test_ambiguous_unnamed_parent_is_not_chosen_to_fill_reference_counts(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = reference_structure("NLE", with_h=True)
    names = tuple(
        atom.name
        for atom in original.constitution.ligands[0].atom_sites
        if atom.is_hydrogen()
    )
    target = f"Q{names.index('HXT'):03d}"
    source = _renamed_hydrogens(original, permute=False)
    residue = source.constitution.ligands[0]
    geometry = source.residue_geometry(source.constitution.residue_index(LIGAND))
    geometry = geometry.with_atom_geometries(
        (name, geometry.atom_geometry(name).with_position(geometry.position("C")))
        for name in (target, "OXT")
    )
    source = source.with_updated_residue_facets(
        replace(
            residue,
            atom_sites=(
                residue.atom_site(target),
                *(atom for atom in residue.atom_sites if atom.name != target),
            ),
        ),
        residue_geometry=geometry,
    )
    prepare = Mock(side_effect=AssertionError("ambiguous parent must not be assigned"))
    monkeypatch.setattr(
        reference_chemistry, "component_reference_hydrogen_bond_lengths", prepare
    )
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.hydrogen_coverage_state is None
    assert fact.chemistry_failure is not None
    assert (
        fact.chemistry_failure.kind
        is RetainedNonPolymerChemistryFailureKind.HYDROGEN_PARENT_UNRESOLVED
    )
    prepare.assert_not_called()


def test_renamed_partial_h_inventory_is_verified_without_generation() -> None:
    source = _renamed_hydrogens(reference_structure("3CF", with_h=True), permute=False)
    source = source.without_atom_refs((AtomRef(LIGAND, "Q000"),))
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert fact.reference_hydrogens_verified
    assert fact.hydrogen_coverage_state is HydrogenCoverageState.PARTIAL
    assert not fact.requires_hydrogen_completion()
    result = add_retained_non_polymer_hydrogens(source)
    assert result.structure == source
    assert len(result.issues) == 1

"""Known polymer reference chemistry must not imply admitted H preparation."""

from dataclasses import replace

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_bonded_structure,
    chain_payload,
    residue_payload,
)
from tests.support.request_builders import whole_structure_requested_goals

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.nonstandard.registry import build_bundled_nonstandard_registry
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.state import HydrogenCoverageState, TopologyAvailabilityState
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.structure import AtomRef, ProteinStructure, ResidueId
from protrepair.structure.constitution import AtomSite
from protrepair.structure.provenance import FileFormat

RESIDUE = ResidueId("X", 2)


def _reference_polymer(
    component_id: str,
    *,
    with_h: bool,
    linked: bool,
    rename_h: bool = False,
    isotope: str = "H",
) -> ProteinStructure:
    record = build_bundled_nonstandard_registry().get(component_id)
    assert record is not None
    reference = record.to_idealized_component()
    atoms = []
    nitrogen_h = 0
    for atom in record.atoms:
        if atom.is_hydrogen() and not with_h:
            continue
        if linked:
            parent = reference.hydrogen_anchor_atom_name(atom.atom_name)
            if atom.atom_name == "OXT" or parent == "OXT":
                continue
            if parent == "N":
                nitrogen_h += 1
                if nitrogen_h > 1:
                    continue
        assert atom.ideal_position is not None
        name = (
            f"Q{len(atoms):03d}" if rename_h and atom.is_hydrogen() else atom.atom_name
        )
        atoms.append(
            atom_payload(
                name,
                isotope if atom.is_hydrogen() else atom.element,
                Vec3(*atom.ideal_position),
            )
        )
    payload = residue_payload(
        component_id=component_id, residue_id=RESIDUE, atoms=tuple(atoms)
    )
    residues = (payload,)
    if linked:
        neighbors = tuple(
            residue_payload(
                component_id="ALA",
                residue_id=ResidueId("X", number),
                atoms=tuple(
                    atom_payload(name, name[0], Vec3(number * 30, index * 3, 0))
                    for index, name in enumerate(("N", "CA", "C", "O", "CB"))
                ),
            )
            for number in (1, 3)
        )
        residues = (neighbors[0], payload, neighbors[1])
    return build_bonded_structure(
        chains=(chain_payload("X", residues),), source_format=FileFormat.MMCIF
    )


@pytest.mark.parametrize("component_id", ("3CF", "CGU", "NLE", "KCX"))
@pytest.mark.parametrize("linked", (False, True))
def test_missing_body_h_is_indeterminate_not_complete(
    component_id: str, linked: bool
) -> None:
    source = _reference_polymer(component_id, with_h=False, linked=linked)
    observation = StructureChemistryObservation(source)
    facts = observation.whole_structure_facts()[1]
    fact = next(fact for fact in facts.residue_facts if fact.residue_id == RESIDUE)
    assert fact.hydrogen_coverage_state is None
    assert facts.hydrogen_coverage_state is None
    assessment = observation.hydrogen_expectation_model.polymer_reference_assessments[
        RESIDUE
    ]
    assert assessment.failure is not None
    assert assessment.failure.kind == "no_admitted_preparation"


@pytest.mark.parametrize("component_id", ("3CF", "CGU", "NLE", "KCX"))
@pytest.mark.parametrize("linked", (False, True))
def test_full_h_is_verified_with_actual_peptide_boundary(
    component_id: str, linked: bool
) -> None:
    source = _reference_polymer(component_id, with_h=True, linked=linked)
    before = (source.constitution, source.geometry, source.topology, source.provenance)
    observation = StructureChemistryObservation(source)
    fact = observation.residue_completion_facts(RESIDUE).chemistry_readiness_facts
    assessment = observation.hydrogen_expectation_model.polymer_reference_assessments[
        RESIDUE
    ]
    assert assessment.failure is None
    assert fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
    assert (
        source.constitution,
        source.geometry,
        source.topology,
        source.provenance,
    ) == before


@pytest.mark.parametrize("component_id", ("3CF", "NLE", "CGU"))
def test_public_missing_body_h_returns_unmet_reason_once(component_id: str) -> None:
    source = _reference_polymer(component_id, with_h=False, linked=False)
    result = process_structure(
        source,
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert not result.requested_goals_fully_satisfied()
    assert result.requested_goal_report is not None
    outcome = next(
        outcome
        for outcome in result.requested_goal_report.outcomes
        if isinstance(outcome.requested_goal.value, HydrogenCoverageState)
    )
    assert outcome.observed_state is None
    assert outcome.blocking_scopes
    assert outcome.details
    warnings = [
        issue
        for issue in result.issues
        if issue.kind is ValidationIssueKind.UNRESOLVED_POLYMER_HYDROGEN
        and issue.residue_id == RESIDUE
    ]
    assert len(warnings) == 1
    residue = result.structure.constitution.residue_or_ligand(RESIDUE)
    assert residue is not None
    assert not residue.has_atom_site("HA")


@pytest.mark.parametrize("component_id", ("3CF", "CGU", "KCX"))
def test_public_complete_source_is_preserved(component_id: str) -> None:
    source = _reference_polymer(component_id, with_h=True, linked=False)
    result = process_structure(
        source,
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert result.requested_goals_fully_satisfied()
    assert result.structure == source


def test_partial_reference_h_is_not_promoted_to_complete() -> None:
    source = _reference_polymer("3CF", with_h=True, linked=True)
    source = source.without_atom_refs((AtomRef(RESIDUE, "HA"),))
    fact = (
        StructureChemistryObservation(source)
        .residue_completion_facts(RESIDUE)
        .chemistry_readiness_facts
    )
    assert fact.hydrogen_coverage_state is None


def test_reference_elements_are_not_inferred_from_atom_names() -> None:
    library = build_default_component_library()
    templates = tuple(
        template
        for template in library.templates.values()
        if template.reference_component is not None
    )
    halogenated = next(
        template
        for template in templates
        if template.reference_component is not None
        and any(
            atom.element in {"CL", "BR"}
            for atom in template.reference_component.heavy_atoms()
        )
    )
    source = _reference_polymer(halogenated.component_id, with_h=False, linked=False)
    preparation = StructureChemistryObservation(
        source
    ).hydrogen_expectation_model.polymer_preparation
    targets = preparation.targets_for(RESIDUE)
    assert targets
    assert all(
        "scaffold" not in " ".join(target.resolution.details) for target in targets
    )


@pytest.mark.parametrize("linked", (False, True))
@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_renamed_isotopic_h_are_verified_without_missing_heavy_coordinates(
    linked: bool,
    isotope: str,
) -> None:
    source = _reference_polymer(
        "3CF", with_h=True, linked=linked, rename_h=True, isotope=isotope
    )
    before = source
    observation = StructureChemistryObservation(source)
    assessment = observation.hydrogen_expectation_model.polymer_reference_assessments[
        RESIDUE
    ]
    assert assessment.failure is None
    assert assessment.coverage_state is HydrogenCoverageState.COMPLETE
    assert source == before


@pytest.mark.parametrize("linked", (False, True))
@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_reference_validation_survives_file_ingress(
    linked: bool, file_format: FileFormat
) -> None:
    source = _reference_polymer("3CF", with_h=True, linked=linked)
    source = read_structure_string(
        write_structure_string(source, file_format), file_format
    )
    fact = (
        StructureChemistryObservation(source)
        .residue_completion_facts(RESIDUE)
        .chemistry_readiness_facts
    )
    assert fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE


def test_body_charge_conflict_is_not_hidden_by_reference_completeness() -> None:
    source = _reference_polymer("3CF", with_h=True, linked=True)
    index = source.constitution.residue_index(RESIDUE)
    residue = source.constitution.residue_site_at(index)
    source = source.with_updated_residue_facets(
        residue,
        residue_geometry=source.residue_geometry(index),
        formal_charge_by_atom_name=(("CA", 1),),
    )
    assessment = StructureChemistryObservation(
        source
    ).hydrogen_expectation_model.polymer_reference_assessments[RESIDUE]
    assert assessment.coverage_state is None
    assert assessment.failure is not None
    assert assessment.failure.kind == "reference_state_not_covered"


def test_reference_parent_observation_is_invalidated_after_geometry_change() -> None:
    source = _reference_polymer("3CF", with_h=True, linked=False)
    model = StructureChemistryObservation(source).hydrogen_expectation_model
    index = source.constitution.residue_index(RESIDUE)
    geometry = source.residue_geometry(index)
    moved = source.with_updated_residue_geometries(
        (
            (
                RESIDUE,
                geometry.with_atom_geometry(
                    "HA",
                    geometry.atom_geometry("HA").with_position(geometry.position("N")),
                ),
            ),
        )
    )
    assert not model.matches_inputs(
        moved, component_library=build_default_component_library()
    )
    assessment = StructureChemistryObservation(
        moved
    ).hydrogen_expectation_model.polymer_reference_assessments[RESIDUE]
    assert assessment.coverage_state is None


def test_missing_heavy_scaffold_cannot_be_verified() -> None:
    source = _reference_polymer("3CF", with_h=True, linked=True)
    source = source.without_atom_refs((AtomRef(RESIDUE, "N1"),))
    assessment = StructureChemistryObservation(
        source
    ).hydrogen_expectation_model.polymer_reference_assessments[RESIDUE]
    assert assessment.coverage_state is None
    assert assessment.failure is not None


def test_verified_coverage_does_not_admit_h_topology_completion() -> None:
    source = _reference_polymer("3CF", with_h=True, linked=False)
    fact = (
        StructureChemistryObservation(source)
        .residue_completion_facts(RESIDUE)
        .chemistry_readiness_facts
    )
    assert fact.hydrogen_coverage_state is HydrogenCoverageState.COMPLETE
    assert (
        fact.hydrogen_topology_availability_state
        is TopologyAvailabilityState.UNSUPPORTED
    )


def test_custom_reference_does_not_authorize_stock_terminal_states() -> None:
    source = _reference_polymer("3CF", with_h=True, linked=False)
    index = source.constitution.residue_index(RESIDUE)
    residue = source.constitution.residue_site_at(index)
    geometry = source.residue_geometry(index)
    source = source.with_updated_residue_facets(
        replace(residue, atom_sites=(*residue.atom_sites, AtomSite("H3", "H"))),
        residue_geometry=geometry.with_atom_geometry(
            "H3",
            geometry.atom_geometry("H").with_position(
                geometry.position("N").with_offset(0, 0, 0.99)
            ),
        ),
    )
    library = build_default_component_library()
    stock = library.require("3CF")
    custom = replace(
        stock, definition=replace(stock.definition, aliases=(*stock.aliases, "ZCF"))
    )
    stock_facts = StructureChemistryObservation(source, component_library=library)
    assert (
        stock_facts.residue_completion_facts(RESIDUE).hydrogen_coverage_state
        is HydrogenCoverageState.COMPLETE
    )
    custom_facts = StructureChemistryObservation(
        source, component_library=library.with_template(custom)
    )
    assert (
        custom_facts.residue_completion_facts(RESIDUE).hydrogen_coverage_state is None
    )

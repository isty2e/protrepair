"""Unit tests for localized side-chain stereochemistry correction."""

from collections.abc import Callable, Collection, Iterable
from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    CanonicalResiduePayload,
    chain_payload,
)
from tests.support.canonical_builders import (
    build_structure as build_canonical_structure,
)

import protrepair.transformer.completion.stereochemistry.batch as batch_module
import protrepair.transformer.completion.stereochemistry.correction as correction_module
from protrepair.chemistry import ComponentLibrary
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.diagnostics import (
    RepairEventKind,
    SidechainStereochemistryViolation,
    StereochemistryReport,
    detect_sidechain_stereochemistry,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.io import read_structure
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.stereochemistry import (
    correct_sidechain_stereochemistry,
)
from protrepair.transformer.completion.stereochemistry.batch import (
    StereochemistryCorrectionBatch,
)
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.contracts import StructureIngressOptions


def test_correct_sidechain_stereochemistry_repairs_inverted_threonine() -> None:
    """Swapped THR substituents should be rebuilt into the supported chirality."""

    structure = focused_structure_for_residue(
        seq_num=30,
        mutate_residue=invert_threonine_residue,
    )

    result = correct_sidechain_stereochemistry(structure)

    assert result.issue_count() == 0
    assert stereochemistry_report(result).is_empty()
    assert any(
        repair.kind is RepairEventKind.STEREOCHEMISTRY_CORRECTED
        for repair in result.repairs
    )
    assert any(
        repair.kind is RepairEventKind.HEAVY_ATOMS_ADDED
        and set(repair.atom_names) == {"OG1", "CG2"}
        for repair in result.repairs
    )


def test_correct_sidechain_stereochemistry_repairs_inverted_isoleucine() -> None:
    """An inverted ILE side chain should be locally rebuilt into native chirality."""

    structure = focused_structure_for_residue(
        seq_num=25,
        mutate_residue=invert_isoleucine_residue,
    )

    result = correct_sidechain_stereochemistry(structure)

    assert result.issue_count() == 0
    assert stereochemistry_report(result).is_empty()
    assert any(
        repair.kind is RepairEventKind.STEREOCHEMISTRY_CORRECTED
        for repair in result.repairs
    )
    assert any(
        repair.kind is RepairEventKind.HEAVY_ATOMS_ADDED
        and set(repair.atom_names) == {"CG1", "CG2", "CD1"}
        for repair in result.repairs
    )


def test_correct_sidechain_stereochemistry_repairs_multiple_residues() -> None:
    """One correction batch should rebuild each selected residue independently."""

    structure = focused_structure_for_residues(
        (
            (25, invert_isoleucine_residue),
            (30, invert_threonine_residue),
        )
    )

    result = correct_sidechain_stereochemistry(structure)

    assert result.issue_count() == 0
    assert stereochemistry_report(result).is_empty()
    assert {
        repair.residue_id
        for repair in result.repairs
        if repair.kind is RepairEventKind.STEREOCHEMISTRY_CORRECTED
    } == {ResidueId("A", 25), ResidueId("A", 30)}
    assert set(
        result.structure.constitution.residue_site_at(
            result.structure.constitution.residue_index(ResidueId("A", 25))
        ).atom_site_names()
    ) >= {"CG1", "CG2", "CD1"}
    assert set(
        result.structure.constitution.residue_site_at(
            result.structure.constitution.residue_index(ResidueId("A", 30))
        ).atom_site_names()
    ) >= {"OG1", "CG2"}


def test_stereochemistry_batch_updates_multiple_residue_facets_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Preparation and merge should each perform one structure batch update."""

    structure = focused_structure_for_residues(
        (
            (25, invert_isoleucine_residue),
            (30, invert_threonine_residue),
        )
    )
    component_library = build_standard_component_library()
    batch = StereochemistryCorrectionBatch.from_violations(
        detect_sidechain_stereochemistry(
            structure,
            component_library=component_library,
        ).violations
    )
    original_batch_update = ProteinStructure.with_updated_residue_facets_batch
    observed_batch_sizes: list[int] = []

    def recording_batch_update(
        candidate_structure: ProteinStructure,
        residue_facets: Iterable[CanonicalResiduePayload],
    ) -> ProteinStructure:
        materialized_facets = tuple(residue_facets)
        observed_batch_sizes.append(len(materialized_facets))
        return original_batch_update(candidate_structure, materialized_facets)

    monkeypatch.setattr(
        ProteinStructure,
        "with_updated_residue_facets_batch",
        recording_batch_update,
    )

    batch.prepared_structure(structure, component_library=component_library)
    batch.merged_structure(
        original_structure=structure,
        corrected_heavy_structure=structure,
    )

    assert observed_batch_sizes == [2, 2]


def test_stereochemistry_batch_matches_sequential_topology_remapping() -> None:
    """Batch preparation and merge should match sequential facet replacement."""

    structure = read_structure(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    component_library = build_standard_component_library()
    residue_ids = (ResidueId("A", 30), ResidueId("B", 30))
    batch = StereochemistryCorrectionBatch(
        violations_by_residue={
            residue_id: (_stereochemistry_violation(residue_id),)
            for residue_id in residue_ids
        }
    )

    sequentially_prepared = structure
    for residue_id in batch.corrected_residue_ids():
        prepared_residue = batch.prepared_payload(
            batch_module._completion_payload_for_structure(
                sequentially_prepared,
                residue_id,
            ),
            component_library=component_library,
        )
        sequentially_prepared = sequentially_prepared.with_updated_residue_facets(
            prepared_residue.residue_site,
            residue_geometry=prepared_residue.residue_geometry,
            formal_charge_by_atom_name=(
                prepared_residue.formal_charge_by_atom_name
            ),
        )

    batch_prepared = batch.prepared_structure(
        structure,
        component_library=component_library,
    )
    assert structure.topology.bonds
    assert batch_prepared == sequentially_prepared
    assert batch_prepared.topology.bonds == sequentially_prepared.topology.bonds

    sequentially_merged = structure
    for residue_id in batch.corrected_residue_ids():
        corrected_residue = batch_module._completion_payload_for_structure(
            batch_prepared,
            residue_id,
        )
        sequentially_merged = sequentially_merged.with_updated_residue_facets(
            corrected_residue.residue_site,
            residue_geometry=corrected_residue.residue_geometry,
            formal_charge_by_atom_name=(
                corrected_residue.formal_charge_by_atom_name
            ),
        )

    batch_merged = batch.merged_structure(
        original_structure=structure,
        corrected_heavy_structure=batch_prepared,
    )
    assert batch_merged == sequentially_merged
    assert batch_merged.topology.bonds == sequentially_merged.topology.bonds
    assert tuple(
        residue_site.residue_id for residue_site in batch_merged.iter_residue_sites()
    ) == tuple(
        residue_site.residue_id
        for residue_site in sequentially_merged.iter_residue_sites()
    )


def test_correct_sidechain_stereochemistry_is_noop_for_native_residues() -> None:
    """Native supported residues should pass through without corrections."""

    structure = focused_structure_for_residue(seq_num=30)

    result = correct_sidechain_stereochemistry(structure)

    assert result == TransformationResult(
        structure=structure,
        repairs=(),
        issues=(),
    )


def test_targeted_stereochemistry_correction_keeps_diagnostics_focused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Initial and residual diagnostics should retain the requested scope."""

    residue_id = ResidueId("A", 30)
    target_residue_ids = frozenset({residue_id})
    structure = focused_structure_for_residue(
        seq_num=residue_id.seq_num,
        mutate_residue=invert_threonine_residue,
    )
    observed_focuses: list[Collection[ResidueId] | None] = []

    def focused_detector(
        candidate_structure: ProteinStructure,
        *,
        component_library: ComponentLibrary,
        residue_ids: Collection[ResidueId] | None = None,
    ) -> StereochemistryReport:
        observed_focuses.append(residue_ids)
        return detect_sidechain_stereochemistry(
            candidate_structure,
            component_library=component_library,
            residue_ids=residue_ids,
        )

    monkeypatch.setattr(
        correction_module,
        "detect_sidechain_stereochemistry",
        focused_detector,
    )
    monkeypatch.setattr(
        batch_module,
        "detect_sidechain_stereochemistry",
        focused_detector,
    )

    result = correct_sidechain_stereochemistry(
        structure,
        target_residue_ids=target_residue_ids,
    )

    assert result.issue_count() == 0
    assert observed_focuses == [target_residue_ids, target_residue_ids]


def test_stereochemistry_batch_remaining_issues_filters_to_corrected_residues() -> None:
    """Batch issue projection should keep only unresolved selected residues."""

    selected_thr = ResidueId("A", 30)
    selected_ile = ResidueId("A", 25)
    unrelated_val = ResidueId("A", 10)
    same_seq_other_chain = ResidueId("B", 30)
    same_seq_insertion = ResidueId("A", 30, "A")
    batch = StereochemistryCorrectionBatch(
        violations_by_residue={
            selected_thr: (_stereochemistry_violation(selected_thr),),
            selected_ile: (_stereochemistry_violation(selected_ile),),
        }
    )
    selected_ile_issue = _residue_issue(selected_ile, "selected ILE remains bad")
    selected_thr_issue = _residue_issue(selected_thr, "selected THR remains bad")
    selected_ile_duplicate = _residue_issue(selected_ile, "second ILE center bad")
    report = _ReportWithIssues(
        (
            _structure_issue("global parser issue"),
            selected_ile_issue,
            _residue_issue(unrelated_val, "unrelated residue remains bad"),
            _residue_issue(same_seq_other_chain, "same sequence on another chain"),
            _residue_issue(same_seq_insertion, "same sequence with insertion code"),
            selected_thr_issue,
            selected_ile_duplicate,
        )
    )

    assert batch.remaining_issues(report) == (
        selected_ile_issue,
        selected_thr_issue,
        selected_ile_duplicate,
    )


def test_stereochemistry_batch_remaining_issues_handles_empty_inputs() -> None:
    """Empty batch or empty report should not manufacture remaining issues."""

    selected_thr = ResidueId("A", 30)
    non_empty_report = _ReportWithIssues(
        (
            _residue_issue(selected_thr, "selected THR remains bad"),
            _structure_issue("global parser issue"),
        )
    )
    non_empty_batch = StereochemistryCorrectionBatch(
        violations_by_residue={
            selected_thr: (_stereochemistry_violation(selected_thr),),
        }
    )

    assert (
        StereochemistryCorrectionBatch(violations_by_residue={}).remaining_issues(
            non_empty_report
        )
        == ()
    )
    assert non_empty_batch.remaining_issues(_ReportWithIssues(())) == ()


def focused_structure_for_residue(
    *,
    seq_num: int,
    mutate_residue: Callable[[CanonicalResiduePayload], CanonicalResiduePayload]
    | None = None,
) -> ProteinStructure:
    """Return a one-residue canonical structure from the representative fixture."""

    return focused_structure_for_residues(((seq_num, mutate_residue),))


def focused_structure_for_residues(
    residue_specs: tuple[
        tuple[
            int,
            Callable[[CanonicalResiduePayload], CanonicalResiduePayload] | None,
        ],
        ...,
    ],
) -> ProteinStructure:
    """Return selected canonical residues from the representative fixture."""

    structure = read_structure(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    residues: list[CanonicalResiduePayload] = []
    for seq_num, mutate_residue in residue_specs:
        residue_site = next(
            residue_site
            for residue_site in structure.chain_site("A").residues
            if residue_site.residue_id.seq_num == seq_num
        )
        residue_id = residue_site.residue_id
        residue_geometry = structure.geometry.residue_geometry(
            constitution=structure.constitution,
            residue_index=structure.constitution.residue_index(residue_id),
        )
        residue: CanonicalResiduePayload = (
            residue_site,
            residue_geometry,
            structure.topology.residue_formal_charge_by_atom_name(
                constitution=structure.constitution,
                residue_index=structure.constitution.residue_index(residue_id),
            ),
        )
        residues.append(
            mutate_residue(residue) if mutate_residue is not None else residue
        )

    return build_canonical_structure(
        chains=(chain_payload("A", tuple(residues)),),
        source_format=FileFormat.PDB,
        source_name="stereochemistry-focused",
    )


def invert_threonine_residue(
    residue: CanonicalResiduePayload,
) -> CanonicalResiduePayload:
    """Swap THR substituent coordinates to invert the CB tetrahedral center."""

    residue_site, residue_geometry, formal_charge_by_atom_name = residue
    return (
        residue_site,
        residue_geometry.with_atom_geometries(
            (
                ("OG1", residue_geometry.atom_geometry("CG2")),
                ("CG2", residue_geometry.atom_geometry("OG1")),
            )
        ),
        formal_charge_by_atom_name,
    )


def invert_isoleucine_residue(
    residue: CanonicalResiduePayload,
) -> CanonicalResiduePayload:
    """Swap ILE branch roots to invert the CB tetrahedral center."""

    residue_site, residue_geometry, formal_charge_by_atom_name = residue
    return (
        residue_site,
        residue_geometry.with_atom_geometries(
            (
                ("CG1", residue_geometry.atom_geometry("CG2")),
                ("CG2", residue_geometry.atom_geometry("CG1")),
            )
        ),
        formal_charge_by_atom_name,
    )


def stereochemistry_report(result: TransformationResult):
    """Return the supported stereochemistry diagnostic report for one result."""

    return detect_sidechain_stereochemistry(
        result.structure,
        component_library=build_standard_component_library(),
    )


def _stereochemistry_violation(
    residue_id: ResidueId,
) -> SidechainStereochemistryViolation:
    return SidechainStereochemistryViolation(
        residue_id=residue_id,
        component_id="THR",
        center_atom_name="CB",
        ordered_neighbor_atom_names=("CA", "OG1", "CG2"),
        expected_orientation_sign=1,
        observed_signed_volume=-1.0,
    )


def _residue_issue(residue_id: ResidueId, message: str) -> ValidationIssue:
    return ValidationIssue.for_residue(
        kind=ValidationIssueKind.INVALID_STEREOCHEMISTRY,
        severity=IssueSeverity.WARNING,
        message=message,
        residue_id=residue_id,
    )


def _structure_issue(message: str) -> ValidationIssue:
    return ValidationIssue(
        kind=ValidationIssueKind.PARSER_READABILITY,
        severity=IssueSeverity.ERROR,
        message=message,
    )


class _ReportWithIssues(StereochemistryReport):
    __slots__ = ("_issues",)

    def __init__(self, issues: tuple[ValidationIssue, ...]) -> None:
        object.__setattr__(self, "_issues", issues)

    def to_issues(self) -> tuple[ValidationIssue, ...]:
        return self._issues

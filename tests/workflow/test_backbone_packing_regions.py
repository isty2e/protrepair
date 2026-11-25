"""FASPR packing of usable regions next to unresolved backbone sites.

Every test runs the vendored FASPR executable. Chain A carries the
zero-occupancy 1UBQ fixture: A:3 N/CA stay unresolved, while A:5 C, A:8 CA+C and
A:10 C are rebuilt and accepted. Other chains are fully observed copies.
"""

from collections.abc import Mapping
from functools import cache
from pathlib import Path
from types import MappingProxyType

import numpy as np
import pytest

from protrepair import process_structure
from protrepair.chemistry.component.defaults import build_default_component_library
from protrepair.diagnostics.clashes import (
    ClashPolicy,
    detect_clashes_involving_residues,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import RepairEventKind, ValidationIssueKind
from protrepair.errors import PackingError
from protrepair.io import read_structure_string
from protrepair.structure import FileFormat, ResidueId
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.backbone_availability import (
    BackboneAvailability,
    backbone_repair_run,
)
from protrepair.transformer.completion.backbone_rebuild import (
    rebuild_selected_backbones,
)
from protrepair.transformer.packing import (
    PackingPlan,
    PackingResult,
    PackingScope,
    PackingSpec,
    execute_sidechain_packing,
)
from protrepair.transformer.packing.faspr import backend as faspr_backend
from protrepair.transformer.packing.faspr.backend import (
    FasprPackingBackend,
    PackingBackendError,
    PackingBackendExecutionError,
)
from protrepair.workflow.contracts import WorkflowTransformRequests

pytestmark = pytest.mark.workflow

_PDB = Path(__file__).parents[1] / "fixtures/pdb/span-reconstruction/1ubq.pdb"
_MASK: Mapping[int, frozenset[str]] = MappingProxyType(
    {
        3: frozenset({"N", "CA"}),
        5: frozenset({"C"}),
        8: frozenset({"CA", "C"}),
        10: frozenset({"C"}),
    }
)
_RESIDUES = tuple(range(3, 14))
_REMOTE = np.array((80.0, 0.0, 0.0))
_FULL = PackingSpec(backend_name="faspr")


@cache
def _fixture_rows() -> tuple[str, ...]:
    return tuple(
        row
        for row in _PDB.read_text().splitlines()
        if row.startswith("ATOM  ") and int(row[22:26]) in _RESIDUES
    )


def _row_point(row: str) -> np.ndarray:
    return np.array([float(row[i : i + 8]) for i in (30, 38, 46)])


def _fixture_point(number: int, atom: str) -> np.ndarray:
    return _row_point(
        next(
            row
            for row in _fixture_rows()
            if int(row[22:26]) == number and row[12:16].strip() == atom
        )
    )


def _chain_rows(
    chain: str,
    offset: np.ndarray,
    *,
    numbers: tuple[int, ...] = _RESIDUES,
    mask: Mapping[int, frozenset[str]] | None = None,
    shift: float = 0.0,
    drop: frozenset[tuple[int, str]] = frozenset(),
    rename: Mapping[int, str] | None = None,
    ter_before: int | None = None,
) -> list[str]:
    rows: list[str] = []
    for row in _fixture_rows():
        number, atom = int(row[22:26]), row[12:16].strip()
        if number not in numbers or (number, atom) in drop:
            continue
        if number == ter_before and atom == "N":
            rows.append("TER")

        point = _row_point(row) + offset
        occupancy = row[54:60]
        if mask is not None and atom in mask.get(number, frozenset()):
            point = point + shift
            occupancy = "  0.00"
        component = (rename or {}).get(number, row[17:20])
        coordinates = "".join(f"{value:8.3f}" for value in point)
        rows.append(
            f"{row[:17]}{component} {chain}{row[22:30]}{coordinates}"
            f"{occupancy}{row[60:]}"
        )
    return [*rows, "TER"]


def _read(*chains: list[str]) -> ProteinStructure:
    lines = [row for chain in chains for row in chain]
    return read_structure_string("\n".join([*lines, "END", ""]), FileFormat.PDB)


def _with_refused_site(*remote: list[str], shift: float = 6.0) -> ProteinStructure:
    return _read(_chain_rows("A", np.zeros(3), mask=_MASK, shift=shift), *remote)


def _methanol_rows(carbon: np.ndarray, away: np.ndarray) -> list[str]:
    """Return a methanol ligand Z:1 whose O points from ``carbon`` along ``away``."""

    oxygen = carbon + 1.43 * away / np.linalg.norm(away)
    return [
        f"HETATM{serial:5d}  {name:<3} MOH Z   1    "
        + "".join(f"{value:8.3f}" for value in point)
        + f"  1.00 20.00          {name:>2}"
        for serial, (name, point) in enumerate(
            (("C", carbon), ("O", oxygen)), start=9001
        )
    ]


def _rebuilt(source: ProteinStructure) -> tuple[ProteinStructure, BackboneAvailability]:
    result, availability = rebuild_selected_backbones(
        source, build_default_component_library()
    )
    assert availability is not None
    return result.structure, availability


def _pack(
    carrier: ProteinStructure, availability: BackboneAvailability, spec: PackingSpec
) -> PackingResult:
    with backbone_repair_run(availability):
        return execute_sidechain_packing(carrier, spec)


def _ordinary(structure: ProteinStructure, spec: PackingSpec) -> PackingResult:
    return FasprPackingBackend().pack(PackingPlan.from_inputs(structure, spec))


def _chain_ids(chain: str, numbers: tuple[int, ...] = _RESIDUES) -> set[ResidueId]:
    return {ResidueId(chain, number) for number in numbers}


def _refused(issues: tuple[ValidationIssue, ...]) -> dict[ResidueId, str]:
    return {
        issue.residue_id: issue.message
        for issue in issues
        if issue.kind is ValidationIssueKind.PACKING_REGION_REFUSED
        and issue.residue_id is not None
    }


def _refused_messages(result: PackingResult) -> tuple[str, ...]:
    return tuple(
        issue.message
        for issue in result.issues
        if issue.kind is ValidationIssueKind.PACKING_REGION_REFUSED
    )


def _positions(
    structure: ProteinStructure, residue_id: ResidueId
) -> dict[str, tuple[float, ...]]:
    site = structure.constitution.residue_or_ligand(residue_id)
    assert site is not None
    geometry = structure.residue_geometry(
        structure.constitution.residue_index(residue_id)
    )
    return {
        atom.name: tuple(
            round(float(value), 3) for value in geometry.position(atom.name).to_array()
        )
        for atom in site.atom_sites
    }


def test_no_unresolved_backbone_keeps_the_ordinary_call():
    accepted = {number: names for number, names in _MASK.items() if number != 3}
    carrier, availability = _rebuilt(
        _read(_chain_rows("A", np.zeros(3), mask=accepted, shift=6.0))
    )
    assert all(p.reason is None for p in availability.placements.values())

    assert _pack(carrier, availability, _FULL) == _ordinary(carrier, _FULL)


def test_placeholder_coordinates_do_not_reach_region_packing():
    remote = _chain_rows("B", _REMOTE)
    runs = []
    for shift in (-8.0, 14.0):
        carrier, availability = _rebuilt(_with_refused_site(remote, shift=shift))
        runs.append((carrier, _pack(carrier, availability, _FULL)))

    (left, left_result), (_, right_result) = runs
    assert left_result.changed_residue_ids == right_result.changed_residue_ids
    assert left_result.changed_residue_ids
    assert _refused(left_result.issues) == _refused(right_result.issues)
    for site in left.constitution.iter_residues():
        placeholder = {"N", "CA"} if site.residue_id == ResidueId("A", 3) else set()
        left_atoms, right_atoms = (
            {
                name: point
                for name, point in _positions(
                    result.packed_structure, site.residue_id
                ).items()
                if name not in placeholder
            }
            for result in (left_result, right_result)
        )
        assert left_atoms == right_atoms
    for carrier, result in runs:
        assert _positions(result.packed_structure, ResidueId("A", 3)) == _positions(
            carrier, ResidueId("A", 3)
        )


def test_targets_near_the_refused_site_are_withheld_and_a_remote_chain_repacks():
    carrier, availability = _rebuilt(_with_refused_site(_chain_rows("B", _REMOTE)))

    result = _pack(carrier, availability, _FULL)

    refused = _refused(result.issues)
    assert "unresolved backbone" in refused[ResidueId("A", 3)]
    assert "chain neighbor A:3" in refused[ResidueId("A", 4)]
    for number in (5, 12, 13):
        assert (
            "possible FASPR contact with unsubmitted residue A:3"
            in (refused[ResidueId("A", number)])
        )
    assert ResidueId("A", 8) in result.changed_residue_ids
    assert not _chain_ids("B") & refused.keys()
    for residue_id in _chain_ids("A") & refused.keys():
        assert _positions(result.packed_structure, residue_id) == _positions(
            carrier, residue_id
        )

    ordinary = _ordinary(carrier.select_chains(("B",)), _FULL)
    assert ordinary.changed_residue_ids
    assert set(ordinary.changed_residue_ids) <= set(result.changed_residue_ids)
    for residue_id in _chain_ids("B"):
        assert _positions(result.packed_structure, residue_id) == _positions(
            ordinary.packed_structure, residue_id
        )


def test_usable_region_matches_the_joint_call_given_true_coordinates():
    targets = tuple(ResidueId("A", number) for number in (7, 8, 9, 11))
    spec = PackingSpec(
        backend_name="faspr", scope=PackingScope.LOCAL, mutable_residue_ids=targets
    )
    carrier, availability = _rebuilt(_with_refused_site(shift=6.0))
    truth, _ = _rebuilt(_with_refused_site(shift=0.0))

    result = _pack(carrier, availability, spec)
    joint = _ordinary(truth, spec)

    assert not _refused(result.issues)
    assert result.changed_residue_ids == joint.changed_residue_ids
    for residue_id in targets:
        assert _positions(result.packed_structure, residue_id) == _positions(
            joint.packed_structure, residue_id
        )


def test_contacting_chains_pack_jointly_across_a_far_transport_join(monkeypatch):
    contact = _REMOTE + np.array((0.0, 0.0, 9.0))
    carrier, availability = _rebuilt(
        _with_refused_site(_chain_rows("B", _REMOTE), _chain_rows("C", contact))
    )
    submitted: list[tuple[str, ...]] = []
    real_run = faspr_backend.run_faspr

    def recording_run(execution_input, *args, **kwargs):
        submitted.append(execution_input.structure.constitution.chain_ids())
        return real_run(execution_input, *args, **kwargs)

    monkeypatch.setattr(faspr_backend, "run_faspr", recording_run)
    result = _pack(carrier, availability, _FULL)

    assert ("B", "C") in submitted
    both = _chain_ids("B") | _chain_ids("C")
    assert not both & _refused(result.issues).keys()
    ordinary = _ordinary(carrier.select_chains(("B", "C")), _FULL)
    for residue_id in both:
        assert _positions(result.packed_structure, residue_id) == _positions(
            ordinary.packed_structure, residue_id
        )


@pytest.mark.parametrize("deeper", (True, False))
def test_unseen_ligand_contact_is_compared_with_its_baseline(deeper):
    remote = _chain_rows("B", _REMOTE)
    plain, _ = _rebuilt(_with_refused_site(remote))
    ordinary = _ordinary(plain.select_chains(("B",)), _FULL)
    leucine, glycine = ResidueId("B", 8), ResidueId("B", 10)
    if deeper:
        moved = np.array(_positions(ordinary.packed_structure, leucine)["CD1"])
        away = moved - np.array(_positions(plain, leucine)["CD1"])
        carbon = moved + 1.5 * away / np.linalg.norm(away)
        contact = leucine
    else:
        carbonyl = _positions(plain, glycine)
        away = np.subtract(carbonyl["O"], carbonyl["C"])
        carbon = np.array(carbonyl["O"]) + 2.4 * away / np.linalg.norm(away)
        contact = glycine
    carrier, availability = _rebuilt(
        _with_refused_site(remote, _methanol_rows(carbon, away))
    )
    baseline = detect_clashes_involving_residues(
        carrier,
        residue_ids=frozenset({contact}),
        component_library=build_default_component_library(),
        policy=ClashPolicy(include_ligands=True),
    )
    assert any(
        {clash.left_residue_id, clash.right_residue_id} == {contact, ResidueId("Z", 1)}
        for clash in baseline.clashes
    )

    result = _pack(carrier, availability, _FULL)

    refused = _refused(result.issues)
    if deeper:
        assert (
            "packed B:8 clashes with Z:1, which FASPR did not see" in (refused[leucine])
        )
        assert _positions(result.packed_structure, leucine) == _positions(
            carrier, leucine
        )
    else:
        assert not _chain_ids("B") & refused.keys()
        for residue_id in _chain_ids("B"):
            assert _positions(result.packed_structure, residue_id) == _positions(
                ordinary.packed_structure, residue_id
            )


def test_same_chain_ter_break_withholds_its_target():
    carrier, availability = _rebuilt(
        _with_refused_site(_chain_rows("B", _REMOTE, ter_before=8))
    )
    target = ResidueId("B", 8)

    result = _pack(
        carrier,
        availability,
        PackingSpec(
            backend_name="faspr",
            scope=PackingScope.LOCAL,
            mutable_residue_ids=(ResidueId("A", 3), target),
        ),
    )

    refused = _refused(result.issues)
    assert "unresolved backbone" in refused[ResidueId("A", 3)]
    assert "FASPR input would join B:7 and B:8" in refused[target]
    assert result.changed_residue_ids == ()
    assert _positions(result.packed_structure, target) == _positions(carrier, target)


def test_unproven_transport_join_withholds_its_targets():
    carbon = _fixture_point(13, "C")
    direction = carbon - _fixture_point(13, "CA")
    joined_nitrogen = carbon + 1.5 * direction / np.linalg.norm(direction)
    offset = _REMOTE + joined_nitrogen - _fixture_point(3, "N")
    carrier, availability = _rebuilt(
        _with_refused_site(
            _chain_rows("B", _REMOTE), _chain_rows("C", offset, numbers=(3, 4, 5))
        )
    )

    result = _pack(carrier, availability, _FULL)

    refused = _refused(result.issues)
    for residue_id in (ResidueId("B", 13), ResidueId("C", 3)):
        assert "FASPR input would join B:13 and C:3" in refused[residue_id]
        assert _positions(result.packed_structure, residue_id) == _positions(
            carrier, residue_id
        )


def test_unsupported_remote_residue_withholds_only_its_dependents():
    carrier, availability = _rebuilt(
        _with_refused_site(
            _chain_rows("B", _REMOTE),
            _chain_rows("C", 2 * _REMOTE, rename={8: "DLE"}),
        )
    )
    target = PackingSpec(
        backend_name="faspr",
        scope=PackingScope.LOCAL,
        mutable_residue_ids=(ResidueId("B", 8),),
    )
    with pytest.raises(PackingError, match="does not support component DLE"):
        _ordinary(carrier, target)

    local = _pack(carrier, availability, target)
    full = _pack(carrier, availability, _FULL)

    assert not _refused(local.issues)
    ordinary = _ordinary(carrier.select_chains(("B",)), target)
    assert _positions(local.packed_structure, ResidueId("B", 8)) == _positions(
        ordinary.packed_structure, ResidueId("B", 8)
    )
    refused = _refused(full.issues)
    assert "does not support component DLE" in refused[ResidueId("C", 8)]
    assert "chain neighbor C:8" in refused[ResidueId("C", 7)]
    assert "chain neighbor C:8" in refused[ResidueId("C", 9)]
    assert not _chain_ids("B") & refused.keys()


def test_workflow_reference_packing_reports_partial_coverage():
    source = _with_refused_site(
        _chain_rows("B", _REMOTE), _chain_rows("C", 2 * _REMOTE, rename={8: "DLE"})
    )

    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            rebuild_zero_occupancy_backbone=True,
            reference_sidechain_packing=_FULL,
        ),
    )

    rebuilt = {
        event.residue_id
        for event in result.repairs
        if event.kind is RepairEventKind.BACKBONE_ATOMS_REBUILT
    }
    assert rebuilt == {ResidueId("A", 5), ResidueId("A", 8), ResidueId("A", 10)}
    refused = _refused(result.issues)
    assert {
        ResidueId("A", 3),
        ResidueId("C", 7),
        ResidueId("C", 8),
        ResidueId("C", 9),
    } <= refused.keys()
    assert not _chain_ids("B") & refused.keys()


def test_workflow_committed_packing_completes_a_remote_side_chain():
    source = _with_refused_site(_chain_rows("B", _REMOTE, drop=frozenset({(8, "CD1")})))
    incomplete = source.constitution.residue_or_ligand(ResidueId("B", 8))
    assert incomplete is not None
    assert not incomplete.has_atom_site("CD1")

    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            rebuild_zero_occupancy_backbone=True,
            committed_sidechain_packing=PackingSpec(
                backend_name="faspr",
                scope=PackingScope.LOCAL,
                mutable_residue_ids=(ResidueId("A", 3), ResidueId("B", 8)),
            ),
        ),
    )

    packed = result.structure.constitution.residue_or_ligand(ResidueId("B", 8))
    assert packed is not None
    assert packed.has_atom_site("CD1")
    assert ResidueId("A", 3) in _refused(result.issues)
    rebuilt = {
        event.residue_id
        for event in result.repairs
        if event.kind is RepairEventKind.BACKBONE_ATOMS_REBUILT
    }
    assert rebuilt == {ResidueId("A", 5), ResidueId("A", 8), ResidueId("A", 10)}


def test_incomplete_fixed_residue_is_omitted_instead_of_failing_the_call():
    carrier, availability = _rebuilt(
        _with_refused_site(
            _chain_rows("B", _REMOTE),
            _chain_rows("C", 2 * _REMOTE, drop=frozenset({(6, "NZ")})),
        )
    )
    incomplete = ResidueId("C", 6)
    spec = PackingSpec(
        backend_name="faspr",
        scope=PackingScope.LOCAL,
        mutable_residue_ids=tuple(
            sorted((_chain_ids("B") | _chain_ids("C")) - {incomplete})
        ),
    )
    with pytest.raises(PackingBackendError, match="incomplete fixed residue C:6"):
        _ordinary(carrier, spec)

    result = _pack(carrier, availability, spec)

    refused = _refused(result.issues)
    assert "chain neighbor C:6" in refused[ResidueId("C", 5)]
    assert "chain neighbor C:6" in refused[ResidueId("C", 7)]
    assert incomplete not in result.changed_residue_ids
    assert _positions(result.packed_structure, incomplete) == _positions(
        carrier, incomplete
    )
    assert not _chain_ids("B") & refused.keys()
    assert _chain_ids("B") & set(result.changed_residue_ids)


@pytest.mark.parametrize("failing_chain", ("B", "C"))
def test_region_failure_discards_only_that_region(monkeypatch, failing_chain):
    carrier, availability = _rebuilt(
        _with_refused_site(_chain_rows("B", _REMOTE), _chain_rows("C", 2 * _REMOTE))
    )
    real_run = faspr_backend.run_faspr

    def failing_run(execution_input, *args, **kwargs):
        if failing_chain in execution_input.structure.constitution.chain_ids():
            raise PackingBackendExecutionError("synthetic region failure")
        return real_run(execution_input, *args, **kwargs)

    monkeypatch.setattr(faspr_backend, "run_faspr", failing_run)
    result = _pack(carrier, availability, _FULL)

    healthy = "C" if failing_chain == "B" else "B"
    refused = _refused(result.issues)
    failed = {rid for rid, message in refused.items() if "synthetic" in message}
    assert failed == _chain_ids(failing_chain)
    assert not _chain_ids(failing_chain) & set(result.changed_residue_ids)
    assert _chain_ids(healthy) & set(result.changed_residue_ids)
    assert not _chain_ids(healthy) & refused.keys()
    for residue_id in _chain_ids(failing_chain):
        assert _positions(result.packed_structure, residue_id) == _positions(
            carrier, residue_id
        )


def test_explicit_mutation_is_all_or_nothing():
    carrier, availability = _rebuilt(_with_refused_site(_chain_rows("B", _REMOTE)))
    remote = ResidueId("B", 8)

    coupled = _pack(
        carrier,
        availability,
        PackingSpec(
            backend_name="faspr",
            scope=PackingScope.LOCAL,
            target_sequence="AA",
            mutable_residue_ids=(ResidueId("A", 4), remote),
        ),
    )
    alone = _pack(
        carrier,
        availability,
        PackingSpec(
            backend_name="faspr",
            scope=PackingScope.LOCAL,
            target_sequence="a",
            mutable_residue_ids=(remote,),
        ),
    )

    assert coupled.packed_structure == carrier
    assert coupled.changed_residue_ids == ()
    assert any("target_sequence" in message for message in _refused_messages(coupled))
    site = alone.packed_structure.constitution.residue_or_ligand(remote)
    assert site is not None
    assert site.component_id == "ALA"
    assert alone.changed_residue_ids == (remote,)


@pytest.mark.parametrize(
    ("spec", "message"),
    (
        (
            PackingSpec(backend_name="faspr", target_sequence="A"),
            "must match polymer residue count",
        ),
        (
            PackingSpec(
                backend_name="faspr",
                scope=PackingScope.LOCAL,
                target_sequence="X",
                mutable_residue_ids=(ResidueId("B", 8),),
            ),
            "does not support token 'X'",
        ),
    ),
)
def test_malformed_request_keeps_the_ordinary_error(spec, message):
    carrier, availability = _rebuilt(_with_refused_site(_chain_rows("B", _REMOTE)))

    with pytest.raises(PackingError, match=message):
        _ordinary(carrier, spec)
    with pytest.raises(PackingError, match=message):
        _pack(carrier, availability, spec)


def test_missing_backend_is_reported_once_without_writes(monkeypatch):
    def unavailable(self):
        raise OSError("no FASPR here")

    carrier, availability = _rebuilt(_with_refused_site(_chain_rows("B", _REMOTE)))
    monkeypatch.setattr(FasprPackingBackend, "runtime_assets", unavailable)

    result = _pack(carrier, availability, _FULL)

    assert result.packed_structure == carrier
    assert result.changed_residue_ids == ()
    missing = [m for m in _refused_messages(result) if "unavailable" in m]
    assert len(missing) == 1

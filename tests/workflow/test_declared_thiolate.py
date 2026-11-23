"""Ordinary H preparation preserves deposited cysteine sulfur chemistry."""

from dataclasses import replace
from pathlib import Path

import pytest

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.io import read_structure_string, write_structure_string
from protrepair.scope import WholeStructureScope
from protrepair.state import HydrogenCoverageState
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.hydrogen.repair import add_hydrogens
from protrepair.workflow.contracts.request import (
    RequestedGoalSet,
    requested_process_goal,
)
from protrepair.workflow.engine.hydrogen import project_hydrogen_issues

_CYS = ResidueId("A", 3)
_SG = AtomRef(_CYS, "SG")
_H_GOAL = RequestedGoalSet(
    (
        requested_process_goal(
            scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
        ),
    )
)


def _read_cysteine(charge, isotope=None):
    records = []
    for line in (
        Path("tests/fixtures/pdb/span-reconstruction/1crn.pdb").read_text().splitlines()
    ):
        if line.startswith("ATOM") and int(line[22:26]) in (2, 3, 4):
            if int(line[22:26]) == 3 and line[12:16].strip() == "SG":
                line = line[:78] + charge
            records.append(line)
    if isotope is not None:
        records.append(
            f"ATOM     99 {isotope + 'G':4s} CYS A   3      15.885  12.426  11.356"
            f"  1.00  7.01          {isotope:>2}  "
        )
        records.append("CONECT   20   99")
    return read_structure_string("\n".join((*records, "END")), FileFormat.PDB)


def _complete(structure, workflow):
    if workflow:
        return process_structure(structure, requested_goals=_H_GOAL)
    return add_hydrogens(
        structure, prepare_heavy_atoms=False, target_residue_ids=frozenset((_CYS,))
    )


@pytest.mark.parametrize("charge,expected", (("  ", None), ("0+", 0), ("1-", -1)))
@pytest.mark.parametrize("workflow", (False, True))
def test_sulfur_charge_controls_hydrogen_completion(charge, expected, workflow):
    source = _read_cysteine(charge)
    result = _complete(source, workflow)
    assert not any(
        i.kind is ValidationIssueKind.UNRESOLVED_POLYMER_HYDROGEN for i in result.issues
    )
    structure = result.structure
    assert (
        structure.topology.formal_charge(structure.constitution.atom_index(_SG))
        == expected
    )
    assert (
        structure.constitution.resolve_atom_index(AtomRef(_CYS, "HG")) is not None
    ) is (expected != -1)


@pytest.mark.parametrize("fmt", (FileFormat.MMCIF, FileFormat.PDB))
def test_reread_thiolate_needs_no_additional_request(fmt):
    first = add_hydrogens(_read_cysteine("1-")).structure
    reread = read_structure_string(write_structure_string(first, fmt), fmt)
    again = add_hydrogens(reread)
    assert again.structure.constitution.resolve_atom_index(AtomRef(_CYS, "HG")) is None
    assert (
        again.structure.topology.formal_charge(
            again.structure.constitution.atom_index(_SG)
        )
        == -1
    )


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
@pytest.mark.parametrize("workflow", (False, True))
def test_contradictory_original_sulfur_h_is_reported_and_kept(isotope, workflow):
    source = _read_cysteine("1-", isotope)
    ref = AtomRef(_CYS, isotope + "G")
    before = source.constitution.atom_index(ref)
    carrier = source
    for _ in range(2):
        result = _complete(carrier, workflow)
        assert (
            sum(
                i.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
                for i in result.issues
            )
            == 1
        )
        carrier = result.structure
        after = carrier.constitution.atom_index(ref)
        assert carrier.constitution.atom_site_at(
            after
        ) == source.constitution.atom_site_at(before)
        assert carrier.geometry.atom_geometry(after) == source.geometry.atom_geometry(
            before
        )
        assert (
            carrier.topology.bond_between(after, carrier.constitution.atom_index(_SG))
            is not None
        )
        assert (
            carrier.topology.formal_charge(carrier.constitution.atom_index(_SG)) == -1
        )


@pytest.mark.parametrize("isotope", (None, "D"))
@pytest.mark.parametrize("workflow", (False, True))
def test_canonical_charge_without_ingress_observation_protects_current_atoms(
    isotope, workflow
):
    source = _read_cysteine("1-", isotope)
    source = source.with_provenance(
        replace(
            source.provenance,
            ingress=replace(source.provenance.ingress, observation=None),
        )
    )
    for _ in range(2):
        result = _complete(source, workflow)
        source = result.structure
        assert source.topology.formal_charge(source.constitution.atom_index(_SG)) == -1
        assert source.constitution.resolve_atom_index(AtomRef(_CYS, "HG")) is None
        assert (
            source.constitution.resolve_atom_index(AtomRef(_CYS, "DG")) is not None
        ) is (isotope is not None)
        assert sum(
            i.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION for i in result.issues
        ) == int(isotope is not None)


def test_projection_reports_one_current_sulfur_conflict_after_repeated_stages():
    result = process_structure(_read_cysteine("1-", "D"), requested_goals=_H_GOAL)
    (conflict,) = (
        i
        for i in result.issues
        if i.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
    )
    projected = project_hydrogen_issues(
        replace(result, issues=(*result.issues, conflict)),
        requested_goal_report=result.requested_goal_report,
        component_library=build_default_component_library(),
    )
    assert projected.count(conflict) == 1

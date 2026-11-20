"""Linked NME atom names from a populated legacy Amber PDB export."""

from collections.abc import Mapping
from pathlib import Path

import pytest
from rdkit import Chem

from protrepair import process_structure
from protrepair.io import read_structure, read_structure_string, write_structure_string
from protrepair.scope import WholeStructureScope
from protrepair.state import HydrogenCoverageState
from protrepair.structure import AtomRef, FileFormat, ProteinStructure, ResidueId
from protrepair.workflow.contracts import (
    LigandPolicy,
    StructureIngressOptions,
    requested_process_goal,
)

pytestmark = pytest.mark.workflow
_AMBER = Path(__file__).parents[1] / "fixtures/pdb/nme-names/amber-ff94-nme.pdb"
_GLY = ResidueId("_", 1)
_NME = ResidueId("_", 2)
_GOALS = (
    requested_process_goal(
        scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
    ),
)
_KEEP = StructureIngressOptions(ligand_policy=LigandPolicy.KEEP)
_NME_PARENTS = {"H": "N", "HH31": "CH3", "HH32": "CH3", "HH33": "CH3"}


def _read(
    payload: str, *, retained: bool = False, chain_id: str = " "
) -> ProteinStructure:
    if chain_id != " ":
        payload = "".join(
            row[:21] + chain_id + row[22:] if row.startswith("ATOM") else row
            for row in payload.splitlines(keepends=True)
        )
    if retained:
        link = f"LINK         C   GLY {chain_id}   1                 N   NME {chain_id}   2\n"
        payload = link + (
            "\n".join(
                "HETATM" + line[6:]
                if line.startswith("ATOM  ") and line[17:20] == "NME"
                else line
                for line in payload.splitlines()
            )
            + "\n"
        )
    return read_structure_string(
        payload, FileFormat.PDB, policy=_KEEP.structure_normalization_policy()
    )


def _parents(
    structure: ProteinStructure, residue_id: ResidueId = _NME
) -> Mapping[str, str]:
    constitution = structure.constitution
    return structure.topology.residue_hydrogen_anchor_by_name(
        constitution=constitution, residue_index=constitution.residue_index(residue_id)
    )


def _assert_input_atoms_preserved(
    source: ProteinStructure, prepared: ProteinStructure, residue_id: ResidueId = _NME
) -> None:
    for index in source.constitution.atom_indices_for_residue(residue_id):
        ref = source.constitution.atom_ref_at(index)
        target = prepared.constitution.atom_index(ref)
        assert prepared.constitution.atom_site_at(
            target
        ) == source.constitution.atom_site_at(index)
        assert prepared.geometry.atom_geometry(target) == source.geometry.atom_geometry(
            index
        )
        assert prepared.topology.formal_charge(target) == source.topology.formal_charge(
            index
        )


@pytest.mark.parametrize("retained", [False, True])
def test_populated_amber_names_bind_without_renaming_source_atoms(retained):
    source = (
        _read(_AMBER.read_text(), retained=True) if retained else read_structure(_AMBER)
    )
    result = process_structure(
        source if retained else _AMBER, ingress=_KEEP, requested_goals=_GOALS
    )
    assert result.requested_goal_report is not None
    assert result.requested_goal_report.outcomes[0].is_satisfied()
    prepared = result.structure
    assert result.issues == ()
    assert _parents(prepared) == _NME_PARENTS
    assert prepared.constitution == source.constitution
    _assert_input_atoms_preserved(source, prepared)
    nme = prepared.constitution.residue_or_ligand(_NME)
    assert nme is not None and nme.is_hetero is retained
    assert prepared.constitution.resolve_atom_index(AtomRef(_GLY, "OXT")) is None


def _rename_nme_atom(
    payload: str, name: str, replacement: str, *, element: str | None = None
) -> str:
    (row,) = (
        row
        for row in payload.splitlines()
        if row.startswith("ATOM") and row[17:20] == "NME" and row[12:16].strip() == name
    )
    label = f" {replacement:<3}" if len(replacement) < 4 else replacement
    updated = row[:12] + label + row[16:]
    if element is not None:
        updated = updated[:76] + f"{element:>2}"
    return payload.replace(row, updated)


@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("keep_first_slot", [False, True])
def test_missing_h_uses_legacy_spelling_without_duplicating_mixed_slots(
    retained, keep_first_slot
):
    payload = _AMBER.read_text()
    keep = {"HH31"} if keep_first_slot else set()
    remove = _NME_PARENTS.keys() - keep
    payload = (
        "\n".join(
            line
            for line in payload.splitlines()
            if not (line[17:20] == "NME" and line[12:16].strip() in remove)
        )
        + "\n"
    )
    if keep_first_slot:
        payload = _rename_nme_atom(payload, "HH31", "H1")
    source = _read(payload, retained=retained)
    result = process_structure(source, ingress=_KEEP, requested_goals=_GOALS)
    assert result.requested_goal_report is not None
    assert result.requested_goal_report.outcomes[0].is_satisfied()
    assert result.issues == ()
    expected = dict(_NME_PARENTS)
    if keep_first_slot:
        expected["H1"] = expected.pop("HH31")
    assert _parents(result.structure) == expected
    _assert_input_atoms_preserved(source, result.structure)


@pytest.mark.parametrize("file_format", [FileFormat.PDB, FileFormat.MMCIF])
@pytest.mark.parametrize("retained", [False, True])
def test_legacy_hydrogen_names_and_parents_survive_format_round_trip(
    file_format, retained
):
    # The blank PDB chain becomes "_", which has a separate mmCIF writer issue.
    chain_id = "A" if file_format is FileFormat.MMCIF else " "
    cap_id = ResidueId("A", 2) if chain_id == "A" else _NME
    source = _read(_AMBER.read_text(), retained=retained, chain_id=chain_id)
    prepared = process_structure(
        source, ingress=_KEEP, requested_goals=_GOALS
    ).structure
    payload = write_structure_string(prepared, file_format)
    restored = read_structure_string(
        payload, file_format, policy=_KEEP.structure_normalization_policy()
    )
    result = process_structure(restored, ingress=_KEEP, requested_goals=_GOALS)
    assert result.requested_goal_report is not None
    assert result.requested_goal_report.outcomes[0].is_satisfied()
    assert result.issues == ()
    assert _parents(result.structure, cap_id) == _NME_PARENTS
    assert result.structure.constitution == restored.constitution
    _assert_input_atoms_preserved(restored, result.structure, cap_id)


@pytest.mark.parametrize("isotope", ["D", "T"])
def test_legacy_methyl_isotope_occupies_its_named_slot(isotope):
    payload = _rename_nme_atom(
        _AMBER.read_text(), "HH31", isotope + "H31", element=isotope
    )
    source = _read(payload)
    index = source.constitution.atom_index(AtomRef(_NME, isotope + "H31"))
    assert source.constitution.atom_site_at(index).element == isotope
    result = process_structure(source, ingress=_KEEP, requested_goals=_GOALS)
    assert result.requested_goal_report is not None
    assert result.requested_goal_report.outcomes[0].is_satisfied()
    expected = dict(_NME_PARENTS)
    expected[isotope + "H31"] = expected.pop("HH31")
    assert _parents(result.structure) == expected
    _assert_input_atoms_preserved(source, result.structure)


def test_native_parser_reads_the_same_methyl_and_amide_parents():
    source = _read(_AMBER.read_text())
    prepared = process_structure(
        source, ingress=_KEEP, requested_goals=_GOALS
    ).structure
    pdb = write_structure_string(prepared, FileFormat.PDB)
    coordinate_only = (
        "\n".join(
            line for line in pdb.splitlines() if not line.startswith(("CONECT", "LINK"))
        )
        + "\n"
    )
    molecule = Chem.MolFromPDBBlock(coordinate_only, sanitize=True, removeHs=False)
    assert molecule is not None
    assert sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()) == 0
    parents = {}
    for atom in molecule.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if atom.GetAtomicNum() != 1 or info.GetResidueName().strip() != "NME":
            continue
        (parent,) = atom.GetNeighbors()
        parent_info = parent.GetPDBResidueInfo()
        assert parent_info.GetResidueName().strip() == "NME"
        parents[info.GetName().strip()] = parent_info.GetName().strip()
    assert parents == _NME_PARENTS


@pytest.mark.parametrize(
    "problem",
    ["same-slot", "two-methyl-names", "wrong-element", "wrong-bond", "renamed"],
)
def test_incompatible_legacy_inventory_is_not_promoted(problem):
    payload = _AMBER.read_text()
    original = _read(payload)
    if problem == "same-slot":
        payload = _rename_nme_atom(payload, "HH32", "H1")
    elif problem == "two-methyl-names":
        row = next(
            line for line in payload.splitlines() if line.startswith("ATOM     12")
        )
        payload = payload.replace(row, row + "\n" + "ATOM     16  C  " + row[16:])
    elif problem == "wrong-element":
        payload = _rename_nme_atom(payload, "CH3", "CH3", element="O")
    elif problem == "wrong-bond":
        payload = payload.replace("END\n", "CONECT   10   13\nEND\n")
    else:
        payload = _rename_nme_atom(payload, "CH3", "C")

    source = _read(payload)
    if problem == "renamed":
        source = source.with_provenance(original.provenance)
    result = process_structure(source, ingress=_KEEP, requested_goals=_GOALS)
    assert result.requested_goal_report is not None
    assert not result.requested_goal_report.outcomes[0].is_satisfied()
    assert _parents(result.structure) == _parents(source)
    assert result.structure.constitution.residue_or_ligand(_NME) == (
        source.constitution.residue_or_ligand(_NME)
    )
    _assert_input_atoms_preserved(source, result.structure)

"""Hydrogen completion for cap residues already present in the input."""

from dataclasses import replace

import pytest
from rdkit import Chem

from protrepair import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.io import (
    read_structure_string,
    to_rdkit_molecule,
    write_structure_string,
)
from protrepair.scope import WholeStructureScope
from protrepair.state import BackboneHeavyAtomCompletenessState, HydrogenCoverageState
from protrepair.structure import AtomRef, FileFormat, ProteinStructure, ResidueId
from protrepair.structure.peptide import are_peptide_adjacent
from protrepair.structure.topology import BondRelationshipType, TopologyBond
from protrepair.transformer.completion.hydrogen.core import materialize_hydrogens_core
from protrepair.transformer.completion.hydrogen.topology import (
    resolve_fixed_hydrogen_topology,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
    resolve_retained_non_polymer_topology,
)
from protrepair.transformer.continuous.binding_policy import (
    ManualContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationForceField
from protrepair.transformer.local import LocalScopeSpec
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.workflow.contracts import (
    LigandPolicy,
    ProcessResult,
    RequestedGoalOutcome,
    RequestedGoalSet,
    RequestedGoalStatus,
    StructureIngressOptions,
    WorkflowTransformRequests,
    requested_process_goal,
)
from protrepair.workflow.planning.completion.transformer_candidates import (
    plan_hydrogen_completion_transformers,
)

pytestmark = pytest.mark.workflow

# Heavy-atom coordinates from the report, including both pre-existing caps.
_PDB = """\
ATOM      1  C   ACE A   1       4.658   1.185  -0.417  1.00 20.00           C
ATOM      2  O   ACE A   1       5.124   0.479   0.475  1.00 20.00           O
ATOM      3  CH3 ACE A   1       5.557   2.090  -1.247  1.00 20.00           C
ATOM      4  N   GLY A   2       3.358   1.209  -0.718  1.00 20.00           N
ATOM      5  CA  GLY A   2       2.403   0.401  -0.009  1.00 20.00           C
ATOM      6  C   GLY A   2       1.013   0.571  -0.493  1.00 20.00           C
ATOM      7  O   GLY A   2       0.787   1.363  -1.431  1.00 20.00           O
ATOM      8  N   ALA A   3      -0.012  -0.190   0.149  1.00 20.00           N
ATOM      9  CA  ALA A   3      -1.375  -0.051  -0.294  1.00 20.00           C
ATOM     10  C   ALA A   3      -2.339   0.209   0.802  1.00 20.00           C
ATOM     11  O   ALA A   3      -3.553   0.343   0.497  1.00 20.00           O
ATOM     12  CB  ALA A   3      -1.774  -1.424  -0.850  1.00 20.00           C
ATOM     13  N   NME A   4      -1.827   0.279   2.033  1.00 20.00           N
ATOM     14  C   NME A   4      -2.657   0.528   3.194  1.00 20.00           C
TER      15      NME A   4
END
"""
_TYPED_LINK = "LINK         C   ALA A   3                 N   NME A   4\n"
_NME = ResidueId("A", 4)
_ALA = ResidueId("A", 3)
_GOALS = (
    requested_process_goal(
        scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
    ),
)
_KEEP = StructureIngressOptions(ligand_policy=LigandPolicy.KEEP)


def _h_outcome(result: ProcessResult) -> RequestedGoalOutcome:
    report = result.requested_goal_report
    assert report is not None
    (outcome,) = report.outcomes
    return outcome


def _input(*, retained: bool = False, pdb: str = _PDB) -> ProteinStructure:
    if not retained:
        return read_structure_string(pdb, FileFormat.PDB)
    pdb = _TYPED_LINK + "\n".join(
        "HETATM" + row[6:] if row.startswith("ATOM") and row[17:20] == "NME" else row
        for row in pdb.splitlines()
    )
    return read_structure_string(
        pdb, FileFormat.PDB, policy=_KEEP.structure_normalization_policy()
    )


@pytest.mark.parametrize("retained", [False, True])
def test_linked_nme_gets_one_amide_and_three_methyl_hydrogens(retained):
    source = _input(retained=retained)
    result = process_structure(source, ingress=_KEEP, requested_goals=_GOALS)
    structure = result.structure
    constitution = structure.constitution
    nme = constitution.residue_or_ligand(_NME)
    assert nme is not None and nme.is_hetero is retained
    parents = structure.topology.residue_hydrogen_anchor_by_name(
        constitution=constitution, residue_index=constitution.residue_index(_NME)
    )
    assert sorted(parents.values()) == ["C", "C", "C", "N"]
    assert {atom.name for atom in nme.atom_sites if atom.is_hydrogen()} == set(parents)
    assert constitution.resolve_atom_index(AtomRef(_ALA, "OXT")) is None
    assert not any(
        issue.kind is ValidationIssueKind.UNRESOLVED_POLYMER_HYDROGEN
        for issue in result.issues
    )
    assert _h_outcome(result).status is RequestedGoalStatus.SATISFIED
    for residue in source.constitution.iter_residues(include_ligands=True):
        for index in source.constitution.atom_indices_for_residue(residue.residue_id):
            ref = source.constitution.atom_ref_at(index)
            assert structure.geometry.atom_geometry(constitution.atom_index(ref)) == (
                source.geometry.atom_geometry(index)
            )


@pytest.mark.parametrize("retained", [False, True])
def test_cap_graph_can_be_realized_with_no_hydrogens(retained):
    source = _input(retained=retained)
    resolve = (
        resolve_retained_non_polymer_topology
        if retained
        else resolve_fixed_hydrogen_topology
    )
    library = build_default_component_library()
    result = resolve(source, library, target_residue_ids=frozenset((_NME,)))
    structure = result.structure
    assert structure.constitution == source.constitution
    assert structure.geometry == source.geometry
    for carbon in (AtomRef(_ALA, "C"), AtomRef(_NME, "C")):
        bond = structure.topology.bond_between(
            structure.constitution.atom_index(carbon),
            structure.constitution.atom_index(AtomRef(_NME, "N")),
        )
        assert bond is not None and bond.order == 1
        if retained and carbon.residue_id == _ALA:
            assert bond.source_metadata is not None
            assert bond.source_metadata.reported_order is None
    assert (
        resolve(structure, library, target_residue_ids=frozenset((_NME,))).structure
        is structure
    )


def _hydrogen_names(structure: ProteinStructure) -> set[str]:
    residue = structure.constitution.residue_or_ligand(_NME)
    assert residue is not None
    return {atom.name for atom in residue.atom_sites if atom.is_hydrogen()}


def _nme_bond(structure: ProteinStructure, other: AtomRef) -> TopologyBond | None:
    constitution = structure.constitution
    return structure.topology.bond_between(
        constitution.atom_index(AtomRef(_NME, "N")), constitution.atom_index(other)
    )


def _with_nme_hydrogen_rows(*rows: str) -> str:
    return _PDB.replace("TER      15", "\n".join(rows) + "\nTER      20")


@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize("file_format", [FileFormat.PDB, FileFormat.MMCIF])
def test_capped_chain_round_trip_and_no_conect_native_graph(retained, file_format):
    source = _input(retained=retained)
    result = process_structure(source, ingress=_KEEP, requested_goals=_GOALS)
    actual = to_rdkit_molecule(result.structure)
    assert sum(atom.GetNumRadicalElectrons() for atom in actual.GetAtoms()) == 0
    pdb = write_structure_string(result.structure, FileFormat.PDB)
    no_conect = (
        "\n".join(row for row in pdb.splitlines() if not row.startswith("CONECT"))
        + "\n"
    )
    proximity = Chem.MolFromPDBBlock(no_conect, sanitize=True, removeHs=False)
    assert proximity is not None
    nme_hydrogen_parents = []
    for atom in proximity.GetAtoms():
        info = atom.GetPDBResidueInfo()
        if info.GetResidueName() == "NME" and atom.GetAtomicNum() == 1:
            (parent,) = atom.GetNeighbors()
            parent_info = parent.GetPDBResidueInfo()
            assert parent_info.GetResidueName() == "NME"
            nme_hydrogen_parents.append(parent_info.GetName().strip())
    assert sorted(nme_hydrogen_parents) == ["C", "C", "C", "N"]

    exported = write_structure_string(result.structure, file_format)
    reread = read_structure_string(
        exported, file_format, policy=_KEEP.structure_normalization_policy()
    )
    second = process_structure(reread, ingress=_KEEP, requested_goals=_GOALS)
    assert _hydrogen_names(reread) == _hydrogen_names(second.structure)
    assert second.structure.constitution == reread.constitution
    assert second.structure.geometry == reread.geometry
    assert _h_outcome(second).is_satisfied()


@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize(("name", "element"), [("H", "H"), ("DN", "D"), ("TN1", "T")])
def test_amide_isotope_identity_and_pose_are_not_replaced(retained, name, element):
    source = _input(
        retained=retained,
        pdb=_with_nme_hydrogen_rows(
            f"ATOM     15 {name:>4} NME A   4      -0.813   0.152   2.161  0.63 13.12          {element:>2}"
        ),
    )
    result = process_structure(source, ingress=_KEEP, requested_goals=_GOALS)
    before = source.constitution.atom_index(AtomRef(_NME, name))
    after = result.structure.constitution.atom_index(AtomRef(_NME, name))
    assert result.structure.constitution.atom_site_at(
        after
    ) == source.constitution.atom_site_at(before)
    assert result.structure.geometry.atom_geometry(
        after
    ) == source.geometry.atom_geometry(before)
    assert _hydrogen_names(result.structure) == {name, "H1", "H2", "H3"}
    assert _nme_bond(result.structure, AtomRef(_NME, name)) is not None


@pytest.mark.parametrize("retained", [False, True])
@pytest.mark.parametrize(
    "evidence", ["source", "no_observation", "removed_from_current"]
)
def test_excess_amide_h_cannot_be_deleted_or_count_as_complete(retained, evidence):
    source = _input(
        retained=retained,
        pdb=_with_nme_hydrogen_rows(
            "ATOM     15  HN1 NME A   4      -0.813   0.152   2.161  1.00 20.00           H",
            "ATOM     16  HN2 NME A   4      -0.916   0.829   2.389  1.00 20.00           H",
        ),
    )
    if evidence == "no_observation":
        provenance = source.provenance
        source = source.with_provenance(
            replace(provenance, ingress=replace(provenance.ingress, observation=None))
        )
    elif evidence == "removed_from_current":
        source = source.without_atom_refs((AtomRef(_NME, "HN2"),))
    action = (
        add_retained_non_polymer_hydrogens if retained else materialize_hydrogens_core
    )
    direct = action(source, target_residue_ids=frozenset((_NME,)))
    assert direct.structure is source
    assert (
        any("excess" in issue.message for issue in direct.issues)
        or evidence == "removed_from_current"
    )
    result = process_structure(source, ingress=_KEEP, requested_goals=_GOALS)
    assert not _h_outcome(result).is_satisfied()
    assert _hydrogen_names(result.structure) == _hydrogen_names(source)
    assert (
        result.structure.constitution.resolve_atom_index(AtomRef(_ALA, "OXT")) is None
    )


@pytest.mark.parametrize("retained", [False, True])
def test_placement_failure_keeps_the_admitted_heavy_graph(retained):
    collinear = _PDB.replace("-2.657   0.528   3.194", "-1.315   0.349   3.264")
    source = _input(retained=retained, pdb=collinear)
    result = process_structure(source, ingress=_KEEP, requested_goals=_GOALS)
    assert not _h_outcome(result).is_satisfied()
    assert not _hydrogen_names(result.structure)
    for ref in (AtomRef(_NME, "C"), AtomRef(_ALA, "C")):
        bond = _nme_bond(result.structure, ref)
        assert bond is not None and bond.order == 1
    assert (
        result.structure.constitution.resolve_atom_index(AtomRef(_ALA, "OXT")) is None
    )


def test_no_h_request_does_not_realize_the_retained_cap_bonds():
    source = _input(retained=True)
    result = process_structure(source, ingress=_KEEP)
    assert result.structure.topology == source.topology
    assert not _hydrogen_names(result.structure)


@pytest.mark.parametrize("retained", [False, True])
def test_local_force_field_request_preserves_cap_and_can_prepare_independent_h(
    retained,
):
    source = _input(retained=retained)
    result = process_structure(
        source,
        ingress=_KEEP,
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((_ALA,)),
                binding=ManualContinuousRelaxationBinding(
                    ContinuousRelaxationForceField.UFF
                ),
            )
        ),
    )
    assert not _hydrogen_names(result.structure)
    assert (
        result.structure.constitution.resolve_atom_index(AtomRef(_ALA, "HA"))
        is not None
    )
    bond = _nme_bond(result.structure, AtomRef(_ALA, "C"))
    assert bond is not None and bond.order == (None if retained else 1)
    assert _nme_bond(result.structure, AtomRef(_NME, "C")) is None
    assert not result.relaxation_executions


def test_unavailable_native_placement_does_not_undo_the_cap_graph(monkeypatch):
    monkeypatch.setattr(
        "protrepair.transformer.completion.hydrogen.linked_caps.Chem", None
    )
    source = _input(retained=True)
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((_NME,))
    )
    assert not _hydrogen_names(result.structure)
    bond = _nme_bond(result.structure, AtomRef(_NME, "C"))
    assert bond is not None and bond.order == 1
    assert any("operational RDKit" in issue.message for issue in result.issues)


def test_local_hydrogen_plan_preserves_ace_preparation_but_does_not_select_nme():
    ace = ResidueId("A", 1)
    plan = plan_hydrogen_completion_transformers(
        _input(),
        requested_goals=RequestedGoalSet(()),
        component_library=build_default_component_library(),
        required_residue_ids=(ace, _ALA, _NME),
    )
    (transformer,) = plan.transformers
    assert ace in transformer.scope.residue_ids
    assert _ALA in transformer.scope.residue_ids
    assert _NME not in transformer.scope.residue_ids


def test_hydrogen_support_does_not_grant_general_heavy_completion():
    heavy = requested_process_goal(
        scope=WholeStructureScope(), value=BackboneHeavyAtomCompletenessState.COMPLETE
    )
    result = process_structure(_input(), requested_goals=(*_GOALS, heavy))
    report = result.requested_goal_report
    assert report is not None
    hydrogen_outcome, heavy_outcome = report.outcomes
    assert hydrogen_outcome.is_satisfied()
    assert heavy_outcome.status is RequestedGoalStatus.BLOCKED


@pytest.mark.parametrize("retained", [False, True])
def test_full_inventory_with_unpublished_bonds_is_realized_without_adding_h(retained):
    source = _input(
        retained=retained,
        pdb=_with_nme_hydrogen_rows(
            "ATOM     15  H1  NME A   4      -2.071   0.352   4.108  1.00 20.00           H",
            "ATOM     16  H2  NME A   4      -3.007   1.571   3.178  1.00 20.00           H",
            "ATOM     17  H3  NME A   4      -3.523  -0.150   3.178  1.00 20.00           H",
            "ATOM     18  HN2 NME A   4      -0.813   0.152   2.161  1.00 20.00           H",
        ),
    )
    assert _nme_bond(source, AtomRef(_NME, "HN2")) is None
    result = process_structure(source, ingress=_KEEP, requested_goals=_GOALS)
    assert _h_outcome(result).status is RequestedGoalStatus.SATISFIED
    assert _hydrogen_names(result.structure) == _hydrogen_names(source)
    assert _nme_bond(result.structure, AtomRef(_NME, "HN2")) is not None
    assert result.structure.residue_geometry(
        result.structure.constitution.residue_index(_NME)
    ) == (source.residue_geometry(source.constitution.residue_index(_NME)))


def test_a_missing_retained_cap_partner_cannot_trigger_isolated_fallback():
    source = _input(retained=True).without_atom_refs((AtomRef(_ALA, "C"),))
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((_NME,))
    )
    assert result.structure is source
    assert any("surviving covalent partner" in issue.message for issue in result.issues)


def test_polymer_only_direct_target_cannot_publish_retained_cap_topology():
    source = _input(retained=True)
    result = materialize_hydrogens_core(source, target_residue_ids=frozenset((_ALA,)))
    assert not _hydrogen_names(result.structure)
    assert _nme_bond(result.structure, AtomRef(_NME, "C")) is None
    bond = _nme_bond(result.structure, AtomRef(_ALA, "C"))
    assert bond is not None and bond.order is None
    assert (
        result.structure.constitution.resolve_atom_index(AtomRef(_ALA, "OXT")) is None
    )


def test_standalone_retained_nme_keeps_its_five_h_path():
    source = read_structure_string(
        "\n".join(
            "HETATM" + row[6:]
            for row in _PDB.splitlines()
            if row.startswith("ATOM") and row[17:20] == "NME"
        )
        + "\nEND\n",
        FileFormat.PDB,
        policy=_KEEP.structure_normalization_policy(),
    )
    result = process_structure(source, ingress=_KEEP, requested_goals=_GOALS)
    assert len(_hydrogen_names(result.structure)) == 5
    assert _h_outcome(result).is_satisfied()


def test_bare_conect_does_not_admit_a_retained_cap():
    pdb = _PDB.replace("ATOM     13", "HETATM   13").replace(
        "ATOM     14", "HETATM   14"
    )
    pdb = pdb.replace("END\n", "CONECT   10   13\nEND\n")
    source = read_structure_string(
        pdb, FileFormat.PDB, policy=_KEEP.structure_normalization_policy()
    )
    result = process_structure(source, ingress=_KEEP, requested_goals=_GOALS)
    assert not _hydrogen_names(result.structure)
    assert not _h_outcome(result).is_satisfied()
    bond = _nme_bond(result.structure, AtomRef(_ALA, "C"))
    assert bond is not None and bond.relationship_type is BondRelationshipType.UNKNOWN


@pytest.mark.parametrize("explicit", [False, True])
def test_nme_methyl_is_never_an_outgoing_peptide_port(explicit):
    gly = []
    for serial, row in enumerate(_PDB.splitlines()[3:7], start=15):
        gly.append(
            f"HETATM{serial:5d}{row[11:22]}   5{row[26:30]}"
            f"{float(row[30:38]) - 4.685:8.3f}{float(row[38:46]) - 0.681:8.3f}"
            f"{float(row[46:54]) + 3.912:8.3f}{row[54:]}"
        )
    wrong_port_link = "LINK         C   NME A   4                 N   GLY A   5\n"
    source = read_structure_string(
        (wrong_port_link if explicit else "")
        + _PDB.split("TER")[0]
        + "\n".join(gly)
        + "\nEND\n",
        FileFormat.PDB,
        policy=_KEEP.structure_normalization_policy(),
    )
    constitution = source.constitution
    left, right = (
        constitution.residue_or_ligand(ref) for ref in (_NME, ResidueId("A", 5))
    )
    assert left is not None and right is not None and right.is_hetero
    assert not are_peptide_adjacent(left, right, structure=source)
    bond = source.topology.bond_between(
        constitution.atom_index(AtomRef(_NME, "C")),
        constitution.atom_index(AtomRef(right.residue_id, "N")),
    )
    assert (bond is not None) is explicit

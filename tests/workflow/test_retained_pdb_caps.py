"""Retained caps whose PDB connectivity comes from OpenMM CONECT records."""

from dataclasses import replace
from pathlib import Path

import pytest
from rdkit import Chem

from protrepair import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.io import read_structure, read_structure_string, write_structure_string
from protrepair.io.pdb_projection import (
    pdb_without_conect,
    pdb_without_conect_for_parser_probe,
    prepare_rdkit_no_conect_pdb_block_projector,
)
from protrepair.scope import WholeStructureScope
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.state import HydrogenCoverageState
from protrepair.structure import AtomRef, FileFormat, ProteinStructure, ResidueId
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondRecordType,
    TopologyBond,
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
    StructureIngressOptions,
    WorkflowTransformRequests,
    requested_process_goal,
)

pytestmark = pytest.mark.workflow

_FIXTURES = Path(__file__).parents[1] / "fixtures" / "pdb" / "linked-caps"
_HEAVY = _FIXTURES / "openmm-heavy.pdb"
_KEEP = StructureIngressOptions(ligand_policy=LigandPolicy.KEEP)
_GOALS = (
    requested_process_goal(
        scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
    ),
)
_ACE, _GLY, _ALA, _NME = (ResidueId("A", number) for number in range(1, 5))
_ACE_LINK = (AtomRef(_ACE, "C"), AtomRef(_GLY, "N"))
_NME_LINK = (AtomRef(_ALA, "C"), AtomRef(_NME, "N"))


def _bond(structure: ProteinStructure, pair: tuple[AtomRef, AtomRef]) -> TopologyBond:
    indices = tuple(structure.constitution.atom_index(ref) for ref in pair)
    bond = structure.topology.bond_between(*indices)
    assert bond is not None
    return bond


@pytest.mark.parametrize("name", ["openmm-heavy.pdb", "openmm-populated.pdb"])
@pytest.mark.parametrize("canonical_input", [False, True])
def test_openmm_pdb_h_goal_prepares_both_retained_caps(name, canonical_input):
    path = _FIXTURES / name
    source = read_structure(path, policy=_KEEP.structure_normalization_policy())
    for pair in (_ACE_LINK, _NME_LINK):
        bond = _bond(source, pair)
        assert bond.relationship_type is BondRelationshipType.UNKNOWN
        assert bond.order is None

    result = process_structure(
        source if canonical_input else path, ingress=_KEEP, requested_goals=_GOALS
    )
    assert result.requested_goal_report is not None
    assert result.requested_goal_report.outcomes[0].is_satisfied()
    assert result.issues == ()
    repaired = result.structure
    assert repaired.provenance.ingress == source.provenance.ingress
    assert repaired.constitution.resolve_atom_index(AtomRef(_ALA, "OXT")) is None

    for pair in (_ACE_LINK, _NME_LINK):
        bond = _bond(repaired, pair)
        assert bond.relationship_type is BondRelationshipType.COVALENT
        assert bond.order == 1
        assert bond.provenance is BondProvenance.SOURCE_EXPLICIT
        assert bond.source_metadata == _bond(source, pair).source_metadata
        assert bond.source_metadata is not None
        assert bond.source_metadata.record_type is SourceBondRecordType.PDB_CONECT
        assert bond.source_metadata.reported_order is None
        assert (
            bond.source_metadata.reported_relationship_type
            is BondRelationshipType.UNKNOWN
        )

    for residue_id, expected in ((_ACE, ["CH3"] * 3), (_NME, ["C", "C", "C", "N"])):
        residue = repaired.constitution.residue_or_ligand(residue_id)
        assert residue is not None and residue.is_hetero
        parents = repaired.topology.residue_hydrogen_anchor_by_name(
            constitution=repaired.constitution,
            residue_index=repaired.constitution.residue_index(residue_id),
        )
        assert sorted(parents.values()) == expected
        assert set(parents) == {
            atom.name for atom in residue.atom_sites if atom.is_hydrogen()
        }

    for residue in source.constitution.iter_residues(include_ligands=True):
        for index in source.constitution.atom_indices_for_residue(residue.residue_id):
            target = repaired.constitution.atom_index(
                source.constitution.atom_ref_at(index)
            )
            assert repaired.constitution.atom_site_at(
                target
            ) == source.constitution.atom_site_at(index)
            assert repaired.geometry.atom_geometry(
                target
            ) == source.geometry.atom_geometry(index)
            declared_charge = source.topology.formal_charge(index)
            effective_charge = repaired.topology.formal_charge(target)
            if residue.is_hetero or declared_charge is not None:
                assert effective_charge == declared_charge
            else:
                assert effective_charge in (None, 0)
    if name == "openmm-populated.pdb":
        assert repaired.constitution == source.constitution


def _read(text: str) -> ProteinStructure:
    return read_structure_string(
        text, FileFormat.PDB, policy=_KEEP.structure_normalization_policy()
    )


@pytest.mark.parametrize(
    ("cap", "pair", "other"),
    [
        (_ACE, _ACE_LINK, _NME_LINK),
        (_NME, _NME_LINK, _ACE_LINK),
    ],
)
def test_retained_cap_direct_helpers_keep_target_scope(cap, pair, other):
    source = _read(_HEAVY.read_text())
    library = build_default_component_library()
    targets = frozenset((cap,))
    graph = resolve_retained_non_polymer_topology(
        source, library, target_residue_ids=targets
    )
    assert graph.issues == ()
    assert graph.structure.geometry is source.geometry
    assert graph.structure.constitution is source.constitution
    assert graph.structure.topology.atom_topologies == source.topology.atom_topologies
    assert _bond(graph.structure, pair).order == 1
    assert _bond(graph.structure, other) == _bond(source, other)

    h = add_retained_non_polymer_hydrogens(source, library, target_residue_ids=targets)
    assert h.issues == ()
    assert _bond(h.structure, pair).order == 1
    assert _bond(h.structure, other).relationship_type is BondRelationshipType.UNKNOWN
    residue = h.structure.constitution.residue_or_ligand(cap)
    assert residue is not None
    assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == (
        3 if cap == _ACE else 4
    )


def test_pdb_retention_does_not_realize_the_bond_without_an_h_request():
    source = read_structure(_HEAVY, policy=_KEEP.structure_normalization_policy())
    readonly = process_structure(source, ingress=_KEEP)
    assert readonly.structure == source
    assert readonly.repairs == ()
    assert not any(
        row.startswith("LINK")
        for row in write_structure_string(source, FileFormat.PDB).splitlines()
    )

    dropped = process_structure(_HEAVY, requested_goals=_GOALS)
    assert dropped.structure.constitution.ligands == ()
    assert (
        dropped.structure.constitution.resolve_atom_index(AtomRef(_ALA, "OXT"))
        is not None
    )
    assert dropped.requested_goal_report is not None
    assert dropped.requested_goal_report.outcomes[0].is_satisfied()


def test_local_refinement_does_not_authorize_retained_cap_preparation():
    source = _read(_HEAVY.read_text())
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
    assert (
        result.structure.constitution.resolve_atom_index(AtomRef(_ALA, "HA"))
        is not None
    )
    for pair in (_ACE_LINK, _NME_LINK):
        assert (
            _bond(result.structure, pair).relationship_type
            is BondRelationshipType.UNKNOWN
        )
    for cap in result.structure.constitution.ligands:
        assert not any(atom.is_hydrogen() for atom in cap.atom_sites)


def test_zero_occupancy_projection_keeps_unaffected_cap_evidence():
    text = _HEAVY.read_text()
    row = next(row for row in text.splitlines() if row.startswith("ATOM     12"))
    ingress = StructureIngressOptions(
        ligand_policy=LigandPolicy.KEEP, omit_zero_occupancy_sidechain_heavy_atoms=True
    )
    source = read_structure_string(
        text.replace(row, row[:54] + "  0.00" + row[60:]),
        FileFormat.PDB,
        policy=ingress.structure_normalization_policy(),
    )
    assert source.constitution.resolve_atom_index(AtomRef(_ALA, "CB")) is None
    graph = resolve_retained_non_polymer_topology(
        source, build_default_component_library()
    )
    assert graph.issues == ()
    for pair in (_ACE_LINK, _NME_LINK):
        assert _bond(graph.structure, pair).order == 1


def test_realized_links_export_without_changing_the_rdkit_diagnostic():
    repaired = process_structure(
        _HEAVY, ingress=_KEEP, requested_goals=_GOALS
    ).structure
    pdb = write_structure_string(repaired, FileFormat.PDB)
    assert sum(row.startswith("LINK") for row in pdb.splitlines()) == 2
    assert any(row.startswith("CONECT") for row in pdb.splitlines())
    native_pdb = pdb_without_conect(repaired)
    assert not any(
        row.startswith(("CONECT", "LINK")) for row in native_pdb.splitlines()
    )
    projector = prepare_rdkit_no_conect_pdb_block_projector(repaired)
    assert projector is not None
    assert pdb_without_conect_for_parser_probe(repaired, projector) == native_pdb
    molecule = Chem.MolFromPDBBlock(native_pdb, sanitize=True, removeHs=False)
    assert molecule is not None
    assert sum(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms()) == 0
    native_links = {
        frozenset(
            (info.GetResidueName().strip(), info.GetName().strip())
            for info in (
                bond.GetBeginAtom().GetPDBResidueInfo(),
                bond.GetEndAtom().GetPDBResidueInfo(),
            )
        )
        for bond in molecule.GetBonds()
    }
    assert frozenset((("ACE", "C"), ("GLY", "N"))) in native_links
    assert frozenset((("ALA", "C"), ("NME", "N"))) in native_links

    for file_format in (FileFormat.PDB, FileFormat.MMCIF):
        payload = write_structure_string(repaired, file_format)
        source = read_structure_string(
            payload, file_format, policy=_KEEP.structure_normalization_policy()
        )
        result = process_structure(source, ingress=_KEEP, requested_goals=_GOALS)
        assert result.requested_goal_report is not None
        assert result.requested_goal_report.outcomes[0].is_satisfied()
        assert result.issues == ()
        assert result.structure.constitution == source.constitution
        assert result.structure.geometry == source.geometry
        for pair in (_ACE_LINK, _NME_LINK):
            assert (
                _bond(source, pair).relationship_type is BondRelationshipType.COVALENT
            )
            assert _bond(result.structure, pair).order == 1


@pytest.mark.parametrize(
    ("old", "new", "rejected", "admitted"),
    [
        pytest.param("HETATM   13", "TER\nHETATM   13", _NME_LINK, _ACE_LINK, id="ter"),
        pytest.param(
            "HETATM   14", "TER\nHETATM   14", _NME_LINK, _ACE_LINK, id="split-slot"
        ),
        pytest.param(
            "CONECT   10   13",
            "CONECT   10   13   13",
            _NME_LINK,
            _ACE_LINK,
            id="multiple-order",
        ),
        pytest.param(
            "CONECT   10   13",
            "CONECT   10   13   99",
            _NME_LINK,
            _ACE_LINK,
            id="missing-target",
        ),
        pytest.param(
            "END\n",
            "CONECT   99   13\nEND\n",
            _NME_LINK,
            _ACE_LINK,
            id="missing-source",
        ),
        pytest.param(
            "END\n", "CONECT        13\nEND\n", _NME_LINK, _ACE_LINK, id="blank-source"
        ),
        pytest.param(
            "CONECT    3    1",
            "CONECT    3    1   99",
            _ACE_LINK,
            _NME_LINK,
            id="cap-local-row",
        ),
        pytest.param(
            "CONECT   10   13",
            "HETATM   99  O   HOH B   1       0.100   5.000   1.000  1.00  0.00           O\n"
            "CONECT   10   13   99",
            _NME_LINK,
            _ACE_LINK,
            id="excluded-competitor",
        ),
    ],
)
def test_unqualified_attachment_stays_unknown(old, new, rejected, admitted):
    text = _HEAVY.read_text()
    assert old in text
    source = _read(text.replace(old, new))
    result = process_structure(source, ingress=_KEEP, requested_goals=_GOALS)
    assert result.requested_goal_report is not None
    assert not result.requested_goal_report.outcomes[0].is_satisfied()
    rejected_bond = _bond(result.structure, rejected)
    assert rejected_bond.relationship_type is BondRelationshipType.UNKNOWN
    assert rejected_bond.order == _bond(source, rejected).order
    assert _bond(result.structure, admitted).order == 1
    assert (
        result.structure.constitution.resolve_atom_index(AtomRef(_ALA, "OXT")) is None
    )


def test_unbound_canonical_attachments_do_not_gain_pdb_authority():
    source = _read(_HEAVY.read_text())
    provenance = source.provenance
    carrier = source.with_provenance(
        replace(
            provenance, ingress=replace(provenance.ingress, pdb_adjacent_cap_links=())
        )
    )
    result = process_structure(carrier, ingress=_KEEP, requested_goals=_GOALS)
    for pair in (_ACE_LINK, _NME_LINK):
        assert (
            _bond(result.structure, pair).relationship_type
            is BondRelationshipType.UNKNOWN
        )
    assert result.requested_goal_report is not None
    assert not result.requested_goal_report.outcomes[0].is_satisfied()


@pytest.mark.parametrize("selected", ["A", "B"])
def test_bare_conect_cannot_rebind_to_another_selected_conformer(selected):
    text = _HEAVY.read_text()
    row = next(row for row in text.splitlines() if row.startswith("HETATM   13"))
    alternatives = []
    for serial, altloc in ((13, "A"), (40, "B")):
        occupancy = 0.75 if altloc == selected else 0.25
        alternatives.append(
            f"HETATM{serial:5d}{row[11:16]}{altloc}{row[17:54]}{occupancy:6.2f}{row[60:]}"
        )
    source = _read(text.replace(row, "\n".join(alternatives)))
    nitrogen = source.constitution.atom_index(AtomRef(_NME, "N"))
    assert source.geometry.atom_geometry(nitrogen).altloc == selected
    graph = resolve_retained_non_polymer_topology(
        source, build_default_component_library(), target_residue_ids=frozenset((_NME,))
    )
    indices = tuple(graph.structure.constitution.atom_index(ref) for ref in _NME_LINK)
    bond = graph.structure.topology.bond_between(*indices)
    if selected == "A":
        assert (
            bond is not None and bond.relationship_type is BondRelationshipType.COVALENT
        )
        payload = write_structure_string(graph.structure, FileFormat.PDB)
        assert (
            _bond(_read(payload), _NME_LINK).relationship_type
            is BondRelationshipType.COVALENT
        )
    else:
        assert bond is None


def test_different_serials_for_one_cap_atom_do_not_prove_an_exact_occurrence():
    text = _HEAVY.read_text()
    row = next(row for row in text.splitlines() if row.startswith("HETATM   13"))
    source = _read(text.replace(row, row + "\n" + "HETATM   40" + row[11:]))
    assert _bond(source, _NME_LINK).relationship_type is BondRelationshipType.UNKNOWN
    graph = resolve_retained_non_polymer_topology(
        source, build_default_component_library(), target_residue_ids=frozenset((_NME,))
    )
    assert graph.structure is source
    assert graph.issues


@pytest.mark.parametrize("later_model_only", [False, True])
def test_later_model_conect_cannot_establish_first_model_chemistry(later_model_only):
    coordinates, _, remainder = _HEAVY.read_text().partition("CONECT")
    conect = ("CONECT" + remainder).removesuffix("END\n")
    models = (
        f"MODEL        1\n{coordinates}ENDMDL\n"
        f"MODEL        2\n{coordinates}{conect if later_model_only else ''}ENDMDL\n"
    )
    source = _read(models + ("" if later_model_only else conect) + "END\n")
    for pair in (_ACE_LINK, _NME_LINK):
        assert _bond(source, pair).relationship_type is BondRelationshipType.UNKNOWN
    result = process_structure(source, ingress=_KEEP, requested_goals=_GOALS)
    assert result.requested_goal_report is not None
    assert result.requested_goal_report.outcomes[0].is_satisfied() != later_model_only
    expected = (
        BondRelationshipType.UNKNOWN
        if later_model_only
        else BondRelationshipType.COVALENT
    )
    for pair in (_ACE_LINK, _NME_LINK):
        assert _bond(result.structure, pair).relationship_type is expected


@pytest.mark.parametrize("isotope", ["D", "T"])
def test_openmm_cap_repair_preserves_a_source_amide_isotope(isotope):
    text = (_FIXTURES / "openmm-populated.pdb").read_text()
    row = next(row for row in text.splitlines() if row.startswith("HETATM   25"))
    source = _read(text.replace(row, row[:13] + isotope + row[14:77] + isotope))
    ref = AtomRef(_NME, isotope)
    index = source.constitution.atom_index(ref)
    assert source.constitution.atom_site_at(index).element == isotope
    result = process_structure(source, ingress=_KEEP, requested_goals=_GOALS)
    assert result.requested_goal_report is not None
    assert result.requested_goal_report.outcomes[0].is_satisfied()
    prepared = result.structure
    target = prepared.constitution.atom_index(ref)
    assert prepared.constitution.atom_site_at(
        target
    ) == source.constitution.atom_site_at(index)
    assert prepared.geometry.atom_geometry(target) == source.geometry.atom_geometry(
        index
    )
    assert _bond(prepared, (AtomRef(_NME, "N"), ref)).order == 1


def test_staged_ace_rejects_a_partner_with_free_terminal_hydrogens():
    text = _HEAVY.read_text().replace(
        "ATOM      5",
        "ATOM     40  H1  GLY A   2       3.000   2.100  -0.700  1.00  0.00           H\n"
        "ATOM     41  H2  GLY A   2       3.200   0.300  -1.200  1.00  0.00           H\n"
        "ATOM      5",
    )
    source = _read(text)
    result = add_retained_non_polymer_hydrogens(
        source, build_default_component_library(), target_residue_ids=frozenset((_ACE,))
    )
    assert result.issues
    assert result.structure is source
    assert not result.repairs


def test_retained_chemistry_override_takes_precedence_over_the_stock_cap():
    source = _read(_HEAVY.read_text())
    override = RetainedNonPolymerChemistryOverride(_NME, "NC", ("N", "C"))
    ingress = StructureIngressOptions(
        ligand_policy=LigandPolicy.KEEP,
        retained_non_polymer_chemistry_overrides=(override,),
    )
    result = process_structure(source, ingress=ingress, requested_goals=_GOALS)
    assert _bond(result.structure, _ACE_LINK).order == 1
    assert (
        _bond(result.structure, _NME_LINK).relationship_type
        is BondRelationshipType.UNKNOWN
    )

from typing import NoReturn
from unittest.mock import patch

import pytest
from rdkit import Chem
from rdkit.Chem.rdchem import Mol
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    residue_payload,
)

from protrepair.diagnostics.parser_readability import (
    measure_rdkit_no_conect_sanitize_readability,
    probe_rdkit_no_conect_parser_readability,
)
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.io.pdb_projection import (
    pdb_without_conect,
    prepare_rdkit_no_conect_pdb_block_projector,
)
from protrepair.state import ParserCompatibilityState, StructureParserCompatibilityFacts
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat


def _cleanup_repairable_structure(
    *,
    hydrogen_element: str | None = "H",
    metal_first: bool = True,
    with_invalid_carbon: bool = False,
    invalid_first: bool = False,
    with_extra_carbon: bool = False,
) -> ProteinStructure:
    atoms = (
        atom_payload("FE", "FE", Vec3(0.0, 0.0, 2.3), formal_charge=3),
        atom_payload("N1", "N", Vec3(0.0, 0.0, 0.0), formal_charge=-1),
        atom_payload("C1", "C", Vec3(1.3, 0.0, 0.0)),
        atom_payload("C2", "C", Vec3(-1.3, 0.0, 0.0)),
    )
    if not metal_first:
        atoms = (*atoms[1:], atoms[0])
    if with_extra_carbon:
        atoms = (*atoms, atom_payload("C3", "C", Vec3(0.0, 4.0, 0.0)))
    if hydrogen_element is not None:
        atoms = (
            *atoms,
            atom_payload("H1", hydrogen_element, Vec3(2.35, 0.0, 0.0)),
        )
    residues = (
        residue_payload(
            component_id="LIG",
            residue_id=ResidueId("A", 1),
            is_hetero=True,
            atoms=atoms,
        ),
    )
    if with_invalid_carbon:
        invalid = residue_payload(
            component_id="BAD",
            residue_id=ResidueId("A", 2),
            is_hetero=True,
            atoms=(
                atom_payload("C1", "C", Vec3(20.0, 0.0, 0.0)),
                atom_payload("H1", "H", Vec3(21.0, 0.0, 0.0)),
                atom_payload("H2", "H", Vec3(19.0, 0.0, 0.0)),
                atom_payload("H3", "H", Vec3(20.0, 1.0, 0.0)),
                atom_payload("H4", "H", Vec3(20.0, -1.0, 0.0)),
                atom_payload("H5", "H", Vec3(20.0, 0.0, 1.0)),
            ),
        )
        residues = (invalid, *residues) if invalid_first else (*residues, invalid)
    return build_structure(chains=(), ligands=residues, source_format=FileFormat.PDB)


@pytest.mark.parametrize("hydrogen_element", ("H", "D", "T"))
@pytest.mark.parametrize("metal_first", (False, True))
def test_probe_honors_native_organometallic_cleanup(
    hydrogen_element: str, metal_first: bool
) -> None:
    structure = _cleanup_repairable_structure(
        hydrogen_element=hydrogen_element, metal_first=metal_first
    )
    text = pdb_without_conect(structure)
    raw = Chem.MolFromPDBBlock(
        text, sanitize=False, removeHs=False, proximityBonding=True
    )
    assert raw is not None
    assert Chem.DetectChemistryProblems(raw)
    native = Chem.MolFromPDBBlock(
        text, sanitize=True, removeHs=False, proximityBonding=True
    )
    assert native is not None
    assert any(bond.GetBondType() == Chem.BondType.DATIVE for bond in native.GetBonds())

    before = structure.topology
    before_positions = tuple(structure.geometry.iter_positions())
    probe = probe_rdkit_no_conect_parser_readability(structure)

    assert probe.sanitize_readable is True
    assert measure_rdkit_no_conect_sanitize_readability(structure) is True
    assert not probe.parser_problem_witnesses()
    assert not probe.issues()
    assert structure.topology == before
    assert tuple(structure.geometry.iter_positions()) == before_positions
    assert (
        structure.topology.formal_charge(
            structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "N1"))
        )
        == -1
    )


@pytest.mark.parametrize("invalid_first", (False, True))
def test_probe_localizes_real_failure_after_metal_cleanup(invalid_first: bool) -> None:
    structure = _cleanup_repairable_structure(
        with_invalid_carbon=True, invalid_first=invalid_first
    )
    assert measure_rdkit_no_conect_sanitize_readability(structure) is False

    probe = probe_rdkit_no_conect_parser_readability(structure)

    assert probe.sanitize_readable is False
    assert tuple(w.residue_id for w in probe.parser_problem_witnesses()) == (
        ResidueId("A", 2),
    )
    assert probe.parser_problem_witnesses()[0].atom_names == ("C1",)
    assert probe.issues()


def test_probe_keeps_missing_hydrogen_readability_unknown() -> None:
    structure = _cleanup_repairable_structure(hydrogen_element=None)

    with patch.object(Chem, "MolFromPDBBlock", wraps=Chem.MolFromPDBBlock) as parser:
        probe = probe_rdkit_no_conect_parser_readability(structure)

    assert probe.sanitize_readable is None
    assert not probe.issues()
    assert parser.call_count == 0


def test_successful_probe_parses_and_sanitizes_once() -> None:
    structure = _cleanup_repairable_structure()
    with (
        patch.object(Chem, "MolFromPDBBlock", wraps=Chem.MolFromPDBBlock) as parser,
        patch.object(Chem, "SanitizeMol", wraps=Chem.SanitizeMol) as sanitizer,
        patch.object(
            Chem, "DetectChemistryProblems", wraps=Chem.DetectChemistryProblems
        ) as localizer,
    ):
        probe = probe_rdkit_no_conect_parser_readability(structure)

    assert probe.sanitize_readable is True
    assert parser.call_count == 1
    assert sanitizer.call_count == 1
    assert localizer.call_count == 0


def test_probe_keeps_unlocalized_native_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structure = _cleanup_repairable_structure()

    def _fail_sanitation(molecule: Mol) -> NoReturn:
        raise RuntimeError("native sanitation failed outside the valence checks")

    monkeypatch.setattr(Chem, "SanitizeMol", _fail_sanitation)
    monkeypatch.setattr(Chem, "DetectChemistryProblems", lambda molecule: ())

    probe = probe_rdkit_no_conect_parser_readability(structure)

    assert probe.sanitize_readable is False
    assert not probe.parser_problem_witnesses()
    assert probe.structure_messages == (
        "native sanitation failed outside the valence checks",
    )
    assert probe.issues()


def test_probe_retains_native_failure_when_localization_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structure = _cleanup_repairable_structure(with_invalid_carbon=True)

    def _fail_localization(molecule: Mol) -> NoReturn:
        raise RuntimeError("localization is unavailable")

    monkeypatch.setattr(Chem, "DetectChemistryProblems", _fail_localization)

    probe = probe_rdkit_no_conect_parser_readability(structure)

    assert probe.sanitize_readable is False
    assert not probe.parser_problem_witnesses()
    assert "Explicit valence" in probe.structure_messages[0]
    assert "localization is unavailable" not in probe.structure_messages[0]
    assert probe.issues()


def test_probe_keeps_parse_failure_distinct_from_sanitation_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    structure = _cleanup_repairable_structure()
    monkeypatch.setattr(Chem, "MolFromPDBBlock", lambda *args, **kwargs: None)

    with patch.object(Chem, "SanitizeMol", wraps=Chem.SanitizeMol) as sanitizer:
        probe = probe_rdkit_no_conect_parser_readability(structure)

    assert probe.sanitize_readable is False
    assert probe.structure_messages == ("_pdb_parse_failed_before_sanitize",)
    assert sanitizer.call_count == 0
    assert probe.issues()


def test_readability_does_not_establish_native_charge_fidelity() -> None:
    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("A", 1),
                is_hetero=True,
                atoms=(
                    atom_payload("N1", "N", Vec3(0.0, 0.0, 0.0), formal_charge=0),
                    atom_payload("H1", "H", Vec3(1.0, 0.0, 0.0)),
                    atom_payload("H2", "H", Vec3(-1.0, 0.0, 0.0)),
                    atom_payload("H3", "H", Vec3(0.0, 1.0, 0.0)),
                    atom_payload("H4", "H", Vec3(0.0, -1.0, 0.0)),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    native = Chem.MolFromPDBBlock(
        pdb_without_conect(structure),
        sanitize=True,
        removeHs=False,
        proximityBonding=True,
    )
    assert native is not None
    assert native.GetAtomWithIdx(0).GetFormalCharge() == 1

    probe = probe_rdkit_no_conect_parser_readability(structure)

    assert probe.sanitize_readable is True
    assert (
        structure.topology.formal_charge(
            structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "N1"))
        )
        == 0
    )


@pytest.mark.parametrize("file_format", (FileFormat.PDB, FileFormat.MMCIF))
def test_cleanup_readability_survives_coordinate_ingress(
    file_format: FileFormat,
) -> None:
    source = _cleanup_repairable_structure(hydrogen_element="D")
    restored = read_structure_string(
        write_structure_string(source, file_format), file_format
    )

    probe = probe_rdkit_no_conect_parser_readability(restored)

    assert probe.sanitize_readable is True
    assert measure_rdkit_no_conect_sanitize_readability(restored) is True
    assert not probe.issues()
    nitrogen = restored.constitution.atom_index(AtomRef(ResidueId("A", 1), "N1"))
    assert restored.topology.formal_charge(nitrogen) == -1
    deuterium = restored.constitution.resolve_atom_site(
        AtomRef(ResidueId("A", 1), "H1")
    )
    assert deuterium is not None
    assert deuterium.element_identity.isotope_mass_number == 2


def test_prepared_projector_reobserves_failure_after_coordinate_change() -> None:
    structure = _cleanup_repairable_structure(with_extra_carbon=True)
    projector = prepare_rdkit_no_conect_pdb_block_projector(structure)
    assert projector is not None
    assert (
        probe_rdkit_no_conect_parser_readability(
            structure, pdb_block_projector=projector
        ).sanitize_readable
        is True
    )

    residue_id = ResidueId("A", 1)
    geometry = structure.residue_geometry(
        structure.constitution.residue_index(residue_id)
    )
    moved_geometry = geometry.with_atom_geometry(
        "C3", geometry.atom_geometry("C3").with_position(Vec3(0.0, 1.3, 0.0))
    )
    moved = structure.with_updated_residue_geometries(((residue_id, moved_geometry),))
    prepared = probe_rdkit_no_conect_parser_readability(
        moved, pdb_block_projector=projector
    )
    fresh = probe_rdkit_no_conect_parser_readability(moved)

    assert measure_rdkit_no_conect_sanitize_readability(moved) is False
    assert prepared == fresh
    assert prepared.sanitize_readable is False
    assert prepared.parser_problem_witnesses()[0].atom_names == ("N1",)
    assert prepared.issues()


@pytest.mark.parametrize(
    ("hydrogen", "invalid_carbon", "expected"),
    (
        ("H", False, ParserCompatibilityState.COMPATIBLE),
        ("H", True, ParserCompatibilityState.INCOMPATIBLE),
        (None, False, ParserCompatibilityState.NOT_OBSERVED),
    ),
)
def test_parser_state_uses_the_native_sanitation_observation(
    hydrogen: str | None, invalid_carbon: bool, expected: ParserCompatibilityState
) -> None:
    structure = _cleanup_repairable_structure(
        hydrogen_element=hydrogen, with_invalid_carbon=invalid_carbon
    )

    facts = StructureParserCompatibilityFacts.from_structure(structure)

    assert facts.carrier is structure
    assert facts.compatibility_state is expected

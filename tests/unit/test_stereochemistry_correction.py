"""Unit tests for localized side-chain stereochemistry correction."""

from collections.abc import Callable
from pathlib import Path

from tests.support.canonical_builders import (
    CanonicalResiduePayload,
    chain_payload,
)
from tests.support.canonical_builders import (
    build_structure as build_canonical_structure,
)

from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.diagnostics import (
    RepairEventKind,
    detect_sidechain_stereochemistry,
)
from protrepair.io import read_structure
from protrepair.structure import ProteinStructure
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.stereochemistry import (
    correct_sidechain_stereochemistry,
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


def test_correct_sidechain_stereochemistry_is_noop_for_native_residues() -> None:
    """Native supported residues should pass through without corrections."""

    structure = focused_structure_for_residue(seq_num=30)

    result = correct_sidechain_stereochemistry(structure)

    assert result == TransformationResult(
        structure=structure,
        repairs=(),
        issues=(),
    )


def focused_structure_for_residue(
    *,
    seq_num: int,
    mutate_residue: Callable[[CanonicalResiduePayload], CanonicalResiduePayload]
    | None = None,
) -> ProteinStructure:
    """Return a one-residue canonical structure from the representative fixture."""

    structure = read_structure(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
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
    residue = (
        residue_site,
        residue_geometry,
        structure.topology.residue_formal_charge_by_atom_name(
            constitution=structure.constitution,
            residue_index=structure.constitution.residue_index(residue_id),
        ),
    )
    if mutate_residue is not None:
        residue = mutate_residue(residue)

    return build_canonical_structure(
        chains=(chain_payload("A", (residue,)),),
        source_format=FileFormat.PDB,
        source_name=f"stereochemistry-{seq_num}",
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

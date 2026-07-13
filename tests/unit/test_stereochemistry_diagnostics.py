"""Unit tests for explicit side-chain stereochemistry diagnostics."""

from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    build_structure,
    chain_payload,
)

from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.diagnostics import ValidationIssueKind, detect_sidechain_stereochemistry
from protrepair.io import read_structure
from protrepair.structure import ProteinStructure, ResidueSite, StructureConstitution
from protrepair.structure.labels import ResidueId
from protrepair.workflow.contracts import StructureIngressOptions


def test_detect_sidechain_stereochemistry_accepts_supported_native_orientation() -> (
    None
):
    """Canonical supported residues should satisfy encoded chirality expectations."""

    structure = read_structure(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    focused_structure = focused_chain_structure(
        structure,
        chain_id="A",
        residue_ids={ResidueId("A", 25), ResidueId("A", 30)},
        source_name="native-chirality",
    )

    report = detect_sidechain_stereochemistry(
        focused_structure,
        component_library=build_standard_component_library(),
    )

    assert report.is_empty()


def test_detect_sidechain_stereochemistry_flags_inverted_threonine_center() -> None:
    """Swapping THR side-chain substituents should trigger a stereochemistry issue."""

    structure = read_structure(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    residue_id = ResidueId("A", 30)
    residue_site = structure.constitution.chain("A").residue(residue_id)
    residue_geometry = structure.geometry.residue_geometry(
        constitution=structure.constitution,
        residue_index=structure.constitution.residue_index(residue_id),
    )
    structure = structure.with_updated_residue_facets(
        residue_site,
        residue_geometry=residue_geometry.with_atom_geometries(
            (
                ("OG1", residue_geometry.atom_geometry("CG2")),
                ("CG2", residue_geometry.atom_geometry("OG1")),
            )
        ),
        formal_charge_by_atom_name=structure.topology.residue_formal_charge_by_atom_name(
            constitution=structure.constitution,
            residue_index=structure.constitution.residue_index(residue_id),
        ),
    )
    structure = focused_chain_structure(
        structure,
        chain_id="A",
        residue_ids={residue_id},
        source_name="inverted-thr",
    )

    report = detect_sidechain_stereochemistry(
        structure,
        component_library=build_standard_component_library(),
    )

    assert len(report.violations) == 1
    violation = report.violations[0]
    assert violation.component_id == "THR"
    assert violation.center_atom_name == "CB"
    assert violation.observed_signed_volume < 0
    assert report.to_issues()[0].kind is ValidationIssueKind.INVALID_STEREOCHEMISTRY


def test_detect_sidechain_stereochemistry_skips_missing_center_atoms() -> None:
    """Incomplete residues should not emit stereochemistry warnings."""

    structure = read_structure(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    residue_id = ResidueId("A", 25)
    residue_site = structure.constitution.chain("A").residue(residue_id)
    residue_geometry = structure.geometry.residue_geometry(
        constitution=structure.constitution,
        residue_index=structure.constitution.residue_index(residue_id),
    )
    structure = structure.with_updated_residue_facets(
        residue_site.without_atom_sites({"CG2"}),
        residue_geometry=residue_geometry.without_atoms({"CG2"}),
        formal_charge_by_atom_name=tuple(
            (
                atom_name,
                formal_charge,
            )
            for atom_name, formal_charge in (
                structure.topology.residue_formal_charge_by_atom_name(
                    constitution=structure.constitution,
                    residue_index=structure.constitution.residue_index(residue_id),
                )
            )
            if atom_name != "CG2"
        ),
    )
    focused_structure = focused_chain_structure(
        structure,
        chain_id="A",
        residue_ids={residue_id},
        source_name="missing-ile-atom",
    )

    report = detect_sidechain_stereochemistry(
        focused_structure,
        component_library=build_standard_component_library(),
    )

    assert report.is_empty()


def test_focused_stereochemistry_uses_direct_residue_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Focused diagnostics must not traverse every polymer residue."""

    structure = read_structure(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    residue_id = ResidueId("A", 30)
    structure = _with_swapped_residue_atom_geometries(
        structure,
        residue_id=residue_id,
        first_atom_name="OG1",
        second_atom_name="CG2",
    )
    monkeypatch.setattr(
        StructureConstitution,
        "iter_residues",
        lambda *_args, **_kwargs: pytest.fail(
            "focused stereochemistry must not scan all polymer residues"
        ),
    )

    empty_report = detect_sidechain_stereochemistry(
        structure,
        component_library=build_standard_component_library(),
        residue_ids=(),
    )
    report = detect_sidechain_stereochemistry(
        structure,
        component_library=build_standard_component_library(),
        residue_ids=frozenset({residue_id, ResidueId("missing", 999)}),
    )

    assert tuple(violation.residue_id for violation in report.violations) == (
        residue_id,
    )
    assert empty_report.is_empty()


def test_focused_stereochemistry_preserves_polymer_order_and_scope() -> None:
    """Focused reports follow canonical polymer order despite misleading flags."""

    structure = read_structure(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    structure = structure.select_chains(("B", "A"))
    first_residue_id = ResidueId("B", 30)
    second_residue_id = ResidueId("A", 30)
    for residue_id in (first_residue_id, second_residue_id):
        structure = _with_swapped_residue_atom_geometries(
            structure,
            residue_id=residue_id,
            first_atom_name="OG1",
            second_atom_name="CG2",
        )

    source_residue = structure.constitution.chain("B").residue(first_residue_id)
    source_geometry = structure.residue_geometry(
        structure.constitution.residue_index(first_residue_id)
    )
    ligand_residue_id = ResidueId("L", 1)
    structure = structure.with_ligand_facets(
        ligand_sites=(
            ResidueSite(
                component_id=source_residue.component_id,
                residue_id=ligand_residue_id,
                atom_sites=source_residue.atom_sites,
                is_hetero=False,
            ),
        ),
        ligand_geometries=(source_geometry,),
    )
    structure = structure.with_updated_residue_facets(
        ResidueSite(
            component_id=source_residue.component_id,
            residue_id=source_residue.residue_id,
            atom_sites=source_residue.atom_sites,
            is_hetero=True,
        ),
        residue_geometry=source_geometry,
        formal_charge_by_atom_name=(
            structure.topology.residue_formal_charge_by_atom_name(
                constitution=structure.constitution,
                residue_index=structure.constitution.residue_index(first_residue_id),
            )
        ),
    )

    report = detect_sidechain_stereochemistry(
        structure,
        component_library=build_standard_component_library(),
        residue_ids=(
            second_residue_id,
            ligand_residue_id,
            first_residue_id,
            second_residue_id,
            ResidueId("missing", 999),
        ),
    )

    assert tuple(violation.residue_id for violation in report.violations) == (
        first_residue_id,
        second_residue_id,
    )


def focused_chain_structure(
    structure: ProteinStructure,
    *,
    chain_id: str,
    residue_ids: set[ResidueId],
    source_name: str,
) -> ProteinStructure:
    """Return one focused structure over selected residues in one chain."""

    residue_payloads = []
    for residue_site in structure.constitution.chain(chain_id).residues:
        if residue_site.residue_id not in residue_ids:
            continue

        residue_geometry = structure.geometry.residue_geometry(
            constitution=structure.constitution,
            residue_index=structure.constitution.residue_index(residue_site.residue_id),
        )
        residue_payloads.append(
            (
                residue_site,
                residue_geometry,
                structure.topology.residue_formal_charge_by_atom_name(
                    constitution=structure.constitution,
                    residue_index=structure.constitution.residue_index(
                        residue_site.residue_id
                    ),
                ),
            )
        )

    return build_structure(
        chains=(chain_payload(chain_id, tuple(residue_payloads)),),
        ligands=(),
        source_format=structure.provenance.ingress.source_format,
        source_name=source_name,
    )


def _with_swapped_residue_atom_geometries(
    structure: ProteinStructure,
    *,
    residue_id: ResidueId,
    first_atom_name: str,
    second_atom_name: str,
) -> ProteinStructure:
    """Return a structure with two residue-local atom geometries swapped."""

    residue_site = structure.constitution.residue_or_ligand(residue_id)
    assert residue_site is not None
    residue_index = structure.constitution.residue_index(residue_id)
    residue_geometry = structure.residue_geometry(residue_index)
    return structure.with_updated_residue_facets(
        residue_site,
        residue_geometry=residue_geometry.with_atom_geometries(
            (
                (
                    first_atom_name,
                    residue_geometry.atom_geometry(second_atom_name),
                ),
                (
                    second_atom_name,
                    residue_geometry.atom_geometry(first_atom_name),
                ),
            )
        ),
        formal_charge_by_atom_name=(
            structure.topology.residue_formal_charge_by_atom_name(
                constitution=structure.constitution,
                residue_index=residue_index,
            )
        ),
    )

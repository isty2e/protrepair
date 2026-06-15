"""Unit tests for continuous local bond-planning support resolution."""

import pytest
from tests.support.canonical_builders import (
    CanonicalAtomPayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import build_default_component_library
from protrepair.geometry import Vec3
from protrepair.io import FileFormat
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.continuous import (
    LocalBondPlanningSupportMode,
    LocalBondPlanningSupportResolution,
    resolve_local_bond_planning_support,
)

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - optional dependency
    Chem = None

RDKIT_AVAILABLE = Chem is not None


def test_resolve_local_bond_planning_support_prefers_component_templates() -> None:
    """Known template residues should resolve to template-backed support."""

    snapshot = ProteinStructureSnapshot.from_structure(
        build_ser_with_template_less_ligand_structure(
            ligand_atoms=(atom_payload("ZN1", "Zn", Vec3(1.7, 1.4, 0.0)),),
            source_name="template-backed-ser-support",
        )
    )
    residue_id = ResidueId("A", 1)

    resolution = support_resolution_for_residue(
        snapshot,
        residue_id=residue_id,
    )

    assert resolution.mode is LocalBondPlanningSupportMode.TEMPLATE
    assert resolution.fallback_bond_definitions == ()


def test_resolve_local_bond_planning_support_allows_single_center_passive_context() -> (
    None
):
    """Template-less passive single-center context needs no bonded fallback."""

    snapshot = ProteinStructureSnapshot.from_structure(
        build_ser_with_template_less_ligand_structure(
            ligand_atoms=(atom_payload("ZN1", "Zn", Vec3(1.7, 1.4, 0.0)),),
            source_name="single-center-passive-context-support",
        )
    )
    residue_id = ResidueId("L", 1)

    resolution = support_resolution_for_residue(
        snapshot,
        residue_id=residue_id,
    )

    assert resolution.mode is LocalBondPlanningSupportMode.NONBONDING_PASSIVE_CONTEXT
    assert resolution.fallback_bond_definitions == ()


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="requires RDKit fallback chemistry")
def test_resolve_local_bond_planning_support_returns_passive_fallback_bonds() -> None:
    """Connected passive retained residues should expose fallback heavy bonds."""

    snapshot = ProteinStructureSnapshot.from_structure(
        build_ser_with_template_less_ligand_structure(
            ligand_atoms=(
                atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
                atom_payload("O1", "O", Vec3(2.7, 1.4, 0.0)),
                atom_payload("H1", "H", Vec3(1.5, 0.7, 0.0)),
                atom_payload("H2", "H", Vec3(1.7, -0.7, 0.0)),
                atom_payload("H3", "H", Vec3(1.9, 0.7, 0.0)),
                atom_payload("H4", "H", Vec3(2.1, -0.7, 0.0)),
            ),
            source_name="connected-passive-context-support",
        )
    )
    residue_id = ResidueId("L", 1)

    resolution = support_resolution_for_residue(
        snapshot,
        residue_id=residue_id,
    )

    assert resolution.mode is LocalBondPlanningSupportMode.TEMPLATE_LESS_PASSIVE_CONTEXT
    assert tuple(
        sorted(
            tuple(sorted((bond.atom_name_1, bond.atom_name_2)))
            for bond in resolution.fallback_bond_definitions
        )
    ) == (("C1", "O1"),)


def test_resolve_local_bond_planning_support_blocks_selected_template_less_ligand() -> (
    None
):
    """Template-less fallback support must not make selected ligands editable."""

    snapshot = ProteinStructureSnapshot.from_structure(
        build_ser_with_template_less_ligand_structure(
            ligand_atoms=(
                atom_payload("C1", "C", Vec3(1.7, 1.4, 0.0)),
                atom_payload("O1", "O", Vec3(2.7, 1.4, 0.0)),
                atom_payload("H1", "H", Vec3(1.5, 0.7, 0.0)),
                atom_payload("H2", "H", Vec3(1.7, -0.7, 0.0)),
                atom_payload("H3", "H", Vec3(1.9, 0.7, 0.0)),
                atom_payload("H4", "H", Vec3(2.1, -0.7, 0.0)),
            ),
            source_name="selected-template-less-support-blocker",
        )
    )
    residue_id = ResidueId("L", 1)

    resolution = support_resolution_for_residue(
        snapshot,
        residue_id=residue_id,
        movable_atom_names=("C1",),
    )

    assert resolution.mode is LocalBondPlanningSupportMode.UNSUPPORTED
    assert resolution.fallback_bond_definitions == ()


def support_resolution_for_residue(
    snapshot: ProteinStructureSnapshot,
    *,
    residue_id: ResidueId,
    movable_atom_names: tuple[str, ...] = (),
) -> LocalBondPlanningSupportResolution:
    """Resolve local bond-planning support for one residue in a test snapshot."""

    constitution = snapshot.structure.constitution
    residue_index = constitution.residue_index(residue_id)
    residue_site = constitution.residue_site_at(residue_index)
    movable_atom_indices = tuple(
        constitution.atom_index(AtomRef(residue_id, atom_name))
        for atom_name in movable_atom_names
    )
    return resolve_local_bond_planning_support(
        snapshot,
        residue_index,
        residue_site,
        movable_atom_indices=movable_atom_indices,
        component_library=build_default_component_library(),
    )


def build_ser_with_template_less_ligand_structure(
    *,
    ligand_atoms: tuple[CanonicalAtomPayload, ...],
    source_name: str,
) -> ProteinStructure:
    """Build one SER fixture with one template-less retained non-polymer ligand."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="SER",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.4, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.8, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.8, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.4, 1.4, 0.0)),
                            atom_payload("OG", "O", Vec3(1.4, 2.6, 0.0)),
                            atom_payload("H1", "H", Vec3(-0.7, 0.0, 0.0)),
                            atom_payload("H2", "H", Vec3(0.0, 0.7, 0.0)),
                            atom_payload("H3", "H", Vec3(0.0, -0.7, 0.0)),
                            atom_payload("HA", "H", Vec3(1.4, -0.9, 0.0)),
                            atom_payload("HB1", "H", Vec3(0.8, 1.9, 0.8)),
                            atom_payload("HB2", "H", Vec3(2.0, 1.9, -0.8)),
                            atom_payload("HG", "H", Vec3(1.4, 3.3, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("L", 1),
                atoms=ligand_atoms,
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name=source_name,
    )

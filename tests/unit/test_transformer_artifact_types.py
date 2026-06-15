"""Tests for transformation artifact and delta contracts."""

import pytest
from tests.support.refinement_type_fixtures import build_toy_structure

from protrepair.structure import AtomSite
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex, ResidueIndex
from protrepair.transformer.artifacts import (
    AtomElementRewrite,
    AtomFormalChargeRewrite,
    BondTopologyRewrite,
    GraphBondState,
    MovedAtomDelta,
    RegionTransformationResult,
    ResidueIdentityRewrite,
    ResidueTopologyRewrite,
    StructureDelta,
)


def test_refinement_result_normalizes_backend_name_and_moved_atom_deltas() -> None:
    """Results should normalize moved atom deltas and backend metadata."""

    structure = build_toy_structure()
    atom_index = structure.constitution.atom_index(
        AtomRef(
            residue_id=ResidueId(chain_id="A", seq_num=1),
            atom_name="H1",
        )
    )
    result = RegionTransformationResult(
        refined_structure=structure,
        delta=StructureDelta(
            before_constitution=structure.constitution,
            after_constitution=structure.constitution,
            moved_atoms=(
                MovedAtomDelta(
                    before_atom_index=atom_index,
                    after_atom_index=atom_index,
                ),
                MovedAtomDelta(
                    before_atom_index=atom_index,
                    after_atom_index=atom_index,
                ),
            ),
        ),
        issues=(),
        backend_name=" RDKIT ",
        backend_version=" 2025.9.6 ",
    )

    assert result.backend_name == "rdkit"
    assert result.backend_version == "2025.9.6"
    assert result.delta.moved_atoms == (
        MovedAtomDelta(
            before_atom_index=atom_index,
            after_atom_index=atom_index,
        ),
    )
    assert result.moved_atom_count() == 1


def test_structure_delta_tracks_inventory_changes() -> None:
    """Structure deltas should distinguish moved vs created/deleted inventory."""

    structure = build_toy_structure()
    delta = StructureDelta(
        before_constitution=structure.constitution,
        after_constitution=structure.constitution,
        moved_atoms=(
            MovedAtomDelta(
                before_atom_index=AtomIndex(0),
                after_atom_index=AtomIndex(0),
            ),
        ),
        created_atom_indices=(AtomIndex(1),),
        deleted_atom_indices=(AtomIndex(2),),
        created_residue_indices=(ResidueIndex(0),),
        deleted_residue_indices=(ResidueIndex(1),),
    )

    assert delta.moved_atoms == (
        MovedAtomDelta(
            before_atom_index=AtomIndex(0),
            after_atom_index=AtomIndex(0),
        ),
    )
    assert delta.created_atom_indices == (AtomIndex(1),)
    assert delta.deleted_atom_indices == (AtomIndex(2),)
    assert delta.created_residue_indices == (ResidueIndex(0),)
    assert delta.deleted_residue_indices == (ResidueIndex(1),)


def test_structure_delta_tracks_identity_and_topology_rewrites() -> None:
    """Structure deltas should model rewrite-only residue deltas explicitly."""

    structure = build_toy_structure()
    residue_index = structure.constitution.residue_index(
        ResidueId(chain_id="A", seq_num=1)
    )
    atom_index = structure.constitution.atom_index_in_residue(residue_index, "C1")
    patch = StructureDelta(
        before_constitution=structure.constitution,
        after_constitution=structure.constitution,
        residue_identity_rewrites=(
            ResidueIdentityRewrite(
                after_residue_index=residue_index,
                previous_component_id=" ser ",
                current_component_id="sep",
            ),
        ),
        residue_topology_rewrites=(
            ResidueTopologyRewrite(
                after_residue_index=residue_index,
                affected_atom_indices=(atom_index, atom_index),
            ),
        ),
    )

    assert patch.residue_identity_rewrites == (
        ResidueIdentityRewrite(
            after_residue_index=residue_index,
            previous_component_id="SER",
            current_component_id="SEP",
        ),
    )
    assert patch.residue_topology_rewrites == (
        ResidueTopologyRewrite(
            after_residue_index=residue_index,
            affected_atom_indices=(atom_index,),
        ),
    )


def test_structure_delta_tracks_atom_and_bond_rewrites() -> None:
    """Structure deltas should model constitution and topology rewrites separately."""

    structure = build_toy_structure()
    left_atom_index = structure.constitution.atom_index(
        AtomRef(ResidueId(chain_id="A", seq_num=1), "c1")
    )
    right_atom_index = structure.constitution.atom_index(
        AtomRef(ResidueId(chain_id="L", seq_num=1), "o1")
    )
    patch = StructureDelta(
        before_constitution=structure.constitution,
        after_constitution=structure.constitution,
        atom_element_rewrites=(
            AtomElementRewrite(
                after_atom_index=left_atom_index,
                previous_element=" c ",
                current_element=" n ",
            ),
        ),
        atom_formal_charge_rewrites=(
            AtomFormalChargeRewrite(
                after_atom_index=left_atom_index,
                previous_formal_charge=None,
                current_formal_charge=1,
            ),
        ),
        bond_topology_rewrites=(
            BondTopologyRewrite(
                left_after_atom_index=right_atom_index,
                right_after_atom_index=left_atom_index,
                previous_bond=None,
                current_bond=GraphBondState(order=1),
            ),
        ),
    )

    assert patch.atom_element_rewrites == (
        AtomElementRewrite(
            after_atom_index=left_atom_index,
            previous_element="C",
            current_element="N",
        ),
    )
    assert patch.atom_formal_charge_rewrites == (
        AtomFormalChargeRewrite(
            after_atom_index=left_atom_index,
            previous_formal_charge=None,
            current_formal_charge=1,
        ),
    )
    assert patch.bond_topology_rewrites == (
        BondTopologyRewrite(
            left_after_atom_index=left_atom_index,
            right_after_atom_index=right_atom_index,
            previous_bond=None,
            current_bond=GraphBondState(order=1),
        ),
    )


def test_structure_delta_rejects_inventory_and_rewrite_overlap() -> None:
    """Structure deltas should keep after-space creation disjoint from rewrites."""

    structure = build_toy_structure()
    residue_index = structure.constitution.residue_index(
        ResidueId(chain_id="A", seq_num=1)
    )
    atom_index = structure.constitution.atom_index_in_residue(residue_index, "C1")

    with pytest.raises(ValueError, match="created and identity-rewritten"):
        StructureDelta(
            before_constitution=structure.constitution,
            after_constitution=structure.constitution,
            created_residue_indices=(residue_index,),
            residue_identity_rewrites=(
                ResidueIdentityRewrite(
                    after_residue_index=residue_index,
                    previous_component_id="SER",
                    current_component_id="SEP",
                ),
            ),
        )

    with pytest.raises(ValueError, match="created and topology-rewritten"):
        StructureDelta(
            before_constitution=structure.constitution,
            after_constitution=structure.constitution,
            created_residue_indices=(residue_index,),
            residue_topology_rewrites=(
                ResidueTopologyRewrite(after_residue_index=residue_index),
            ),
        )

    with pytest.raises(ValueError, match="created and rewritten"):
        StructureDelta(
            before_constitution=structure.constitution,
            after_constitution=structure.constitution,
            created_atom_indices=(atom_index,),
            atom_formal_charge_rewrites=(
                AtomFormalChargeRewrite(
                    after_atom_index=atom_index,
                    previous_formal_charge=None,
                    current_formal_charge=1,
                ),
            ),
        )


def test_delta_rewrite_nouns_reject_noop_payloads() -> None:
    """Delta rewrite nouns should reject no-op constitution and topology changes."""

    with pytest.raises(ValueError, match="must change the element"):
        AtomElementRewrite(
            after_atom_index=AtomIndex(0),
            previous_element="C",
            current_element="C",
        )

    with pytest.raises(ValueError, match="must change the formal charge"):
        AtomFormalChargeRewrite(
            after_atom_index=AtomIndex(0),
            previous_formal_charge=1,
            current_formal_charge=1,
        )

    with pytest.raises(ValueError, match="must describe a previous or current bond"):
        BondTopologyRewrite(
            left_after_atom_index=AtomIndex(0),
            right_after_atom_index=AtomIndex(1),
        )

    with pytest.raises(ValueError, match="must change bond semantics"):
        BondTopologyRewrite(
            left_after_atom_index=AtomIndex(0),
            right_after_atom_index=AtomIndex(1),
            previous_bond=GraphBondState(order=1),
            current_bond=GraphBondState(order=1),
        )

    with pytest.raises(ValueError, match="two distinct atom slots"):
        BondTopologyRewrite(
            left_after_atom_index=AtomIndex(0),
            right_after_atom_index=AtomIndex(0),
            current_bond=GraphBondState(order=1),
        )


def test_refinement_result_validates_identity_and_topology_rewrites() -> None:
    """Refinement results should validate rewrite targets against the result."""

    structure = build_toy_structure()
    residue_index = structure.constitution.residue_index(
        ResidueId(chain_id="A", seq_num=1)
    )
    atom_index = structure.constitution.atom_index_in_residue(residue_index, "C1")
    result = RegionTransformationResult(
        refined_structure=structure,
        delta=StructureDelta(
            before_constitution=structure.constitution,
            after_constitution=structure.constitution,
            residue_identity_rewrites=(
                ResidueIdentityRewrite(
                    after_residue_index=residue_index,
                    previous_component_id="OLD",
                    current_component_id="MOV",
                ),
            ),
            residue_topology_rewrites=(
                ResidueTopologyRewrite(
                    after_residue_index=residue_index,
                    affected_atom_indices=(atom_index,),
                ),
            ),
        ),
        issues=(),
        backend_name="rdkit",
    )

    assert result.delta.residue_identity_rewrites[0].current_component_id == "MOV"
    assert result.delta.residue_topology_rewrites[0].affected_atom_indices == (
        atom_index,
    )

    with pytest.raises(ValueError, match="outside the constitution residue slots"):
        StructureDelta(
            before_constitution=structure.constitution,
            after_constitution=structure.constitution,
            residue_identity_rewrites=(
                ResidueIdentityRewrite(
                    after_residue_index=ResidueIndex(999),
                    previous_component_id="OLD",
                    current_component_id="MOV",
                ),
            ),
        )

    with pytest.raises(ValueError, match="expects component SEP"):
        RegionTransformationResult(
            refined_structure=structure,
            delta=StructureDelta(
                before_constitution=structure.constitution,
                after_constitution=structure.constitution,
                residue_identity_rewrites=(
                    ResidueIdentityRewrite(
                        after_residue_index=residue_index,
                        previous_component_id="OLD",
                        current_component_id="SEP",
                    ),
                ),
            ),
            issues=(),
            backend_name="rdkit",
        )

    with pytest.raises(
        ValueError,
        match="must only reference atom slots inside the rewritten residue",
    ):
        StructureDelta(
            before_constitution=structure.constitution,
            after_constitution=structure.constitution,
            residue_topology_rewrites=(
                ResidueTopologyRewrite(
                    after_residue_index=residue_index,
                    affected_atom_indices=(
                        structure.constitution.atom_index(
                            AtomRef(ResidueId(chain_id="L", seq_num=1), "O1")
                        ),
                    ),
                ),
            ),
        )


def test_refinement_result_validates_atom_and_bond_rewrites() -> None:
    """Refinement results should validate rewrite targets and current semantics."""

    structure = build_toy_structure()
    residue_id = ResidueId(chain_id="A", seq_num=1)
    residue_index = structure.constitution.residue_index(residue_id)
    residue_site = structure.constitution.residue_or_ligand(residue_id)
    assert residue_site is not None
    residue_geometry = structure.geometry.residue_geometry(
        constitution=structure.constitution,
        residue_index=residue_index,
    )
    refined_structure = structure.with_updated_residue_facets(
        residue_site=residue_site.with_atom_site(AtomSite(name="C1", element="N")),
        residue_geometry=residue_geometry.with_atom_geometry(
            "C1",
            residue_geometry.atom_geometry("C1"),
        ),
        formal_charge_by_atom_name=(("C1", 1),),
    )
    atom_index = refined_structure.constitution.atom_index(
        AtomRef(ResidueId(chain_id="A", seq_num=1), "C1")
    )
    link_atom_index = refined_structure.constitution.atom_index(
        AtomRef(ResidueId(chain_id="L", seq_num=1), "O1")
    )

    result = RegionTransformationResult(
        refined_structure=refined_structure,
        delta=StructureDelta(
            before_constitution=structure.constitution,
            after_constitution=refined_structure.constitution,
            atom_element_rewrites=(
                AtomElementRewrite(
                    after_atom_index=atom_index,
                    previous_element="C",
                    current_element="N",
                ),
            ),
            atom_formal_charge_rewrites=(
                AtomFormalChargeRewrite(
                    after_atom_index=atom_index,
                    previous_formal_charge=None,
                    current_formal_charge=1,
                ),
            ),
            bond_topology_rewrites=(
                BondTopologyRewrite(
                    left_after_atom_index=atom_index,
                    right_after_atom_index=link_atom_index,
                    previous_bond=None,
                    current_bond=GraphBondState(order=1),
                ),
            ),
        ),
        issues=(),
        backend_name="rdkit",
    )

    assert result.delta.atom_element_rewrites[0].current_element == "N"
    assert result.delta.atom_formal_charge_rewrites[0].current_formal_charge == 1
    assert result.delta.bond_topology_rewrites[0] == BondTopologyRewrite(
        left_after_atom_index=atom_index,
        right_after_atom_index=link_atom_index,
        previous_bond=None,
        current_bond=GraphBondState(order=1),
    )

    with pytest.raises(ValueError, match="expects element O"):
        RegionTransformationResult(
            refined_structure=refined_structure,
            delta=StructureDelta(
                before_constitution=structure.constitution,
                after_constitution=refined_structure.constitution,
                atom_element_rewrites=(
                    AtomElementRewrite(
                        after_atom_index=atom_index,
                        previous_element="C",
                        current_element="O",
                    ),
                ),
            ),
            issues=(),
            backend_name="rdkit",
        )

    with pytest.raises(ValueError, match="expects charge 2"):
        RegionTransformationResult(
            refined_structure=refined_structure,
            delta=StructureDelta(
                before_constitution=structure.constitution,
                after_constitution=refined_structure.constitution,
                atom_formal_charge_rewrites=(
                    AtomFormalChargeRewrite(
                        after_atom_index=atom_index,
                        previous_formal_charge=None,
                        current_formal_charge=2,
                    ),
                ),
            ),
            issues=(),
            backend_name="rdkit",
        )

    with pytest.raises(ValueError, match="outside the constitution atom slots"):
        StructureDelta(
            before_constitution=structure.constitution,
            after_constitution=refined_structure.constitution,
            bond_topology_rewrites=(
                BondTopologyRewrite(
                    left_after_atom_index=atom_index,
                    right_after_atom_index=AtomIndex(999),
                    previous_bond=None,
                    current_bond=GraphBondState(order=1),
                ),
            ),
        )

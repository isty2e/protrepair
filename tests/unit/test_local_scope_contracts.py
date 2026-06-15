"""Tests for local scope contracts and snapshot-bound normalization."""

from typing import cast

import pytest
from tests.support.refinement_benchmarks import load_case_structure
from tests.support.refinement_cases import REFINEMENT_BENCHMARK_CASES
from tests.support.refinement_type_fixtures import (
    build_toy_component_library,
    build_toy_structure,
)

from protrepair.scope import (
    AbsentResidueSpanScope,
    AnchorAtomPairScope,
    AtomSetScope,
    CompositeScope,
)
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.local import (
    LocalScopeSpec,
    atom_input_from_local_scope_spec,
)


def test_refinement_selection_normalizes_one_boundary_mode() -> None:
    """Local scope specs should deduplicate their active residue or atom focus."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    scope_spec = LocalScopeSpec.from_residues((residue_id, residue_id))
    atom_scope_spec = LocalScopeSpec.from_atoms(
        (
            AtomRef(residue_id=residue_id, atom_name=" ca "),
            AtomRef(residue_id=residue_id, atom_name="CA"),
        )
    )
    attached_hydrogen_scope_spec = LocalScopeSpec.from_atoms_with_attached_hydrogens(
        (
            AtomRef(residue_id=residue_id, atom_name=" c1 "),
            AtomRef(residue_id=residue_id, atom_name="C1"),
        )
    )
    sidechain_scope_spec = LocalScopeSpec.from_residue_sidechains((residue_id,))

    assert scope_spec.is_residuewise()
    assert scope_spec.referenced_residue_ids() == (residue_id,)
    assert sidechain_scope_spec.is_residuewise()
    assert sidechain_scope_spec.lowers_to_residue_sidechains()
    assert atom_scope_spec.is_atomwise()
    assert atom_scope_spec.as_scope() == AtomSetScope(
        atom_refs=(AtomRef(residue_id=residue_id, atom_name="CA"),)
    )
    assert attached_hydrogen_scope_spec.is_atomwise()
    assert attached_hydrogen_scope_spec.as_scope() == AtomSetScope(
        atom_refs=(AtomRef(residue_id=residue_id, atom_name="C1"),)
    )

    with pytest.raises(ValueError, match="at least one residue"):
        LocalScopeSpec.from_residues(())

    with pytest.raises(ValueError, match="at least one atom"):
        LocalScopeSpec.from_atoms(())

    with pytest.raises(ValueError, match="at least one atom"):
        LocalScopeSpec.from_atoms_with_attached_hydrogens(())


def test_atomwise_selection_reports_first_seen_residue_order() -> None:
    """Atomwise local scope specs should preserve first-seen residue order."""

    scope_spec = LocalScopeSpec.from_atoms(
        (
            AtomRef(ResidueId(chain_id="L", seq_num=1), "O1"),
            AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),
            AtomRef(ResidueId(chain_id="L", seq_num=1), "C2"),
        )
    )

    assert scope_spec.referenced_residue_ids() == (
        ResidueId(chain_id="L", seq_num=1),
        ResidueId(chain_id="A", seq_num=1),
    )


def test_local_scope_ontology_represents_absent_and_relational_scopes() -> None:
    """Local scope ontology should express absent and relational scopes directly."""

    absent_scope = AbsentResidueSpanScope(
        preceding_residue_id=ResidueId(chain_id="A", seq_num=10),
        following_residue_id=ResidueId(chain_id="A", seq_num=13),
        absent_residue_ids=(
            ResidueId(chain_id="A", seq_num=11),
            ResidueId(chain_id="A", seq_num=12),
            ResidueId(chain_id="A", seq_num=11),
        ),
    )
    relational_scope = AnchorAtomPairScope(
        left_anchor_atom_ref=AtomRef(ResidueId(chain_id="A", seq_num=10), "C"),
        right_anchor_atom_ref=AtomRef(ResidueId(chain_id="A", seq_num=13), "N"),
    )

    assert absent_scope.anchor_residue_ids() == (
        ResidueId(chain_id="A", seq_num=10),
        ResidueId(chain_id="A", seq_num=13),
    )
    assert absent_scope.absent_residue_ids == (
        ResidueId(chain_id="A", seq_num=11),
        ResidueId(chain_id="A", seq_num=12),
    )
    assert relational_scope.anchor_atom_refs() == (
        AtomRef(ResidueId(chain_id="A", seq_num=10), "C"),
        AtomRef(ResidueId(chain_id="A", seq_num=13), "N"),
    )
    assert relational_scope.anchor_residue_ids() == (
        ResidueId(chain_id="A", seq_num=10),
        ResidueId(chain_id="A", seq_num=13),
    )

    with pytest.raises(ValueError, match="at least one anchor residue"):
        AbsentResidueSpanScope()

    with pytest.raises(ValueError, match="distinct anchor residues"):
        AbsentResidueSpanScope(
            preceding_residue_id=ResidueId(chain_id="A", seq_num=10),
            following_residue_id=ResidueId(chain_id="A", seq_num=10),
        )

    with pytest.raises(ValueError, match="must not list the preceding anchor"):
        AbsentResidueSpanScope(
            preceding_residue_id=ResidueId(chain_id="A", seq_num=10),
            absent_residue_ids=(ResidueId(chain_id="A", seq_num=10),),
        )

    with pytest.raises(ValueError, match="two distinct anchor atoms"):
        AnchorAtomPairScope(
            left_anchor_atom_ref=AtomRef(ResidueId(chain_id="A", seq_num=10), "C"),
            right_anchor_atom_ref=AtomRef(ResidueId(chain_id="A", seq_num=10), "C"),
        )


def test_composite_scope_preserves_multiple_primitive_roles() -> None:
    """Composite scopes should retain distinct primitive local scope roles."""

    residue_scope = LocalScopeSpec.from_residues(
        (ResidueId(chain_id="A", seq_num=10),)
    ).as_scope()
    absent_scope = AbsentResidueSpanScope(
        preceding_residue_id=ResidueId(chain_id="A", seq_num=10),
        following_residue_id=ResidueId(chain_id="A", seq_num=13),
        absent_residue_ids=(
            ResidueId(chain_id="A", seq_num=11),
            ResidueId(chain_id="A", seq_num=12),
        ),
    )
    anchor_scope = AnchorAtomPairScope(
        left_anchor_atom_ref=AtomRef(ResidueId(chain_id="A", seq_num=10), "C"),
        right_anchor_atom_ref=AtomRef(ResidueId(chain_id="A", seq_num=13), "N"),
    )
    composite_scope = CompositeScope(
        scopes=(residue_scope, absent_scope, anchor_scope, residue_scope)
    )

    assert composite_scope.scopes == (residue_scope, absent_scope, anchor_scope)

    with pytest.raises(ValueError, match="at least two distinct scopes"):
        CompositeScope(scopes=(residue_scope,))

    with pytest.raises(TypeError, match="Scope values"):
        CompositeScope(
            scopes=(
                residue_scope,
                cast(CompositeScope, "not-a-scope"),
            )
        )


def test_atom_domain_normalization_rejects_unknown_residue_and_atom_references() -> (
    None
):
    """Selection normalization should reject unknown refs at snapshot ingress."""

    structure = build_toy_structure()
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    residue_selection = LocalScopeSpec.from_residues(
        (ResidueId(chain_id="A", seq_num=9),)
    )
    atom_selection = LocalScopeSpec.from_atoms(
        (
            AtomRef(
                residue_id=ResidueId(chain_id="A", seq_num=1),
                atom_name="XX",
            ),
        )
    )

    with pytest.raises(ValueError, match="unknown residue"):
        atom_input_from_local_scope_spec(snapshot, residue_selection)

    with pytest.raises(ValueError, match="unknown atom"):
        atom_input_from_local_scope_spec(snapshot, atom_selection)


def test_attached_hydrogen_atom_selection_expands_present_attached_hydrogens() -> None:
    """Attached-H local scope specs should encode closure intent explicitly."""

    structure = build_toy_structure()
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    atom_input = atom_input_from_local_scope_spec(
        snapshot,
        LocalScopeSpec.from_atoms_with_attached_hydrogens(
            (AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),)
        ),
        component_library=build_toy_component_library(),
    )

    assert atom_input.as_scope() == AtomSetScope(
        atom_refs=(
            AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),
            AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),
        )
    )
    assert atom_input.atom_indices == tuple(
        structure.constitution.atom_index(atom_ref)
        for atom_ref in (
            AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),
            AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),
        )
    )


def test_residue_sidechain_selection_realizes_sidechain_atoms() -> None:
    """Residue-sidechain lowering should preserve semantic scope and narrow atoms."""

    case = REFINEMENT_BENCHMARK_CASES["1bkr-thr101"]
    structure = load_case_structure(case)
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    atom_input = atom_input_from_local_scope_spec(
        snapshot,
        LocalScopeSpec.from_residue_sidechains(case.focus_residue_ids),
    )

    assert atom_input.is_residuewise()
    assert atom_input.realizes_residue_sidechains()
    assert atom_input.as_scope() == LocalScopeSpec.from_residue_sidechains(
        case.focus_residue_ids
    ).as_scope()

    observed_atom_scope = atom_input.observed_atom_scope(snapshot)
    observed_atom_names = {
        atom_ref.atom_name
        for atom_ref in observed_atom_scope.atom_refs
        if atom_ref.residue_id == case.focus_residue_ids[0]
    }

    assert {"CB", "OG1", "CG2"}.issubset(observed_atom_names)
    assert {"N", "CA", "C", "O"}.isdisjoint(observed_atom_names)

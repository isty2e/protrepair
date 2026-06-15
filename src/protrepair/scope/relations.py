"""Semantic relations between protein scope values."""

from protrepair.scope.base import (
    AbsentResidueSpanScope,
    AnchorAtomPairScope,
    AtomSetScope,
    ChainSetScope,
    CompositeScope,
    ResidueBoundaryScope,
    ResidueSetScope,
    Scope,
    WholeStructureScope,
)


def scope_refines(finer: Scope, coarser: Scope) -> bool:
    """Return whether `finer` is fully contained in `coarser`."""

    if finer == coarser:
        return True

    if isinstance(coarser, WholeStructureScope):
        return True

    if isinstance(finer, CompositeScope):
        return all(scope_refines(child_scope, coarser) for child_scope in finer.scopes)

    if isinstance(coarser, CompositeScope):
        return any(scope_refines(finer, child_scope) for child_scope in coarser.scopes)

    if isinstance(coarser, ChainSetScope):
        allowed_chain_ids = set(coarser.chain_ids)
        if isinstance(finer, ChainSetScope):
            return set(finer.chain_ids).issubset(allowed_chain_ids)
        if isinstance(finer, ResidueSetScope):
            return all(
                residue_id.chain_id in allowed_chain_ids
                for residue_id in finer.residue_ids
            )
        if isinstance(finer, AtomSetScope):
            return all(
                atom_ref.residue_id.chain_id in allowed_chain_ids
                for atom_ref in finer.atom_refs
            )
        if isinstance(finer, ResidueBoundaryScope):
            return finer.residue_id.chain_id in allowed_chain_ids
        return False

    if isinstance(coarser, ResidueSetScope):
        allowed_residue_ids = set(coarser.residue_ids)
        if isinstance(finer, ResidueSetScope):
            return set(finer.residue_ids).issubset(allowed_residue_ids)
        if isinstance(finer, AtomSetScope):
            return all(
                atom_ref.residue_id in allowed_residue_ids
                for atom_ref in finer.atom_refs
            )
        if isinstance(finer, ResidueBoundaryScope):
            return finer.residue_id in allowed_residue_ids
        return False

    if isinstance(coarser, ResidueBoundaryScope) and isinstance(
        finer, ResidueBoundaryScope
    ):
        return finer == coarser

    if isinstance(coarser, AtomSetScope) and isinstance(finer, AtomSetScope):
        return set(finer.atom_refs).issubset(set(coarser.atom_refs))

    if isinstance(coarser, AbsentResidueSpanScope) and isinstance(
        finer, AbsentResidueSpanScope
    ):
        return (
            finer.preceding_residue_id == coarser.preceding_residue_id
            and finer.following_residue_id == coarser.following_residue_id
            and set(finer.absent_residue_ids) <= set(coarser.absent_residue_ids)
        )

    if isinstance(coarser, AnchorAtomPairScope) and isinstance(
        finer, AnchorAtomPairScope
    ):
        return finer == coarser

    return False


def scope_contains(coarser: Scope, finer: Scope) -> bool:
    """Return whether `coarser` contains `finer`."""

    return scope_refines(finer, coarser)


def scopes_overlap(left: Scope, right: Scope) -> bool:
    """Return whether two scopes overlap semantically."""

    if left == right:
        return True

    if isinstance(left, WholeStructureScope) or isinstance(right, WholeStructureScope):
        return True

    if isinstance(left, CompositeScope):
        return any(scopes_overlap(child_scope, right) for child_scope in left.scopes)

    if isinstance(right, CompositeScope):
        return any(scopes_overlap(left, child_scope) for child_scope in right.scopes)

    if isinstance(left, ChainSetScope) and isinstance(right, ChainSetScope):
        return bool(set(left.chain_ids) & set(right.chain_ids))

    if isinstance(left, ResidueSetScope) and isinstance(right, ResidueSetScope):
        return bool(set(left.residue_ids) & set(right.residue_ids))

    if isinstance(left, ResidueBoundaryScope) and isinstance(
        right, ResidueBoundaryScope
    ):
        return left == right

    if isinstance(left, AtomSetScope) and isinstance(right, AtomSetScope):
        return bool(set(left.atom_refs) & set(right.atom_refs))

    if isinstance(left, ChainSetScope) and isinstance(right, ResidueSetScope):
        return any(
            residue_id.chain_id in set(left.chain_ids)
            for residue_id in right.residue_ids
        )
    if isinstance(left, ResidueSetScope) and isinstance(right, ChainSetScope):
        return scopes_overlap(right, left)

    if isinstance(left, ChainSetScope) and isinstance(right, AtomSetScope):
        return any(
            atom_ref.residue_id.chain_id in set(left.chain_ids)
            for atom_ref in right.atom_refs
        )
    if isinstance(left, AtomSetScope) and isinstance(right, ChainSetScope):
        return scopes_overlap(right, left)

    if isinstance(left, ResidueSetScope) and isinstance(right, AtomSetScope):
        return any(
            atom_ref.residue_id in set(left.residue_ids)
            for atom_ref in right.atom_refs
        )
    if isinstance(left, AtomSetScope) and isinstance(right, ResidueSetScope):
        return scopes_overlap(right, left)

    if isinstance(left, ChainSetScope) and isinstance(right, ResidueBoundaryScope):
        return right.residue_id.chain_id in set(left.chain_ids)
    if isinstance(left, ResidueBoundaryScope) and isinstance(right, ChainSetScope):
        return scopes_overlap(right, left)

    if isinstance(left, ResidueSetScope) and isinstance(right, ResidueBoundaryScope):
        return right.residue_id in set(left.residue_ids)
    if isinstance(left, ResidueBoundaryScope) and isinstance(right, ResidueSetScope):
        return scopes_overlap(right, left)

    if isinstance(left, AbsentResidueSpanScope) and isinstance(
        right, AbsentResidueSpanScope
    ):
        return (
            left.preceding_residue_id == right.preceding_residue_id
            and left.following_residue_id == right.following_residue_id
            and bool(set(left.absent_residue_ids) & set(right.absent_residue_ids))
        )

    if isinstance(left, AnchorAtomPairScope) and isinstance(right, AnchorAtomPairScope):
        return left == right

    return False

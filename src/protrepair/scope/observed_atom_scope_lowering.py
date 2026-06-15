"""Observed atom-scope realization from semantic scope."""

from collections.abc import Callable
from dataclasses import dataclass

from protrepair.scope.base import (
    AtomSetScope,
    ChainSetScope,
    ResidueBoundaryScope,
    ResidueSetScope,
    Scope,
    WholeStructureScope,
)
from protrepair.scope.lowering import ScopeLowering
from protrepair.structure.labels import AtomRef
from protrepair.structure.snapshot import ProteinStructureSnapshot


@dataclass(frozen=True, slots=True)
class ObservedAtomScopeLoweringPolicy:
    """Policy for observed atom-scope realization.

    This policy is intentionally empty today. Shared observed atom realization
    should depend only on the current carrier and semantic scope. Template-aware
    closure belongs to consumer-specific lowering, not to the shared scope
    operator.
    """


class ObservedAtomScopeLowering(
    ScopeLowering[
        Scope,
        ProteinStructureSnapshot,
        ObservedAtomScopeLoweringPolicy,
        AtomSetScope,
    ]
):
    """Lower supported semantic scopes into one observed atom scope."""

    def __init__(
        self,
        *,
        default_policy: ObservedAtomScopeLoweringPolicy | None = None,
    ) -> None:
        self._default_policy = (
            ObservedAtomScopeLoweringPolicy()
            if default_policy is None
            else default_policy
        )

    def supports(self, scope: Scope) -> bool:
        """Return whether one semantic scope can lower into an atom scope."""

        return isinstance(
            scope,
            (
                WholeStructureScope,
                ChainSetScope,
                ResidueSetScope,
                ResidueBoundaryScope,
                AtomSetScope,
            ),
        )

    def lower(
        self,
        scope: Scope,
        *,
        carrier: ProteinStructureSnapshot,
        policy: ObservedAtomScopeLoweringPolicy | None = None,
    ) -> AtomSetScope:
        """Lower one supported semantic scope into one observed atom scope."""

        active_policy = self._default_policy if policy is None else policy
        del active_policy
        if isinstance(scope, WholeStructureScope):
            return _project_observed_atom_scope(
                carrier,
                predicate=lambda atom_ref: True,
            )

        if isinstance(scope, ChainSetScope):
            allowed_chain_ids = set(scope.chain_ids)
            return _project_observed_atom_scope(
                carrier,
                predicate=lambda atom_ref: (
                    atom_ref.residue_id.chain_id in allowed_chain_ids
                ),
            )

        if isinstance(scope, ResidueSetScope):
            residue_id_set = set(scope.residue_ids)
            for residue_id in residue_id_set:
                if carrier.structure.constitution.residue_or_ligand(residue_id) is None:
                    raise ValueError(
                        "residue-set scope references unknown residue "
                        f"{residue_id.display_token()}"
                    )

            return _project_observed_atom_scope(
                carrier,
                predicate=lambda atom_ref: atom_ref.residue_id in residue_id_set,
            )

        if isinstance(scope, ResidueBoundaryScope):
            residue = carrier.structure.constitution.residue_or_ligand(scope.residue_id)
            if residue is None:
                raise ValueError(
                    "residue-boundary scope references unknown residue "
                    f"{scope.residue_id.display_token()}"
                )

            return AtomSetScope(
                atom_refs=tuple(
                    atom_ref
                    for atom_ref in structure_ordered_atom_refs(carrier)
                    if atom_ref.residue_id == residue.residue_id
                )
            )

        if isinstance(scope, AtomSetScope):
            selected_atom_ref_set = _selected_atom_ref_set(
                carrier,
                scope.atom_refs,
            )
            return AtomSetScope(
                atom_refs=tuple(
                    atom_ref
                    for atom_ref in structure_ordered_atom_refs(carrier)
                    if atom_ref in selected_atom_ref_set
                )
            )

        raise TypeError(
            "semantic scope does not support observed atom-scope lowering: "
            f"{type(scope).__name__}"
        )


OBSERVED_ATOM_SCOPE_LOWERING = ObservedAtomScopeLowering()


def structure_ordered_atom_refs(
    snapshot: ProteinStructureSnapshot,
) -> tuple[AtomRef, ...]:
    """Return atom refs in canonical structure order including ligands."""

    return tuple(
        AtomRef(residue_id=residue.residue_id, atom_name=atom_site.name)
        for residue in snapshot.structure.constitution.iter_residues(
            include_ligands=True
        )
        for atom_site in residue.atom_sites
    )


def _project_observed_atom_scope(
    snapshot: ProteinStructureSnapshot,
    *,
    predicate: Callable[[AtomRef], bool],
) -> AtomSetScope:
    """Project one observed atom scope from a canonical atom predicate."""

    return AtomSetScope(
        atom_refs=tuple(
            atom_ref
            for atom_ref in structure_ordered_atom_refs(snapshot)
            if predicate(atom_ref)
        )
    )


def _selected_atom_ref_set(
    snapshot: ProteinStructureSnapshot,
    atom_refs: tuple[AtomRef, ...],
) -> set[AtomRef]:
    """Return one validated explicit atom-ref set."""

    selected_atom_ref_set = set(atom_refs)
    for atom_ref in selected_atom_ref_set:
        if snapshot.structure.constitution.resolve_atom_site(atom_ref) is None:
            raise ValueError(
                f"atom-set scope references unknown atom {atom_ref.display_token()}"
            )

    return selected_atom_ref_set

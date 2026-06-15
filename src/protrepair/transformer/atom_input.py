"""Canonical realized atom selections over one structure carrier."""

from dataclasses import dataclass
from enum import Enum

from protrepair.scope import AtomSetScope, ResidueSetScope, Scope
from protrepair.scope.observed_atom_scope_lowering import structure_ordered_atom_refs
from protrepair.structure.labels import ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot

__all__ = [
    "AtomInput",
    "AtomInputBasis",
    "AtomInputRealization",
    "structure_ordered_atom_refs",
]


class AtomInputBasis(str, Enum):
    """How one atom input was derived from boundary semantics."""

    ATOMWISE = "atomwise"
    RESIDUEWISE = "residuewise"


class AtomInputRealization(str, Enum):
    """How one semantic selection was realized into concrete movable atoms."""

    EXACT_ATOMS = "exact_atoms"
    RESIDUE_ATOMS = "residue_atoms"
    RESIDUE_BACKBONE_ATOMS = "residue_backbone_atoms"
    RESIDUE_SIDECHAIN_ATOMS = "residue_sidechain_atoms"


@dataclass(frozen=True, slots=True)
class AtomInput:
    """Realized atom input over one structure carrier."""

    atom_indices: tuple[AtomIndex, ...]
    basis: AtomInputBasis
    selected_scope: Scope
    realization: AtomInputRealization | None = None

    def __post_init__(self) -> None:
        atom_indices = _normalize_atom_index_tuple(self.atom_indices)
        if not atom_indices:
            raise ValueError("atom inputs must contain at least one atom index")

        selected_scope = self.selected_scope
        if self.basis is AtomInputBasis.ATOMWISE:
            if not isinstance(selected_scope, AtomSetScope):
                raise TypeError(
                    "atomwise atom inputs require an AtomSetScope selected_scope"
                )
        elif self.basis is AtomInputBasis.RESIDUEWISE:
            if not isinstance(selected_scope, ResidueSetScope):
                raise TypeError(
                    "residuewise atom inputs require a ResidueSetScope selected_scope"
                )
        else:
            raise TypeError("atom inputs require a valid AtomInputBasis")

        realization = self.realization
        if realization is None:
            realization = (
                AtomInputRealization.EXACT_ATOMS
                if self.basis is AtomInputBasis.ATOMWISE
                else AtomInputRealization.RESIDUE_ATOMS
            )
        if not isinstance(realization, AtomInputRealization):
            raise TypeError(
                "atom inputs require an AtomInputRealization value when one is "
                "provided"
            )
        residuewise_realizations = {
            AtomInputRealization.RESIDUE_BACKBONE_ATOMS,
            AtomInputRealization.RESIDUE_SIDECHAIN_ATOMS,
        }
        if realization in residuewise_realizations and (
            self.basis is not AtomInputBasis.RESIDUEWISE
        ):
            raise TypeError(
                "residue-local atom inputs require a residuewise semantic basis"
            )

        object.__setattr__(self, "atom_indices", atom_indices)
        object.__setattr__(self, "realization", realization)

    def is_atomwise(self) -> bool:
        """Return whether this input preserves exact atomwise boundary intent."""

        return self.basis is AtomInputBasis.ATOMWISE

    def is_residuewise(self) -> bool:
        """Return whether this input preserves residuewise boundary intent."""

        return self.basis is AtomInputBasis.RESIDUEWISE

    def realizes_residue_sidechains(self) -> bool:
        """Return whether this input lowers residuewise semantics to side chains."""

        return self.realization is AtomInputRealization.RESIDUE_SIDECHAIN_ATOMS

    def realizes_residue_backbones(self) -> bool:
        """Return whether this input lowers residuewise semantics to backbone atoms."""

        return self.realization is AtomInputRealization.RESIDUE_BACKBONE_ATOMS

    def referenced_residue_ids(self) -> tuple[ResidueId, ...]:
        """Return referenced residue ids in first-seen order."""

        if isinstance(self.selected_scope, ResidueSetScope):
            return self.selected_scope.residue_ids

        assert isinstance(self.selected_scope, AtomSetScope)
        ordered_residue_ids: list[ResidueId] = []
        seen_residue_ids: set[ResidueId] = set()
        for atom_ref in self.selected_scope.atom_refs:
            if atom_ref.residue_id in seen_residue_ids:
                continue

            ordered_residue_ids.append(atom_ref.residue_id)
            seen_residue_ids.add(atom_ref.residue_id)

        return tuple(ordered_residue_ids)

    def as_scope(self) -> AtomSetScope | ResidueSetScope:
        """Return the semantic scope covered by this atom input."""

        if isinstance(self.selected_scope, (AtomSetScope, ResidueSetScope)):
            return self.selected_scope

        raise AssertionError("atom input selected_scope must be atom or residue set")

    def observed_atom_scope(
        self,
        snapshot: ProteinStructureSnapshot,
    ) -> AtomSetScope:
        """Return the realized observed atom scope for this atom input."""

        ordered_atom_refs = structure_ordered_atom_refs(snapshot)
        selected_atom_index_set = set(self.atom_indices)
        return AtomSetScope(
            atom_refs=tuple(
                atom_ref
                for atom_ref in ordered_atom_refs
                if snapshot.structure.constitution.atom_index(atom_ref)
                in selected_atom_index_set
            )
        )


def _normalize_atom_index_tuple(
    atom_indices: tuple[AtomIndex, ...],
) -> tuple[AtomIndex, ...]:
    """Normalize atom indices into first-seen order without duplicates."""

    normalized_atom_indices: list[AtomIndex] = []
    seen_atom_indices: set[AtomIndex] = set()
    for atom_index in atom_indices:
        if atom_index in seen_atom_indices:
            continue

        normalized_atom_indices.append(atom_index)
        seen_atom_indices.add(atom_index)

    return tuple(normalized_atom_indices)

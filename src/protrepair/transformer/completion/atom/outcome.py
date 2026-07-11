"""Typed outcomes for residue-local geometry placement."""

from dataclasses import dataclass

from protrepair.structure.snapshot import ProteinStructureSnapshot


@dataclass(frozen=True, slots=True)
class GeometryPlacementOutcome:
    """One geometry-placement result with any requested atoms left unplaced."""

    snapshot: ProteinStructureSnapshot
    skipped_atom_names: tuple[str, ...] = ()
    failure_reason: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.snapshot, ProteinStructureSnapshot):
            raise TypeError("geometry placement outcomes require a structure snapshot")

        skipped_atom_names: list[str] = []
        for atom_name in self.skipped_atom_names:
            if not isinstance(atom_name, str):
                raise TypeError("skipped atom names must contain strings")
            normalized_atom_name = atom_name.strip().upper()
            if not normalized_atom_name:
                raise ValueError("skipped atom names must not be blank")
            if normalized_atom_name not in skipped_atom_names:
                skipped_atom_names.append(normalized_atom_name)

        failure_reason = self.failure_reason
        if failure_reason is not None:
            if not isinstance(failure_reason, str):
                raise TypeError("geometry placement failure reason must be a string")
            failure_reason = failure_reason.strip() or None
        if skipped_atom_names and failure_reason is None:
            raise ValueError("skipped geometry placement requires a failure reason")

        object.__setattr__(self, "skipped_atom_names", tuple(skipped_atom_names))
        object.__setattr__(self, "failure_reason", failure_reason)

    def has_skipped_atoms(self) -> bool:
        """Return whether requested atoms remain absent after placement."""

        return bool(self.skipped_atom_names)

"""Source-coordinate identity normalization contracts."""

from dataclasses import dataclass

from protrepair.structure.labels import AtomRef, ResidueId

DEFAULT_CHAIN_ID = "_"


def normalize_chain_id(raw_chain_id: str) -> str:
    """Normalize a possibly blank source chain identifier."""

    normalized_chain_id = raw_chain_id.strip()
    return normalized_chain_id or DEFAULT_CHAIN_ID


def normalize_insertion_code(raw_insertion_code: str) -> str | None:
    """Normalize a possibly blank source insertion code."""

    normalized_insertion_code = raw_insertion_code.strip()
    return normalized_insertion_code or None


def normalize_altloc(raw_altloc: str) -> str | None:
    """Normalize a possibly blank source alternative-location code."""

    normalized_altloc = raw_altloc.strip("\x00 ").strip()
    return normalized_altloc or None


@dataclass(frozen=True, slots=True)
class SourceAtomIdentity:
    """Source endpoint identity before selected canonical lowering."""

    atom_ref: AtomRef
    component_id: str
    altloc: str | None = None

    def __post_init__(self) -> None:
        component_id = self.component_id.strip().upper()
        if not component_id:
            raise ValueError("source atom identities require a component id")

        altloc = self.altloc
        if altloc is not None:
            altloc = altloc.strip() or None

        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "altloc", altloc)

    def sort_key(self) -> tuple[ResidueId, str, str, str]:
        """Return a deterministic key for unordered source endpoint pairs."""

        return (
            self.atom_ref.residue_id,
            self.atom_ref.atom_name,
            self.component_id,
            self.altloc or "",
        )

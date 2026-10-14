"""Source-coordinate identity normalization contracts."""

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

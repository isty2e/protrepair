"""Canonical UniProt reference identifiers for external sequence relations."""

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class UniProtSequenceReference:
    """One UniProt accession plus an optional isoform and residue interval."""

    accession: str
    isoform_accession: str | None = None
    residue_start: int | None = None
    residue_end: int | None = None

    def __post_init__(self) -> None:
        accession = self.accession.strip().upper()
        if not accession:
            raise ValueError("UniProt accession must not be blank")
        if not accession.replace("-", "").isalnum():
            raise ValueError("UniProt accession must be alphanumeric")

        isoform_accession = self.isoform_accession
        if isoform_accession is not None:
            isoform_accession = isoform_accession.strip().upper() or None
        if isoform_accession is not None:
            if not isoform_accession.startswith(f"{accession}-"):
                raise ValueError(
                    "UniProt isoform accession must extend the base accession"
                )
            if isoform_accession == accession:
                raise ValueError(
                    "UniProt isoform accession must differ from base accession"
                )

        residue_start = self.residue_start
        residue_end = self.residue_end
        if residue_start is None and residue_end is None:
            pass
        elif residue_start is None or residue_end is None:
            raise ValueError(
                "UniProt residue intervals require both residue_start and residue_end"
            )
        else:
            if residue_start <= 0 or residue_end <= 0:
                raise ValueError("UniProt residue intervals must be positive")
            if residue_start > residue_end:
                raise ValueError("UniProt residue intervals must not be reversed")

        object.__setattr__(self, "accession", accession)
        object.__setattr__(self, "isoform_accession", isoform_accession)

    def effective_accession(self) -> str:
        """Return the isoform-specific accession when present."""

        if self.isoform_accession is not None:
            return self.isoform_accession

        return self.accession

    def references_interval(self) -> bool:
        """Return whether this reference names one residue interval."""

        return self.residue_start is not None

    def projected_sequence(self, sequence: str) -> str:
        """Return the full sequence or requested residue interval projection."""

        if self.residue_start is None or self.residue_end is None:
            return sequence

        return sequence[self.residue_start - 1 : self.residue_end]

"""UniProt source records and fetch outcomes."""

from dataclasses import dataclass
from enum import IntEnum

from protrepair.relation.uniprot_reference import UniProtSequenceReference


class UniProtSequenceFetchFailureKind(IntEnum):
    """Closed failure kinds for external UniProt sequence retrieval."""

    NOT_FOUND = 1
    REMOTE_ERROR = 2
    INVALID_RESPONSE = 3


@dataclass(frozen=True, slots=True)
class UniProtSequenceRecord:
    """One fetched UniProt sequence plus retrieval provenance/versioning."""

    uniprot_reference: UniProtSequenceReference
    primary_accession: str
    isoform_accession: str | None
    sequence: str
    entry_version: int | None = None
    sequence_version: int | None = None
    entry_type: str | None = None
    organism_scientific_name: str | None = None
    last_annotation_update_date: str | None = None
    last_sequence_update_date: str | None = None
    declared_isoform_accessions: tuple[str, ...] = ()
    source_url: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.uniprot_reference, UniProtSequenceReference):
            raise TypeError(
                "UniProt sequence records require a UniProtSequenceReference"
            )
        if self.uniprot_reference.references_interval():
            raise ValueError(
                "UniProt sequence records require one raw reference without "
                "residue-interval shaping"
            )

        primary_accession = self.primary_accession.strip().upper()
        if not primary_accession:
            raise ValueError("UniProt primary_accession must not be blank")

        isoform_accession = self.isoform_accession
        if isoform_accession is not None:
            isoform_accession = isoform_accession.strip().upper() or None

        sequence = "".join(self.sequence.split()).upper()
        if not sequence:
            raise ValueError("UniProt sequence records require a non-empty sequence")
        if not sequence.isalpha():
            raise ValueError(
                "UniProt sequence records require alphabetic amino-acid tokens"
            )

        declared_isoform_accessions = tuple(
            dict.fromkeys(
                accession.strip().upper()
                for accession in self.declared_isoform_accessions
                if accession.strip()
            )
        )
        for declared_isoform_accession in declared_isoform_accessions:
            if not declared_isoform_accession.startswith(
                f"{self.uniprot_reference.accession}-"
            ):
                raise ValueError(
                    "UniProt declared isoform accessions must extend the base accession"
                )

        object.__setattr__(self, "primary_accession", primary_accession)
        object.__setattr__(self, "isoform_accession", isoform_accession)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(
            self,
            "declared_isoform_accessions",
            declared_isoform_accessions,
        )
        object.__setattr__(
            self,
            "entry_type",
            _normalized_optional_label(self.entry_type),
        )
        object.__setattr__(
            self,
            "organism_scientific_name",
            _normalized_optional_label(self.organism_scientific_name),
        )
        object.__setattr__(
            self,
            "last_annotation_update_date",
            _normalized_optional_label(self.last_annotation_update_date),
        )
        object.__setattr__(
            self,
            "last_sequence_update_date",
            _normalized_optional_label(self.last_sequence_update_date),
        )
        object.__setattr__(
            self,
            "source_url",
            _normalized_optional_label(self.source_url),
        )

    def effective_accession(self) -> str:
        """Return the isoform-specific accession when present."""

        if self.isoform_accession is not None:
            return self.isoform_accession

        return self.primary_accession


@dataclass(frozen=True, slots=True)
class UniProtSequenceFetchFailure:
    """Typed failure result for one external UniProt sequence fetch."""

    requested_reference: UniProtSequenceReference
    kind: UniProtSequenceFetchFailureKind
    message: str
    status_code: int | None = None
    source_url: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.requested_reference, UniProtSequenceReference):
            raise TypeError(
                "UniProt sequence fetch failures require a UniProtSequenceReference"
            )
        if not isinstance(self.kind, UniProtSequenceFetchFailureKind):
            raise TypeError("UniProt sequence fetch failures require a failure kind")

        message = self.message.strip()
        if not message:
            raise ValueError(
                "UniProt sequence fetch failures require a non-blank message"
            )

        object.__setattr__(self, "message", message)
        object.__setattr__(
            self,
            "source_url",
            _normalized_optional_label(self.source_url),
        )


@dataclass(frozen=True, slots=True)
class UniProtSequenceFetchOutcome:
    """Closed fetch outcome that carries one request plus success or failure."""

    requested_reference: UniProtSequenceReference
    record: UniProtSequenceRecord | None = None
    failure: UniProtSequenceFetchFailure | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.requested_reference, UniProtSequenceReference):
            raise TypeError(
                "UniProt sequence fetch outcomes require a UniProtSequenceReference"
            )

        has_record = self.record is not None
        has_failure = self.failure is not None
        if has_record == has_failure:
            raise ValueError(
                "UniProt sequence fetch outcomes require exactly one of "
                "record or failure"
            )

        if self.record is not None:
            if not isinstance(self.record, UniProtSequenceRecord):
                raise TypeError(
                    "UniProt sequence fetch outcomes require a UniProtSequenceRecord"
                )
            if self.record.uniprot_reference.effective_accession() != (
                self.requested_reference.effective_accession()
            ):
                raise ValueError(
                    "UniProt sequence fetch outcomes require the record to match "
                    "the requested accession"
                )

        if self.failure is not None:
            if not isinstance(self.failure, UniProtSequenceFetchFailure):
                raise TypeError(
                    "UniProt sequence fetch outcomes require a "
                    "UniProtSequenceFetchFailure"
                )
            if self.failure.requested_reference != self.requested_reference:
                raise ValueError(
                    "UniProt sequence fetch outcomes require the failure to carry "
                    "the requested reference"
                )

    @classmethod
    def success(
        cls,
        requested_reference: UniProtSequenceReference,
        record: UniProtSequenceRecord,
    ) -> "UniProtSequenceFetchOutcome":
        """Return a successful UniProt sequence fetch outcome."""

        return cls(requested_reference=requested_reference, record=record)

    @classmethod
    def failure_result(
        cls,
        failure: UniProtSequenceFetchFailure,
    ) -> "UniProtSequenceFetchOutcome":
        """Return a failed UniProt sequence fetch outcome."""

        return cls(
            requested_reference=failure.requested_reference,
            failure=failure,
        )

    def is_success(self) -> bool:
        """Return whether the fetch succeeded."""

        return self.record is not None

    def require_record(self) -> UniProtSequenceRecord:
        """Return the fetched record or raise when the fetch failed."""

        if self.record is None:
            raise ValueError("UniProt sequence fetch outcome has no record")

        return self.record


@dataclass(frozen=True, slots=True)
class UniProtSequenceFamily:
    """One pure UniProt family containing successful canonical and isoform records."""

    canonical_record: UniProtSequenceRecord
    isoform_records: tuple[UniProtSequenceRecord, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.canonical_record, UniProtSequenceRecord):
            raise TypeError(
                "UniProt sequence families require a canonical UniProtSequenceRecord"
            )
        if self.canonical_record.uniprot_reference.isoform_accession is not None:
            raise ValueError(
                "UniProt sequence family canonical records must use the base accession"
            )

        isoform_records = tuple(self.isoform_records)
        seen_accessions = {self.canonical_record.effective_accession()}
        for record in isoform_records:
            if not isinstance(record, UniProtSequenceRecord):
                raise TypeError(
                    "UniProt sequence families require UniProtSequenceRecord values"
                )
            if record.uniprot_reference.accession != self.base_accession():
                raise ValueError(
                    "UniProt sequence family records must share the same base accession"
                )
            if record.uniprot_reference.isoform_accession is None:
                raise ValueError(
                    "UniProt sequence family isoform records must use "
                    "isoform-specific references"
                )
            accession = record.effective_accession()
            if accession in seen_accessions:
                raise ValueError(
                    "UniProt sequence families must not repeat the same "
                    "effective accession"
                )
            seen_accessions.add(accession)

        object.__setattr__(self, "isoform_records", isoform_records)

    def base_accession(self) -> str:
        """Return the base accession shared by this family."""

        return self.canonical_record.uniprot_reference.accession

    def records(self) -> tuple[UniProtSequenceRecord, ...]:
        """Return canonical and isoform records in stable order."""

        return (self.canonical_record, *self.isoform_records)

    def record_for_accession(
        self,
        accession: str,
    ) -> UniProtSequenceRecord | None:
        """Return one successful family record for the given accession."""

        normalized_accession = accession.strip().upper()
        for record in self.records():
            if record.effective_accession() == normalized_accession:
                return record

        return None


@dataclass(frozen=True, slots=True)
class UniProtSequenceFamilyFetchResult:
    """One UniProt family plus the request context and retained isoform failures."""

    requested_reference: UniProtSequenceReference
    family: UniProtSequenceFamily
    isoform_failures: tuple[UniProtSequenceFetchFailure, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.requested_reference, UniProtSequenceReference):
            raise TypeError(
                "UniProt sequence family fetch results require a "
                "UniProtSequenceReference"
            )
        if not isinstance(self.family, UniProtSequenceFamily):
            raise TypeError(
                "UniProt sequence family fetch results require a UniProtSequenceFamily"
            )
        if self.family.base_accession() != self.requested_reference.accession:
            raise ValueError(
                "UniProt sequence family fetch results must share the same base "
                "accession as the requested reference"
            )

        isoform_failures = tuple(self.isoform_failures)
        seen_accessions = {
            record.effective_accession() for record in self.family.records()
        }
        for failure in isoform_failures:
            if not isinstance(failure, UniProtSequenceFetchFailure):
                raise TypeError(
                    "UniProt sequence family fetch results require fetch failures"
                )
            reference = failure.requested_reference
            if reference.accession != self.requested_reference.accession:
                raise ValueError(
                    "UniProt sequence family fetch failures must share the same "
                    "base accession as the requested reference"
                )
            if reference.isoform_accession is None:
                raise ValueError(
                    "UniProt sequence family fetch failures must use "
                    "isoform-specific references"
                )
            accession = reference.effective_accession()
            if accession in seen_accessions:
                raise ValueError(
                    "UniProt sequence family fetch results must not repeat "
                    "the same effective accession"
                )
            seen_accessions.add(accession)

        object.__setattr__(self, "isoform_failures", isoform_failures)

    def base_accession(self) -> str:
        """Return the base accession shared by this fetch result."""

        return self.requested_reference.accession

    def successful_records(self) -> tuple[UniProtSequenceRecord, ...]:
        """Return successful canonical and isoform records in stable order."""

        return self.family.records()

    def failures(self) -> tuple[UniProtSequenceFetchFailure, ...]:
        """Return retained isoform fetch failures."""

        return self.isoform_failures

    def record_for_accession(
        self,
        accession: str,
    ) -> UniProtSequenceRecord | None:
        """Return one successful family record for the given accession."""

        return self.family.record_for_accession(accession)

    def requested_record(self) -> UniProtSequenceRecord | None:
        """Return the successful record matching the originally requested accession."""

        return self.record_for_accession(self.requested_reference.effective_accession())


@dataclass(frozen=True, slots=True)
class UniProtSequenceFamilyFetchOutcome:
    """Closed family fetch outcome with success or canonical-family failure."""

    result: UniProtSequenceFamilyFetchResult | None = None
    failure: UniProtSequenceFetchFailure | None = None

    def __post_init__(self) -> None:
        has_result = self.result is not None
        has_failure = self.failure is not None
        if has_result == has_failure:
            raise ValueError(
                "UniProt sequence family fetch outcomes require exactly one of "
                "result or failure"
            )

    @classmethod
    def success(
        cls,
        result: UniProtSequenceFamilyFetchResult,
    ) -> "UniProtSequenceFamilyFetchOutcome":
        """Return a successful family fetch outcome."""

        return cls(result=result)

    @classmethod
    def failure_result(
        cls,
        failure: UniProtSequenceFetchFailure,
    ) -> "UniProtSequenceFamilyFetchOutcome":
        """Return a failed family fetch outcome."""

        return cls(failure=failure)

    def is_success(self) -> bool:
        """Return whether the family fetch succeeded."""

        return self.result is not None

    def require_result(self) -> UniProtSequenceFamilyFetchResult:
        """Return the fetched family result or raise when the fetch failed."""

        if self.result is None:
            raise ValueError("UniProt sequence family fetch outcome has no result")

        return self.result


def _normalized_optional_label(value: str | None) -> str | None:
    """Return one stripped optional string value."""

    if value is None:
        return None

    return value.strip() or None

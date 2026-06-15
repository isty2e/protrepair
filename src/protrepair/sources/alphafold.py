"""AlphaFold external-reference value objects."""

from dataclasses import dataclass
from enum import IntEnum

from protrepair.sources.uniprot import UniProtSequenceReference
from protrepair.structure.provenance import FileFormat


class AlphaFoldFetchFailureKind(IntEnum):
    """Closed failure kinds for AlphaFold metadata or artifact retrieval."""

    NOT_FOUND = 1
    REMOTE_ERROR = 2
    INVALID_RESPONSE = 3
    ARTIFACT_UNAVAILABLE = 4


@dataclass(frozen=True, slots=True)
class AlphaFoldModelRecord:
    """One AlphaFold DB model record keyed by one resolved UniProt reference."""

    uniprot_reference: UniProtSequenceReference
    entry_id: str
    model_entity_id: str
    provider_id: str
    tool_used: str
    sequence: str
    model_created_date: str | None = None
    sequence_version_date: str | None = None
    global_metric_value: float | None = None
    latest_version: int | None = None
    pdb_url: str | None = None
    cif_url: str | None = None
    pae_doc_url: str | None = None
    plddt_doc_url: str | None = None
    source_api_url: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.uniprot_reference, UniProtSequenceReference):
            raise TypeError(
                "AlphaFold model records require a UniProtSequenceReference"
            )
        if self.uniprot_reference.references_interval():
            raise ValueError(
                "AlphaFold model records require one raw reference without "
                "residue-interval shaping"
            )

        entry_id = self.entry_id.strip()
        model_entity_id = self.model_entity_id.strip()
        provider_id = self.provider_id.strip()
        tool_used = self.tool_used.strip()
        if not entry_id or not model_entity_id or not provider_id or not tool_used:
            raise ValueError(
                "AlphaFold model records require non-blank ids, provider, and tool"
            )

        sequence = "".join(self.sequence.split()).upper()
        if not sequence or not sequence.isalpha():
            raise ValueError(
                "AlphaFold model records require one alphabetic amino-acid sequence"
            )

        if (
            self.global_metric_value is not None
            and not 0.0 <= self.global_metric_value <= 100.0
        ):
            raise ValueError(
                "AlphaFold global_metric_value must lie in [0, 100]"
            )
        if self.latest_version is not None and self.latest_version <= 0:
            raise ValueError("AlphaFold latest_version must be positive")

        pdb_url = _normalized_optional_url(self.pdb_url)
        cif_url = _normalized_optional_url(self.cif_url)
        if pdb_url is None and cif_url is None:
            raise ValueError(
                "AlphaFold model records require at least one structure artifact URL"
            )

        object.__setattr__(self, "entry_id", entry_id)
        object.__setattr__(self, "model_entity_id", model_entity_id)
        object.__setattr__(self, "provider_id", provider_id)
        object.__setattr__(self, "tool_used", tool_used)
        object.__setattr__(self, "sequence", sequence)
        object.__setattr__(
            self,
            "model_created_date",
            _normalized_optional_label(self.model_created_date),
        )
        object.__setattr__(
            self,
            "sequence_version_date",
            _normalized_optional_label(self.sequence_version_date),
        )
        object.__setattr__(self, "pdb_url", pdb_url)
        object.__setattr__(self, "cif_url", cif_url)
        object.__setattr__(
            self,
            "pae_doc_url",
            _normalized_optional_url(self.pae_doc_url),
        )
        object.__setattr__(
            self,
            "plddt_doc_url",
            _normalized_optional_url(self.plddt_doc_url),
        )
        object.__setattr__(
            self,
            "source_api_url",
            _normalized_optional_url(self.source_api_url),
        )

    def effective_accession(self) -> str:
        """Return the UniProt accession resolved by this AlphaFold record."""

        return self.uniprot_reference.effective_accession()

    def structure_url(
        self,
        file_format: FileFormat,
    ) -> str | None:
        """Return the artifact URL for one supported structure format."""

        if file_format is FileFormat.PDB:
            return self.pdb_url
        if file_format is FileFormat.MMCIF:
            return self.cif_url

        raise ValueError(f"unsupported AlphaFold structure format: {file_format!r}")


@dataclass(frozen=True, slots=True)
class AlphaFoldModelSet:
    """All AlphaFold model records returned for one requested UniProt reference."""

    requested_reference: UniProtSequenceReference
    models: tuple[AlphaFoldModelRecord, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.requested_reference, UniProtSequenceReference):
            raise TypeError(
                "AlphaFold model sets require a requested UniProtSequenceReference"
            )

        models = tuple(self.models)
        if not models:
            raise ValueError("AlphaFold model sets require at least one model record")

        seen_accessions: set[str] = set()
        for model in models:
            if not isinstance(model, AlphaFoldModelRecord):
                raise TypeError(
                    "AlphaFold model sets require AlphaFoldModelRecord values"
                )
            if model.uniprot_reference.accession != self.requested_reference.accession:
                raise ValueError(
                    "AlphaFold model records in one set must share the requested "
                    "base accession"
                )
            accession = model.effective_accession()
            if accession in seen_accessions:
                raise ValueError(
                    "AlphaFold model sets must not repeat the same UniProt accession"
                )
            seen_accessions.add(accession)

        object.__setattr__(self, "models", models)

    def record_for_accession(
        self,
        accession: str,
    ) -> AlphaFoldModelRecord | None:
        """Return one model record for the given accession when present."""

        normalized_accession = accession.strip().upper()
        for model in self.models:
            if model.effective_accession() == normalized_accession:
                return model

        return None

    def requested_record(self) -> AlphaFoldModelRecord | None:
        """Return the model record matching the originally requested accession."""

        return self.record_for_accession(self.requested_reference.effective_accession())


@dataclass(frozen=True, slots=True)
class AlphaFoldModelFetchFailure:
    """Typed failure result for AlphaFold model discovery."""

    requested_reference: UniProtSequenceReference
    kind: AlphaFoldFetchFailureKind
    message: str
    status_code: int | None = None
    source_url: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.requested_reference, UniProtSequenceReference):
            raise TypeError(
                "AlphaFold fetch failures require a UniProtSequenceReference"
            )
        if not isinstance(self.kind, AlphaFoldFetchFailureKind):
            raise TypeError(
                "AlphaFold fetch failures require an AlphaFoldFetchFailureKind"
            )

        message = self.message.strip()
        if not message:
            raise ValueError("AlphaFold fetch failures require a non-blank message")

        object.__setattr__(self, "message", message)
        object.__setattr__(
            self,
            "source_url",
            _normalized_optional_url(self.source_url),
        )


@dataclass(frozen=True, slots=True)
class AlphaFoldModelFetchOutcome:
    """Closed AlphaFold model discovery outcome."""

    model_set: AlphaFoldModelSet | None = None
    failure: AlphaFoldModelFetchFailure | None = None

    def __post_init__(self) -> None:
        has_model_set = self.model_set is not None
        has_failure = self.failure is not None
        if has_model_set == has_failure:
            raise ValueError(
                "AlphaFold model fetch outcomes require exactly one of "
                "model_set or failure"
            )

    @classmethod
    def success(
        cls,
        model_set: AlphaFoldModelSet,
    ) -> "AlphaFoldModelFetchOutcome":
        """Return a successful AlphaFold model discovery outcome."""

        return cls(model_set=model_set)

    @classmethod
    def failure_result(
        cls,
        failure: AlphaFoldModelFetchFailure,
    ) -> "AlphaFoldModelFetchOutcome":
        """Return a failed AlphaFold model discovery outcome."""

        return cls(failure=failure)

    def is_success(self) -> bool:
        """Return whether AlphaFold model discovery succeeded."""

        return self.model_set is not None

    def require_model_set(self) -> AlphaFoldModelSet:
        """Return the discovered model set or raise when discovery failed."""

        if self.model_set is None:
            raise ValueError("AlphaFold model fetch outcome has no model set")

        return self.model_set


@dataclass(frozen=True, slots=True)
class AlphaFoldStructureArtifact:
    """One fetched AlphaFold structure artifact kept outside canonical state."""

    model: AlphaFoldModelRecord
    file_format: FileFormat
    structure_text: str
    source_url: str

    def __post_init__(self) -> None:
        if not isinstance(self.model, AlphaFoldModelRecord):
            raise TypeError(
                "AlphaFold structure artifacts require an AlphaFoldModelRecord"
            )
        if not isinstance(self.file_format, FileFormat):
            raise TypeError(
                "AlphaFold structure artifacts require a FileFormat value"
            )

        structure_text = self.structure_text.strip()
        if not structure_text:
            raise ValueError(
                "AlphaFold structure artifacts require non-empty structure text"
            )

        source_url = self.source_url.strip()
        if not source_url:
            raise ValueError(
                "AlphaFold structure artifacts require a non-blank source URL"
            )

        object.__setattr__(self, "structure_text", structure_text)
        object.__setattr__(self, "source_url", source_url)

    def cache_key(self) -> str:
        """Return one deterministic cache key for this AlphaFold artifact."""

        return f"{self.model.entry_id}:{self.file_format.value}"


@dataclass(frozen=True, slots=True)
class AlphaFoldStructureFetchFailure:
    """Typed failure result for AlphaFold structure-artifact retrieval."""

    model: AlphaFoldModelRecord
    kind: AlphaFoldFetchFailureKind
    message: str
    status_code: int | None = None
    source_url: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.model, AlphaFoldModelRecord):
            raise TypeError(
                "AlphaFold structure fetch failures require an AlphaFoldModelRecord"
            )
        if not isinstance(self.kind, AlphaFoldFetchFailureKind):
            raise TypeError(
                "AlphaFold structure fetch failures require an "
                "AlphaFoldFetchFailureKind"
            )

        message = self.message.strip()
        if not message:
            raise ValueError(
                "AlphaFold structure fetch failures require a non-blank message"
            )

        object.__setattr__(self, "message", message)
        object.__setattr__(
            self,
            "source_url",
            _normalized_optional_url(self.source_url),
        )


@dataclass(frozen=True, slots=True)
class AlphaFoldStructureFetchOutcome:
    """Closed AlphaFold structure-artifact fetch outcome."""

    artifact: AlphaFoldStructureArtifact | None = None
    failure: AlphaFoldStructureFetchFailure | None = None

    def __post_init__(self) -> None:
        has_artifact = self.artifact is not None
        has_failure = self.failure is not None
        if has_artifact == has_failure:
            raise ValueError(
                "AlphaFold structure fetch outcomes require exactly one of "
                "artifact or failure"
            )

    @classmethod
    def success(
        cls,
        artifact: AlphaFoldStructureArtifact,
    ) -> "AlphaFoldStructureFetchOutcome":
        """Return a successful AlphaFold structure-artifact fetch outcome."""

        return cls(artifact=artifact)

    @classmethod
    def failure_result(
        cls,
        failure: AlphaFoldStructureFetchFailure,
    ) -> "AlphaFoldStructureFetchOutcome":
        """Return a failed AlphaFold structure-artifact fetch outcome."""

        return cls(failure=failure)

    def is_success(self) -> bool:
        """Return whether AlphaFold structure-artifact fetch succeeded."""

        return self.artifact is not None

    def require_artifact(self) -> AlphaFoldStructureArtifact:
        """Return the fetched artifact or raise when fetch failed."""

        if self.artifact is None:
            raise ValueError("AlphaFold structure fetch outcome has no artifact")

        return self.artifact


def _normalized_optional_label(value: str | None) -> str | None:
    """Return one stripped optional string value."""

    if value is None:
        return None

    return value.strip() or None


def _normalized_optional_url(value: str | None) -> str | None:
    """Return one stripped optional URL string value."""

    if value is None:
        return None

    normalized_value = value.strip()
    if not normalized_value:
        return None
    if "://" not in normalized_value:
        raise ValueError("AlphaFold URLs must be absolute")

    return normalized_value

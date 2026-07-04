"""AlphaFold source retrieval boundary."""

import json
from typing import Protocol, runtime_checkable
from urllib.error import HTTPError, URLError
from urllib.parse import urlsplit
from urllib.request import Request, urlopen

from protrepair.sources._network import (
    DEFAULT_SOURCE_RETRIEVAL_TIMEOUT_SECONDS,
    SourceResponseTooLargeError,
    normalize_source_retrieval_timeout,
    read_bounded_response_text,
)
from protrepair.sources.alphafold import (
    AlphaFoldFetchFailureKind,
    AlphaFoldModelFetchFailure,
    AlphaFoldModelFetchOutcome,
    AlphaFoldModelRecord,
    AlphaFoldModelSet,
    AlphaFoldStructureArtifact,
    AlphaFoldStructureFetchFailure,
    AlphaFoldStructureFetchOutcome,
)
from protrepair.sources.uniprot import UniProtSequenceReference
from protrepair.structure.provenance import FileFormat

JsonMapping = dict[str, object]
_ALLOWED_ALPHAFOLD_HOSTS = frozenset({"alphafold.ebi.ac.uk"})


@runtime_checkable
class _ResponseWithUrl(Protocol):
    """Response object exposing urllib's final URL after redirects."""

    def geturl(self) -> str:
        """Return the final response URL."""

        ...


class _FetchFailure:
    """Internal failure skeleton before binding to one public owner."""

    def __init__(
        self,
        *,
        kind: AlphaFoldFetchFailureKind,
        message: str,
        status_code: int | None = None,
    ) -> None:
        self.kind = kind
        self.message = message
        self.status_code = status_code


def fetch_alphafold_model_set(
    reference: UniProtSequenceReference,
    *,
    timeout_seconds: float = DEFAULT_SOURCE_RETRIEVAL_TIMEOUT_SECONDS,
) -> AlphaFoldModelFetchOutcome:
    """Fetch AlphaFold model metadata for one UniProt accession family."""

    timeout = normalize_source_retrieval_timeout(timeout_seconds)
    request_url = _alphafold_prediction_url(reference)
    payload, failure = _fetch_json_payload(
        request_url,
        timeout_seconds=timeout,
    )
    if failure is not None:
        return AlphaFoldModelFetchOutcome.failure_result(
            AlphaFoldModelFetchFailure(
                requested_reference=reference,
                kind=failure.kind,
                message=failure.message,
                status_code=failure.status_code,
                source_url=request_url,
            )
        )

    if not isinstance(payload, list):
        return AlphaFoldModelFetchOutcome.failure_result(
            AlphaFoldModelFetchFailure(
                requested_reference=reference,
                kind=AlphaFoldFetchFailureKind.INVALID_RESPONSE,
                message="AlphaFold prediction response root must be a JSON array",
                source_url=request_url,
            )
        )

    try:
        model_set = AlphaFoldModelSet(
            requested_reference=reference,
            models=tuple(
                _model_record_from_payload(
                    payload=model_payload,
                    source_api_url=request_url,
                )
                for model_payload in payload
            ),
        )
    except ValueError as error:
        return AlphaFoldModelFetchOutcome.failure_result(
            AlphaFoldModelFetchFailure(
                requested_reference=reference,
                kind=AlphaFoldFetchFailureKind.INVALID_RESPONSE,
                message=str(error),
                source_url=request_url,
            )
        )

    return AlphaFoldModelFetchOutcome.success(model_set)


def fetch_alphafold_structure_artifact(
    model: AlphaFoldModelRecord,
    *,
    file_format: FileFormat = FileFormat.PDB,
    timeout_seconds: float = DEFAULT_SOURCE_RETRIEVAL_TIMEOUT_SECONDS,
) -> AlphaFoldStructureFetchOutcome:
    """Fetch one AlphaFold structure artifact without canonicalizing it."""

    timeout = normalize_source_retrieval_timeout(timeout_seconds)
    artifact_url = model.structure_url(file_format)
    if artifact_url is None:
        return AlphaFoldStructureFetchOutcome.failure_result(
            AlphaFoldStructureFetchFailure(
                model=model,
                kind=AlphaFoldFetchFailureKind.ARTIFACT_UNAVAILABLE,
                message=(
                    "AlphaFold model does not expose an artifact for "
                    f"{file_format.value}"
                ),
            )
        )

    invalid_artifact_url = _alphafold_url_failure(artifact_url)
    if invalid_artifact_url is not None:
        return AlphaFoldStructureFetchOutcome.failure_result(
            AlphaFoldStructureFetchFailure(
                model=model,
                kind=invalid_artifact_url.kind,
                message=invalid_artifact_url.message,
                status_code=invalid_artifact_url.status_code,
                source_url=artifact_url,
            )
        )

    artifact_text, failure = _fetch_text_payload(
        artifact_url,
        timeout_seconds=timeout,
    )
    if failure is not None:
        return AlphaFoldStructureFetchOutcome.failure_result(
            AlphaFoldStructureFetchFailure(
                model=model,
                kind=failure.kind,
                message=failure.message,
                status_code=failure.status_code,
                source_url=artifact_url,
            )
        )

    if artifact_text is None:
        raise AssertionError("AlphaFold text fetch must return text or failure")

    return AlphaFoldStructureFetchOutcome.success(
        AlphaFoldStructureArtifact(
            model=model,
            file_format=file_format,
            structure_text=artifact_text,
            source_url=artifact_url,
        )
    )


def _alphafold_prediction_url(reference: UniProtSequenceReference) -> str:
    """Return the AlphaFold DB metadata endpoint for one reference."""

    return (
        "https://alphafold.ebi.ac.uk/api/prediction/"
        f"{reference.effective_accession()}"
    )


def _model_record_from_payload(
    *,
    payload: object,
    source_api_url: str,
) -> AlphaFoldModelRecord:
    """Decode one AlphaFold metadata payload into a typed model record."""

    if not isinstance(payload, dict):
        raise ValueError("AlphaFold model payload must be a JSON object")

    uniprot_accession = payload.get("uniprotAccession")
    entry_id = payload.get("entryId")
    model_entity_id = payload.get("modelEntityId")
    provider_id = payload.get("providerId")
    tool_used = payload.get("toolUsed")
    sequence = payload.get("sequence")
    if not isinstance(uniprot_accession, str):
        raise ValueError("AlphaFold model payload omitted uniprotAccession")
    if not isinstance(entry_id, str):
        raise ValueError("AlphaFold model payload omitted entryId")
    if not isinstance(model_entity_id, str):
        raise ValueError("AlphaFold model payload omitted modelEntityId")
    if not isinstance(provider_id, str):
        raise ValueError("AlphaFold model payload omitted providerId")
    if not isinstance(tool_used, str):
        raise ValueError("AlphaFold model payload omitted toolUsed")
    if not isinstance(sequence, str):
        raise ValueError("AlphaFold model payload omitted sequence")

    return AlphaFoldModelRecord(
        uniprot_reference=_uniprot_reference_from_accession(uniprot_accession),
        entry_id=entry_id,
        model_entity_id=model_entity_id,
        provider_id=provider_id,
        tool_used=tool_used,
        sequence=sequence,
        model_created_date=_optional_str(payload, "modelCreatedDate"),
        sequence_version_date=_optional_str(payload, "sequenceVersionDate"),
        global_metric_value=_optional_float(payload, "globalMetricValue"),
        latest_version=_optional_int(payload, "latestVersion"),
        pdb_url=_optional_str(payload, "pdbUrl"),
        cif_url=_optional_str(payload, "cifUrl"),
        pae_doc_url=_optional_str(payload, "paeDocUrl"),
        plddt_doc_url=_optional_str(payload, "plddtDocUrl"),
        source_api_url=source_api_url,
    )


def _uniprot_reference_from_accession(
    accession: str,
) -> UniProtSequenceReference:
    """Return one UniProt reference parsed from one accession string."""

    normalized_accession = accession.strip().upper()
    if "-" in normalized_accession:
        base_accession, _ = normalized_accession.split("-", maxsplit=1)
        return UniProtSequenceReference(
            accession=base_accession,
            isoform_accession=normalized_accession,
        )

    return UniProtSequenceReference(accession=normalized_accession)


def _alphafold_url_failure(request_url: str) -> _FetchFailure | None:
    """Return a fetch failure when one AlphaFold URL is outside AFDB HTTPS."""

    try:
        parsed_url = urlsplit(request_url)
        port = parsed_url.port
    except ValueError as error:
        return _FetchFailure(
            kind=AlphaFoldFetchFailureKind.INVALID_RESPONSE,
            message=f"AlphaFold URL is invalid: {error}",
        )

    if parsed_url.scheme.lower() != "https":
        return _FetchFailure(
            kind=AlphaFoldFetchFailureKind.INVALID_RESPONSE,
            message="AlphaFold URL must use HTTPS",
        )

    host = parsed_url.hostname
    if host is None or host.lower() not in _ALLOWED_ALPHAFOLD_HOSTS:
        return _FetchFailure(
            kind=AlphaFoldFetchFailureKind.INVALID_RESPONSE,
            message="AlphaFold URL host is not trusted",
        )

    if port not in (None, 443):
        return _FetchFailure(
            kind=AlphaFoldFetchFailureKind.INVALID_RESPONSE,
            message="AlphaFold URL must use the default HTTPS port",
        )

    return None


def _response_url(response: object, *, fallback_url: str) -> str:
    """Return the final response URL when urllib exposes one."""

    if isinstance(response, _ResponseWithUrl):
        response_url = response.geturl().strip()
        if response_url:
            return response_url

    return fallback_url


def _fetch_json_payload(
    request_url: str,
    *,
    timeout_seconds: float,
) -> tuple[object | None, _FetchFailure | None]:
    """Fetch and decode one JSON payload or return a typed failure."""

    request = Request(request_url, headers={"Accept": "application/json"})
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            final_url = _response_url(response, fallback_url=request_url)
            redirect_failure = _alphafold_url_failure(final_url)
            if redirect_failure is not None:
                return None, redirect_failure

            return json.loads(
                read_bounded_response_text(response, source_name="AlphaFold")
            ), None
    except HTTPError as error:
        return None, _FetchFailure(
            kind=(
                AlphaFoldFetchFailureKind.NOT_FOUND
                if error.code == 404
                else AlphaFoldFetchFailureKind.REMOTE_ERROR
            ),
            message=f"AlphaFold request failed with HTTP {error.code}",
            status_code=error.code,
        )
    except URLError as error:
        if isinstance(error.reason, TimeoutError):
            return None, _FetchFailure(
                kind=AlphaFoldFetchFailureKind.REMOTE_ERROR,
                message=(
                    "AlphaFold request timed out after "
                    f"{timeout_seconds:g} seconds: {error.reason}"
                ),
            )

        return None, _FetchFailure(
            kind=AlphaFoldFetchFailureKind.REMOTE_ERROR,
            message=f"AlphaFold request failed: {error.reason}",
        )
    except TimeoutError as error:
        return None, _FetchFailure(
            kind=AlphaFoldFetchFailureKind.REMOTE_ERROR,
            message=(
                "AlphaFold request timed out after "
                f"{timeout_seconds:g} seconds: {error}"
            ),
        )
    except json.JSONDecodeError as error:
        return None, _FetchFailure(
            kind=AlphaFoldFetchFailureKind.INVALID_RESPONSE,
            message=f"AlphaFold response was not valid JSON: {error.msg}",
        )
    except UnicodeDecodeError as error:
        return None, _FetchFailure(
            kind=AlphaFoldFetchFailureKind.INVALID_RESPONSE,
            message=f"AlphaFold response was not valid UTF-8: {error.reason}",
        )
    except SourceResponseTooLargeError as error:
        return None, _FetchFailure(
            kind=AlphaFoldFetchFailureKind.INVALID_RESPONSE,
            message=str(error),
        )


def _fetch_text_payload(
    request_url: str,
    *,
    timeout_seconds: float,
) -> tuple[str | None, _FetchFailure | None]:
    """Fetch one text artifact or return a typed failure skeleton."""

    request = Request(request_url)
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            final_url = _response_url(response, fallback_url=request_url)
            redirect_failure = _alphafold_url_failure(final_url)
            if redirect_failure is not None:
                return None, redirect_failure

            return (
                read_bounded_response_text(
                    response,
                    source_name="AlphaFold artifact",
                ),
                None,
            )
    except HTTPError as error:
        return None, _FetchFailure(
            kind=(
                AlphaFoldFetchFailureKind.NOT_FOUND
                if error.code == 404
                else AlphaFoldFetchFailureKind.REMOTE_ERROR
            ),
            message=f"AlphaFold artifact request failed with HTTP {error.code}",
            status_code=error.code,
        )
    except URLError as error:
        if isinstance(error.reason, TimeoutError):
            return None, _FetchFailure(
                kind=AlphaFoldFetchFailureKind.REMOTE_ERROR,
                message=(
                    "AlphaFold artifact request timed out after "
                    f"{timeout_seconds:g} seconds: {error.reason}"
                ),
            )

        return None, _FetchFailure(
            kind=AlphaFoldFetchFailureKind.REMOTE_ERROR,
            message=f"AlphaFold artifact request failed: {error.reason}",
        )
    except TimeoutError as error:
        return None, _FetchFailure(
            kind=AlphaFoldFetchFailureKind.REMOTE_ERROR,
            message=(
                "AlphaFold artifact request timed out after "
                f"{timeout_seconds:g} seconds: {error}"
            ),
        )
    except UnicodeDecodeError as error:
        return None, _FetchFailure(
            kind=AlphaFoldFetchFailureKind.INVALID_RESPONSE,
            message=(
                "AlphaFold artifact response was not valid UTF-8: "
                f"{error.reason}"
            ),
        )
    except SourceResponseTooLargeError as error:
        return None, _FetchFailure(
            kind=AlphaFoldFetchFailureKind.INVALID_RESPONSE,
            message=str(error),
        )


def _optional_int(
    payload: JsonMapping,
    key: str,
) -> int | None:
    """Return one optional integer field from a JSON mapping."""

    value = payload.get(key)
    if isinstance(value, int):
        return value

    return None


def _optional_float(
    payload: JsonMapping,
    key: str,
) -> float | None:
    """Return one optional float field from a JSON mapping."""

    value = payload.get(key)
    if isinstance(value, int | float):
        return float(value)

    return None


def _optional_str(
    payload: JsonMapping,
    key: str,
) -> str | None:
    """Return one optional string field from a JSON mapping."""

    value = payload.get(key)
    if isinstance(value, str):
        return value

    return None

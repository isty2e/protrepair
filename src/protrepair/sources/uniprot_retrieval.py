"""UniProt source retrieval boundary."""

import json
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from protrepair.sources._network import (
    DEFAULT_SOURCE_RETRIEVAL_TIMEOUT_SECONDS,
    normalize_source_retrieval_timeout,
)
from protrepair.sources.uniprot import (
    UniProtSequenceFamily,
    UniProtSequenceFamilyFetchOutcome,
    UniProtSequenceFamilyFetchResult,
    UniProtSequenceFetchFailure,
    UniProtSequenceFetchFailureKind,
    UniProtSequenceFetchOutcome,
    UniProtSequenceRecord,
    UniProtSequenceReference,
)

JsonMapping = dict[str, object]


def fetch_uniprot_sequence(
    reference: UniProtSequenceReference,
    *,
    timeout_seconds: float = DEFAULT_SOURCE_RETRIEVAL_TIMEOUT_SECONDS,
) -> UniProtSequenceFetchOutcome:
    """Fetch one canonical or isoform UniProt sequence record."""

    timeout = normalize_source_retrieval_timeout(timeout_seconds)
    request_url = _uniprot_record_url(reference)
    payload, failure = _fetch_uniprot_payload(
        reference=reference,
        request_url=request_url,
        timeout_seconds=timeout,
    )
    if failure is not None:
        return UniProtSequenceFetchOutcome.failure_result(failure)

    if payload is None:
        raise AssertionError("UniProt payload fetch must return payload or failure")

    try:
        record = _sequence_record_from_payload(
            reference=reference,
            payload=payload,
            request_url=request_url,
        )
    except ValueError as error:
        return UniProtSequenceFetchOutcome.failure_result(
            UniProtSequenceFetchFailure(
                requested_reference=reference,
                kind=UniProtSequenceFetchFailureKind.INVALID_RESPONSE,
                message=str(error),
                source_url=request_url,
            )
        )

    return UniProtSequenceFetchOutcome.success(reference, record)


def fetch_uniprot_sequence_family(
    reference: UniProtSequenceReference,
    *,
    timeout_seconds: float = DEFAULT_SOURCE_RETRIEVAL_TIMEOUT_SECONDS,
) -> UniProtSequenceFamilyFetchOutcome:
    """Fetch one canonical UniProt sequence plus all declared isoform sequences."""

    timeout = normalize_source_retrieval_timeout(timeout_seconds)
    canonical_reference = UniProtSequenceReference(
        accession=reference.accession,
        residue_start=reference.residue_start,
        residue_end=reference.residue_end,
    )
    canonical_request_url = _uniprot_record_url(canonical_reference)
    canonical_payload, canonical_failure = _fetch_uniprot_payload(
        reference=canonical_reference,
        request_url=canonical_request_url,
        timeout_seconds=timeout,
    )
    if canonical_failure is not None:
        return UniProtSequenceFamilyFetchOutcome.failure_result(canonical_failure)

    if canonical_payload is None:
        raise AssertionError("UniProt payload fetch must return payload or failure")

    try:
        canonical_record = _sequence_record_from_payload(
            reference=canonical_reference,
            payload=canonical_payload,
            request_url=canonical_request_url,
        )
    except ValueError as error:
        return UniProtSequenceFamilyFetchOutcome.failure_result(
            UniProtSequenceFetchFailure(
                requested_reference=canonical_reference,
                kind=UniProtSequenceFetchFailureKind.INVALID_RESPONSE,
                message=str(error),
                source_url=canonical_request_url,
            )
        )

    isoform_accessions = list(canonical_record.declared_isoform_accessions)
    requested_isoform_accession = reference.isoform_accession
    if (
        requested_isoform_accession is not None
        and requested_isoform_accession not in isoform_accessions
    ):
        isoform_accessions.append(requested_isoform_accession)

    isoform_outcomes = tuple(
        fetch_uniprot_sequence(
            UniProtSequenceReference(
                accession=reference.accession,
                isoform_accession=isoform_accession,
                residue_start=reference.residue_start,
                residue_end=reference.residue_end,
            ),
            timeout_seconds=timeout,
        )
        for isoform_accession in isoform_accessions
    )

    return UniProtSequenceFamilyFetchOutcome.success(
        UniProtSequenceFamilyFetchResult(
            requested_reference=reference,
            family=UniProtSequenceFamily(
                canonical_record=canonical_record,
                isoform_records=tuple(
                    outcome.require_record()
                    for outcome in isoform_outcomes
                    if outcome.is_success()
                ),
            ),
            isoform_failures=tuple(
                outcome.failure
                for outcome in isoform_outcomes
                if outcome.failure is not None
            ),
        )
    )


def _fetch_uniprot_payload(
    *,
    reference: UniProtSequenceReference,
    request_url: str,
    timeout_seconds: float,
) -> tuple[JsonMapping | None, UniProtSequenceFetchFailure | None]:
    """Fetch and decode one UniProt JSON payload or return a typed failure."""

    request = Request(
        request_url,
        headers={"Accept": "application/json"},
    )
    try:
        with urlopen(request, timeout=timeout_seconds) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as error:
        return None, UniProtSequenceFetchFailure(
            requested_reference=reference,
            kind=(
                UniProtSequenceFetchFailureKind.NOT_FOUND
                if error.code == 404
                else UniProtSequenceFetchFailureKind.REMOTE_ERROR
            ),
            message=f"UniProt request failed with HTTP {error.code}",
            status_code=error.code,
            source_url=request_url,
        )
    except URLError as error:
        return None, UniProtSequenceFetchFailure(
            requested_reference=reference,
            kind=UniProtSequenceFetchFailureKind.REMOTE_ERROR,
            message=f"UniProt request failed: {error.reason}",
            source_url=request_url,
        )
    except TimeoutError as error:
        return None, UniProtSequenceFetchFailure(
            requested_reference=reference,
            kind=UniProtSequenceFetchFailureKind.REMOTE_ERROR,
            message=(
                "UniProt request timed out after "
                f"{timeout_seconds:g} seconds: {error}"
            ),
            source_url=request_url,
        )
    except json.JSONDecodeError as error:
        return None, UniProtSequenceFetchFailure(
            requested_reference=reference,
            kind=UniProtSequenceFetchFailureKind.INVALID_RESPONSE,
            message=f"UniProt response was not valid JSON: {error.msg}",
            source_url=request_url,
        )

    if not isinstance(payload, dict):
        return None, UniProtSequenceFetchFailure(
            requested_reference=reference,
            kind=UniProtSequenceFetchFailureKind.INVALID_RESPONSE,
            message="UniProt response root payload was not a JSON object",
            source_url=request_url,
        )

    return payload, None


def _sequence_record_from_payload(
    *,
    reference: UniProtSequenceReference,
    payload: JsonMapping,
    request_url: str,
) -> UniProtSequenceRecord:
    """Decode one UniProt JSON payload into one typed sequence record."""

    sequence_payload = payload.get("sequence")
    entry_audit = payload.get("entryAudit")
    organism_payload = payload.get("organism")
    if not isinstance(sequence_payload, dict):
        raise ValueError("UniProt response omitted sequence payload")

    sequence_value = sequence_payload.get("value")
    primary_accession = payload.get("primaryAccession")
    if not isinstance(sequence_value, str) or not isinstance(primary_accession, str):
        raise ValueError(
            "UniProt response omitted primary accession or sequence value"
        )

    isoform_accession = (
        reference.isoform_accession
        if reference.isoform_accession is not None
        else None
    )
    entry_type = payload.get("entryType")
    entry_audit_payload = entry_audit if isinstance(entry_audit, dict) else None
    organism_payload_mapping = (
        organism_payload if isinstance(organism_payload, dict) else None
    )
    organism_scientific_name = (
        None
        if organism_payload_mapping is None
        else _optional_str(organism_payload_mapping, "scientificName")
    )

    return UniProtSequenceRecord(
        uniprot_reference=UniProtSequenceReference(
            accession=reference.accession,
            isoform_accession=reference.isoform_accession,
        ),
        primary_accession=primary_accession,
        isoform_accession=isoform_accession,
        sequence=sequence_value,
        entry_version=_optional_int(entry_audit_payload, "entryVersion"),
        sequence_version=_optional_int(entry_audit_payload, "sequenceVersion"),
        entry_type=entry_type if isinstance(entry_type, str) else None,
        organism_scientific_name=organism_scientific_name,
        last_annotation_update_date=_optional_str(
            entry_audit_payload,
            "lastAnnotationUpdateDate",
        ),
        last_sequence_update_date=_optional_str(
            entry_audit_payload,
            "lastSequenceUpdateDate",
        ),
        declared_isoform_accessions=_declared_isoform_accessions(payload),
        source_url=request_url,
    )


def _uniprot_record_url(reference: UniProtSequenceReference) -> str:
    """Return the UniProt REST JSON URL for one reference."""

    return f"https://rest.uniprot.org/uniprotkb/{reference.effective_accession()}.json"


def _declared_isoform_accessions(payload: JsonMapping) -> tuple[str, ...]:
    """Return declared isoform accessions from UniProt alternative-products comments."""

    comments_payload = payload.get("comments")
    if not isinstance(comments_payload, list):
        return ()

    isoform_accessions: list[str] = []
    for comment_payload in comments_payload:
        if not isinstance(comment_payload, dict):
            continue
        if comment_payload.get("commentType") != "ALTERNATIVE PRODUCTS":
            continue

        isoforms_payload = comment_payload.get("isoforms")
        if not isinstance(isoforms_payload, list):
            continue

        for isoform_payload in isoforms_payload:
            if not isinstance(isoform_payload, dict):
                continue

            isoform_ids_payload = isoform_payload.get("isoformIds")
            if not isinstance(isoform_ids_payload, list):
                continue

            for isoform_id in isoform_ids_payload:
                if isinstance(isoform_id, str) and isoform_id.strip():
                    isoform_accessions.append(isoform_id.strip().upper())

    return tuple(dict.fromkeys(isoform_accessions))


def _optional_int(
    payload: JsonMapping | None,
    key: str,
) -> int | None:
    """Return one optional integer field from a JSON mapping."""

    if payload is None:
        return None

    value = payload.get(key)
    if isinstance(value, int):
        return value

    return None


def _optional_str(
    payload: JsonMapping | None,
    key: str,
) -> str | None:
    """Return one optional string field from a JSON mapping."""

    if payload is None:
        return None

    value = payload.get(key)
    if isinstance(value, str):
        return value

    return None

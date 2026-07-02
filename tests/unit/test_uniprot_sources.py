"""UniProt source retrieval tests."""

import json
from email.message import Message
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from protrepair.sources import (
    UniProtSequenceFetchFailureKind,
    UniProtSequenceReference,
)
from protrepair.sources.uniprot_retrieval import (
    DEFAULT_SOURCE_RETRIEVAL_TIMEOUT_SECONDS,
    fetch_uniprot_sequence,
    fetch_uniprot_sequence_family,
)


class FakeResponse:
    """Minimal context-managed HTTP response for UniProt boundary tests."""

    def __init__(self, payload: dict[str, object]) -> None:
        self._payload = payload

    def __enter__(self) -> "FakeResponse":
        """Enter the fake response context."""

        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: object | None,
    ) -> None:
        """Exit the fake response context."""

        return None

    def read(self) -> bytes:
        """Return one JSON-encoded payload body."""

        return json.dumps(self._payload).encode("utf-8")


def test_fetch_uniprot_sequence_family_fetches_canonical_and_isoforms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Family fetch should keep canonical and isoform records together."""

    payload_by_accession = {
        "P04406": {
            "primaryAccession": "P04406",
            "entryAudit": {
                "entryVersion": 279,
                "sequenceVersion": 3,
            },
            "sequence": {"value": "MPEPTIDE"},
            "comments": [
                {
                    "commentType": "ALTERNATIVE PRODUCTS",
                    "isoforms": [
                        {"isoformIds": ["P04406-1"]},
                        {"isoformIds": ["P04406-2"]},
                    ],
                }
            ],
        },
        "P04406-1": {
            "primaryAccession": "P04406-1",
            "sequence": {"value": "MPEPTIDE"},
        },
        "P04406-2": {
            "primaryAccession": "P04406-2",
            "sequence": {"value": "MPTIDE"},
        },
    }

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        assert timeout == DEFAULT_SOURCE_RETRIEVAL_TIMEOUT_SECONDS
        full_url = request.full_url
        accession = full_url.removesuffix(".json").rsplit("/", maxsplit=1)[-1]
        return FakeResponse(payload_by_accession[accession])

    import protrepair.sources.uniprot_retrieval as uniprot_io

    monkeypatch.setattr(uniprot_io, "urlopen", fake_urlopen)

    outcome = fetch_uniprot_sequence_family(
        UniProtSequenceReference(accession="P04406")
    )

    assert outcome.is_success()
    family = outcome.require_result()
    assert family.family.canonical_record.effective_accession() == "P04406"
    assert family.family.canonical_record.declared_isoform_accessions == (
        "P04406-1",
        "P04406-2",
    )
    assert tuple(
        record.effective_accession()
        for record in family.family.isoform_records
    ) == ("P04406-1", "P04406-2")


def test_fetch_uniprot_sequence_family_preserves_isoform_fetch_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Family fetch should retain per-isoform failures without dropping success."""

    payload_by_accession = {
        "P04406": {
            "primaryAccession": "P04406",
            "sequence": {"value": "MPEPTIDE"},
            "comments": [
                {
                    "commentType": "ALTERNATIVE PRODUCTS",
                    "isoforms": [{"isoformIds": ["P04406-2"]}],
                }
            ],
        }
    }

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        assert timeout == DEFAULT_SOURCE_RETRIEVAL_TIMEOUT_SECONDS
        full_url = request.full_url
        accession = full_url.removesuffix(".json").rsplit("/", maxsplit=1)[-1]
        if accession == "P04406-2":
            raise HTTPError(
                full_url,
                404,
                "not found",
                hdrs=Message(),
                fp=None,
            )

        return FakeResponse(payload_by_accession[accession])

    import protrepair.sources.uniprot_retrieval as uniprot_io

    monkeypatch.setattr(uniprot_io, "urlopen", fake_urlopen)

    outcome = fetch_uniprot_sequence_family(
        UniProtSequenceReference(
            accession="P04406",
            isoform_accession="P04406-2",
        )
    )

    assert outcome.is_success()
    family = outcome.require_result()
    assert family.requested_record() is None
    assert len(family.failures()) == 1
    assert family.failures()[0].kind is UniProtSequenceFetchFailureKind.NOT_FOUND


def test_fetch_uniprot_sequence_family_fails_when_canonical_fetch_fails(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Family fetch should fail closed when the canonical base accession fails."""

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        assert timeout == DEFAULT_SOURCE_RETRIEVAL_TIMEOUT_SECONDS
        raise HTTPError(
            request.full_url,
            404,
            "not found",
            hdrs=Message(),
            fp=None,
        )

    import protrepair.sources.uniprot_retrieval as uniprot_io

    monkeypatch.setattr(uniprot_io, "urlopen", fake_urlopen)

    outcome = fetch_uniprot_sequence_family(
        UniProtSequenceReference(accession="P99999")
    )

    assert outcome.is_success() is False
    assert outcome.failure is not None
    assert outcome.failure.kind is UniProtSequenceFetchFailureKind.NOT_FOUND


def test_fetch_uniprot_sequence_passes_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Direct UniProt fetch should pass caller timeout to urlopen."""

    seen_timeouts: list[float] = []
    payload = {
        "primaryAccession": "P04406",
        "sequence": {"value": "MPEPTIDE"},
    }

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        seen_timeouts.append(timeout)
        return FakeResponse(payload)

    import protrepair.sources.uniprot_retrieval as uniprot_io

    monkeypatch.setattr(uniprot_io, "urlopen", fake_urlopen)

    outcome = fetch_uniprot_sequence(
        UniProtSequenceReference(accession="P04406"),
        timeout_seconds=6.25,
    )

    assert outcome.is_success()
    assert seen_timeouts == [6.25]


def test_fetch_uniprot_sequence_family_passes_timeout_to_each_fetch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Family fetch should pass one timeout through canonical and isoform fetches."""

    payload_by_accession = {
        "P04406": {
            "primaryAccession": "P04406",
            "sequence": {"value": "MPEPTIDE"},
            "comments": [
                {
                    "commentType": "ALTERNATIVE PRODUCTS",
                    "isoforms": [
                        {"isoformIds": ["P04406-1"]},
                        {"isoformIds": ["P04406-2"]},
                    ],
                }
            ],
        },
        "P04406-1": {
            "primaryAccession": "P04406-1",
            "sequence": {"value": "MPEPTIDE"},
        },
        "P04406-2": {
            "primaryAccession": "P04406-2",
            "sequence": {"value": "MPTIDE"},
        },
    }
    timeout_by_accession: dict[str, float] = {}

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        accession = request.full_url.removesuffix(".json").rsplit("/", maxsplit=1)[-1]
        timeout_by_accession[accession] = timeout
        return FakeResponse(payload_by_accession[accession])

    import protrepair.sources.uniprot_retrieval as uniprot_io

    monkeypatch.setattr(uniprot_io, "urlopen", fake_urlopen)

    outcome = fetch_uniprot_sequence_family(
        UniProtSequenceReference(accession="P04406"),
        timeout_seconds=8.0,
    )

    assert outcome.is_success()
    assert timeout_by_accession == {
        "P04406": 8.0,
        "P04406-1": 8.0,
        "P04406-2": 8.0,
    }


def test_fetch_uniprot_sequence_family_converts_timeout_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A socket timeout should become a typed UniProt remote failure."""

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        raise TimeoutError("timed out")

    import protrepair.sources.uniprot_retrieval as uniprot_io

    monkeypatch.setattr(uniprot_io, "urlopen", fake_urlopen)

    outcome = fetch_uniprot_sequence_family(
        UniProtSequenceReference(accession="P04406"),
        timeout_seconds=0.5,
    )

    assert outcome.is_success() is False
    assert outcome.failure is not None
    assert outcome.failure.kind is UniProtSequenceFetchFailureKind.REMOTE_ERROR
    assert "timed out" in outcome.failure.message
    assert "0.5" in outcome.failure.message


def test_fetch_uniprot_sequence_family_preserves_isoform_timeout_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Isoform timeout should be retained as a per-isoform failure."""

    canonical_payload = {
        "primaryAccession": "P04406",
        "sequence": {"value": "MPEPTIDE"},
        "comments": [
            {
                "commentType": "ALTERNATIVE PRODUCTS",
                "isoforms": [{"isoformIds": ["P04406-2"]}],
            }
        ],
    }

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        accession = request.full_url.removesuffix(".json").rsplit("/", maxsplit=1)[-1]
        if accession == "P04406-2":
            raise URLError(TimeoutError("timed out"))

        return FakeResponse(canonical_payload)

    import protrepair.sources.uniprot_retrieval as uniprot_io

    monkeypatch.setattr(uniprot_io, "urlopen", fake_urlopen)

    outcome = fetch_uniprot_sequence_family(
        UniProtSequenceReference(accession="P04406"),
        timeout_seconds=1.25,
    )

    assert outcome.is_success()
    family = outcome.require_result()
    assert (
        tuple(
            record.effective_accession()
            for record in family.family.isoform_records
        )
        == ()
    )
    assert len(family.failures()) == 1
    assert family.failures()[0].kind is UniProtSequenceFetchFailureKind.REMOTE_ERROR
    assert "timed out" in family.failures()[0].message
    assert "1.25" in family.failures()[0].message

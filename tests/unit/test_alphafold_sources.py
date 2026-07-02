"""AlphaFold source retrieval tests."""

import json
import math
from collections.abc import Mapping
from email.message import Message
from typing import cast
from urllib.error import HTTPError, URLError
from urllib.request import Request

import pytest

from protrepair.sources import (
    AlphaFoldFetchFailureKind,
    AlphaFoldModelRecord,
    UniProtSequenceReference,
)
from protrepair.sources.alphafold_retrieval import (
    DEFAULT_SOURCE_RETRIEVAL_TIMEOUT_SECONDS,
    fetch_alphafold_model_set,
    fetch_alphafold_structure_artifact,
)
from protrepair.structure.provenance import FileFormat


class FakeResponse:
    """Minimal context-managed response for AlphaFold boundary tests."""

    def __init__(
        self,
        payload: str | Mapping[str, object] | tuple[Mapping[str, object], ...],
    ) -> None:
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
        """Return one encoded payload body."""

        if isinstance(self._payload, str):
            return self._payload.encode("utf-8")

        return json.dumps(self._payload).encode("utf-8")


def test_fetch_alphafold_model_set_preserves_canonical_and_isoform_records(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AlphaFold metadata fetch should keep the returned accession family together."""

    metadata_payload = (
        {
            "entryId": "AF-P04406-F1",
            "modelEntityId": "AF-P04406-F1",
            "providerId": "GDM",
            "toolUsed": "AlphaFold Monomer v2.0 pipeline",
            "uniprotAccession": "P04406",
            "sequence": "MPEPTIDE",
            "latestVersion": 6,
            "globalMetricValue": 98.12,
            "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P04406-F1-model_v6.pdb",
            "cifUrl": "https://alphafold.ebi.ac.uk/files/AF-P04406-F1-model_v6.cif",
        },
        {
            "entryId": "AF-P04406-2-F1",
            "modelEntityId": "AF-P04406-2-F1",
            "providerId": "GDM",
            "toolUsed": "AlphaFold Monomer v2.0 pipeline",
            "uniprotAccession": "P04406-2",
            "sequence": "MPTIDE",
            "latestVersion": 6,
            "globalMetricValue": 97.5,
            "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P04406-2-F1-model_v6.pdb",
            "cifUrl": "https://alphafold.ebi.ac.uk/files/AF-P04406-2-F1-model_v6.cif",
        },
    )

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        assert timeout == DEFAULT_SOURCE_RETRIEVAL_TIMEOUT_SECONDS
        return FakeResponse(metadata_payload)

    import protrepair.sources.alphafold_retrieval as alphafold_io

    monkeypatch.setattr(alphafold_io, "urlopen", fake_urlopen)

    outcome = fetch_alphafold_model_set(
        UniProtSequenceReference(accession="P04406")
    )

    assert outcome.is_success()
    model_set = outcome.require_model_set()
    assert tuple(model.effective_accession() for model in model_set.models) == (
        "P04406",
        "P04406-2",
    )
    assert model_set.requested_record() is model_set.models[0]


def test_fetch_alphafold_structure_artifact_downloads_raw_text(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AlphaFold artifact fetch should return external raw structure text."""

    metadata_payload = (
        {
            "entryId": "AF-P04406-F1",
            "modelEntityId": "AF-P04406-F1",
            "providerId": "GDM",
            "toolUsed": "AlphaFold Monomer v2.0 pipeline",
            "uniprotAccession": "P04406",
            "sequence": "MPEPTIDE",
            "latestVersion": 6,
            "globalMetricValue": 98.12,
            "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P04406-F1-model_v6.pdb",
            "cifUrl": "https://alphafold.ebi.ac.uk/files/AF-P04406-F1-model_v6.cif",
        },
    )

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        assert timeout == DEFAULT_SOURCE_RETRIEVAL_TIMEOUT_SECONDS
        if request.full_url.endswith(".pdb"):
            return FakeResponse("HEADER    ALPHAFOLD MODEL\nATOM      1")

        return FakeResponse(metadata_payload)

    import protrepair.sources.alphafold_retrieval as alphafold_io

    monkeypatch.setattr(alphafold_io, "urlopen", fake_urlopen)

    model_outcome = fetch_alphafold_model_set(
        UniProtSequenceReference(accession="P04406")
    )
    model = model_outcome.require_model_set().models[0]

    artifact_outcome = fetch_alphafold_structure_artifact(
        model,
        file_format=FileFormat.PDB,
    )

    assert artifact_outcome.is_success()
    artifact = artifact_outcome.require_artifact()
    assert artifact.file_format is FileFormat.PDB
    assert artifact.cache_key() == "AF-P04406-F1:pdb"
    assert "HEADER" in artifact.structure_text


def test_fetch_alphafold_model_set_surfaces_not_found_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AlphaFold metadata fetch should report typed not-found failures."""

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        assert timeout == DEFAULT_SOURCE_RETRIEVAL_TIMEOUT_SECONDS
        raise HTTPError(
            request.full_url,
            404,
            "not found",
            hdrs=Message(),
            fp=None,
        )

    import protrepair.sources.alphafold_retrieval as alphafold_io

    monkeypatch.setattr(alphafold_io, "urlopen", fake_urlopen)

    outcome = fetch_alphafold_model_set(
        UniProtSequenceReference(accession="P99999")
    )

    assert outcome.is_success() is False
    assert outcome.failure is not None
    assert outcome.failure.kind is AlphaFoldFetchFailureKind.NOT_FOUND


def test_fetch_alphafold_model_set_passes_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AlphaFold metadata fetch should pass caller timeout to urlopen."""

    seen_timeouts: list[float] = []
    metadata_payload = (
        {
            "entryId": "AF-P04406-F1",
            "modelEntityId": "AF-P04406-F1",
            "providerId": "GDM",
            "toolUsed": "AlphaFold Monomer v2.0 pipeline",
            "uniprotAccession": "P04406",
            "sequence": "MPEPTIDE",
            "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P04406-F1-model_v6.pdb",
        },
    )

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        seen_timeouts.append(timeout)
        return FakeResponse(metadata_payload)

    import protrepair.sources.alphafold_retrieval as alphafold_io

    monkeypatch.setattr(alphafold_io, "urlopen", fake_urlopen)

    outcome = fetch_alphafold_model_set(
        UniProtSequenceReference(accession="P04406"),
        timeout_seconds=7.5,
    )

    assert outcome.is_success()
    assert seen_timeouts == [7.5]


def test_fetch_alphafold_structure_artifact_passes_configured_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """AlphaFold artifact fetch should pass caller timeout to text retrieval."""

    metadata_payload = (
        {
            "entryId": "AF-P04406-F1",
            "modelEntityId": "AF-P04406-F1",
            "providerId": "GDM",
            "toolUsed": "AlphaFold Monomer v2.0 pipeline",
            "uniprotAccession": "P04406",
            "sequence": "MPEPTIDE",
            "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P04406-F1-model_v6.pdb",
        },
    )
    timeout_by_url: dict[str, float] = {}

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        timeout_by_url[request.full_url] = timeout
        if request.full_url.endswith(".pdb"):
            return FakeResponse("HEADER    ALPHAFOLD MODEL\nATOM      1")

        return FakeResponse(metadata_payload)

    import protrepair.sources.alphafold_retrieval as alphafold_io

    monkeypatch.setattr(alphafold_io, "urlopen", fake_urlopen)

    model = fetch_alphafold_model_set(
        UniProtSequenceReference(accession="P04406"),
        timeout_seconds=3.0,
    ).require_model_set().models[0]

    outcome = fetch_alphafold_structure_artifact(
        model,
        file_format=FileFormat.PDB,
        timeout_seconds=4.25,
    )

    assert outcome.is_success()
    assert timeout_by_url == {
        "https://alphafold.ebi.ac.uk/api/prediction/P04406": 3.0,
        "https://alphafold.ebi.ac.uk/files/AF-P04406-F1-model_v6.pdb": 4.25,
    }


def test_fetch_alphafold_model_set_converts_timeout_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A socket timeout should become a typed AlphaFold remote failure."""

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        raise TimeoutError("timed out")

    import protrepair.sources.alphafold_retrieval as alphafold_io

    monkeypatch.setattr(alphafold_io, "urlopen", fake_urlopen)

    outcome = fetch_alphafold_model_set(
        UniProtSequenceReference(accession="P04406"),
        timeout_seconds=0.5,
    )

    assert outcome.is_success() is False
    assert outcome.failure is not None
    assert outcome.failure.kind is AlphaFoldFetchFailureKind.REMOTE_ERROR
    assert "timed out" in outcome.failure.message
    assert "0.5" in outcome.failure.message


def test_fetch_alphafold_structure_artifact_converts_urlerror_timeout_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A urllib-wrapped timeout should become a typed artifact failure."""

    metadata_payload = (
        {
            "entryId": "AF-P04406-F1",
            "modelEntityId": "AF-P04406-F1",
            "providerId": "GDM",
            "toolUsed": "AlphaFold Monomer v2.0 pipeline",
            "uniprotAccession": "P04406",
            "sequence": "MPEPTIDE",
            "pdbUrl": "https://alphafold.ebi.ac.uk/files/AF-P04406-F1-model_v6.pdb",
        },
    )

    def fake_urlopen(request: Request, *, timeout: float) -> FakeResponse:
        if request.full_url.endswith(".pdb"):
            raise URLError(TimeoutError("timed out"))

        return FakeResponse(metadata_payload)

    import protrepair.sources.alphafold_retrieval as alphafold_io

    monkeypatch.setattr(alphafold_io, "urlopen", fake_urlopen)

    model = fetch_alphafold_model_set(
        UniProtSequenceReference(accession="P04406")
    ).require_model_set().models[0]
    outcome = fetch_alphafold_structure_artifact(model)

    assert outcome.is_success() is False
    assert outcome.failure is not None
    assert outcome.failure.kind is AlphaFoldFetchFailureKind.REMOTE_ERROR
    assert "timed out" in outcome.failure.message


@pytest.mark.parametrize("timeout_seconds", [0.0, -1.0, math.inf, math.nan])
def test_fetch_alphafold_model_set_rejects_invalid_timeout(
    timeout_seconds: float,
) -> None:
    """Public source retrieval should reject non-positive or non-finite timeouts."""

    with pytest.raises(ValueError, match="timeout_seconds"):
        fetch_alphafold_model_set(
            UniProtSequenceReference(accession="P04406"),
            timeout_seconds=timeout_seconds,
        )


@pytest.mark.parametrize(
    "timeout_seconds",
    [cast(float, True), cast(float, "5")],
)
def test_fetch_alphafold_model_set_rejects_non_numeric_timeout(
    timeout_seconds: float,
) -> None:
    """Runtime timeout validation should not accept bools or numeric strings."""

    with pytest.raises(TypeError, match="timeout_seconds"):
        fetch_alphafold_model_set(
            UniProtSequenceReference(accession="P04406"),
            timeout_seconds=timeout_seconds,
        )


def test_fetch_alphafold_structure_artifact_rejects_invalid_timeout_first() -> None:
    """Invalid timeout should fail before artifact-availability branching."""

    model = AlphaFoldModelRecord(
        uniprot_reference=UniProtSequenceReference(accession="P04406"),
        entry_id="AF-P04406-F1",
        model_entity_id="AF-P04406-F1",
        provider_id="GDM",
        tool_used="AlphaFold Monomer v2.0 pipeline",
        sequence="MPEPTIDE",
        pdb_url="https://alphafold.ebi.ac.uk/files/AF-P04406-F1-model_v6.pdb",
    )

    with pytest.raises(ValueError, match="timeout_seconds"):
        fetch_alphafold_structure_artifact(
            model,
            file_format=FileFormat.MMCIF,
            timeout_seconds=0.0,
        )

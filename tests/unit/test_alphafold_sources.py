"""AlphaFold source retrieval tests."""

import json
from email.message import Message
from urllib.error import HTTPError
from urllib.request import Request

import pytest

from protrepair.sources import (
    AlphaFoldFetchFailureKind,
    UniProtSequenceReference,
)
from protrepair.sources.alphafold_retrieval import (
    fetch_alphafold_model_set,
    fetch_alphafold_structure_artifact,
)
from protrepair.structure.provenance import FileFormat


class FakeResponse:
    """Minimal context-managed response for AlphaFold boundary tests."""

    def __init__(
        self,
        payload: str | dict[str, object] | tuple[dict[str, object], ...],
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

    def fake_urlopen(request: Request) -> FakeResponse:
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

    def fake_urlopen(request: Request) -> FakeResponse:
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

    def fake_urlopen(request: Request) -> FakeResponse:
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



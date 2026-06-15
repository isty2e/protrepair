"""Downloaded whole-structure corpus source fixtures for exploratory tracking."""

from dataclasses import dataclass
from pathlib import Path

CORPUS_FIXTURE_ROOT = Path("tests/fixtures/corpus")


@dataclass(frozen=True, slots=True)
class WholeStructureCorpusSource:
    """One whole-structure corpus source plus optional raw-source chain subset."""

    case_id: str
    source_pdb_id: str
    output_path: Path
    selected_source_chain_ids: tuple[str, ...] = ()
    download_format_priority: tuple[str, ...] = ("pdb", "cif")


def corpus_source_fixture_path(filename: str) -> Path:
    """Return one canonical whole-structure corpus fixture path."""

    return CORPUS_FIXTURE_ROOT / filename


WHOLE_STRUCTURE_CORPUS_SOURCES: dict[str, WholeStructureCorpusSource] = {
    "2dn2-whole-structure": WholeStructureCorpusSource(
        case_id="2dn2-whole-structure",
        source_pdb_id="2DN2",
        output_path=corpus_source_fixture_path("pdb2dn2.ent"),
    ),
    "2qls-whole-structure": WholeStructureCorpusSource(
        case_id="2qls-whole-structure",
        source_pdb_id="2QLS",
        output_path=corpus_source_fixture_path("pdb2qls.ent"),
    ),
    "2z62-whole-structure": WholeStructureCorpusSource(
        case_id="2z62-whole-structure",
        source_pdb_id="2Z62",
        output_path=corpus_source_fixture_path("pdb2z62.ent"),
    ),
    "3i1n-whole-structure": WholeStructureCorpusSource(
        case_id="3i1n-whole-structure",
        source_pdb_id="3I1N",
        output_path=corpus_source_fixture_path("pdb3i1n.ent"),
    ),
    "3j4p-whole-structure": WholeStructureCorpusSource(
        case_id="3j4p-whole-structure",
        source_pdb_id="3J4P",
        output_path=corpus_source_fixture_path("pdb3j4p.ent"),
    ),
    "3j7h-binding-site-backbone": WholeStructureCorpusSource(
        case_id="3j7h-binding-site-backbone",
        source_pdb_id="3J7H",
        output_path=corpus_source_fixture_path("pdb3j7h.ent"),
    ),
    "3j7l-whole-structure": WholeStructureCorpusSource(
        case_id="3j7l-whole-structure",
        source_pdb_id="3J7L",
        output_path=corpus_source_fixture_path("pdb3j7l.ent"),
    ),
    "3j9z-chain-sb": WholeStructureCorpusSource(
        case_id="3j9z-chain-sb",
        source_pdb_id="3J9Z",
        output_path=corpus_source_fixture_path("3j9z.cif"),
        selected_source_chain_ids=("SB",),
        download_format_priority=("cif", "pdb"),
    ),
    "3ja8-whole-structure": WholeStructureCorpusSource(
        case_id="3ja8-whole-structure",
        source_pdb_id="3JA8",
        output_path=corpus_source_fixture_path("pdb3ja8.ent"),
    ),
    "4gd1-whole-structure": WholeStructureCorpusSource(
        case_id="4gd1-whole-structure",
        source_pdb_id="4GD1",
        output_path=corpus_source_fixture_path("pdb4gd1.ent"),
    ),
    "5n9y-whole-structure": WholeStructureCorpusSource(
        case_id="5n9y-whole-structure",
        source_pdb_id="5N9Y",
        output_path=corpus_source_fixture_path("5n9y.cif"),
        download_format_priority=("cif", "pdb"),
    ),
    "5owx-whole-structure": WholeStructureCorpusSource(
        case_id="5owx-whole-structure",
        source_pdb_id="5OWX",
        output_path=corpus_source_fixture_path("5owx.cif"),
        download_format_priority=("cif", "pdb"),
    ),
    "5wsn-whole-structure": WholeStructureCorpusSource(
        case_id="5wsn-whole-structure",
        source_pdb_id="5WSN",
        output_path=corpus_source_fixture_path("5wsn.cif"),
        download_format_priority=("cif", "pdb"),
    ),
    "6d83-whole-structure": WholeStructureCorpusSource(
        case_id="6d83-whole-structure",
        source_pdb_id="6D83",
        output_path=corpus_source_fixture_path("6d83.cif"),
        download_format_priority=("cif", "pdb"),
    ),
    "6dmb-chain-a": WholeStructureCorpusSource(
        case_id="6dmb-chain-a",
        source_pdb_id="6DMB",
        output_path=corpus_source_fixture_path("6dmb.cif"),
        selected_source_chain_ids=("A",),
        download_format_priority=("cif", "pdb"),
    ),
    "6eyc-whole-structure": WholeStructureCorpusSource(
        case_id="6eyc-whole-structure",
        source_pdb_id="6EYC",
        output_path=corpus_source_fixture_path("pdb6eyc.ent"),
    ),
    "6gz3-chain-bj": WholeStructureCorpusSource(
        case_id="6gz3-chain-bj",
        source_pdb_id="6GZ3",
        output_path=corpus_source_fixture_path("6gz3.cif"),
        selected_source_chain_ids=("BJ",),
        download_format_priority=("cif", "pdb"),
    ),
    "6htx-whole-structure": WholeStructureCorpusSource(
        case_id="6htx-whole-structure",
        source_pdb_id="6HTX",
        output_path=corpus_source_fixture_path("pdb6htx.ent"),
    ),
    "6l54-chain-c": WholeStructureCorpusSource(
        case_id="6l54-chain-c",
        source_pdb_id="6L54",
        output_path=corpus_source_fixture_path("pdb6l54.ent"),
        selected_source_chain_ids=("C",),
    ),
    "6nbb-whole-structure": WholeStructureCorpusSource(
        case_id="6nbb-whole-structure",
        source_pdb_id="6NBB",
        output_path=corpus_source_fixture_path("pdb6nbb.ent"),
    ),
    "6z3r-chain-c": WholeStructureCorpusSource(
        case_id="6z3r-chain-c",
        source_pdb_id="6Z3R",
        output_path=corpus_source_fixture_path("pdb6z3r.ent"),
        selected_source_chain_ids=("C",),
    ),
}

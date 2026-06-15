"""Literature-backed local refinement fixture and benchmark case definitions."""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from protrepair.structure.labels import AtomRef, ResidueId

REFINEMENT_FIXTURE_ROOT = Path("tests/fixtures/pdb/refinement")


class RefinementBenchmarkExpectedOutcome(str, Enum):
    """Closed semantic expectation for one direct local-refinement benchmark."""

    POSITIVE_IMPROVEMENT = "positive_improvement"
    RUNNABLE_SUCCESS = "runnable_success"
    EXPECTED_NO_OP = "expected_no_op"
    EXPECTED_OPERATOR_SUPPORT_GAP = "expected_operator_support_gap"
    EXPECTED_ERROR = "expected_error"

    def expects_error(self) -> bool:
        """Return whether the case is expected to fail before usable output."""

        return self is RefinementBenchmarkExpectedOutcome.EXPECTED_ERROR

    def expects_no_op(self) -> bool:
        """Return whether the case is expected to return a no-op output."""

        return self in {
            RefinementBenchmarkExpectedOutcome.EXPECTED_NO_OP,
            RefinementBenchmarkExpectedOutcome.EXPECTED_OPERATOR_SUPPORT_GAP,
        }

    def expects_runnable_output(self) -> bool:
        """Return whether the case is expected to produce a usable output."""

        return not self.expects_error()

    def matches_status(self, status: str) -> bool:
        """Return whether one actual benchmark status satisfies this expectation."""

        if self is RefinementBenchmarkExpectedOutcome.EXPECTED_ERROR:
            return status == "error"
        if self.expects_no_op():
            return status == "no_op"

        return status == "success"


@dataclass(frozen=True, slots=True)
class RefinementFixtureSource:
    """One source-PDB extraction recipe for one local refinement fixture."""

    fixture_id: str
    source_pdb_id: str
    output_path: Path
    seed_residue_ids: tuple[ResidueId, ...]
    extraction_radius_angstrom: float
    description: str


@dataclass(frozen=True, slots=True)
class RefinementBenchmarkCase:
    """One benchmarkable local refinement case over one extracted fixture."""

    case_id: str
    title: str
    fixture_path: Path
    focus_residue_ids: tuple[ResidueId, ...]
    focus_atom_refs: tuple[AtomRef, ...]
    expected_outcome: RefinementBenchmarkExpectedOutcome
    description: str
    reference_fixture_path: Path | None = None

    @property
    def expected_success(self) -> bool:
        """Return the legacy runnable-output expectation projection."""

        return self.expected_outcome.expects_runnable_output()


def refinement_fixture_path(filename: str) -> Path:
    """Return one canonical extracted-fixture path."""

    return REFINEMENT_FIXTURE_ROOT / filename


REFINEMENT_FIXTURE_SOURCES: dict[str, RefinementFixtureSource] = {
    "1bkr-thr101": RefinementFixtureSource(
        fixture_id="1bkr-thr101",
        source_pdb_id="1BKR",
        output_path=refinement_fixture_path("1bkr_thr101_local.pdb"),
        seed_residue_ids=(ResidueId("A", 101),),
        extraction_radius_angstrom=8.0,
        description="Thr101 local misfit and clash neighborhood",
    ),
    "1bkr-his42": RefinementFixtureSource(
        fixture_id="1bkr-his42",
        source_pdb_id="1BKR",
        output_path=refinement_fixture_path("1bkr_his42_local.pdb"),
        seed_residue_ids=(ResidueId("A", 42),),
        extraction_radius_angstrom=8.0,
        description="His42 histidine-flip local contact neighborhood",
    ),
    "1lxa-his125": RefinementFixtureSource(
        fixture_id="1lxa-his125",
        source_pdb_id="1LXA",
        output_path=refinement_fixture_path("1lxa_his125_local.pdb"),
        seed_residue_ids=(ResidueId("A", 125),),
        extraction_radius_angstrom=8.0,
        description="1LXA apo LpxA His125 histidine-flip neighborhood",
    ),
    "1uaq-gln55": RefinementFixtureSource(
        fixture_id="1uaq-gln55",
        source_pdb_id="1UAQ",
        output_path=refinement_fixture_path("1uaq_gln55_local.pdb"),
        seed_residue_ids=(ResidueId("A", 55),),
        extraction_radius_angstrom=8.0,
        description="1UAQ Gln55 ligand-free amide-flip neighborhood",
    ),
    "1xgo-leu253": RefinementFixtureSource(
        fixture_id="1xgo-leu253",
        source_pdb_id="1XGO",
        output_path=refinement_fixture_path("1xgo_leu253_local.pdb"),
        seed_residue_ids=(ResidueId("A", 253),),
        extraction_radius_angstrom=8.0,
        description="1XGO distorted Leu253 helix neighborhood",
    ),
    "1xgs-leu253-reference": RefinementFixtureSource(
        fixture_id="1xgs-leu253-reference",
        source_pdb_id="1XGS",
        output_path=refinement_fixture_path("1xgs_leu253_reference_local.pdb"),
        seed_residue_ids=(ResidueId("A", 253),),
        extraction_radius_angstrom=8.0,
        description="1XGS higher-quality Leu253 local reference neighborhood",
    ),
    "3g8l-asn182": RefinementFixtureSource(
        fixture_id="3g8l-asn182",
        source_pdb_id="3G8L",
        output_path=refinement_fixture_path("3g8l_asn182_local.pdb"),
        seed_residue_ids=(ResidueId("A", 182),),
        extraction_radius_angstrom=8.0,
        description="3G8L Asn182 helix-cap contact neighborhood",
    ),
    "1jd0-gln92": RefinementFixtureSource(
        fixture_id="1jd0-gln92",
        source_pdb_id="1JD0",
        output_path=refinement_fixture_path("1jd0_gln92_local.pdb"),
        seed_residue_ids=(ResidueId("A", 92),),
        extraction_radius_angstrom=8.0,
        description="1JD0 Gln92 active-site amide-flip neighborhood",
    ),
    "1ywr-his148": RefinementFixtureSource(
        fixture_id="1ywr-his148",
        source_pdb_id="1YWR",
        output_path=refinement_fixture_path("1ywr_his148_local.pdb"),
        seed_residue_ids=(ResidueId("A", 148),),
        extraction_radius_angstrom=8.0,
        description="1YWR His148 active-site histidine-flip neighborhood",
    ),
    "1ywr-asn155": RefinementFixtureSource(
        fixture_id="1ywr-asn155",
        source_pdb_id="1YWR",
        output_path=refinement_fixture_path("1ywr_asn155_local.pdb"),
        seed_residue_ids=(ResidueId("A", 155),),
        extraction_radius_angstrom=8.0,
        description="1YWR Asn155 active-site amide-flip neighborhood",
    ),
    "2q6f-cys143-pje": RefinementFixtureSource(
        fixture_id="2q6f-cys143-pje",
        source_pdb_id="2Q6F",
        output_path=refinement_fixture_path("2q6f_cys143_pje_local.pdb"),
        seed_residue_ids=(ResidueId("A", 143), ResidueId("D", 5)),
        extraction_radius_angstrom=8.0,
        description="2Q6F covalent-link negative-control neighborhood",
    ),
    "4pep-sep68": RefinementFixtureSource(
        fixture_id="4pep-sep68",
        source_pdb_id="4PEP",
        output_path=refinement_fixture_path("4pep_sep68_local.pdb"),
        seed_residue_ids=(ResidueId("A", 68),),
        extraction_radius_angstrom=8.0,
        description="4PEP phosphoserine neighborhood in local corpus",
    ),
    "1uhg-sep68": RefinementFixtureSource(
        fixture_id="1uhg-sep68",
        source_pdb_id="1UHG",
        output_path=refinement_fixture_path("1uhg_sep68_local.pdb"),
        seed_residue_ids=(ResidueId("A", 68),),
        extraction_radius_angstrom=8.0,
        description="1UHG phosphoserine stress neighborhood from local corpus",
    ),
    "8pyr-sep164": RefinementFixtureSource(
        fixture_id="8pyr-sep164",
        source_pdb_id="8PYR",
        output_path=refinement_fixture_path("8pyr_sep164_local.pdb"),
        seed_residue_ids=(ResidueId("A", 164),),
        extraction_radius_angstrom=8.0,
        description="8PYR full-chain phosphoserine neighborhood",
    ),
    "8pyr-tpo170": RefinementFixtureSource(
        fixture_id="8pyr-tpo170",
        source_pdb_id="8PYR",
        output_path=refinement_fixture_path("8pyr_tpo170_local.pdb"),
        seed_residue_ids=(ResidueId("A", 170),),
        extraction_radius_angstrom=8.0,
        description="8PYR full-chain phosphothreonine neighborhood",
    ),
    "4jfx-ptr8": RefinementFixtureSource(
        fixture_id="4jfx-ptr8",
        source_pdb_id="4JFX",
        output_path=refinement_fixture_path("4jfx_ptr8_local.pdb"),
        seed_residue_ids=(ResidueId("P", 8),),
        extraction_radius_angstrom=8.0,
        description="4JFX phosphotyrosine peptide neighborhood",
    ),
}

REFINEMENT_BENCHMARK_CASES: dict[str, RefinementBenchmarkCase] = {
    "1bkr-thr101": RefinementBenchmarkCase(
        case_id="1bkr-thr101",
        title="1BKR Thr101 local side-chain misfit",
        fixture_path=REFINEMENT_FIXTURE_SOURCES["1bkr-thr101"].output_path,
        focus_residue_ids=(ResidueId("A", 101),),
        focus_atom_refs=(
            AtomRef(ResidueId("A", 101), "CB"),
            AtomRef(ResidueId("A", 101), "OG1"),
            AtomRef(ResidueId("A", 101), "CG2"),
        ),
        expected_outcome=RefinementBenchmarkExpectedOutcome.POSITIVE_IMPROVEMENT,
        description="Backward-fit Thr branch assignment and local clashes",
    ),
    "1bkr-his42": RefinementBenchmarkCase(
        case_id="1bkr-his42",
        title="1BKR His42 histidine-flip neighborhood",
        fixture_path=REFINEMENT_FIXTURE_SOURCES["1bkr-his42"].output_path,
        focus_residue_ids=(ResidueId("A", 42),),
        focus_atom_refs=(
            AtomRef(ResidueId("A", 42), "CB"),
            AtomRef(ResidueId("A", 42), "CG"),
            AtomRef(ResidueId("A", 42), "ND1"),
            AtomRef(ResidueId("A", 42), "CD2"),
            AtomRef(ResidueId("A", 42), "CE1"),
            AtomRef(ResidueId("A", 42), "NE2"),
        ),
        expected_outcome=RefinementBenchmarkExpectedOutcome.POSITIVE_IMPROVEMENT,
        description="All-atom-contact-driven histidine orientation cleanup case",
    ),
    "1lxa-his125": RefinementBenchmarkCase(
        case_id="1lxa-his125",
        title="1LXA His125 apo LpxA histidine flip",
        fixture_path=REFINEMENT_FIXTURE_SOURCES["1lxa-his125"].output_path,
        focus_residue_ids=(ResidueId("A", 125),),
        focus_atom_refs=(
            AtomRef(ResidueId("A", 125), "CB"),
            AtomRef(ResidueId("A", 125), "CG"),
            AtomRef(ResidueId("A", 125), "ND1"),
            AtomRef(ResidueId("A", 125), "CD2"),
            AtomRef(ResidueId("A", 125), "CE1"),
            AtomRef(ResidueId("A", 125), "NE2"),
        ),
        expected_outcome=RefinementBenchmarkExpectedOutcome.POSITIVE_IMPROVEMENT,
        description="Classical apo-LpxA histidine orientation cleanup case",
    ),
    "1uaq-gln55": RefinementBenchmarkCase(
        case_id="1uaq-gln55",
        title="1UAQ Gln55 ligand-free amide flip",
        fixture_path=REFINEMENT_FIXTURE_SOURCES["1uaq-gln55"].output_path,
        focus_residue_ids=(ResidueId("A", 55),),
        focus_atom_refs=(
            AtomRef(ResidueId("A", 55), "CB"),
            AtomRef(ResidueId("A", 55), "CG"),
            AtomRef(ResidueId("A", 55), "CD"),
            AtomRef(ResidueId("A", 55), "OE1"),
            AtomRef(ResidueId("A", 55), "NE2"),
        ),
        expected_outcome=RefinementBenchmarkExpectedOutcome.POSITIVE_IMPROVEMENT,
        description="Ligand-free glutamine amide orientation cleanup case",
    ),
    "1xgo-leu253": RefinementBenchmarkCase(
        case_id="1xgo-leu253",
        title="1XGO Leu253 with 1XGS local reference",
        fixture_path=REFINEMENT_FIXTURE_SOURCES["1xgo-leu253"].output_path,
        focus_residue_ids=(ResidueId("A", 253),),
        focus_atom_refs=(
            AtomRef(ResidueId("A", 253), "CB"),
            AtomRef(ResidueId("A", 253), "CG"),
            AtomRef(ResidueId("A", 253), "CD1"),
            AtomRef(ResidueId("A", 253), "CD2"),
        ),
        expected_outcome=RefinementBenchmarkExpectedOutcome.POSITIVE_IMPROVEMENT,
        description="Low-resolution Leu rotamer and local helix distortion case",
        reference_fixture_path=REFINEMENT_FIXTURE_SOURCES[
            "1xgs-leu253-reference"
        ].output_path,
    ),
    "3g8l-asn182": RefinementBenchmarkCase(
        case_id="3g8l-asn182",
        title="3G8L Asn182 helix-cap contact case",
        fixture_path=REFINEMENT_FIXTURE_SOURCES["3g8l-asn182"].output_path,
        focus_residue_ids=(ResidueId("A", 182),),
        focus_atom_refs=(
            AtomRef(ResidueId("A", 182), "CB"),
            AtomRef(ResidueId("A", 182), "CG"),
            AtomRef(ResidueId("A", 182), "OD1"),
            AtomRef(ResidueId("A", 182), "ND2"),
        ),
        expected_outcome=RefinementBenchmarkExpectedOutcome.POSITIVE_IMPROVEMENT,
        description="Helix-cap steric-contact cleanup case",
    ),
    "1jd0-gln92": RefinementBenchmarkCase(
        case_id="1jd0-gln92",
        title="1JD0 Gln92 active-site amide flip",
        fixture_path=REFINEMENT_FIXTURE_SOURCES["1jd0-gln92"].output_path,
        focus_residue_ids=(ResidueId("A", 92),),
        focus_atom_refs=(
            AtomRef(ResidueId("A", 92), "CB"),
            AtomRef(ResidueId("A", 92), "CG"),
            AtomRef(ResidueId("A", 92), "CD"),
            AtomRef(ResidueId("A", 92), "OE1"),
            AtomRef(ResidueId("A", 92), "NE2"),
        ),
        expected_outcome=RefinementBenchmarkExpectedOutcome.EXPECTED_NO_OP,
        description="Active-site glutamine amide orientation cleanup case",
    ),
    "1ywr-his148": RefinementBenchmarkCase(
        case_id="1ywr-his148",
        title="1YWR His148 active-site histidine flip",
        fixture_path=REFINEMENT_FIXTURE_SOURCES["1ywr-his148"].output_path,
        focus_residue_ids=(ResidueId("A", 148),),
        focus_atom_refs=(
            AtomRef(ResidueId("A", 148), "CB"),
            AtomRef(ResidueId("A", 148), "CG"),
            AtomRef(ResidueId("A", 148), "ND1"),
            AtomRef(ResidueId("A", 148), "CD2"),
            AtomRef(ResidueId("A", 148), "CE1"),
            AtomRef(ResidueId("A", 148), "NE2"),
        ),
        expected_outcome=RefinementBenchmarkExpectedOutcome.POSITIVE_IMPROVEMENT,
        description="Active-site histidine orientation cleanup case",
    ),
    "1ywr-asn155": RefinementBenchmarkCase(
        case_id="1ywr-asn155",
        title="1YWR Asn155 active-site amide flip",
        fixture_path=REFINEMENT_FIXTURE_SOURCES["1ywr-asn155"].output_path,
        focus_residue_ids=(ResidueId("A", 155),),
        focus_atom_refs=(
            AtomRef(ResidueId("A", 155), "CB"),
            AtomRef(ResidueId("A", 155), "CG"),
            AtomRef(ResidueId("A", 155), "OD1"),
            AtomRef(ResidueId("A", 155), "ND2"),
        ),
        expected_outcome=RefinementBenchmarkExpectedOutcome.EXPECTED_NO_OP,
        description="Active-site asparagine amide orientation cleanup case",
    ),
    "4pep-sep68": RefinementBenchmarkCase(
        case_id="4pep-sep68",
        title="4PEP SEP68 phosphoserine local case",
        fixture_path=REFINEMENT_FIXTURE_SOURCES["4pep-sep68"].output_path,
        focus_residue_ids=(ResidueId("A", 68),),
        focus_atom_refs=(
            AtomRef(ResidueId("A", 68), "CB"),
            AtomRef(ResidueId("A", 68), "OG"),
            AtomRef(ResidueId("A", 68), "P"),
            AtomRef(ResidueId("A", 68), "O1P"),
            AtomRef(ResidueId("A", 68), "O2P"),
            AtomRef(ResidueId("A", 68), "O3P"),
        ),
        expected_outcome=RefinementBenchmarkExpectedOutcome.RUNNABLE_SUCCESS,
        description="Real phosphoserine local refinement smoke case",
    ),
    "8pyr-tpo170": RefinementBenchmarkCase(
        case_id="8pyr-tpo170",
        title="8PYR TPO170 phosphothreonine local case",
        fixture_path=REFINEMENT_FIXTURE_SOURCES["8pyr-tpo170"].output_path,
        focus_residue_ids=(ResidueId("A", 170),),
        focus_atom_refs=(
            AtomRef(ResidueId("A", 170), "CB"),
            AtomRef(ResidueId("A", 170), "CG2"),
            AtomRef(ResidueId("A", 170), "OG1"),
            AtomRef(ResidueId("A", 170), "P"),
            AtomRef(ResidueId("A", 170), "O1P"),
            AtomRef(ResidueId("A", 170), "O2P"),
            AtomRef(ResidueId("A", 170), "O3P"),
        ),
        expected_outcome=RefinementBenchmarkExpectedOutcome.RUNNABLE_SUCCESS,
        description="Real phosphothreonine local refinement smoke case",
    ),
    "4jfx-ptr8": RefinementBenchmarkCase(
        case_id="4jfx-ptr8",
        title="4JFX PTR8 phosphotyrosine local case",
        fixture_path=REFINEMENT_FIXTURE_SOURCES["4jfx-ptr8"].output_path,
        focus_residue_ids=(ResidueId("P", 8),),
        focus_atom_refs=(
            AtomRef(ResidueId("P", 8), "CB"),
            AtomRef(ResidueId("P", 8), "CG"),
            AtomRef(ResidueId("P", 8), "CD1"),
            AtomRef(ResidueId("P", 8), "CD2"),
            AtomRef(ResidueId("P", 8), "CE1"),
            AtomRef(ResidueId("P", 8), "CE2"),
            AtomRef(ResidueId("P", 8), "CZ"),
            AtomRef(ResidueId("P", 8), "OH"),
            AtomRef(ResidueId("P", 8), "P"),
            AtomRef(ResidueId("P", 8), "O1P"),
            AtomRef(ResidueId("P", 8), "O2P"),
            AtomRef(ResidueId("P", 8), "O3P"),
        ),
        expected_outcome=RefinementBenchmarkExpectedOutcome.RUNNABLE_SUCCESS,
        description="Real phosphotyrosine local refinement smoke case",
    ),
    "2q6f-cys143-pje": RefinementBenchmarkCase(
        case_id="2q6f-cys143-pje",
        title="2Q6F covalent-link negative control",
        fixture_path=REFINEMENT_FIXTURE_SOURCES["2q6f-cys143-pje"].output_path,
        focus_residue_ids=(ResidueId("A", 143), ResidueId("D", 5)),
        focus_atom_refs=(
            AtomRef(ResidueId("A", 143), "SG"),
            AtomRef(ResidueId("D", 5), "C20"),
            AtomRef(ResidueId("D", 5), "C21"),
        ),
        expected_outcome=RefinementBenchmarkExpectedOutcome.EXPECTED_ERROR,
        description="Missing covalent-link chemistry should remain unsupported",
    ),
}


EXPLORATORY_REFINEMENT_FIXTURE_SOURCES: dict[str, RefinementFixtureSource] = {
    "2h6o-glycan-problematic": RefinementFixtureSource(
        fixture_id="2h6o-glycan-problematic",
        source_pdb_id="2H6O",
        output_path=refinement_fixture_path("2h6o_glycan_problematic_local.pdb"),
        seed_residue_ids=(ResidueId("A", 195),),
        extraction_radius_angstrom=12.0,
        description="2H6O retained glycan neighborhood around glycosylated Asn195",
    ),
    "3j5p-gln519-sidechain": RefinementFixtureSource(
        fixture_id="3j5p-gln519-sidechain",
        source_pdb_id="3J5P",
        output_path=refinement_fixture_path("3j5p_gln519_sidechain_local.pdb"),
        seed_residue_ids=(ResidueId("C", 519),),
        extraction_radius_angstrom=8.0,
        description="3J5P chain C Gln519 side-chain density-misfit neighborhood",
    ),
    "3j6b-terminal-helix-misthread": RefinementFixtureSource(
        fixture_id="3j6b-terminal-helix-misthread",
        source_pdb_id="3J6B",
        output_path=refinement_fixture_path("3j6b_terminal_helix_misthread_local.pdb"),
        seed_residue_ids=tuple(ResidueId("9", seq_num) for seq_num in range(228, 238)),
        extraction_radius_angstrom=8.0,
        description="3J6B chain 9 terminal-helix misthread segment Ser228-Val237",
    ),
    "3j9e-loop-backbone-error": RefinementFixtureSource(
        fixture_id="3j9e-loop-backbone-error",
        source_pdb_id="3J9E",
        output_path=refinement_fixture_path("3j9e_loop_backbone_error_local.pdb"),
        seed_residue_ids=tuple(ResidueId("D", seq_num) for seq_num in range(370, 376)),
        extraction_radius_angstrom=8.0,
        description="3J9E chain D incorrect loop/backbone segment residues 370-375",
    ),
    "5k12-tyr471-sidechain": RefinementFixtureSource(
        fixture_id="5k12-tyr471-sidechain",
        source_pdb_id="5K12",
        output_path=refinement_fixture_path("5k12_tyr471_sidechain_local.pdb"),
        seed_residue_ids=(ResidueId("F", 471),),
        extraction_radius_angstrom=8.0,
        description="5K12 chain F Tyr471 cryo-EM side-chain correction neighborhood",
    ),
    "5xm5-metal-sites": RefinementFixtureSource(
        fixture_id="5xm5-metal-sites",
        source_pdb_id="5XM5",
        output_path=refinement_fixture_path("5xm5_metal_sites_local.pdb"),
        seed_residue_ids=(ResidueId("A", 201), ResidueId("B", 202)),
        extraction_radius_angstrom=8.0,
        description=(
            "5XM5 paired zinc-site admissibility neighborhood around A201 and B202"
        ),
    ),
    "7s9d-prestin-segment": RefinementFixtureSource(
        fixture_id="7s9d-prestin-segment",
        source_pdb_id="7S9D",
        output_path=refinement_fixture_path("7s9d_prestin_segment_local.pdb"),
        seed_residue_ids=tuple(ResidueId("A", seq_num) for seq_num in range(615, 637)),
        extraction_radius_angstrom=8.0,
        description="7S9D chain A prestin error segment residues 615-636",
    ),
}


EXPLORATORY_REFINEMENT_BENCHMARK_CASES: dict[str, RefinementBenchmarkCase] = {
    "3j5p-gln519-sidechain": RefinementBenchmarkCase(
        case_id="3j5p-gln519-sidechain",
        title="3J5P Gln519 low-resolution side-chain orientation",
        fixture_path=EXPLORATORY_REFINEMENT_FIXTURE_SOURCES[
            "3j5p-gln519-sidechain"
        ].output_path,
        focus_residue_ids=(ResidueId("C", 519),),
        focus_atom_refs=(
            AtomRef(ResidueId("C", 519), "CB"),
            AtomRef(ResidueId("C", 519), "CG"),
            AtomRef(ResidueId("C", 519), "CD"),
            AtomRef(ResidueId("C", 519), "OE1"),
            AtomRef(ResidueId("C", 519), "NE2"),
        ),
        expected_outcome=(
            RefinementBenchmarkExpectedOutcome.EXPECTED_OPERATOR_SUPPORT_GAP
        ),
        description=(
            "Literature-backed low-resolution GLN side-chain density-misfit "
            "case. Current clash/geometry/RDKit-readability acceptance axes do "
            "not provide a positive signal for choosing a new orientation, so "
            "plain local FF should remain a stable no-op until a density- or "
            "reference-aware side-chain operator exists."
        ),
    ),
    "5k12-tyr471-sidechain": RefinementBenchmarkCase(
        case_id="5k12-tyr471-sidechain",
        title="5K12 Tyr471 low-resolution side-chain orientation",
        fixture_path=EXPLORATORY_REFINEMENT_FIXTURE_SOURCES[
            "5k12-tyr471-sidechain"
        ].output_path,
        focus_residue_ids=(ResidueId("F", 471),),
        focus_atom_refs=(
            AtomRef(ResidueId("F", 471), "CB"),
            AtomRef(ResidueId("F", 471), "CG"),
            AtomRef(ResidueId("F", 471), "CD1"),
            AtomRef(ResidueId("F", 471), "CD2"),
            AtomRef(ResidueId("F", 471), "CE1"),
            AtomRef(ResidueId("F", 471), "CE2"),
            AtomRef(ResidueId("F", 471), "CZ"),
            AtomRef(ResidueId("F", 471), "OH"),
        ),
        expected_outcome=(
            RefinementBenchmarkExpectedOutcome.EXPECTED_OPERATOR_SUPPORT_GAP
        ),
        description=(
            "Literature-backed low-resolution TYR side-chain density-misfit "
            "case. Current local FF and packing-free rotamer support cannot "
            "rank the density-preferred side-chain orientation, so the "
            "benchmark should distinguish the stable no-op from ordinary "
            "solver failure."
        ),
    ),
}


ALL_REFINEMENT_BENCHMARK_CASES: dict[str, RefinementBenchmarkCase] = {
    **REFINEMENT_BENCHMARK_CASES,
    **EXPLORATORY_REFINEMENT_BENCHMARK_CASES,
}


ALL_REFINEMENT_FIXTURE_SOURCES: dict[str, RefinementFixtureSource] = {
    **REFINEMENT_FIXTURE_SOURCES,
    **EXPLORATORY_REFINEMENT_FIXTURE_SOURCES,
}

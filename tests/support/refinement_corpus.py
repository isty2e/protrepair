"""Corpus metadata over representative and local refinement benchmark cases."""

from dataclasses import dataclass
from enum import Enum

from protrepair.transformer.refinement.spec import RefinementOperatorFamily
from tests.support.refinement_cases import REFINEMENT_BENCHMARK_CASES
from tests.support.representative_cases import REPRESENTATIVE_CASES


class CorpusSurface(str, Enum):
    """Top-level execution surface represented by one corpus case."""

    WHOLE_STRUCTURE_REPRESENTATIVE = "whole_structure_representative"
    LOCAL_REFINEMENT_BENCHMARK = "local_refinement_benchmark"


class CorpusEvaluationMode(str, Enum):
    """Primary evaluation semantics carried by one corpus case."""

    OPERATOR_BENCHMARK = "operator_benchmark"
    CHEMISTRY_ADMISSIBILITY = "chemistry_admissibility"


class CorpusPanel(str, Enum):
    """Execution panel membership used to stage corpus breadth."""

    ACTIVE = "active"
    EXTENDED = "extended"
    EXPLORATORY = "exploratory"


class CorpusExecutionPolicy(str, Enum):
    """Current execution expectation for one tracked corpus case."""

    MUST_PASS = "must_pass"
    TRACKING_ONLY = "tracking_only"


class CorpusEvidenceBasis(str, Enum):
    """Authority basis behind one corpus case."""

    INTERNAL_REGRESSION = "internal_regression"
    LITERATURE_BACKED = "literature_backed"
    KNOWN_PROBLEMATIC_DEPOSITION = "known_problematic_deposition"


class StructureDeterminationMethod(str, Enum):
    """Experimental structure-determination method for one corpus case."""

    X_RAY_DIFFRACTION = "x_ray_diffraction"
    CRYO_EM = "cryo_em"
    UNKNOWN = "unknown"


class CorpusCoverageTag(str, Enum):
    """Closed coverage tags used to query corpus breadth and panel gaps."""

    AMIDE_ORIENTATION = "amide_orientation"
    BACKBONE_DRIVEN_CLASH = "backbone_driven_clash"
    BACKBONE_MISTHREAD = "backbone_misthread"
    COFACTOR_KEPT = "cofactor_kept"
    COVALENT_NEGATIVE_CONTROL = "covalent_negative_control"
    GLYCAN_KEPT = "glycan_kept"
    HISTIDINE_PROTOMER = "histidine_protomer"
    HYDROGENATION_STRESS = "hydrogenation_stress"
    LOW_RESOLUTION = "low_resolution"
    METAL_SITE = "metal_site"
    MULTI_RESIDUE_JOINT_CORRECTION = "multi_residue_joint_correction"
    NO_CONECT_INGEST_SENSITIVE = "no_conect_ingest_sensitive"
    NOT_APPLICABLE_CHEMISTRY = "not_applicable_chemistry"
    PERTURB_REFERENCE_PAIRED = "perturb_reference_paired"
    PHOSPHO = "phospho"
    RETAINED_NON_POLYMER_KEPT = "retained_non_polymer_kept"
    SIDECHAIN_ORIENTATION = "sidechain_orientation"


class CorpusTrackingPressure(str, Enum):
    """Primary pressure that keeps one non-gating case in the tracking corpus."""

    CHEMISTRY_BLOCKER = "chemistry_blocker"
    OPERATOR_CAPABILITY = "operator_capability"
    SCALING_GAP = "scaling_gap"
    SCOPING_GAP = "scoping_gap"


@dataclass(frozen=True, slots=True)
class CorpusCaseMetadata:
    """Orthogonal metadata describing one corpus case."""

    case_id: str
    surface: CorpusSurface
    panels: frozenset[CorpusPanel]
    evidence_basis: CorpusEvidenceBasis
    structure_method: StructureDeterminationMethod
    resolution_angstrom: float | None
    coverage_tags: frozenset[CorpusCoverageTag]
    evaluation_mode: CorpusEvaluationMode = CorpusEvaluationMode.OPERATOR_BENCHMARK
    execution_policy: CorpusExecutionPolicy = CorpusExecutionPolicy.MUST_PASS
    tracking_pressure: CorpusTrackingPressure | None = None
    required_operator_family: RefinementOperatorFamily | None = None
    literature_references: tuple[str, ...] = ()


REPRESENTATIVE_CORPUS_METADATA: dict[str, CorpusCaseMetadata] = {
    "1aar-heavy-chain-1": CorpusCaseMetadata(
        case_id="1aar-heavy-chain-1",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.ACTIVE}),
        evidence_basis=CorpusEvidenceBasis.INTERNAL_REGRESSION,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset(),
    ),
    "1afc-hydrogen-his-protonated": CorpusCaseMetadata(
        case_id="1afc-hydrogen-his-protonated",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.ACTIVE}),
        evidence_basis=CorpusEvidenceBasis.INTERNAL_REGRESSION,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.HISTIDINE_PROTOMER,
                CorpusCoverageTag.HYDROGENATION_STRESS,
                CorpusCoverageTag.MULTI_RESIDUE_JOINT_CORRECTION,
                CorpusCoverageTag.NO_CONECT_INGEST_SENSITIVE,
            }
        ),
    ),
    "1aho-heavy-default": CorpusCaseMetadata(
        case_id="1aho-heavy-default",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.ACTIVE}),
        evidence_basis=CorpusEvidenceBasis.INTERNAL_REGRESSION,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset(),
    ),
    "1aho-hydrogen-default": CorpusCaseMetadata(
        case_id="1aho-hydrogen-default",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.ACTIVE}),
        evidence_basis=CorpusEvidenceBasis.INTERNAL_REGRESSION,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset({CorpusCoverageTag.HYDROGENATION_STRESS}),
    ),
    "1cjc-heavy-keep-ligand": CorpusCaseMetadata(
        case_id="1cjc-heavy-keep-ligand",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.ACTIVE}),
        evidence_basis=CorpusEvidenceBasis.INTERNAL_REGRESSION,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.COFACTOR_KEPT,
                CorpusCoverageTag.RETAINED_NON_POLYMER_KEPT,
            }
        ),
    ),
    "1cjc-hydrogen-keep-ligand": CorpusCaseMetadata(
        case_id="1cjc-hydrogen-keep-ligand",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.ACTIVE}),
        evidence_basis=CorpusEvidenceBasis.INTERNAL_REGRESSION,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.COFACTOR_KEPT,
                CorpusCoverageTag.HYDROGENATION_STRESS,
                CorpusCoverageTag.NO_CONECT_INGEST_SENSITIVE,
                CorpusCoverageTag.RETAINED_NON_POLYMER_KEPT,
            }
        ),
    ),
}


REFINEMENT_CORPUS_METADATA: dict[str, CorpusCaseMetadata] = {
    "1bkr-his42": CorpusCaseMetadata(
        case_id="1bkr-his42",
        surface=CorpusSurface.LOCAL_REFINEMENT_BENCHMARK,
        panels=frozenset({CorpusPanel.ACTIVE}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.HISTIDINE_PROTOMER,
                CorpusCoverageTag.SIDECHAIN_ORIENTATION,
            }
        ),
    ),
    "1bkr-thr101": CorpusCaseMetadata(
        case_id="1bkr-thr101",
        surface=CorpusSurface.LOCAL_REFINEMENT_BENCHMARK,
        panels=frozenset({CorpusPanel.ACTIVE}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset({CorpusCoverageTag.SIDECHAIN_ORIENTATION}),
    ),
    "1jd0-gln92": CorpusCaseMetadata(
        case_id="1jd0-gln92",
        surface=CorpusSurface.LOCAL_REFINEMENT_BENCHMARK,
        panels=frozenset({CorpusPanel.EXTENDED}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.AMIDE_ORIENTATION,
                CorpusCoverageTag.SIDECHAIN_ORIENTATION,
            }
        ),
    ),
    "1lxa-his125": CorpusCaseMetadata(
        case_id="1lxa-his125",
        surface=CorpusSurface.LOCAL_REFINEMENT_BENCHMARK,
        panels=frozenset({CorpusPanel.EXTENDED}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.HISTIDINE_PROTOMER,
                CorpusCoverageTag.SIDECHAIN_ORIENTATION,
            }
        ),
    ),
    "1uaq-gln55": CorpusCaseMetadata(
        case_id="1uaq-gln55",
        surface=CorpusSurface.LOCAL_REFINEMENT_BENCHMARK,
        panels=frozenset({CorpusPanel.ACTIVE}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.AMIDE_ORIENTATION,
                CorpusCoverageTag.SIDECHAIN_ORIENTATION,
            }
        ),
    ),
    "1xgo-leu253": CorpusCaseMetadata(
        case_id="1xgo-leu253",
        surface=CorpusSurface.LOCAL_REFINEMENT_BENCHMARK,
        panels=frozenset({CorpusPanel.ACTIVE}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.BACKBONE_DRIVEN_CLASH,
                CorpusCoverageTag.PERTURB_REFERENCE_PAIRED,
            }
        ),
    ),
    "1ywr-asn155": CorpusCaseMetadata(
        case_id="1ywr-asn155",
        surface=CorpusSurface.LOCAL_REFINEMENT_BENCHMARK,
        panels=frozenset({CorpusPanel.EXTENDED}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.AMIDE_ORIENTATION,
                CorpusCoverageTag.SIDECHAIN_ORIENTATION,
            }
        ),
    ),
    "1ywr-his148": CorpusCaseMetadata(
        case_id="1ywr-his148",
        surface=CorpusSurface.LOCAL_REFINEMENT_BENCHMARK,
        panels=frozenset({CorpusPanel.EXTENDED}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.HISTIDINE_PROTOMER,
                CorpusCoverageTag.SIDECHAIN_ORIENTATION,
            }
        ),
    ),
    "2q6f-cys143-pje": CorpusCaseMetadata(
        case_id="2q6f-cys143-pje",
        surface=CorpusSurface.LOCAL_REFINEMENT_BENCHMARK,
        panels=frozenset({CorpusPanel.ACTIVE}),
        evidence_basis=CorpusEvidenceBasis.KNOWN_PROBLEMATIC_DEPOSITION,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.COVALENT_NEGATIVE_CONTROL,
                CorpusCoverageTag.MULTI_RESIDUE_JOINT_CORRECTION,
                CorpusCoverageTag.NOT_APPLICABLE_CHEMISTRY,
                CorpusCoverageTag.RETAINED_NON_POLYMER_KEPT,
            }
        ),
        evaluation_mode=CorpusEvaluationMode.CHEMISTRY_ADMISSIBILITY,
        tracking_pressure=CorpusTrackingPressure.CHEMISTRY_BLOCKER,
    ),
    "3g8l-asn182": CorpusCaseMetadata(
        case_id="3g8l-asn182",
        surface=CorpusSurface.LOCAL_REFINEMENT_BENCHMARK,
        panels=frozenset({CorpusPanel.ACTIVE}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.BACKBONE_DRIVEN_CLASH,
                CorpusCoverageTag.MULTI_RESIDUE_JOINT_CORRECTION,
            }
        ),
    ),
    "4jfx-ptr8": CorpusCaseMetadata(
        case_id="4jfx-ptr8",
        surface=CorpusSurface.LOCAL_REFINEMENT_BENCHMARK,
        panels=frozenset({CorpusPanel.EXTENDED}),
        evidence_basis=CorpusEvidenceBasis.INTERNAL_REGRESSION,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset({CorpusCoverageTag.PHOSPHO}),
    ),
    "4pep-sep68": CorpusCaseMetadata(
        case_id="4pep-sep68",
        surface=CorpusSurface.LOCAL_REFINEMENT_BENCHMARK,
        panels=frozenset({CorpusPanel.EXTENDED}),
        evidence_basis=CorpusEvidenceBasis.INTERNAL_REGRESSION,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset({CorpusCoverageTag.PHOSPHO}),
    ),
    "8pyr-tpo170": CorpusCaseMetadata(
        case_id="8pyr-tpo170",
        surface=CorpusSurface.LOCAL_REFINEMENT_BENCHMARK,
        panels=frozenset({CorpusPanel.EXTENDED}),
        evidence_basis=CorpusEvidenceBasis.INTERNAL_REGRESSION,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset({CorpusCoverageTag.PHOSPHO}),
    ),
}


EXPLORATORY_CORPUS_METADATA: dict[str, CorpusCaseMetadata] = {
    "2h6o-glycan-problematic": CorpusCaseMetadata(
        case_id="2h6o-glycan-problematic",
        surface=CorpusSurface.LOCAL_REFINEMENT_BENCHMARK,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.GLYCAN_KEPT,
                CorpusCoverageTag.NOT_APPLICABLE_CHEMISTRY,
                CorpusCoverageTag.RETAINED_NON_POLYMER_KEPT,
            }
        ),
        evaluation_mode=CorpusEvaluationMode.CHEMISTRY_ADMISSIBILITY,
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.CHEMISTRY_BLOCKER,
        literature_references=(
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC7554661/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC6616339/",
        ),
    ),
    "3j5p-gln519-sidechain": CorpusCaseMetadata(
        case_id="3j5p-gln519-sidechain",
        surface=CorpusSurface.LOCAL_REFINEMENT_BENCHMARK,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=3.27,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.LOW_RESOLUTION,
                CorpusCoverageTag.SIDECHAIN_ORIENTATION,
            }
        ),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCOPING_GAP,
        literature_references=(
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC4589481/",
        ),
    ),
    "3j6b-terminal-helix-misthread": CorpusCaseMetadata(
        case_id="3j6b-terminal-helix-misthread",
        surface=CorpusSurface.LOCAL_REFINEMENT_BENCHMARK,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=3.2,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.BACKBONE_MISTHREAD,
                CorpusCoverageTag.LOW_RESOLUTION,
                CorpusCoverageTag.MULTI_RESIDUE_JOINT_CORRECTION,
            }
        ),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.OPERATOR_CAPABILITY,
        required_operator_family=(
            RefinementOperatorFamily.BACKBONE_WINDOW_REFINEMENT
        ),
        literature_references=(
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC10024464/",
        ),
    ),
    "3j7h-binding-site-backbone": CorpusCaseMetadata(
        case_id="3j7h-binding-site-backbone",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.BACKBONE_DRIVEN_CLASH,
                CorpusCoverageTag.BACKBONE_MISTHREAD,
                CorpusCoverageTag.LOW_RESOLUTION,
                CorpusCoverageTag.NO_CONECT_INGEST_SENSITIVE,
            }
        ),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCALING_GAP,
        literature_references=(
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC4990421/",
        ),
    ),
    "3j9e-loop-backbone-error": CorpusCaseMetadata(
        case_id="3j9e-loop-backbone-error",
        surface=CorpusSurface.LOCAL_REFINEMENT_BENCHMARK,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.BACKBONE_DRIVEN_CLASH,
                CorpusCoverageTag.BACKBONE_MISTHREAD,
                CorpusCoverageTag.LOW_RESOLUTION,
            }
        ),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.OPERATOR_CAPABILITY,
        required_operator_family=(
            RefinementOperatorFamily.BACKBONE_WINDOW_REFINEMENT
        ),
        literature_references=(
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC10330749/",
        ),
    ),
    "5k12-tyr471-sidechain": CorpusCaseMetadata(
        case_id="5k12-tyr471-sidechain",
        surface=CorpusSurface.LOCAL_REFINEMENT_BENCHMARK,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.LOW_RESOLUTION,
                CorpusCoverageTag.SIDECHAIN_ORIENTATION,
            }
        ),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCOPING_GAP,
        literature_references=(
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC7787109/",
        ),
    ),
    "5xm5-metal-sites": CorpusCaseMetadata(
        case_id="5xm5-metal-sites",
        surface=CorpusSurface.LOCAL_REFINEMENT_BENCHMARK,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.METAL_SITE,
                CorpusCoverageTag.NOT_APPLICABLE_CHEMISTRY,
            }
        ),
        evaluation_mode=CorpusEvaluationMode.CHEMISTRY_ADMISSIBILITY,
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.CHEMISTRY_BLOCKER,
        literature_references=(
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC10321142/",
        ),
    ),
    "7s9d-prestin-segment": CorpusCaseMetadata(
        case_id="7s9d-prestin-segment",
        surface=CorpusSurface.LOCAL_REFINEMENT_BENCHMARK,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.BACKBONE_DRIVEN_CLASH,
                CorpusCoverageTag.BACKBONE_MISTHREAD,
                CorpusCoverageTag.LOW_RESOLUTION,
                CorpusCoverageTag.MULTI_RESIDUE_JOINT_CORRECTION,
            }
        ),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.OPERATOR_CAPABILITY,
        required_operator_family=(
            RefinementOperatorFamily.BACKBONE_WINDOW_REFINEMENT
        ),
        literature_references=(
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC10330749/",
        ),
    ),
    "2dn2-whole-structure": CorpusCaseMetadata(
        case_id="2dn2-whole-structure",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset({CorpusCoverageTag.SIDECHAIN_ORIENTATION}),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCOPING_GAP,
        literature_references=("https://pmc.ncbi.nlm.nih.gov/articles/PMC10833350/",),
    ),
    "2qls-whole-structure": CorpusCaseMetadata(
        case_id="2qls-whole-structure",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=3.5,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.LOW_RESOLUTION,
                CorpusCoverageTag.SIDECHAIN_ORIENTATION,
            }
        ),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCOPING_GAP,
        literature_references=("https://pmc.ncbi.nlm.nih.gov/articles/PMC10833350/",),
    ),
    "2z62-whole-structure": CorpusCaseMetadata(
        case_id="2z62-whole-structure",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.GLYCAN_KEPT,
                CorpusCoverageTag.NOT_APPLICABLE_CHEMISTRY,
                CorpusCoverageTag.RETAINED_NON_POLYMER_KEPT,
            }
        ),
        evaluation_mode=CorpusEvaluationMode.CHEMISTRY_ADMISSIBILITY,
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.CHEMISTRY_BLOCKER,
        literature_references=(
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC7554661/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC6616339/",
        ),
    ),
    "3i1n-whole-structure": CorpusCaseMetadata(
        case_id="3i1n-whole-structure",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset({CorpusCoverageTag.BACKBONE_MISTHREAD}),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCALING_GAP,
    ),
    "3j4p-whole-structure": CorpusCaseMetadata(
        case_id="3j4p-whole-structure",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=4.8,
        coverage_tags=frozenset({CorpusCoverageTag.LOW_RESOLUTION}),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCOPING_GAP,
        literature_references=("https://pmc.ncbi.nlm.nih.gov/articles/PMC7787109/",),
    ),
    "3j7l-whole-structure": CorpusCaseMetadata(
        case_id="3j7l-whole-structure",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.BACKBONE_MISTHREAD,
                CorpusCoverageTag.LOW_RESOLUTION,
            }
        ),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCOPING_GAP,
        literature_references=("https://pmc.ncbi.nlm.nih.gov/articles/PMC6163098/",),
    ),
    "3j9z-chain-sb": CorpusCaseMetadata(
        case_id="3j9z-chain-sb",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=3.6,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.BACKBONE_MISTHREAD,
                CorpusCoverageTag.LOW_RESOLUTION,
                CorpusCoverageTag.MULTI_RESIDUE_JOINT_CORRECTION,
            }
        ),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCOPING_GAP,
        literature_references=("https://journals.iucr.org/m/issues/2024/06/00/lz5074/",),
    ),
    "3ja8-whole-structure": CorpusCaseMetadata(
        case_id="3ja8-whole-structure",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=3.8,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.BACKBONE_MISTHREAD,
                CorpusCoverageTag.LOW_RESOLUTION,
            }
        ),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCALING_GAP,
        literature_references=(
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC10833350/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC6163098/",
        ),
    ),
    "4gd1-whole-structure": CorpusCaseMetadata(
        case_id="4gd1-whole-structure",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.X_RAY_DIFFRACTION,
        resolution_angstrom=None,
        coverage_tags=frozenset({CorpusCoverageTag.BACKBONE_MISTHREAD}),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCALING_GAP,
    ),
    "5n9y-whole-structure": CorpusCaseMetadata(
        case_id="5n9y-whole-structure",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=4.2,
        coverage_tags=frozenset({CorpusCoverageTag.LOW_RESOLUTION}),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCOPING_GAP,
        literature_references=("https://pubmed.ncbi.nlm.nih.gov/35831370/",),
    ),
    "5owx-whole-structure": CorpusCaseMetadata(
        case_id="5owx-whole-structure",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=5.2,
        coverage_tags=frozenset({CorpusCoverageTag.LOW_RESOLUTION}),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCOPING_GAP,
        literature_references=("https://pubmed.ncbi.nlm.nih.gov/35831370/",),
    ),
    "5wsn-whole-structure": CorpusCaseMetadata(
        case_id="5wsn-whole-structure",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=4.3,
        coverage_tags=frozenset({CorpusCoverageTag.LOW_RESOLUTION}),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCOPING_GAP,
        literature_references=("https://pubmed.ncbi.nlm.nih.gov/35831370/",),
    ),
    "6d83-whole-structure": CorpusCaseMetadata(
        case_id="6d83-whole-structure",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=4.27,
        coverage_tags=frozenset({CorpusCoverageTag.LOW_RESOLUTION}),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCOPING_GAP,
        literature_references=("https://pubmed.ncbi.nlm.nih.gov/35831370/",),
    ),
    "6dmb-chain-a": CorpusCaseMetadata(
        case_id="6dmb-chain-a",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=3.9,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.BACKBONE_MISTHREAD,
                CorpusCoverageTag.LOW_RESOLUTION,
                CorpusCoverageTag.MULTI_RESIDUE_JOINT_CORRECTION,
            }
        ),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCOPING_GAP,
        literature_references=("https://pmc.ncbi.nlm.nih.gov/articles/PMC10330749/",),
    ),
    "6eyc-whole-structure": CorpusCaseMetadata(
        case_id="6eyc-whole-structure",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.BACKBONE_MISTHREAD,
                CorpusCoverageTag.LOW_RESOLUTION,
            }
        ),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCALING_GAP,
        literature_references=(
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC10833350/",
            "https://pmc.ncbi.nlm.nih.gov/articles/PMC6163098/",
        ),
    ),
    "6gz3-chain-bj": CorpusCaseMetadata(
        case_id="6gz3-chain-bj",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=3.6,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.BACKBONE_MISTHREAD,
                CorpusCoverageTag.LOW_RESOLUTION,
                CorpusCoverageTag.MULTI_RESIDUE_JOINT_CORRECTION,
            }
        ),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCOPING_GAP,
        literature_references=("https://journals.iucr.org/m/issues/2024/06/00/lz5074/",),
    ),
    "6htx-whole-structure": CorpusCaseMetadata(
        case_id="6htx-whole-structure",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=None,
        coverage_tags=frozenset({CorpusCoverageTag.LOW_RESOLUTION}),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCOPING_GAP,
        literature_references=("https://pmc.ncbi.nlm.nih.gov/articles/PMC7787109/",),
    ),
    "6l54-chain-c": CorpusCaseMetadata(
        case_id="6l54-chain-c",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.KNOWN_PROBLEMATIC_DEPOSITION,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.BACKBONE_DRIVEN_CLASH,
                CorpusCoverageTag.LOW_RESOLUTION,
                CorpusCoverageTag.NO_CONECT_INGEST_SENSITIVE,
            }
        ),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCOPING_GAP,
        literature_references=("https://pmc.ncbi.nlm.nih.gov/articles/PMC10024464/",),
    ),
    "6nbb-whole-structure": CorpusCaseMetadata(
        case_id="6nbb-whole-structure",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.METAL_SITE,
                CorpusCoverageTag.RETAINED_NON_POLYMER_KEPT,
            }
        ),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCOPING_GAP,
        literature_references=("https://pmc.ncbi.nlm.nih.gov/articles/PMC8167059/",),
    ),
    "6z3r-chain-c": CorpusCaseMetadata(
        case_id="6z3r-chain-c",
        surface=CorpusSurface.WHOLE_STRUCTURE_REPRESENTATIVE,
        panels=frozenset({CorpusPanel.EXPLORATORY}),
        evidence_basis=CorpusEvidenceBasis.LITERATURE_BACKED,
        structure_method=StructureDeterminationMethod.CRYO_EM,
        resolution_angstrom=None,
        coverage_tags=frozenset(
            {
                CorpusCoverageTag.BACKBONE_DRIVEN_CLASH,
                CorpusCoverageTag.LOW_RESOLUTION,
                CorpusCoverageTag.NO_CONECT_INGEST_SENSITIVE,
            }
        ),
        execution_policy=CorpusExecutionPolicy.TRACKING_ONLY,
        tracking_pressure=CorpusTrackingPressure.SCOPING_GAP,
        literature_references=("https://pmc.ncbi.nlm.nih.gov/articles/PMC10024464/",),
    ),
}


ALL_CORPUS_METADATA: dict[str, CorpusCaseMetadata] = {
    **REPRESENTATIVE_CORPUS_METADATA,
    **REFINEMENT_CORPUS_METADATA,
    **EXPLORATORY_CORPUS_METADATA,
}


def corpus_case_ids_for_panel(panel: CorpusPanel) -> tuple[str, ...]:
    """Return corpus case ids assigned to one execution panel."""

    return tuple(
        case_id
        for case_id, metadata in sorted(ALL_CORPUS_METADATA.items())
        if panel in metadata.panels
    )


def corpus_case_ids_for_execution_policy(
    execution_policy: CorpusExecutionPolicy,
) -> tuple[str, ...]:
    """Return corpus case ids sharing one execution-policy classification."""

    return tuple(
        case_id
        for case_id, metadata in sorted(ALL_CORPUS_METADATA.items())
        if metadata.execution_policy is execution_policy
    )


def corpus_case_ids_for_evaluation_mode(
    evaluation_mode: CorpusEvaluationMode,
) -> tuple[str, ...]:
    """Return corpus case ids sharing one evaluation semantics."""

    return tuple(
        case_id
        for case_id, metadata in sorted(ALL_CORPUS_METADATA.items())
        if metadata.evaluation_mode is evaluation_mode
    )


def corpus_case_ids_for_tag(tag: CorpusCoverageTag) -> tuple[str, ...]:
    """Return corpus case ids covering one named validation axis."""

    return tuple(
        case_id
        for case_id, metadata in sorted(ALL_CORPUS_METADATA.items())
        if tag in metadata.coverage_tags
    )


def corpus_case_ids_for_tracking_pressure(
    tracking_pressure: CorpusTrackingPressure,
) -> tuple[str, ...]:
    """Return tracking-only corpus cases grouped by their primary pressure."""

    return tuple(
        case_id
        for case_id, metadata in sorted(ALL_CORPUS_METADATA.items())
        if metadata.tracking_pressure is tracking_pressure
    )


def corpus_case_ids_for_required_operator_family(
    operator_family: RefinementOperatorFamily,
) -> tuple[str, ...]:
    """Return corpus cases requiring one unsupported operator family."""

    return tuple(
        case_id
        for case_id, metadata in sorted(ALL_CORPUS_METADATA.items())
        if metadata.required_operator_family is operator_family
    )


def uncovered_tags_for_panel(
    panel: CorpusPanel,
    *,
    tags: frozenset[CorpusCoverageTag],
) -> frozenset[CorpusCoverageTag]:
    """Return requested coverage tags not represented in one panel."""

    covered_tags = set()
    for metadata in ALL_CORPUS_METADATA.values():
        if panel not in metadata.panels:
            continue
        covered_tags.update(metadata.coverage_tags)

    return frozenset(tag for tag in tags if tag not in covered_tags)


assert set(REPRESENTATIVE_CORPUS_METADATA) == set(REPRESENTATIVE_CASES)
assert set(REFINEMENT_CORPUS_METADATA) == set(REFINEMENT_BENCHMARK_CASES)

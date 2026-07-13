"""Topology-preserving pre-untangle for parser-witness local refinement."""

from collections.abc import Callable
from dataclasses import dataclass, field

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.diagnostics.clashes import (
    ClashDetectionBasis,
    prepare_clash_detection_basis,
)
from protrepair.diagnostics.near_covalent import (
    NearCovalentContactBasis,
    prepare_near_covalent_contact_basis,
)
from protrepair.diagnostics.parser_readability import (
    RDKitKnownBondLookup,
    RDKitProximityBondCluster,
    RDKitProximityBondWitness,
    prepare_rdkit_no_conect_known_bond_lookup,
    rdkit_no_conect_extra_proximity_bond_clusters,
)
from protrepair.geometry import Vec3
from protrepair.io.pdb_projection import (
    RDKitNoConectPDBBlockProjector,
    prepare_rdkit_no_conect_pdb_block_projector,
)
from protrepair.scope import ResidueSetScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.atom_input import AtomInput
from protrepair.transformer.base import DeterministicContextOperation
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.discrete import (
    parser_witness_pre_untangle_materialization as parser_materialization,
)
from protrepair.transformer.discrete.parser_witness_pre_untangle_scoring import (
    ParserWitnessContactAssessmentBasis,
    atom_ref_position_from_scoring_context,
    materialized_atom_ref_position,
    parser_extra_heavy_proximity_bond_count_for_structure,
    parser_witness_pre_untangle_candidate_rank,
    parser_witness_pre_untangle_score_from_context,
    parser_witness_scoring_context,
    parser_witness_target_distance_angstrom,
)
from protrepair.transformer.discrete.parser_witness_pre_untangle_types import (
    MeasuredRotatedSidechainCandidate,
    ParserWitnessPreUntangleCandidate,
    ParserWitnessPreUntangleCandidateRank,
    ParserWitnessPreUntangleScore,
    ParserWitnessScoringContext,
    RankedRotatedSidechainCandidate,
    RotatedSidechainPayload,
    SidechainRootRotationPlan,
)

BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})
SIDECHAIN_ROOT_AXIS_ATOM_NAMES = ("CA", "CB")
PARSER_WITNESS_PRE_UNTANGLE_COVALENT_MARGIN_ANGSTROM = 0.45
PARSER_WITNESS_PRE_UNTANGLE_PRIMARY_ANGLE_DEGREES = (-30, 30, -20, 20, -10, 10)
PARSER_WITNESS_PRE_UNTANGLE_EXTENDED_ANGLE_DEGREES = (
    -90,
    90,
    -75,
    75,
    -60,
    60,
    -45,
    45,
)
PARSER_WITNESS_PRE_UNTANGLE_ANGLE_DEGREES = (
    PARSER_WITNESS_PRE_UNTANGLE_PRIMARY_ANGLE_DEGREES
    + PARSER_WITNESS_PRE_UNTANGLE_EXTENDED_ANGLE_DEGREES
)
PARSER_WITNESS_PRE_UNTANGLE_MAX_PASSES = 3


_MeasuredRotatedSidechainCandidate = MeasuredRotatedSidechainCandidate
_ParserWitnessPreUntangleCandidateRank = ParserWitnessPreUntangleCandidateRank
_ParserWitnessScoringContext = ParserWitnessScoringContext
_RankedRotatedSidechainCandidate = RankedRotatedSidechainCandidate
_RotatedSidechainPayload = RotatedSidechainPayload
_SidechainRootRotationPlan = SidechainRootRotationPlan


@dataclass(slots=True)
class _ParserWitnessContactBasisCache:
    """Lazily share structure-wide contact facts across cluster searches."""

    component_library: ComponentLibrary
    initial_clash_basis: ClashDetectionBasis | None = None
    _prepared_bases: tuple[ClashDetectionBasis, NearCovalentContactBasis] | None = (
        field(default=None, init=False, repr=False)
    )

    def basis_for(
        self,
        structure: ProteinStructure,
        cluster: RDKitProximityBondCluster,
    ) -> ParserWitnessContactAssessmentBasis:
        """Return one cluster view, preparing shared facts on first demand."""

        prepared_bases = self._prepared_bases
        if prepared_bases is None:
            clash_basis = self.initial_clash_basis
            clash_basis = (
                prepare_clash_detection_basis(
                    structure,
                    component_library=self.component_library,
                )
                if clash_basis is None
                else clash_basis
            )
            if not clash_basis.is_compatible_with(structure):
                raise ValueError(
                    "parser-witness contact cache requires a matching clash basis"
                )
            near_covalent_basis = prepare_near_covalent_contact_basis(
                structure,
                pair_policy=clash_basis.policy,
            )
            prepared_bases = (clash_basis, near_covalent_basis)
            self._prepared_bases = prepared_bases
        else:
            clash_basis, near_covalent_basis = prepared_bases
            near_covalent_basis.require_compatible_structure(structure)

        return ParserWitnessContactAssessmentBasis.from_bases(
            cluster,
            clash_basis=clash_basis,
            near_covalent_basis=near_covalent_basis,
        )


@dataclass(frozen=True, slots=True)
class _ParserWitnessPreUntangleSearchSession:
    """Private search session for one parser-witness cluster and baseline."""

    snapshot: ProteinStructureSnapshot
    atom_input: AtomInput | None
    cluster: RDKitProximityBondCluster
    component_library: ComponentLibrary
    clash_basis: ClashDetectionBasis | None = None
    baseline_extra_heavy_proximity_bond_count: int | None = None
    baseline_rank: _ParserWitnessPreUntangleCandidateRank | None = None
    known_bond_lookup: RDKitKnownBondLookup | None = None
    pdb_block_projector: RDKitNoConectPDBBlockProjector | None = None
    contact_basis_cache: _ParserWitnessContactBasisCache | None = field(
        default=None,
        repr=False,
        compare=False,
    )
    contact_assessment_basis: ParserWitnessContactAssessmentBasis | None = field(
        default=None,
        repr=False,
        compare=False,
    )

    def resolved_contact_assessment_basis(
        self,
    ) -> ParserWitnessContactAssessmentBasis:
        """Return contact facts cached for this cluster search session."""

        contact_assessment_basis = self.contact_assessment_basis
        if contact_assessment_basis is None:
            contact_assessment_basis = (
                ParserWitnessContactAssessmentBasis.prepare(
                    self.snapshot.structure,
                    self.cluster,
                    component_library=self.component_library,
                    clash_basis=self.clash_basis,
                )
                if self.contact_basis_cache is None
                else self.contact_basis_cache.basis_for(
                    self.snapshot.structure,
                    self.cluster,
                )
            )
            object.__setattr__(
                self,
                "contact_assessment_basis",
                contact_assessment_basis,
            )

        return contact_assessment_basis

    def build_ranked_candidate(self) -> _RankedRotatedSidechainCandidate | None:
        """Return the best ranked torsion move for this search session."""

        atom_input = self.atom_input
        if atom_input is None:
            return None

        if not _context_can_pre_untangle_parser_witness(atom_input):
            return None

        selected_residue_ids = frozenset(atom_input.referenced_residue_ids())
        if not set(self.cluster.residue_ids) <= selected_residue_ids:
            return None

        scoring_context = _parser_witness_scoring_context(
            self.snapshot.structure,
            self.cluster,
        )
        baseline_score = _parser_witness_pre_untangle_score_from_context(
            scoring_context
        )
        plans = _deduplicated_sidechain_root_rotation_plans(
            self.snapshot.structure,
            self.cluster,
            component_library=self.component_library,
        )
        if not plans:
            return None

        baseline_rank = self.normalized_baseline_rank(
            baseline_score=baseline_score,
        )
        for angle_degrees_group, exhaustive in (
            (
                PARSER_WITNESS_PRE_UNTANGLE_PRIMARY_ANGLE_DEGREES,
                False,
            ),
            (
                PARSER_WITNESS_PRE_UNTANGLE_EXTENDED_ANGLE_DEGREES,
                True,
            ),
        ):
            candidates = _ranked_parser_witness_pre_untangle_candidates(
                self.snapshot.structure,
                plans=plans,
                angle_degrees_group=angle_degrees_group,
                exhaustive=exhaustive,
                scoring_context=scoring_context,
                cluster=self.cluster,
                baseline_score=baseline_score,
                baseline_rank=baseline_rank,
                component_library=self.component_library,
                clash_basis=self.clash_basis,
                known_bond_lookup=self.known_bond_lookup,
                pdb_block_projector=self.pdb_block_projector,
                _session=self,
            )
            if candidates:
                return min(
                    candidates,
                    key=lambda candidate: (
                        candidate.rank,
                        candidate.payload.plan.residue_site.residue_id.display_token(),
                        candidate.payload.moved_atom_names,
                    ),
                )

        return None

    def normalized_baseline_rank(
        self,
        *,
        baseline_score: ParserWitnessPreUntangleScore,
    ) -> _ParserWitnessPreUntangleCandidateRank:
        """Return the baseline rank for this pass and target score."""

        if (
            self.baseline_rank is not None
            and self.baseline_rank.target_score == baseline_score
        ):
            return self.baseline_rank

        return self.rank_structure(
            self.snapshot.structure,
            target_score=baseline_score,
            angle_degrees=0,
            parser_extra_heavy_proximity_bond_count=(
                self.baseline_extra_heavy_proximity_bond_count
            ),
        )

    def ranked_candidates_for_angle_group(
        self,
        *,
        plans: tuple[_SidechainRootRotationPlan, ...],
        angle_degrees_group: tuple[int, ...],
        exhaustive: bool,
        scoring_context: _ParserWitnessScoringContext,
        baseline_score: ParserWitnessPreUntangleScore,
        baseline_rank: _ParserWitnessPreUntangleCandidateRank,
    ) -> tuple[_RankedRotatedSidechainCandidate, ...]:
        """Return safe ranked pre-untangle candidates for one angle group."""

        payloads = self.improved_payloads_for_angle_group(
            plans=plans,
            angle_degrees_group=angle_degrees_group,
            scoring_context=scoring_context,
            baseline_score=baseline_score,
        )
        if not payloads:
            return ()

        if not exhaustive:
            payloads = (
                min(
                    payloads,
                    key=lambda payload: (
                        payload.score,
                        abs(payload.angle_degrees),
                        payload.angle_degrees,
                        payload.plan.residue_site.residue_id.display_token(),
                        payload.moved_atom_names,
                    ),
                ),
            )

        measured_candidates = self.measured_candidates_not_worse_than_baseline(
            payloads,
            baseline_rank=baseline_rank,
        )
        if not measured_candidates:
            return ()

        rankable_candidates = self.rankable_minimum_parser_burden_candidates(
            measured_candidates,
            baseline_rank=baseline_rank,
        )
        candidates: list[_RankedRotatedSidechainCandidate] = []
        for measured_candidate in rankable_candidates:
            candidate = measured_candidate.candidate
            payload = measured_candidate.payload
            candidate_rank = self.rank_structure(
                candidate.structure,
                target_score=payload.score,
                angle_degrees=payload.angle_degrees,
                parser_extra_heavy_proximity_bond_count=(
                    measured_candidate.parser_extra_heavy_proximity_bond_count
                ),
            )
            if candidate_rank >= baseline_rank:
                continue

            candidates.append(
                _RankedRotatedSidechainCandidate(
                    candidate=ParserWitnessPreUntangleCandidate(
                        structure=candidate.structure,
                        moved_atom_indices=candidate.moved_atom_indices,
                        score=candidate.score,
                        parser_extra_heavy_proximity_bond_count=(
                            candidate_rank.parser_extra_heavy_proximity_bond_count
                        ),
                    ),
                    payload=payload,
                    rank=candidate_rank,
                )
            )

        return tuple(candidates)

    def improved_payloads_for_angle_group(
        self,
        *,
        plans: tuple[_SidechainRootRotationPlan, ...],
        angle_degrees_group: tuple[int, ...],
        scoring_context: _ParserWitnessScoringContext,
        baseline_score: ParserWitnessPreUntangleScore,
    ) -> tuple[_RotatedSidechainPayload, ...]:
        """Return rotated payloads that improve the target witness score."""

        payloads: list[_RotatedSidechainPayload] = []
        for plan in plans:
            for angle_degrees in angle_degrees_group:
                payload = _build_rotated_sidechain_payload(
                    self.snapshot.structure,
                    scoring_context,
                    plan,
                    angle_degrees=angle_degrees,
                )
                if payload.score >= baseline_score:
                    continue

                payloads.append(payload)

        return tuple(payloads)

    def measured_candidates_not_worse_than_baseline(
        self,
        payloads: tuple[_RotatedSidechainPayload, ...],
        *,
        baseline_rank: _ParserWitnessPreUntangleCandidateRank,
    ) -> tuple[_MeasuredRotatedSidechainCandidate, ...]:
        """Materialize payloads and keep those not worsening parser burden."""

        measured_candidates: list[_MeasuredRotatedSidechainCandidate] = []
        for payload in payloads:
            candidate = _materialize_rotated_sidechain_candidate(
                self.snapshot.structure,
                payload,
            )
            parser_extra_heavy_proximity_bond_count = self.parser_burden(
                candidate.structure
            )
            if (
                parser_extra_heavy_proximity_bond_count
                > baseline_rank.parser_extra_heavy_proximity_bond_count
            ):
                continue

            measured_candidates.append(
                _MeasuredRotatedSidechainCandidate(
                    candidate=candidate,
                    payload=payload,
                    parser_extra_heavy_proximity_bond_count=(
                        parser_extra_heavy_proximity_bond_count
                    ),
                )
            )

        return tuple(measured_candidates)

    def rankable_minimum_parser_burden_candidates(
        self,
        measured_candidates: tuple[_MeasuredRotatedSidechainCandidate, ...],
        *,
        baseline_rank: _ParserWitnessPreUntangleCandidateRank,
    ) -> tuple[_MeasuredRotatedSidechainCandidate, ...]:
        """Return candidates worth full ranking after parser-burden measurement."""

        minimum_parser_extra_heavy_proximity_bond_count = min(
            candidate.parser_extra_heavy_proximity_bond_count
            for candidate in measured_candidates
        )
        return tuple(
            candidate
            for candidate in measured_candidates
            if (
                minimum_parser_extra_heavy_proximity_bond_count
                == baseline_rank.parser_extra_heavy_proximity_bond_count
                or candidate.parser_extra_heavy_proximity_bond_count
                == minimum_parser_extra_heavy_proximity_bond_count
            )
        )

    def rank_structure(
        self,
        structure: ProteinStructure,
        *,
        target_score: ParserWitnessPreUntangleScore,
        angle_degrees: int,
        parser_extra_heavy_proximity_bond_count: int | None = None,
    ) -> _ParserWitnessPreUntangleCandidateRank:
        """Return whole-candidate rank for one materialized structure."""

        return _parser_witness_pre_untangle_candidate_rank(
            structure,
            self.cluster,
            target_score=target_score,
            angle_degrees=angle_degrees,
            component_library=self.component_library,
            clash_basis=self.clash_basis,
            contact_assessment_basis=self.resolved_contact_assessment_basis(),
            parser_extra_heavy_proximity_bond_count=(
                parser_extra_heavy_proximity_bond_count
            ),
            known_bond_lookup=self.known_bond_lookup,
            pdb_block_projector=self.pdb_block_projector,
        )

    def parser_burden(self, structure: ProteinStructure) -> int:
        """Return whole-structure parser proximity burden for one candidate."""

        return _parser_extra_heavy_proximity_bond_count(
            structure,
            component_library=self.component_library,
            known_bond_lookup=self.known_bond_lookup,
            pdb_block_projector=self.pdb_block_projector,
        )


class ParserWitnessPreUntangleTransformer(
    DeterministicContextOperation[ProteinStructureSnapshot]
):
    """Deterministically reduce parser-witness contacts before FF refinement."""

    def __init__(
        self,
        component_library: ComponentLibrary,
        *,
        clash_basis: ClashDetectionBasis | None = None,
        pdb_block_projector: RDKitNoConectPDBBlockProjector | None = None,
    ) -> None:
        self._component_library = component_library
        self._clash_basis = clash_basis
        self._pdb_block_projector = pdb_block_projector

    def is_applicable(
        self,
        context: ProteinTransformationContext,
    ) -> bool:
        """Return whether the current local domain can be pre-untangled."""

        if not _context_can_pre_untangle_parser_witness(context.atom_input):
            return False

        return bool(
            _eligible_parser_witness_clusters(
                context.source_snapshot.structure,
                context.atom_input,
                component_library=self._component_library,
            )
        )

    def transform(
        self,
        context: ProteinTransformationContext,
    ) -> ProteinStructureSnapshot:
        """Return one snapshot after bounded parser-witness pre-untangling."""

        if not _context_can_pre_untangle_parser_witness(context.atom_input):
            return context.source_snapshot

        pdb_block_projector = self._pdb_block_projector
        if pdb_block_projector is None or not pdb_block_projector.can_render(
            context.source_snapshot.structure
        ):
            pdb_block_projector = prepare_rdkit_no_conect_pdb_block_projector(
                context.source_snapshot.structure,
            )
        parser_witness_clusters = rdkit_no_conect_extra_proximity_bond_clusters(
            context.source_snapshot.structure,
            component_library=self._component_library,
            pdb_block_projector=pdb_block_projector,
        )
        clusters = _eligible_parser_witness_clusters_from_candidates(
            parser_witness_clusters,
            context.atom_input,
        )
        if not clusters:
            return context.source_snapshot

        baseline_extra_heavy_proximity_bond_count = sum(
            len(cluster.bonds) for cluster in parser_witness_clusters
        )
        known_bond_lookup = prepare_rdkit_no_conect_known_bond_lookup(
            context.source_snapshot.structure,
            component_library=self._component_library,
        )
        contact_basis_cache = _ParserWitnessContactBasisCache(
            component_library=self._component_library,
            initial_clash_basis=self._clash_basis,
        )
        working_snapshot = context.source_snapshot
        for cluster in clusters:
            baseline_rank: _ParserWitnessPreUntangleCandidateRank | None = None
            for _ in range(PARSER_WITNESS_PRE_UNTANGLE_MAX_PASSES):
                ranked_candidate = _build_ranked_parser_witness_pre_untangle_candidate(
                    working_snapshot,
                    context.atom_input,
                    cluster,
                    component_library=self._component_library,
                    clash_basis=self._clash_basis,
                    contact_basis_cache=contact_basis_cache,
                    baseline_extra_heavy_proximity_bond_count=(
                        baseline_extra_heavy_proximity_bond_count
                    ),
                    baseline_rank=baseline_rank,
                    known_bond_lookup=known_bond_lookup,
                    pdb_block_projector=pdb_block_projector,
                )
                if ranked_candidate is None:
                    break

                candidate = ranked_candidate.candidate
                working_snapshot = working_snapshot.with_structure(
                    candidate.structure
                )
                baseline_rank = _baseline_rank_from_accepted_candidate_rank(
                    ranked_candidate.rank
                )
                if candidate.parser_extra_heavy_proximity_bond_count is None:
                    baseline_extra_heavy_proximity_bond_count = (
                        _parser_extra_heavy_proximity_bond_count(
                            working_snapshot.structure,
                            component_library=self._component_library,
                            known_bond_lookup=known_bond_lookup,
                            pdb_block_projector=pdb_block_projector,
                        )
                    )
                    continue

                baseline_extra_heavy_proximity_bond_count = (
                    candidate.parser_extra_heavy_proximity_bond_count
                )

        return working_snapshot


def build_parser_witness_pre_untangle_candidate(
    snapshot: ProteinStructureSnapshot,
    atom_input: AtomInput,
    cluster: RDKitProximityBondCluster,
    *,
    component_library: ComponentLibrary,
    clash_basis: ClashDetectionBasis | None = None,
    baseline_extra_heavy_proximity_bond_count: int | None = None,
) -> ParserWitnessPreUntangleCandidate | None:
    """Return the best single torsion move for one parser-witness cluster."""

    ranked_candidate = _build_ranked_parser_witness_pre_untangle_candidate(
        snapshot,
        atom_input,
        cluster,
        component_library=component_library,
        clash_basis=clash_basis,
        baseline_extra_heavy_proximity_bond_count=(
            baseline_extra_heavy_proximity_bond_count
        ),
    )
    if ranked_candidate is None:
        return None

    return ranked_candidate.candidate


def _build_ranked_parser_witness_pre_untangle_candidate(
    snapshot: ProteinStructureSnapshot,
    atom_input: AtomInput,
    cluster: RDKitProximityBondCluster,
    *,
    component_library: ComponentLibrary,
    clash_basis: ClashDetectionBasis | None = None,
    contact_basis_cache: _ParserWitnessContactBasisCache | None = None,
    contact_assessment_basis: ParserWitnessContactAssessmentBasis | None = None,
    baseline_extra_heavy_proximity_bond_count: int | None = None,
    baseline_rank: _ParserWitnessPreUntangleCandidateRank | None = None,
    known_bond_lookup: RDKitKnownBondLookup | None = None,
    pdb_block_projector: RDKitNoConectPDBBlockProjector | None = None,
) -> _RankedRotatedSidechainCandidate | None:
    """Return the best ranked torsion move for one parser-witness cluster."""

    return _ParserWitnessPreUntangleSearchSession(
        snapshot=snapshot,
        atom_input=atom_input,
        cluster=cluster,
        component_library=component_library,
        clash_basis=clash_basis,
        contact_basis_cache=contact_basis_cache,
        contact_assessment_basis=contact_assessment_basis,
        baseline_extra_heavy_proximity_bond_count=(
            baseline_extra_heavy_proximity_bond_count
        ),
        baseline_rank=baseline_rank,
        known_bond_lookup=known_bond_lookup,
        pdb_block_projector=pdb_block_projector,
    ).build_ranked_candidate()


def _deduplicated_sidechain_root_rotation_plans(
    structure: ProteinStructure,
    cluster: RDKitProximityBondCluster,
    *,
    component_library: ComponentLibrary,
) -> tuple[_SidechainRootRotationPlan, ...]:
    """Return at most one CA-CB sidechain rotation plan per residue."""

    return parser_materialization.deduplicated_sidechain_root_rotation_plans(
        structure,
        cluster,
        component_library=component_library,
    )


def _ranked_parser_witness_pre_untangle_candidates(
    structure: ProteinStructure,
    *,
    plans: tuple[_SidechainRootRotationPlan, ...],
    angle_degrees_group: tuple[int, ...],
    exhaustive: bool,
    scoring_context: _ParserWitnessScoringContext,
    cluster: RDKitProximityBondCluster,
    baseline_score: ParserWitnessPreUntangleScore,
    baseline_rank: _ParserWitnessPreUntangleCandidateRank,
    component_library: ComponentLibrary,
    clash_basis: ClashDetectionBasis | None,
    known_bond_lookup: RDKitKnownBondLookup | None,
    pdb_block_projector: RDKitNoConectPDBBlockProjector | None,
    _session: _ParserWitnessPreUntangleSearchSession | None = None,
) -> tuple[_RankedRotatedSidechainCandidate, ...]:
    """Return safe ranked pre-untangle candidates for one angle group."""

    if _session is None:
        _session = _ParserWitnessPreUntangleSearchSession(
            snapshot=ProteinStructureSnapshot.from_structure(structure),
            atom_input=None,
            cluster=cluster,
            component_library=component_library,
            clash_basis=clash_basis,
            known_bond_lookup=known_bond_lookup,
            pdb_block_projector=pdb_block_projector,
        )

    return _session.ranked_candidates_for_angle_group(
        plans=plans,
        angle_degrees_group=angle_degrees_group,
        exhaustive=exhaustive,
        scoring_context=scoring_context,
        baseline_score=baseline_score,
        baseline_rank=baseline_rank,
    )


def _baseline_rank_from_accepted_candidate_rank(
    rank: _ParserWitnessPreUntangleCandidateRank,
) -> _ParserWitnessPreUntangleCandidateRank:
    """Return an accepted candidate rank normalized for the next baseline state."""

    return _ParserWitnessPreUntangleCandidateRank(
        parser_extra_heavy_proximity_bond_count=(
            rank.parser_extra_heavy_proximity_bond_count
        ),
        focus_near_covalent_contact_count=rank.focus_near_covalent_contact_count,
        focus_total_near_covalent_overlap_angstrom=(
            rank.focus_total_near_covalent_overlap_angstrom
        ),
        focus_clash_count=rank.focus_clash_count,
        target_score=rank.target_score,
        absolute_angle_degrees=0,
        angle_degrees=0,
    )


def parser_witness_pre_untangle_score(
    structure: ProteinStructure,
    cluster: RDKitProximityBondCluster,
) -> ParserWitnessPreUntangleScore:
    """Return unresolved false-proximity score for one witness cluster."""

    return parser_witness_pre_untangle_score_from_context(
        parser_witness_scoring_context(structure, cluster)
    )


def _parser_witness_pre_untangle_score_from_context(
    scoring_context: _ParserWitnessScoringContext,
    *,
    residue_id: ResidueId | None = None,
    residue_geometry: ResidueGeometry | None = None,
) -> ParserWitnessPreUntangleScore:
    """Return unresolved false-proximity score from cached witness data."""

    return parser_witness_pre_untangle_score_from_context(
        scoring_context,
        residue_id=residue_id,
        residue_geometry=residue_geometry,
    )


def _context_can_pre_untangle_parser_witness(atom_input: AtomInput) -> bool:
    """Return whether one atom input is a safe parser-witness preconditioner scope."""

    selected_scope = atom_input.as_scope()
    return (
        atom_input.is_residuewise()
        and isinstance(selected_scope, ResidueSetScope)
        and len(selected_scope.residue_ids) > 1
    )


def _eligible_parser_witness_clusters(
    structure: ProteinStructure,
    atom_input: AtomInput,
    *,
    component_library: ComponentLibrary,
) -> tuple[RDKitProximityBondCluster, ...]:
    """Return parser-witness clusters fully covered by the selected residues."""

    return _eligible_parser_witness_clusters_from_candidates(
        rdkit_no_conect_extra_proximity_bond_clusters(
            structure,
            component_library=component_library,
        ),
        atom_input,
    )


def _eligible_parser_witness_clusters_from_candidates(
    clusters: tuple[RDKitProximityBondCluster, ...],
    atom_input: AtomInput,
) -> tuple[RDKitProximityBondCluster, ...]:
    """Return candidate parser-witness clusters covered by selected residues."""

    selected_residue_ids = frozenset(atom_input.referenced_residue_ids())
    return tuple(
        cluster
        for cluster in clusters
        if set(cluster.residue_ids) <= selected_residue_ids
    )


def _ordered_cluster_endpoint_atom_refs(
    cluster: RDKitProximityBondCluster,
) -> tuple[AtomRef, ...]:
    """Return deduplicated witness endpoints in deterministic order."""

    return parser_materialization.ordered_cluster_endpoint_atom_refs(cluster)


def _sidechain_root_rotation_plan(
    structure: ProteinStructure,
    atom_ref: AtomRef,
    *,
    component_library: ComponentLibrary,
) -> _SidechainRootRotationPlan | None:
    """Return a safe CA-CB side-chain rotation plan for one witness endpoint."""

    return parser_materialization.sidechain_root_rotation_plan(
        structure,
        atom_ref,
        component_library=component_library,
    )


def _sidechain_root_rotating_atom_names(
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    *,
    atom_ref: AtomRef,
    component_bonded_atom_names: Callable[[str], frozenset[str]],
) -> frozenset[str]:
    """Return atoms that can rotate around CA-CB without moving backbone atoms."""

    return parser_materialization.sidechain_root_rotating_atom_names(
        residue_site,
        residue_geometry,
        atom_ref=atom_ref,
        component_bonded_atom_names=component_bonded_atom_names,
    )


def _reachable_component_atom_names_after_cut(
    *,
    start_atom_name: str,
    blocked_atom_name: str,
    present_atom_names: set[str],
    component_bonded_atom_names: Callable[[str], frozenset[str]],
) -> set[str]:
    """Return present component atoms reachable after cutting one graph edge."""

    return parser_materialization.reachable_component_atom_names_after_cut(
        start_atom_name=start_atom_name,
        blocked_atom_name=blocked_atom_name,
        present_atom_names=present_atom_names,
        component_bonded_atom_names=component_bonded_atom_names,
    )


def _build_rotated_sidechain_payload(
    structure: ProteinStructure,
    scoring_context: _ParserWitnessScoringContext,
    plan: _SidechainRootRotationPlan,
    *,
    angle_degrees: int,
) -> _RotatedSidechainPayload:
    """Return one parser-witness payload after a bounded side-chain rotation."""

    del structure
    return parser_materialization.build_rotated_sidechain_payload(
        scoring_context,
        plan,
        angle_degrees=angle_degrees,
    )


def _materialize_rotated_sidechain_candidate(
    structure: ProteinStructure,
    payload: _RotatedSidechainPayload,
) -> ParserWitnessPreUntangleCandidate:
    """Return one fully materialized parser-witness pre-untangle candidate."""

    return parser_materialization.materialize_rotated_sidechain_candidate(
        structure,
        payload,
    )


def _parser_witness_pre_untangle_candidate_rank(
    structure: ProteinStructure,
    cluster: RDKitProximityBondCluster,
    *,
    target_score: ParserWitnessPreUntangleScore,
    angle_degrees: int,
    component_library: ComponentLibrary,
    clash_basis: ClashDetectionBasis | None = None,
    contact_assessment_basis: ParserWitnessContactAssessmentBasis | None = None,
    parser_extra_heavy_proximity_bond_count: int | None = None,
    known_bond_lookup: RDKitKnownBondLookup | None = None,
    pdb_block_projector: RDKitNoConectPDBBlockProjector | None = None,
) -> _ParserWitnessPreUntangleCandidateRank:
    """Return whole-candidate rank for one parser-witness pre-untangle state."""

    return parser_witness_pre_untangle_candidate_rank(
        structure,
        cluster,
        target_score=target_score,
        angle_degrees=angle_degrees,
        component_library=component_library,
        clash_basis=clash_basis,
        contact_assessment_basis=contact_assessment_basis,
        parser_extra_heavy_proximity_bond_count=(
            parser_extra_heavy_proximity_bond_count
        ),
        known_bond_lookup=known_bond_lookup,
        pdb_block_projector=pdb_block_projector,
    )


def _parser_extra_heavy_proximity_bond_count(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    known_bond_lookup: RDKitKnownBondLookup | None = None,
    pdb_block_projector: RDKitNoConectPDBBlockProjector | None = None,
) -> int:
    """Return whole-structure extra heavy-heavy parser proximity bond count."""

    return parser_extra_heavy_proximity_bond_count_for_structure(
        structure,
        component_library=component_library,
        known_bond_lookup=known_bond_lookup,
        pdb_block_projector=pdb_block_projector,
    )


def _parser_witness_scoring_context(
    structure: ProteinStructure,
    cluster: RDKitProximityBondCluster,
) -> _ParserWitnessScoringContext:
    """Return cached geometry and target distances for one witness cluster."""

    return parser_witness_scoring_context(structure, cluster)


def _atom_ref_position_from_scoring_context(
    scoring_context: _ParserWitnessScoringContext,
    atom_ref: AtomRef,
    *,
    residue_id: ResidueId | None,
    residue_geometry: ResidueGeometry | None,
) -> Vec3 | None:
    """Return an atom-ref position with one residue-local geometry override."""

    return atom_ref_position_from_scoring_context(
        scoring_context,
        atom_ref,
        residue_id=residue_id,
        residue_geometry=residue_geometry,
    )


def _materialized_atom_ref_position(
    structure: ProteinStructure,
    atom_ref: AtomRef,
) -> Vec3 | None:
    """Return an atom-ref position from the materialized structure."""

    return materialized_atom_ref_position(structure, atom_ref)


def _parser_witness_target_distance_angstrom(
    witness: RDKitProximityBondWitness,
) -> float:
    """Return the clearance target for one RDKit false-proximity witness."""

    return parser_witness_target_distance_angstrom(witness)

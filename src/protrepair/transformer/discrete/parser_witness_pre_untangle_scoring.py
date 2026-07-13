"""Scoring context and ranking for parser-witness pre-untangle search."""

from dataclasses import dataclass

from protrepair.chemistry import ElementRadiusLookup, RadiusKind, prepare_radius_lookup
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.diagnostics.clash_pair_generation import PreparedAtomSitePairIndex
from protrepair.diagnostics.clashes import (
    ClashDetectionBasis,
    detect_clashes_from_context,
    prepare_clash_detection_basis,
)
from protrepair.diagnostics.near_covalent import (
    NearCovalentContactBasis,
    detect_near_covalent_contacts_from_context,
    prepare_near_covalent_contact_basis,
)
from protrepair.diagnostics.parser_readability import (
    RDKitKnownBondLookup,
    RDKitProximityBondCluster,
    RDKitProximityBondWitness,
    measure_rdkit_no_conect_extra_heavy_proximity_bond_count,
)
from protrepair.geometry import Vec3
from protrepair.io.pdb_projection import RDKitNoConectPDBBlockProjector
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.transformer.discrete.parser_witness_pre_untangle_types import (
    ParserWitnessPreUntangleCandidateRank,
    ParserWitnessPreUntangleScore,
    ParserWitnessScoringContext,
)

PARSER_WITNESS_PRE_UNTANGLE_COVALENT_MARGIN_ANGSTROM = 0.45


@dataclass(frozen=True, slots=True)
class ParserWitnessContactAssessment:
    """Contact-derived rank components for one parser-witness candidate."""

    focus_near_covalent_contact_count: int
    focus_total_near_covalent_overlap_angstrom: float
    focus_clash_count: int


@dataclass(frozen=True, slots=True)
class ParserWitnessContactAssessmentBasis:
    """Reusable facts for independent clash and near-covalent candidate scoring."""

    clash_basis: ClashDetectionBasis
    near_covalent_basis: NearCovalentContactBasis
    focus_residue_ids: frozenset[ResidueId]

    def __post_init__(self) -> None:
        if (
            self.near_covalent_basis.constitution is not self.clash_basis.constitution
            and self.near_covalent_basis.constitution
            != self.clash_basis.constitution
        ):
            raise ValueError(
                "parser-witness contact bases require one immutable constitution"
            )
        if (
            self.near_covalent_basis.topology is not self.clash_basis.topology
            and self.near_covalent_basis.topology != self.clash_basis.topology
        ):
            raise ValueError(
                "parser-witness contact bases require one immutable topology"
            )
        if (
            self.near_covalent_basis.pair_policy
            != self.clash_basis.policy.as_contact_pair_policy()
        ):
            raise ValueError("parser-witness contact bases require one pair policy")

        object.__setattr__(
            self,
            "focus_residue_ids",
            frozenset(self.focus_residue_ids),
        )

    @property
    def shared_candidate_cell_size_angstrom(self) -> float:
        """Return the cell size required by both independent metrics."""

        return max(
            self.clash_basis.candidate_cell_size_angstrom,
            self.near_covalent_basis.candidate_cell_size_angstrom,
        )

    @classmethod
    def from_bases(
        cls,
        cluster: RDKitProximityBondCluster,
        *,
        clash_basis: ClashDetectionBasis,
        near_covalent_basis: NearCovalentContactBasis,
    ) -> "ParserWitnessContactAssessmentBasis":
        """Bind one cluster focus to reusable structure-wide contact facts."""

        return cls(
            clash_basis=clash_basis,
            near_covalent_basis=near_covalent_basis,
            focus_residue_ids=frozenset(cluster.residue_ids),
        )

    @classmethod
    def prepare(
        cls,
        structure: ProteinStructure,
        cluster: RDKitProximityBondCluster,
        *,
        component_library: ComponentLibrary,
        clash_basis: ClashDetectionBasis | None = None,
    ) -> "ParserWitnessContactAssessmentBasis":
        """Prepare coordinate-independent contact facts for one witness cluster."""

        active_clash_basis = (
            prepare_clash_detection_basis(
                structure,
                component_library=component_library,
            )
            if clash_basis is None
            else clash_basis
        )
        if not active_clash_basis.is_compatible_with(structure):
            raise ValueError(
                "parser-witness contact assessment requires a matching clash basis"
            )
        near_covalent_basis = prepare_near_covalent_contact_basis(
            structure,
            pair_policy=active_clash_basis.policy,
        )
        return cls.from_bases(
            cluster,
            clash_basis=active_clash_basis,
            near_covalent_basis=near_covalent_basis,
        )

    def assess(self, structure: ProteinStructure) -> ParserWitnessContactAssessment:
        """Assess both contact metrics over one shared coordinate-bound index."""

        self.near_covalent_basis.require_compatible_structure(structure)
        clash_context = self.clash_basis.bind_context(
            structure,
            candidate_cell_size_angstrom=(
                self.shared_candidate_cell_size_angstrom
            ),
        )
        prepared_pair_index = PreparedAtomSitePairIndex(
            atom_sites=clash_context.atom_sites,
            focus_residue_ids=self.focus_residue_ids,
        )
        focus_clashes = detect_clashes_from_context(
            clash_context,
            focus_residue_ids=self.focus_residue_ids,
            prepared_pair_index=prepared_pair_index,
        ).clashes
        near_covalent_contacts = detect_near_covalent_contacts_from_context(
            structure,
            clash_context,
            focus_residue_ids=self.focus_residue_ids,
            basis=self.near_covalent_basis,
            prepared_pair_index=prepared_pair_index,
        )
        return ParserWitnessContactAssessment(
            focus_near_covalent_contact_count=len(near_covalent_contacts),
            focus_total_near_covalent_overlap_angstrom=sum(
                contact.overlap_angstrom for contact in near_covalent_contacts
            ),
            focus_clash_count=len(focus_clashes),
        )


def parser_witness_pre_untangle_score(
    structure: ProteinStructure,
    cluster: RDKitProximityBondCluster,
) -> ParserWitnessPreUntangleScore:
    """Return unresolved false-proximity score for one witness cluster."""

    return parser_witness_pre_untangle_score_from_context(
        parser_witness_scoring_context(structure, cluster)
    )


def parser_witness_pre_untangle_score_from_context(
    scoring_context: ParserWitnessScoringContext,
    *,
    residue_id: ResidueId | None = None,
    residue_geometry: ResidueGeometry | None = None,
) -> ParserWitnessPreUntangleScore:
    """Return unresolved false-proximity score from cached witness data."""

    overlaps: list[float] = []
    for witness in scoring_context.cluster.bonds:
        position_1 = atom_ref_position_from_scoring_context(
            scoring_context,
            witness.atom_ref_1,
            residue_id=residue_id,
            residue_geometry=residue_geometry,
        )
        position_2 = atom_ref_position_from_scoring_context(
            scoring_context,
            witness.atom_ref_2,
            residue_id=residue_id,
            residue_geometry=residue_geometry,
        )
        if position_1 is None or position_2 is None:
            continue

        overlap = max(
            0.0,
            scoring_context.target_distance_by_witness[witness]
            - position_1.distance_to(position_2),
        )
        if overlap > 0.0:
            overlaps.append(overlap)

    if not overlaps:
        return ParserWitnessPreUntangleScore(
            unresolved_contact_count=0,
            total_overlap_angstrom=0.0,
            worst_overlap_angstrom=0.0,
        )

    return ParserWitnessPreUntangleScore(
        unresolved_contact_count=len(overlaps),
        total_overlap_angstrom=sum(overlaps),
        worst_overlap_angstrom=max(overlaps),
    )


def parser_witness_pre_untangle_candidate_rank(
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
) -> ParserWitnessPreUntangleCandidateRank:
    """Return whole-candidate rank for one parser-witness pre-untangle state."""

    if parser_extra_heavy_proximity_bond_count is None:
        parser_extra_heavy_proximity_bond_count = (
            parser_extra_heavy_proximity_bond_count_for_structure(
                structure,
                component_library=component_library,
                known_bond_lookup=known_bond_lookup,
                pdb_block_projector=pdb_block_projector,
            )
        )
    active_contact_assessment_basis = (
        ParserWitnessContactAssessmentBasis.prepare(
            structure,
            cluster,
            component_library=component_library,
            clash_basis=clash_basis,
        )
        if contact_assessment_basis is None
        else contact_assessment_basis
    )
    if active_contact_assessment_basis.focus_residue_ids != frozenset(
        cluster.residue_ids
    ):
        raise ValueError(
            "parser-witness contact assessment basis requires a matching cluster"
        )
    if (
        clash_basis is not None
        and clash_basis is not active_contact_assessment_basis.clash_basis
    ):
        raise ValueError(
            "parser-witness contact assessment basis conflicts with clash basis"
        )
    contact_assessment = active_contact_assessment_basis.assess(structure)
    return ParserWitnessPreUntangleCandidateRank(
        parser_extra_heavy_proximity_bond_count=(
            parser_extra_heavy_proximity_bond_count
        ),
        focus_near_covalent_contact_count=(
            contact_assessment.focus_near_covalent_contact_count
        ),
        focus_total_near_covalent_overlap_angstrom=(
            contact_assessment.focus_total_near_covalent_overlap_angstrom
        ),
        focus_clash_count=contact_assessment.focus_clash_count,
        target_score=target_score,
        absolute_angle_degrees=abs(angle_degrees),
        angle_degrees=angle_degrees,
    )


def parser_extra_heavy_proximity_bond_count_for_structure(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    known_bond_lookup: RDKitKnownBondLookup | None = None,
    pdb_block_projector: RDKitNoConectPDBBlockProjector | None = None,
) -> int:
    """Return whole-structure extra heavy-heavy parser proximity bond count."""

    return measure_rdkit_no_conect_extra_heavy_proximity_bond_count(
        structure,
        component_library=component_library,
        known_bond_lookup=known_bond_lookup,
        pdb_block_projector=pdb_block_projector,
    )


def parser_witness_scoring_context(
    structure: ProteinStructure,
    cluster: RDKitProximityBondCluster,
) -> ParserWitnessScoringContext:
    """Return cached geometry and target distances for one witness cluster."""

    covalent_radius_lookup = prepare_radius_lookup(
        _cluster_witness_elements(cluster),
        RadiusKind.COVALENT,
    )
    covalent_radius_lookup.require_complete("parser-witness pre-untangle scoring")
    return ParserWitnessScoringContext(
        cluster=cluster,
        position_by_atom_ref={
            atom_ref: materialized_atom_ref_position(structure, atom_ref)
            for atom_ref in _ordered_cluster_endpoint_atom_refs(cluster)
        },
        target_distance_by_witness={
            witness: parser_witness_target_distance_angstrom(
                witness,
                covalent_radius_lookup=covalent_radius_lookup,
            )
            for witness in cluster.bonds
        },
    )


def atom_ref_position_from_scoring_context(
    scoring_context: ParserWitnessScoringContext,
    atom_ref: AtomRef,
    *,
    residue_id: ResidueId | None,
    residue_geometry: ResidueGeometry | None,
) -> Vec3 | None:
    """Return an atom-ref position with one residue-local geometry override."""

    if atom_ref.residue_id == residue_id and residue_geometry is not None:
        if not residue_geometry.has_atom(atom_ref.atom_name):
            return None

        return residue_geometry.position(atom_ref.atom_name)

    return scoring_context.position_by_atom_ref.get(atom_ref)


def materialized_atom_ref_position(
    structure: ProteinStructure,
    atom_ref: AtomRef,
) -> Vec3 | None:
    """Return an atom-ref position from the materialized structure."""

    if structure.constitution.resolve_atom_site(atom_ref) is None:
        return None

    return structure.geometry.position(structure.constitution.atom_index(atom_ref))


def parser_witness_target_distance_angstrom(
    witness: RDKitProximityBondWitness,
    *,
    covalent_radius_lookup: ElementRadiusLookup | None = None,
) -> float:
    """Return the clearance target for one RDKit false-proximity witness."""

    active_radius_lookup = (
        prepare_radius_lookup(
            (witness.element_1, witness.element_2),
            RadiusKind.COVALENT,
        )
        if covalent_radius_lookup is None
        else covalent_radius_lookup
    )
    active_radius_lookup.require_kind(
        RadiusKind.COVALENT,
        "parser-witness target distance",
    )
    active_radius_lookup.require_complete("parser-witness target distance")
    return (
        active_radius_lookup.radius_angstrom(witness.element_1)
        + active_radius_lookup.radius_angstrom(witness.element_2)
        + PARSER_WITNESS_PRE_UNTANGLE_COVALENT_MARGIN_ANGSTROM
    )


def _cluster_witness_elements(cluster: RDKitProximityBondCluster) -> tuple[str, ...]:
    """Return witness elements used by one parser-witness cluster."""

    return tuple(
        element
        for witness in cluster.bonds
        for element in (witness.element_1, witness.element_2)
    )


def _ordered_cluster_endpoint_atom_refs(
    cluster: RDKitProximityBondCluster,
) -> tuple[AtomRef, ...]:
    return AtomRef.deduplicated(
        atom_ref
        for witness in cluster.bonds
        for atom_ref in (witness.atom_ref_1, witness.atom_ref_2)
    )

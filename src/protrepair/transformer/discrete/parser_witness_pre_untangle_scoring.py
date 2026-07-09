"""Scoring context and ranking for parser-witness pre-untangle search."""

from protrepair.chemistry import ElementRadiusLookup, RadiusKind, prepare_radius_lookup
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.diagnostics.clashes import (
    ClashDetectionBasis,
    detect_clashes_from_context,
    prepare_clash_detection_context,
)
from protrepair.diagnostics.near_covalent import (
    detect_near_covalent_contacts_from_context,
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
    clash_context = prepare_clash_detection_context(
        structure,
        component_library=component_library,
        basis=clash_basis,
    )
    focus_clashes = detect_clashes_from_context(
        clash_context,
        focus_residue_ids=frozenset(cluster.residue_ids),
    ).clashes
    near_covalent_contacts = detect_near_covalent_contacts_from_context(
        clash_context,
        focus_residue_ids=frozenset(cluster.residue_ids),
    )
    return ParserWitnessPreUntangleCandidateRank(
        parser_extra_heavy_proximity_bond_count=(
            parser_extra_heavy_proximity_bond_count
        ),
        focus_near_covalent_contact_count=len(near_covalent_contacts),
        focus_total_near_covalent_overlap_angstrom=sum(
            contact.overlap_angstrom for contact in near_covalent_contacts
        ),
        focus_clash_count=len(focus_clashes),
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

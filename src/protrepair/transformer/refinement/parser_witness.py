"""Parser-witness local-refinement candidate construction."""

from dataclasses import dataclass
from enum import Enum
from math import ceil

from protrepair.chemistry import ComponentLibrary
from protrepair.diagnostics.parser_readability import (
    RDKitProximityBondCluster,
    RDKitProximityBondWitness,
    rdkit_no_conect_extra_proximity_bond_clusters,
)
from protrepair.diagnostics.parser_topology import (
    ambiguous_disulfide_partner_residue_ids_by_residue,
    parser_witness_cluster_is_ambiguous_disulfide_blocker,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.transformer.continuous.binding import (
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationConfig
from protrepair.transformer.local.models import LocalScopeSpec
from protrepair.transformer.refinement.spec import RepairRefinementSpec

POLYMER_BACKBONE_HEAVY_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})


@dataclass(frozen=True, slots=True)
class ParserWitnessRepairBudget:
    """Bounded execution budget for parser-witness local refinement."""

    max_passes: int = 64
    base_passes: int = 3
    extra_heavy_bonds_per_pass: int = 1
    max_clusters_per_pass: int = 4
    max_cluster_residues: int = 6
    context_radius_angstrom: float = 3.0
    max_iterations: int = 20
    sidechain_max_iterations: int = 50

    def __post_init__(self) -> None:
        if self.max_passes <= 0:
            raise ValueError("max_passes must be positive")
        if self.base_passes <= 0:
            raise ValueError("base_passes must be positive")
        if self.extra_heavy_bonds_per_pass <= 0:
            raise ValueError("extra_heavy_bonds_per_pass must be positive")
        if self.max_clusters_per_pass <= 0:
            raise ValueError("max_clusters_per_pass must be positive")
        if self.max_cluster_residues <= 0:
            raise ValueError("max_cluster_residues must be positive")
        if self.context_radius_angstrom <= 0.0:
            raise ValueError("context_radius_angstrom must be positive")
        if self.max_iterations <= 0:
            raise ValueError("max_iterations must be positive")
        if self.sidechain_max_iterations <= 0:
            raise ValueError("sidechain_max_iterations must be positive")

    def pass_limit_for_initial_extra_heavy_bond_count(
        self,
        extra_heavy_bond_count: int,
    ) -> int:
        """Return the pass cap for one parser-witness repair invocation."""

        if extra_heavy_bond_count < 0:
            raise ValueError("extra_heavy_bond_count must be non-negative")

        scaled_passes = ceil(
            extra_heavy_bond_count / self.extra_heavy_bonds_per_pass
        )
        return min(self.max_passes, max(self.base_passes, scaled_passes))


@dataclass(frozen=True, slots=True)
class ParserWitnessRepairCandidate:
    """One parser-witness cluster paired with a bounded repair request."""

    cluster: RDKitProximityBondCluster
    repair_refinement: RepairRefinementSpec
    budget: ParserWitnessRepairBudget


DEFAULT_PARSER_WITNESS_REPAIR_BUDGET = ParserWitnessRepairBudget()


class ParserWitnessRepairExclusionReason(Enum):
    """Reason one parser-witness cluster is not an ordinary FF repair candidate."""

    AMBIGUOUS_DISULFIDE_TOPOLOGY = "ambiguous_disulfide_topology"
    CLUSTER_TOO_LARGE = "cluster_too_large"


@dataclass(frozen=True, slots=True)
class ParserWitnessRepairExclusion:
    """One parser-witness cluster excluded from ordinary local refinement."""

    cluster: RDKitProximityBondCluster
    reason: ParserWitnessRepairExclusionReason
    budget: ParserWitnessRepairBudget

    def display_token(self) -> str:
        """Return a compact diagnostic token for the excluded cluster."""

        return f"{self.reason.value}:{self.cluster.display_token()}"


def parser_witness_repair_candidates(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    budget: ParserWitnessRepairBudget = DEFAULT_PARSER_WITNESS_REPAIR_BUDGET,
    clusters: tuple[RDKitProximityBondCluster, ...] | None = None,
) -> tuple[ParserWitnessRepairCandidate, ...]:
    """Return deterministic bounded repair candidates from RDKit parser witnesses."""

    candidates: list[ParserWitnessRepairCandidate] = []
    active_clusters = _active_parser_witness_clusters(
        structure,
        component_library=component_library,
        clusters=clusters,
    )
    if not active_clusters:
        return ()

    ambiguous_disulfide_partners = (
        ambiguous_disulfide_partner_residue_ids_by_residue(structure)
    )
    for cluster in active_clusters:
        if _parser_witness_repair_exclusion_for_cluster(
            cluster,
            budget=budget,
            ambiguous_disulfide_partners=ambiguous_disulfide_partners,
        ) is not None:
            continue

        scope_spec = _parser_witness_repair_scope_spec(structure, cluster)
        candidates.append(
            ParserWitnessRepairCandidate(
                cluster=cluster,
                repair_refinement=RepairRefinementSpec(
                    scope_spec=scope_spec,
                    binding=RecommendedContinuousRelaxationBinding(),
                    config=_parser_witness_repair_config(
                        budget,
                        scope_spec=scope_spec,
                    ),
                ),
                budget=budget,
            )
        )
        if len(candidates) >= budget.max_clusters_per_pass:
            break

    return tuple(candidates)


def parser_witness_repair_exclusions(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    budget: ParserWitnessRepairBudget = DEFAULT_PARSER_WITNESS_REPAIR_BUDGET,
    clusters: tuple[RDKitProximityBondCluster, ...] | None = None,
) -> tuple[ParserWitnessRepairExclusion, ...]:
    """Return parser-witness clusters excluded from ordinary local FF repair."""

    active_clusters = _active_parser_witness_clusters(
        structure,
        component_library=component_library,
        clusters=clusters,
    )
    if not active_clusters:
        return ()

    ambiguous_disulfide_partners = (
        ambiguous_disulfide_partner_residue_ids_by_residue(structure)
    )
    return tuple(
        exclusion
        for cluster in active_clusters
        if (
            exclusion := _parser_witness_repair_exclusion_for_cluster(
                cluster,
                budget=budget,
                ambiguous_disulfide_partners=ambiguous_disulfide_partners,
            )
        )
        is not None
    )


def _active_parser_witness_clusters(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    clusters: tuple[RDKitProximityBondCluster, ...] | None,
) -> tuple[RDKitProximityBondCluster, ...]:
    """Return supplied or freshly derived parser-witness clusters."""

    if clusters is not None:
        return clusters

    return rdkit_no_conect_extra_proximity_bond_clusters(
        structure,
        component_library=component_library,
    )


def _parser_witness_repair_exclusion_for_cluster(
    cluster: RDKitProximityBondCluster,
    *,
    budget: ParserWitnessRepairBudget,
    ambiguous_disulfide_partners: dict[ResidueId, frozenset[ResidueId]],
) -> ParserWitnessRepairExclusion | None:
    """Return why one parser-witness cluster is not repairable, if excluded."""

    if parser_witness_cluster_is_ambiguous_disulfide_blocker(
        cluster,
        ambiguous_disulfide_partners=ambiguous_disulfide_partners,
    ):
        return ParserWitnessRepairExclusion(
            cluster=cluster,
            reason=ParserWitnessRepairExclusionReason.AMBIGUOUS_DISULFIDE_TOPOLOGY,
            budget=budget,
        )

    if len(cluster.residue_ids) > budget.max_cluster_residues:
        return ParserWitnessRepairExclusion(
            cluster=cluster,
            reason=ParserWitnessRepairExclusionReason.CLUSTER_TOO_LARGE,
            budget=budget,
        )

    return None


def _parser_witness_repair_scope_spec(
    structure: ProteinStructure,
    cluster: RDKitProximityBondCluster,
) -> LocalScopeSpec:
    """Return the narrowest safe motion scope for one parser-witness cluster."""

    if _cluster_supports_sidechain_only_motion(structure, cluster):
        return LocalScopeSpec.from_residue_sidechains(cluster.residue_ids)

    return LocalScopeSpec.from_residues(cluster.residue_ids)


def _parser_witness_repair_config(
    budget: ParserWitnessRepairBudget,
    *,
    scope_spec: LocalScopeSpec,
) -> ContinuousRelaxationConfig:
    """Return parser-witness local refinement settings for one motion scope."""

    return ContinuousRelaxationConfig(
        context_radius_angstrom=budget.context_radius_angstrom,
        max_iterations=(
            budget.sidechain_max_iterations
            if scope_spec.lowers_to_residue_sidechains()
            else budget.max_iterations
        ),
    )


def _cluster_supports_sidechain_only_motion(
    structure: ProteinStructure,
    cluster: RDKitProximityBondCluster,
) -> bool:
    """Return whether parser-witness repair can preserve polymer backbones."""

    return all(
        _residue_supports_sidechain_only_motion(structure, residue_id)
        for residue_id in cluster.residue_ids
    ) and all(
        _witness_supports_sidechain_only_motion(structure, witness)
        for witness in cluster.bonds
    )


def _residue_supports_sidechain_only_motion(
    structure: ProteinStructure,
    residue_id: ResidueId,
) -> bool:
    """Return whether one residue can use polymer side-chain lowering."""

    residue = structure.constitution.residue_or_ligand(residue_id)
    return residue is not None and not residue.is_hetero


def _witness_supports_sidechain_only_motion(
    structure: ProteinStructure,
    witness: RDKitProximityBondWitness,
) -> bool:
    """Return whether both parser-witness endpoints are side-chain atoms."""

    return (
        _atom_ref_supports_sidechain_only_motion(structure, witness.atom_ref_1)
        and _atom_ref_supports_sidechain_only_motion(structure, witness.atom_ref_2)
    )


def _atom_ref_supports_sidechain_only_motion(
    structure: ProteinStructure,
    atom_ref: AtomRef,
) -> bool:
    """Return whether one atom belongs to a polymer side-chain motion group."""

    residue = structure.constitution.residue_or_ligand(atom_ref.residue_id)
    if residue is None or residue.is_hetero:
        return False

    atom_site = residue.atom_site(atom_ref.atom_name)
    if atom_site.element == "H":
        return False

    return atom_site.name.strip().upper() not in POLYMER_BACKBONE_HEAVY_ATOM_NAMES

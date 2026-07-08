"""Joint local-correction scope proposals derived from severe contact burden."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum

from protrepair.chemistry import ComponentLibrary
from protrepair.diagnostics.clashes import (
    ClashPolicy,
    prepare_clash_detection_context,
)
from protrepair.diagnostics.geometry import residue_sort_key
from protrepair.diagnostics.near_covalent import (
    NearCovalentContact,
    NearCovalentContactPolicy,
    detect_near_covalent_contacts_from_context,
)
from protrepair.geometry import Vec3
from protrepair.scope import ResidueSetScope
from protrepair.state import HydrogenCoverageState, StructureChemistryReadinessFacts
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.local.models import LocalScopeSpec

POLYMER_BACKBONE_HEAVY_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})
HYDROGEN_ATTACHMENT_DISTANCE_MAX_ANGSTROM = 1.35


class JointCorrectionMotionClass(str, Enum):
    """Execution-local motion class implied by one joint pathology cluster."""

    RESIDUE_SIDECHAINS = "residue_sidechains"
    RESIDUE_ATOMS = "residue_atoms"

    def to_local_scope_spec(
        self,
        residue_ids: tuple[ResidueId, ...],
    ) -> LocalScopeSpec:
        """Project this motion class into one local lowering over residue ids."""

        if self is JointCorrectionMotionClass.RESIDUE_ATOMS:
            return LocalScopeSpec.from_residues(residue_ids)

        return LocalScopeSpec.from_residue_sidechains(residue_ids)

    @classmethod
    def widened(
        cls,
        motion_classes: Iterable["JointCorrectionMotionClass"],
    ) -> "JointCorrectionMotionClass":
        """Return the least-narrow motion class that covers all inputs."""

        if any(
            motion_class is JointCorrectionMotionClass.RESIDUE_ATOMS
            for motion_class in motion_classes
        ):
            return JointCorrectionMotionClass.RESIDUE_ATOMS

        return JointCorrectionMotionClass.RESIDUE_SIDECHAINS


@dataclass(frozen=True, slots=True)
class JointCorrectionScopePolicy:
    """Policy for promoting near-covalent contacts into joint scopes."""

    minimum_overlap_angstrom: float = 0.0
    covalent_distance_margin_angstrom: float = 0.45

    def __post_init__(self) -> None:
        if self.minimum_overlap_angstrom < 0.0:
            raise ValueError("minimum_overlap_angstrom must be non-negative")
        if self.covalent_distance_margin_angstrom < 0.0:
            raise ValueError(
                "covalent_distance_margin_angstrom must be non-negative"
            )


@dataclass(frozen=True, slots=True)
class JointCorrectionBatchingPolicy:
    """Policy for batching near-covalent clusters into local scopes."""

    minimum_cluster_worst_overlap_angstrom: float = 0.25
    minimum_cluster_total_overlap_angstrom: float = 0.75
    compatibility_context_distance_angstrom: float = 6.0

    def __post_init__(self) -> None:
        if self.minimum_cluster_worst_overlap_angstrom <= 0.0:
            raise ValueError(
                "minimum_cluster_worst_overlap_angstrom must be positive"
            )
        if self.minimum_cluster_total_overlap_angstrom <= 0.0:
            raise ValueError(
                "minimum_cluster_total_overlap_angstrom must be positive"
            )
        if self.compatibility_context_distance_angstrom < 0.0:
            raise ValueError(
                "compatibility_context_distance_angstrom must be non-negative"
            )


@dataclass(frozen=True, slots=True)
class JointCorrectionScopeProposal:
    """One multi-residue refinement scope proposed from near-covalent contacts."""

    residue_ids: tuple[ResidueId, ...]
    contact_pair_count: int
    worst_overlap_angstrom: float
    total_overlap_angstrom: float
    motion_class: JointCorrectionMotionClass = (
        JointCorrectionMotionClass.RESIDUE_SIDECHAINS
    )

    def __post_init__(self) -> None:
        object.__setattr__(self, "residue_ids", tuple(self.residue_ids))
        if len(self.residue_ids) < 2:
            raise ValueError(
                "joint correction scopes require at least two residue ids"
            )
        if self.contact_pair_count <= 0:
            raise ValueError(
                "joint correction scopes require at least one severe contact"
            )
        if not isinstance(self.motion_class, JointCorrectionMotionClass):
            raise TypeError(
                "joint correction scopes require a JointCorrectionMotionClass value"
            )

    def as_local_scope_spec(self) -> LocalScopeSpec:
        """Project this proposal into one residuewise local scope spec."""

        return self.motion_class.to_local_scope_spec(self.residue_ids)

    def as_scope(self) -> ResidueSetScope:
        """Project this proposal into one canonical residue-set scope."""

        return ResidueSetScope(residue_ids=self.residue_ids)


def execution_scope_spec_for_joint_correction(
    structure: ProteinStructure,
    *,
    proposal: JointCorrectionScopeProposal,
) -> LocalScopeSpec | None:
    """Return an optional execution-local widening for one joint proposal.

    Semantic joint-correction scopes stay on the pathology residue cluster.
    Execution-local widening is derived separately so backbone-involving
    clusters can move one peptide-neighborhood window without broadening the
    planner-visible canonical scope.
    """

    if proposal.motion_class is not JointCorrectionMotionClass.RESIDUE_ATOMS:
        return None

    widened_residue_ids = _execution_residue_ids_for_backbone_motion(
        structure,
        residue_ids=proposal.residue_ids,
    )
    if widened_residue_ids == proposal.residue_ids:
        return None

    return LocalScopeSpec.from_residues(widened_residue_ids)


def propose_joint_correction_scopes(
    structure: ProteinStructure,
    *,
    focus_residue_ids: tuple[ResidueId, ...],
    component_library: ComponentLibrary,
    chemistry_readiness_facts: StructureChemistryReadinessFacts,
    include_ligands: bool = False,
    policy: JointCorrectionScopePolicy | None = None,
) -> tuple[JointCorrectionScopeProposal, ...]:
    """Return correction scopes for near-covalent focus-connected contacts."""

    if not focus_residue_ids:
        return ()

    active_policy = (
        JointCorrectionScopePolicy()
        if policy is None
        else policy
    )
    clash_context = prepare_clash_detection_context(
        structure,
        component_library=component_library,
        policy=ClashPolicy(
            include_hydrogens=(
                chemistry_readiness_facts.hydrogen_coverage_state
                is HydrogenCoverageState.COMPLETE
            ),
            include_ligands=include_ligands,
        ),
    )
    focus_residue_id_set = frozenset(focus_residue_ids)
    near_covalent_contacts = tuple(
        detect_near_covalent_contacts_from_context(
            clash_context,
            focus_residue_ids=focus_residue_id_set,
            policy=NearCovalentContactPolicy(
                minimum_overlap_angstrom=active_policy.minimum_overlap_angstrom,
                covalent_distance_margin_angstrom=(
                    active_policy.covalent_distance_margin_angstrom
                ),
            ),
        )
    )
    if not near_covalent_contacts:
        return ()

    proposals: list[JointCorrectionScopeProposal] = []
    for residue_cluster in _contact_connected_residue_clusters(
        near_covalent_contacts,
        focus_residue_ids=focus_residue_ids,
    ):
        cluster_contacts = tuple(
            contact
            for contact in near_covalent_contacts
            if contact.left_residue_id in residue_cluster
            and contact.right_residue_id in residue_cluster
        )
        if not cluster_contacts:
            continue

        proposals.append(
            JointCorrectionScopeProposal(
                residue_ids=tuple(
                    sorted(residue_cluster, key=residue_sort_key)
                ),
                contact_pair_count=len(cluster_contacts),
                worst_overlap_angstrom=max(
                    contact.overlap_angstrom for contact in cluster_contacts
                ),
                total_overlap_angstrom=sum(
                    contact.overlap_angstrom for contact in cluster_contacts
                ),
                motion_class=_cluster_motion_class(
                    structure,
                    cluster_contacts=cluster_contacts,
                    component_library=component_library,
                ),
            )
        )

    proposals.sort(
        key=lambda proposal: (
            -proposal.worst_overlap_angstrom,
            -proposal.total_overlap_angstrom,
            -proposal.contact_pair_count,
            tuple(residue_sort_key(residue_id) for residue_id in proposal.residue_ids),
        )
    )
    return tuple(proposals)


def batch_joint_correction_scope_proposals(
    structure: ProteinStructure,
    *,
    proposals: tuple[JointCorrectionScopeProposal, ...],
    batching_policy: JointCorrectionBatchingPolicy | None = None,
) -> tuple[JointCorrectionScopeProposal, ...]:
    """Return thresholded joint-correction proposals merged by local compatibility."""

    if not proposals:
        return ()

    active_policy = (
        JointCorrectionBatchingPolicy()
        if batching_policy is None
        else batching_policy
    )
    thresholded_proposals = tuple(
        proposal
        for proposal in proposals
        if _proposal_passes_batching_threshold(
            proposal,
            policy=active_policy,
        )
    )
    if not thresholded_proposals:
        return ()

    residue_atom_positions = {
        residue.residue_id: tuple(
            structure.geometry.position(atom_index)
            for atom_index in structure.constitution.atom_indices_for_residue(
                residue.residue_id
            )
        )
        for residue in structure.constitution.iter_residues(include_ligands=False)
    }
    context_residue_ids_by_index = {
        proposal_index: _proposal_context_residue_ids(
            proposal,
            residue_atom_positions=residue_atom_positions,
            compatibility_context_distance_angstrom=(
                active_policy.compatibility_context_distance_angstrom
            ),
        )
        for proposal_index, proposal in enumerate(thresholded_proposals)
    }
    batched_proposals: list[JointCorrectionScopeProposal] = []
    for proposal_indices in _compatible_proposal_components(
        thresholded_proposals,
        context_residue_ids_by_index=context_residue_ids_by_index,
    ):
        component_proposals = tuple(
            thresholded_proposals[proposal_index]
            for proposal_index in proposal_indices
        )
        batched_proposals.append(
            JointCorrectionScopeProposal(
                residue_ids=tuple(
                    sorted(
                        {
                            residue_id
                            for proposal in component_proposals
                            for residue_id in proposal.residue_ids
                        },
                        key=residue_sort_key,
                    )
                ),
                contact_pair_count=sum(
                    proposal.contact_pair_count
                    for proposal in component_proposals
                ),
                worst_overlap_angstrom=max(
                    proposal.worst_overlap_angstrom
                    for proposal in component_proposals
                ),
                total_overlap_angstrom=sum(
                    proposal.total_overlap_angstrom
                    for proposal in component_proposals
                ),
                motion_class=JointCorrectionMotionClass.widened(
                    proposal.motion_class for proposal in component_proposals
                ),
            )
        )

    batched_proposals.sort(
        key=lambda proposal: (
            -proposal.worst_overlap_angstrom,
            -proposal.total_overlap_angstrom,
            -proposal.contact_pair_count,
            tuple(
                residue_sort_key(residue_id)
                for residue_id in proposal.residue_ids
            ),
        )
    )
    return tuple(batched_proposals)


def _cluster_motion_class(
    structure: ProteinStructure,
    *,
    cluster_contacts: tuple[NearCovalentContact, ...],
    component_library: ComponentLibrary,
) -> JointCorrectionMotionClass:
    """Return the minimum motion class required by one contact cluster."""

    if any(
        _contact_requires_residue_atoms(
            structure,
            contact=contact,
            component_library=component_library,
        )
        for contact in cluster_contacts
    ):
        return JointCorrectionMotionClass.RESIDUE_ATOMS

    return JointCorrectionMotionClass.RESIDUE_SIDECHAINS


def _execution_residue_ids_for_backbone_motion(
    structure: ProteinStructure,
    *,
    residue_ids: tuple[ResidueId, ...],
) -> tuple[ResidueId, ...]:
    """Return one peptide-neighborhood execution residue set.

    Backbone motion often needs one adjacent polymer residue on either side to
    keep the local peptide frame addressable. This widening is execution-only;
    the canonical joint-correction scope remains the original pathology cluster.
    """

    widened_residue_ids: set[ResidueId] = set(residue_ids)
    constitution = structure.constitution
    for residue_id in residue_ids:
        residue = constitution.residue_or_ligand(residue_id)
        if residue is None or residue.is_hetero:
            continue

        chain_index = constitution.chain_index(residue_id.chain_id)
        chain_residue_indices = constitution.residue_indices_for_chain_index(
            chain_index
        )
        residue_index = constitution.residue_index(residue_id)
        ordinal = chain_residue_indices.index(residue_index)
        if ordinal > 0:
            widened_residue_ids.add(
                constitution.residue_site_at(
                    chain_residue_indices[ordinal - 1]
                ).residue_id
            )
        if ordinal + 1 < len(chain_residue_indices):
            widened_residue_ids.add(
                constitution.residue_site_at(
                    chain_residue_indices[ordinal + 1]
                ).residue_id
            )

    return tuple(sorted(widened_residue_ids, key=residue_sort_key))


def _contact_requires_residue_atoms(
    structure: ProteinStructure,
    *,
    contact: NearCovalentContact,
    component_library: ComponentLibrary,
) -> bool:
    """Return whether one contact involves polymer-backbone-local motion."""

    return _atom_requires_residue_atoms(
        structure,
        residue_id=contact.left_residue_id,
        atom_name=contact.left_atom_name,
        component_library=component_library,
    ) or _atom_requires_residue_atoms(
        structure,
        residue_id=contact.right_residue_id,
        atom_name=contact.right_atom_name,
        component_library=component_library,
    )


def _atom_requires_residue_atoms(
    structure: ProteinStructure,
    *,
    residue_id: ResidueId,
    atom_name: str,
    component_library: ComponentLibrary,
) -> bool:
    """Return whether one atom belongs to the polymer-backbone motion group."""

    residue = structure.constitution.residue_or_ligand(residue_id)
    if residue is None or residue.is_hetero:
        return False

    normalized_atom_name = atom_name.strip().upper()
    atom_site = residue.atom_site(normalized_atom_name)
    if atom_site.element != "H":
        return normalized_atom_name in POLYMER_BACKBONE_HEAVY_ATOM_NAMES

    anchor_atom_name = _hydrogen_anchor_atom_name(
        structure,
        residue_id=residue_id,
        hydrogen_atom_name=normalized_atom_name,
        component_library=component_library,
    )
    return anchor_atom_name in POLYMER_BACKBONE_HEAVY_ATOM_NAMES


def _hydrogen_anchor_atom_name(
    structure: ProteinStructure,
    *,
    residue_id: ResidueId,
    hydrogen_atom_name: str,
    component_library: ComponentLibrary,
) -> str | None:
    """Return the best local heavy-atom anchor for one present hydrogen atom."""

    residue = structure.constitution.residue_or_ligand(residue_id)
    if residue is None:
        return None

    template = component_library.get(residue.component_id)
    if template is not None:
        anchor_atom_name = template.template_hydrogen_anchor_by_name(
            residue.atom_site_names()
        ).get(hydrogen_atom_name)
        if anchor_atom_name is not None:
            return anchor_atom_name

    residue_geometry = structure.residue_geometry(
        structure.constitution.residue_index(residue_id)
    )
    hydrogen_atom_geometry = residue_geometry.atom_geometry(hydrogen_atom_name)
    nearest_anchor_atom_name: str | None = None
    nearest_anchor_distance = float("inf")
    for atom_site in residue.atom_sites:
        if atom_site.element == "H":
            continue

        pair_distance = hydrogen_atom_geometry.distance_to(
            residue_geometry.atom_geometry(atom_site.name)
        )
        if pair_distance > HYDROGEN_ATTACHMENT_DISTANCE_MAX_ANGSTROM:
            continue
        if pair_distance < nearest_anchor_distance:
            nearest_anchor_atom_name = atom_site.name
            nearest_anchor_distance = pair_distance

    return nearest_anchor_atom_name


def _contact_connected_residue_clusters(
    contacts: Iterable[NearCovalentContact],
    *,
    focus_residue_ids: tuple[ResidueId, ...],
) -> tuple[frozenset[ResidueId], ...]:
    """Return focus-connected clusters from near-covalent inter-residue contacts."""

    adjacency: dict[ResidueId, set[ResidueId]] = defaultdict(set)
    for clash in contacts:
        if clash.left_residue_id == clash.right_residue_id:
            continue
        adjacency[clash.left_residue_id].add(clash.right_residue_id)
        adjacency[clash.right_residue_id].add(clash.left_residue_id)

    visited: set[ResidueId] = set()
    clusters: list[frozenset[ResidueId]] = []
    for focus_residue_id in focus_residue_ids:
        if focus_residue_id in visited or focus_residue_id not in adjacency:
            continue
        pending = [focus_residue_id]
        cluster: set[ResidueId] = set()
        while pending:
            residue_id = pending.pop()
            if residue_id in visited:
                continue
            visited.add(residue_id)
            cluster.add(residue_id)
            pending.extend(
                neighbor
                for neighbor in adjacency[residue_id]
                if neighbor not in visited
            )

        if len(cluster) > 1:
            clusters.append(frozenset(cluster))

    return tuple(clusters)


def _proposal_passes_batching_threshold(
    proposal: JointCorrectionScopeProposal,
    *,
    policy: JointCorrectionBatchingPolicy,
) -> bool:
    """Return whether one raw proposal has enough near-covalent burden."""

    return (
        proposal.worst_overlap_angstrom
        >= policy.minimum_cluster_worst_overlap_angstrom
        or proposal.total_overlap_angstrom
        >= policy.minimum_cluster_total_overlap_angstrom
    )


def _compatible_proposal_components(
    proposals: tuple[JointCorrectionScopeProposal, ...],
    *,
    context_residue_ids_by_index: dict[int, frozenset[ResidueId]],
) -> tuple[tuple[int, ...], ...]:
    """Return connected proposal components under the merge-compatibility relation."""

    adjacency: dict[int, set[int]] = {
        proposal_index: set()
        for proposal_index in range(len(proposals))
    }
    for left_index, left_proposal in enumerate(proposals):
        for right_index in range(left_index + 1, len(proposals)):
            right_proposal = proposals[right_index]
            if not _proposals_are_merge_compatible(
                left_proposal,
                right_proposal,
                left_context_residue_ids=context_residue_ids_by_index[left_index],
                right_context_residue_ids=context_residue_ids_by_index[right_index],
            ):
                continue
            adjacency[left_index].add(right_index)
            adjacency[right_index].add(left_index)

    visited: set[int] = set()
    components: list[tuple[int, ...]] = []
    for proposal_index in range(len(proposals)):
        if proposal_index in visited:
            continue
        pending = [proposal_index]
        component: list[int] = []
        while pending:
            active_index = pending.pop()
            if active_index in visited:
                continue
            visited.add(active_index)
            component.append(active_index)
            pending.extend(
                neighbor_index
                for neighbor_index in adjacency[active_index]
                if neighbor_index not in visited
            )
        components.append(tuple(sorted(component)))

    return tuple(components)


def _proposals_are_merge_compatible(
    left_proposal: JointCorrectionScopeProposal,
    right_proposal: JointCorrectionScopeProposal,
    *,
    left_context_residue_ids: frozenset[ResidueId],
    right_context_residue_ids: frozenset[ResidueId],
) -> bool:
    """Return whether two raw proposals should share one correction scope."""

    if set(left_proposal.residue_ids) & set(right_proposal.residue_ids):
        return True

    return bool(left_context_residue_ids & right_context_residue_ids)


def _proposal_context_residue_ids(
    proposal: JointCorrectionScopeProposal,
    *,
    residue_atom_positions: dict[ResidueId, tuple[Vec3, ...]],
    compatibility_context_distance_angstrom: float,
) -> frozenset[ResidueId]:
    """Return the residue neighborhood covered by one proposal's local context."""

    proposal_residue_ids = frozenset(proposal.residue_ids)
    selected_positions = tuple(
        position
        for residue_id in proposal_residue_ids
        for position in residue_atom_positions[residue_id]
    )
    context_residue_ids = set(proposal_residue_ids)
    for residue_id, residue_positions in residue_atom_positions.items():
        if residue_id in context_residue_ids:
            continue
        if _positions_overlap_within_distance(
            selected_positions,
            residue_positions,
            compatibility_context_distance_angstrom=(
                compatibility_context_distance_angstrom
            ),
        ):
            context_residue_ids.add(residue_id)

    return frozenset(context_residue_ids)


def _positions_overlap_within_distance(
    left_positions: tuple[Vec3, ...],
    right_positions: tuple[Vec3, ...],
    *,
    compatibility_context_distance_angstrom: float,
) -> bool:
    """Return whether two atom-position clouds overlap within one distance bound."""

    return any(
        left_position.distance_to(right_position)
        <= compatibility_context_distance_angstrom
        for left_position in left_positions
        for right_position in right_positions
    )

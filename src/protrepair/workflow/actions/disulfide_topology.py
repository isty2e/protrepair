"""Planner-visible disulfide topology resolution action."""

from dataclasses import dataclass

from protrepair.diagnostics.events import EventScope, RepairEvent, ResidueAtomImpact
from protrepair.diagnostics.kinds import RepairEventKind
from protrepair.diagnostics.topology import LikelyDisulfideBond
from protrepair.scope import ResidueSetScope
from protrepair.state.structure_topology import StructureDisulfideTopologyFacts
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.base import ProjectedCodomainState, ProjectedDomainState
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.actions.base import WorkflowStructureTransformer
from protrepair.workflow.actions.context import TransformerExecutionContext


@dataclass(frozen=True, slots=True)
class DisulfideTopologyResolutionTransformer(WorkflowStructureTransformer):
    """Promote one endpoint-disjoint batch of likely disulfides to topology."""

    candidates: tuple[LikelyDisulfideBond, ...]

    def __post_init__(self) -> None:
        candidates = tuple(
            sorted(self.candidates, key=LikelyDisulfideBond.residue_pair)
        )
        if not candidates:
            raise ValueError(
                "disulfide topology resolution requires at least one candidate"
            )
        candidate_pairs = tuple(candidate.residue_pair() for candidate in candidates)
        if len(set(candidate_pairs)) != len(candidate_pairs):
            raise ValueError(
                "disulfide topology resolution candidates must not repeat"
            )
        endpoint_residue_ids = tuple(
            residue_id
            for candidate in candidates
            for residue_id in candidate.residue_pair()
        )
        if len(set(endpoint_residue_ids)) != len(endpoint_residue_ids):
            raise ValueError(
                "disulfide topology resolution candidates must not share endpoints"
            )
        object.__setattr__(self, "candidates", candidates)

    @property
    def workflow_scope(self) -> ResidueSetScope:
        """Return all candidate cysteine residues transformed by this action."""

        return ResidueSetScope(
            residue_ids=tuple(
                residue_id
                for candidate in self.candidates
                for residue_id in candidate.residue_pair()
            )
        )

    def accepts_projected_domain(
        self,
        projected_domain: ProjectedDomainState[ProteinStructure],
        *,
        context: TransformerExecutionContext,
    ) -> bool:
        """Return whether every planned candidate remains promotable."""

        del context
        active_candidates = frozenset(
            StructureDisulfideTopologyFacts.from_structure(
                projected_domain.state
            ).promotable_candidates
        )
        return all(candidate in active_candidates for candidate in self.candidates)

    def transform_projected_domain(
        self,
        projected_domain: ProjectedDomainState[ProteinStructure],
        *,
        carrier: TransformationResult,
        context: TransformerExecutionContext,
    ) -> ProjectedCodomainState[ProteinStructure]:
        """Return the structure with explicit evidence-resolved S-S bonds."""

        del carrier, context
        structure = projected_domain.state
        added_bonds = tuple(
            _topology_bond_for_candidate(structure, candidate)
            for candidate in self.candidates
        )
        updated_topology = StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=structure.topology.bonds + added_bonds,
        )
        updated_structure = type(structure).from_payload(
            constitution=structure.constitution,
            geometry=structure.geometry,
            topology=updated_topology,
            polymer_blueprint=structure.polymer_blueprint,
            provenance=structure.provenance,
        )
        return ProjectedCodomainState(
            scope=self.workflow_scope,
            state=updated_structure,
            repairs=tuple(
                _repair_event_for_candidate(candidate)
                for candidate in self.candidates
            ),
        )


def _topology_bond_for_candidate(
    structure: ProteinStructure,
    candidate: LikelyDisulfideBond,
) -> TopologyBond:
    """Return one canonical evidence-resolved disulfide bond."""

    return TopologyBond(
        atom_index_1=structure.constitution.atom_index(
            AtomRef(candidate.left_residue_id, "SG")
        ),
        atom_index_2=structure.constitution.atom_index(
            AtomRef(candidate.right_residue_id, "SG")
        ),
        order=1,
        relationship_type=BondRelationshipType.DISULFIDE,
        provenance=BondProvenance.EVIDENCE_RESOLVED,
    )


def _repair_event_for_candidate(
    candidate: LikelyDisulfideBond,
) -> RepairEvent:
    """Return one repair event for an evidence-resolved disulfide."""

    return RepairEvent(
        kind=RepairEventKind.DISULFIDE_TOPOLOGY_RESOLVED,
        scope=EventScope.for_residue_pair(*candidate.residue_pair()),
        residue_impacts=tuple(
            ResidueAtomImpact(
                residue_id=residue_id,
                component_id="CYS",
                atom_names=("SG",),
            )
            for residue_id in candidate.residue_pair()
        ),
        details=(
            "promoted unique SG-SG geometry evidence at "
            f"{candidate.sg_distance_angstrom:.3f} A"
        ),
    )

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
    """Publish one endpoint-disjoint batch of disulfide resolutions.

    Parameters
    ----------
    candidates : tuple[LikelyDisulfideBond, ...], default=()
        Unique geometric candidates to revalidate before publication.
    source_atom_ref_pairs : tuple[tuple[AtomRef, AtomRef], ...], default=()
        Untyped source pairs to revalidate against the current chemical graph.
        Their source declarations remain attached to the resolved bonds.

    Raises
    ------
    ValueError
        The batch is empty, contains invalid pairs or shares endpoints.
    """

    candidates: tuple[LikelyDisulfideBond, ...] = ()
    source_atom_ref_pairs: tuple[tuple[AtomRef, AtomRef], ...] = ()

    def __post_init__(self) -> None:
        candidates = tuple(
            sorted(self.candidates, key=LikelyDisulfideBond.residue_pair)
        )
        source_pairs = tuple(sorted(self.source_atom_ref_pairs))
        if any(
            len(pair) != 2
            or pair[0] >= pair[1]
            or pair[0].residue_id == pair[1].residue_id
            or any(ref.atom_name != "SG" for ref in pair)
            for pair in source_pairs
        ):
            raise ValueError("source disulfide candidates require canonical SG pairs")
        if not candidates and not source_pairs:
            raise ValueError(
                "disulfide topology resolution requires at least one candidate"
            )
        candidate_pairs = tuple(candidate.residue_pair() for candidate in candidates)
        if len(set(candidate_pairs)) != len(candidate_pairs):
            raise ValueError("disulfide topology resolution candidates must not repeat")
        endpoint_residue_ids = tuple(
            residue_id
            for candidate in candidates
            for residue_id in candidate.residue_pair()
        ) + tuple(ref.residue_id for pair in source_pairs for ref in pair)
        if len(set(endpoint_residue_ids)) != len(endpoint_residue_ids):
            raise ValueError(
                "disulfide topology resolution candidates must not share endpoints"
            )
        object.__setattr__(self, "candidates", candidates)
        object.__setattr__(self, "source_atom_ref_pairs", source_pairs)

    @property
    def workflow_scope(self) -> ResidueSetScope:
        """Return all candidate cysteine residues transformed by this action."""

        return ResidueSetScope(
            residue_ids=tuple(
                residue_id
                for candidate in self.candidates
                for residue_id in candidate.residue_pair()
            )
            + tuple(
                ref.residue_id for pair in self.source_atom_ref_pairs for ref in pair
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
        facts = StructureDisulfideTopologyFacts.from_structure(projected_domain.state)
        active_candidates = frozenset(facts.promotable_candidates)
        return all(
            candidate in active_candidates for candidate in self.candidates
        ) and all(
            pair in facts.source_atom_ref_pairs for pair in self.source_atom_ref_pairs
        )

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
        source_resolutions = {}
        for pair in self.source_atom_ref_pairs:
            first_index = structure.constitution.atom_index(pair[0])
            second_index = structure.constitution.atom_index(pair[1])
            existing = structure.topology.bond_between(first_index, second_index)
            assert existing is not None
            resolved = existing.with_resolved_chemistry(
                TopologyBond(
                    first_index,
                    second_index,
                    relationship_type=BondRelationshipType.DISULFIDE,
                    provenance=BondProvenance.EVIDENCE_RESOLVED,
                )
            )
            source_resolutions[resolved.endpoint_pair()] = resolved

        updated_topology = StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=tuple(
                source_resolutions.get(bond.endpoint_pair(), bond)
                for bond in structure.topology.bonds
            )
            + added_bonds,
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
                _repair_event_for_candidate(candidate) for candidate in self.candidates
            )
            + tuple(
                RepairEvent(
                    kind=RepairEventKind.DISULFIDE_TOPOLOGY_RESOLVED,
                    scope=EventScope.for_residue_pair(
                        *(ref.residue_id for ref in pair)
                    ),
                    residue_impacts=tuple(
                        ResidueAtomImpact(
                            residue_id=ref.residue_id,
                            component_id="CYS",
                            atom_names=("SG",),
                        )
                        for ref in pair
                    ),
                    details="resolved exclusive neutral CYS SG-SG source connectivity",
                )
                for pair in self.source_atom_ref_pairs
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

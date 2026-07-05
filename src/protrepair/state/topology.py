"""Shared topology-readiness basis facts and observations."""

from collections.abc import Collection
from dataclasses import dataclass
from enum import Enum

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.topology import (
    template_heavy_bond_definitions_for_present_atoms,
    template_hydrogen_bond_definitions_for_names,
)
from protrepair.scope import ResidueSetScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot


class TopologyAvailabilityState(str, Enum):
    """Observed topology-availability state for one selected residue."""

    NOT_APPLICABLE = "not_applicable"
    ABSENT = "absent"
    PRESENT = "present"


class TopologyAvailabilityAspect(str, Enum):
    """Observed topology-availability aspect for one selected residue."""

    HEAVY_ATOM_CONNECTIVITY = "heavy_atom_connectivity"
    HYDROGEN_ATTACHMENTS = "hydrogen_attachments"


def residue_bond_topology_availability_state(
    structure: ProteinStructure,
    residue: ResidueSite,
    *,
    expected_bond_definitions: tuple[BondDefinition, ...],
    empty_state: TopologyAvailabilityState,
    covalent_like_endpoint_pairs: Collection[tuple[AtomIndex, AtomIndex]] | None = None,
) -> TopologyAvailabilityState:
    """Return whether expected residue-local bonds exist in canonical topology."""

    if not expected_bond_definitions:
        return empty_state

    endpoint_pairs = (
        structure.topology.covalent_like_endpoint_pairs()
        if covalent_like_endpoint_pairs is None
        else covalent_like_endpoint_pairs
    )
    for bond_definition in expected_bond_definitions:
        if not _has_residue_topology_bond(
            structure,
            residue.residue_id,
            bond_definition.atom_name_1,
            bond_definition.atom_name_2,
            covalent_like_endpoint_pairs=endpoint_pairs,
        ):
            return TopologyAvailabilityState.ABSENT

    return TopologyAvailabilityState.PRESENT


@dataclass(frozen=True, slots=True)
class ResidueTopologyAvailabilityFact:
    """Canonical topology-availability fact for one selected residue."""

    residue_id: ResidueId
    aspect: TopologyAvailabilityAspect
    state: TopologyAvailabilityState


@dataclass(frozen=True, slots=True)
class TopologyAvailabilityFacts:
    """Primitive topology-availability facts over one selected residue scope."""

    carrier: ProteinStructure
    scope: ResidueSetScope
    residue_facts: tuple[ResidueTopologyAvailabilityFact, ...]

    def __post_init__(self) -> None:
        scoped_residue_ids = tuple(self.scope.residue_ids)
        fact_residue_ids = tuple(
            dict.fromkeys(
                residue_fact.residue_id for residue_fact in self.residue_facts
            )
        )
        if fact_residue_ids != scoped_residue_ids:
            raise ValueError(
                "topology availability facts must cover exactly the scoped residues"
            )

    @classmethod
    def from_projection(
        cls,
        snapshot: ProteinStructureSnapshot,
        *,
        residue_ids: tuple[ResidueId, ...],
        component_library: ComponentLibrary | None,
    ) -> "TopologyAvailabilityFacts":
        """Derive topology-availability facts over one selected residue projection."""

        library = (
            build_default_component_library()
            if component_library is None
            else component_library
        )
        covalent_like_endpoint_pairs = (
            snapshot.structure.topology.covalent_like_endpoint_pairs()
        )
        residue_facts_by_key: dict[
            tuple[ResidueId, TopologyAvailabilityAspect],
            ResidueTopologyAvailabilityFact,
        ] = {}
        for residue_id in residue_ids:
            residue = snapshot.structure.constitution.residue_or_ligand(residue_id)
            if residue is None:
                continue

            template = library.get(residue.component_id)
            if template is None:
                for aspect in TopologyAvailabilityAspect:
                    residue_facts_by_key[(residue.residue_id, aspect)] = (
                        ResidueTopologyAvailabilityFact(
                            residue_id=residue.residue_id,
                            aspect=aspect,
                            state=TopologyAvailabilityState.ABSENT,
                        )
                    )
                continue

            present_heavy_atom_names = {
                atom_site.name
                for atom_site in residue.atom_sites
                if atom_site.element != "H"
            }
            expected_heavy_atom_names = set(template.expected_heavy_atom_names())
            if not expected_heavy_atom_names.issubset(present_heavy_atom_names):
                for aspect in TopologyAvailabilityAspect:
                    residue_facts_by_key[(residue.residue_id, aspect)] = (
                        ResidueTopologyAvailabilityFact(
                            residue_id=residue.residue_id,
                            aspect=aspect,
                            state=TopologyAvailabilityState.ABSENT,
                        )
                )
                continue

            heavy_state = residue_bond_topology_availability_state(
                snapshot.structure,
                residue,
                expected_bond_definitions=(
                    template_heavy_bond_definitions_for_present_atoms(
                        residue,
                        template=template,
                    )
                ),
                empty_state=TopologyAvailabilityState.PRESENT,
                covalent_like_endpoint_pairs=covalent_like_endpoint_pairs,
            )
            residue_facts_by_key[
                (
                    residue.residue_id,
                    TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
                )
            ] = ResidueTopologyAvailabilityFact(
                residue_id=residue.residue_id,
                aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
                state=heavy_state,
            )

            if not template.can_add_hydrogens():
                residue_facts_by_key[
                    (
                        residue.residue_id,
                        TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
                    )
                ] = ResidueTopologyAvailabilityFact(
                    residue_id=residue.residue_id,
                    aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
                    state=TopologyAvailabilityState.NOT_APPLICABLE,
                )
                continue

            present_hydrogen_atom_names = tuple(
                atom_site.name
                for atom_site in residue.atom_sites
                if atom_site.element == "H"
            )
            if not present_hydrogen_atom_names:
                hydrogen_state = TopologyAvailabilityState.ABSENT
            else:
                hydrogen_state = residue_bond_topology_availability_state(
                    snapshot.structure,
                    residue,
                    expected_bond_definitions=(
                        template_hydrogen_bond_definitions_for_names(
                            template,
                            hydrogen_atom_names=present_hydrogen_atom_names,
                        )
                    ),
                    empty_state=TopologyAvailabilityState.ABSENT,
                    covalent_like_endpoint_pairs=covalent_like_endpoint_pairs,
                )

            residue_facts_by_key[
                (
                    residue.residue_id,
                    TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
                )
            ] = ResidueTopologyAvailabilityFact(
                residue_id=residue.residue_id,
                aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
                state=hydrogen_state,
            )

        for residue_id in residue_ids:
            for aspect in TopologyAvailabilityAspect:
                residue_facts_by_key.setdefault(
                    (residue_id, aspect),
                    ResidueTopologyAvailabilityFact(
                        residue_id=residue_id,
                        aspect=aspect,
                        state=TopologyAvailabilityState.ABSENT,
                    ),
                )

        return cls(
            carrier=snapshot.structure,
            scope=ResidueSetScope(residue_ids=residue_ids),
            residue_facts=tuple(residue_facts_by_key.values()),
        )


def _has_residue_topology_bond(
    structure: ProteinStructure,
    residue_id: ResidueId,
    atom_name_1: str,
    atom_name_2: str,
    *,
    covalent_like_endpoint_pairs: Collection[tuple[AtomIndex, AtomIndex]],
) -> bool:
    atom_index_1 = structure.constitution.resolve_atom_index(
        AtomRef(residue_id, atom_name_1)
    )
    atom_index_2 = structure.constitution.resolve_atom_index(
        AtomRef(residue_id, atom_name_2)
    )
    if atom_index_1 is None or atom_index_2 is None:
        return False

    endpoint_pair = (
        (atom_index_2, atom_index_1)
        if atom_index_2.value < atom_index_1.value
        else (atom_index_1, atom_index_2)
    )
    return endpoint_pair in covalent_like_endpoint_pairs


class HydrogenAttachmentResolutionState(str, Enum):
    """Observed hydrogen-attachment resolution detail for one selected residue."""

    NOT_APPLICABLE = "not_applicable"
    ABSENT = "absent"
    TEMPLATE_RESOLVED = "template_resolved"
    COORDINATE_INFERRED = "coordinate_inferred"


@dataclass(frozen=True, slots=True)
class ResidueHydrogenAttachmentResolutionFact:
    """Hydrogen-attachment resolution fact for one selected residue."""

    residue_id: ResidueId
    state: HydrogenAttachmentResolutionState


@dataclass(frozen=True, slots=True)
class HydrogenAttachmentResolutionFacts:
    """Primitive hydrogen-attachment resolution facts over one selected scope."""

    carrier: ProteinStructure
    scope: ResidueSetScope
    residue_facts: tuple[ResidueHydrogenAttachmentResolutionFact, ...]

    def __post_init__(self) -> None:
        scoped_residue_ids = tuple(self.scope.residue_ids)
        fact_residue_ids = tuple(
            dict.fromkeys(
                residue_fact.residue_id for residue_fact in self.residue_facts
            )
        )
        if fact_residue_ids != scoped_residue_ids:
            raise ValueError(
                "hydrogen attachment resolution facts must cover exactly the scoped "
                "residues"
            )

    @classmethod
    def from_projection(
        cls,
        snapshot: ProteinStructureSnapshot,
        *,
        residue_ids: tuple[ResidueId, ...],
        component_library: ComponentLibrary | None,
    ) -> "HydrogenAttachmentResolutionFacts":
        """Derive hydrogen-attachment resolution detail over one residue projection."""

        library = (
            build_default_component_library()
            if component_library is None
            else component_library
        )
        residue_facts: list[ResidueHydrogenAttachmentResolutionFact] = []
        for residue_id in residue_ids:
            residue = snapshot.structure.constitution.residue_or_ligand(residue_id)
            if residue is None:
                continue

            template = library.get(residue.component_id)
            if template is None:
                residue_facts.append(
                    ResidueHydrogenAttachmentResolutionFact(
                        residue_id=residue_id,
                        state=HydrogenAttachmentResolutionState.ABSENT,
                    )
                )
                continue

            if not template.can_add_hydrogens():
                residue_facts.append(
                    ResidueHydrogenAttachmentResolutionFact(
                        residue_id=residue_id,
                        state=HydrogenAttachmentResolutionState.NOT_APPLICABLE,
                    )
                )
                continue

            present_hydrogen_atom_names = tuple(
                atom_site.name
                for atom_site in residue.atom_sites
                if atom_site.element == "H"
            )
            if not present_hydrogen_atom_names:
                residue_facts.append(
                    ResidueHydrogenAttachmentResolutionFact(
                        residue_id=residue_id,
                        state=HydrogenAttachmentResolutionState.ABSENT,
                    )
                )
                continue

            if template.requires_coordinate_hydrogen_topology(
                present_hydrogen_atom_names
            ):
                residue_facts.append(
                    ResidueHydrogenAttachmentResolutionFact(
                        residue_id=residue_id,
                        state=HydrogenAttachmentResolutionState.COORDINATE_INFERRED,
                    )
                )
                continue

            residue_facts.append(
                ResidueHydrogenAttachmentResolutionFact(
                    residue_id=residue_id,
                    state=HydrogenAttachmentResolutionState.TEMPLATE_RESOLVED,
                )
            )

        covered_residue_ids = {
            residue_fact.residue_id for residue_fact in residue_facts
        }
        residue_facts.extend(
            ResidueHydrogenAttachmentResolutionFact(
                residue_id=residue_id,
                state=HydrogenAttachmentResolutionState.ABSENT,
            )
            for residue_id in residue_ids
            if residue_id not in covered_residue_ids
        )
        return cls(
            carrier=snapshot.structure,
            scope=ResidueSetScope(residue_ids=residue_ids),
            residue_facts=tuple(residue_facts),
        )

    def any_coordinate_inferred(self) -> bool:
        """Return whether any selected residue needs coordinate-derived attachment."""

        return any(
            residue_fact.state is HydrogenAttachmentResolutionState.COORDINATE_INFERRED
            for residue_fact in self.residue_facts
        )


@dataclass(frozen=True, slots=True)
class TopologyAvailabilityObservation:
    """Per-residue topology-availability facts over one selected scope."""

    residue_facts: tuple[ResidueTopologyAvailabilityFact, ...]

    def __post_init__(self) -> None:
        if not self.residue_facts:
            raise ValueError(
                "topology availability observation requires at least one residue fact"
            )

        ordered_residue_facts = tuple(
            sorted(
                self.residue_facts,
                key=lambda fact: (fact.residue_id, fact.aspect.value),
            )
        )
        fact_keys = tuple(
            (residue_fact.residue_id, residue_fact.aspect)
            for residue_fact in ordered_residue_facts
        )
        if len(set(fact_keys)) != len(fact_keys):
            raise ValueError(
                "topology availability observation must not repeat residue/aspect facts"
            )

        object.__setattr__(self, "residue_facts", ordered_residue_facts)

    @classmethod
    def uniform(
        cls,
        residue_ids: tuple[ResidueId, ...],
        *,
        heavy_atom_state: TopologyAvailabilityState,
        hydrogen_state: TopologyAvailabilityState,
    ) -> "TopologyAvailabilityObservation":
        """Return one uniform observation over the given residue identifiers."""

        return cls(
            residue_facts=tuple(
                residue_fact
                for residue_id in residue_ids
                for residue_fact in (
                    ResidueTopologyAvailabilityFact(
                        residue_id=residue_id,
                        aspect=TopologyAvailabilityAspect.HEAVY_ATOM_CONNECTIVITY,
                        state=heavy_atom_state,
                    ),
                    ResidueTopologyAvailabilityFact(
                        residue_id=residue_id,
                        aspect=TopologyAvailabilityAspect.HYDROGEN_ATTACHMENTS,
                        state=hydrogen_state,
                    ),
                )
            )
        )

    @classmethod
    def from_facts(
        cls,
        facts: TopologyAvailabilityFacts,
    ) -> "TopologyAvailabilityObservation":
        """Rebuild topology-availability observation from primitive facts."""

        return cls(residue_facts=facts.residue_facts)

    def is_uniform(
        self,
        *,
        aspect: TopologyAvailabilityAspect,
        state: TopologyAvailabilityState,
    ) -> bool:
        """Return whether one topology aspect is uniform over selected residues."""

        return all(
            residue_fact.state is state
            for residue_fact in self.residue_facts
            if residue_fact.aspect is aspect
        )

    def state_for(
        self,
        residue_id: ResidueId,
        *,
        aspect: TopologyAvailabilityAspect,
    ) -> TopologyAvailabilityState:
        """Return one topology-availability state for one residue/aspect pair."""

        for residue_fact in self.residue_facts:
            if residue_fact.residue_id == residue_id and residue_fact.aspect is aspect:
                return residue_fact.state

        raise KeyError(
            f"{residue_id.display_token()} / {aspect.value} is not part of this "
            "topology observation"
        )

    def facts_for_residue(
        self,
        residue_id: ResidueId,
    ) -> tuple[ResidueTopologyAvailabilityFact, ...]:
        """Return all topology facts recorded for one selected residue."""

        return tuple(
            residue_fact
            for residue_fact in self.residue_facts
            if residue_fact.residue_id == residue_id
        )


@dataclass(frozen=True, slots=True)
class HydrogenAttachmentResolutionObservation:
    """Per-residue hydrogen-attachment resolution detail over one scope."""

    residue_facts: tuple[ResidueHydrogenAttachmentResolutionFact, ...]

    def __post_init__(self) -> None:
        if not self.residue_facts:
            raise ValueError(
                "hydrogen attachment resolution observation requires at least one "
                "residue fact"
            )

        ordered_residue_facts = tuple(
            sorted(self.residue_facts, key=lambda fact: fact.residue_id)
        )
        fact_residue_ids = tuple(
            residue_fact.residue_id for residue_fact in ordered_residue_facts
        )
        if len(set(fact_residue_ids)) != len(fact_residue_ids):
            raise ValueError(
                "hydrogen attachment resolution observation must not repeat residues"
            )

        object.__setattr__(self, "residue_facts", ordered_residue_facts)

    @classmethod
    def uniform(
        cls,
        residue_ids: tuple[ResidueId, ...],
        *,
        state: HydrogenAttachmentResolutionState,
    ) -> "HydrogenAttachmentResolutionObservation":
        """Return one uniform observation over the given residue identifiers."""

        return cls(
            residue_facts=tuple(
                ResidueHydrogenAttachmentResolutionFact(
                    residue_id=residue_id,
                    state=state,
                )
                for residue_id in residue_ids
            )
        )

    @classmethod
    def from_facts(
        cls,
        facts: HydrogenAttachmentResolutionFacts,
    ) -> "HydrogenAttachmentResolutionObservation":
        """Rebuild hydrogen-attachment resolution observation from facts."""

        return cls(residue_facts=facts.residue_facts)

    def is_uniform(
        self,
        *,
        state: HydrogenAttachmentResolutionState,
    ) -> bool:
        """Return whether one resolution state is uniform over selected residues."""

        return all(residue_fact.state is state for residue_fact in self.residue_facts)

    def state_for(
        self,
        residue_id: ResidueId,
    ) -> HydrogenAttachmentResolutionState:
        """Return the hydrogen-attachment resolution state for one residue."""

        for residue_fact in self.residue_facts:
            if residue_fact.residue_id == residue_id:
                return residue_fact.state

        raise KeyError(
            f"{residue_id.display_token()} is not part of this hydrogen attachment "
            "resolution observation"
        )

    def any_coordinate_inferred(self) -> bool:
        """Return whether any selected residue needs coordinate-derived attachment."""

        return any(
            residue_fact.state is HydrogenAttachmentResolutionState.COORDINATE_INFERRED
            for residue_fact in self.residue_facts
        )

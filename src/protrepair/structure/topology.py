"""Structure topology facets aligned to constitution-owned atom slots."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from enum import Enum
from math import isfinite
from types import MappingProxyType

from protrepair.errors import ModelInvariantError
from protrepair.structure.address_space import (
    StructureAddressSpaceKey,
    atom_count_in_address_space,
)
from protrepair.structure.constitution import StructureConstitution
from protrepair.structure.slots import AtomIndex, ResidueIndex


@dataclass(frozen=True, slots=True)
class AtomTopology:
    """Topology payload attached to one constitution-native atom slot."""

    formal_charge: int | None = None

    def __post_init__(self) -> None:
        formal_charge = self.formal_charge
        if formal_charge is not None and (
            isinstance(formal_charge, bool) or not isinstance(formal_charge, int)
        ):
            raise TypeError("atom topology formal_charge must be an integer or None")


class BondProvenance(str, Enum):
    """Evidence family supporting one canonical topology bond.

    This is a support-mode axis, not a lifecycle or egress axis. A bond created
    during repair can still be template-resolved, sequence-inferred, evidence-
    resolved, or repair-inferred depending on what justified the endpoint pair.
    Writers and readiness checks must project from provenance plus their own
    boundary context instead of treating provenance as a serialization flag.
    """

    SOURCE_EXPLICIT = "source_explicit"
    TEMPLATE_RESOLVED = "template_resolved"
    SEQUENCE_INFERRED = "sequence_inferred"
    EVIDENCE_RESOLVED = "evidence_resolved"
    REPAIR_INFERRED = "repair_inferred"


class BondRelationshipType(str, Enum):
    """Physical relationship type of a canonical topology bond.

    DISULFIDE is topology truth only when supplied by source-explicit records,
    chemistry/template evidence, or an explicit topology-writing transformer.
    Geometry-only SG-SG proximity is a diagnostic/execution candidate, not
    default-ingress topology truth.
    """

    COVALENT = "covalent"
    DISULFIDE = "disulfide"
    HYDROGEN_BOND = "hydrogen_bond"
    METAL_COORDINATION = "metal_coordination"
    UNKNOWN = "unknown"


class SourceBondRecordType(str, Enum):
    """Boundary format that originated a source-explicit topology bond."""

    PDB_LINK = "pdb_link"
    PDB_CONECT = "pdb_conect"
    MMCIF_STRUCT_CONN = "mmcif_struct_conn"


@dataclass(frozen=True, slots=True)
class SourceBondMetadata:
    """Source-origin metadata for one SOURCE_EXPLICIT topology bond."""

    record_type: SourceBondRecordType
    source_id: str | None = None
    reported_distance_angstrom: float | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.record_type, SourceBondRecordType):
            raise TypeError(
                "source bond metadata record_type must be a SourceBondRecordType"
            )

        source_id = None if self.source_id is None else self.source_id.strip() or None
        reported_distance = self.reported_distance_angstrom
        if reported_distance is not None:
            if isinstance(reported_distance, bool):
                raise ValueError(
                    "source bond metadata reported distance must be finite, positive, "
                    "or None"
                )
            try:
                reported_distance = float(reported_distance)
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "source bond metadata reported distance must be finite, positive, "
                    "or None"
                ) from error
            if not isfinite(reported_distance) or reported_distance <= 0.0:
                raise ValueError(
                    "source bond metadata reported distance must be finite, positive, "
                    "or None"
                )

        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "reported_distance_angstrom", reported_distance)


@dataclass(frozen=True, slots=True)
class TopologyBond:
    """One canonical bond in the structure topology bond graph."""

    atom_index_1: AtomIndex
    atom_index_2: AtomIndex
    order: int = 1
    aromatic: bool = False
    relationship_type: BondRelationshipType = BondRelationshipType.COVALENT
    provenance: BondProvenance = BondProvenance.TEMPLATE_RESOLVED
    source_metadata: SourceBondMetadata | None = None

    def __post_init__(self) -> None:
        atom_index_1 = self.atom_index_1
        atom_index_2 = self.atom_index_2
        if atom_index_1 == atom_index_2:
            raise ValueError("topology bonds require two distinct atom slots")
        if atom_index_2.value < atom_index_1.value:
            atom_index_1, atom_index_2 = atom_index_2, atom_index_1

        if isinstance(self.order, bool) or not isinstance(self.order, int):
            raise TypeError("topology bond order must be an integer")
        if self.order <= 0:
            raise ValueError("topology bond order must be positive")

        if not isinstance(self.relationship_type, BondRelationshipType):
            raise TypeError(
                "topology bond relationship_type must be a BondRelationshipType"
            )
        if not isinstance(self.provenance, BondProvenance):
            raise TypeError("topology bond provenance must be a BondProvenance")

        if (
            self.source_metadata is not None
            and self.provenance is not BondProvenance.SOURCE_EXPLICIT
        ):
            raise ValueError(
                "topology bond source_metadata requires SOURCE_EXPLICIT provenance"
            )

        object.__setattr__(self, "atom_index_1", atom_index_1)
        object.__setattr__(self, "atom_index_2", atom_index_2)

    def endpoint_pair(self) -> tuple[AtomIndex, AtomIndex]:
        """Return the canonically ordered endpoint pair."""

        return (self.atom_index_1, self.atom_index_2)

    def involves(self, atom_index: AtomIndex) -> bool:
        """Return whether this bond references one atom slot."""

        return atom_index in (self.atom_index_1, self.atom_index_2)


_COVALENT_LIKE_RELATIONSHIP_TYPES: frozenset[BondRelationshipType] = frozenset(
    {
        BondRelationshipType.COVALENT,
        BondRelationshipType.DISULFIDE,
    }
)


def is_covalent_like_relationship(bond: TopologyBond) -> bool:
    """Return whether a topology bond has covalent-like relationship type."""

    return bond.relationship_type in _COVALENT_LIKE_RELATIONSHIP_TYPES


def is_source_provenance(bond: TopologyBond) -> bool:
    """Return whether a topology bond has source-explicit provenance."""

    return bond.provenance is BondProvenance.SOURCE_EXPLICIT


def is_model_resolved_provenance(bond: TopologyBond) -> bool:
    """Return whether a topology bond was resolved by canonical model policy."""

    return not is_source_provenance(bond)


@dataclass(frozen=True, slots=True, init=False)
class StructureTopology:
    """Structure-level topology aligned to constitution-owned atom slots."""

    atom_topologies: tuple[AtomTopology | None, ...]
    bonds: tuple[TopologyBond, ...]
    _address_space_key: StructureAddressSpaceKey
    _bond_by_endpoint_pair: Mapping[tuple[AtomIndex, AtomIndex], TopologyBond] = field(
        init=False,
        repr=False,
        compare=False,
    )

    def __init__(
        self,
        *,
        constitution: StructureConstitution,
        atom_topologies: tuple[AtomTopology | None, ...],
        bonds: tuple[TopologyBond, ...] = (),
    ) -> None:
        structure_topology = type(self)._from_address_space_key(
            atom_topologies=tuple(atom_topologies),
            bonds=tuple(bonds),
            address_space_key=constitution.address_space_key,
        )
        object.__setattr__(self, "atom_topologies", structure_topology.atom_topologies)
        object.__setattr__(self, "bonds", structure_topology.bonds)
        object.__setattr__(
            self,
            "_bond_by_endpoint_pair",
            structure_topology._bond_by_endpoint_pair,
        )
        object.__setattr__(
            self,
            "_address_space_key",
            structure_topology._address_space_key,
        )

    @classmethod
    def _from_address_space_key(
        cls,
        *,
        atom_topologies: tuple[AtomTopology | None, ...],
        bonds: tuple[TopologyBond, ...] = (),
        address_space_key: StructureAddressSpaceKey,
    ) -> "StructureTopology":
        structure_topology = object.__new__(cls)
        object.__setattr__(
            structure_topology,
            "atom_topologies",
            tuple(atom_topologies),
        )
        object.__setattr__(
            structure_topology,
            "_address_space_key",
            tuple(address_space_key),
        )
        atom_slot_count = atom_count_in_address_space(
            structure_topology._address_space_key
        )
        if len(structure_topology.atom_topologies) != atom_slot_count:
            raise ModelInvariantError(
                "structure topology atom slots must align with the constitution "
                "address space"
            )
        deduplicated_bonds = _deduplicate_topology_bonds(
            bonds, atom_slot_count=atom_slot_count
        )
        object.__setattr__(structure_topology, "bonds", deduplicated_bonds)
        object.__setattr__(
            structure_topology,
            "_bond_by_endpoint_pair",
            MappingProxyType(
                {bond.endpoint_pair(): bond for bond in deduplicated_bonds}
            ),
        )
        return structure_topology

    def is_aligned_to(self, constitution: StructureConstitution) -> bool:
        """Return whether this topology payload matches one constitution."""

        return self._address_space_key == constitution.address_space_key

    @classmethod
    def empty(
        cls,
        *,
        constitution: StructureConstitution,
    ) -> "StructureTopology":
        """Return an empty topology aligned to one constitution payload."""

        return cls(
            constitution=constitution,
            atom_topologies=(None,) * len(constitution.atom_slots),
        )

    def bonds_for_constitution(
        self,
        *,
        source_constitution: StructureConstitution,
        target_constitution: StructureConstitution,
    ) -> tuple[TopologyBond, ...]:
        """Return topology bonds remapped to a target constitution.

        Bonds whose endpoint atom references do not both survive in the target
        constitution are intentionally filtered out.
        """

        if not self.is_aligned_to(source_constitution):
            raise ModelInvariantError(
                "topology bond remapping requires the matching source "
                "constitution address space"
            )

        remapped_bonds: list[TopologyBond] = []
        for bond in self.bonds:
            source_ref_1 = source_constitution.atom_ref_at(bond.atom_index_1)
            source_ref_2 = source_constitution.atom_ref_at(bond.atom_index_2)
            target_atom_index_1 = target_constitution.resolve_atom_index(source_ref_1)
            target_atom_index_2 = target_constitution.resolve_atom_index(source_ref_2)
            if target_atom_index_1 is None or target_atom_index_2 is None:
                continue

            remapped_bonds.append(
                TopologyBond(
                    atom_index_1=target_atom_index_1,
                    atom_index_2=target_atom_index_2,
                    order=bond.order,
                    aromatic=bond.aromatic,
                    relationship_type=bond.relationship_type,
                    provenance=bond.provenance,
                    source_metadata=bond.source_metadata,
                )
            )

        return tuple(remapped_bonds)

    def bond_between(
        self,
        atom_index_1: AtomIndex,
        atom_index_2: AtomIndex,
    ) -> TopologyBond | None:
        """Return the topology bond between two atom slots when present."""

        if atom_index_1 == atom_index_2:
            return None

        endpoint_pair = _canonical_endpoint_pair(atom_index_1, atom_index_2)
        return self._bond_by_endpoint_pair.get(endpoint_pair)

    def covalent_like_endpoint_pairs(self) -> frozenset[tuple[AtomIndex, AtomIndex]]:
        """Return canonical endpoint pairs for all covalent-like topology bonds."""

        return frozenset(
            bond.endpoint_pair()
            for bond in self.bonds
            if is_covalent_like_relationship(bond)
        )

    def atom_count(self) -> int:
        """Return the number of stored atom-topology slots."""

        return len(self.atom_topologies)

    def atom_topology(self, atom_index: AtomIndex) -> AtomTopology | None:
        """Return topology payload for one atom slot when present."""

        return self.atom_topologies[atom_index.value]

    def formal_charge(self, atom_index: AtomIndex) -> int | None:
        """Return formal-charge payload for one atom slot when present."""

        atom_topology = self.atom_topology(atom_index)
        if atom_topology is None:
            return None

        return atom_topology.formal_charge

    def formal_charge_entries(self) -> tuple[tuple[AtomIndex, int | None], ...]:
        """Return explicit formal-charge payload keyed by atom slot index."""

        return tuple(
            (
                AtomIndex(atom_index),
                atom_topology.formal_charge,
            )
            for atom_index, atom_topology in enumerate(self.atom_topologies)
            if atom_topology is not None
        )

    def residue_formal_charge_by_atom_name(
        self,
        *,
        constitution: StructureConstitution,
        residue_index: ResidueIndex,
    ) -> tuple[tuple[str, int | None], ...]:
        """Return residue-local formal-charge payload keyed by atom name."""

        if not self.is_aligned_to(constitution):
            raise ModelInvariantError(
                "structure topology residue projection requires the matching "
                "constitution address space"
            )

        return tuple(
            (
                constitution.atom_site_at(atom_index).name,
                atom_topology.formal_charge,
            )
            for atom_index in constitution.atom_indices_for_residue_index(residue_index)
            for atom_topology in (self.atom_topology(atom_index),)
            if atom_topology is not None
        )


def _canonical_endpoint_pair(
    atom_index_1: AtomIndex,
    atom_index_2: AtomIndex,
) -> tuple[AtomIndex, AtomIndex]:
    """Return a canonically ordered atom endpoint pair."""

    if atom_index_2.value < atom_index_1.value:
        return (atom_index_2, atom_index_1)

    return (atom_index_1, atom_index_2)


def _deduplicate_topology_bonds(
    bonds: tuple[TopologyBond, ...],
    *,
    atom_slot_count: int,
) -> tuple[TopologyBond, ...]:
    """Return topology bonds validated, deduplicated, and canonically sorted."""

    seen: dict[tuple[AtomIndex, AtomIndex], TopologyBond] = {}
    for bond in bonds:
        if (
            bond.atom_index_1.value >= atom_slot_count
            or bond.atom_index_2.value >= atom_slot_count
        ):
            raise ModelInvariantError(
                "topology bond endpoints must reference valid atom slots within "
                "the constitution address space"
            )
        endpoints = bond.endpoint_pair()
        existing = seen.get(endpoints)
        if existing is None:
            seen[endpoints] = bond
            continue
        if existing == bond:
            continue
        raise ModelInvariantError(
            "topology bond graph contains conflicting bonds for the same "
            f"endpoint pair ({endpoints[0].value}, {endpoints[1].value}): "
            "caller must resolve conflicts before constructing topology"
        )
    return tuple(
        bond
        for bond in sorted(
            seen.values(),
            key=lambda b: (b.atom_index_1.value, b.atom_index_2.value),
        )
    )

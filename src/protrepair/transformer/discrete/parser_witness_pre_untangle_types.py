"""Domain types for parser-witness pre-untangle search."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from protrepair.diagnostics.parser_readability import (
    RDKitProximityBondCluster,
    RDKitProximityBondWitness,
)
from protrepair.geometry import Vec3
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.labels import AtomRef
from protrepair.structure.slots import AtomIndex, ResidueIndex


@dataclass(frozen=True, order=True, slots=True)
class ParserWitnessPreUntangleScore:
    """Lexicographic score for parser-witness proximity contacts."""

    unresolved_contact_count: int
    total_overlap_angstrom: float
    worst_overlap_angstrom: float


@dataclass(frozen=True, order=True, slots=True)
class ParserWitnessPreUntangleCandidateRank:
    """Whole-candidate rank for safe parser-witness pre-untangle selection."""

    parser_extra_heavy_proximity_bond_count: int
    focus_near_covalent_contact_count: int
    focus_total_near_covalent_overlap_angstrom: float
    focus_clash_count: int
    target_score: ParserWitnessPreUntangleScore
    absolute_angle_degrees: int
    angle_degrees: int


@dataclass(frozen=True, slots=True)
class ParserWitnessPreUntangleCandidate:
    """One torsion-rotated pre-untangle candidate."""

    structure: ProteinStructure
    moved_atom_indices: tuple[AtomIndex, ...]
    score: ParserWitnessPreUntangleScore
    parser_extra_heavy_proximity_bond_count: int | None = None


@dataclass(frozen=True, slots=True)
class SidechainRootRotationPlan:
    """One safe side-chain rotation plan around the CA-CB root axis."""

    residue_site: ResidueSite
    residue_geometry: ResidueGeometry
    residue_index: ResidueIndex
    axis_atom_names: tuple[str, str]
    rotating_atom_names: frozenset[str]
    formal_charge_by_atom_name: tuple[tuple[str, int | None], ...]


@dataclass(frozen=True, slots=True)
class RotatedSidechainPayload:
    """One residue-local rotation payload before structure materialization."""

    plan: SidechainRootRotationPlan
    angle_degrees: int
    residue_geometry: ResidueGeometry
    moved_atom_names: tuple[str, ...]
    score: ParserWitnessPreUntangleScore


@dataclass(frozen=True, slots=True)
class RankedRotatedSidechainCandidate:
    """One materialized pre-untangle candidate plus its selection rank."""

    candidate: ParserWitnessPreUntangleCandidate
    payload: RotatedSidechainPayload
    rank: ParserWitnessPreUntangleCandidateRank


@dataclass(frozen=True, slots=True)
class MeasuredRotatedSidechainCandidate:
    """One materialized candidate with its exact parser-burden count."""

    candidate: ParserWitnessPreUntangleCandidate
    payload: RotatedSidechainPayload
    parser_extra_heavy_proximity_bond_count: int


@dataclass(frozen=True, slots=True)
class ParserWitnessScoringContext:
    """Cached geometry and target distances for one parser-witness cluster."""

    cluster: RDKitProximityBondCluster
    position_by_atom_ref: Mapping[AtomRef, Vec3 | None]
    target_distance_by_witness: Mapping[RDKitProximityBondWitness, float]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "position_by_atom_ref",
            MappingProxyType(dict(self.position_by_atom_ref)),
        )
        object.__setattr__(
            self,
            "target_distance_by_witness",
            MappingProxyType(dict(self.target_distance_by_witness)),
        )

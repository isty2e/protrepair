"""Canonical structure provenance facets."""

from dataclasses import dataclass, field
from enum import Enum
from itertools import pairwise

from protrepair.chemistry.microstate.resolution import AppliedMicrostateChoice
from protrepair.structure.cap_attachment import PeptideCapAttachment
from protrepair.structure.constitution import ResidueRole, StructureConstitution
from protrepair.structure.element import ElementIdentity
from protrepair.structure.endpoint import StructureEndpoint
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.sequence_boundary import DepositedPeptideSequence
from protrepair.structure.source_connection import (
    PdbAdjacentCapLink,
    SourceAtomIdentity,
)
from protrepair.structure.topology import BondProvenance


class FileFormat(str, Enum):
    """Canonical coordinate file formats recorded in structure provenance."""

    PDB = "pdb"
    MMCIF = "mmcif"


@dataclass(frozen=True, slots=True)
class ZeroOccupancyOmission:
    """Original position and scalars of a zero-occupancy coordinate.

    Parameters
    ----------
    source_identity : SourceAtomIdentity
        Selected atom identity, including component and alternate location.
    element : ElementIdentity
        Source element and isotope, independent of the name.
    geometry : AtomGeometry
        Original coordinate and scalars with an occupancy of zero.

    Raises
    ------
    TypeError
        A field has a noncanonical type.
    ValueError
        The occupancy or alternate location contradicts the omission.
    """

    source_identity: SourceAtomIdentity
    element: ElementIdentity
    geometry: AtomGeometry

    def __post_init__(self) -> None:
        if not isinstance(self.source_identity, SourceAtomIdentity):
            raise TypeError("an occupancy omission requires a SourceAtomIdentity")
        if not isinstance(self.element, ElementIdentity):
            raise TypeError("an occupancy omission requires an ElementIdentity")
        if not isinstance(self.geometry, AtomGeometry):
            raise TypeError("an occupancy omission requires AtomGeometry")
        if self.geometry.occupancy != 0.0:
            raise ValueError("an occupancy omission requires zero occupancy")
        if self.geometry.altloc != self.source_identity.altloc:
            raise ValueError("an occupancy omission must match the source altloc")


@dataclass(frozen=True, slots=True)
class WithheldBackboneHydrogen:
    """Original isotope and parent of a zero hydrogen with no determined position."""

    original: ZeroOccupancyOmission
    parent: AtomRef
    former_bond: BondProvenance | None


@dataclass(frozen=True, slots=True)
class DeferredSidechainOmission:
    """Side-chain decision conditional on restoration of this residue's backbone."""

    residue_id: ResidueId
    component_id: str
    atoms: tuple[ZeroOccupancyOmission, ...]
    refusal: str | None


@dataclass(frozen=True, slots=True)
class ZeroOccupancyResidueOmission:
    """An original internal residue withheld as one coordinate gap.

    Parameters
    ----------
    preceding_residue_id, following_residue_id : ResidueId
        The selected source flanks in chain order.
    atoms : tuple of ZeroOccupancyOmission
        Selected original positions and their common residue identity.

    Raises
    ------
    TypeError
        The flanks or atom records are noncanonical.
    ValueError
        The atoms do not describe a single ordinary-numbered internal slot.
    """

    preceding_residue_id: ResidueId
    following_residue_id: ResidueId
    atoms: tuple[ZeroOccupancyOmission, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.preceding_residue_id, ResidueId) or not isinstance(
            self.following_residue_id, ResidueId
        ):
            raise TypeError("a residue omission requires ResidueId flanks")
        if not isinstance(self.atoms, tuple) or any(
            not isinstance(atom, ZeroOccupancyOmission) for atom in self.atoms
        ):
            raise TypeError("a residue omission requires a tuple of atom omissions")
        if not self.atoms:
            raise ValueError("a residue omission requires original atom records")

        path = (self.preceding_residue_id, self.residue_id, self.following_residue_id)
        if (
            any(residue_id.insertion_code is not None for residue_id in path)
            or not path[0].immediately_precedes(path[1])
            or not path[1].immediately_precedes(path[2])
        ):
            raise ValueError(
                "a residue omission requires consecutive ordinary source ids"
            )
        refs = {atom.source_identity.atom_ref for atom in self.atoms}
        if len(refs) != len(self.atoms) or any(
            atom.source_identity.atom_ref.residue_id != self.residue_id
            or atom.source_identity.component_id != self.component_id
            for atom in self.atoms
        ):
            raise ValueError("omitted atom records must be distinct within one residue")

    @property
    def residue_id(self) -> ResidueId:
        """Return the source residue ID shared by the archived positions."""

        return self.atoms[0].source_identity.atom_ref.residue_id

    @property
    def component_id(self) -> str:
        """Return the source component ID shared by the archived positions."""

        return self.atoms[0].source_identity.component_id

    @property
    def residue_ids(self) -> tuple[ResidueId, ...]:
        """Return the biological positions in source order."""
        return (self.residue_id,)

    @property
    def component_ids(self) -> tuple[str, ...]:
        """Return the biological sequence at the omitted positions."""
        return (self.component_id,)

    def contains_position(self, residue_id: ResidueId) -> bool:
        """Return whether a proposed slot lies strictly between these flanks.

        Parameters
        ----------
        residue_id : ResidueId
            Proposed source slot, including insertion codes.

        Returns
        -------
        bool
            Whether the proposal lies inside this recorded gap.
        """

        return self.preceding_residue_id < residue_id < self.following_residue_id


@dataclass(frozen=True, slots=True)
class ConsecutiveZeroOccupancyResidueOmission:
    """Consecutive zero-occupancy residues withheld for reconstruction together.

    Parameters
    ----------
    preceding_residue_id, following_residue_id : ResidueId
        Residues immediately before and after the omitted interval.
    preceding_component_id, following_component_id : str
        Deposited component identities of the two flanks.
    members : tuple of tuples of ZeroOccupancyOmission
        Selected source atoms grouped by residue in biological order.

    Raises
    ------
    ValueError
        The archive does not describe distinct atoms in at least two consecutive slots.
    TypeError
        A flank or atom archive has a noncanonical type.
    """

    preceding_residue_id: ResidueId
    following_residue_id: ResidueId
    preceding_component_id: str
    following_component_id: str
    members: tuple[tuple[ZeroOccupancyOmission, ...], ...]

    def __post_init__(self) -> None:
        if not all(
            isinstance(r, ResidueId)
            for r in (self.preceding_residue_id, self.following_residue_id)
        ):
            raise TypeError(
                "the residues flanking an omission must have ResidueId addresses"
            )
        if not all(
            isinstance(c, str) and c and c == c.strip().upper()
            for c in (self.preceding_component_id, self.following_component_id)
        ):
            raise ValueError("an omitted interval requires canonical flank components")
        if not isinstance(self.members, tuple) or any(
            not isinstance(group, tuple)
            or any(not isinstance(a, ZeroOccupancyOmission) for a in group)
            for group in self.members
        ):
            raise TypeError("omitted atoms must be grouped into tuples by residue")
        if len(self.members) < 2 or not all(self.members):
            raise ValueError(
                "at least two omitted residues are required, each with deposited atoms"
            )
        for group in self.members:
            first = group[0].source_identity
            if len({a.source_identity.atom_ref for a in group}) != len(group) or any(
                a.source_identity.atom_ref.residue_id != first.atom_ref.residue_id
                or a.source_identity.component_id != first.component_id
                for a in group
            ):
                raise ValueError(
                    "each omitted residue must contain distinct atoms of one component"
                )
        path = (self.preceding_residue_id, *self.residue_ids, self.following_residue_id)
        if any(r.insertion_code is not None for r in path) or not all(
            a.immediately_precedes(b) for a, b in pairwise(path)
        ):
            raise ValueError(
                "the omitted residues and their flanks must have consecutive ordinary source ids"
            )

    @property
    def residue_ids(self) -> tuple[ResidueId, ...]:
        """Return the member addresses in source order."""
        return tuple(
            group[0].source_identity.atom_ref.residue_id for group in self.members
        )

    @property
    def component_ids(self) -> tuple[str, ...]:
        """Return the source sequence that an explicit donor must restore."""
        return tuple(group[0].source_identity.component_id for group in self.members)

    def contains_position(self, residue_id: ResidueId) -> bool:
        """Return whether a proposed slot lies inside the recorded interval."""
        return self.preceding_residue_id < residue_id < self.following_residue_id


@dataclass(frozen=True, slots=True)
class StructureIngress:
    """Input origin and observations, independent of the current structure.

    Parameters
    ----------
    source_format : FileFormat
        Coordinate format of the input.
    source_name : str or None, default=None
        Optional input name; blank names become None.
    observation : StructureObservation or None, default=None
        Selected original input facts. None denotes unavailable observations,
        including manual structures without an explicitly supplied snapshot.
    zero_occupancy_omissions : tuple of ZeroOccupancyOmission or None, default=None
        Side-chain positions withheld from the observation. An empty tuple means
        the policy was applied without omissions; None means it was not applied.
    zero_occupancy_residue_omissions : tuple of ZeroOccupancyResidueOmission or None, default=None
        Whole source residues withheld from the observation. Empty and None
        distinguish applied without omissions and not applied independently
        of the side-chain policy.
    consecutive_zero_occupancy_residue_omissions : tuple of ConsecutiveZeroOccupancyResidueOmission or None, default=None
        Groups of consecutive residues withheld together. None means the option
        was off; an empty tuple means no interval qualified.
    pdb_adjacent_cap_links : tuple of PdbAdjacentCapLink, default=()
        Selected bare CONECT attachments corroborated by source PDB slot order.
        These are source observations, not realized peptide bonds.
    uncertain_peptide_link_ports : frozenset[AtomRef] or None, default=None
        N/C addresses on incomplete, conflicting or non-local declarations.
        None denotes an input whose raw records were not inspected.
    deposited_peptide_sequences : tuple[DepositedPeptideSequence, ...]
        Verified constructs with their original coordinate position intervals.
    backbone_neighbors : frozenset of residue ID pairs
        Directed peptide neighbors verified without using the C--N distance.
    backbone_link_vetoes : frozenset of AtomRef
        Ambiguous source declarations that prevent using a port as a fit anchor.
    deferred_sidechains : tuple of DeferredSidechainOmission
        Side-chain omission decisions to revisit if the backbone is rebuilt.

    Raises
    ------
    TypeError
        The source format, observation, omissions or cap links have noncanonical types.
    ValueError
        Omission history has no source observation, or an omitted atom appears
        in the observation or more than once in the history.
        A cap link does not match a unique retained cap in the PDB observation.
    """

    source_format: FileFormat
    source_name: str | None = None
    observation: StructureObservation | None = field(default=None, repr=False)
    zero_occupancy_omissions: tuple[ZeroOccupancyOmission, ...] | None = field(
        default=None, kw_only=True, repr=False
    )
    zero_occupancy_residue_omissions: (
        tuple[ZeroOccupancyResidueOmission, ...] | None
    ) = field(default=None, kw_only=True, repr=False)
    consecutive_zero_occupancy_residue_omissions: (
        tuple[ConsecutiveZeroOccupancyResidueOmission, ...] | None
    ) = field(default=None, kw_only=True, repr=False)
    pdb_adjacent_cap_links: tuple[PdbAdjacentCapLink, ...] = field(
        default=(), kw_only=True, repr=False
    )
    uncertain_peptide_link_ports: frozenset[AtomRef] | None = field(
        default=None, kw_only=True, repr=False
    )
    deposited_peptide_sequences: tuple[DepositedPeptideSequence, ...] = field(
        default=(), kw_only=True, repr=False
    )
    backbone_neighbors: frozenset[tuple[ResidueId, ResidueId]] = field(
        default_factory=frozenset, kw_only=True, repr=False
    )
    backbone_link_vetoes: frozenset[AtomRef] = field(
        default_factory=frozenset, kw_only=True, repr=False
    )
    deferred_sidechains: tuple[DeferredSidechainOmission, ...] = field(
        default=(), kw_only=True, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.source_format, FileFormat):
            raise TypeError("structure ingress requires a FileFormat source_format")

        if not isinstance(self.backbone_neighbors, frozenset) or not all(
            isinstance(edge, tuple)
            and len(edge) == 2
            and all(isinstance(rid, ResidueId) for rid in edge)
            and edge[0] != edge[1]
            and edge[0].chain_id == edge[1].chain_id
            for edge in self.backbone_neighbors
        ):
            raise ValueError(
                "backbone neighbors must be distinct residue pairs in one chain"
            )
        if not isinstance(self.backbone_link_vetoes, frozenset) or not all(
            isinstance(ref, AtomRef) for ref in self.backbone_link_vetoes
        ):
            raise TypeError(
                "backbone link exclusions require a frozen set of atom references"
            )
        if not isinstance(self.deferred_sidechains, tuple) or not all(
            isinstance(decision, DeferredSidechainOmission)
            for decision in self.deferred_sidechains
        ):
            raise TypeError(
                "deferred side-chain decisions require a tuple of omission assessments"
            )

        if self.observation is not None and not isinstance(
            self.observation, StructureObservation
        ):
            raise TypeError(
                "structure ingress observation must be a StructureObservation or None"
            )

        ports = self.uncertain_peptide_link_ports
        if ports is not None and (
            not isinstance(ports, frozenset)
            or any(
                not isinstance(port, AtomRef) or port.atom_name not in ("N", "C")
                for port in ports
            )
        ):
            raise TypeError(
                "uncertain peptide ports require a frozen set of backbone atom references"
            )
        paths = self.deposited_peptide_sequences
        if not isinstance(paths, tuple) or any(
            not isinstance(path, DepositedPeptideSequence) for path in paths
        ):
            raise TypeError(
                "deposited peptide sequences require a tuple of bound paths"
            )
        if len({path.chain_id for path in paths}) != len(paths):
            raise ValueError("multiple deposited peptide paths for one author chain")

        omissions = self.zero_occupancy_omissions
        residue_omissions = self.zero_occupancy_residue_omissions
        if omissions is not None and (
            not isinstance(omissions, tuple)
            or any(
                not isinstance(record, ZeroOccupancyOmission) for record in omissions
            )
        ):
            raise TypeError("occupancy omission history must be a tuple of omissions")
        if residue_omissions is not None and (
            not isinstance(residue_omissions, tuple)
            or any(
                not isinstance(record, ZeroOccupancyResidueOmission)
                for record in residue_omissions
            )
        ):
            raise TypeError("residue omission history must be a tuple of omissions")
        consecutive_omissions = self.consecutive_zero_occupancy_residue_omissions
        if consecutive_omissions is not None and (
            not isinstance(consecutive_omissions, tuple)
            or any(
                not isinstance(record, ConsecutiveZeroOccupancyResidueOmission)
                for record in consecutive_omissions
            )
        ):
            raise TypeError(
                "consecutive residue omissions must be a tuple of ConsecutiveZeroOccupancyResidueOmission records"
            )

        observation = self.observation
        cap_links = self.pdb_adjacent_cap_links
        if not isinstance(cap_links, tuple) or any(
            not isinstance(link, PdbAdjacentCapLink) for link in cap_links
        ):
            raise TypeError(
                "PDB cap links must be a tuple of PdbAdjacentCapLink values"
            )
        if cap_links and (
            self.source_format is not FileFormat.PDB
            or observation is None
            or len({link.cap_residue_id for link in cap_links}) != len(cap_links)
            or any(
                link.bond_in(
                    observation.constitution, observation.geometry, observation.topology
                )
                is None
                for link in cap_links
            )
        ):
            raise ValueError(
                "PDB cap links must identify distinct observed retained attachments"
            )
        if (
            omissions is not None
            or residue_omissions is not None
            or consecutive_omissions is not None
        ) and observation is None:
            raise ValueError("occupancy omission history requires a source observation")
        if omissions is not None:
            assert observation is not None
            refs = {omission.source_identity.atom_ref for omission in omissions}
            if len(refs) != len(omissions) or any(
                observation.constitution.resolve_atom_index(ref) is not None
                for ref in refs
            ):
                raise ValueError(
                    "occupancy omissions must be distinct and absent from the source observation"
                )
        if residue_omissions or consecutive_omissions:
            assert observation is not None
            gaps = self.recorded_residue_gaps()
            slots = tuple(r for gap in gaps for r in gap.residue_ids)
            residue_ids = set(slots)
            if len(residue_ids) != len(slots) or any(
                observation.constitution.residue_or_ligand(residue_id) is not None
                for residue_id in residue_ids
            ):
                raise ValueError(
                    "omitted residues must be distinct and absent from the observation"
                )
            if any(
                omission.source_identity.atom_ref.residue_id in residue_ids
                for omission in omissions or ()
            ):
                raise ValueError(
                    "side-chain and whole-residue omissions must be disjoint"
                )
            for record in gaps:
                for flank in (record.preceding_residue_id, record.following_residue_id):
                    residue = observation.constitution.residue_or_ligand(flank)
                    if residue is None or residue.is_hetero:
                        raise ValueError(
                            "residue omission requires present polymer source flanks"
                        )
                    if isinstance(record, ConsecutiveZeroOccupancyResidueOmission):
                        expected = (
                            record.preceding_component_id
                            if flank == record.preceding_residue_id
                            else record.following_component_id
                        )
                        if (
                            residue.component_id != expected
                            or observation.constitution.role_at(
                                observation.constitution.residue_index(flank)
                            )
                            is not ResidueRole.POLYMER
                        ):
                            raise ValueError(
                                "omitted interval flank components must match the observation"
                            )

        source_name = self.source_name
        if source_name is not None:
            source_name = source_name.strip() or None

        object.__setattr__(self, "source_name", source_name)

    def recorded_residue_gaps(
        self,
    ) -> tuple[
        ZeroOccupancyResidueOmission | ConsecutiveZeroOccupancyResidueOmission, ...
    ]:
        """Return biological intervals withheld by either whole-residue policy."""
        return (
            *(self.zero_occupancy_residue_omissions or ()),
            *(self.consecutive_zero_occupancy_residue_omissions or ()),
        )

    def unrestored_residue_gaps(
        self, constitution: StructureConstitution, chain_id: str
    ) -> tuple[
        ZeroOccupancyResidueOmission | ConsecutiveZeroOccupancyResidueOmission, ...
    ]:
        """Find historical gaps that still prevent publishing a full sequence.

        Parameters
        ----------
        constitution : StructureConstitution
            Current structure, which may contain a subsequently restored span.
        chain_id : str
            Author chain whose sequence is to be projected.

        Returns
        -------
        tuple of ZeroOccupancyResidueOmission or ConsecutiveZeroOccupancyResidueOmission
            Intervals with an absent, non-polymer or changed component.
        """
        return tuple(
            gap
            for gap in self.recorded_residue_gaps()
            if gap.preceding_residue_id.chain_id == chain_id
            and not all(
                (site := constitution.residue_or_ligand(rid)) is not None
                and site.component_id == component
                and constitution.role_at(constitution.residue_index(rid))
                is ResidueRole.POLYMER
                for rid, component in zip(
                    gap.residue_ids, gap.component_ids, strict=True
                )
            )
        )

    def pdb_adjacent_cap_link_for(
        self, residue_id: ResidueId
    ) -> PdbAdjacentCapLink | None:
        """Return the source evidence for a retained cap occurrence.

        Parameters
        ----------
        residue_id : ResidueId
            Retained cap to look up in the original observation.

        Returns
        -------
        PdbAdjacentCapLink or None
            Original attachment evidence, if available. Rebind it before use.
        """
        return next(
            (
                link
                for link in self.pdb_adjacent_cap_links
                if link.cap_residue_id == residue_id
            ),
            None,
        )


@dataclass(frozen=True, slots=True)
class StructureProvenance:
    """Original observations, lineage and applied chemistry choices.

    Parameters
    ----------
    ingress : StructureIngress
        Immutable original input observations.
    lineage : tuple[StructureEndpoint, ...]
        Source or supporting-structure endpoints, deduplicated in input order.
    microstate_choices : tuple[AppliedMicrostateChoice, ...]
        Disjoint applied site choices with their original authority. Defaults
        remain preferences, not observations or explicit overrides.
    omitted_peptide_caps : frozenset[PeptideCapAttachment]
        Qualified occupied ports whose cap was removed by retention. Their
        history survives re-entry even if the source observation predates the cap.
    original_backbone_coordinates : tuple of ZeroOccupancyOmission
        Selected zero coordinates before an accepted backbone repair, including
        dependent hydrogens that moved or were withheld.
    withheld_backbone_hydrogens : tuple of WithheldBackboneHydrogen
        Undetermined dependent hydrogen identities and their original parents.

    Raises
    ------
    TypeError
        An ingress, endpoint or choice has a noncanonical type.
    ValueError
        Applied choices overlap on a residue atom.
    """

    ingress: StructureIngress
    lineage: tuple[StructureEndpoint, ...] = ()
    microstate_choices: tuple[AppliedMicrostateChoice, ...] = ()
    omitted_peptide_caps: frozenset[PeptideCapAttachment] = field(
        default_factory=frozenset, kw_only=True, repr=False
    )
    original_backbone_coordinates: tuple[ZeroOccupancyOmission, ...] = field(
        default=(), kw_only=True, repr=False
    )
    withheld_backbone_hydrogens: tuple[WithheldBackboneHydrogen, ...] = field(
        default=(), kw_only=True, repr=False
    )

    def __post_init__(self) -> None:
        if not isinstance(self.ingress, StructureIngress):
            raise TypeError("structure provenance requires a StructureIngress value")
        if not isinstance(self.original_backbone_coordinates, tuple) or not all(
            isinstance(r, ZeroOccupancyOmission)
            for r in self.original_backbone_coordinates
        ):
            raise TypeError(
                "original backbone coordinates require zero-occupancy atom records"
            )
        if len({r.source_identity for r in self.original_backbone_coordinates}) != len(
            self.original_backbone_coordinates
        ):
            raise ValueError("duplicate original backbone coordinate")
        if not isinstance(self.withheld_backbone_hydrogens, tuple) or not all(
            isinstance(r, WithheldBackboneHydrogen)
            for r in self.withheld_backbone_hydrogens
        ):
            raise TypeError(
                "withheld hydrogens require typed parent and isotope records"
            )
        if self.withheld_backbone_hydrogens and not {
            r.original for r in self.withheld_backbone_hydrogens
        } <= set(self.original_backbone_coordinates):
            raise ValueError("withheld hydrogen is missing its original coordinate")

        if not isinstance(self.omitted_peptide_caps, frozenset) or not all(
            isinstance(link, PeptideCapAttachment) for link in self.omitted_peptide_caps
        ):
            raise TypeError("omitted peptide caps require a frozenset of attachments")

        lineage_list: list[StructureEndpoint] = []
        for lineage_scope in self.lineage:
            if not isinstance(lineage_scope, StructureEndpoint):
                raise TypeError(
                    "structure provenance lineage requires StructureEndpoint values"
                )
            if lineage_scope not in lineage_list:
                lineage_list.append(lineage_scope)

        object.__setattr__(self, "lineage", tuple(lineage_list))
        overrides = tuple(self.microstate_choices)
        occupied: set[AtomRef] = set()
        for override in overrides:
            if not isinstance(override, AppliedMicrostateChoice):
                raise TypeError(
                    "structure provenance requires applied microstate choices"
                )
            # Site identity must not depend on atom indices that H edits can remap.
            keys = {
                AtomRef(override.residue_id, atom.name) for atom in override.graph.atoms
            }
            if occupied & keys:
                raise ValueError("applied microstate choices must not overlap")
            occupied.update(keys)
        object.__setattr__(
            self,
            "microstate_choices",
            tuple(
                sorted(
                    overrides,
                    key=lambda entry: (
                        entry.residue_id,
                        tuple(atom.name for atom in entry.graph.atoms),
                    ),
                )
            ),
        )

    def withheld_hydrogen_residues(
        self, constitution: StructureConstitution
    ) -> frozenset[ResidueId]:
        """Return residues where an absent original isotope still has its original parent."""
        return frozenset(
            r.parent.residue_id
            for r in self.withheld_backbone_hydrogens
            if (site := constitution.residue_or_ligand(r.parent.residue_id)) is not None
            and site.component_id == r.original.source_identity.component_id
            and site.has_atom_site(r.parent.atom_name)
            and not site.has_atom_site(r.original.source_identity.atom_ref.atom_name)
        )


@dataclass(frozen=True, slots=True, init=False)
class StructureProvenanceOrigin:
    """One canonical origin pointer into a source or supporting structure."""

    structure_endpoint: StructureEndpoint

    def __init__(
        self,
        *,
        structure_endpoint: StructureEndpoint,
    ) -> None:
        """Build one provenance origin from one canonical carrier scope."""

        if not isinstance(structure_endpoint, StructureEndpoint):
            raise TypeError(
                "structure provenance origins require a StructureEndpoint value"
            )

        object.__setattr__(self, "structure_endpoint", structure_endpoint)

    def is_source(self) -> bool:
        """Return whether this origin points into the source structure."""

        return self.structure_endpoint.is_source()

    def is_supporting(self) -> bool:
        """Return whether this origin points into a supporting structure."""

        return self.structure_endpoint.is_supporting()

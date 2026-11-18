"""Canonical structure provenance facets."""

from dataclasses import dataclass, field
from enum import Enum

from protrepair.chemistry.microstate.resolution import AppliedMicrostateChoice
from protrepair.structure.element import ElementIdentity
from protrepair.structure.endpoint import StructureEndpoint
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.source_connection import SourceAtomIdentity


class FileFormat(str, Enum):
    """Canonical coordinate file formats recorded in structure provenance."""

    PDB = "pdb"
    MMCIF = "mmcif"


@dataclass(frozen=True, slots=True)
class ZeroOccupancyOmission:
    """Historical position withheld during source normalization.

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

    Raises
    ------
    TypeError
        The source format, observation or omissions have noncanonical types.
    ValueError
        Omission history has no source observation, or an omitted atom appears
        in the observation or more than once in the history.
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

    def __post_init__(self) -> None:
        if not isinstance(self.source_format, FileFormat):
            raise TypeError("structure ingress requires a FileFormat source_format")

        if self.observation is not None and not isinstance(
            self.observation, StructureObservation
        ):
            raise TypeError(
                "structure ingress observation must be a StructureObservation or None"
            )

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

        observation = self.observation
        if (
            omissions is not None or residue_omissions is not None
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
        if residue_omissions is not None:
            assert observation is not None
            residue_ids = {record.residue_id for record in residue_omissions}
            if len(residue_ids) != len(residue_omissions) or any(
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
            for record in residue_omissions:
                for flank in (record.preceding_residue_id, record.following_residue_id):
                    residue = observation.constitution.residue_or_ligand(flank)
                    if residue is None or residue.is_hetero:
                        raise ValueError(
                            "residue omission requires present polymer source flanks"
                        )

        source_name = self.source_name
        if source_name is not None:
            source_name = source_name.strip() or None

        object.__setattr__(self, "source_name", source_name)


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

    def __post_init__(self) -> None:
        if not isinstance(self.ingress, StructureIngress):
            raise TypeError("structure provenance requires a StructureIngress value")

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

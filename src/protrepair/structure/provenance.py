"""Canonical structure provenance facets."""

from dataclasses import dataclass, field
from enum import Enum

from protrepair.chemistry.microstate.resolution import AppliedMicrostateOverride
from protrepair.structure.endpoint import StructureEndpoint
from protrepair.structure.labels import AtomRef
from protrepair.structure.observation import StructureObservation


class FileFormat(str, Enum):
    """Canonical coordinate file formats recorded in structure provenance."""

    PDB = "pdb"
    MMCIF = "mmcif"


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

    Raises
    ------
    TypeError
        The source format or observation has a noncanonical type.
    """

    source_format: FileFormat
    source_name: str | None = None
    observation: StructureObservation | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if not isinstance(self.source_format, FileFormat):
            raise TypeError("structure ingress requires a FileFormat source_format")

        if self.observation is not None and not isinstance(
            self.observation, StructureObservation
        ):
            raise TypeError(
                "structure ingress observation must be a StructureObservation or None"
            )

        source_name = self.source_name
        if source_name is not None:
            source_name = source_name.strip() or None

        object.__setattr__(self, "source_name", source_name)


@dataclass(frozen=True, slots=True)
class StructureProvenance:
    """Original observations, lineage and explicitly applied chemistry choices.

    Parameters
    ----------
    ingress : StructureIngress
        Immutable original input observations.
    lineage : tuple[StructureEndpoint, ...]
        Source or supporting-structure endpoints, deduplicated in input order.
    microstate_overrides : tuple[AppliedMicrostateOverride, ...]
        Disjoint applied site choices. These are not observations and must be
        rebound to current chemistry before reuse.

    Raises
    ------
    TypeError
        An ingress, endpoint or override has a noncanonical type.
    ValueError
        Applied overrides overlap on a residue atom.
    """

    ingress: StructureIngress
    lineage: tuple[StructureEndpoint, ...] = ()
    microstate_overrides: tuple[AppliedMicrostateOverride, ...] = ()

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
        overrides = tuple(self.microstate_overrides)
        occupied: set[AtomRef] = set()
        for override in overrides:
            if not isinstance(override, AppliedMicrostateOverride):
                raise TypeError(
                    "structure provenance requires applied microstate overrides"
                )
            # Site identity must not depend on atom indices that H edits can remap.
            keys = {
                AtomRef(override.residue_id, atom.name) for atom in override.graph.atoms
            }
            if occupied & keys:
                raise ValueError("applied microstate overrides must not overlap")
            occupied.update(keys)
        object.__setattr__(
            self,
            "microstate_overrides",
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

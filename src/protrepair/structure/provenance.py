"""Canonical structure provenance facets."""

from dataclasses import dataclass
from enum import Enum

from protrepair.relation.endpoint import StructureEndpoint


class FileFormat(str, Enum):
    """Canonical coordinate file formats recorded in structure provenance."""

    PDB = "pdb"
    MMCIF = "mmcif"


@dataclass(frozen=True, slots=True)
class StructureIngress:
    """Whole-structure ingress metadata detached from the aggregate root."""

    source_format: FileFormat
    source_name: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_format, FileFormat):
            raise TypeError(
                "structure ingress requires a FileFormat source_format"
            )

        source_name = self.source_name
        if source_name is not None:
            source_name = source_name.strip() or None

        object.__setattr__(self, "source_name", source_name)


@dataclass(frozen=True, slots=True)
class StructureProvenance:
    """Whole-structure provenance as ingress metadata plus lineage endpoints."""

    ingress: StructureIngress
    lineage: tuple[StructureEndpoint, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ingress, StructureIngress):
            raise TypeError(
                "structure provenance requires a StructureIngress value"
            )

        lineage_list: list[StructureEndpoint] = []
        for lineage_scope in self.lineage:
            if not isinstance(lineage_scope, StructureEndpoint):
                raise TypeError(
                    "structure provenance lineage requires StructureEndpoint values"
                )
            if lineage_scope not in lineage_list:
                lineage_list.append(lineage_scope)

        object.__setattr__(self, "lineage", tuple(lineage_list))


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

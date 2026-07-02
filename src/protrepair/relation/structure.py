"""Canonical scoped structure-relation value objects."""

from dataclasses import dataclass
from enum import Enum

from protrepair.scope import Scope
from protrepair.structure.endpoint import (
    StructureCarrierKind,
    StructureEndpoint,
)


def _require_structure_endpoint(
    endpoint: StructureEndpoint,
    *,
    field_name: str,
    expected_kind: StructureCarrierKind,
) -> StructureEndpoint:
    """Return one canonical structure endpoint with the required carrier kind."""

    if isinstance(endpoint, StructureEndpoint):
        structure_endpoint = endpoint
    else:
        raise TypeError(f"{field_name} requires a StructureEndpoint value")

    if structure_endpoint.carrier_handle.kind is not expected_kind:
        raise ValueError(
            f"{field_name} must point into a {expected_kind.value} carrier"
        )

    return structure_endpoint


@dataclass(frozen=True, slots=True)
class StructureEndpointMapping:
    """Canonical one-to-one mapping between two scoped carrier endpoints."""

    source_structure_endpoint: StructureEndpoint
    supporting_structure_endpoint: StructureEndpoint

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "source_structure_endpoint",
            _require_structure_endpoint(
                self.source_structure_endpoint,
                field_name="source_structure_endpoint",
                expected_kind=StructureCarrierKind.SOURCE,
            ),
        )
        object.__setattr__(
            self,
            "supporting_structure_endpoint",
            _require_structure_endpoint(
                self.supporting_structure_endpoint,
                field_name="supporting_structure_endpoint",
                expected_kind=StructureCarrierKind.SUPPORTING,
            ),
        )

    def cardinality_signature(self) -> tuple[int, int]:
        """Return source/supporting member counts for this mapping."""

        return (
            self.source_structure_endpoint.cardinality(),
            self.supporting_structure_endpoint.cardinality(),
        )


@dataclass(frozen=True, slots=True)
class SupportingStructureCorrespondence:
    """Canonical correspondence between one source region and one support region."""

    source_structure_endpoint: StructureEndpoint
    supporting_structure_endpoint: StructureEndpoint
    mappings: tuple[StructureEndpointMapping, ...] = ()

    def __post_init__(self) -> None:
        source_structure_endpoint = _require_structure_endpoint(
            self.source_structure_endpoint,
            field_name="source_structure_endpoint",
            expected_kind=StructureCarrierKind.SOURCE,
        )
        supporting_structure_endpoint = _require_structure_endpoint(
            self.supporting_structure_endpoint,
            field_name="supporting_structure_endpoint",
            expected_kind=StructureCarrierKind.SUPPORTING,
        )
        mapping_values = tuple(dict.fromkeys(self.mappings))
        for mapping in mapping_values:
            if not isinstance(mapping, StructureEndpointMapping):
                raise TypeError(
                    "supporting structure correspondences require "
                    "StructureEndpointMapping values"
                )
            if not source_structure_endpoint.covers(mapping.source_structure_endpoint):
                raise ValueError(
                    "supporting structure correspondence mapping source endpoint "
                    "must fall inside the correspondence source endpoint"
                )
            if not supporting_structure_endpoint.covers(
                mapping.supporting_structure_endpoint
            ):
                raise ValueError(
                    "supporting structure correspondence mapping supporting "
                    "endpoint must fall inside the correspondence supporting "
                    "endpoint"
                )

        object.__setattr__(self, "source_structure_endpoint", source_structure_endpoint)
        object.__setattr__(
            self,
            "supporting_structure_endpoint",
            supporting_structure_endpoint,
        )
        object.__setattr__(self, "mappings", mapping_values)

    @property
    def source_scope(self) -> Scope:
        """Return the source semantic scope carried by this correspondence."""

        return self.source_structure_endpoint.scope

    @property
    def supporting_scope(self) -> Scope:
        """Return the support semantic scope carried by this correspondence."""

        return self.supporting_structure_endpoint.scope


class SupportingStructureAuthorityAspect(str, Enum):
    """Closed semantic aspects that a support structure may authoritatively supply."""

    COORDINATES = "coordinates"
    COMPONENT_IDENTITY = "component_identity"
    HEAVY_ATOM_TOPOLOGY = "heavy_atom_topology"
    HYDROGEN_TOPOLOGY = "hydrogen_topology"


@dataclass(frozen=True, slots=True)
class SupportingStructureAuthorityGrant:
    """Authority grant over one correspondence and one or more semantic aspects."""

    correspondence: SupportingStructureCorrespondence
    authoritative_aspects: tuple[SupportingStructureAuthorityAspect, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.correspondence, SupportingStructureCorrespondence):
            raise TypeError(
                "supporting structure authority grants require one "
                "SupportingStructureCorrespondence"
            )

        authoritative_aspects = tuple(dict.fromkeys(self.authoritative_aspects))
        if not authoritative_aspects:
            raise ValueError(
                "supporting structure authority grants require at least one "
                "authoritative aspect"
            )
        for aspect in authoritative_aspects:
            if not isinstance(aspect, SupportingStructureAuthorityAspect):
                raise TypeError(
                    "supporting structure authority grants require "
                    "SupportingStructureAuthorityAspect values"
                )

        object.__setattr__(self, "authoritative_aspects", authoritative_aspects)

    def grants(self, aspect: SupportingStructureAuthorityAspect) -> bool:
        """Return whether this grant authorizes one semantic aspect."""

        return aspect in self.authoritative_aspects

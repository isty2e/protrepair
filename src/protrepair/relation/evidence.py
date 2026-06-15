"""Relation-owned evidence and constraint value objects."""

from dataclasses import dataclass
from math import isfinite
from typing import TypeAlias

from protrepair.relation.endpoint import StructureEndpoint
from protrepair.scope import AtomSetScope
from protrepair.structure.provenance import StructureProvenanceOrigin


def _normalized_optional_label(value: str | None) -> str | None:
    """Normalize one optional user-facing label or token."""

    if value is None:
        return None

    return value.strip() or None


@dataclass(frozen=True, slots=True)
class ExternalCorrespondenceEvidence:
    """Externally supplied correspondence between two provenance origins."""

    source_origin: StructureProvenanceOrigin
    counterpart_origin: StructureProvenanceOrigin
    evidence_label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.source_origin, StructureProvenanceOrigin):
            raise TypeError(
                "external correspondence evidence requires a "
                "StructureProvenanceOrigin source_origin"
            )
        if not isinstance(self.counterpart_origin, StructureProvenanceOrigin):
            raise TypeError(
                "external correspondence evidence requires a "
                "StructureProvenanceOrigin counterpart_origin"
            )
        if self.source_origin == self.counterpart_origin:
            raise ValueError(
                "external correspondence evidence requires two distinct "
                "provenance origins"
            )

        object.__setattr__(
            self,
            "evidence_label",
            _normalized_optional_label(self.evidence_label),
        )


@dataclass(frozen=True, slots=True)
class DensityEvidence:
    """External density evidence attached to one target structure subregion."""

    target_structure_endpoint: StructureEndpoint
    map_label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target_structure_endpoint, StructureEndpoint):
            raise TypeError(
                "density evidence requires a StructureEndpoint "
                "target_structure_endpoint"
            )
        if not self.target_structure_endpoint.is_source():
            raise ValueError(
                "density evidence requires target_structure_endpoint to point "
                "into the active source structure"
            )

        object.__setattr__(
            self,
            "map_label",
            _normalized_optional_label(self.map_label),
        )


@dataclass(frozen=True, slots=True)
class SymmetryContactEvidence:
    """External symmetry-contact evidence over one target structure subregion."""

    target_structure_endpoint: StructureEndpoint
    symmetry_operator_label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.target_structure_endpoint, StructureEndpoint):
            raise TypeError(
                "symmetry-contact evidence requires a StructureEndpoint "
                "target_structure_endpoint"
            )
        if not self.target_structure_endpoint.is_source():
            raise ValueError(
                "symmetry-contact evidence requires target_structure_endpoint "
                "to point into the active source structure"
            )

        object.__setattr__(
            self,
            "symmetry_operator_label",
            _normalized_optional_label(self.symmetry_operator_label),
        )


@dataclass(frozen=True, slots=True)
class AnchorDistanceConstraint:
    """External anchor-distance constraint over two atom-local origins."""

    left_anchor_origin: StructureProvenanceOrigin
    right_anchor_origin: StructureProvenanceOrigin
    target_distance_angstrom: float
    tolerance_angstrom: float

    def __post_init__(self) -> None:
        if not isinstance(self.left_anchor_origin, StructureProvenanceOrigin):
            raise TypeError(
                "anchor-distance constraints require a StructureProvenanceOrigin "
                "left_anchor_origin"
            )
        if not isinstance(self.right_anchor_origin, StructureProvenanceOrigin):
            raise TypeError(
                "anchor-distance constraints require a StructureProvenanceOrigin "
                "right_anchor_origin"
            )
        if self.left_anchor_origin == self.right_anchor_origin:
            raise ValueError(
                "anchor-distance constraints require two distinct anchor origins"
            )
        left_scope = self.left_anchor_origin.structure_endpoint.scope
        right_scope = self.right_anchor_origin.structure_endpoint.scope
        if (
            not isinstance(left_scope, AtomSetScope)
            or len(left_scope.atom_refs) != 1
            or not isinstance(right_scope, AtomSetScope)
            or len(right_scope.atom_refs) != 1
        ):
            raise ValueError(
                "anchor-distance constraints require atom-local provenance origins"
            )

        for field_name, value in (
            ("target_distance_angstrom", self.target_distance_angstrom),
            ("tolerance_angstrom", self.tolerance_angstrom),
        ):
            if isinstance(value, bool) or not isinstance(value, int | float):
                raise TypeError(
                    "anchor-distance constraints require finite numeric "
                    f"{field_name}"
                )
            if not isfinite(float(value)):
                raise ValueError(
                    "anchor-distance constraints require finite "
                    f"{field_name}"
                )

        if self.target_distance_angstrom <= 0.0:
            raise ValueError(
                "anchor-distance constraints require positive "
                "target_distance_angstrom"
            )
        if self.tolerance_angstrom <= 0.0:
            raise ValueError(
                "anchor-distance constraints require positive tolerance_angstrom"
            )


TransformationExternalEvidence: TypeAlias = (
    DensityEvidence
    | SymmetryContactEvidence
    | ExternalCorrespondenceEvidence
)

TransformationExternalConstraint: TypeAlias = AnchorDistanceConstraint

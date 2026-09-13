"""Canonical primitive directives for hydrogen-completion workflows."""

from dataclasses import dataclass

from protrepair.chemistry import (
    HydrogenSemantics,
    IdealGeometryHydrogenSemantics,
    ResidueTemplate,
)
from protrepair.structure.slots import ResidueIndex


class HydrogenCompletionDirective:
    """One ordered primitive hydrogen-completion operation."""


@dataclass(frozen=True, slots=True)
class StaticHydrogenPlacementDirective(HydrogenCompletionDirective):
    """Place one residue's hydrogens from template-static hydrogen semantics."""

    residue_index: ResidueIndex
    template: ResidueTemplate
    semantics: HydrogenSemantics
    next_residue_index: ResidueIndex | None = None


@dataclass(frozen=True, slots=True)
class RigidHydrogenPlacementDirective(HydrogenCompletionDirective):
    """Place one residue's hydrogens from idealized rigid-frame semantics."""

    residue_index: ResidueIndex
    template: ResidueTemplate
    semantics: IdealGeometryHydrogenSemantics
    next_residue_index: ResidueIndex | None = None


def hydrogen_placement_directive(
    *,
    residue_index: ResidueIndex,
    template: ResidueTemplate | None,
    next_residue_index: ResidueIndex | None = None,
) -> StaticHydrogenPlacementDirective | RigidHydrogenPlacementDirective | None:
    """Return the primitive placement directive for one supported residue."""

    if template is None:
        return None

    hydrogen_semantics = template.hydrogen_semantics
    if hydrogen_semantics is None:
        return None

    if isinstance(hydrogen_semantics, IdealGeometryHydrogenSemantics):
        return RigidHydrogenPlacementDirective(
            residue_index=residue_index,
            template=template,
            semantics=hydrogen_semantics,
            next_residue_index=next_residue_index,
        )

    return StaticHydrogenPlacementDirective(
        residue_index=residue_index,
        template=template,
        semantics=hydrogen_semantics,
        next_residue_index=next_residue_index,
    )

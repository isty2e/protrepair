"""Canonical primitive directives for hydrogen-completion workflows."""

from collections.abc import Sequence
from dataclasses import dataclass

from protrepair.chemistry import (
    HydrogenSemantics,
    IdealGeometryHydrogenSemantics,
    ResidueTemplate,
)
from protrepair.structure.constitution import ChainSite
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


@dataclass(frozen=True, slots=True)
class HistidineDeltaProtonationDirective(HydrogenCompletionDirective):
    """Append the delta hydrogen for one selected histidine residue."""

    residue_index: ResidueIndex
    template: ResidueTemplate


@dataclass(frozen=True, slots=True)
class BackboneHydrogenPropagationDirective(HydrogenCompletionDirective):
    """Propagate one residue's backbone hydrogen onto the next residue."""

    residue_index: ResidueIndex
    template: ResidueTemplate
    next_residue_index: ResidueIndex


@dataclass(frozen=True, slots=True)
class NTerminalHydrogenPlacementDirective(HydrogenCompletionDirective):
    """Place N-terminal hydrogens on one supported first residue."""

    residue_index: ResidueIndex
    template: ResidueTemplate
    backbone_family_component_id: str


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


def derive_hydrogen_directives(
    chain: ChainSite,
    *,
    templates: Sequence[ResidueTemplate | None],
    protonate_histidines: bool,
) -> tuple[HydrogenCompletionDirective, ...]:
    """Derive the ordered primitive directives for one chain hydrogenation run."""

    directives: list[HydrogenCompletionDirective] = []
    supported_templates_by_index: dict[ResidueIndex, ResidueTemplate] = {}

    for residue_offset, _residue in enumerate(chain.residues):
        residue_index = ResidueIndex(residue_offset)
        if (
            hydrogen_placement_directive(
                residue_index=residue_index,
                template=templates[residue_index.value],
            )
            is None
        ):
            continue

        template = templates[residue_index.value]
        assert template is not None
        supported_templates_by_index[residue_index] = template

    placement_directives_by_index: dict[
        ResidueIndex,
        StaticHydrogenPlacementDirective | RigidHydrogenPlacementDirective,
    ] = {}
    for residue_index, template in supported_templates_by_index.items():
        next_residue_index: ResidueIndex | None = None
        next_index = ResidueIndex(residue_index.value + 1)
        next_template = supported_templates_by_index.get(next_index)
        if next_template is not None:
            next_backbone_family_component_id = (
                next_template.backbone_family_component_id
            )
            if next_backbone_family_component_id != "PRO":
                next_residue_index = next_index

        placement_directive = hydrogen_placement_directive(
            residue_index=residue_index,
            template=template,
            next_residue_index=next_residue_index,
        )
        assert placement_directive is not None
        placement_directives_by_index[residue_index] = placement_directive
        directives.append(placement_directive)

    if protonate_histidines:
        histidine_indices = tuple(
            ResidueIndex(residue_offset)
            for residue_offset, residue in enumerate(chain.residues)
            if residue.component_id == "HIS"
        )
        if len(histidine_indices) > 4:
            for residue_index in histidine_indices[: len(histidine_indices) // 5]:
                template = templates[residue_index.value]
                if template is None:
                    continue
                directives.append(
                    HistidineDeltaProtonationDirective(
                        residue_index=residue_index,
                        template=template,
                    )
                )

    for residue_index, placement_directive in placement_directives_by_index.items():
        if residue_index.value >= len(chain.residues) - 1:
            continue

        next_placement_directive = placement_directives_by_index.get(
            ResidueIndex(residue_index.value + 1)
        )
        if next_placement_directive is None:
            continue

        next_backbone_family_component_id = (
            next_placement_directive.template.backbone_family_component_id
        )
        if next_backbone_family_component_id == "PRO":
            continue

        directives.append(
            BackboneHydrogenPropagationDirective(
                residue_index=placement_directive.residue_index,
                template=placement_directive.template,
                next_residue_index=next_placement_directive.residue_index,
            )
        )

    first_placement_directive = placement_directives_by_index.get(ResidueIndex(0))
    if first_placement_directive is not None:
        directives.append(
            NTerminalHydrogenPlacementDirective(
                residue_index=first_placement_directive.residue_index,
                template=first_placement_directive.template,
                backbone_family_component_id=(
                    first_placement_directive.template.backbone_family_component_id
                ),
            )
        )

    return tuple(directives)

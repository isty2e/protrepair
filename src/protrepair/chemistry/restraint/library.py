"""Canonical lookup boundary for residue restraint templates."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from typing_extensions import Self

from protrepair.chemistry.restraint.template import ResidueRestraintTemplate
from protrepair.errors import UnknownComponentError


def _register_restraint_alias(
    alias_to_component_id: dict[str, str],
    alias: str,
    component_id: str,
) -> None:
    """Register one restraint alias mapping and reject ambiguous reuse."""

    existing_component_id = alias_to_component_id.get(alias)
    if existing_component_id is not None and existing_component_id != component_id:
        raise ValueError(
            f"ambiguous restraint alias {alias!r}: "
            f"{existing_component_id} vs {component_id}"
        )

    alias_to_component_id[alias] = component_id


@dataclass(frozen=True, slots=True)
class RestraintLibrary:
    """Canonical lookup boundary for residue restraint templates."""

    templates: Mapping[str, ResidueRestraintTemplate] = field(default_factory=dict)
    alias_to_component_id: Mapping[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        templates = {
            component_id.strip().upper(): template
            for component_id, template in self.templates.items()
        }
        alias_to_component_id = {
            alias.strip().upper(): component_id.strip().upper()
            for alias, component_id in self.alias_to_component_id.items()
        }

        for template in templates.values():
            _register_restraint_alias(
                alias_to_component_id,
                template.component_id,
                template.component_id,
            )

        object.__setattr__(self, "templates", MappingProxyType(templates))
        object.__setattr__(
            self,
            "alias_to_component_id",
            MappingProxyType(alias_to_component_id),
        )

    def normalize_component_id(self, component_id: str) -> str:
        """Normalize one component identifier to the canonical restraint key."""

        normalized_component_id = component_id.strip().upper()
        return self.alias_to_component_id.get(
            normalized_component_id,
            normalized_component_id,
        )

    def get(self, component_id: str) -> ResidueRestraintTemplate | None:
        """Return one restraint template if the library can resolve it."""

        return self.templates.get(self.normalize_component_id(component_id))

    def require(self, component_id: str) -> ResidueRestraintTemplate:
        """Return one restraint template or raise when unavailable."""

        template = self.get(component_id)
        if template is None:
            raise UnknownComponentError(f"Unknown restraint template: {component_id}")

        return template

    def with_template(self, template: ResidueRestraintTemplate) -> Self:
        """Return a copy with one added or replaced restraint template."""

        templates = dict(self.templates)
        templates[template.component_id] = template
        return type(self)(
            templates=templates,
            alias_to_component_id=self.alias_to_component_id,
        )

    def merged_with(self, override: "RestraintLibrary") -> Self:
        """Return a copy with templates and aliases overridden by another library."""

        return type(self)(
            templates={
                **self.templates,
                **override.templates,
            },
            alias_to_component_id={
                **self.alias_to_component_id,
                **override.alias_to_component_id,
            },
        )

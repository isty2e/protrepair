"""Component library abstractions for the redesigned ProtRepair package."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from typing_extensions import Self

from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.errors import UnknownComponentError


def register_component_alias(
    alias_to_component_id: dict[str, str],
    alias: str,
    component_id: str,
) -> None:
    """Register one alias mapping and reject ambiguous reuse."""

    existing_component_id = alias_to_component_id.get(alias)
    if existing_component_id is not None and existing_component_id != component_id:
        raise ValueError(
            f"ambiguous alias {alias!r}: {existing_component_id} vs {component_id}"
        )

    alias_to_component_id[alias] = component_id


@dataclass(frozen=True, slots=True)
class ComponentLibrary:
    """Canonical lookup boundary for residue templates."""

    templates: Mapping[str, ResidueTemplate] = field(default_factory=dict)
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
            register_component_alias(
                alias_to_component_id,
                template.component_id,
                template.component_id,
            )
            for alias in template.aliases:
                register_component_alias(
                    alias_to_component_id,
                    alias,
                    template.component_id,
                )

        object.__setattr__(self, "templates", MappingProxyType(templates))
        object.__setattr__(
            self, "alias_to_component_id", MappingProxyType(alias_to_component_id)
        )

    def normalize_component_id(self, component_id: str) -> str:
        """Normalize a component identifier to the canonical definition key."""

        normalized = component_id.strip().upper()
        return self.alias_to_component_id.get(normalized, normalized)

    def has(self, component_id: str) -> bool:
        """Return whether the library can resolve a component identifier."""

        normalized = self.normalize_component_id(component_id)
        return normalized in self.templates

    def get(self, component_id: str) -> ResidueTemplate | None:
        """Return a residue template if available."""

        normalized = self.normalize_component_id(component_id)
        return self.templates.get(normalized)

    def require(self, component_id: str) -> ResidueTemplate:
        """Return a residue template or raise if it is unavailable."""

        template = self.get(component_id)
        if template is None:
            raise UnknownComponentError(f"Unknown component: {component_id}")

        return template

    def with_template(self, template: ResidueTemplate) -> Self:
        """Return a copy with an added or replaced residue template."""

        templates = dict(self.templates)
        templates[template.component_id] = template

        alias_to_component_id = dict(self.alias_to_component_id)
        alias_to_component_id[template.component_id] = template.component_id
        for alias in template.aliases:
            alias_to_component_id[alias] = template.component_id

        return type(self)(
            templates=templates,
            alias_to_component_id=alias_to_component_id,
        )

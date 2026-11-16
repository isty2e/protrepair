"""Canonical lookup boundary for residue restraint templates."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from typing_extensions import Self

from protrepair.chemistry.restraint.reference import ComponentRestraintReference
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
    """Own static dictionaries and their optional state-matched references.

    Parameters
    ----------
    templates : Mapping[str, ResidueRestraintTemplate]
        Static fallbacks keyed by canonical component identity.
    alias_to_component_id : Mapping[str, str]
        Alternate names normalized to the template keys.
    component_references : Mapping[str, tuple[ComponentRestraintReference, ...]]
        Noncompeting state alternatives sharing one scope per component.

    Raises
    ------
    ValueError
        Aliases conflict or references lack compatible fallbacks or unique states.
    """

    templates: Mapping[str, ResidueRestraintTemplate] = field(default_factory=dict)
    alias_to_component_id: Mapping[str, str] = field(default_factory=dict)
    component_references: Mapping[str, tuple[ComponentRestraintReference, ...]] = field(
        default_factory=dict
    )

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

        references = {
            component_id.strip().upper(): tuple(entries)
            for component_id, entries in self.component_references.items()
        }
        for component_id, entries in references.items():
            if component_id not in templates:
                raise ValueError(
                    "component references require a static fallback template"
                )
            for index, reference in enumerate(entries):
                if not reference.same_scope_as(entries[0]):
                    raise ValueError(
                        "a component's reference alternatives must share one scope"
                    )
                reference.validate_targets(templates[component_id])
                if any(
                    reference.same_state_as(previous) for previous in entries[:index]
                ):
                    raise ValueError("component reference states must not repeat")

        object.__setattr__(self, "templates", MappingProxyType(templates))
        object.__setattr__(self, "component_references", MappingProxyType(references))
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
        """Replace a template and remove its automatic component references."""

        templates = dict(self.templates)
        templates[template.component_id] = template
        return type(self)(
            templates=templates,
            alias_to_component_id=self.alias_to_component_id,
            component_references={
                key: entries
                for key, entries in self.component_references.items()
                if key != template.component_id
            },
        )

    def with_component_reference(self, reference: ComponentRestraintReference) -> Self:
        """Attach a state-specific reference without changing the fallback.

        Parameters
        ----------
        reference : ComponentRestraintReference
            Additional state alternative for an existing static dictionary.

        Returns
        -------
        Self
            New library; the original library and fallback remain unchanged.

        Raises
        ------
        ValueError
            The reference conflicts with the fallback or existing alternatives.
        """

        component_id = reference.component.component_id
        references = dict(self.component_references)
        references[component_id] = (*references.get(component_id, ()), reference)
        return type(self)(
            templates=self.templates,
            alias_to_component_id=self.alias_to_component_id,
            component_references=references,
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
            component_references={
                **{
                    key: entries
                    for key, entries in self.component_references.items()
                    if key not in override.templates
                },
                **override.component_references,
            },
        )

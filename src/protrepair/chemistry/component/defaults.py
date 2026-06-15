"""Internal default component-library construction."""

from functools import lru_cache

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.nonstandard.registry import (
    build_bundled_nonstandard_component_library,
)
from protrepair.chemistry.retained_non_polymer.registry import (
    build_bundled_retained_non_polymer_component_library,
)
from protrepair.chemistry.standard.components import build_standard_component_library


@lru_cache(maxsize=1)
def build_default_component_library() -> ComponentLibrary:
    """Return the internal default library with bundled nonstandard components."""

    standard_library = build_standard_component_library()
    bundled_nonstandard_library = build_bundled_nonstandard_component_library()
    bundled_retained_non_polymer_library = (
        build_bundled_retained_non_polymer_component_library()
    )
    nonstandard_templates = {
        component_id: template
        for component_id, template in bundled_nonstandard_library.templates.items()
        if standard_library.alias_to_component_id.get(component_id)
        in {None, component_id}
    }
    retained_non_polymer_templates = {
        component_id: template
        for (
            component_id,
            template,
        ) in bundled_retained_non_polymer_library.templates.items()
        if standard_library.alias_to_component_id.get(component_id)
        in {None, component_id}
    }
    return ComponentLibrary(
        templates={
            **standard_library.templates,
            **nonstandard_templates,
            **retained_non_polymer_templates,
        },
        alias_to_component_id={
            **standard_library.alias_to_component_id,
        },
    )

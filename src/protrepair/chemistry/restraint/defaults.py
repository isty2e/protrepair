"""Internal default restraint-library construction."""

from functools import lru_cache

from protrepair.chemistry.nonstandard.registry import (
    build_bundled_nonstandard_restraint_library,
)
from protrepair.chemistry.restraint.library import RestraintLibrary
from protrepair.chemistry.retained_non_polymer.registry import (
    build_bundled_retained_non_polymer_restraint_library,
)
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.chemistry.standard.restraints import build_standard_restraint_library


@lru_cache(maxsize=1)
def build_default_restraint_library() -> RestraintLibrary:
    """Return the internal default library with bundled nonstandard restraints."""

    standard_components = build_standard_component_library()
    standard_restraints = build_standard_restraint_library()
    bundled_nonstandard_restraints = build_bundled_nonstandard_restraint_library()
    bundled_retained_non_polymer_restraints = (
        build_bundled_retained_non_polymer_restraint_library()
    )
    nonstandard_templates = {
        component_id: template
        for component_id, template in bundled_nonstandard_restraints.templates.items()
        if standard_components.normalize_component_id(component_id) == component_id
    }
    retained_non_polymer_templates = {
        component_id: template
        for (
            component_id,
            template,
        ) in bundled_retained_non_polymer_restraints.templates.items()
        if standard_components.normalize_component_id(component_id) == component_id
    }
    nonstandard_restraints = RestraintLibrary(
        templates=nonstandard_templates,
        alias_to_component_id={
            alias: component_id
            for alias, component_id in (
                bundled_nonstandard_restraints.alias_to_component_id.items()
            )
            if component_id in nonstandard_templates
        },
    )
    retained_non_polymer_restraints = RestraintLibrary(
        templates=retained_non_polymer_templates,
        alias_to_component_id={
            alias: component_id
            for alias, component_id in (
                bundled_retained_non_polymer_restraints.alias_to_component_id.items()
            )
            if component_id in retained_non_polymer_templates
        },
    )
    return standard_restraints.merged_with(
        nonstandard_restraints
    ).merged_with(retained_non_polymer_restraints)

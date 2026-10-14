"""Assembly of standard residue geometry-restraint libraries."""

from functools import lru_cache

from protrepair.chemistry.restraint.library import RestraintLibrary
from protrepair.chemistry.standard.components import (
    STANDARD_COMPONENT_ATOMS,
    build_standard_component_library,
)
from protrepair.chemistry.standard.histidine_restraints import (
    histidine_restraint_template,
)
from protrepair.chemistry.standard.restraint_targets import (
    STANDARD_RESTRAINT_TEMPLATES as _STANDARD_RESTRAINT_TEMPLATES,
)

STANDARD_RESTRAINT_TEMPLATES = {
    component_id: (
        histidine_restraint_template()
        if component_id == "HIS"
        else _STANDARD_RESTRAINT_TEMPLATES[component_id]
    )
    for component_id in STANDARD_COMPONENT_ATOMS
}


@lru_cache(maxsize=1)
def build_standard_restraint_library() -> RestraintLibrary:
    """Return the built-in standard residue restraint templates."""

    return RestraintLibrary(
        templates=STANDARD_RESTRAINT_TEMPLATES,
        alias_to_component_id=build_standard_component_library().alias_to_component_id,
    )

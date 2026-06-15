"""Shared retained non-polymer component fixtures for workflow tests."""

from protrepair.chemistry import (
    ChemicalComponentDefinition,
    ComponentLibrary,
    HeavyAtomSemantics,
    HydrogenSemantics,
    ResidueTemplate,
)
from protrepair.chemistry.internal_coordinates import InternalCoordinateProgram


def build_retained_non_polymer_component_library() -> ComponentLibrary:
    """Return a small component library with retained non-polymer fixtures."""

    return ComponentLibrary(
        templates={
            "LIG": ResidueTemplate(
                definition=ChemicalComponentDefinition(
                    component_id="LIG",
                    atom_names=("C1", "O1", "N1", "H1"),
                ),
                heavy_atom_semantics=HeavyAtomSemantics(
                    program=InternalCoordinateProgram.backbone_only(),
                    atom_order=("C1", "O1", "N1"),
                ),
                hydrogen_semantics=HydrogenSemantics(
                    plan_with_backbone=((("H1",), "class5", ("O1", "C1", "N1", 1)),),
                ),
            ),
            "ION": ResidueTemplate(
                definition=ChemicalComponentDefinition(
                    component_id="ION",
                    atom_names=("ZN",),
                ),
            ),
        }
    )

"""Bundled retained non-polymer registry over packaged canonical assets."""

import gzip
import json
from collections.abc import Mapping
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType
from typing import cast

from protrepair.chemistry.component.graph import ChemicalComponentDefinition
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.component.semantics import IdealGeometryHydrogenSemantics
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.nonstandard.registry import (
    NonstandardComponentRecord,
    expect_mapping,
    expect_sequence,
    parse_nonstandard_component_record,
)
from protrepair.chemistry.restraint.library import RestraintLibrary
from protrepair.chemistry.retained_non_polymer.support_policy import (
    supports_bundled_retained_non_polymer_hydrogenation,
)


@dataclass(frozen=True, slots=True)
class BundledRetainedNonPolymerRegistry:
    """Lazy-loaded registry for curated retained non-polymer records."""

    records: Mapping[str, NonstandardComponentRecord]

    def __post_init__(self) -> None:
        normalized_records = {
            component_id.strip().upper(): record
            for component_id, record in self.records.items()
        }
        object.__setattr__(self, "records", MappingProxyType(normalized_records))

    def get(self, component_id: str) -> NonstandardComponentRecord | None:
        """Return one bundled retained non-polymer record if present."""

        return self.records.get(component_id.strip().upper())

    def component_library(self) -> ComponentLibrary:
        """Project bundled retained non-polymers into a component library."""

        return ComponentLibrary(
            templates={
                component_id: retained_non_polymer_record_to_template(record)
                for component_id, record in self.records.items()
            }
        )

    def restraint_library(self) -> RestraintLibrary:
        """Project bundled retained non-polymers into restraint templates."""

        return RestraintLibrary(
            templates={
                component_id: record.to_restraint_template()
                for component_id, record in self.records.items()
            }
        )


def retained_non_polymer_record_to_template(
    record: NonstandardComponentRecord,
) -> ResidueTemplate:
    """Project one retained non-polymer record into a canonical template."""

    idealized_component = record.to_idealized_component()
    heavy_atom_names = idealized_component.heavy_atom_names()
    formal_charges = {
        atom.atom_name: atom.formal_charge
        for atom in idealized_component.heavy_atoms()
        if atom.formal_charge != 0
    }
    hydrogen_semantics = None
    if supports_bundled_retained_non_polymer_hydrogenation(record.component_id):
        hydrogen_semantics = IdealGeometryHydrogenSemantics(
            component=idealized_component
        )

    return ResidueTemplate(
        definition=ChemicalComponentDefinition(
            component_id=record.component_id,
            atom_names=heavy_atom_names,
            bonds=idealized_component.heavy_bonds(),
            formal_charges=formal_charges,
        ),
        lineage_parent_component_id=record.parent_standard_id,
        backbone_family_component_id=(
            record.parent_standard_id or record.component_id
        ),
        preferred_atom_order=heavy_atom_names,
        heavy_atom_semantics=None,
        hydrogen_semantics=hydrogen_semantics,
    )


def bundled_retained_non_polymer_asset_path() -> str:
    """Return the packaged path for the retained non-polymer asset."""

    return str(
        files("protrepair.chemistry.resources").joinpath(
            "retained_non_polymer_components.json.gz"
        )
    )


@lru_cache(maxsize=1)
def build_bundled_retained_non_polymer_registry(
) -> BundledRetainedNonPolymerRegistry:
    """Return the bundled retained non-polymer registry."""

    asset_path = files("protrepair.chemistry.resources").joinpath(
        "retained_non_polymer_components.json.gz"
    )
    with asset_path.open("rb") as handle:
        payload = gzip.decompress(handle.read()).decode("utf-8")

    raw_payload = cast(object, json.loads(payload))
    return parse_bundled_retained_non_polymer_registry(raw_payload)


@lru_cache(maxsize=1)
def build_bundled_retained_non_polymer_component_library() -> ComponentLibrary:
    """Return the component-library projection of the bundled registry."""

    return build_bundled_retained_non_polymer_registry().component_library()


@lru_cache(maxsize=1)
def build_bundled_retained_non_polymer_restraint_library() -> RestraintLibrary:
    """Return the restraint-library projection of the bundled registry."""

    return build_bundled_retained_non_polymer_registry().restraint_library()


def parse_bundled_retained_non_polymer_registry(
    payload: object,
) -> BundledRetainedNonPolymerRegistry:
    """Parse one packaged retained non-polymer registry payload."""

    payload_mapping = expect_mapping(payload, path="root")
    components = expect_sequence(payload_mapping.get("components"), path="components")
    records = {
        record.component_id: record
        for record in (
            parse_nonstandard_component_record(
                item,
                path=f"components[{index}]",
            )
            for index, item in enumerate(components)
        )
    }
    return BundledRetainedNonPolymerRegistry(records=records)

"""Packaged mapped coordination data, independent of execution policy."""

import json
from collections.abc import Mapping
from functools import lru_cache
from importlib.resources import files
from types import MappingProxyType
from typing import cast

from protrepair.chemistry.inference.coordination_microstates import (
    CoordinationMicrostates,
)
from protrepair.chemistry.nonstandard.registry import (
    expect_int,
    expect_mapping,
    expect_sequence,
    expect_string,
    expect_string_sequence,
)


def coordination_microstate_family(component_id: str) -> CoordinationMicrostates | None:
    """Load registered chemistry; an unknown component has no implied profile.

    Parameters
    ----------
    component_id : str
        Component data key, normalized once before lookup.

    Returns
    -------
    CoordinationMicrostates or None
        Validated finite chemistry, or None without invoking native chemistry.

    Raises
    ------
    ValueError
        Registered data are malformed or chemically inconsistent.
    RdkitUnavailableError
        A registered family's native interpretation is unavailable.
    """
    return _load_family(component_id.strip().upper())


@lru_cache(maxsize=64)
def _load_family(component_id: str) -> CoordinationMicrostates | None:
    record = _catalogue_records().get(component_id)
    if record is None:
        return None
    return CoordinationMicrostates(
        heavy_atom_names=expect_string_sequence(
            record.get("heavy_atom_names"), path="heavy_atom_names"
        ),
        smiles_states=expect_string_sequence(
            record.get("smiles_states"), path="smiles_states"
        ),
        provenance=expect_string(record.get("provenance"), path="provenance"),
    )


@lru_cache(maxsize=1)
def _catalogue_records() -> Mapping[str, Mapping[str, object]]:
    asset = files("protrepair.chemistry.resources").joinpath(
        "coordination_microstates.json"
    )
    payload = expect_mapping(
        cast(object, json.loads(asset.read_text(encoding="utf-8"))),
        path="coordination_microstates",
    )
    if expect_int(payload.get("schema_version"), path="schema_version") != 1:
        raise ValueError("unsupported coordination catalogue schema")
    records = {}
    for item in expect_sequence(payload.get("components"), path="components"):
        record = expect_mapping(item, path="component")
        key = expect_string(record.get("component_id"), path="component_id").upper()
        if key in records:
            raise ValueError("coordination catalogue repeats a component")
        records[key] = MappingProxyType(dict(record))
    return MappingProxyType(records)

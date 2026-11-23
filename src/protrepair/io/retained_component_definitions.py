"""Read named covalent chemistry from explicitly supplied PDBx CCD files."""

from collections.abc import Iterator, Mapping
from os import PathLike
from pathlib import Path
from types import MappingProxyType

import gemmi

from protrepair.chemistry.inference.ccd_graph import component_heavy_molecule
from protrepair.chemistry.retained_non_polymer.definition import (
    NAMED_COMPONENT_ELEMENTS,
    ComponentAtom,
    ComponentBond,
    RetainedComponentDefinition,
)
from protrepair.errors import RdkitUnavailableError


class _ComponentLookup(Mapping[str, RetainedComponentDefinition]):
    def __init__(self, models: dict[str, RetainedComponentDefinition]) -> None:
        self._models = MappingProxyType(models)

    def __getitem__(self, key: str) -> RetainedComponentDefinition:
        return self._models[key.strip().upper()]

    def __iter__(self) -> Iterator[str]:
        return iter(self._models)

    def __len__(self) -> int:
        return len(self._models)


def read_retained_component_definitions(
    path: str | PathLike[str], *more_paths: str | PathLike[str]
) -> Mapping[str, RetainedComponentDefinition]:
    """Read named molecular graphs for use with site-specific chemistry overrides.

    Parameters
    ----------
    path, *more_paths : str or PathLike
        Local PDBx CCD files describing isolated molecules with explicit hydrogens.

    Returns
    -------
    Mapping[str, RetainedComponentDefinition]
        Read-only lookup by component ID. Bind a selected definition with
        `for_residue(residue_id)`.

    Raises
    ------
    ValueError
        A file is not a supported component dictionary, a component ID is
        repeated, or the declared graph cannot be represented faithfully.
    RdkitUnavailableError
        RDKit is unavailable for graph validation.
    """
    definitions: dict[str, RetainedComponentDefinition] = {}
    for source in (path, *more_paths):
        file = Path(source)
        try:
            document = gemmi.cif.read_file(str(file))
        except (RuntimeError, ValueError) as error:
            raise ValueError(f"{file}: {error}") from error
        if not len(document):
            raise ValueError(f"{file}: empty component dictionary")
        for block in document:
            component = _text(block.find_value("_chem_comp.id"))
            if not component or block.find_values("_atom_site.id"):
                raise ValueError(f"{file}: block {block.name!r} is not a CCD component")
            if component in definitions:
                raise ValueError(f"{file}: duplicate component ID {component}")
            try:
                definition = _read_definition(block, component)
                component_heavy_molecule(definition)
            except RdkitUnavailableError:
                raise
            except (ValueError, RuntimeError, KeyError, OverflowError) as error:
                raise ValueError(f"{file}: component {component}: {error}") from error
            definitions[component] = definition
    return _ComponentLookup(definitions)


def _text(raw: str | None) -> str:
    return "" if raw is None else gemmi.cif.as_string(raw).strip().upper()


def _charge(raw: str, field: str) -> int:
    text = _text(raw)
    digits = text[1:] if text.startswith(("+", "-")) else text
    if not digits.isascii() or not digits.isdecimal():
        raise ValueError(f"expected integer {field}, found {raw!r}")
    return int(text)


def _column_value(
    row: gemmi.cif.Table.Row, index: int, allowed: tuple[str, ...], field: str
) -> str | None:
    if not row.has(index):
        return None
    value = _text(row[index])
    if value not in allowed:
        raise ValueError(f"unsupported {field}: {row[index]}")
    return value


def _read_definition(
    block: gemmi.cif.Block, component: str
) -> RetainedComponentDefinition:
    atoms = {}
    for row in block.find(
        "_chem_comp_atom.",
        [
            "comp_id",
            "atom_id",
            "type_symbol",
            "?charge",
            "?pdbx_aromatic_flag",
            "?pdbx_stereo_config",
        ],
    ):
        name, element = _text(row[1]), _text(row[2])
        if _text(row[0]) != component or not name or name in atoms:
            raise ValueError(f"invalid or duplicate atom ID {name!r} / component ID")
        if element not in NAMED_COMPONENT_ELEMENTS:
            raise ValueError(f"unsupported element or isotope {element!r} at {name}")
        charge = _charge(row[3], f"charge at {name}") if row.has(3) else 0
        aromatic = _column_value(row, 4, ("Y", "N"), f"{name} aromaticity")
        stereo = _column_value(row, 5, ("N", "R", "S"), f"{name} stereo")
        atoms[name] = ComponentAtom(
            name,
            element,
            charge,
            None if aromatic is None else aromatic == "Y",
            None if stereo == "N" else stereo,
        )
    if sum(a.element != "H" for a in atoms.values()) < 2:
        raise ValueError("expected a multi-heavy-atom CCD atom table")
    net = block.find_value("_chem_comp.pdbx_formal_charge")
    if (
        net is not None
        and not gemmi.cif.is_null(net)
        and _charge(net, "component formal charge")
        != sum(a.charge for a in atoms.values())
    ):
        raise ValueError("component formal charge disagrees with atom charges")

    bonds = {}
    for row in block.find(
        "_chem_comp_bond.",
        [
            "comp_id",
            "atom_id_1",
            "atom_id_2",
            "value_order",
            "?pdbx_aromatic_flag",
            "?pdbx_stereo_config",
            "?pdbx_metal_coordination_flag",
            "?pdbx_metal_pi_flag",
        ],
    ):
        a, b = sorted((_text(row[1]), _text(row[2])))
        if _text(row[0]) != component or a not in atoms or b not in atoms:
            raise ValueError(f"unknown endpoint or wrong component at {a}-{b}")
        if a == b or (a, b) in bonds:
            raise ValueError(f"duplicate or self bond {a}-{b}")
        order = _text(row[3])
        if order not in {"SING", "DOUB", "TRIP", "AROM"}:
            raise ValueError(f"unsupported bond order {order!r} at {a}-{b}")
        aromatic = _column_value(row, 4, ("Y", "N"), f"{a}-{b} aromaticity")
        stereo = _column_value(row, 5, ("N", "E", "Z"), f"{a}-{b} stereo")
        for index in (6, 7):
            _column_value(row, index, ("N",), f"{a}-{b} metal flag")
        bonds[a, b] = ComponentBond(
            a,
            b,
            order,
            None if aromatic is None else aromatic == "Y",
            None if stereo == "N" else stereo,
        )
    if not bonds:
        raise ValueError("expected a named CCD bond table")
    return RetainedComponentDefinition(
        component,
        tuple(sorted(atoms.values(), key=lambda a: (a.element == "H", a.name))),
        tuple(bonds[k] for k in sorted(bonds)),
    )

"""Caller-supplied geometry restraints for observed polymer X-H bonds."""

from collections import Counter
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from decimal import Decimal, InvalidOperation
from enum import Enum
from math import isfinite
from types import MappingProxyType

import gemmi

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.idealized import IdealizedComponentAtom
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.restraint.monomer import read_monomer_chemistry
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import AtomSite, ResidueSite
from protrepair.structure.labels import AtomRef
from protrepair.structure.topology import BondRelationshipType

BACKBONE_HEAVY_NAMES = frozenset(("N", "CA", "C", "O", "OXT"))
_FIXED_AROMATIC_COMPONENTS = frozenset(("PHE", "TYR", "TRP"))
_RESTRAINT_SIGMAS = 4


class HydrogenLengthConvention(str, Enum):
    """Caller-declared coordinate basis for X-H restraint comparisons."""

    UNKNOWN = "unknown"
    ELECTRON_CLOUD = "electron_cloud"
    NUCLEAR = "nuclear"


class HydrogenLengthStatus(str, Enum):
    """Length comparison, parent-position contradiction or lack of assessment."""

    WITHIN_REFERENCE = "within_reference"
    BETWEEN_REFERENCES = "between_references"
    TOO_SHORT = "too_short"
    TOO_LONG = "too_long"
    PARENT_POSITION_CONFLICT = "parent_position_conflict"
    UNASSESSED = "unassessed"


@dataclass(frozen=True, slots=True)
class HydrogenLengthComparison:
    """A distance compared to qualified restraints, not a validity judgment.

    Parameters
    ----------
    source : str
        Identity and digest of the restraint input.
    bounds_angstrom : tuple[float, float]
        Closed limits used for this comparison.
    profiles : tuple[tuple[HydrogenLengthConvention, tuple[float, float]], ...]
        Individual intervals used to build the decision limits.
    matching_conventions : tuple[HydrogenLengthConvention, ...]
        Individual intervals that contain the measurement, possibly none.
    outer_margin_angstrom : float
        Distance from the nearer comparison limit, negative outside the
        limits. A value between disjoint intervals has a positive margin
        even though no individual convention matches it.
    target_angstrom : float or None
        Exact reference target for a declared convention. An envelope has no
        placement target.
    """

    source: str
    bounds_angstrom: tuple[float, float]
    profiles: tuple[tuple[HydrogenLengthConvention, tuple[float, float]], ...]
    matching_conventions: tuple[HydrogenLengthConvention, ...]
    outer_margin_angstrom: float
    target_angstrom: float | None = None

    def outside_status(self, distance_angstrom: float) -> HydrogenLengthStatus | None:
        """Return the direction of a deviation from the closed comparison bounds.

        Parameters
        ----------
        distance_angstrom : float
            Measured finite distance in angstroms.

        Returns
        -------
        HydrogenLengthStatus or None
            TOO_SHORT or TOO_LONG, or no outer-bound deviation. This does not
            test agreement with either individual convention.
        """
        low, high = self.bounds_angstrom
        if distance_angstrom < low:
            return HydrogenLengthStatus.TOO_SHORT
        if distance_angstrom > high:
            return HydrogenLengthStatus.TOO_LONG
        return None


@dataclass(frozen=True, slots=True)
class _Band:
    target: float
    low: float
    high: float


_Bands = tuple[tuple[HydrogenLengthConvention, _Band], ...]
_Pair = tuple[str, str]


@dataclass(frozen=True, slots=True, init=False)
class HydrogenLengthReference:
    """State-qualified X-H restraints from one caller-supplied standard monomer.

    Parameters
    ----------
    text : str
        Loaded monomer CIF text with an explicit closed-shell graph. A missing
        target or ESD leaves that convention unavailable.
    component_id : str
        Standard amino-acid code of the ``comp_<code>`` block.
    source : str
        Reference identity. A SHA-256 of the loaded text is appended.

    Raises
    ------
    ValueError
        The graph, source identity or a supplied side-chain interval is invalid.
    RdkitUnavailableError
        RDKit is unavailable for validating the explicit chemical graph.

    Notes
    -----
    The reference must include the complete standard side-chain heavy scaffold
    attached to CA. Side chains also bonded to another backbone atom, such as
    proline, are outside this domain. Equivalent aromatic encodings are
    normalized for fixed rings; state-dependent ring bond orders must match.
    """

    component_id: str
    source: str
    _atoms: tuple[IdealizedComponentAtom, ...] = field(repr=False)
    _bonds: tuple[BondDefinition, ...] = field(repr=False)
    _hydrogen_counts: Mapping[str, int] = field(repr=False)
    _profiles: Mapping[_Pair, _Bands] = field(repr=False)

    def __init__(self, text: str, *, component_id: str, source: str) -> None:
        component_id = component_id.strip().upper()
        standard = build_standard_component_library().templates.get(component_id)
        if standard is None:
            raise ValueError("hydrogen length references require a standard amino acid")
        excluded = BACKBONE_HEAVY_NAMES - {"CA"}
        if any(
            standard.definition.bonded_atom_names(name) - BACKBONE_HEAVY_NAMES
            for name in excluded
        ):
            raise ValueError(
                "side-chain bonds to the backbone outside CA are unsupported"
            )
        block, component, provenance = read_monomer_chemistry(
            text,
            component_id=component_id,
            source=source,
            normalize_aromatic=component_id in _FIXED_AROMATIC_COMPONENTS,
        )
        atoms = tuple(a for a in component.heavy_atoms() if a.atom_name not in excluded)
        names = frozenset(a.atom_name for a in atoms)
        standard_names = set(standard.expected_polymer_heavy_atom_names()) - excluded
        bonds = tuple(
            b
            for b in component.heavy_bonds()
            if {b.atom_name_1, b.atom_name_2} <= names
        )
        expected_pairs = {
            frozenset((b.atom_name_1, b.atom_name_2))
            for b in standard.definition.bonds
            if {b.atom_name_1, b.atom_name_2} <= standard_names
        }
        if (
            names != standard_names
            or any(a.element != a.atom_name[0] for a in atoms)
            or expected_pairs
            != {frozenset((b.atom_name_1, b.atom_name_2)) for b in bonds}
        ):
            raise ValueError(
                "hydrogen length reference lacks the standard side-chain scaffold"
            )

        hydrogen_atoms = {a.atom_name: a for a in component.hydrogen_atoms()}
        attachments = [
            (parent, child, bond)
            for bond in component.bonds
            for parent, child in (
                (bond.atom_name_1, bond.atom_name_2),
                (bond.atom_name_2, bond.atom_name_1),
            )
            if parent in names - {"CA"} and child in hydrogen_atoms
        ]
        selected_pairs = {
            frozenset((parent, child)) for parent, child, _ in attachments
        }
        pairs: dict[_Pair, _Bands] = {}
        counts: Counter[str] = Counter()
        electron = _read_bands(block, "value_dist", selected_pairs)
        nuclear = _read_bands(block, "value_dist_nucleus", selected_pairs)
        for parent, child, bond in attachments:
            atom = hydrogen_atoms[child]
            if atom.formal_charge != 0 or bond.order != 1 or bond.aromatic:
                raise ValueError(
                    "hydrogen length references require neutral single-bonded H"
                )
            name = PolymerMicrostateSite.hydrogen_slot_name(
                AtomSite(child, atom.element)
            )
            key = (parent, name)
            if key in pairs:
                raise ValueError("hydrogen length reference repeats an isotope slot")
            counts[parent] += 1
            lookup = frozenset((parent, child))
            pairs[key] = tuple(
                (convention, interval)
                for convention, values in (
                    (HydrogenLengthConvention.ELECTRON_CLOUD, electron),
                    (HydrogenLengthConvention.NUCLEAR, nuclear),
                )
                if (interval := values.get(lookup)) is not None
            )

        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "source", provenance)
        object.__setattr__(self, "_atoms", atoms)
        object.__setattr__(self, "_bonds", bonds)
        object.__setattr__(self, "_hydrogen_counts", MappingProxyType(dict(counts)))
        object.__setattr__(self, "_profiles", MappingProxyType(pairs))

    def _matches_scaffold(
        self, structure: ProteinStructure, residue: ResidueSite
    ) -> bool:
        if residue.component_id != self.component_id:
            return False
        c = structure.constitution
        indices = {
            site.name: c.atom_index(AtomRef(residue.residue_id, site.name))
            for site in residue.atom_sites
            if not site.is_hydrogen()
            and (site.name == "CA" or site.name not in BACKBONE_HEAVY_NAMES)
        }
        if set(indices) != {a.atom_name for a in self._atoms}:
            return False

        topology = structure.topology
        for atom in self._atoms:
            index = indices[atom.atom_name]
            if c.atom_site_at(index).element_identity.chemical_symbol != atom.element:
                return False
            charge = topology.formal_charge(index)
            if (0 if charge is None else charge) != atom.formal_charge:
                return False

        expected = {
            frozenset((indices[b.atom_name_1], indices[b.atom_name_2])): b
            for b in self._bonds
        }
        standard = build_standard_component_library().require(self.component_id)
        stock = {
            frozenset((b.atom_name_1, b.atom_name_2)): b.order
            for b in standard.definition.bonds
        }
        observed = set()
        incident = topology.incident_bonds_by_atom()
        for name, index in indices.items():
            for bond in incident.get(index, ()):
                other = next(other for other in bond.endpoint_pair() if other != index)
                if bond.relationship_type is BondRelationshipType.HYDROGEN_BOND:
                    continue
                other_ref = c.atom_ref_at(other)
                if other_ref.residue_id != residue.residue_id:
                    return False
                if c.atom_site_at(other).is_hydrogen():
                    continue
                if name == "CA" and other_ref.atom_name in BACKBONE_HEAVY_NAMES:
                    continue
                pair = frozenset((index, other))
                definition = expected.get(pair)
                if (
                    definition is None
                    or bond.relationship_type is not BondRelationshipType.COVALENT
                ):
                    return False
                if (
                    definition.aromatic
                    and self.component_id in _FIXED_AROMATIC_COMPONENTS
                ):
                    # Accept the standard named Kekule form, not arbitrary single bonds.
                    if bond.order not in (1, 2) or (
                        not bond.aromatic
                        and bond.order != stock[frozenset((name, other_ref.atom_name))]
                    ):
                        return False
                elif (
                    bond.aromatic != definition.aromatic
                    or bond.order != definition.order
                ):
                    return False
                observed.add(pair)
        return observed == expected.keys()

    def _expected_hydrogen_count(self, parent: str) -> int:
        return self._hydrogen_counts.get(parent, 0)

    def _assigns_slot_elsewhere(self, parent: str, atom: AtomSite) -> bool:
        slot = PolymerMicrostateSite.hydrogen_slot_name(atom)
        return (parent, slot) not in self._profiles and any(
            name == slot for _, name in self._profiles
        )

    def _compare(
        self,
        parent: str,
        atom: AtomSite,
        convention: HydrogenLengthConvention,
        distance: float,
    ) -> tuple[HydrogenLengthStatus, HydrogenLengthComparison] | None:
        profiles = self._profiles.get(
            (parent, PolymerMicrostateSite.hydrogen_slot_name(atom)), ()
        )
        if convention is HydrogenLengthConvention.UNKNOWN:
            if len(profiles) != 2:
                return None
        else:
            profiles = tuple(
                (kind, band) for kind, band in profiles if kind is convention
            )
            if not profiles:
                return None

        lower = min(band.low for _, band in profiles)
        upper = max(band.high for _, band in profiles)
        matching = tuple(
            kind for kind, band in profiles if band.low <= distance <= band.high
        )
        comparison = HydrogenLengthComparison(
            self.source,
            (lower, upper),
            tuple((kind, (band.low, band.high)) for kind, band in profiles),
            matching,
            min(distance - lower, upper - distance),
            target_angstrom=profiles[0][1].target if len(profiles) == 1 else None,
        )
        outside = comparison.outside_status(distance)
        if outside is not None:
            return outside, comparison
        status = (
            HydrogenLengthStatus.WITHIN_REFERENCE
            if matching
            else HydrogenLengthStatus.BETWEEN_REFERENCES
        )
        return status, comparison


def index_hydrogen_length_references(
    references: Iterable[HydrogenLengthReference],
) -> dict[str, HydrogenLengthReference]:
    """Index valid references without silently replacing a component entry."""
    by_component: dict[str, HydrogenLengthReference] = {}
    for reference in references:
        if not isinstance(reference, HydrogenLengthReference):
            raise TypeError("references must contain HydrogenLengthReference values")
        if reference.component_id in by_component:
            raise ValueError(
                "supply at most one hydrogen length reference per component"
            )
        by_component[reference.component_id] = reference
    return by_component


def _read_bands(
    block: gemmi.cif.Block, target: str, selected_pairs: set[frozenset[str]]
) -> dict[frozenset[str], _Band]:
    rows = block.find(
        "_chem_comp_bond.", ("atom_id_1", "atom_id_2", target, target + "_esd")
    )
    result: dict[frozenset[str], _Band] = {}
    for row in rows:
        key = frozenset((row.str(0).upper(), row.str(1).upper()))
        center_text, esd_text = row.str(2), row.str(3)
        if key not in selected_pairs or not center_text or not esd_text:
            continue
        try:
            center, esd = Decimal(center_text), Decimal(esd_text)
            if not center.is_finite() or not esd.is_finite():
                raise InvalidOperation
            low, high = (
                float(center - _RESTRAINT_SIGMAS * esd),
                float(center + _RESTRAINT_SIGMAS * esd),
            )
        except (ArithmeticError, ValueError) as error:
            raise ValueError("invalid hydrogen length restraint interval") from error
        if not isfinite(high) or not 0 < low < high:
            raise ValueError("invalid hydrogen length restraint interval")
        result[key] = _Band(float(center), low, high)
    return result

"""Bond planning and materialization for continuous-relaxation problems."""

from collections import defaultdict
from collections.abc import Iterable
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from protrepair.chemistry import BondDefinition, HydrogenSemantics, ResidueTemplate
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.errors import RefinementError
from protrepair.state.structure_topology import (
    DisulfideEndpointMultiplicityContradiction,
)
from protrepair.structure.constitution import (
    AtomSite as ConstitutionAtomSite,
)
from protrepair.structure.constitution import ResidueSite, StructureConstitution
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.labels import ResidueId
from protrepair.structure.slots import AtomIndex, ResidueIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.structure.topology import (
    BondRelationshipType,
    TopologyBond,
    is_covalent_like_relationship,
)
from protrepair.transformer.continuous.support import (
    LocalBondPlanningSupportMode,
    LocalBondPlanningSupportResolution,
)

if TYPE_CHECKING:
    from protrepair.transformer.continuous.domain import ContinuousRelaxationRegion

PEPTIDE_BOND_ORDER = 1
HYDROGEN_ATTACHMENT_DISTANCE_MAX_ANGSTROM = 1.35


def residue_ids_are_sequential_peptide_neighbors(
    left_residue_id: ResidueId,
    right_residue_id: ResidueId,
) -> bool:
    """Return whether adjacent residue slots are plausibly sequence-adjacent."""

    return left_residue_id.immediately_precedes(right_residue_id)


@dataclass(frozen=True, order=True, slots=True)
class PlannedBond:
    """A covalent edge selected for a region, with order resolved before binding.

    Parameters
    ----------
    atom_index_1, atom_index_2 : AtomIndex
        Distinct constitution-native endpoints, stored in ascending order.
    order : int or None, default=1
        Integral order or unresolved evidence retained during context selection.
    aromatic : bool, default=False
        Whether binding uses an aromatic rather than integral RDKit bond type.

    Raises
    ------
    ValueError
        Endpoints coincide or a supplied order is nonpositive.
    """

    atom_index_1: AtomIndex
    atom_index_2: AtomIndex
    order: int | None = 1
    aromatic: bool = False

    def __post_init__(self) -> None:
        atom_index_1 = self.atom_index_1
        atom_index_2 = self.atom_index_2
        if atom_index_1 == atom_index_2:
            raise ValueError("planned bonds require two distinct atom slots")

        if atom_index_2.value < atom_index_1.value:
            atom_index_1, atom_index_2 = atom_index_2, atom_index_1

        if self.order is not None and self.order <= 0:
            raise ValueError("planned bond order must be positive")

        object.__setattr__(self, "atom_index_1", atom_index_1)
        object.__setattr__(self, "atom_index_2", atom_index_2)

    def involves(self, atom_index: AtomIndex) -> bool:
        """Return whether the bond references one atom slot."""

        return atom_index in (self.atom_index_1, self.atom_index_2)

    def other(self, atom_index: AtomIndex) -> AtomIndex | None:
        """Return the opposite endpoint for one bond endpoint if present."""

        if atom_index == self.atom_index_1:
            return self.atom_index_2

        if atom_index == self.atom_index_2:
            return self.atom_index_1

        return None

    def sort_key(self) -> tuple[int, int]:
        """Return one total ordering key for stable bond ordering."""

        return (self.atom_index_1.value, self.atom_index_2.value)


def planned_bond_from_topology_bond(bond: TopologyBond) -> PlannedBond | None:
    """Project a covalent-like edge for planning; binding also requires chemistry."""

    if not is_covalent_like_relationship(bond):
        return None

    return PlannedBond(
        atom_index_1=bond.atom_index_1,
        atom_index_2=bond.atom_index_2,
        order=bond.order,
        aromatic=bond.aromatic,
    )


def inter_residue_bonds(
    snapshot: ProteinStructureSnapshot,
) -> tuple[PlannedBond, ...]:
    """Return inter-residue bonds across one whole snapshot."""

    canonical_pairs = {
        (bond.atom_index_1.value, bond.atom_index_2.value)
        for bond in snapshot.structure.topology.bonds
        if bond.relationship_type is not BondRelationshipType.UNKNOWN
    }
    by_pair = {
        bond.sort_key(): bond
        for bond in _peptide_bonds(snapshot)
        if bond.sort_key() not in canonical_pairs
    }
    by_pair.update(
        (bond.sort_key(), bond) for bond in _topology_inter_residue_bonds(snapshot)
    )
    return tuple(by_pair[pair] for pair in sorted(by_pair))


def directly_bonded_context_residue_indices(
    constitution: StructureConstitution,
    inter_residue_bonds: tuple[PlannedBond, ...],
    *,
    selected_residue_index_set: set[ResidueIndex],
) -> set[ResidueIndex]:
    """Return residues directly bonded to selected residues."""

    context_residue_index_set: set[ResidueIndex] = set()
    for bond in inter_residue_bonds:
        left_residue_index = constitution.residue_index_for_atom_index(
            bond.atom_index_1
        )
        right_residue_index = constitution.residue_index_for_atom_index(
            bond.atom_index_2
        )
        left_selected = left_residue_index in selected_residue_index_set
        right_selected = right_residue_index in selected_residue_index_set
        if left_selected and not right_selected:
            context_residue_index_set.add(right_residue_index)
        elif right_selected and not left_selected:
            context_residue_index_set.add(left_residue_index)

    return context_residue_index_set


def plan_continuous_region_bonds(
    region: "ContinuousRelaxationRegion",
    component_library: ComponentLibrary,
    *,
    support_by_residue_index: dict[ResidueIndex, LocalBondPlanningSupportResolution]
    | None = None,
) -> tuple[PlannedBond, ...]:
    """Return all planned bonds inside one included region."""

    constitution = region.snapshot.structure.constitution
    included_residue_index_set = set(region.included_residue_indices)
    _require_disulfide_endpoint_multiplicity_realizability(
        region,
        included_residue_index_set=included_residue_index_set,
    )

    if support_by_residue_index is None:
        support_by_residue_index = region.require_local_bond_planning_support(
            component_library
        )

    bond_set: set[PlannedBond] = set(
        _topology_local_bonds(
            region,
            included_residue_index_set=included_residue_index_set,
        )
    )
    canonical_hydrogen_pairs: dict[AtomIndex, set[tuple[int, int]]] = defaultdict(set)
    for bond in bond_set:
        for index in (bond.atom_index_1, bond.atom_index_2):
            if constitution.atom_site_at(index).is_hydrogen():
                canonical_hydrogen_pairs[index].add(bond.sort_key())
    for residue_index in region.included_residue_indices:
        residue_site = region.residue_site(residue_index)
        template = component_library.get(residue_site.component_id)
        residue_geometry = region.residue_geometry(residue_index)
        if template is None:
            support_resolution = support_by_residue_index[residue_index]
            if (
                support_resolution.mode
                is LocalBondPlanningSupportMode.NONBONDING_PASSIVE_CONTEXT
            ):
                continue

            if (
                support_resolution.mode
                is not LocalBondPlanningSupportMode.TEMPLATE_LESS_PASSIVE_CONTEXT
            ):
                continue

            bond_set.update(
                _residue_local_bonds(
                    region.snapshot.structure.constitution,
                    residue_index,
                    residue_site,
                    support_resolution.fallback_bond_definitions,
                )
            )
            bond_set.update(
                _inferred_hydrogen_bonds(
                    region.snapshot.structure.constitution,
                    residue_index,
                    residue_site,
                    residue_geometry=residue_geometry,
                    explicit_hydrogen_atom_names=(
                        _explicit_hydrogen_atom_names_in_bonds(
                            region.snapshot.structure.constitution,
                            residue_index,
                            bond_set,
                        )
                    ),
                )
            )
            continue

        bond_set.update(
            _residue_local_bonds(
                region.snapshot.structure.constitution,
                residue_index,
                residue_site,
                template.definition.bonds,
            )
        )
        template_hydrogen_bonds = _template_hydrogen_bonds(
            region.snapshot.structure.constitution,
            residue_index,
            residue_site,
            template=template,
        )
        bond_set.update(template_hydrogen_bonds)
        bond_set.update(
            _inferred_hydrogen_bonds(
                region.snapshot.structure.constitution,
                residue_index,
                residue_site,
                residue_geometry=residue_geometry,
                explicit_hydrogen_atom_names={
                    region.snapshot.structure.constitution.atom_site_at(atom_index).name
                    for bond in template_hydrogen_bonds
                    for atom_index in (
                        bond.atom_index_1,
                        bond.atom_index_2,
                    )
                    if region.snapshot.structure.constitution.atom_site_at(
                        atom_index
                    ).name.startswith("H")
                },
            )
        )

    bond_set.update(
        bond
        for bond in region.inter_residue_bonds
        if constitution.residue_index_for_atom_index(bond.atom_index_1)
        in included_residue_index_set
        and constitution.residue_index_for_atom_index(bond.atom_index_2)
        in included_residue_index_set
    )
    # A canonical H attachment excludes competing template/proximity parents,
    # not just competing orders at the same endpoint pair.
    by_pair = {
        bond.sort_key(): bond
        for bond in bond_set
        if all(
            index not in canonical_hydrogen_pairs
            or bond.sort_key() in canonical_hydrogen_pairs[index]
            for index in (bond.atom_index_1, bond.atom_index_2)
        )
    }
    # Fallback templates supply missing edges, never competing orders for a
    # canonical edge whose chemistry may differ from the default microstate.
    for topology_bond in region.snapshot.structure.topology.bonds:
        if any(
            constitution.residue_index_for_atom_index(index)
            not in included_residue_index_set
            for index in topology_bond.endpoint_pair()
        ):
            continue
        planned = planned_bond_from_topology_bond(topology_bond)
        if planned is not None:
            by_pair[planned.sort_key()] = planned
        elif topology_bond.relationship_type is not BondRelationshipType.UNKNOWN:
            by_pair.pop(
                (topology_bond.atom_index_1.value, topology_bond.atom_index_2.value),
                None,
            )
        elif topology_bond.order is not None:
            pair = (topology_bond.atom_index_1.value, topology_bond.atom_index_2.value)
            fallback = by_pair.get(pair)
            if fallback is not None:
                by_pair[pair] = replace(fallback, order=topology_bond.order)
    return tuple(by_pair[pair] for pair in sorted(by_pair))


def _require_disulfide_endpoint_multiplicity_realizability(
    region: "ContinuousRelaxationRegion",
    *,
    included_residue_index_set: set[ResidueIndex],
) -> None:
    """Reject projected bond graphs with multiply assigned disulfide sulfur."""

    structure = region.snapshot.structure
    included_residue_ids = frozenset(
        structure.constitution.residue_site_at(residue_index).residue_id
        for residue_index in included_residue_index_set
    )
    contradictions = tuple(
        contradiction
        for contradiction in (
            DisulfideEndpointMultiplicityContradiction.all_from_structure(structure)
        )
        if contradiction.is_contradictory_in_residue_projection(included_residue_ids)
    )
    if not contradictions:
        return

    contradiction_details = "; ".join(
        f"{contradiction.sulfur_atom_ref.display_token()} has "
        f"{contradiction.projected_pair_count(included_residue_ids)} relationships"
        for contradiction in contradictions
    )
    raise RefinementError(
        "continuous relaxation cannot realize multiple canonical disulfide "
        f"relationships at one CYS SG endpoint: {contradiction_details}"
    )


def _topology_inter_residue_bonds(
    snapshot: ProteinStructureSnapshot,
) -> tuple[PlannedBond, ...]:
    """Return execution-admissible topology inter-residue bonds."""

    constitution = snapshot.structure.constitution
    planned_bonds: list[PlannedBond] = []
    for topology_bond in snapshot.structure.topology.bonds:
        if constitution.residue_index_for_atom_index(
            topology_bond.atom_index_1
        ) == constitution.residue_index_for_atom_index(topology_bond.atom_index_2):
            continue

        planned_bond = planned_bond_from_topology_bond(topology_bond)
        if planned_bond is not None:
            planned_bonds.append(planned_bond)

    return tuple(planned_bonds)


def _peptide_bonds(
    snapshot: ProteinStructureSnapshot,
) -> tuple[PlannedBond, ...]:
    """Return peptide C-N bonds across adjacent polymer residues."""

    constitution = snapshot.structure.constitution
    bonds: list[PlannedBond] = []
    for chain_site in snapshot.structure.constitution.chains:
        for left_residue, right_residue in zip(
            chain_site.residues,
            chain_site.residues[1:],
            strict=False,
        ):
            if (
                left_residue.is_hetero
                or right_residue.is_hetero
                or not residue_ids_are_sequential_peptide_neighbors(
                    left_residue.residue_id,
                    right_residue.residue_id,
                )
                or not left_residue.has_atom_site("C")
                or not right_residue.has_atom_site("N")
            ):
                continue

            bonds.append(
                PlannedBond(
                    atom_index_1=constitution.atom_index_in_residue(
                        constitution.residue_index(left_residue.residue_id),
                        "C",
                    ),
                    atom_index_2=constitution.atom_index_in_residue(
                        constitution.residue_index(right_residue.residue_id),
                        "N",
                    ),
                    order=PEPTIDE_BOND_ORDER,
                )
            )

    return tuple(bonds)


def _topology_local_bonds(
    region: "ContinuousRelaxationRegion",
    *,
    included_residue_index_set: set[ResidueIndex],
) -> tuple[PlannedBond, ...]:
    """Return execution-admissible topology bonds inside included residues."""

    constitution = region.snapshot.structure.constitution
    planned_bonds: list[PlannedBond] = []
    for topology_bond in region.snapshot.structure.topology.bonds:
        left_residue_index = constitution.residue_index_for_atom_index(
            topology_bond.atom_index_1
        )
        right_residue_index = constitution.residue_index_for_atom_index(
            topology_bond.atom_index_2
        )
        if left_residue_index != right_residue_index:
            continue
        if left_residue_index not in included_residue_index_set:
            continue

        planned_bond = planned_bond_from_topology_bond(topology_bond)
        if planned_bond is not None:
            planned_bonds.append(planned_bond)

    return tuple(planned_bonds)


def _residue_local_bonds(
    constitution: StructureConstitution,
    residue_index: ResidueIndex,
    residue_site: ResidueSite,
    bond_definitions: Iterable[BondDefinition],
) -> tuple[PlannedBond, ...]:
    """Return present residue-local template bonds for one residue."""

    present_atom_names = set(residue_site.atom_site_names())
    return tuple(
        PlannedBond(
            atom_index_1=constitution.atom_index_in_residue(
                residue_index,
                bond.atom_name_1,
            ),
            atom_index_2=constitution.atom_index_in_residue(
                residue_index,
                bond.atom_name_2,
            ),
            order=bond.order,
            aromatic=bond.aromatic,
        )
        for bond in bond_definitions
        if bond.atom_name_1 in present_atom_names
        and bond.atom_name_2 in present_atom_names
    )


def _inferred_hydrogen_bonds(
    constitution: StructureConstitution,
    residue_index: ResidueIndex,
    residue_site: ResidueSite,
    *,
    residue_geometry: ResidueGeometry,
    explicit_hydrogen_atom_names: set[str],
) -> tuple[PlannedBond, ...]:
    """Infer residue-local hydrogen attachments when templates omit hydrogens."""

    heavy_atom_sites = tuple(
        atom_site
        for atom_site in residue_site.atom_sites
        if not atom_site.is_hydrogen()
    )
    inferred_bonds: list[PlannedBond] = []
    for hydrogen_atom_site in residue_site.atom_sites:
        if not hydrogen_atom_site.is_hydrogen():
            continue

        if hydrogen_atom_site.name in explicit_hydrogen_atom_names:
            continue

        anchor_atom = _nearest_hydrogen_anchor(
            hydrogen_atom_name=hydrogen_atom_site.name,
            residue_geometry=residue_geometry,
            heavy_atom_sites=heavy_atom_sites,
        )
        if anchor_atom is None:
            continue

        inferred_bonds.append(
            PlannedBond(
                atom_index_1=constitution.atom_index_in_residue(
                    residue_index,
                    anchor_atom,
                ),
                atom_index_2=constitution.atom_index_in_residue(
                    residue_index,
                    hydrogen_atom_site.name,
                ),
            )
        )

    return tuple(inferred_bonds)


def _explicit_hydrogen_atom_names_in_bonds(
    constitution: StructureConstitution,
    residue_index: ResidueIndex,
    bonds: Iterable[PlannedBond],
) -> set[str]:
    """Return residue-local hydrogen names already represented by planned bonds."""

    hydrogen_atom_names: set[str] = set()
    for bond in bonds:
        for atom_index in (bond.atom_index_1, bond.atom_index_2):
            if constitution.residue_index_for_atom_index(atom_index) != residue_index:
                continue

            atom_site = constitution.atom_site_at(atom_index)
            if atom_site.is_hydrogen():
                hydrogen_atom_names.add(atom_site.name)

    return hydrogen_atom_names


def _template_hydrogen_bonds(
    constitution: StructureConstitution,
    residue_index: ResidueIndex,
    residue_site: ResidueSite,
    *,
    template: ResidueTemplate,
) -> tuple[PlannedBond, ...]:
    """Return planned hydrogen bonds derived from one template hydrogen plan."""

    hydrogen_semantics = template.hydrogen_semantics
    if (
        hydrogen_semantics is None
        or not isinstance(hydrogen_semantics, HydrogenSemantics)
        or hydrogen_semantics.rotatable_kind is not None
    ):
        return ()

    bond_set: set[PlannedBond] = set()
    for plan in (
        hydrogen_semantics.plan_with_backbone,
        hydrogen_semantics.plan_without_backbone,
    ):
        if plan is None:
            continue

        for hydrogen_atom_names, method_name, arguments in plan:
            anchor_atom_name = _hydrogen_plan_anchor_atom_name(
                method_name=method_name,
                arguments=arguments,
            )
            if anchor_atom_name is None or not residue_site.has_atom_site(
                anchor_atom_name
            ):
                continue

            for hydrogen_atom_name in hydrogen_atom_names:
                if not residue_site.has_atom_site(hydrogen_atom_name):
                    continue

                bond_set.add(
                    PlannedBond(
                        atom_index_1=constitution.atom_index_in_residue(
                            residue_index,
                            anchor_atom_name,
                        ),
                        atom_index_2=constitution.atom_index_in_residue(
                            residue_index,
                            hydrogen_atom_name,
                        ),
                    )
                )

    return tuple(sorted(bond_set, key=lambda bond: bond.sort_key()))


def _hydrogen_plan_anchor_atom_name(
    *,
    method_name: str,
    arguments: tuple[str | float | int, ...],
) -> str | None:
    """Return the anchor atom name encoded by one hydrogen-plan operation."""

    if method_name in {"class2", "class3", "class4", "calcCoordinate"}:
        anchor_index = 2
    elif method_name == "class5":
        anchor_index = 1
    else:
        return None

    if len(arguments) <= anchor_index:
        return None

    anchor_atom_name = arguments[anchor_index]
    if not isinstance(anchor_atom_name, str):
        return None

    normalized_anchor_atom_name = anchor_atom_name.strip().upper()
    return normalized_anchor_atom_name or None


def _nearest_hydrogen_anchor(
    *,
    hydrogen_atom_name: str,
    residue_geometry: ResidueGeometry,
    heavy_atom_sites: tuple[ConstitutionAtomSite, ...],
) -> str | None:
    """Return the nearest plausible heavy-atom anchor for one hydrogen atom."""

    nearest_atom_name: str | None = None
    nearest_distance = float("inf")
    hydrogen_atom_geometry = residue_geometry.atom_geometry(hydrogen_atom_name)
    for heavy_atom_site in heavy_atom_sites:
        pair_distance = hydrogen_atom_geometry.distance_to(
            residue_geometry.atom_geometry(heavy_atom_site.name)
        )
        if pair_distance > HYDROGEN_ATTACHMENT_DISTANCE_MAX_ANGSTROM:
            continue

        if pair_distance < nearest_distance:
            nearest_atom_name = heavy_atom_site.name
            nearest_distance = pair_distance

    return nearest_atom_name

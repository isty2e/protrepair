"""Transformer output deltas over constitution-owned structure slot spaces."""

from dataclasses import dataclass

from protrepair.structure.constitution import StructureConstitution
from protrepair.structure.slots import AtomIndex, ResidueIndex


def _normalized_component_id(component_id: str, *, field_name: str) -> str:
    """Normalize one residue component identifier for canonical rewrite facts."""

    normalized_component_id = component_id.strip().upper()
    if not normalized_component_id:
        raise ValueError(f"{field_name} must not be blank")

    return normalized_component_id


def _normalized_element(element: str, *, field_name: str) -> str:
    """Normalize one atom element token for canonical rewrite facts."""

    normalized_element = element.strip().upper()
    if not normalized_element:
        raise ValueError(f"{field_name} must not be blank")

    return normalized_element


def _deduplicated_atom_indices(
    atom_indices: tuple[AtomIndex, ...],
) -> tuple[AtomIndex, ...]:
    """Return atom indices deduplicated in first-seen order."""

    return tuple(dict.fromkeys(atom_indices))


def _deduplicated_residue_indices(
    residue_indices: tuple[ResidueIndex, ...],
) -> tuple[ResidueIndex, ...]:
    """Return residue indices deduplicated in first-seen order."""

    return tuple(dict.fromkeys(residue_indices))


@dataclass(frozen=True, order=True, slots=True)
class MovedAtomDelta:
    """One retained atom whose coordinates changed across one delta."""

    before_atom_index: AtomIndex
    after_atom_index: AtomIndex


@dataclass(frozen=True, slots=True)
class ResidueIdentityRewrite:
    """Canonical residue-level component identity rewrite in after-space."""

    after_residue_index: ResidueIndex
    previous_component_id: str
    current_component_id: str

    def __post_init__(self) -> None:
        previous_component_id = _normalized_component_id(
            self.previous_component_id,
            field_name="residue identity rewrite previous_component_id",
        )
        current_component_id = _normalized_component_id(
            self.current_component_id,
            field_name="residue identity rewrite current_component_id",
        )
        if previous_component_id == current_component_id:
            raise ValueError(
                "residue identity rewrite must change the component identity"
            )

        object.__setattr__(self, "previous_component_id", previous_component_id)
        object.__setattr__(self, "current_component_id", current_component_id)


@dataclass(frozen=True, slots=True)
class ResidueTopologyRewrite:
    """Canonical residue-level topology reinterpretation in after-space."""

    after_residue_index: ResidueIndex
    affected_atom_indices: tuple[AtomIndex, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "affected_atom_indices",
            _deduplicated_atom_indices(self.affected_atom_indices),
        )


@dataclass(frozen=True, slots=True)
class AtomElementRewrite:
    """Canonical atom-site element rewrite in after-space."""

    after_atom_index: AtomIndex
    previous_element: str
    current_element: str

    def __post_init__(self) -> None:
        previous_element = _normalized_element(
            self.previous_element,
            field_name="atom element rewrite previous_element",
        )
        current_element = _normalized_element(
            self.current_element,
            field_name="atom element rewrite current_element",
        )
        if previous_element == current_element:
            raise ValueError("atom element rewrite must change the element")

        object.__setattr__(self, "previous_element", previous_element)
        object.__setattr__(self, "current_element", current_element)


@dataclass(frozen=True, slots=True)
class AtomFormalChargeRewrite:
    """Canonical atom-topology formal-charge rewrite in after-space."""

    after_atom_index: AtomIndex
    previous_formal_charge: int | None = None
    current_formal_charge: int | None = None

    def __post_init__(self) -> None:
        for field_name, value in (
            (
                "atom formal-charge rewrite previous_formal_charge",
                self.previous_formal_charge,
            ),
            (
                "atom formal-charge rewrite current_formal_charge",
                self.current_formal_charge,
            ),
        ):
            if value is not None and (
                isinstance(value, bool) or not isinstance(value, int)
            ):
                raise TypeError(f"{field_name} must be an integer or None")

        if self.previous_formal_charge == self.current_formal_charge:
            raise ValueError(
                "atom formal-charge rewrite must change the formal charge"
            )


@dataclass(frozen=True, slots=True)
class GraphBondState:
    """Canonical bond semantics on one graph edge."""

    order: int = 1
    aromatic: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.order, bool) or not isinstance(self.order, int):
            raise TypeError("graph bond state order must be an integer")
        if self.order <= 0:
            raise ValueError("graph bond state order must be positive")


@dataclass(frozen=True, slots=True)
class BondTopologyRewrite:
    """Canonical bond-semantics rewrite in after-space."""

    left_after_atom_index: AtomIndex
    right_after_atom_index: AtomIndex
    previous_bond: GraphBondState | None = None
    current_bond: GraphBondState | None = None

    def __post_init__(self) -> None:
        left_after_atom_index = self.left_after_atom_index
        right_after_atom_index = self.right_after_atom_index
        if left_after_atom_index == right_after_atom_index:
            raise ValueError(
                "bond topology rewrites require two distinct atom slots"
            )
        if right_after_atom_index.value < left_after_atom_index.value:
            left_after_atom_index, right_after_atom_index = (
                right_after_atom_index,
                left_after_atom_index,
            )

        if self.previous_bond is None and self.current_bond is None:
            raise ValueError(
                "bond topology rewrite must describe a previous or current bond"
            )
        if self.previous_bond == self.current_bond:
            raise ValueError("bond topology rewrite must change bond semantics")

        object.__setattr__(
            self,
            "left_after_atom_index",
            left_after_atom_index,
        )
        object.__setattr__(
            self,
            "right_after_atom_index",
            right_after_atom_index,
        )


def _deduplicated_moved_atoms(
    moved_atoms: tuple[MovedAtomDelta, ...],
) -> tuple[MovedAtomDelta, ...]:
    """Return moved-atom facts deduplicated by before/after correspondence."""

    ordered_moved_atoms: list[MovedAtomDelta] = []
    moved_atom_set: set[MovedAtomDelta] = set()
    for moved_atom in moved_atoms:
        if not isinstance(moved_atom, MovedAtomDelta):
            raise TypeError(
                "structure delta moved_atoms must be MovedAtomDelta values"
            )
        if moved_atom in moved_atom_set:
            continue

        ordered_moved_atoms.append(moved_atom)
        moved_atom_set.add(moved_atom)

    return tuple(ordered_moved_atoms)


def _deduplicated_residue_identity_rewrites(
    rewrites: tuple[ResidueIdentityRewrite, ...],
) -> tuple[ResidueIdentityRewrite, ...]:
    """Return residue identity rewrites deduplicated by after residue slot."""

    ordered_rewrites: list[ResidueIdentityRewrite] = []
    rewrites_by_index: dict[ResidueIndex, ResidueIdentityRewrite] = {}
    for rewrite in rewrites:
        if not isinstance(rewrite, ResidueIdentityRewrite):
            raise TypeError(
                "structure delta residue_identity_rewrites must be "
                "ResidueIdentityRewrite values"
            )

        existing_rewrite = rewrites_by_index.get(rewrite.after_residue_index)
        if existing_rewrite is None:
            rewrites_by_index[rewrite.after_residue_index] = rewrite
            ordered_rewrites.append(rewrite)
            continue

        if existing_rewrite != rewrite:
            raise ValueError(
                "structure delta cannot carry multiple residue identity "
                f"rewrites for residue slot {rewrite.after_residue_index.value}"
            )

    return tuple(ordered_rewrites)


def _deduplicated_residue_topology_rewrites(
    rewrites: tuple[ResidueTopologyRewrite, ...],
) -> tuple[ResidueTopologyRewrite, ...]:
    """Return residue topology rewrites deduplicated by after residue slot."""

    ordered_rewrites: list[ResidueTopologyRewrite] = []
    rewrites_by_index: dict[ResidueIndex, ResidueTopologyRewrite] = {}
    for rewrite in rewrites:
        if not isinstance(rewrite, ResidueTopologyRewrite):
            raise TypeError(
                "structure delta residue_topology_rewrites must be "
                "ResidueTopologyRewrite values"
            )

        existing_rewrite = rewrites_by_index.get(rewrite.after_residue_index)
        if existing_rewrite is None:
            rewrites_by_index[rewrite.after_residue_index] = rewrite
            ordered_rewrites.append(rewrite)
            continue

        if existing_rewrite != rewrite:
            raise ValueError(
                "structure delta cannot carry multiple residue topology "
                f"rewrites for residue slot {rewrite.after_residue_index.value}"
            )

    return tuple(ordered_rewrites)


def _deduplicated_atom_element_rewrites(
    rewrites: tuple[AtomElementRewrite, ...],
) -> tuple[AtomElementRewrite, ...]:
    """Return atom element rewrites deduplicated by after atom slot."""

    ordered_rewrites: list[AtomElementRewrite] = []
    rewrites_by_index: dict[AtomIndex, AtomElementRewrite] = {}
    for rewrite in rewrites:
        if not isinstance(rewrite, AtomElementRewrite):
            raise TypeError(
                "structure delta atom_element_rewrites must be "
                "AtomElementRewrite values"
            )

        existing_rewrite = rewrites_by_index.get(rewrite.after_atom_index)
        if existing_rewrite is None:
            rewrites_by_index[rewrite.after_atom_index] = rewrite
            ordered_rewrites.append(rewrite)
            continue

        if existing_rewrite != rewrite:
            raise ValueError(
                "structure delta cannot carry multiple atom element rewrites "
                f"for atom slot {rewrite.after_atom_index.value}"
            )

    return tuple(ordered_rewrites)


def _deduplicated_atom_formal_charge_rewrites(
    rewrites: tuple[AtomFormalChargeRewrite, ...],
) -> tuple[AtomFormalChargeRewrite, ...]:
    """Return atom formal-charge rewrites deduplicated by after atom slot."""

    ordered_rewrites: list[AtomFormalChargeRewrite] = []
    rewrites_by_index: dict[AtomIndex, AtomFormalChargeRewrite] = {}
    for rewrite in rewrites:
        if not isinstance(rewrite, AtomFormalChargeRewrite):
            raise TypeError(
                "structure delta atom_formal_charge_rewrites must be "
                "AtomFormalChargeRewrite values"
            )

        existing_rewrite = rewrites_by_index.get(rewrite.after_atom_index)
        if existing_rewrite is None:
            rewrites_by_index[rewrite.after_atom_index] = rewrite
            ordered_rewrites.append(rewrite)
            continue

        if existing_rewrite != rewrite:
            raise ValueError(
                "structure delta cannot carry multiple atom formal-charge "
                f"rewrites for atom slot {rewrite.after_atom_index.value}"
            )

    return tuple(ordered_rewrites)


def _deduplicated_bond_topology_rewrites(
    rewrites: tuple[BondTopologyRewrite, ...],
) -> tuple[BondTopologyRewrite, ...]:
    """Return bond rewrites deduplicated by canonical after-space atom pair."""

    ordered_rewrites: list[BondTopologyRewrite] = []
    rewrites_by_pair: dict[tuple[AtomIndex, AtomIndex], BondTopologyRewrite] = {}
    for rewrite in rewrites:
        if not isinstance(rewrite, BondTopologyRewrite):
            raise TypeError(
                "structure delta bond_topology_rewrites must be "
                "BondTopologyRewrite values"
            )

        pair = (
            rewrite.left_after_atom_index,
            rewrite.right_after_atom_index,
        )
        existing_rewrite = rewrites_by_pair.get(pair)
        if existing_rewrite is None:
            rewrites_by_pair[pair] = rewrite
            ordered_rewrites.append(rewrite)
            continue

        if existing_rewrite != rewrite:
            raise ValueError(
                "structure delta cannot carry multiple bond topology rewrites "
                f"for atom slots {pair[0].value} and {pair[1].value}"
            )

    return tuple(ordered_rewrites)


def _validate_atom_index(
    atom_index: AtomIndex,
    *,
    constitution: StructureConstitution,
    field_name: str,
) -> None:
    """Raise when one atom index falls outside one constitution."""

    if atom_index.value >= len(constitution.atom_slots):
        raise ValueError(
            f"{field_name} {atom_index.value} is outside the constitution atom slots"
        )


def _validate_residue_index(
    residue_index: ResidueIndex,
    *,
    constitution: StructureConstitution,
    field_name: str,
) -> None:
    """Raise when one residue index falls outside one constitution."""

    if residue_index.value >= len(constitution.residue_slots):
        raise ValueError(
            f"{field_name} {residue_index.value} is outside the "
            "constitution residue slots"
        )


@dataclass(frozen=True, slots=True)
class StructureDelta:
    """Canonical before/after slot delta from one structure mutation."""

    before_constitution: StructureConstitution
    after_constitution: StructureConstitution
    moved_atoms: tuple[MovedAtomDelta, ...] = ()
    created_atom_indices: tuple[AtomIndex, ...] = ()
    deleted_atom_indices: tuple[AtomIndex, ...] = ()
    created_residue_indices: tuple[ResidueIndex, ...] = ()
    deleted_residue_indices: tuple[ResidueIndex, ...] = ()
    residue_identity_rewrites: tuple[ResidueIdentityRewrite, ...] = ()
    residue_topology_rewrites: tuple[ResidueTopologyRewrite, ...] = ()
    atom_element_rewrites: tuple[AtomElementRewrite, ...] = ()
    atom_formal_charge_rewrites: tuple[AtomFormalChargeRewrite, ...] = ()
    bond_topology_rewrites: tuple[BondTopologyRewrite, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.before_constitution, StructureConstitution):
            raise TypeError(
                "structure delta before_constitution must be a StructureConstitution"
            )
        if not isinstance(self.after_constitution, StructureConstitution):
            raise TypeError(
                "structure delta after_constitution must be a StructureConstitution"
            )

        moved_atoms = _deduplicated_moved_atoms(self.moved_atoms)
        created_atom_indices = _deduplicated_atom_indices(self.created_atom_indices)
        deleted_atom_indices = _deduplicated_atom_indices(self.deleted_atom_indices)
        created_residue_indices = _deduplicated_residue_indices(
            self.created_residue_indices
        )
        deleted_residue_indices = _deduplicated_residue_indices(
            self.deleted_residue_indices
        )
        residue_identity_rewrites = _deduplicated_residue_identity_rewrites(
            self.residue_identity_rewrites
        )
        residue_topology_rewrites = _deduplicated_residue_topology_rewrites(
            self.residue_topology_rewrites
        )
        atom_element_rewrites = _deduplicated_atom_element_rewrites(
            self.atom_element_rewrites
        )
        atom_formal_charge_rewrites = _deduplicated_atom_formal_charge_rewrites(
            self.atom_formal_charge_rewrites
        )
        bond_topology_rewrites = _deduplicated_bond_topology_rewrites(
            self.bond_topology_rewrites
        )

        for moved_atom in moved_atoms:
            _validate_atom_index(
                moved_atom.before_atom_index,
                constitution=self.before_constitution,
                field_name="structure delta moved_atoms before_atom_index",
            )
            _validate_atom_index(
                moved_atom.after_atom_index,
                constitution=self.after_constitution,
                field_name="structure delta moved_atoms after_atom_index",
            )

        for atom_index in created_atom_indices:
            _validate_atom_index(
                atom_index,
                constitution=self.after_constitution,
                field_name="structure delta created_atom_indices",
            )
        for atom_index in deleted_atom_indices:
            _validate_atom_index(
                atom_index,
                constitution=self.before_constitution,
                field_name="structure delta deleted_atom_indices",
            )
        for residue_index in created_residue_indices:
            _validate_residue_index(
                residue_index,
                constitution=self.after_constitution,
                field_name="structure delta created_residue_indices",
            )
        for residue_index in deleted_residue_indices:
            _validate_residue_index(
                residue_index,
                constitution=self.before_constitution,
                field_name="structure delta deleted_residue_indices",
            )

        moved_before_atom_index_set = {
            moved_atom.before_atom_index for moved_atom in moved_atoms
        }
        moved_after_atom_index_set = {
            moved_atom.after_atom_index for moved_atom in moved_atoms
        }
        created_atom_index_set = set(created_atom_indices)
        deleted_atom_index_set = set(deleted_atom_indices)
        if moved_before_atom_index_set & deleted_atom_index_set:
            raise ValueError(
                "structure delta atom slots cannot be both moved and deleted "
                "in before-space"
            )
        if moved_after_atom_index_set & created_atom_index_set:
            raise ValueError(
                "structure delta atom slots cannot be both moved and created "
                "in after-space"
            )

        created_residue_index_set = set(created_residue_indices)
        identity_rewrite_residue_index_set = {
            rewrite.after_residue_index for rewrite in residue_identity_rewrites
        }
        topology_rewrite_residue_index_set = {
            rewrite.after_residue_index for rewrite in residue_topology_rewrites
        }
        if created_residue_index_set & identity_rewrite_residue_index_set:
            raise ValueError(
                "structure delta residue slots cannot be both created and "
                "identity-rewritten"
            )
        if created_residue_index_set & topology_rewrite_residue_index_set:
            raise ValueError(
                "structure delta residue slots cannot be both created and "
                "topology-rewritten"
            )

        atom_rewrite_index_set = {
            rewrite.after_atom_index for rewrite in atom_element_rewrites
        } | {
            rewrite.after_atom_index for rewrite in atom_formal_charge_rewrites
        }
        for rewrite in residue_topology_rewrites:
            for atom_index in rewrite.affected_atom_indices:
                _validate_atom_index(
                    atom_index,
                    constitution=self.after_constitution,
                    field_name=(
                        "structure delta residue_topology_rewrites "
                        "affected_atom_indices"
                    ),
                )
                if (
                    self.after_constitution.residue_index_for_atom_index(atom_index)
                    != rewrite.after_residue_index
                ):
                    raise ValueError(
                        "structure delta residue topology rewrites must only "
                        "reference atom slots inside the rewritten residue"
                    )
                atom_rewrite_index_set.add(atom_index)
        if created_atom_index_set & atom_rewrite_index_set:
            raise ValueError(
                "structure delta atom slots cannot be both created and rewritten"
            )

        for rewrite in atom_element_rewrites:
            _validate_atom_index(
                rewrite.after_atom_index,
                constitution=self.after_constitution,
                field_name="structure delta atom_element_rewrites",
            )
        for rewrite in atom_formal_charge_rewrites:
            _validate_atom_index(
                rewrite.after_atom_index,
                constitution=self.after_constitution,
                field_name="structure delta atom_formal_charge_rewrites",
            )
        for rewrite in residue_identity_rewrites:
            _validate_residue_index(
                rewrite.after_residue_index,
                constitution=self.after_constitution,
                field_name="structure delta residue_identity_rewrites",
            )
        for rewrite in residue_topology_rewrites:
            _validate_residue_index(
                rewrite.after_residue_index,
                constitution=self.after_constitution,
                field_name="structure delta residue_topology_rewrites",
            )
        for rewrite in bond_topology_rewrites:
            _validate_atom_index(
                rewrite.left_after_atom_index,
                constitution=self.after_constitution,
                field_name="structure delta bond_topology_rewrites left atom",
            )
            _validate_atom_index(
                rewrite.right_after_atom_index,
                constitution=self.after_constitution,
                field_name="structure delta bond_topology_rewrites right atom",
            )
            if created_atom_index_set & {
                rewrite.left_after_atom_index,
                rewrite.right_after_atom_index,
            }:
                raise ValueError(
                    "structure delta atom slots cannot be both created and "
                    "bond-topology-rewritten"
                )

        object.__setattr__(self, "moved_atoms", moved_atoms)
        object.__setattr__(
            self,
            "created_atom_indices",
            created_atom_indices,
        )
        object.__setattr__(
            self,
            "deleted_atom_indices",
            deleted_atom_indices,
        )
        object.__setattr__(
            self,
            "created_residue_indices",
            created_residue_indices,
        )
        object.__setattr__(
            self,
            "deleted_residue_indices",
            deleted_residue_indices,
        )
        object.__setattr__(
            self,
            "residue_identity_rewrites",
            residue_identity_rewrites,
        )
        object.__setattr__(
            self,
            "residue_topology_rewrites",
            residue_topology_rewrites,
        )
        object.__setattr__(
            self,
            "atom_element_rewrites",
            atom_element_rewrites,
        )
        object.__setattr__(
            self,
            "atom_formal_charge_rewrites",
            atom_formal_charge_rewrites,
        )
        object.__setattr__(
            self,
            "bond_topology_rewrites",
            bond_topology_rewrites,
        )

    def moved_atom_count(self) -> int:
        """Return the number of atom slots whose coordinates were updated."""

        return len(self.moved_atoms)

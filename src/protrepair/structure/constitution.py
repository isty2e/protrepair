"""Structure constitution facets for the redesigned ProtRepair package."""

from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, field

from protrepair.errors import (
    AtomNotFoundError,
    ChainNotFoundError,
    ModelInvariantError,
    ResidueNotFoundError,
)
from protrepair.structure.address_space import StructureAddressSpaceKey
from protrepair.structure.element import ElementIdentity
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex, ChainIndex, ResidueIndex


@dataclass(frozen=True, slots=True)
class AtomSite:
    """One materialized atom site without geometry or topology payload."""

    name: str
    element: str
    element_identity: ElementIdentity = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        name = self.name.strip().upper()
        element_identity = ElementIdentity(self.element)
        element = element_identity.source_symbol

        if not name:
            raise ValueError("atom-site name must not be blank")

        object.__setattr__(self, "name", name)
        object.__setattr__(self, "element", element)
        object.__setattr__(self, "element_identity", element_identity)

    def is_named(self, atom_name: str) -> bool:
        """Return whether this site matches the requested canonical atom name."""

        return self.name == atom_name.strip().upper()

    def is_hydrogen(self) -> bool:
        """Return whether this site has hydrogen chemical behavior."""

        return self.element_identity.is_hydrogen()

    def with_name(self, atom_name: str) -> "AtomSite":
        """Return a copy with an updated canonical atom-site name."""

        return type(self)(
            name=atom_name,
            element=self.element,
        )


@dataclass(frozen=True, slots=True)
class ResidueSite:
    """One residue scaffold without geometry or topology payload."""

    component_id: str
    residue_id: ResidueId
    atom_sites: tuple[AtomSite, ...]
    is_hetero: bool = False

    def __post_init__(self) -> None:
        component_id = self.component_id.strip().upper()
        atom_sites = tuple(self.atom_sites)
        atom_names = tuple(atom_site.name for atom_site in atom_sites)

        if not component_id:
            raise ValueError("component_id must not be blank")

        if len(atom_names) != len(set(atom_names)):
            raise ModelInvariantError(
                f"residue {self.residue_id.display_token()} contains duplicate "
                "atom-site names"
            )

        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "atom_sites", atom_sites)

    def atom_site_names(self) -> tuple[str, ...]:
        """Return atom-site names in residue order."""

        return tuple(atom_site.name for atom_site in self.atom_sites)

    def atom_site(self, atom_name: str) -> AtomSite:
        """Return one materialized atom site by name or raise if it is absent."""

        normalized_atom_name = atom_name.strip().upper()
        for atom_site in self.atom_sites:
            if atom_site.name == normalized_atom_name:
                return atom_site

        raise AtomNotFoundError(
            f"{self.residue_id.display_token()} has no atom named "
            f"{normalized_atom_name}"
        )

    def has_atom_site(self, atom_name: str) -> bool:
        """Return whether the residue contains the requested atom site."""

        normalized_atom_name = atom_name.strip().upper()
        return any(
            atom_site.name == normalized_atom_name for atom_site in self.atom_sites
        )

    def with_component_id(self, component_id: str) -> "ResidueSite":
        """Return a copy with an updated residue component identifier."""

        return type(self)(
            component_id=component_id,
            residue_id=self.residue_id,
            atom_sites=self.atom_sites,
            is_hetero=self.is_hetero,
        )

    def with_residue_id(self, residue_id: ResidueId) -> "ResidueSite":
        """Return a copy with an updated residue identifier."""

        return type(self)(
            component_id=self.component_id,
            residue_id=residue_id,
            atom_sites=self.atom_sites,
            is_hetero=self.is_hetero,
        )

    def with_atom_site(self, atom_site: AtomSite) -> "ResidueSite":
        """Return a copy with one atom site added or replaced by name."""

        updated_atom_sites = list(self.atom_sites)
        replaced_existing_site = False
        for index, current_atom_site in enumerate(updated_atom_sites):
            if current_atom_site.name == atom_site.name:
                updated_atom_sites[index] = atom_site
                replaced_existing_site = True
                break

        if not replaced_existing_site:
            updated_atom_sites.append(atom_site)

        return type(self)(
            component_id=self.component_id,
            residue_id=self.residue_id,
            atom_sites=tuple(updated_atom_sites),
            is_hetero=self.is_hetero,
        )

    def with_atom_sites(self, atom_sites: Iterable[AtomSite]) -> "ResidueSite":
        """Return a copy with multiple atom sites added or replaced by name."""

        residue = self
        for atom_site in atom_sites:
            residue = residue.with_atom_site(atom_site)

        return residue

    def without_atom_sites(
        self,
        atom_names: Collection[str],
    ) -> "ResidueSite":
        """Return a copy without the requested atom-site names."""

        names_to_remove = {atom_name.strip().upper() for atom_name in atom_names}
        return type(self)(
            component_id=self.component_id,
            residue_id=self.residue_id,
            atom_sites=tuple(
                atom_site
                for atom_site in self.atom_sites
                if atom_site.name not in names_to_remove
            ),
            is_hetero=self.is_hetero,
        )

    def reordered_atom_sites(
        self,
        ordered_atom_names: Sequence[str],
    ) -> "ResidueSite":
        """Return a copy with atom sites projected into a preferred order."""

        normalized_order = tuple(
            atom_name.strip().upper() for atom_name in ordered_atom_names
        )
        if not normalized_order:
            return self

        atom_order = {
            atom_name: index for index, atom_name in enumerate(normalized_order)
        }
        reordered_atom_sites = tuple(
            sorted(
                self.atom_sites,
                key=lambda atom_site: atom_order.get(
                    atom_site.name,
                    len(atom_order),
                ),
            )
        )
        if reordered_atom_sites == self.atom_sites:
            return self

        return type(self)(
            component_id=self.component_id,
            residue_id=self.residue_id,
            atom_sites=reordered_atom_sites,
            is_hetero=self.is_hetero,
        )


@dataclass(frozen=True, slots=True)
class ChainSite:
    """One polymer chain scaffold."""

    chain_id: str
    residues: tuple[ResidueSite, ...]

    def __post_init__(self) -> None:
        chain_id = self.chain_id.strip()
        residues = tuple(self.residues)

        if not chain_id:
            raise ValueError("chain_id must not be blank")

        residue_ids = tuple(residue.residue_id for residue in residues)
        if len(residue_ids) != len(set(residue_ids)):
            raise ModelInvariantError(
                f"chain {chain_id} contains duplicate residue ids"
            )

        for residue in residues:
            if residue.residue_id.chain_id != chain_id:
                raise ModelInvariantError(
                    f"residue {residue.residue_id.display_token()} does not belong "
                    f"to chain {chain_id}"
                )

        object.__setattr__(self, "chain_id", chain_id)
        object.__setattr__(self, "residues", residues)

    def residue_ids(self) -> tuple[ResidueId, ...]:
        """Return materialized residue identifiers in chain order."""

        return tuple(residue.residue_id for residue in self.residues)

    def has_residue(self, residue_id: ResidueId) -> bool:
        """Return whether the chain contains the requested residue id."""

        return residue_id in self.residue_ids()

    def residue(self, residue_id: ResidueId) -> ResidueSite:
        """Return one residue constitution by id or raise if it is absent."""

        for residue in self.residues:
            if residue.residue_id == residue_id:
                return residue

        raise ResidueNotFoundError(
            f"chain {self.chain_id} has no residue {residue_id.display_token()}"
        )

    def with_residues(self, residues: Iterable[ResidueSite]) -> "ChainSite":
        """Return a copy with an explicit full residue replacement."""

        return type(self)(
            chain_id=self.chain_id,
            residues=tuple(residues),
        )

    def with_updated_residue(self, residue: ResidueSite) -> "ChainSite":
        """Return a copy with one residue replaced by identifier."""

        updated_residues = list(self.residues)
        for index, current_residue in enumerate(updated_residues):
            if current_residue.residue_id == residue.residue_id:
                updated_residues[index] = residue
                return self.with_residues(updated_residues)

        raise ResidueNotFoundError(
            f"chain {self.chain_id} has no residue {residue.residue_id.display_token()}"
        )

    def with_inserted_residues(
        self,
        residues: Iterable[ResidueSite],
    ) -> "ChainSite":
        """Return a copy with newly inserted residues merged in chain order."""

        inserted_residues = tuple(residues)
        if not inserted_residues:
            return self

        existing_residue_ids = set(self.residue_ids())
        inserted_residue_ids: list[ResidueId] = []
        for residue in inserted_residues:
            if residue.residue_id.chain_id != self.chain_id:
                raise ModelInvariantError(
                    f"residue {residue.residue_id.display_token()} does not belong "
                    f"to chain {self.chain_id}"
                )
            if residue.residue_id in existing_residue_ids:
                raise ModelInvariantError(
                    f"chain {self.chain_id} already contains residue "
                    f"{residue.residue_id.display_token()}"
                )
            if residue.residue_id in inserted_residue_ids:
                raise ModelInvariantError(
                    f"chain {self.chain_id} received duplicate inserted residue "
                    f"{residue.residue_id.display_token()}"
                )

            inserted_residue_ids.append(residue.residue_id)

        return self.with_residues(
            sorted(
                (*self.residues, *inserted_residues),
                key=lambda residue: residue.residue_id,
            )
        )


@dataclass(frozen=True, slots=True)
class StructureConstitution:
    """One structure-level constitution over chains and ligands."""

    chains: tuple[ChainSite, ...]
    ligands: tuple[ResidueSite, ...]
    residue_slots: tuple[ResidueSite, ...] = field(init=False, repr=False)
    atom_slots: tuple[AtomSite, ...] = field(init=False, repr=False)
    address_space_key: StructureAddressSpaceKey = field(init=False, repr=False)
    _chain_index_by_id: dict[str, ChainIndex] = field(init=False, repr=False)
    _residue_index_by_id: dict[ResidueId, ResidueIndex] = field(
        init=False,
        repr=False,
    )
    _residue_indices_by_chain_index: dict[ChainIndex, tuple[ResidueIndex, ...]] = field(
        init=False,
        repr=False,
    )
    _residue_index_by_atom_index: dict[AtomIndex, ResidueIndex] = field(
        init=False,
        repr=False,
    )
    _atom_indices_by_residue_index: dict[ResidueIndex, tuple[AtomIndex, ...]] = field(
        init=False,
        repr=False,
    )
    _atom_index_by_residue_atom_name: dict[tuple[ResidueIndex, str], AtomIndex] = field(
        init=False,
        repr=False,
    )

    def __post_init__(self) -> None:
        chains = tuple(self.chains)
        ligands = tuple(self.ligands)
        chain_ids = tuple(chain.chain_id for chain in chains)

        if len(chain_ids) != len(set(chain_ids)):
            raise ModelInvariantError(
                "structure constitution contains duplicate chain ids"
            )

        residue_slots = (
            tuple(residue for chain in chains for residue in chain.residues) + ligands
        )
        residue_ids = tuple(residue.residue_id for residue in residue_slots)
        if len(residue_ids) != len(set(residue_ids)):
            raise ModelInvariantError(
                "structure constitution contains duplicate residue ids"
            )

        atom_slot_keys = tuple(
            (residue.residue_id, atom_site.name)
            for residue in residue_slots
            for atom_site in residue.atom_sites
        )
        if len(atom_slot_keys) != len(set(atom_slot_keys)):
            raise ModelInvariantError(
                "structure constitution contains duplicate atom references"
            )

        atom_slots: list[AtomSite] = []
        atom_indices_by_residue_index: dict[ResidueIndex, tuple[AtomIndex, ...]] = {}
        atom_index_by_residue_atom_name: dict[tuple[ResidueIndex, str], AtomIndex] = {}
        residue_indices_by_chain_index: dict[ChainIndex, tuple[ResidueIndex, ...]] = {}
        residue_cursor = 0
        for chain_index_value, chain in enumerate(chains):
            chain_index = ChainIndex(chain_index_value)
            residue_indices_by_chain_index[chain_index] = tuple(
                ResidueIndex(residue_cursor + residue_offset)
                for residue_offset in range(len(chain.residues))
            )
            residue_cursor += len(chain.residues)
        for residue_index_value, residue in enumerate(residue_slots):
            residue_index = ResidueIndex(residue_index_value)
            residue_atom_indices: list[AtomIndex] = []
            for atom_site in residue.atom_sites:
                atom_index = AtomIndex(len(atom_slots))
                atom_slots.append(atom_site)
                residue_atom_indices.append(atom_index)
                atom_index_by_residue_atom_name[(residue_index, atom_site.name)] = (
                    atom_index
                )

            atom_indices_by_residue_index[residue_index] = tuple(residue_atom_indices)

        object.__setattr__(self, "chains", chains)
        object.__setattr__(self, "ligands", ligands)
        object.__setattr__(self, "residue_slots", residue_slots)
        object.__setattr__(self, "atom_slots", tuple(atom_slots))
        object.__setattr__(
            self,
            "address_space_key",
            tuple(
                (
                    residue.residue_id,
                    residue.atom_site_names(),
                )
                for residue in residue_slots
            ),
        )
        object.__setattr__(
            self,
            "_chain_index_by_id",
            {chain.chain_id: ChainIndex(index) for index, chain in enumerate(chains)},
        )
        object.__setattr__(
            self,
            "_residue_index_by_id",
            {
                residue.residue_id: ResidueIndex(index)
                for index, residue in enumerate(residue_slots)
            },
        )
        object.__setattr__(
            self,
            "_residue_indices_by_chain_index",
            residue_indices_by_chain_index,
        )
        object.__setattr__(
            self,
            "_atom_indices_by_residue_index",
            atom_indices_by_residue_index,
        )
        object.__setattr__(
            self,
            "_residue_index_by_atom_index",
            {
                atom_index: residue_index
                for residue_index, atom_indices in (
                    self._atom_indices_by_residue_index.items()
                )
                for atom_index in atom_indices
            },
        )
        object.__setattr__(
            self,
            "_atom_index_by_residue_atom_name",
            atom_index_by_residue_atom_name,
        )

    def chain_ids(self) -> tuple[str, ...]:
        """Return materialized chain identifiers in structure order."""

        return tuple(chain.chain_id for chain in self.chains)

    def has_chain(self, chain_id: str) -> bool:
        """Return whether the scaffold contains the requested chain."""

        return chain_id.strip() in self.chain_ids()

    def chain(self, chain_id: str) -> ChainSite:
        """Return one chain constitution by id or raise if it is absent."""

        normalized_chain_id = chain_id.strip()
        for chain in self.chains:
            if chain.chain_id == normalized_chain_id:
                return chain

        raise ChainNotFoundError(
            f"structure constitution has no chain {normalized_chain_id}"
        )

    def chain_index(self, chain_id: str) -> ChainIndex:
        """Return the canonical slot index for one chain identifier."""

        normalized_chain_id = chain_id.strip()
        try:
            return self._chain_index_by_id[normalized_chain_id]
        except KeyError as error:
            raise ChainNotFoundError(
                f"structure constitution has no chain {normalized_chain_id}"
            ) from error

    def chain_site_at(self, chain_index: ChainIndex) -> ChainSite:
        """Return the chain site at one canonical chain slot index."""

        return self.chains[chain_index.value]

    def residue_or_ligand(
        self,
        residue_id: ResidueId,
    ) -> ResidueSite | None:
        """Return one residue or ligand constitution by identifier when present."""

        residue_index = self._residue_index_by_id.get(residue_id)
        if residue_index is None:
            return None

        return self.residue_site_at(residue_index)

    def residue_index(self, residue_id: ResidueId) -> ResidueIndex:
        """Return the canonical slot index for one residue identifier."""

        try:
            return self._residue_index_by_id[residue_id]
        except KeyError as error:
            raise ResidueNotFoundError(
                f"structure constitution has no residue {residue_id.display_token()}"
            ) from error

    def residue_site_at(self, residue_index: ResidueIndex) -> ResidueSite:
        """Return the residue site at one canonical residue slot index."""

        return self.residue_slots[residue_index.value]

    def residue_indices_for_chain_index(
        self,
        chain_index: ChainIndex,
    ) -> tuple[ResidueIndex, ...]:
        """Return residue slot indices in structure order for one chain slot."""

        return self._residue_indices_by_chain_index[chain_index]

    def atom_indices_for_residue(
        self,
        residue_id: ResidueId,
    ) -> tuple[AtomIndex, ...]:
        """Return canonical atom slot indices for one residue identifier."""

        return self.atom_indices_for_residue_index(self.residue_index(residue_id))

    def atom_indices_for_residue_index(
        self,
        residue_index: ResidueIndex,
    ) -> tuple[AtomIndex, ...]:
        """Return canonical atom slot indices for one residue slot index."""

        return self._atom_indices_by_residue_index[residue_index]

    def residue_index_for_atom_index(self, atom_index: AtomIndex) -> ResidueIndex:
        """Return the residue slot index that owns one atom slot index."""

        return self._residue_index_by_atom_index[atom_index]

    def atom_index_in_residue(
        self,
        residue_index: ResidueIndex,
        atom_name: str,
    ) -> AtomIndex:
        """Return the atom slot index for one residue-local atom name."""

        normalized_atom_name = atom_name.strip().upper()
        try:
            return self._atom_index_by_residue_atom_name[
                (residue_index, normalized_atom_name)
            ]
        except KeyError as error:
            residue_site = self.residue_site_at(residue_index)
            raise AtomNotFoundError(
                f"residue {residue_site.residue_id.display_token()} has no atom "
                f"{normalized_atom_name}"
            ) from error

    def with_chains(self, chains: Iterable[ChainSite]) -> "StructureConstitution":
        """Return a copy with an explicit chain replacement."""

        return type(self)(
            chains=tuple(chains),
            ligands=self.ligands,
        )

    def with_ligands(
        self,
        ligands: Iterable[ResidueSite],
    ) -> "StructureConstitution":
        """Return a copy with an explicit ligand replacement."""

        return type(self)(
            chains=self.chains,
            ligands=tuple(ligands),
        )

    def with_updated_chain(self, chain: ChainSite) -> "StructureConstitution":
        """Return a copy with one chain replaced by identifier."""

        updated_chains = list(self.chains)
        for index, current_chain in enumerate(updated_chains):
            if current_chain.chain_id == chain.chain_id:
                updated_chains[index] = chain
                return self.with_chains(updated_chains)

        raise ChainNotFoundError(
            f"structure constitution has no chain {chain.chain_id}"
        )

    def with_inserted_chain_residues(
        self,
        chain_id: str,
        residues: Iterable[ResidueSite],
    ) -> "StructureConstitution":
        """Return a copy with inserted residues merged into one chain."""

        return self.with_updated_chain(
            self.chain(chain_id).with_inserted_residues(residues)
        )

    def resolve_atom_site(self, atom_ref: AtomRef) -> AtomSite | None:
        """Return one materialized atom site by canonical atom reference."""

        atom_index = self.resolve_atom_index(atom_ref)
        if atom_index is None:
            return None

        return self.atom_site_at(atom_index)

    def resolve_atom_index(self, atom_ref: AtomRef) -> AtomIndex | None:
        """Return one atom slot index by canonical atom reference when present."""

        residue_index = self._residue_index_by_id.get(atom_ref.residue_id)
        if residue_index is None:
            return None

        return self._atom_index_by_residue_atom_name.get(
            (residue_index, atom_ref.atom_name.strip().upper())
        )

    def atom_index(self, atom_ref: AtomRef) -> AtomIndex:
        """Return the canonical slot index for one atom reference."""

        residue_index = self.residue_index(atom_ref.residue_id)
        return self.atom_index_in_residue(residue_index, atom_ref.atom_name)

    def atom_ref_at(self, atom_index: AtomIndex) -> AtomRef:
        """Return the canonical atom reference at one atom slot index."""

        residue_index = self.residue_index_for_atom_index(atom_index)
        residue_site = self.residue_site_at(residue_index)
        atom_site = self.atom_site_at(atom_index)
        return AtomRef(residue_id=residue_site.residue_id, atom_name=atom_site.name)

    def atom_site_at(self, atom_index: AtomIndex) -> AtomSite:
        """Return the atom site at one canonical atom slot index."""

        return self.atom_slots[atom_index.value]

    def iter_residues(
        self,
        *,
        include_ligands: bool = False,
    ) -> Iterator[ResidueSite]:
        """Iterate over residue constitutions in structure order."""

        for chain in self.chains:
            yield from chain.residues

        if include_ligands:
            yield from self.ligands

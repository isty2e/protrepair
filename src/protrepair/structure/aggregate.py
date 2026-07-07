"""Canonical structure aggregate over constitution, geometry, topology,
polymer blueprint, and provenance."""

from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass
from typing import TypeVar

from typing_extensions import Self

from protrepair.errors import ModelInvariantError, ResidueNotFoundError
from protrepair.structure.constitution import (
    AtomSite,
    ChainSite,
    ResidueSite,
    StructureConstitution,
)
from protrepair.structure.geometry import ResidueGeometry, StructureGeometry
from protrepair.structure.labels import ResidueId
from protrepair.structure.polymer_blueprint import PolymerBlueprint
from protrepair.structure.provenance import StructureProvenance
from protrepair.structure.slots import ResidueIndex
from protrepair.structure.topology import (
    AtomTopology,
    StructureTopology,
    TopologyBond,
)

StructureT = TypeVar("StructureT", bound="ProteinStructure")


@dataclass(frozen=True, slots=True)
class ResidueFacetPayload:
    """Slot-local residue payload across constitution, geometry, and topology."""

    residue_site: ResidueSite
    residue_geometry: ResidueGeometry
    formal_charge_by_atom_name: tuple[tuple[str, int | None], ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "formal_charge_by_atom_name",
            tuple(self.formal_charge_by_atom_name),
        )

    @property
    def residue_id(self) -> ResidueId:
        """Return the residue identifier owned by this facet payload."""

        return self.residue_site.residue_id

    @property
    def is_hetero(self) -> bool:
        """Return whether this payload describes a hetero residue or ligand."""

        return self.residue_site.is_hetero

    def with_residue_site(self, residue_site: ResidueSite) -> "ResidueFacetPayload":
        """Return a copy with an updated constitution facet."""

        return type(self)(
            residue_site=residue_site,
            residue_geometry=self.residue_geometry,
            formal_charge_by_atom_name=self.formal_charge_by_atom_name,
        )

    def without_atoms(self, atom_names: Collection[str]) -> "ResidueFacetPayload":
        """Return a copy with matching atom names removed from all facets."""

        names_to_remove = {atom_name.strip().upper() for atom_name in atom_names}
        return type(self)(
            residue_site=self.residue_site.without_atom_sites(names_to_remove),
            residue_geometry=self.residue_geometry.without_atoms(names_to_remove),
            formal_charge_by_atom_name=tuple(
                (atom_name, formal_charge)
                for atom_name, formal_charge in self.formal_charge_by_atom_name
                if atom_name not in names_to_remove
            ),
        )


def _residue_entries_by_index(
    structure: "ProteinStructure",
) -> tuple[ResidueFacetPayload, ...]:
    """Return slot-aligned residue payloads for one canonical structure."""

    return tuple(
        ResidueFacetPayload(
            residue_site=residue_site,
            residue_geometry=structure.residue_geometry(ResidueIndex(residue_index)),
            formal_charge_by_atom_name=(
                structure.residue_formal_charge_by_atom_name(
                    ResidueIndex(residue_index)
                )
            ),
        )
        for residue_index, residue_site in enumerate(
            structure.constitution.residue_slots
        )
    )


def _require_complete_residue_entries(
    *,
    constitution: StructureConstitution,
    residue_entries_by_index: Sequence[ResidueFacetPayload | None],
) -> tuple[ResidueFacetPayload, ...]:
    """Return slot-aligned residue payloads or raise when any slot is missing."""

    missing_residue_tokens = tuple(
        constitution.residue_site_at(
            ResidueIndex(residue_index)
        ).residue_id.display_token()
        for residue_index, residue_entry in enumerate(residue_entries_by_index)
        if residue_entry is None
    )
    if missing_residue_tokens:
        raise ModelInvariantError(
            "structure residue payload rebuild is missing slots for "
            f"{', '.join(missing_residue_tokens)}"
        )

    return tuple(
        residue_entry
        for residue_entry in residue_entries_by_index
        if residue_entry is not None
    )


def _structure_from_residue_entries(
    *,
    structure_type: type[StructureT],
    constitution: StructureConstitution,
    residue_entries_by_index: Sequence[ResidueFacetPayload | None],
    polymer_blueprint: PolymerBlueprint | None,
    provenance: StructureProvenance,
    source_bonds: Sequence[TopologyBond] = (),
) -> StructureT:
    """Build one canonical structure from slot-aligned residue payloads."""

    complete_residue_entries = _require_complete_residue_entries(
        constitution=constitution,
        residue_entries_by_index=residue_entries_by_index,
    )
    return structure_type.from_payload(
        constitution=constitution,
        geometry=_geometry_from_residue_entries(
            constitution=constitution,
            residue_entries_by_index=complete_residue_entries,
        ),
        topology=_topology_from_residue_entries(
            constitution=constitution,
            residue_entries_by_index=complete_residue_entries,
            source_bonds=source_bonds,
        ),
        polymer_blueprint=polymer_blueprint,
        provenance=provenance,
    )


def _geometry_from_residue_entries(
    *,
    constitution: StructureConstitution,
    residue_entries_by_index: Sequence[ResidueFacetPayload],
) -> StructureGeometry:
    """Build slot-aligned geometry from residue-local payloads."""

    return StructureGeometry(
        constitution=constitution,
        atom_geometries=tuple(
            residue_entry.residue_geometry.atom_geometry(atom_site.name)
            for residue_entry, residue_slot in zip(
                residue_entries_by_index,
                constitution.residue_slots,
                strict=True,
            )
            for atom_site in residue_slot.atom_sites
        ),
    )


def _topology_from_residue_entries(
    *,
    constitution: StructureConstitution,
    residue_entries_by_index: Sequence[ResidueFacetPayload],
    source_bonds: Sequence[TopologyBond] = (),
) -> StructureTopology:
    """Build slot-aligned topology from residue-local formal-charge payloads."""

    return StructureTopology(
        constitution=constitution,
        atom_topologies=tuple(
            (
                AtomTopology(formal_charge=formal_charge)
                if formal_charge is not None
                else None
            )
            for residue_entry, residue_slot in zip(
                residue_entries_by_index,
                constitution.residue_slots,
                strict=True,
            )
            for atom_site in residue_slot.atom_sites
            for formal_charge in (
                dict(residue_entry.formal_charge_by_atom_name).get(atom_site.name),
            )
        ),
        bonds=tuple(source_bonds),
    )


@dataclass(frozen=True, slots=True, init=False)
class ProteinStructure:
    """Canonical aggregate root over structure facets."""

    constitution: StructureConstitution
    geometry: StructureGeometry
    topology: StructureTopology
    polymer_blueprint: PolymerBlueprint | None
    provenance: StructureProvenance

    @classmethod
    def from_payload(
        cls: type[Self],
        *,
        constitution: StructureConstitution,
        geometry: StructureGeometry,
        topology: StructureTopology,
        polymer_blueprint: PolymerBlueprint | None = None,
        provenance: StructureProvenance,
    ) -> Self:
        """Build one canonical structure directly from facet payload."""

        if geometry.atom_count() != len(constitution.atom_slots):
            raise ModelInvariantError(
                "structure geometry must cover exactly the constitution atom sites"
            )

        if not geometry.is_aligned_to(constitution):
            raise ModelInvariantError(
                "structure geometry must share the constitution address space"
            )

        if topology.atom_count() != len(constitution.atom_slots):
            raise ModelInvariantError(
                "structure topology must cover exactly the constitution atom sites"
            )

        if not topology.is_aligned_to(constitution):
            raise ModelInvariantError(
                "structure topology must share the constitution address space"
            )

        if not isinstance(provenance, StructureProvenance):
            raise TypeError("structure provenance must be a StructureProvenance value")
        if polymer_blueprint is not None and not isinstance(
            polymer_blueprint, PolymerBlueprint
        ):
            raise TypeError(
                "structure polymer_blueprint must be a PolymerBlueprint or None"
            )
        if polymer_blueprint is not None:
            constitution_chain_ids = constitution.chain_ids()
            if polymer_blueprint.chain_ids() != constitution_chain_ids:
                raise ModelInvariantError(
                    "structure-attached polymer blueprint must share the "
                    "constitution polymer chain ids in order"
                )

        structure = object.__new__(cls)
        object.__setattr__(structure, "constitution", constitution)
        object.__setattr__(structure, "geometry", geometry)
        object.__setattr__(structure, "topology", topology)
        object.__setattr__(structure, "polymer_blueprint", polymer_blueprint)
        object.__setattr__(structure, "provenance", provenance)
        return structure

    def chain_ids(self) -> tuple[str, ...]:
        """Return chain identifiers in structure order."""

        return self.constitution.chain_ids()

    def has_chain(self, chain_id: str) -> bool:
        """Return whether the structure contains a chain."""

        return self.constitution.has_chain(chain_id)

    def chain_site(self, chain_id: str) -> ChainSite:
        """Return a chain constitution by identifier or raise if absent."""

        return self.constitution.chain(chain_id)

    def iter_residue_sites(
        self,
        include_ligands: bool = False,
    ) -> Iterator[ResidueSite]:
        """Iterate over residue constitutions in structure order."""

        yield from self.constitution.iter_residues(include_ligands=include_ligands)

    def iter_atom_sites(self, include_ligands: bool = False) -> Iterator[AtomSite]:
        """Iterate over constitution-native atom sites in structure order."""

        for residue in self.constitution.iter_residues(include_ligands=include_ligands):
            yield from residue.atom_sites

    def residue_geometry(self, residue_index: ResidueIndex) -> ResidueGeometry:
        """Return residue-local geometry for one residue slot index."""

        residue_site = self.constitution.residue_site_at(residue_index)
        return ResidueGeometry(
            atoms_by_name={
                atom_site.name: self.geometry.atom_geometry(atom_index)
                for atom_site, atom_index in zip(
                    residue_site.atom_sites,
                    self.constitution.atom_indices_for_residue_index(residue_index),
                    strict=True,
                )
            },
        )

    def residue_formal_charge_by_atom_name(
        self,
        residue_index: ResidueIndex,
    ) -> tuple[tuple[str, int | None], ...]:
        """Return residue-local formal-charge payload keyed by atom name."""

        return tuple(
            (
                self.constitution.atom_site_at(atom_index).name,
                atom_topology.formal_charge,
            )
            for atom_index in self.constitution.atom_indices_for_residue_index(
                residue_index
            )
            for atom_topology in (self.topology.atom_topology(atom_index),)
            if atom_topology is not None
        )

    def select_chains(self, chain_ids: Collection[str]) -> Self:
        """Return a copy containing only selected chains in the given order."""

        normalized_chain_ids = tuple(chain_id.strip() for chain_id in chain_ids)
        selected_chain_sites = tuple(
            self.constitution.chain(chain_id) for chain_id in normalized_chain_ids
        )
        selected_chain_id_set = frozenset(normalized_chain_ids)
        selected_ligands = tuple(
            ligand_site
            for ligand_site in self.constitution.ligands
            if ligand_site.residue_id.chain_id in selected_chain_id_set
        )
        selected_structure_constitution = self.constitution.with_chains(
            selected_chain_sites
        ).with_ligands(selected_ligands)
        selected_entries = tuple(
            residue_entry
            for residue_entry in _residue_entries_by_index(self)
            if residue_entry.residue_id.chain_id in selected_chain_id_set
        )
        return _structure_from_residue_entries(
            structure_type=type(self),
            constitution=selected_structure_constitution,
            residue_entries_by_index=selected_entries,
            polymer_blueprint=(
                None
                if self.polymer_blueprint is None
                else self.polymer_blueprint.select_chains(normalized_chain_ids)
            ),
            provenance=self.provenance,
            source_bonds=self.topology.bonds_for_constitution(
                source_constitution=self.constitution,
                target_constitution=selected_structure_constitution,
            ),
        )

    def with_ligand_facets(
        self,
        *,
        ligand_sites: Sequence[ResidueSite],
        ligand_geometries: Sequence[ResidueGeometry],
        ligand_formal_charge_payloads: Sequence[tuple[tuple[str, int | None], ...]]
        | None = None,
    ) -> Self:
        """Return a copy with an explicit ligand facet replacement."""

        normalized_ligand_sites = tuple(ligand_sites)
        normalized_ligand_geometries = tuple(ligand_geometries)
        if ligand_formal_charge_payloads is None:
            normalized_ligand_formal_charge_payloads = tuple(
                () for _ in normalized_ligand_sites
            )
        else:
            normalized_ligand_formal_charge_payloads = tuple(
                tuple(formal_charge_payload)
                for formal_charge_payload in ligand_formal_charge_payloads
            )
        if len(normalized_ligand_sites) != len(normalized_ligand_geometries):
            raise ModelInvariantError(
                "ligand facet replacement requires one geometry per ligand site"
            )
        if len(normalized_ligand_sites) != len(
            normalized_ligand_formal_charge_payloads
        ):
            raise ModelInvariantError(
                "ligand facet replacement requires one topology payload per ligand site"
            )

        updated_constitution = self.constitution.with_ligands(normalized_ligand_sites)
        polymer_entries = tuple(
            residue_entry
            for residue_entry in _residue_entries_by_index(self)
            if not residue_entry.is_hetero
        )
        updated_entries_by_index = (
            *polymer_entries,
            *(
                ResidueFacetPayload(
                    residue_site=ligand_site,
                    residue_geometry=ligand_geometry,
                    formal_charge_by_atom_name=formal_charge_payload,
                )
                for (
                    ligand_site,
                    ligand_geometry,
                    formal_charge_payload,
                ) in zip(
                    normalized_ligand_sites,
                    normalized_ligand_geometries,
                    normalized_ligand_formal_charge_payloads,
                    strict=True,
                )
            ),
        )

        return _structure_from_residue_entries(
            structure_type=type(self),
            constitution=updated_constitution,
            residue_entries_by_index=updated_entries_by_index,
            polymer_blueprint=self.polymer_blueprint,
            provenance=self.provenance,
            source_bonds=self.topology.bonds_for_constitution(
                source_constitution=self.constitution,
                target_constitution=updated_constitution,
            ),
        )

    def with_ligand_facets_from(self, ligand_source: "ProteinStructure") -> Self:
        """Return a copy with ligand facets copied from another structure."""

        if self.constitution.ligands == ligand_source.constitution.ligands:
            return self

        return self.with_ligand_facets(
            ligand_sites=ligand_source.constitution.ligands,
            ligand_geometries=tuple(
                ligand_source.residue_geometry(
                    ligand_source.constitution.residue_index(ligand.residue_id)
                )
                for ligand in ligand_source.constitution.ligands
            ),
            ligand_formal_charge_payloads=tuple(
                ligand_source.residue_formal_charge_by_atom_name(
                    ligand_source.constitution.residue_index(ligand.residue_id)
                )
                for ligand in ligand_source.constitution.ligands
            ),
        )

    def with_provenance(self, provenance: StructureProvenance) -> Self:
        """Return a copy with a replacement provenance facet."""

        return type(self).from_payload(
            constitution=self.constitution,
            geometry=self.geometry,
            topology=self.topology,
            polymer_blueprint=self.polymer_blueprint,
            provenance=provenance,
        )

    def with_inserted_chain_residue_facets(
        self,
        chain_id: str,
        *,
        residue_sites: Sequence[ResidueSite],
        residue_geometries: Sequence[ResidueGeometry],
        residue_formal_charge_payloads: Sequence[tuple[tuple[str, int | None], ...]]
        | None = None,
    ) -> Self:
        """Return a copy with newly inserted residue facets merged into one chain."""

        normalized_residue_sites = tuple(residue_sites)
        normalized_residue_geometries = tuple(residue_geometries)
        if residue_formal_charge_payloads is None:
            normalized_residue_formal_charge_payloads = tuple(
                () for _ in normalized_residue_sites
            )
        else:
            normalized_residue_formal_charge_payloads = tuple(
                tuple(formal_charge_payload)
                for formal_charge_payload in residue_formal_charge_payloads
            )
        if not normalized_residue_sites:
            return self
        if len(normalized_residue_sites) != len(normalized_residue_geometries):
            raise ModelInvariantError(
                "inserted-chain residue update requires one geometry per residue site"
            )
        if len(normalized_residue_sites) != len(
            normalized_residue_formal_charge_payloads
        ):
            raise ModelInvariantError(
                "inserted-chain residue update requires one topology payload "
                "per residue site"
            )

        updated_constitution = self.constitution.with_inserted_chain_residues(
            chain_id,
            normalized_residue_sites,
        )
        inserted_entries_by_residue_id = {
            residue_site.residue_id: ResidueFacetPayload(
                residue_site=residue_site,
                residue_geometry=residue_geometry,
                formal_charge_by_atom_name=formal_charge_payload,
            )
            for residue_site, residue_geometry, formal_charge_payload in zip(
                normalized_residue_sites,
                normalized_residue_geometries,
                normalized_residue_formal_charge_payloads,
                strict=True,
            )
        }
        carried_entries_by_chain_id: dict[str, list[ResidueFacetPayload]] = {
            chain_site.chain_id: [] for chain_site in self.constitution.chains
        }
        carried_ligand_entries: list[ResidueFacetPayload] = []
        for residue_entry in _residue_entries_by_index(self):
            if residue_entry.is_hetero:
                carried_ligand_entries.append(residue_entry)
                continue

            carried_entries_by_chain_id[residue_entry.residue_id.chain_id].append(
                residue_entry
            )

        updated_entries_by_index: list[ResidueFacetPayload] = []
        for updated_chain_site in updated_constitution.chains:
            if updated_chain_site.chain_id != chain_id:
                updated_entries_by_index.extend(
                    carried_entries_by_chain_id[updated_chain_site.chain_id]
                )
                continue

            carried_chain_entries = iter(
                carried_entries_by_chain_id[updated_chain_site.chain_id]
            )
            for residue_site in updated_chain_site.residues:
                inserted_entry = inserted_entries_by_residue_id.get(
                    residue_site.residue_id
                )
                if inserted_entry is not None:
                    updated_entries_by_index.append(inserted_entry)
                    continue

                carried_entry = next(carried_chain_entries)
                if carried_entry.residue_id != residue_site.residue_id:
                    raise ModelInvariantError(
                        "inserted-chain residue payload rebuild lost canonical "
                        f"slot alignment at {residue_site.residue_id.display_token()}"
                    )
                updated_entries_by_index.append(
                    carried_entry.with_residue_site(residue_site)
                )

        updated_entries_by_index.extend(carried_ligand_entries)

        return _structure_from_residue_entries(
            structure_type=type(self),
            constitution=updated_constitution,
            residue_entries_by_index=updated_entries_by_index,
            polymer_blueprint=self.polymer_blueprint,
            provenance=self.provenance,
            source_bonds=self.topology.bonds_for_constitution(
                source_constitution=self.constitution,
                target_constitution=updated_constitution,
            ),
        )

    def with_updated_residue_facets(
        self,
        residue_site: ResidueSite,
        *,
        residue_geometry: ResidueGeometry,
        formal_charge_by_atom_name: Iterable[tuple[str, int | None]] = (),
    ) -> Self:
        """Return a copy with one residue facet tuple replaced by identifier."""

        if self.constitution.residue_or_ligand(residue_site.residue_id) is None:
            raise ResidueNotFoundError(
                f"structure has no residue {residue_site.residue_id.display_token()}"
            )

        current_residue_site = self.constitution.residue_or_ligand(
            residue_site.residue_id
        )
        assert current_residue_site is not None

        if not current_residue_site.is_hetero:
            updated_constitution = self.constitution.with_updated_chain(
                self.constitution.chain(
                    residue_site.residue_id.chain_id
                ).with_updated_residue(residue_site)
            )
        else:
            updated_constitution = self.constitution.with_ligands(
                tuple(
                    residue_site
                    if ligand.residue_id == residue_site.residue_id
                    else ligand
                    for ligand in self.constitution.ligands
                )
            )

        updated_entries_by_index = list(_residue_entries_by_index(self))
        target_residue_index = self.constitution.residue_index(residue_site.residue_id)
        updated_entries_by_index[target_residue_index.value] = ResidueFacetPayload(
            residue_site=residue_site,
            residue_geometry=residue_geometry,
            formal_charge_by_atom_name=tuple(formal_charge_by_atom_name),
        )

        return _structure_from_residue_entries(
            structure_type=type(self),
            constitution=updated_constitution,
            residue_entries_by_index=updated_entries_by_index,
            polymer_blueprint=self.polymer_blueprint,
            provenance=self.provenance,
            source_bonds=self.topology.bonds_for_constitution(
                source_constitution=self.constitution,
                target_constitution=updated_constitution,
            ),
        )

    def with_updated_residue_facets_batch(
        self,
        residue_facets: Iterable[
            tuple[
                ResidueSite,
                ResidueGeometry,
                Iterable[tuple[str, int | None]],
            ]
        ],
    ) -> Self:
        """Return a copy with multiple residue facet tuples replaced by identifier."""

        normalized_residue_entries_by_id: dict[ResidueId, ResidueFacetPayload] = {}
        residue_indices_by_id: dict[ResidueId, ResidueIndex] = {}
        polymer_residue_sites_by_id: dict[ResidueId, ResidueSite] = {}
        ligand_sites_by_id: dict[ResidueId, ResidueSite] = {}
        for (
            residue_site,
            residue_geometry,
            formal_charge_by_atom_name,
        ) in residue_facets:
            residue_id = residue_site.residue_id
            if residue_id in normalized_residue_entries_by_id:
                raise ModelInvariantError(
                    "batch residue facet update received duplicate residue "
                    f"{residue_id.display_token()}"
                )

            residue_index = self.constitution.residue_index(residue_id)
            current_residue_site = self.constitution.residue_site_at(residue_index)
            normalized_residue_entries_by_id[residue_id] = ResidueFacetPayload(
                residue_site=residue_site,
                residue_geometry=residue_geometry,
                formal_charge_by_atom_name=tuple(formal_charge_by_atom_name),
            )
            residue_indices_by_id[residue_id] = residue_index
            if current_residue_site.is_hetero:
                ligand_sites_by_id[residue_id] = residue_site
            else:
                polymer_residue_sites_by_id[residue_id] = residue_site

        if not normalized_residue_entries_by_id:
            return self

        updated_chain_sites: list[ChainSite] = []
        for chain_site in self.constitution.chains:
            chain_changed = any(
                residue_site.residue_id in polymer_residue_sites_by_id
                for residue_site in chain_site.residues
            )
            updated_chain_sites.append(
                chain_site.with_residues(
                    polymer_residue_sites_by_id.get(
                        residue_site.residue_id,
                        residue_site,
                    )
                    for residue_site in chain_site.residues
                )
                if chain_changed
                else chain_site
            )

        updated_ligand_sites = tuple(
            ligand_sites_by_id.get(ligand_site.residue_id, ligand_site)
            for ligand_site in self.constitution.ligands
        )
        updated_constitution = self.constitution.with_chains(
            updated_chain_sites
        ).with_ligands(updated_ligand_sites)

        updated_entries_by_index = list(_residue_entries_by_index(self))
        for residue_id, residue_entry in normalized_residue_entries_by_id.items():
            target_residue_index = residue_indices_by_id[residue_id]
            updated_entries_by_index[target_residue_index.value] = residue_entry

        return _structure_from_residue_entries(
            structure_type=type(self),
            constitution=updated_constitution,
            residue_entries_by_index=updated_entries_by_index,
            polymer_blueprint=self.polymer_blueprint,
            provenance=self.provenance,
            source_bonds=self.topology.bonds_for_constitution(
                source_constitution=self.constitution,
                target_constitution=updated_constitution,
            ),
        )

    def with_updated_residue_geometries(
        self,
        residue_geometries: Iterable[tuple[ResidueId, ResidueGeometry]],
    ) -> Self:
        """Return a copy with coordinate-only residue geometry replacements."""

        residue_geometries_by_index: dict[ResidueIndex, ResidueGeometry] = {}
        seen_residue_ids: set[ResidueId] = set()
        for residue_id, residue_geometry in residue_geometries:
            if residue_id in seen_residue_ids:
                raise ModelInvariantError(
                    "residue geometry update received duplicate residue "
                    f"{residue_id.display_token()}"
                )

            seen_residue_ids.add(residue_id)
            residue_geometries_by_index[self.constitution.residue_index(residue_id)] = (
                residue_geometry
            )

        if not residue_geometries_by_index:
            return self

        return type(self).from_payload(
            constitution=self.constitution,
            geometry=self.geometry.with_residue_geometries(
                constitution=self.constitution,
                residue_geometries_by_index=residue_geometries_by_index,
            ),
            topology=self.topology,
            polymer_blueprint=self.polymer_blueprint,
            provenance=self.provenance,
        )

    def without_hydrogens(self) -> Self:
        """Return a copy with polymer hydrogen atoms removed."""

        stripped_chain_sites: list[ChainSite] = []
        structure_changed = False
        for chain_site in self.constitution.chains:
            stripped_residue_sites: list[ResidueSite] = []
            chain_changed = False
            for residue_site in chain_site.residues:
                hydrogen_atom_names = {
                    atom_site.name
                    for atom_site in residue_site.atom_sites
                    if atom_site.element == "H"
                }
                if not hydrogen_atom_names:
                    stripped_residue_sites.append(residue_site)
                    continue

                chain_changed = True
                structure_changed = True
                stripped_residue_sites.append(
                    residue_site.without_atom_sites(hydrogen_atom_names)
                )

            stripped_chain_sites.append(
                chain_site.with_residues(stripped_residue_sites)
                if chain_changed
                else chain_site
            )

        if not structure_changed:
            return self

        stripped_constitution = self.constitution.with_chains(stripped_chain_sites)
        updated_entries_by_index: list[ResidueFacetPayload] = []
        for stripped_residue_site, current_payload in zip(
            stripped_constitution.residue_slots,
            _residue_entries_by_index(self),
            strict=True,
        ):
            if current_payload.is_hetero:
                updated_entries_by_index.append(current_payload)
                continue

            hydrogen_atom_names = {
                atom_site.name
                for atom_site in current_payload.residue_site.atom_sites
                if atom_site.element == "H"
            }
            if not hydrogen_atom_names:
                updated_entries_by_index.append(current_payload)
                continue

            updated_entries_by_index.append(
                current_payload.with_residue_site(stripped_residue_site).without_atoms(
                    hydrogen_atom_names
                )
            )

        return _structure_from_residue_entries(
            structure_type=type(self),
            constitution=stripped_constitution,
            residue_entries_by_index=updated_entries_by_index,
            polymer_blueprint=self.polymer_blueprint,
            provenance=self.provenance,
            source_bonds=self.topology.bonds_for_constitution(
                source_constitution=self.constitution,
                target_constitution=stripped_constitution,
            ),
        )

    def without_hydrogens_in_residues(
        self,
        residue_ids: Collection[ResidueId],
    ) -> Self:
        """Return a copy with hydrogen atoms removed only in selected residues."""

        selected_residue_ids = frozenset(residue_ids)
        if not selected_residue_ids:
            return self

        stripped_chain_sites: list[ChainSite] = []
        stripped_ligands: list[ResidueSite] = []
        structure_changed = False

        for chain_site in self.constitution.chains:
            stripped_residues: list[ResidueSite] = []
            chain_changed = False
            for residue_site in chain_site.residues:
                if residue_site.residue_id not in selected_residue_ids:
                    stripped_residues.append(residue_site)
                    continue

                hydrogen_atom_names = {
                    atom_site.name
                    for atom_site in residue_site.atom_sites
                    if atom_site.element == "H"
                }
                if not hydrogen_atom_names:
                    stripped_residues.append(residue_site)
                    continue

                stripped_residue = residue_site.without_atom_sites(hydrogen_atom_names)
                if stripped_residue != residue_site:
                    chain_changed = True
                    structure_changed = True
                stripped_residues.append(stripped_residue)

            if chain_changed:
                stripped_chain_sites.append(chain_site.with_residues(stripped_residues))
                continue

            stripped_chain_sites.append(chain_site)

        for ligand in self.constitution.ligands:
            if ligand.residue_id not in selected_residue_ids:
                stripped_ligands.append(ligand)
                continue

            hydrogen_atom_names = {
                atom_site.name
                for atom_site in ligand.atom_sites
                if atom_site.element == "H"
            }
            if not hydrogen_atom_names:
                stripped_ligands.append(ligand)
                continue

            stripped_ligand = ligand.without_atom_sites(hydrogen_atom_names)
            if stripped_ligand != ligand:
                structure_changed = True
            stripped_ligands.append(stripped_ligand)

        if not structure_changed:
            return self

        stripped_constitution = self.constitution.with_chains(
            stripped_chain_sites
        ).with_ligands(stripped_ligands)
        updated_entries_by_index: list[ResidueFacetPayload] = []
        for stripped_residue_site, current_payload in zip(
            stripped_constitution.residue_slots,
            _residue_entries_by_index(self),
            strict=True,
        ):
            if stripped_residue_site.residue_id not in selected_residue_ids:
                updated_entries_by_index.append(current_payload)
                continue

            hydrogen_atom_names = {
                atom_site.name
                for atom_site in current_payload.residue_site.atom_sites
                if atom_site.element == "H"
            }
            updated_entries_by_index.append(
                current_payload.with_residue_site(stripped_residue_site).without_atoms(
                    hydrogen_atom_names
                )
            )

        return _structure_from_residue_entries(
            structure_type=type(self),
            constitution=stripped_constitution,
            residue_entries_by_index=updated_entries_by_index,
            polymer_blueprint=self.polymer_blueprint,
            provenance=self.provenance,
            source_bonds=self.topology.bonds_for_constitution(
                source_constitution=self.constitution,
                target_constitution=stripped_constitution,
            ),
        )

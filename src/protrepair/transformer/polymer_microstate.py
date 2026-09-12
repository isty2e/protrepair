"""Atomic application of resolved polymer chemistry with caller-placed H atoms."""

from collections import Counter, defaultdict
from dataclasses import InitVar, dataclass, field, replace
from math import isfinite

from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.graph import MicrostateGraph
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.resolution import (
    AppliedMicrostateOverride,
    MicrostateConstraints,
    MicrostateResolution,
    MicrostateSelectionBasis,
)
from protrepair.structure.aggregate import ProteinStructure, ResidueFacetPayload
from protrepair.structure.constitution import AtomSite
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)


@dataclass(frozen=True, slots=True)
class MicrostateHydrogenPlacement:
    """Name and place one H/D/T attached to a resolved site atom.

    Parameters
    ----------
    atom : AtomSite
        Hydrogen identity, including an explicit isotope label.
    parent_name : str
        Canonical heavy-atom name in the same residue.
    geometry : AtomGeometry
        Placement used only when that identity has no surviving current or
        original geometry. This operation does not reposition existing H atoms.

    Raises
    ------
    ValueError
        The atom is not hydrogen, the parent name is noncanonical, or geometry
        contains nonfinite/invalid atom scalars.
    """

    atom: AtomSite
    parent_name: str
    geometry: AtomGeometry

    def __post_init__(self) -> None:
        if not self.atom.is_hydrogen():
            raise ValueError("microstate placements must be H, D or T")
        if not self.parent_name or self.parent_name != self.parent_name.strip().upper():
            raise ValueError("microstate H parents must be canonical atom names")
        if (
            not all(isfinite(value) for value in self.geometry.position)
            or not isfinite(self.geometry.occupancy)
            or not 0.0 <= self.geometry.occupancy <= 1.0
            or (
                self.geometry.b_factor is not None
                and (not isfinite(self.geometry.b_factor) or self.geometry.b_factor < 0)
            )
        ):
            raise ValueError("microstate H geometry must have valid finite scalars")


@dataclass(frozen=True, slots=True, eq=False)
class PolymerMicrostatePatch:
    """Bind a complete site update to the immutable structure used to resolve it.

    Parameters
    ----------
    source : ProteinStructure
        Exact source snapshot; a batch cannot apply this decision to a new one.
    residue_id : ResidueId
        Polymer residue containing the site.
    site : PolymerMicrostateSite
        Standard chemistry and caller-established linkage, not a chain-position
        guess. Actual current boundary bonds are checked during application.
    hydrogens : tuple[MicrostateHydrogenPlacement, ...]
        All final H identities for the site, not only newly added atoms.
    override : MicrostateConstraints or None
        Explicit replacement authority passed to the chemical resolver.
    applied_override : AppliedMicrostateOverride or None
        Explicit choice retained by a prior transformation, if the caller requests
        reuse. A new override supersedes it. Omission does not inherit authority.
    preferences : tuple[MicrostateConstraints, ...]
        Ordered, source-compatible preparation preferences.

    Raises
    ------
    ResidueNotFoundError
        The residue identity is absent from the source snapshot.
    ValueError
        The residue is not polymer, chemistry is unresolved, placements disagree
        with the graph, or source H identities are unnecessarily replaced.
    """

    source: ProteinStructure = field(repr=False, compare=False)
    residue_id: ResidueId
    site: PolymerMicrostateSite
    hydrogens: tuple[MicrostateHydrogenPlacement, ...]
    override: InitVar[MicrostateConstraints | None] = None
    applied_override: InitVar[AppliedMicrostateOverride | None] = None
    preferences: InitVar[tuple[MicrostateConstraints, ...]] = ()
    resolution: MicrostateResolution = field(init=False)

    def __post_init__(
        self,
        override: MicrostateConstraints | None,
        applied_override: AppliedMicrostateOverride | None,
        preferences: tuple[MicrostateConstraints, ...],
    ) -> None:
        index = self.source.constitution.residue_index(self.residue_id)
        if index.value >= (
            len(self.source.constitution.residue_slots)
            - len(self.source.constitution.ligands)
        ):
            raise ValueError("polymer microstate patches require a polymer residue")
        residue = self.source.constitution.residue_site_at(index)
        resolution = self.site.resolve(
            residue,
            self.source.provenance.ingress.observation,
            override=override,
            applied_override=applied_override,
            preferences=preferences,
        )
        graph = resolution.graph
        if graph is None:
            raise ValueError(
                f"cannot apply {resolution.status.value} polymer chemistry"
            )
        object.__setattr__(self, "resolution", resolution)
        object.__setattr__(self, "hydrogens", tuple(self.hydrogens))

        names = {entry.atom.name for entry in self.hydrogens}
        if len(names) != len(self.hydrogens):
            raise ValueError("microstate H placements require unique atom names")
        counts = Counter(entry.parent_name for entry in self.hydrogens)
        expected = Counter({atom.name: atom.hydrogens for atom in graph.atoms})
        if +counts != +expected:
            raise ValueError(
                "microstate H placements must exactly match the resolved graph"
            )
        if any(
            residue.has_atom_site(name) and not residue.atom_site(name).is_hydrogen()
            for name in names
        ):
            raise ValueError("microstate H names collide with heavy atoms")
        self._validate_observed_hydrogens()

    @property
    def graph(self) -> MicrostateGraph:
        """Return the resolved coupled graph.

        Returns
        -------
        MicrostateGraph
            Validated graph; unresolved outcomes cannot construct a patch.
        """
        graph = self.resolution.graph
        assert graph is not None
        return graph

    def _validate_observed_hydrogens(self) -> None:
        placements = {entry.atom.name: entry for entry in self.hydrogens}
        observed_by_parent: dict[str, list[str]] = defaultdict(list)
        for attachment in self.resolution.observed_hydrogens:
            if (
                attachment.hydrogen.residue_id != self.residue_id
                or attachment.parent.residue_id != self.residue_id
            ):
                raise ValueError("microstate source H belongs to another residue")
            name = attachment.hydrogen.atom_name
            parent = attachment.parent.atom_name
            observed_by_parent[parent].append(name)
            if name in placements and placements[name].parent_name != parent:
                raise ValueError(
                    "microstate patch cannot reparent an original H identity"
                )

        superseded = dict(self.resolution.superseded_source.minimum_hydrogens)
        for parent, names in observed_by_parent.items():
            retained = sum(name in placements for name in names)
            required = min(len(names), self.graph.atom(parent).hydrogens)
            if retained != required:
                raise ValueError(
                    "microstate patch must preserve compatible original H identities"
                )
            if retained < len(names) and (
                self.resolution.basis is not MicrostateSelectionBasis.OVERRIDE
                or superseded.get(parent) != len(names)
            ):
                raise ValueError(
                    "removing original H requires a reported explicit override"
                )

    def _materialize(
        self,
        payload: ResidueFacetPayload,
        current_hydrogens: dict[str, str],
    ) -> ResidueFacetPayload:
        source = self.source
        observation = source.provenance.ingress.observation
        observed = {
            entry.hydrogen.atom_name: entry.parent.atom_name
            for entry in self.resolution.observed_hydrogens
        }
        managed = set(current_hydrogens) | observed.keys()
        final_names = {entry.atom.name for entry in self.hydrogens}
        removed = managed - final_names
        residue = payload.residue_site.without_atom_sites(removed)
        geometry = payload.residue_geometry.without_atoms(removed)
        charges = {
            name: charge
            for name, charge in payload.formal_charge_by_atom_name
            if name not in removed
        }
        charges.update((atom.name, atom.charge) for atom in self.graph.atoms)
        for entry in self.hydrogens:
            name = entry.atom.name
            if residue.has_atom_site(name) and name not in managed:
                raise ValueError("microstate H name collides with an unmanaged atom")
            if (
                name in current_hydrogens
                and current_hydrogens[name] != entry.parent_name
            ):
                raise ValueError(
                    "microstate patch cannot reparent a current H identity"
                )
            atom = entry.atom
            placed = entry.geometry
            if name in observed and observation is not None:
                original_index = observation.constitution.resolve_atom_index(
                    AtomRef(self.residue_id, name)
                )
                assert original_index is not None
                atom = observation.constitution.atom_site_at(original_index)
                placed = observation.geometry.atom_geometry(original_index)
            if residue.has_atom_site(name):
                if name not in observed:
                    atom = residue.atom_site(name)
                placed = geometry.atom_geometry(name)
            residue = residue.with_atom_site(atom)
            geometry = geometry.with_atom_geometry(name, placed)
            charges[name] = 0

        return ResidueFacetPayload(residue, geometry, tuple(charges.items()))


def apply_polymer_microstate_patches(
    structure: ProteinStructure,
    patches: tuple[PolymerMicrostatePatch, ...],
) -> ProteinStructure:
    """Commit disjoint resolved sites without exposing a partly changed graph.

    Parameters
    ----------
    structure : ProteinStructure
        Exact source snapshot shared by every patch.
    patches : tuple[PolymerMicrostatePatch, ...]
        Complete site decisions and H placements; multiple sites may share a residue.

    Returns
    -------
    ProteinStructure
        Updated H constitution, charges and bonds, preserving heavy coordinates,
        surviving H geometry, unrelated facets and original observations. Applied
        explicit overrides replace prior choices for the touched sites; applying
        a site without override authority clears its previous choice.

    Raises
    ------
    ValueError
        A patch is stale, sites/placements overlap, or current chemical boundaries
        disagree with the resolved sites. No caller-owned structure is mutated.
    """
    if not patches:
        return structure
    context = PolymerMicrostateContext(structure)

    touched: set[AtomRef] = set()
    payloads: dict[ResidueId, ResidueFacetPayload] = {}
    for patch in patches:
        if patch.source is not structure:
            raise ValueError("microstate patch requires its original source snapshot")
        refs = {
            AtomRef(patch.residue_id, name)
            for name in (
                *(atom.name for atom in patch.graph.atoms),
                *(entry.atom.name for entry in patch.hydrogens),
                *(
                    entry.hydrogen.atom_name
                    for entry in patch.resolution.observed_hydrogens
                ),
            )
        }
        current = context.hydrogen_attachments(
            patch.residue_id,
            patch.site,
            patch.resolution,
            proposed_parents={
                entry.atom.name: entry.parent_name for entry in patch.hydrogens
            },
        )
        refs.update(AtomRef(patch.residue_id, name) for name in current)
        if refs & touched:
            raise ValueError("microstate patches must not overlap")
        touched.update(refs)
        payload = payloads.get(patch.residue_id)
        if payload is None:
            residue_index = structure.constitution.residue_index(patch.residue_id)
            payload = ResidueFacetPayload(
                structure.constitution.residue_site_at(residue_index),
                structure.residue_geometry(residue_index),
                structure.residue_formal_charge_by_atom_name(residue_index),
            )
        payloads[patch.residue_id] = patch._materialize(payload, current)

    prepared = structure.with_updated_residue_facets_batch(
        (
            payload.residue_site,
            payload.residue_geometry,
            payload.formal_charge_by_atom_name,
        )
        for payload in payloads.values()
    )
    bonds = {bond.endpoint_pair(): bond for bond in prepared.topology.bonds}
    for patch in patches:
        observation = structure.provenance.ingress.observation
        observed_bonds = (
            {}
            if observation is None
            else {
                frozenset(
                    observation.constitution.atom_ref_at(index)
                    for index in bond.endpoint_pair()
                ): bond
                for bond in observation.bonds_for_residue(patch.residue_id)
            }
        )
        updates = [
            (bond.atom_name_1, bond.atom_name_2, bond.order)
            for bond in patch.graph.bonds
        ]
        updates.extend(
            (entry.parent_name, entry.atom.name, 1) for entry in patch.hydrogens
        )
        for first, second, order in updates:
            first_index = prepared.constitution.resolve_atom_index(
                AtomRef(patch.residue_id, first)
            )
            second_index = prepared.constitution.resolve_atom_index(
                AtomRef(patch.residue_id, second)
            )
            assert first_index is not None and second_index is not None
            bond = TopologyBond(
                first_index,
                second_index,
                order=order,
                provenance=BondProvenance.TEMPLATE_RESOLVED,
            )
            existing = bonds.get(bond.endpoint_pair())
            if existing is not None and (
                existing.relationship_type is not BondRelationshipType.COVALENT
                and not (
                    existing.relationship_type is BondRelationshipType.UNKNOWN
                    and existing.provenance is BondProvenance.SOURCE_EXPLICIT
                )
            ):
                raise ValueError(
                    "a noncovalent relationship occupies a required microstate bond"
                )
            if existing is None:
                original = observed_bonds.get(
                    frozenset(
                        (
                            AtomRef(patch.residue_id, first),
                            AtomRef(patch.residue_id, second),
                        )
                    )
                )
                if original is not None and original.relationship_type in (
                    BondRelationshipType.COVALENT,
                    BondRelationshipType.UNKNOWN,
                ):
                    existing = replace(
                        original,
                        atom_index_1=bond.atom_index_1,
                        atom_index_2=bond.atom_index_2,
                        relationship_type=BondRelationshipType.COVALENT,
                    )
            if existing is not None:
                bond = replace(
                    existing,
                    order=order,
                    aromatic=False,
                    relationship_type=BondRelationshipType.COVALENT,
                )
            bonds[bond.endpoint_pair()] = bond

    overrides = [
        override
        for override in structure.provenance.microstate_overrides
        if not any(
            AtomRef(override.residue_id, atom.name) in touched
            for atom in override.graph.atoms
        )
    ]
    overrides.extend(
        AppliedMicrostateOverride(
            patch.residue_id, patch.site.template.component_id, patch.graph
        )
        for patch in patches
        if patch.resolution.basis is MicrostateSelectionBasis.OVERRIDE
    )
    provenance = structure.provenance
    if tuple(overrides) != provenance.microstate_overrides:
        provenance = replace(provenance, microstate_overrides=tuple(overrides))

    return ProteinStructure.from_payload(
        constitution=prepared.constitution,
        geometry=prepared.geometry,
        topology=StructureTopology(
            constitution=prepared.constitution,
            atom_topologies=prepared.topology.atom_topologies,
            bonds=tuple(bonds.values()),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=provenance,
    )

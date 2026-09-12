"""Read current polymer chemistry without confusing selection with realization."""

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.microstate.catalog import (
    PeptideLinkage,
    PolymerChemicalSite,
    standard_microstate_candidates,
)
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.resolution import (
    AppliedMicrostateOverride,
    MicrostateConstraints,
    MicrostateResolution,
    MicrostateResolutionStatus,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    TopologyBond,
)


@dataclass(frozen=True, slots=True, eq=False)
class PolymerMicrostateContext:
    """Bind source evidence, applied choices and current bonds to one snapshot.

    Parameters
    ----------
    source : ProteinStructure
        Immutable snapshot. Build a new context after any chemistry edit.

    Notes
    -----
    The adjacency index is built once and shared across site queries. Resolution
    says which graph is selected; realization separately checks current atoms,
    charges and bonds. Neither operation changes the structure or places H.

    Raises
    ------
    TypeError
        The source is not a canonical structure.
    """

    source: ProteinStructure = field(repr=False)
    _bonds_by_atom: Mapping[AtomIndex, tuple[TopologyBond, ...]] = field(
        init=False, repr=False
    )
    _overrides_by_residue: Mapping[ResidueId, tuple[AppliedMicrostateOverride, ...]] = (
        field(init=False, repr=False)
    )

    def __post_init__(self) -> None:
        if not isinstance(self.source, ProteinStructure):
            raise TypeError("microstate context requires a canonical structure")
        by_atom: dict[AtomIndex, list[TopologyBond]] = defaultdict(list)
        for bond in self.source.topology.bonds:
            for index in bond.endpoint_pair():
                by_atom[index].append(bond)
        object.__setattr__(
            self,
            "_bonds_by_atom",
            MappingProxyType({index: tuple(bonds) for index, bonds in by_atom.items()}),
        )
        overrides: dict[ResidueId, list[AppliedMicrostateOverride]] = defaultdict(list)
        for entry in self.source.provenance.microstate_overrides:
            overrides[entry.residue_id].append(entry)
        object.__setattr__(
            self,
            "_overrides_by_residue",
            MappingProxyType(
                {
                    residue_id: tuple(entries)
                    for residue_id, entries in overrides.items()
                }
            ),
        )

    def resolve(
        self,
        residue_id: ResidueId,
        site: PolymerMicrostateSite,
        *,
        override: MicrostateConstraints | None = None,
        preferences: tuple[MicrostateConstraints, ...] = (),
    ) -> MicrostateResolution:
        """Resolve a site, retaining a compatible previously applied override.

        Parameters
        ----------
        residue_id : ResidueId
            Current polymer residue identity.
        site : PolymerMicrostateSite
            Standard site with caller-established linkage.
        override : MicrostateConstraints or None
            New explicit request, taking precedence over a saved choice.
        preferences : tuple[MicrostateConstraints, ...]
            Ordered preparation preferences; never original observations.

        Returns
        -------
        MicrostateResolution
            Desired chemistry, not current conformance. Saved choices with stale
            component or boundary identity are refused rather than ignored.

        Raises
        ------
        ResidueNotFoundError
            The residue is absent from the snapshot.
        ValueError
            An applicable explicit request is empty.
        """
        constitution = self.source.constitution
        index = constitution.residue_index(residue_id)
        if index.value >= len(constitution.residue_slots) - len(constitution.ligands):
            return MicrostateResolution(
                MicrostateResolutionStatus.UNSUPPORTED,
                details=("polymer microstates require a polymer residue",),
            )
        residue = constitution.residue_site_at(index)
        catalog = standard_microstate_candidates(
            site.template.component_id, site.kind, site.linkage
        )
        names = (
            set()
            if catalog is None
            else {atom.name for atom in catalog.candidates[0].atoms}
        )
        saved = tuple(
            entry
            for entry in self._overrides_by_residue.get(residue_id, ())
            if any(atom.name in names for atom in entry.graph.atoms)
        )
        if override is None and len(saved) > 1:
            return MicrostateResolution(
                MicrostateResolutionStatus.UNSUPPORTED,
                details=("multiple applied choices overlap the current site",),
            )
        return site.resolve(
            residue,
            self.source.provenance.ingress.observation,
            override=override,
            applied_override=saved[0] if len(saved) == 1 else None,
            preferences=preferences,
        )

    def backbone_site(
        self,
        residue_id: ResidueId,
        template: ResidueTemplate,
        kind: PolymerChemicalSite,
        *,
        assume_free_chain_ends: bool = False,
    ) -> PolymerMicrostateSite:
        """Bind a backbone site to actual links or an explicit preparation assumption.

        Parameters
        ----------
        residue_id : ResidueId
            Current polymer residue identity.
        template : ResidueTemplate
            Active component template, checked for applicability by resolution.
        kind : PolymerChemicalSite
            BACKBONE_N or BACKBONE_C, not a side-chain decision.
        assume_free_chain_ends : bool
            Permit FREE at the corresponding outer chain end only when no current
            external relationship occupies that backbone atom. False leaves such
            ends UNKNOWN. This does not certify biological chain completeness.

        Returns
        -------
        PolymerMicrostateSite
            LINKED for one supported canonical C-N bond, marked FREE for the
            permitted preparation assumption, or UNKNOWN. Original observations
            and the full current site boundary still require validation.

        Raises
        ------
        ResidueNotFoundError
            The residue is absent from this snapshot.
        ValueError
            The target is not a polymer backbone site.
        TypeError
            The kind is noncanonical or the assumption option is not boolean.
        """
        if not isinstance(kind, PolymerChemicalSite):
            raise TypeError("backbone site kind must be canonical")
        if kind not in (PolymerChemicalSite.BACKBONE_N, PolymerChemicalSite.BACKBONE_C):
            raise ValueError("backbone binding requires a backbone site kind")
        if type(assume_free_chain_ends) is not bool:
            raise TypeError("chain-end assumption must be a boolean")
        constitution = self.source.constitution
        residue_index = constitution.residue_index(residue_id)
        polymer_count = len(constitution.residue_slots) - len(constitution.ligands)
        if residue_index.value >= polymer_count:
            raise ValueError("backbone binding requires a polymer residue")
        name, partner_name = (
            ("N", "C") if kind is PolymerChemicalSite.BACKBONE_N else ("C", "N")
        )
        index = constitution.resolve_atom_index(AtomRef(residue_id, name))
        linkage = PeptideLinkage.UNKNOWN
        assumed = False
        if index is not None:
            external = tuple(
                (bond, other)
                for bond in self._bonds_by_atom.get(index, ())
                if bond.relationship_type is not BondRelationshipType.HYDROGEN_BOND
                for other in bond.endpoint_pair()
                if other != index
                and constitution.atom_ref_at(other).residue_id != residue_id
            )
            if len(external) == 1:
                bond, other = external[0]
                partner = constitution.atom_ref_at(other)
                if (
                    bond.relationship_type is BondRelationshipType.COVALENT
                    and bond.order == 1
                    and not bond.aromatic
                    and partner.atom_name == partner_name
                    and constitution.atom_site_at(other).element == partner_name
                    and constitution.atom_site_at(index).element == name
                    and constitution.residue_index(partner.residue_id).value
                    < polymer_count
                ):
                    linkage = PeptideLinkage.LINKED
            elif not external and assume_free_chain_ends:
                chain = constitution.chain(residue_id.chain_id)
                end = chain.residues[
                    0 if kind is PolymerChemicalSite.BACKBONE_N else -1
                ]
                if end.residue_id == residue_id:
                    linkage = PeptideLinkage.FREE
                    assumed = True
        return PolymerMicrostateSite(
            template, kind, linkage, free_terminal_assumption=assumed
        )

    def hydrogen_attachments(
        self,
        residue_id: ResidueId,
        site: PolymerMicrostateSite,
        resolution: MicrostateResolution,
        *,
        proposed_parents: Mapping[str, str] | None = None,
    ) -> dict[str, str]:
        """Interpret current H identities and validate this site's edit boundary.

        Parameters
        ----------
        residue_id : ResidueId
            Polymer residue in this snapshot.
        site : PolymerMicrostateSite
            Template and explicitly established chemical linkage.
        resolution : MicrostateResolution
            Desired graph resolved for this snapshot; not a claim of realization.
        proposed_parents : Mapping[str, str] or None
            Additional placement names and their graph-local parents. Used to
            reject collisions or reparenting before applying a patch.

        Returns
        -------
        dict[str, str]
            Current H names and parents, including known but detached H atoms.
            Detached atoms can be repaired but do not count as realized bonds.

        Raises
        ------
        ValueError
            The graph is unresolved, the scaffold/boundary changed, or existing
            H atoms have unsupported charges, relationships or attachments.
        """
        graph = resolution.graph
        if graph is None:
            raise ValueError("current chemistry requires a resolved microstate")
        proposed_parents = {} if proposed_parents is None else proposed_parents
        if not set(proposed_parents.values()) <= {atom.name for atom in graph.atoms}:
            raise ValueError("proposed H parent is outside the microstate")
        bonds_by_atom = self._bonds_by_atom
        constitution = self.source.constitution
        site_names = {atom.name for atom in graph.atoms}
        template_heavy_names = set(site.template.expected_heavy_atom_names())
        boundary_pairs = {
            frozenset((bond.atom_name_1, bond.atom_name_2)): bond.order
            for bond in site.template.definition.bonds
            if (bond.atom_name_1 in site_names) != (bond.atom_name_2 in site_names)
            and bond.atom_name_1 in template_heavy_names
            and bond.atom_name_2 in template_heavy_names
        }
        boundary_orders: Counter[str] = Counter()
        hydrogens: dict[str, str] = {}
        external_partners: set[AtomRef] = set()
        resolved_pairs = {
            frozenset((bond.atom_name_1, bond.atom_name_2)) for bond in graph.bonds
        }
        resolved_pairs.update(
            frozenset((parent, name)) for name, parent in proposed_parents.items()
        )
        resolved_pairs.update(
            frozenset((entry.parent.atom_name, entry.hydrogen.atom_name))
            for entry in resolution.observed_hydrogens
        )
        for atom in graph.atoms:
            index = constitution.resolve_atom_index(AtomRef(residue_id, atom.name))
            if (
                index is None
                or constitution.atom_site_at(index).element != atom.element
            ):
                raise ValueError(
                    "current microstate heavy scaffold is incomplete or changed"
                )
            for bond in bonds_by_atom.get(index, ()):
                if bond.relationship_type is BondRelationshipType.HYDROGEN_BOND:
                    continue
                other = (
                    bond.atom_index_2
                    if bond.atom_index_1 == index
                    else bond.atom_index_1
                )
                other_ref = constitution.atom_ref_at(other)
                other_atom = constitution.atom_site_at(other)
                source_pair_resolved = (
                    bond.relationship_type is BondRelationshipType.UNKNOWN
                    and bond.provenance is BondProvenance.SOURCE_EXPLICIT
                    and other_ref.residue_id == residue_id
                    and frozenset((atom.name, other_atom.name)) in resolved_pairs
                )
                if (
                    bond.relationship_type is not BondRelationshipType.COVALENT
                    and not source_pair_resolved
                ):
                    raise ValueError("unsupported current microstate relationship")
                if other_atom.is_hydrogen():
                    if other_ref.residue_id != residue_id or (
                        bond.order != 1
                        and not (source_pair_resolved and bond.order is None)
                    ):
                        raise ValueError("unsupported current microstate H attachment")
                    if (
                        other_atom.name in hydrogens
                        and hydrogens[other_atom.name] != atom.name
                    ):
                        raise ValueError("current microstate H has multiple parents")
                    hydrogens[other_atom.name] = atom.name
                    continue
                if other_ref.residue_id == residue_id and other_atom.name in site_names:
                    if graph.bond_order(atom.name, other_atom.name) is None:
                        raise ValueError("current bonds change the microstate skeleton")
                    continue

                if other_ref.residue_id == residue_id:
                    expected_order = boundary_pairs.get(
                        frozenset((atom.name, other_atom.name))
                    )
                    allowed = (
                        expected_order is not None and bond.order == expected_order
                    )
                else:
                    expected_names = (
                        ("N", "C")
                        if site.kind is PolymerChemicalSite.BACKBONE_N
                        else ("C", "N")
                    )
                    allowed = (
                        site.kind is not PolymerChemicalSite.SIDECHAIN
                        and site.linkage is PeptideLinkage.LINKED
                        and (atom.name, other_atom.name) == expected_names
                        and other_atom.element == expected_names[1]
                        and bond.order == 1
                        and constitution.residue_index(other_ref.residue_id).value
                        < len(constitution.residue_slots) - len(constitution.ligands)
                    )
                    external_partners.add(other_ref)
                if not allowed or bond.order is None or len(external_partners) > 1:
                    raise ValueError("current bonds disagree with microstate boundary")
                boundary_orders[atom.name] += bond.order

            if boundary_orders[atom.name] != atom.boundary_order:
                raise ValueError("current microstate boundary is incomplete or changed")

        for name in hydrogens:
            index = constitution.resolve_atom_index(AtomRef(residue_id, name))
            assert index is not None
            covalent = [
                bond
                for bond in bonds_by_atom.get(index, ())
                if bond.relationship_type is not BondRelationshipType.HYDROGEN_BOND
            ]
            if len(covalent) != 1 or self.source.topology.formal_charge(index) not in (
                None,
                0,
            ):
                raise ValueError("unsupported current microstate hydrogen chemistry")
        expected_parents = {
            entry.hydrogen.atom_name: entry.parent.atom_name
            for entry in resolution.observed_hydrogens
        }
        residue = constitution.residue_site_at(constitution.residue_index(residue_id))
        for name, parent in site.named_hydrogen_parents(residue).items():
            if parent in site_names:
                expected_parents.setdefault(name, parent)
        expected_parents.update(proposed_parents)
        for name, parent in expected_parents.items():
            index = constitution.resolve_atom_index(AtomRef(residue_id, name))
            if index is None:
                continue
            if self.source.topology.formal_charge(index) not in (None, 0):
                raise ValueError("unsupported current microstate hydrogen chemistry")
            for bond in bonds_by_atom.get(index, ()):
                if bond.relationship_type is BondRelationshipType.HYDROGEN_BOND:
                    continue
                other = (
                    bond.atom_index_2
                    if bond.atom_index_1 == index
                    else bond.atom_index_1
                )
                if (
                    (
                        bond.relationship_type is not BondRelationshipType.COVALENT
                        and not (
                            bond.relationship_type is BondRelationshipType.UNKNOWN
                            and bond.provenance is BondProvenance.SOURCE_EXPLICIT
                        )
                    )
                    or bond.order not in (None, 1)
                    or constitution.atom_ref_at(other) != AtomRef(residue_id, parent)
                ):
                    raise ValueError("current H attachment disagrees with the site")
            if not constitution.atom_site_at(index).is_hydrogen():
                raise ValueError("microstate H identity collides with a heavy atom")
            hydrogens[name] = parent
        return hydrogens

    def is_realized(
        self,
        residue_id: ResidueId,
        site: PolymerMicrostateSite,
        resolution: MicrostateResolution,
    ) -> bool:
        """Check exact current H counts, charges and site bonds.

        Parameters
        ----------
        residue_id : ResidueId
            Current polymer residue identity.
        site : PolymerMicrostateSite
            Standard site and established chemical linkage.
        resolution : MicrostateResolution
            Desired chemistry resolved for this snapshot.

        Returns
        -------
        bool
            False for unresolved chemistry or missing/mismatched site chemistry.
            Heavy coordinates are not assessed by this graph-only predicate.

        Raises
        ------
        ValueError
            Current boundary or H chemistry is outside the supported repair
            contract. This is not merely an incomplete but repairable graph.
        """
        graph = resolution.graph
        if graph is None:
            return False
        hydrogens = self.hydrogen_attachments(residue_id, site, resolution)
        constitution = self.source.constitution
        topology = self.source.topology
        counts = Counter(hydrogens.values())
        if any(counts[atom.name] != atom.hydrogens for atom in graph.atoms):
            return False
        for atom in graph.atoms:
            index = constitution.resolve_atom_index(AtomRef(residue_id, atom.name))
            assert index is not None
            if (topology.formal_charge(index) or 0) != atom.charge:
                return False
        pairs = [
            (bond.atom_name_1, bond.atom_name_2, bond.order) for bond in graph.bonds
        ]
        pairs.extend((parent, name, 1) for name, parent in hydrogens.items())
        for first, second, order in pairs:
            first_index = constitution.resolve_atom_index(AtomRef(residue_id, first))
            second_index = constitution.resolve_atom_index(AtomRef(residue_id, second))
            assert first_index is not None and second_index is not None
            bond = next(
                (
                    candidate
                    for candidate in self._bonds_by_atom.get(first_index, ())
                    if second_index in candidate.endpoint_pair()
                ),
                None,
            )
            if (
                bond is None
                or bond.relationship_type is not BondRelationshipType.COVALENT
                or bond.order != order
                or bond.aromatic
            ):
                return False
        return True

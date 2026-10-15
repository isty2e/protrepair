"""Read current polymer chemistry without confusing selection with realization."""

from collections import Counter, defaultdict
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.microstate.catalog import (
    PeptideLinkage,
    PolymerChemicalSite,
)
from protrepair.chemistry.microstate.graph import MicrostateGraph
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.resolution import (
    AppliedMicrostateChoice,
    MicrostateConstraints,
    MicrostateResolution,
    MicrostateResolutionStatus,
)
from protrepair.chemistry.retained_non_polymer.template_binding import (
    is_resolved_acyl_amide_link,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import AtomSite
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
    _choices_by_residue: Mapping[ResidueId, tuple[AppliedMicrostateChoice, ...]] = (
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
        overrides: dict[ResidueId, list[AppliedMicrostateChoice]] = defaultdict(list)
        for entry in self.source.provenance.microstate_choices:
            overrides[entry.residue_id].append(entry)
        object.__setattr__(
            self,
            "_choices_by_residue",
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
        retain_applied_choice: bool = True,
    ) -> MicrostateResolution:
        """Resolve a site, retaining a compatible choice at its original authority.

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
        retain_applied_choice : bool
            Reuse a saved choice when no new constraint is supplied. False resets
            selection to original evidence and preferences, not neutral chemistry.

        Returns
        -------
        MicrostateResolution
            Desired chemistry, not current conformance. Saved choices with stale
            component or boundary identity are reconsidered when DEFAULT, and
            refused when OVERRIDE.

        Raises
        ------
        ResidueNotFoundError
            The residue is absent from the snapshot.
        ValueError
            An applicable explicit request is empty.
        """
        if type(retain_applied_choice) is not bool:
            raise TypeError("retaining an applied microstate choice requires a bool")
        constitution = self.source.constitution
        index = constitution.residue_index(residue_id)
        if index.value >= len(constitution.residue_slots) - len(constitution.ligands):
            return MicrostateResolution(
                MicrostateResolutionStatus.UNSUPPORTED,
                details=("polymer microstates require a polymer residue",),
            )
        residue = constitution.residue_site_at(index)
        catalog = site.candidates()
        names = (
            set()
            if catalog is None
            else {atom.name for atom in catalog.candidates[0].atoms}
        )
        saved = tuple(
            entry
            for entry in (
                self._choices_by_residue.get(residue_id, ())
                if retain_applied_choice
                else ()
            )
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
            applied_choice=saved[0] if len(saved) == 1 else None,
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
                    and self._is_supported_peptide_partner(index, other, kind)
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

    def _is_supported_peptide_partner(
        self, index: AtomIndex, other: AtomIndex, kind: PolymerChemicalSite
    ) -> bool:
        constitution = self.source.constitution
        partner = constitution.atom_ref_at(other)
        polymer_count = len(constitution.residue_slots) - len(constitution.ligands)
        return constitution.residue_index(partner.residue_id).value < polymer_count or (
            kind is PolymerChemicalSite.BACKBONE_N
            and is_resolved_acyl_amide_link(
                self.source, partner, constitution.atom_ref_at(index)
            )
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
        template_heavy_names = set(site.template.expected_polymer_heavy_atom_names())
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
                        and self._is_supported_peptide_partner(index, other, site.kind)
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

    def hydrogen_parents(
        self,
        residue_id: ResidueId,
        site: PolymerMicrostateSite,
        *,
        original: bool = False,
    ) -> dict[str, str]:
        """Interpret H attachment from explicit bonds, then unambiguous names.

        Parameters
        ----------
        residue_id : ResidueId
            Residue whose fixed-template H identities are being selected.
        site : PolymerMicrostateSite
            Owner of the component's H naming grammar.
        original : bool
            Read immutable observations instead of the current graph.

        Returns
        -------
        dict[str, str]
            Local H name to heavy parent name; no connectivity is manufactured.

        Raises
        ------
        ValueError
            A hydrogen has unresolved, nonlocal, charged or contradictory chemistry.
        """
        observation = self.source.provenance.ingress.observation
        if original and observation is None:
            return {}
        constitution = (
            observation.constitution
            if original and observation is not None
            else self.source.constitution
        )
        residue = constitution.residue_or_ligand(residue_id)
        if residue is None:
            return {}
        by_atom = self._bonds_by_atom
        if original and observation is not None:
            indexed: dict[AtomIndex, list[TopologyBond]] = defaultdict(list)
            for bond in observation.bonds_for_residue(residue_id):
                for index in bond.endpoint_pair():
                    indexed[index].append(bond)
            by_atom = {index: tuple(bonds) for index, bonds in indexed.items()}

        named = site.named_hydrogen_parents(residue)
        original_parents = (
            {} if original else self.hydrogen_parents(residue_id, site, original=True)
        )
        named.update(original_parents)
        result = {}
        for atom in residue.atom_sites:
            if not atom.is_hydrogen():
                continue
            ref = AtomRef(residue_id, atom.name)
            index = constitution.atom_index(ref)
            charge = (
                observation.formal_charge(ref)
                if original and observation is not None
                else self.source.topology.formal_charge(index)
            )
            if charge not in (None, 0):
                raise ValueError("charged H is outside fixed-template chemistry")
            parents = []
            untyped = []
            for bond in by_atom.get(index, ()):
                if bond.relationship_type is BondRelationshipType.HYDROGEN_BOND:
                    continue
                other = (
                    bond.atom_index_2
                    if index == bond.atom_index_1
                    else bond.atom_index_1
                )
                other_ref = constitution.atom_ref_at(other)
                if (
                    other_ref.residue_id != residue_id
                    or constitution.atom_site_at(other).is_hydrogen()
                    or bond.order not in (None, 1)
                    or bond.aromatic
                ):
                    raise ValueError("H attachment requires one local heavy parent")
                if bond.relationship_type is BondRelationshipType.COVALENT:
                    parents.append(other_ref.atom_name)
                elif (
                    bond.relationship_type is BondRelationshipType.UNKNOWN
                    and bond.provenance is BondProvenance.SOURCE_EXPLICIT
                ):
                    untyped.append(other_ref.atom_name)
                else:
                    raise ValueError("unsupported H relationship")
            if len(parents) > 1:
                raise ValueError("H has multiple covalent parents")
            parent = parents[0] if parents else named.get(atom.name)
            if parent is None or any(other != parent for other in untyped):
                raise ValueError(f"unresolved H attachment: {atom.name}")
            if atom.name in original_parents and parent != original_parents[atom.name]:
                raise ValueError("current H attachment contradicts its source parent")
            result[atom.name] = parent
        return result

    def hydrogen_topology_updates(
        self, residue_id: ResidueId, site: PolymerMicrostateSite
    ) -> tuple[TopologyBond, ...]:
        """Propose missing single bonds for supported present polymer H.

        Parameters
        ----------
        residue_id : ResidueId
            Polymer residue in this snapshot.
        site : PolymerMicrostateSite
            Component naming grammar, not a new protonation selection.

        Returns
        -------
        tuple[TopologyBond, ...]
            Missing bonds or resolutions of incomplete source bonds. Existing
            source metadata and endpoint provenance survive resolution.
            This observation does not change the structure.

        Raises
        ------
        ValueError
            The residue is not polymer, a parent is absent or unsupported, or H
            chemistry contradicts source evidence or a known relationship.
        """
        constitution = self.source.constitution
        index = constitution.residue_index(residue_id)
        if index.value >= len(constitution.residue_slots) - len(constitution.ligands):
            raise ValueError("polymer H topology requires a polymer residue")
        updates = []
        for name, parent in self.hydrogen_parents(residue_id, site).items():
            hydrogen_index = constitution.atom_index(AtomRef(residue_id, name))
            parent_index = constitution.resolve_atom_index(AtomRef(residue_id, parent))
            if parent_index is None or constitution.atom_site_at(
                parent_index
            ).element not in {"C", "N", "O", "S", "SE"}:
                raise ValueError(f"unsupported H parent: {parent}")
            existing = next(
                (
                    bond
                    for bond in self._bonds_by_atom.get(hydrogen_index, ())
                    if parent_index in bond.endpoint_pair()
                ),
                None,
            )
            if existing is None:
                updates.append(TopologyBond(parent_index, hydrogen_index))
            elif existing.relationship_type not in {
                BondRelationshipType.COVALENT,
                BondRelationshipType.UNKNOWN,
            }:
                raise ValueError("H parent contradicts a known relationship")
            elif (
                existing.relationship_type is not BondRelationshipType.COVALENT
                or existing.order != 1
            ):
                updates.append(
                    replace(
                        existing,
                        relationship_type=BondRelationshipType.COVALENT,
                        order=1,
                    )
                )
        return tuple(updates)

    def realized_heavy_graph(
        self,
        residue_id: ResidueId,
        site: PolymerMicrostateSite,
        resolution: MicrostateResolution,
    ) -> MicrostateGraph | None:
        """Match current heavy chemistry to an independently selected state.

        Parameters
        ----------
        residue_id : ResidueId
            Current polymer residue identity.
        site : PolymerMicrostateSite
            Supported site with established linkage.
        resolution : MicrostateResolution
            Selection from original evidence or an applied choice, not geometry.

        Returns
        -------
        MicrostateGraph or None
            A protonation-equivalent graph matching current charges and bonds.
            Missing H is allowed; incompatible present H, unsupported boundaries
            or a changed heavy graph refuse the projection. No coordinates are read.
        """
        selected = resolution.graph
        catalog = site.candidates()
        if selected is None or catalog is None:
            return None
        try:
            hydrogens = Counter(
                self.hydrogen_attachments(residue_id, site, resolution).values()
            )
        except ValueError:
            return None
        constitution = self.source.constitution
        topology = self.source.topology
        indices = {
            atom.name: constitution.atom_index(AtomRef(residue_id, atom.name))
            for atom in selected.atoms
        }
        bonds_by_pair = {
            frozenset(bond.endpoint_pair()): bond
            for index in indices.values()
            for bond in self._bonds_by_atom.get(index, ())
        }
        for graph in catalog.candidates:
            if graph.protonation_key() != selected.protonation_key():
                continue
            if any(hydrogens[a.name] > a.hydrogens for a in graph.atoms):
                continue
            if any(
                (topology.formal_charge(indices[a.name]) or 0) != a.charge
                for a in graph.atoms
            ):
                continue
            if all(
                (
                    bond := bonds_by_pair.get(
                        frozenset(
                            (indices[target.atom_name_1], indices[target.atom_name_2])
                        )
                    )
                )
                is not None
                and bond.relationship_type is BondRelationshipType.COVALENT
                and bond.order == target.order
                and not bond.aromatic
                for target in graph.bonds
            ):
                return graph
        return None

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

    def hydrogen_atom_sites(
        self,
        residue_id: ResidueId,
        site: PolymerMicrostateSite,
        resolution: MicrostateResolution,
    ) -> tuple[tuple[AtomSite, str], ...]:
        """Select final H identities and parents for a resolved site.

        Parameters
        ----------
        residue_id : ResidueId
            Current polymer residue identity.
        site : PolymerMicrostateSite
            Standard chemical site with established linkage.
        resolution : MicrostateResolution
            Desired coupled graph and original H attachment evidence.

        Returns
        -------
        tuple[tuple[AtomSite, str], ...]
            Final atoms and their parent names, preserving original H/D/T first,
            then current H identities, then choosing unoccupied standard names.
            No coordinates or source-removal authority are inferred here.

        Raises
        ------
        ValueError
            Chemistry is unresolved, current attachments are unsupported, or
            standard names cannot represent the selected graph without collision.
        """
        current = self.hydrogen_attachments(residue_id, site, resolution)
        graph = resolution.graph
        assert graph is not None
        constitution = self.source.constitution
        residue = constitution.residue_site_at(constitution.residue_index(residue_id))
        observation = self.source.provenance.ingress.observation
        original: dict[str, tuple[AtomSite, str]] = {}
        for attachment in resolution.observed_hydrogens:
            assert observation is not None
            index = observation.constitution.resolve_atom_index(attachment.hydrogen)
            assert index is not None
            original[attachment.hydrogen.atom_name] = (
                observation.constitution.atom_site_at(index),
                attachment.parent.atom_name,
            )

        reserved = dict(current)
        reserved.update((name, parent) for name, (_atom, parent) in original.items())
        final: list[tuple[AtomSite, str]] = []
        for parent in graph.atoms:
            candidates = [
                atom
                for name, (atom, owner) in sorted(original.items())
                if owner == parent.name
            ]
            candidates.extend(
                residue.atom_site(name)
                for name, owner in sorted(current.items())
                if owner == parent.name and name not in original
            )
            selected = candidates[: parent.hydrogens]
            for name in site.hydrogen_name_candidates(parent.name):
                if len(selected) == parent.hydrogens:
                    break
                if name in reserved or residue.has_atom_site(name):
                    continue
                selected.append(AtomSite(name, "H"))
                reserved[name] = parent.name
            if len(selected) != parent.hydrogens:
                raise ValueError(
                    "microstate H identities collide with occupied atom names"
                )
            final.extend((atom, parent.name) for atom in selected)
        return tuple(final)

    def covalent_neighbors(
        self, atom_ref: AtomRef, *, order: int | None = None
    ) -> tuple[AtomRef, ...]:
        """Read current covalent neighbors through the snapshot adjacency index.

        Parameters
        ----------
        atom_ref : AtomRef
            Atom identity in this snapshot.
        order : int or None
            Restrict to an effective integral bond order; None admits any order.

        Returns
        -------
        tuple[AtomRef, ...]
            Stable, sorted neighbor identities. Absent atoms have no neighbors;
            untyped and non-covalent relationships are not guessed as covalent.
        """
        constitution = self.source.constitution
        index = constitution.resolve_atom_index(atom_ref)
        return (
            tuple(
                sorted(
                    constitution.atom_ref_at(other)
                    for bond in self._bonds_by_atom.get(index, ())
                    if bond.relationship_type is BondRelationshipType.COVALENT
                    and (order is None or bond.order == order)
                    for other in bond.endpoint_pair()
                    if other != index
                )
            )
            if index is not None
            else ()
        )

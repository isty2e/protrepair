"""Polymer preparation targets, separate from observed chemistry."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

from protrepair.chemistry.component.defaults import build_default_component_library
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.microstate.catalog import (
    PeptideLinkage,
    PolymerChemicalSite,
    standard_microstate_candidates,
)
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.histidine_environment import HistidineEnvironment
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.resolution import (
    MicrostateConstraints,
    MicrostateResolution,
    MicrostateResolutionStatus,
)
from protrepair.chemistry.retained_non_polymer.covalent_context import (
    RetainedCovalentContext,
)
from protrepair.chemistry.retained_non_polymer.glycosidic import (
    GlycosidicTemplateBinding,
)
from protrepair.chemistry.retained_non_polymer.michael_adduct import (
    CysteineMichaelAdductBinding,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import AtomSite
from protrepair.structure.disulfide import (
    disulfide_atom_ref_pair,
    disulfide_bonded_cysteine_residue_ids,
)
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import (
    BondRelationshipType,
    TopologyBond,
)


@dataclass(frozen=True, slots=True)
class PolymerSitePreparation:
    """Bind desired chemistry and realization queries to the same snapshot.

    Parameters
    ----------
    context : PolymerMicrostateContext
        Current structure and immutable original observations.
    residue_id : ResidueId
        Polymer residue identity.
    site : PolymerMicrostateSite
        Chemical site and established linkage context.
    preferences : tuple[MicrostateConstraints, ...]
        Ordered preparation preferences shared with graph-constrained placement.
    override : MicrostateConstraints or None
        Explicit replacement constraints, if requested.
    retain_applied_choice : bool
        False resets an earlier choice to source-compatible preparation preferences.

    Notes
    -----
    This is a target, not proof that the structure already realizes it. Placement
    and mutation remain transformer responsibilities.
    """

    context: PolymerMicrostateContext = field(repr=False)
    residue_id: ResidueId
    site: PolymerMicrostateSite
    preferences: tuple[MicrostateConstraints, ...]
    override: MicrostateConstraints | None = None
    retain_applied_choice: bool = True
    resolution: MicrostateResolution = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resolution",
            self.context.resolve(
                self.residue_id,
                self.site,
                override=self.override,
                preferences=self.preferences,
                retain_applied_choice=self.retain_applied_choice,
            ),
        )

    def hydrogen_atom_sites(self) -> tuple[tuple[AtomSite, str], ...]:
        """Return selected H identities and parents, or reject an unresolved target.

        Returns
        -------
        tuple[tuple[AtomSite, str], ...]
            Complete desired H inventory, preserving original isotope identities.

        Raises
        ------
        ValueError
            Selection, current boundaries or atom naming cannot be realized.
        """
        return self.context.hydrogen_atom_sites(
            self.residue_id, self.site, self.resolution
        )

    def is_realized(self) -> bool:
        """Test the current graph, without treating selection as conformance.

        Returns
        -------
        bool
            Whether H, charges, orders and current boundaries match the target.
            Unresolved or unsupported current chemistry returns False.
        """
        try:
            return self.context.is_realized(self.residue_id, self.site, self.resolution)
        except ValueError:
            return False

    def controlled_parent_names(self) -> frozenset[str]:
        """Identify this site's atoms even when source evidence prevents selection.

        Returns
        -------
        frozenset[str]
            Parent names whose H cannot be independently chosen by a template.
        """
        if self.site.kind is PolymerChemicalSite.BACKBONE_N:
            return frozenset({"N"})
        if self.site.kind is PolymerChemicalSite.BACKBONE_C:
            return frozenset({"C", "O", "OXT"})
        catalog = standard_microstate_candidates(
            self.site.template.component_id, self.site.kind, self.site.linkage
        )
        return (
            frozenset()
            if catalog is None
            else frozenset(atom.name for atom in catalog.candidates[0].atoms)
        )

    def requires_terminal_oxygen(self) -> bool:
        """Identify the heavy-atom prerequisite for an assumed free C terminus.

        Returns
        -------
        bool
            True only for a free C site without OXT. The heavy-atom transformer
            must still validate its own placement prerequisites.
        """
        residue = self.context.source.constitution.residue_or_ligand(self.residue_id)
        assert residue is not None
        return (
            self.site.kind is PolymerChemicalSite.BACKBONE_C
            and self.site.linkage is PeptideLinkage.FREE
            and not residue.has_atom_site("OXT")
            and self.resolution.status is not MicrostateResolutionStatus.UNSUPPORTED
        )


@dataclass(frozen=True, slots=True, init=False)
class FixedPolymerHydrogenContext:
    """Select fixed H while retaining uncertainty at unresolved linked parents.

    Parameters
    ----------
    context : PolymerMicrostateContext
        Current snapshot and source observations.
    residue_id : ResidueId
        Present polymer residue with a component template.
    template : ResidueTemplate
        Isolated fixed-H chemistry.
    component_library : ComponentLibrary
        Partner definitions for contextual ASN chemistry.
    controlled_parent_names : frozenset[str]
        Parents whose inventory belongs to coupled microstate targets.
    cysteine_adduct : CysteineMichaelAdductBinding or None
        Realized product selected for this polymer site, when available.

    Notes
    -----
    Unexplained external covalent-like relationships leave only the affected
    parents unresolved. Supported products and coupled sites keep their owners.
    This value does not place atoms or certify completed H coverage.
    """

    isolated_template: ResidueTemplate
    glycosidic: GlycosidicTemplateBinding | None
    failure_reason: str
    _unresolved_parent_names: frozenset[str]

    def __init__(
        self,
        context: PolymerMicrostateContext,
        residue_id: ResidueId,
        template: ResidueTemplate,
        component_library: ComponentLibrary,
        *,
        controlled_parent_names: frozenset[str],
        cysteine_adduct: CysteineMichaelAdductBinding | None = None,
    ) -> None:
        if cysteine_adduct is not None:
            template = cysteine_adduct.cysteine_template
        object.__setattr__(self, "isolated_template", template)
        object.__setattr__(self, "glycosidic", None)
        object.__setattr__(self, "failure_reason", "")
        object.__setattr__(self, "_unresolved_parent_names", frozenset())
        residue = context.source.constitution.residue_or_ligand(residue_id)
        if residue is None:
            raise ValueError("fixed H preparation requires a present residue")
        covalent_or_untyped = (
            BondRelationshipType.COVALENT,
            BondRelationshipType.DISULFIDE,
            BondRelationshipType.UNKNOWN,
        )
        cysteine_sg_explained = cysteine_adduct is not None
        if residue.component_id == "CYS" and residue.has_atom_site("SG"):
            external = context.external_heavy_bonds(AtomRef(residue_id, "SG"))
            cysteine_sg_explained = cysteine_sg_explained or any(
                disulfide_atom_ref_pair(context.source.constitution, bond) is not None
                for bond in external
            )
            if not cysteine_sg_explained and any(
                bond.relationship_type
                not in (
                    BondRelationshipType.HYDROGEN_BOND,
                    BondRelationshipType.METAL_COORDINATION,
                )
                for bond in external
            ):
                object.__setattr__(
                    self,
                    "failure_reason",
                    "contextual cysteine product is unsupported or not yet realized",
                )
                object.__setattr__(self, "_unresolved_parent_names", frozenset(("SG",)))
        if (
            residue.component_id == "ASN"
            and residue.has_atom_site("ND2")
            and any(
                bond.relationship_type in covalent_or_untyped
                for bond in context.external_heavy_bonds(AtomRef(residue_id, "ND2"))
            )
        ):
            try:
                binding = GlycosidicTemplateBinding(
                    context.source, residue, template, component_library
                )
            except ValueError as error:
                object.__setattr__(self, "failure_reason", str(error))
                object.__setattr__(
                    self, "_unresolved_parent_names", frozenset(("ND2",))
                )
            else:
                object.__setattr__(self, "glycosidic", binding)

        active = self.template
        parents = set(
            active.template_hydrogen_anchor_by_name(
                active.expected_hydrogen_atom_names()
            ).values()
        ).difference(controlled_parent_names, self.unresolved_parent_names)
        if cysteine_sg_explained:
            parents.discard("SG")
        if self.glycosidic is not None and self.glycosidic.amide_partner is not None:
            parents.discard("ND2")
        unresolved = set(self.unresolved_parent_names)
        reasons = [self.failure_reason] if self.failure_reason else []
        for parent in sorted(parents):
            if not residue.has_atom_site(parent):
                continue
            ref = AtomRef(residue_id, parent)
            external = tuple(
                bond
                for bond in context.external_heavy_bonds(ref)
                if bond.relationship_type in covalent_or_untyped
            )
            if not external:
                continue
            unresolved.add(parent)
            partners = sorted(
                context.source.constitution.atom_ref_at(index).display_token()
                for bond in external
                for index in bond.endpoint_pair()
                if context.source.constitution.atom_ref_at(index) != ref
            )
            reasons.append(
                f"isolated fixed H inventory at {ref.display_token()} cannot "
                f"interpret external covalent-like relationship with {', '.join(partners)}"
            )
        object.__setattr__(self, "_unresolved_parent_names", frozenset(unresolved))
        object.__setattr__(self, "failure_reason", "; ".join(reasons))

    @property
    def template(self) -> ResidueTemplate:
        """Return the product or isolated grammar for unaffected parents.

        Returns
        -------
        ResidueTemplate
            Placement grammar; unresolved parents must still be excluded.
        """
        return (
            self.isolated_template
            if self.glycosidic is None
            else self.glycosidic.template
        )

    @property
    def unresolved_parent_names(self) -> frozenset[str]:
        """Identify fixed parents whose contextual H inventory is unknown.

        Returns
        -------
        frozenset[str]
            Present parents whose linked H inventory is not selected.
        """
        return self._unresolved_parent_names

    def hydrogen_atom_names(self) -> tuple[str, ...]:
        """Return only H identities with a resolved parent inventory.

        Returns
        -------
        tuple[str, ...]
            Selected template names outside unresolved fixed parents.
        """
        template = self.template
        names = template.expected_hydrogen_atom_names()
        anchors = template.template_hydrogen_anchor_by_name(names)
        return tuple(
            name
            for name in names
            if anchors.get(name) not in self.unresolved_parent_names
        )


@dataclass(frozen=True, slots=True, init=False)
class PolymerMicrostatePreparation:
    """Resolve one snapshot's polymer targets for placement and state consumers.

    Parameters
    ----------
    context : PolymerMicrostateContext
        Snapshot whose original and current chemistry are interpreted once.
    component_library : ComponentLibrary
        Active definitions. Custom definitions remain explicitly unsupported by
        the standard-site resolver, not silently replaced by stock chemistry.
    requests : Mapping or None
        Keys are (ResidueId, PolymerChemicalSite), values are constraints or None.
        An absent key preserves a saved choice. A present None resets that site
        to source-compatible preferences; constraints request an explicit override.

    Notes
    -----
    Charge defaults are resolved together before local neutral-HIS preferences.
    Ambiguous neighboring rings supply no polar support, so residue iteration
    order cannot act as an implicit H-bond network optimizer. Unselected HIS
    preferences depend on coordinates and cannot be reused after coordinate edits.

    Raises
    ------
    ValueError
        A request targets a site absent from this polymer preparation.
    """

    _targets: Mapping[ResidueId, tuple[PolymerSitePreparation, ...]]
    context: PolymerMicrostateContext
    component_library: ComponentLibrary
    _disulfides: frozenset[ResidueId]
    _coordinate_dependent: bool
    _cysteine_adducts: Mapping[ResidueId, CysteineMichaelAdductBinding]
    _requests: Mapping[
        tuple[ResidueId, PolymerChemicalSite], MicrostateConstraints | None
    ]

    def __init__(
        self,
        context: PolymerMicrostateContext,
        component_library: ComponentLibrary,
        *,
        requests: Mapping[
            tuple[ResidueId, PolymerChemicalSite], MicrostateConstraints | None
        ]
        | None = None,
    ) -> None:
        requested = {} if requests is None else dict(requests)
        pending = dict(requested)
        adducts = {}
        coordinate_dependent_adducts = False
        for ligand in context.source.constitution.ligands:
            attachments = RetainedCovalentContext(
                context.source, ligand.residue_id
            ).attachments
            if not any(
                bond.relationship_type is BondRelationshipType.COVALENT
                and outside.atom_name == "SG"
                and (
                    partner := context.source.constitution.residue_or_ligand(
                        outside.residue_id
                    )
                )
                is not None
                and partner.component_id == "CYS"
                for bond, _, outside in attachments
            ):
                continue
            coordinate_dependent_adducts = True
            ligand_template = component_library.get(ligand.component_id)
            if ligand_template is None:
                continue
            try:
                binding = CysteineMichaelAdductBinding(
                    context.source, ligand, ligand_template, component_library
                )
                if binding.cysteine_is_realized():
                    adducts[binding.sulfur.residue_id] = binding
            except ValueError:
                continue
        targets = {}
        for chain in context.source.constitution.chains:
            for residue in chain.residues:
                template = component_library.get(residue.component_id)
                if template is None:
                    continue
                if (
                    standard_microstate_candidates(
                        template.backbone_family_component_id,
                        PolymerChemicalSite.BACKBONE_N,
                        PeptideLinkage.LINKED,
                    )
                    is None
                ):
                    continue
                entries = []
                for site in pras_polymer_microstate_sites(
                    context, residue.residue_id, template
                ):
                    key = (residue.residue_id, site.kind)
                    retain = key not in pending
                    preferences = pras_microstate_preferences(site)
                    if site.kind is PolymerChemicalSite.SIDECHAIN:
                        if template.component_id == "HIS":
                            preferences = preferences[:1]
                        elif template.component_id == "CYS":
                            preferences = _cysteine_microstate_preferences(
                                context, residue.residue_id, component_library
                            )
                    entries.append(
                        PolymerSitePreparation(
                            context,
                            residue.residue_id,
                            site,
                            preferences=preferences,
                            override=pending.pop(key, None),
                            retain_applied_choice=retain,
                        )
                    )
                targets[residue.residue_id] = tuple(entries)
        if pending:
            raise ValueError("microstate request targets an absent polymer site")
        unresolved_histidines = tuple(
            target
            for entries in targets.values()
            for target in entries
            if target.site.template.component_id == "HIS"
            and target.site.kind is PolymerChemicalSite.SIDECHAIN
            and target.resolution.status is MicrostateResolutionStatus.AMBIGUOUS
            and all(
                graph.protonation_key()[0] == 0
                for graph in target.resolution.candidates
            )
        )
        if unresolved_histidines:
            environment = HistidineEnvironment(
                context,
                component_library,
                {
                    target.residue_id: target.resolution
                    for entries in targets.values()
                    for target in entries
                    if target.site.kind is PolymerChemicalSite.SIDECHAIN
                },
            )
            for target in unresolved_histidines:
                preference = environment.preference(target.residue_id)
                defaults = pras_microstate_preferences(target.site)
                updated = replace(
                    target,
                    preferences=(
                        defaults[0],
                        *((preference,) if preference is not None else ()),
                        *defaults[1:],
                    ),
                )
                targets[target.residue_id] = tuple(
                    updated if entry is target else entry
                    for entry in targets[target.residue_id]
                )
        object.__setattr__(
            self,
            "_coordinate_dependent",
            bool(unresolved_histidines) or coordinate_dependent_adducts,
        )
        object.__setattr__(self, "_cysteine_adducts", MappingProxyType(adducts))
        object.__setattr__(self, "_targets", MappingProxyType(targets))
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "component_library", component_library)
        object.__setattr__(self, "_requests", MappingProxyType(requested))
        object.__setattr__(
            self, "_disulfides", disulfide_bonded_cysteine_residue_ids(context.source)
        )

    def for_inputs(
        self,
        structure: ProteinStructure,
        component_library: ComponentLibrary,
        *,
        requests: Mapping[
            tuple[ResidueId, PolymerChemicalSite], MicrostateConstraints | None
        ]
        | None = None,
    ) -> "PolymerMicrostatePreparation":
        """Bind placement targets to the exact snapshot and complete requests.

        Parameters
        ----------
        structure : ProteinStructure
            Snapshot whose coordinates and source evidence will be consumed.
        component_library : ComponentLibrary
            Definitions for this operation; identity is required for reuse.
        requests : Mapping or None
            Complete new request map, not a patch over this preparation's map.
            None and an empty map retain saved choices. A present None resets
            that site and is distinct from an absent key.

        Returns
        -------
        PolymerMicrostatePreparation
            This instance only for identical bindings; otherwise a fresh one.

        Raises
        ------
        ValueError
            A request targets an absent polymer site.
        """
        requested = {} if requests is None else requests
        if (
            self.context.source is structure
            and self.component_library is component_library
            and self._requests == requested
        ):
            return self
        return PolymerMicrostatePreparation(
            PolymerMicrostateContext(structure),
            component_library,
            requests=requested,
        )

    def matches_chemistry(self, structure: ProteinStructure) -> bool:
        """Check every input to selection, including an evaluated environment.

        Returns
        -------
        bool
            Whether all inputs to identity selection and realization are equal.
            This does not authorize reuse of coordinate placements.
        """
        source = self.context.source
        return (
            source.constitution == structure.constitution
            and source.topology == structure.topology
            and source.provenance == structure.provenance
            and (
                not self._coordinate_dependent or source.geometry == structure.geometry
            )
        )

    def fixed_hydrogen_context(
        self, residue_id: ResidueId
    ) -> FixedPolymerHydrogenContext | None:
        """Select fixed H inventories and retain unresolved parent identities.

        Parameters
        ----------
        residue_id : ResidueId
            Polymer identity in the preparation snapshot.

        Returns
        -------
        FixedPolymerHydrogenContext or None
            Partial or resolved fixed-H selection; None when no template exists.

        Raises
        ------
        ValueError
            The polymer residue is absent from this snapshot.
        """
        residue = self.context.source.constitution.residue_or_ligand(residue_id)
        if residue is None:
            raise ValueError("fixed H preparation requires a present residue")
        template = self.component_library.get(residue.component_id)
        if template is None:
            return None
        return FixedPolymerHydrogenContext(
            self.context,
            residue_id,
            template,
            self.component_library,
            controlled_parent_names=frozenset(
                name
                for target in self.targets_for(residue_id)
                for name in target.controlled_parent_names()
            ),
            cysteine_adduct=self._cysteine_adducts.get(residue_id),
        )

    def fixed_hydrogen_atom_sites(
        self, residue_id: ResidueId
    ) -> tuple[tuple[AtomSite, str], ...]:
        """Select fixed-template H identities outside coupled chemical sites.

        Parameters
        ----------
        residue_id : ResidueId
            Polymer residue in this snapshot.

        Returns
        -------
        tuple[tuple[AtomSite, str], ...]
            H identity and parent. Original identities precede current generated
            identities, then template names. No coordinates or bonds are inferred.

        Raises
        ------
        ValueError
            Original fixed-parent H exceed the template inventory, or a required
            identity collides with another atom. Unresolved linked parents are
            excluded rather than assigned the isolated template inventory.
        """
        residue = self.context.source.constitution.residue_or_ligand(residue_id)
        if residue is None:
            raise ValueError("fixed H preparation requires a present residue")
        template = self.component_library.get(residue.component_id)
        if template is None or not template.can_add_hydrogens():
            return ()
        isolated = template
        fixed_context = self.fixed_hydrogen_context(residue_id)
        assert fixed_context is not None
        template = fixed_context.template
        controlled = frozenset(
            name
            for target in self.targets_for(residue_id)
            for name in target.controlled_parent_names()
        )
        expected = fixed_context.hydrogen_atom_names()
        anchors = template.template_hydrogen_anchor_by_name(expected)
        names_by_parent: dict[str, list[str]] = {}
        for name in expected:
            parent = anchors.get(name)
            if parent is None or parent in controlled:
                continue
            if residue_id in self._disulfides and parent == "SG":
                continue
            names_by_parent.setdefault(parent, []).append(name)

        observation = self.context.source.provenance.ingress.observation
        original = (
            None
            if observation is None
            else observation.constitution.residue_or_ligand(residue_id)
        )
        grammar = PolymerMicrostateSite(isolated, PolymerChemicalSite.SIDECHAIN)
        selected: dict[str, list[AtomSite]] = {parent: [] for parent in names_by_parent}
        for candidate in (original, residue):
            if candidate is None or candidate.component_id != residue.component_id:
                continue
            named = self.context.hydrogen_parents(
                residue_id, grammar, original=candidate is original
            )
            for atom in candidate.atom_sites:
                parent = named.get(atom.name)
                if parent not in selected:
                    continue
                entries = selected[parent]
                if any(entry.name == atom.name for entry in entries):
                    continue
                if len(entries) >= len(names_by_parent[parent]):
                    if candidate is original:
                        raise ValueError(
                            f"source H exceed fixed template inventory at {parent}"
                        )
                    continue
                entries.append(atom)
        result = []
        used = {atom.name for entries in selected.values() for atom in entries}
        for parent, names in names_by_parent.items():
            entries = selected[parent]
            for name in names:
                if len(entries) == len(names):
                    break
                if name in used:
                    continue
                if residue.has_atom_site(name):
                    raise ValueError(f"fixed H identity collides with {name}")
                entries.append(AtomSite(name, "H"))
                used.add(name)
            if len(entries) != len(names):
                raise ValueError(f"fixed H names exhausted at {parent}")
            result.extend((atom, parent) for atom in entries)
        return tuple(result)

    def fixed_hydrogen_topology_updates(
        self, residue_id: ResidueId
    ) -> tuple[TopologyBond, ...]:
        """Propose bonds for present fixed H without placing or renaming atoms.

        Parameters
        ----------
        residue_id : ResidueId
            Polymer residue whose source inventory is to be retained.

        Returns
        -------
        tuple[TopologyBond, ...]
            Supported updates outside coupled microstate sites. Missing atoms
            remain for H placement; absent component support produces no update.

        Raises
        ------
        ValueError
            H inventory or attachment evidence is contradictory or unsupported.
        """
        selected = {atom.name for atom, _ in self.fixed_hydrogen_atom_sites(residue_id)}
        if not selected:
            return ()
        structure = self.context.source
        residue = structure.constitution.residue_site_at(
            structure.constitution.residue_index(residue_id)
        )
        template = self.component_library.get(residue.component_id)
        assert template is not None
        return tuple(
            bond
            for bond in self.context.hydrogen_topology_updates(
                residue_id,
                PolymerMicrostateSite(template, PolymerChemicalSite.SIDECHAIN),
            )
            if any(
                structure.constitution.atom_site_at(index).is_hydrogen()
                and structure.constitution.atom_ref_at(index).atom_name in selected
                for index in bond.endpoint_pair()
            )
        )

    def targets_for(self, residue_id: ResidueId) -> tuple[PolymerSitePreparation, ...]:
        """Return all chemical targets for a polymer residue.

        Parameters
        ----------
        residue_id : ResidueId
            Residue identity in the bound snapshot.

        Returns
        -------
        tuple[PolymerSitePreparation, ...]
            Empty for an unknown component or non-polymer residue; emptiness is
            not evidence of chemistry readiness.
        """
        return self._targets.get(residue_id, ())

    def has_pending_hydrogen_updates(self, residue_id: ResidueId) -> bool:
        """Return whether supported H or selected chemistry still needs an edit.

        Parameters
        ----------
        residue_id : ResidueId
            Polymer identity in this preparation's snapshot.

        Returns
        -------
        bool
            Whether a selected graph, H identity or fixed H bond needs an edit.
            An unresolved or incompatible fixed inventory does not create work.

        Raises
        ------
        ValueError
            The residue is absent from the snapshot.

        Notes
        -----
        Unresolved parents do not become executable work merely because the
        residue's overall H coverage remains incomplete.
        """
        if any(
            target.resolution.graph is not None and not target.is_realized()
            for target in self.targets_for(residue_id)
        ):
            return True
        residue = self.context.source.constitution.residue_or_ligand(residue_id)
        if residue is None:
            raise ValueError("H update assessment requires a present residue")
        try:
            selected = self.fixed_hydrogen_atom_sites(residue_id)
            if any(not residue.has_atom_site(atom.name) for atom, _ in selected):
                return True
            return bool(self.fixed_hydrogen_topology_updates(residue_id))
        except ValueError:
            return False


def pras_polymer_microstate_sites(
    context: PolymerMicrostateContext,
    residue_id: ResidueId,
    template: ResidueTemplate,
) -> tuple[PolymerMicrostateSite, ...]:
    """Select PRAS preparation contexts without promoting assumptions to evidence.

    Parameters
    ----------
    context : PolymerMicrostateContext
        Snapshot-bound current chemistry and original observations.
    residue_id : ResidueId
        Polymer residue to prepare.
    template : ResidueTemplate
        Active component template; its applicability is checked by site resolution.

    Returns
    -------
    tuple[PolymerMicrostateSite, ...]
        Catalogued side-chain sites plus backbone N/C sites. Unbonded outer
        chain ends receive a marked free-terminal preparation assumption; current
        external relationships and original evidence cannot be overridden by it.
        CYS uses a microstate site here only with a current or original typed
        coordination context; ordinary CYS retains its fixed-template path.
        No pKa prediction or biological completeness claim is made.

    Raises
    ------
    ResidueNotFoundError
        The residue is absent from the snapshot.
    ValueError
        The residue is not polymer.
    """
    sites = []
    if (
        standard_microstate_candidates(
            template.component_id, PolymerChemicalSite.SIDECHAIN, PeptideLinkage.UNKNOWN
        )
        is not None
    ):
        residue = context.source.constitution.residue_or_ligand(residue_id)
        covalent_sulfur = (
            template.component_id == "CYS"
            and residue is not None
            and residue.has_atom_site("SG")
            and any(
                bond.relationship_type
                not in (
                    BondRelationshipType.HYDROGEN_BOND,
                    BondRelationshipType.METAL_COORDINATION,
                )
                for bond in context.external_heavy_bonds(AtomRef(residue_id, "SG"))
            )
        )
        coordinated_sulfur = template.component_id != "CYS" or (
            residue is not None
            and residue.has_atom_site("SG")
            and (
                any(
                    bond.relationship_type is BondRelationshipType.METAL_COORDINATION
                    for bond in context.external_heavy_bonds(AtomRef(residue_id, "SG"))
                )
                or (
                    (observation := context.source.provenance.ingress.observation)
                    is not None
                    and any(
                        bond.relationship_type
                        is BondRelationshipType.METAL_COORDINATION
                        for bond in observation.bonds_for_atom(
                            AtomRef(residue_id, "SG")
                        )
                    )
                )
            )
        )
        if not covalent_sulfur and coordinated_sulfur:
            sites.append(PolymerMicrostateSite(template, PolymerChemicalSite.SIDECHAIN))
    sites.extend(
        context.backbone_site(residue_id, template, kind, assume_free_chain_ends=True)
        for kind in (PolymerChemicalSite.BACKBONE_N, PolymerChemicalSite.BACKBONE_C)
    )
    return tuple(sites)


def pras_microstate_preferences(
    site: PolymerMicrostateSite,
) -> tuple[MicrostateConstraints, ...]:
    """Express the legacy H convention as coupled-graph preferences.

    Parameters
    ----------
    site : PolymerMicrostateSite
        Chemical decision unit with caller-established backbone linkage.
        Applicability, including custom-template refusal, belongs to its resolver.

    Returns
    -------
    tuple[MicrostateConstraints, ...]
        Ordered, source-compatible preferences: charged ARG/LYS, deprotonated
        ASP/GLU, neutral HIS followed by epsilon tautomer, and zwitterionic free
        termini. Linked backbones and sites outside this policy have no preference.

    Notes
    -----
    This is a preparation convention, not a pKa or environmental prediction.
    Optional PRAS histidine-ratio targets supply explicit ring constraints instead
    of changing this default. A predictor can supply different preferences to
    the same resolver without taking ownership of graph application.
    """
    component = site.template.component_id
    if site.kind is PolymerChemicalSite.SIDECHAIN:
        if component == "CYS":
            return (MicrostateConstraints(charges=(("SG", 0),)),)
        if component == "ARG":
            return (
                MicrostateConstraints(hydrogens=(("NE", 1), ("NH1", 2), ("NH2", 2))),
            )
        if component == "LYS":
            return (MicrostateConstraints(hydrogens=(("NZ", 3),)),)
        if component == "ASP":
            return (MicrostateConstraints(hydrogens=(("OD1", 0), ("OD2", 0))),)
        if component == "GLU":
            return (MicrostateConstraints(hydrogens=(("OE1", 0), ("OE2", 0))),)
        if component == "HIS":
            return (
                MicrostateConstraints(charges=(("ND1", 0), ("NE2", 0))),
                MicrostateConstraints(hydrogens=(("ND1", 0), ("NE2", 1))),
            )
    elif site.linkage is PeptideLinkage.FREE:
        if site.kind is PolymerChemicalSite.BACKBONE_N:
            return (
                MicrostateConstraints(
                    hydrogens=(
                        (
                            "N",
                            2
                            if site.template.backbone_family_component_id == "PRO"
                            else 3,
                        ),
                    )
                ),
            )
        if site.kind is PolymerChemicalSite.BACKBONE_C:
            return (MicrostateConstraints(hydrogens=(("O", 0), ("OXT", 0))),)

    return ()


def _cysteine_microstate_preferences(
    context: PolymerMicrostateContext,
    residue_id: ResidueId,
    component_library: ComponentLibrary,
) -> tuple[MicrostateConstraints, ...]:
    sulfur = AtomRef(residue_id, "SG")
    if context.source.constitution.resolve_atom_index(sulfur) is None:
        return ()
    coordination = tuple(
        bond
        for bond in context.external_heavy_bonds(sulfur)
        if bond.relationship_type is BondRelationshipType.METAL_COORDINATION
    )
    if len(coordination) != 1:
        return ()
    partner = next(
        context.source.constitution.atom_ref_at(index)
        for index in coordination[0].endpoint_pair()
        if context.source.constitution.atom_ref_at(index) != sulfur
    )
    residue = context.source.constitution.residue_or_ligand(partner.residue_id)
    if residue is None or residue.component_id != "HEM" or partner.atom_name != "FE":
        return ()
    template = component_library.get("HEM")
    stock = build_default_component_library().get("HEM")
    if (
        template is None
        or template != stock
        or not template.has_complete_heavy_scaffold(residue)
    ):
        return ()

    observation = context.source.provenance.ingress.observation
    if observation is not None and not observation.heavy_atoms_are_preserved_in(
        residue
    ):
        return ()

    # A preparation preference never excludes source-compatible neutral thiol.
    return (MicrostateConstraints(charges=(("SG", -1),)),)

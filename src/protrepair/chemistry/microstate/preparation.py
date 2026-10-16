"""Polymer preparation targets, separate from observed chemistry."""

from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from types import MappingProxyType

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
from protrepair.chemistry.retained_non_polymer.glycosidic import (
    GlycosidicTemplateBinding,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import AtomSite
from protrepair.structure.disulfide import disulfide_bonded_cysteine_residue_ids
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import BondRelationshipType, TopologyBond


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
    """Select fixed H while retaining uncertainty at a linked amide parent.

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

    Notes
    -----
    Failed ASN binding leaves ND2 unresolved, not neutral primary-amide
    chemistry. Other fixed parents retain their independently selected H.
    This value does not place atoms or certify completed H coverage.
    """

    isolated_template: ResidueTemplate
    glycosidic: GlycosidicTemplateBinding | None
    failure_reason: str

    def __init__(
        self,
        context: PolymerMicrostateContext,
        residue_id: ResidueId,
        template: ResidueTemplate,
        component_library: ComponentLibrary,
    ) -> None:
        object.__setattr__(self, "isolated_template", template)
        object.__setattr__(self, "glycosidic", None)
        object.__setattr__(self, "failure_reason", "")
        residue = context.source.constitution.residue_or_ligand(residue_id)
        if residue is None:
            raise ValueError("fixed H preparation requires a present residue")
        if residue.component_id != "ASN" or not residue.has_atom_site("ND2"):
            return
        if not any(
            bond.relationship_type
            not in (
                BondRelationshipType.HYDROGEN_BOND,
                BondRelationshipType.METAL_COORDINATION,
            )
            for bond in context.external_heavy_bonds(AtomRef(residue_id, "ND2"))
        ):
            return
        try:
            binding = GlycosidicTemplateBinding(
                context.source, residue, template, component_library
            )
        except ValueError as error:
            object.__setattr__(self, "failure_reason", str(error))
        else:
            object.__setattr__(self, "glycosidic", binding)

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
            ND2 when linked ASN chemistry could not be selected, otherwise empty.
        """
        return frozenset(("ND2",)) if self.failure_reason else frozenset()

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
        pending = {} if requests is None else dict(requests)
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
                    entries.append(
                        PolymerSitePreparation(
                            context,
                            residue.residue_id,
                            site,
                            preferences=(
                                pras_microstate_preferences(site)[:1]
                                if template.component_id == "HIS"
                                and site.kind is PolymerChemicalSite.SIDECHAIN
                                else pras_microstate_preferences(site)
                            ),
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
        object.__setattr__(self, "_coordinate_dependent", bool(unresolved_histidines))
        object.__setattr__(self, "_targets", MappingProxyType(targets))
        object.__setattr__(self, "context", context)
        object.__setattr__(self, "component_library", component_library)
        object.__setattr__(
            self, "_disulfides", disulfide_bonded_cysteine_residue_ids(context.source)
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
            self.context, residue_id, template, self.component_library
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

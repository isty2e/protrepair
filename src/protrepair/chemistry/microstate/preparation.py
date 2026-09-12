"""PRAS-style preparation targets, separate from observed chemistry."""

from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.microstate.catalog import (
    PeptideLinkage,
    PolymerChemicalSite,
    standard_microstate_candidates,
)
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.resolution import (
    MicrostateConstraints,
    MicrostateResolution,
    MicrostateResolutionStatus,
)
from protrepair.structure.constitution import AtomSite
from protrepair.structure.labels import ResidueId


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
    override : MicrostateConstraints or None
        Explicit replacement constraints, if requested.
    retain_applied_override : bool
        False resets an earlier choice to source-compatible PRAS preferences.

    Notes
    -----
    This is a target, not proof that the structure already realizes it. Placement
    and mutation remain transformer responsibilities.
    """

    context: PolymerMicrostateContext = field(repr=False)
    residue_id: ResidueId
    site: PolymerMicrostateSite
    override: MicrostateConstraints | None = None
    retain_applied_override: bool = True
    resolution: MicrostateResolution = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "resolution",
            self.context.resolve(
                self.residue_id,
                self.site,
                override=self.override,
                preferences=pras_microstate_preferences(self.site),
                retain_applied_override=self.retain_applied_override,
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

    Raises
    ------
    ValueError
        A request targets a site absent from this polymer preparation.
    """

    _targets: Mapping[ResidueId, tuple[PolymerSitePreparation, ...]]

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
                        template.component_id,
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
                            override=pending.pop(key, None),
                            retain_applied_override=retain,
                        )
                    )
                targets[residue.residue_id] = tuple(entries)
        if pending:
            raise ValueError("microstate request targets an absent polymer site")
        object.__setattr__(self, "_targets", MappingProxyType(targets))

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
                    hydrogens=(("N", 2 if component == "PRO" else 3),)
                ),
            )
        if site.kind is PolymerChemicalSite.BACKBONE_C:
            return (MicrostateConstraints(hydrogens=(("O", 0), ("OXT", 0))),)

    return ()

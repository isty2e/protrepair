"""PRAS-style preparation preferences, separate from observed chemistry."""

from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.microstate.catalog import (
    PeptideLinkage,
    PolymerChemicalSite,
    standard_microstate_candidates,
)
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.resolution import MicrostateConstraints
from protrepair.structure.labels import ResidueId


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

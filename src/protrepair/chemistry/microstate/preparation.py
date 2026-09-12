"""PRAS-style preparation preferences, separate from observed chemistry."""

from protrepair.chemistry.microstate.catalog import PeptideLinkage, PolymerChemicalSite
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.resolution import MicrostateConstraints


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

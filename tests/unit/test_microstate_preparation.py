"""Evidence-preserving preparation defaults over coupled chemical graphs."""

from dataclasses import replace

import pytest

from protrepair.chemistry.microstate.catalog import (
    PeptideLinkage,
    PolymerChemicalSite,
    standard_microstate_candidates,
)
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.preparation import pras_microstate_preferences
from protrepair.chemistry.microstate.resolution import (
    MicrostateConstraints,
    MicrostateResolutionStatus,
    MicrostateSelectionBasis,
    MicrostateSite,
)
from protrepair.chemistry.standard.components import build_standard_component_library


def _site(
    component: str,
    kind: PolymerChemicalSite = PolymerChemicalSite.SIDECHAIN,
    linkage: PeptideLinkage = PeptideLinkage.UNKNOWN,
) -> tuple[MicrostateSite, tuple[MicrostateConstraints, ...]]:
    bound = PolymerMicrostateSite(
        build_standard_component_library().require(component), kind, linkage
    )
    catalog = standard_microstate_candidates(component, kind, linkage)
    assert catalog is not None
    return catalog, pras_microstate_preferences(bound)


@pytest.mark.parametrize(
    "component,charge,hydrogens",
    (
        ("ARG", 1, {"NE": 1, "CZ": 0, "NH1": 2, "NH2": 2}),
        ("LYS", 1, {"NZ": 3}),
        ("ASP", -1, {"CG": 0, "OD1": 0, "OD2": 0}),
        ("GLU", -1, {"CD": 0, "OE1": 0, "OE2": 0}),
        ("HIS", 0, {"CG": 0, "ND1": 0, "CE1": 1, "NE2": 1, "CD2": 1}),
    ),
)
def test_pras_defaults_select_coupled_h_and_charge(
    component: str, charge: int, hydrogens: dict[str, int]
) -> None:
    site, preferences = _site(component)
    result = site.resolve(MicrostateConstraints(), preferences=preferences)
    assert result.basis is MicrostateSelectionBasis.DEFAULT
    assert result.superseded_source.is_empty()
    assert result.graph is not None
    assert result.graph.protonation_key()[0] == charge
    assert {atom.name: atom.hydrogens for atom in result.graph.atoms} == hydrogens


@pytest.mark.parametrize("parent", ("ND1", "NE2"))
def test_partial_histidine_h_preserves_its_tautomer_but_prefers_neutral(
    parent: str,
) -> None:
    site, preferences = _site("HIS")
    evidence = MicrostateConstraints(minimum_hydrogens=((parent, 1),))
    assert site.resolve(evidence).status is MicrostateResolutionStatus.AMBIGUOUS
    result = site.resolve(evidence, preferences=preferences)
    assert result.graph is not None
    assert result.graph.protonation_key()[0] == 0
    assert result.graph.atom(parent).hydrogens == 1
    assert result.basis is MicrostateSelectionBasis.DEFAULT
    assert result.superseded_source.is_empty()


def test_both_observed_histidine_hydrogens_override_neutral_preference() -> None:
    site, preferences = _site("HIS")
    evidence = MicrostateConstraints(minimum_hydrogens=(("ND1", 1), ("NE2", 1)))
    result = site.resolve(evidence, preferences=preferences)
    assert result.graph is not None and result.graph.protonation_key()[0] == 1
    assert result.basis is MicrostateSelectionBasis.EVIDENCE
    assert len(result.candidates) == 2


@pytest.mark.parametrize(
    "component,evidence,charge",
    (
        ("LYS", MicrostateConstraints(charges=(("NZ", 0),)), 0),
        ("ARG", MicrostateConstraints(charges=(("NE", 0), ("NH1", 0), ("NH2", 0))), 0),
        ("ASP", MicrostateConstraints(minimum_hydrogens=(("OD2", 1),)), 0),
        ("GLU", MicrostateConstraints(minimum_hydrogens=(("OE1", 1),)), 0),
        ("HIS", MicrostateConstraints(charges=(("ND1", 1),)), 1),
    ),
)
def test_defaults_do_not_erase_plausible_original_chemistry(
    component: str, evidence: MicrostateConstraints, charge: int
) -> None:
    site, preferences = _site(component)
    result = site.resolve(evidence, preferences=preferences)
    assert result.candidates
    assert all(graph.protonation_key()[0] == charge for graph in result.candidates)
    assert all(evidence.violations(graph).is_empty() for graph in result.candidates)
    assert result.basis is MicrostateSelectionBasis.EVIDENCE
    assert result.superseded_source.is_empty()


def test_defaults_cannot_rescue_conflict_or_reintroduce_eliminated_candidates() -> None:
    site, preferences = _site("HIS")
    evidence = MicrostateConstraints(
        charges=(("ND1", 0), ("NE2", 0)),
        minimum_hydrogens=(("ND1", 1), ("NE2", 1)),
    )
    result = site.resolve(evidence, preferences=preferences)
    assert result.status is MicrostateResolutionStatus.CONFLICT

    # The second preference conflicts with the first; it must not restart from
    # the catalog and silently choose the cation.
    preferences = (
        MicrostateConstraints(charges=(("ND1", 0), ("NE2", 0))),
        MicrostateConstraints(hydrogens=(("ND1", 1), ("NE2", 1))),
    )
    result = site.resolve(MicrostateConstraints(), preferences=preferences)
    assert result.status is MicrostateResolutionStatus.AMBIGUOUS
    assert all(graph.protonation_key()[0] == 0 for graph in result.candidates)


def test_explicit_ring_override_bypasses_neutral_defaults_and_reports_source() -> None:
    site, preferences = _site("HIS")
    source = MicrostateConstraints(charges=(("ND1", 0), ("NE2", 0)))
    result = site.resolve(
        source,
        preferences=preferences,
        override=MicrostateConstraints(hydrogens=(("ND1", 1), ("NE2", 1))),
    )
    assert result.graph is not None and result.graph.protonation_key()[0] == 1
    assert result.basis is MicrostateSelectionBasis.OVERRIDE
    assert not result.superseded_source.is_empty()


@pytest.mark.parametrize("component", ("ALA", "PRO"))
@pytest.mark.parametrize(
    "kind", (PolymerChemicalSite.BACKBONE_N, PolymerChemicalSite.BACKBONE_C)
)
@pytest.mark.parametrize("linkage", (PeptideLinkage.LINKED, PeptideLinkage.FREE))
def test_terminal_policy_does_not_infer_linkage(
    component: str, kind: PolymerChemicalSite, linkage: PeptideLinkage
) -> None:
    site, preferences = _site(component, kind, linkage)
    result = site.resolve(MicrostateConstraints(), preferences=preferences)
    assert result.graph is not None
    expected_charge = (
        0
        if linkage is PeptideLinkage.LINKED
        else (1 if kind is PolymerChemicalSite.BACKBONE_N else -1)
    )
    assert result.graph.protonation_key()[0] == expected_charge
    unknown = PolymerMicrostateSite(
        build_standard_component_library().require(component), kind
    )
    assert pras_microstate_preferences(unknown) == ()


def test_other_preferences_use_the_resolver_without_source_override() -> None:
    site, _ = _site("LYS")
    neutral_preference = MicrostateConstraints(hydrogens=(("NZ", 2),))
    result = site.resolve(MicrostateConstraints(), preferences=(neutral_preference,))
    assert result.graph is not None and result.graph.atom("NZ").charge == 0
    source = replace(neutral_preference, hydrogens=(), charges=(("NZ", 1),))
    result = site.resolve(source, preferences=(neutral_preference,))
    assert result.graph is not None and result.graph.atom("NZ").charge == 1
    assert result.basis is MicrostateSelectionBasis.EVIDENCE

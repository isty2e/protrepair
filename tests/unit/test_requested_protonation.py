"""State requests constrain protonation classes, not resonance drawings."""

import pytest

from protrepair.chemistry.microstate.catalog import (
    PeptideLinkage,
    PolymerChemicalSite,
    standard_microstate_candidates,
)
from protrepair.chemistry.microstate.resolution import (
    AppliedMicrostateChoice,
    MicrostateConstraints,
    MicrostateResolutionStatus,
    MicrostateSelectionBasis,
    MicrostateSite,
)
from protrepair.chemistry.standard.components import STANDARD_COMPONENT_ATOMS
from protrepair.structure.labels import ResidueId


@pytest.mark.parametrize(
    "component,kind,linkage,charge,expected_classes,drawings",
    (
        ("ASP", "sidechain", "unknown", 0, 2, 2),
        ("GLU", "sidechain", "unknown", -1, 1, 2),
        ("ARG", "sidechain", "unknown", 0, 3, 3),
        ("ARG", "sidechain", "unknown", 1, 1, 3),
        ("HIS", "sidechain", "unknown", 0, 2, 2),
        ("HIS", "sidechain", "unknown", 1, 1, 2),
        ("LYS", "sidechain", "unknown", 0, 1, 1),
        ("CYS", "sidechain", "unknown", -1, 1, 1),
        ("PRO", "backbone_n", "free", 1, 1, 1),
        ("PRO", "backbone_n", "linked", 0, 1, 1),
        ("ALA", "backbone_c", "free", 0, 2, 2),
        ("ALA", "backbone_c", "linked", 0, 1, 1),
    ),
)
def test_charge_request_counts_protonation_classes(
    component, kind, linkage, charge, expected_classes, drawings
):
    site = standard_microstate_candidates(
        component, PolymerChemicalSite(kind), PeptideLinkage(linkage)
    )
    assert site is not None
    result = site.resolve_protonation(MicrostateConstraints(), site_charge=charge)
    keys = {graph.protonation_key() for graph in result.candidates}
    assert len(keys) == expected_classes
    assert len(result.candidates) == drawings
    assert {level for level, _ in keys} == {charge}
    assert (result.graph is not None) is (expected_classes == 1)
    assert result.status is (
        MicrostateResolutionStatus.RESOLVED
        if expected_classes == 1
        else MicrostateResolutionStatus.AMBIGUOUS
    )


@pytest.mark.parametrize("component", sorted(STANDARD_COMPONENT_ATOMS))
def test_fully_specified_requests_identify_every_catalog_class_in_either_order(
    component,
):
    for kind in PolymerChemicalSite:
        linkages = (
            (PeptideLinkage.UNKNOWN,)
            if kind is PolymerChemicalSite.SIDECHAIN
            else tuple(PeptideLinkage)
        )
        for linkage in linkages:
            site = standard_microstate_candidates(component, kind, linkage)
            if site is None:
                continue
            classes = {}
            for graph in site.candidates:
                charge = sum(atom.charge for atom in graph.atoms)
                parents = tuple((atom.name, atom.hydrogens) for atom in graph.atoms)
                classes.setdefault((charge, parents), set()).add(graph)
            reversed_site = MicrostateSite(site.candidates[::-1])
            for (charge, parents), drawings in classes.items():
                for trial in (site, reversed_site):
                    result = trial.resolve_protonation(
                        MicrostateConstraints(),
                        site_charge=charge,
                        parent_hydrogens=parents,
                    )
                    assert result.status is MicrostateResolutionStatus.RESOLVED
                    assert set(result.candidates) == drawings


def test_source_and_request_must_select_the_same_acid_parent():
    site = standard_microstate_candidates("ASP", PolymerChemicalSite.SIDECHAIN)
    assert site is not None
    source = MicrostateConstraints(minimum_hydrogens=(("OD2", 1),))
    result = site.resolve_protonation(source, site_charge=0)
    assert result.graph is not None and result.graph.atom("OD2").hydrogens == 1
    conflict = site.resolve_protonation(
        source, site_charge=0, parent_hydrogens=(("OD1", 1),)
    )
    assert conflict.status is MicrostateResolutionStatus.CONFLICT
    assert conflict.graph is None and conflict.superseded_source.is_empty()


def test_source_can_pin_resonance_but_request_cannot_bypass_a_donor_requirement():
    site = standard_microstate_candidates("HIS", PolymerChemicalSite.SIDECHAIN)
    assert site is not None
    source = MicrostateConstraints(charges=(("ND1", 0), ("NE2", 1)))
    result = site.resolve_protonation(source, site_charge=1)
    assert result.graph is not None and result.graph.atom("NE2").charge == 1
    blocked = site.resolve_protonation(
        source,
        site_charge=1,
        requirements=MicrostateConstraints(hydrogens=(("NE2", 0),)),
    )
    assert blocked.status is MicrostateResolutionStatus.CONFLICT
    assert blocked.graph is None
    assert site.resolve_protonation(source, site_charge=-1).status is (
        MicrostateResolutionStatus.UNSUPPORTED
    )


def test_partial_parent_count_narrows_the_site_without_replacing_source():
    site = standard_microstate_candidates("HIS", PolymerChemicalSite.SIDECHAIN)
    assert site is not None
    selected = site.resolve_protonation(
        MicrostateConstraints(), site_charge=0, parent_hydrogens=(("ND1", 0),)
    )
    assert selected.basis is MicrostateSelectionBasis.REQUESTED
    assert selected.graph is not None and selected.graph.atom("NE2").hydrogens == 1
    assert selected.superseded_source.is_empty()
    blocked = site.resolve_protonation(
        MicrostateConstraints(minimum_hydrogens=(("ND1", 1),)),
        site_charge=0,
        parent_hydrogens=(("ND1", 0),),
    )
    assert blocked.status is MicrostateResolutionStatus.CONFLICT
    assert blocked.graph is None and blocked.superseded_source.is_empty()


def test_assessed_resonance_drawings_do_not_become_a_saved_choice():
    site = standard_microstate_candidates("HIS", PolymerChemicalSite.SIDECHAIN)
    assert site is not None
    resolved = site.resolve_protonation(MicrostateConstraints(), site_charge=1)
    first, alternative = resolved.candidates
    assert (
        first != alternative
        and first.protonation_key() == alternative.protonation_key()
    )
    for drawing in (first, alternative):
        source = MicrostateConstraints(
            charges=tuple((a.name, a.charge) for a in drawing.atoms)
        )
        assert site.resolve_protonation(source, site_charge=1).graph == drawing
    with pytest.raises(ValueError, match="not an applied choice"):
        AppliedMicrostateChoice(ResidueId("A", 5), "HIS", first, resolved.basis)


def test_partial_arginine_request_can_still_be_ambiguous():
    site = standard_microstate_candidates("ARG", PolymerChemicalSite.SIDECHAIN)
    assert site is not None
    source = MicrostateConstraints()
    still_open = site.resolve_protonation(
        source, site_charge=0, parent_hydrogens=(("NE", 1),)
    )
    assert still_open.graph is None
    assert len({g.protonation_key() for g in still_open.candidates}) == 2
    fixed = site.resolve_protonation(
        source, site_charge=0, parent_hydrogens=(("NE", 0),)
    )
    assert fixed.graph is not None and fixed.graph.atom("NE").hydrogens == 0


def test_physical_constraint_can_narrow_a_charge_only_request():
    site = standard_microstate_candidates("HIS", PolymerChemicalSite.SIDECHAIN)
    assert site is not None
    result = site.resolve_protonation(
        MicrostateConstraints(),
        site_charge=0,
        requirements=MicrostateConstraints(hydrogens=(("NE2", 0),)),
    )
    assert result.graph is not None and result.graph.atom("ND1").hydrogens == 1


@pytest.mark.parametrize(
    "charge,counts,error",
    (
        (True, (), TypeError),
        (0, (("ND1", True),), TypeError),
        (0, (("ND1", -1),), ValueError),
        (0, (("ND1", 1), ("ND1", 0)), ValueError),
    ),
)
def test_malformed_requests_are_not_chemical_refusals(charge, counts, error):
    site = standard_microstate_candidates("HIS", PolymerChemicalSite.SIDECHAIN)
    assert site is not None
    with pytest.raises(error):
        site.resolve_protonation(
            MicrostateConstraints(), site_charge=charge, parent_hydrogens=counts
        )

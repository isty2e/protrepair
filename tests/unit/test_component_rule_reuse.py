"""Immutable component rules are reused without sharing caller-specific facts."""

import pickle
from collections.abc import MutableMapping
from dataclasses import replace
from typing import cast
from unittest.mock import patch

import pytest

from protrepair.chemistry import HydrogenSemantics, build_default_component_library
from protrepair.chemistry.component.semantics import IdealGeometryHydrogenSemantics
from protrepair.chemistry.microstate.catalog import (
    PeptideLinkage,
    PolymerChemicalSite,
    standard_microstate_candidates,
)
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.structure.constitution import AtomSite, ResidueSite
from protrepair.structure.labels import ResidueId


def _semantics(parent: str) -> HydrogenSemantics:
    return HydrogenSemantics(
        plan_with_backbone=((("HX",), "class4", ("C", "CA", parent, 1.0)),)
    )


def test_anchor_projection_is_immutable_and_compiled_once():
    semantics = _semantics("CB")
    cold = replace(semantics)
    interpret = HydrogenSemantics.anchor_atom_name_for_operation
    with patch.object(
        HydrogenSemantics, "anchor_atom_name_for_operation", wraps=interpret
    ) as operation:
        first = semantics.template_hydrogen_anchor_by_name()
        assert dict(first) == {"HX": "CB"}
        assert semantics.template_hydrogen_anchor_by_name() == first
    assert operation.call_count == 1
    assert semantics == cold and hash(semantics) == hash(cold)
    with pytest.raises(TypeError):
        cast(MutableMapping[str, str], first)["HX"] = "CA"
    changed = replace(semantics, plan_with_backbone=_semantics("CA").plan_with_backbone)
    assert dict(changed.template_hydrogen_anchor_by_name()) == {"HX": "CA"}
    assert dict(first) == {"HX": "CB"}


def test_conflicting_full_grammar_never_publishes_a_partial_projection():
    semantics = replace(
        _semantics("CB"), plan_without_backbone=_semantics("CA").plan_with_backbone
    )
    for _ in range(2):
        with pytest.raises(ValueError, match="conflicting anchor atoms"):
            semantics.template_hydrogen_anchor_by_name()


def test_idealized_anchor_projection_preserves_complete_inventory():
    semantics = build_default_component_library().require("FMN").hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    cold = replace(semantics)
    expected = {
        atom.atom_name: anchor
        for atom in cold.component.hydrogen_atoms()
        if (anchor := cold.component.hydrogen_anchor_atom_name(atom.atom_name))
        is not None
    }
    first = cold.template_hydrogen_anchor_by_name()
    assert first == expected
    assert cold.template_hydrogen_anchor_by_name() == first
    assert cold == replace(cold)


@pytest.mark.parametrize("warm", (False, True))
def test_standalone_semantics_preserve_pickle_roundtrip(warm):
    semantics = _semantics("CB")
    if warm:
        semantics.template_hydrogen_anchor_by_name()
    restored = pickle.loads(pickle.dumps(semantics))
    assert restored == semantics
    assert hash(restored) == hash(semantics)
    assert dict(restored.template_hydrogen_anchor_by_name()) == {"HX": "CB"}


def test_same_id_custom_templates_have_independent_naming_rules():
    standard = build_default_component_library().require("ALA")
    first = replace(standard, hydrogen_semantics=_semantics("CB"))
    second = replace(standard, hydrogen_semantics=_semantics("CA"))
    assert first.component_id == second.component_id
    warm = first.polymer_hydrogen_parent_aliases()
    assert warm["HX"] == "CB"
    assert second.polymer_hydrogen_parent_aliases()["HX"] == "CA"
    assert first.polymer_hydrogen_parent_aliases() is warm
    replaced = replace(first, hydrogen_semantics=second.hydrogen_semantics)
    assert replaced.polymer_hydrogen_parent_aliases()["HX"] == "CA"
    assert first == replace(first)
    assert dict(first.template_hydrogen_anchor_by_name((" hx ",))) == {"HX": "CB"}
    assert dict(first.template_hydrogen_anchor_by_name(("OTHER",))) == {}


@pytest.mark.parametrize(
    "component", ("ALA", "HIS", "ARG", "ASN", "GLN", "ASP", "GLU", "ILE")
)
@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_observed_isotope_names_remain_live_after_template_warming(component, isotope):
    template = build_default_component_library().require(component)
    aliases = template.polymer_hydrogen_parent_aliases()
    site = PolymerMicrostateSite(template, PolymerChemicalSite.SIDECHAIN)
    named = {}
    atoms = []
    for name, parent in aliases.items():
        offset = 1 if name[0].isdigit() else 0
        if name[offset : offset + 1] != "H":
            continue
        observed = name[:offset] + isotope + name[offset + 1 :]
        atoms.append(AtomSite(observed, isotope))
        named[observed] = parent
    residue = ResidueSite(component, ResidueId("A", 1), tuple(atoms))
    assert site.named_hydrogen_parents(residue) == named
    assert site.named_hydrogen_parents(replace(residue, atom_sites=())) == {}
    linked = PolymerMicrostateSite(
        template, PolymerChemicalSite.BACKBONE_N, PeptideLinkage.LINKED
    )
    free = replace(linked, linkage=PeptideLinkage.FREE)
    assert linked.hydrogen_name_candidates("N") == ("H",)
    assert free.hydrogen_name_candidates("N") == ("H1", "H2", "H3")


def test_standard_catalog_reuses_immutable_graphs_without_skipping_validation():
    first = standard_microstate_candidates("HIS", PolymerChemicalSite.SIDECHAIN)
    assert first is not None
    assert standard_microstate_candidates("HIS", PolymerChemicalSite.SIDECHAIN) is first
    with pytest.raises(TypeError, match="canonical enum"):
        standard_microstate_candidates("HIS", cast(PolymerChemicalSite, "sidechain"))
    with pytest.raises(TypeError, match="canonical enum"):
        standard_microstate_candidates(
            "HIS", PolymerChemicalSite.SIDECHAIN, cast(PeptideLinkage, "unknown")
        )
    with pytest.raises(ValueError, match="backbone linkage"):
        standard_microstate_candidates(
            "HIS", PolymerChemicalSite.SIDECHAIN, PeptideLinkage.LINKED
        )

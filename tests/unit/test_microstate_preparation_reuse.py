"""Placement reuse is narrower than chemistry-count applicability."""

from dataclasses import replace
from unittest.mock import patch

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_bonded_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.preparation import PolymerMicrostatePreparation
from protrepair.chemistry.microstate.resolution import MicrostateConstraints
from protrepair.geometry import Vec3
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat


@pytest.fixture
def source() -> ProteinStructure:
    return build_bonded_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0, 0, 0)),
                            atom_payload("CA", "C", Vec3(1.45, 0, 0)),
                            atom_payload("C", "C", Vec3(2.1, 1.3, 0)),
                            atom_payload("O", "O", Vec3(1.8, 2.4, 0)),
                            atom_payload("OXT", "O", Vec3(3.35, 1.2, 0)),
                            atom_payload("CB", "C", Vec3(1.9, -0.8, 1.2)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )


def test_exact_binding_reuses_preparation_and_normalizes_empty_request(source):
    library = build_default_component_library()
    prepared = PolymerMicrostatePreparation(PolymerMicrostateContext(source), library)
    assert prepared.for_inputs(source, library) is prepared
    assert prepared.for_inputs(source, library, requests={}) is prepared


@pytest.mark.parametrize("change", ("equal_copy", "geometry", "provenance", "library"))
def test_different_bindings_rebuild_even_when_count_chemistry_matches(source, change):
    library = build_default_component_library()
    prepared = PolymerMicrostatePreparation(PolymerMicrostateContext(source), library)
    current = source.with_provenance(source.provenance)
    if change == "geometry":
        index = source.constitution.residue_index(ResidueId("A", 1))
        atoms = dict(source.residue_geometry(index).atoms_by_name)
        atoms["N"] = atoms["N"].with_position(Vec3(0.1, 0, 0))
        current = source.with_updated_residue_geometries(
            ((ResidueId("A", 1), ResidueGeometry(atoms)),)
        )
    elif change == "provenance":
        current = source.with_provenance(replace(source.provenance))
    elif change == "library":
        current = source
        library = replace(library)
    assert prepared.matches_chemistry(current)
    rebound = prepared.for_inputs(current, library)
    assert rebound is not prepared
    assert rebound.context.source is current
    assert rebound.component_library is library
    fresh = PolymerMicrostatePreparation(PolymerMicrostateContext(current), library)
    assert tuple(
        (target.site.kind, target.resolution)
        for target in rebound.targets_for(ResidueId("A", 1))
    ) == tuple(
        (target.site.kind, target.resolution)
        for target in fresh.targets_for(ResidueId("A", 1))
    )


def test_requests_are_defensively_copied_and_reset_is_not_absence(source):
    library = build_default_component_library()
    key = (ResidueId("A", 1), PolymerChemicalSite.BACKBONE_N)
    neutral = MicrostateConstraints(charges=(("N", 0),))
    requests: dict[
        tuple[ResidueId, PolymerChemicalSite], MicrostateConstraints | None
    ] = {key: neutral}
    prepared = PolymerMicrostatePreparation(
        PolymerMicrostateContext(source), library, requests=requests
    )
    requests.clear()
    assert prepared.for_inputs(source, library, requests={key: neutral}) is prepared
    absent = prepared.for_inputs(source, library)
    reset = prepared.for_inputs(source, library, requests={key: None})
    assert absent is not prepared and reset is not absent
    assert reset.for_inputs(source, library, requests={key: None}) is reset
    assert reset.for_inputs(source, library) is not reset
    neutral_target = next(
        t for t in prepared.targets_for(key[0]) if t.site.kind is key[1]
    )
    reset_target = next(t for t in reset.targets_for(key[0]) if t.site.kind is key[1])
    assert neutral_target.resolution.graph is not None
    assert reset_target.resolution.graph is not None
    assert neutral_target.resolution.graph.atom("N").charge == 0
    assert reset_target.resolution.graph.atom("N").charge == 1


def test_rebinding_keeps_absent_site_request_validation(source):
    library = build_default_component_library()
    prepared = PolymerMicrostatePreparation(PolymerMicrostateContext(source), library)
    for _ in range(2):
        with pytest.raises(ValueError, match="absent polymer site"):
            prepared.for_inputs(
                source,
                library,
                requests={
                    (ResidueId("A", 99), PolymerChemicalSite.BACKBONE_N): None,
                },
            )


def test_attachment_reuse_keeps_empty_results_private_and_keys_exact_objects(source):
    library = build_default_component_library()
    context = PolymerMicrostateContext(source)
    prepared = PolymerMicrostatePreparation(context, library)
    target = prepared.targets_for(ResidueId("A", 1))[0]
    original = PolymerMicrostateContext.coordination_requirements
    with patch.object(
        PolymerMicrostateContext,
        "coordination_requirements",
        autospec=True,
        side_effect=original,
    ) as validate:
        first = context.hydrogen_attachments(
            target.residue_id, target.site, target.resolution
        )
        assert first == {}
        first["POISON"] = "N"
        assert (
            context.hydrogen_attachments(
                target.residue_id, target.site, target.resolution
            )
            == {}
        )
        assert validate.call_count == 1
        context.hydrogen_attachments(
            target.residue_id, replace(target.site), target.resolution
        )
        context.hydrogen_attachments(
            target.residue_id, target.site, replace(target.resolution)
        )
        assert validate.call_count == 3
        for _ in range(2):
            context.hydrogen_attachments(
                target.residue_id, target.site, target.resolution, proposed_parents={}
            )
        assert validate.call_count == 5
        with pytest.raises(ValueError, match="parent is outside"):
            context.hydrogen_attachments(
                target.residue_id,
                target.site,
                target.resolution,
                proposed_parents={"HX": "OUTSIDE"},
            )
        assert validate.call_count == 6
        second_context = PolymerMicrostateContext(source)
        assert (
            second_context.hydrogen_attachments(
                target.residue_id, target.site, target.resolution
            )
            == {}
        )
        assert validate.call_count == 7


def test_attachment_failures_never_become_successful_cached_results(source):
    library = build_default_component_library()
    context = PolymerMicrostateContext(source)
    target = PolymerMicrostatePreparation(context, library).targets_for(
        ResidueId("A", 1)
    )[0]
    missing = source.without_atom_refs((AtomRef(ResidueId("A", 1), "N"),))
    context = PolymerMicrostateContext(missing)
    for _ in range(2):
        with pytest.raises(ValueError, match="scaffold is incomplete"):
            context.hydrogen_attachments(
                target.residue_id, target.site, target.resolution
            )

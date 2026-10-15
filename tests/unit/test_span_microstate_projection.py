"""Donor sidechain chemistry survives H-free grafting and identity remapping."""

from dataclasses import replace
from pathlib import Path

import pytest

from protrepair.api import process_structure
from protrepair.chemistry import (
    build_default_component_library,
    build_default_restraint_library,
)
from protrepair.chemistry.microstate.catalog import (
    PolymerChemicalSite,
    standard_microstate_candidates,
)
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.resolution import (
    AppliedMicrostateChoice,
    MicrostateSelectionBasis,
)
from protrepair.chemistry.restraint.structure import StructureRestraints
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.errors import ModelInvariantError
from protrepair.io import FileFormat, read_structure_string, write_structure_string
from protrepair.scope import AbsentResidueSpanScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.hydrogen.repair import add_hydrogens
from protrepair.transformer.completion.span_reconstruction import (
    ReconstructedSpanCandidate,
    SpanChemistryConflictError,
    reconstruct_donor_span,
)
from protrepair.workflow.contracts import (
    ExternalSpanReconstructionSpec,
    WorkflowTransformRequests,
)

FIXTURE = Path(__file__).parents[1] / "fixtures/pdb/refinement/1bkr_his42_local.pdb"
TARGET = ResidueId("A", 42)
DONOR = ResidueId("X", 142)
SCOPE = AbsentResidueSpanScope(ResidueId("A", 41), ResidueId("A", 43), (TARGET,))


def _inputs(
    counts, *, resonance=0, remove_h=False, basis=MicrostateSelectionBasis.DEFAULT
):
    records = [
        line
        for line in FIXTURE.read_text().splitlines()
        if line.startswith("ATOM") and int(line[22:26]) in (41, 42, 43)
    ]
    source = read_structure_string(
        "\n".join(line for line in records if int(line[22:26]) != 42) + "\nEND\n",
        FileFormat.PDB,
    )
    donor = read_structure_string(
        "\n".join(
            line[:21] + f"X{int(line[22:26]) + 100:4d}" + line[26:] for line in records
        )
        + "\nEND\n",
        FileFormat.PDB,
    )
    catalog = standard_microstate_candidates("HIS", PolymerChemicalSite.SIDECHAIN)
    assert catalog is not None
    graph = [
        g
        for g in catalog.candidates
        if tuple(g.atom(n).hydrogens for n in ("ND1", "NE2")) == counts
    ][resonance]
    donor = donor.with_provenance(
        replace(
            donor.provenance,
            microstate_choices=(AppliedMicrostateChoice(DONOR, "HIS", graph, basis),),
        )
    )
    donor = add_hydrogens(donor).structure
    if remove_h:
        donor = donor.without_hydrogens()
    return source, donor


def _candidate(source, donor):
    result = reconstruct_donor_span(
        source,
        scope=SCOPE,
        donor_structure=donor,
        donor_residue_ids=(DONOR,),
        donor_preceding_residue_id=ResidueId("X", 141),
        donor_following_residue_id=ResidueId("X", 143),
    )
    assert isinstance(result, ReconstructedSpanCandidate), result
    return result


@pytest.mark.parametrize("counts", ((1, 0), (0, 1), (1, 1)))
def test_public_graft_retains_realized_donor_heavy_chemistry(counts):
    source, donor = _inputs(counts, remove_h=True)
    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(
                ExternalSpanReconstructionSpec(
                    scope=SCOPE,
                    donor_structure=donor,
                    donor_residue_ids=(DONOR,),
                ),
            ),
        ),
    )
    assert not result.has_errors(), result.issues
    site = PolymerMicrostateSite(
        build_default_component_library().require("HIS"),
        PolymerChemicalSite.SIDECHAIN,
    )
    context = PolymerMicrostateContext(result.structure)
    resolution = context.resolve(TARGET, site)
    graph = context.realized_heavy_graph(TARGET, site, resolution)
    assert graph is not None
    assert tuple(graph.atom(name).hydrogens for name in ("ND1", "NE2")) == counts
    assert source.constitution.residue_or_ligand(TARGET) is None


@pytest.mark.parametrize(
    "counts,resonance", (((1, 0), 0), ((0, 1), 0), ((1, 1), 0), ((1, 1), 1))
)
@pytest.mark.parametrize("remove_h", (False, True))
def test_graft_remaps_sidechain_graph_and_preserves_source(counts, resonance, remove_h):
    source, donor = _inputs(counts, resonance=resonance, remove_h=remove_h)
    candidate = _candidate(source, donor)
    assert len(candidate.sidechain_choices) == 1
    choice = candidate.sidechain_choices[0]
    assert choice.residue_id == TARGET
    assert choice.basis is MicrostateSelectionBasis.DEFAULT
    assert tuple(choice.graph.atom(n).hydrogens for n in ("ND1", "NE2")) == counts
    result = candidate.materialize(source, build_default_component_library())

    inserted = result.constitution.residue_or_ligand(TARGET)
    assert inserted is not None
    assert not any(atom.is_hydrogen() for atom in inserted.atom_sites)
    assert result.provenance.ingress is source.provenance.ingress
    assert result.provenance.lineage == source.provenance.lineage
    assert result.provenance.microstate_choices == (choice,)
    for before, _charge in source.topology.formal_charge_entries():
        ref = source.constitution.atom_ref_at(before)
        after = result.constitution.atom_index(ref)
        assert result.geometry.atom_geometry(after) == source.geometry.atom_geometry(
            before
        )
        assert result.topology.formal_charge(after) == source.topology.formal_charge(
            before
        )
    for bond in source.topology.bonds:
        refs = tuple(source.constitution.atom_ref_at(i) for i in bond.endpoint_pair())
        actual = result.topology.bond_between(
            *(result.constitution.atom_index(ref) for ref in refs)
        )
        assert actual is not None
        assert (actual.order, actual.relationship_type, actual.provenance) == (
            bond.order,
            bond.relationship_type,
            bond.provenance,
        )

    site = PolymerMicrostateSite(
        build_default_component_library().require("HIS"), PolymerChemicalSite.SIDECHAIN
    )
    context = PolymerMicrostateContext(result)
    resolution = context.resolve(TARGET, site)
    assert context.realized_heavy_graph(TARGET, site, resolution) == choice.graph
    assert not context.is_realized(TARGET, site, resolution)
    template = StructureRestraints(
        result, build_default_component_library(), build_default_restraint_library()
    ).for_residue(TARGET)
    assert template is not None
    target = next(
        b
        for b in template.bond_targets
        if {b.atom_name_1, b.atom_name_2} == {"CE1", "NE2"}
    )
    assert target.target_distance_angstrom == (1.335 if counts == (0, 1) else 1.316)

    hydrogenated = add_hydrogens(
        result, target_residue_ids=frozenset((TARGET,))
    ).structure
    context = PolymerMicrostateContext(hydrogenated)
    resolution = context.resolve(TARGET, site)
    assert resolution.graph is not None
    assert resolution.graph.protonation_key() == choice.graph.protonation_key()
    assert context.is_realized(TARGET, site, resolution)


def test_donor_override_is_adopted_preference_not_transferred_authority():
    source, donor = _inputs((1, 0), basis=MicrostateSelectionBasis.OVERRIDE)
    candidate = _candidate(source, donor)
    assert candidate.sidechain_choices[0].basis is MicrostateSelectionBasis.DEFAULT
    assert not any(
        atom.name in {"N", "C", "O", "OXT"}
        for choice in candidate.sidechain_choices
        for atom in choice.graph.atoms
    )
    assert any(
        a.name == "N"
        for c in donor.provenance.microstate_choices
        for a in c.graph.atoms
    )


def test_existing_source_override_cannot_be_replaced_by_donor_default():
    source, donor = _inputs((0, 1))
    candidate = _candidate(source, donor)
    catalog = standard_microstate_candidates("HIS", PolymerChemicalSite.SIDECHAIN)
    assert catalog is not None
    graph = next(
        g
        for g in catalog.candidates
        if g.atom("ND1").hydrogens == 1 and g.atom("NE2").hydrogens == 0
    )
    source = source.with_provenance(
        replace(
            source.provenance,
            microstate_choices=(
                AppliedMicrostateChoice(
                    TARGET, "HIS", graph, MicrostateSelectionBasis.OVERRIDE
                ),
            ),
        )
    )
    with pytest.raises(SpanChemistryConflictError, match="conflicts with the source"):
        candidate.materialize(source, build_default_component_library())

    result = process_structure(
        source,
        transform_requests=WorkflowTransformRequests(
            external_span_reconstructions=(
                ExternalSpanReconstructionSpec(
                    scope=SCOPE, donor_structure=donor, donor_residue_ids=(DONOR,)
                ),
            ),
        ),
    )
    assert result.structure.constitution == source.constitution
    assert result.structure.geometry == source.geometry
    assert result.structure.topology == source.topology
    assert (
        result.structure.provenance.microstate_choices
        == source.provenance.microstate_choices
    )
    assert any(
        issue.kind is ValidationIssueKind.SPAN_RECONSTRUCTION_FAILED
        and "conflicts with the source" in issue.message
        for issue in result.issues
    )
    assert source.constitution.residue_or_ligand(TARGET) is None
    assert source.provenance.microstate_choices[0].graph == graph


@pytest.mark.parametrize(
    "operation", ("with_inserted_chain_residue_facets", "with_provenance")
)
@pytest.mark.parametrize("error_type", (ValueError, ModelInvariantError))
def test_public_graft_does_not_hide_internal_materialization_errors(
    monkeypatch, operation, error_type
):
    source, donor = _inputs((0, 1))
    original = getattr(ProteinStructure, operation)
    reached = []

    def fail_materialization(structure, *args, **kwargs):
        if operation == "with_inserted_chain_residue_facets" or (
            structure.constitution.residue_or_ligand(TARGET) is not None
        ):
            reached.append(operation)
            raise error_type("internal materialization invariant")
        return original(structure, *args, **kwargs)

    monkeypatch.setattr(ProteinStructure, operation, fail_materialization)
    with pytest.raises(error_type, match="internal materialization invariant"):
        process_structure(
            source,
            transform_requests=WorkflowTransformRequests(
                external_span_reconstructions=(
                    ExternalSpanReconstructionSpec(
                        scope=SCOPE, donor_structure=donor, donor_residue_ids=(DONOR,)
                    ),
                ),
            ),
        )
    assert reached == [operation]
    assert source.constitution.residue_or_ligand(TARGET) is None


def test_repeated_insertion_fails_without_changing_first_result():
    source, donor = _inputs((0, 1))
    candidate = _candidate(source, donor)
    result = candidate.materialize(source, build_default_component_library())
    with pytest.raises(ValueError, match="absent residues"):
        candidate.materialize(result, build_default_component_library())
    assert result.provenance.microstate_choices == candidate.sidechain_choices


def test_serialized_h_free_donor_recovers_choice_from_explicit_chemistry():
    source, donor = _inputs((1, 0), remove_h=True)
    donor = read_structure_string(
        write_structure_string(donor, FileFormat.MMCIF), FileFormat.MMCIF
    )
    assert not donor.provenance.microstate_choices
    candidate = _candidate(source, donor)
    result = candidate.materialize(source, build_default_component_library())
    assert result.provenance.microstate_choices[0].graph.atom("ND1").hydrogens == 1
    assert result.provenance.microstate_choices[0].graph.atom("NE2").hydrogens == 0


def test_candidate_rejects_transferred_override_or_mismatched_heavy_charge():
    source, donor = _inputs((1, 1))
    candidate = _candidate(source, donor)
    choice = candidate.sidechain_choices[0]
    with pytest.raises(ValueError, match="distinct inserted sidechains"):
        replace(
            candidate,
            sidechain_choices=(
                replace(choice, basis=MicrostateSelectionBasis.OVERRIDE),
            ),
        )
    payload = candidate.residue_payloads[0]
    with pytest.raises(ValueError, match="heavy-atom payload"):
        replace(
            candidate,
            residue_payloads=(replace(payload, formal_charge_by_atom_name=()),),
        )


def test_donor_mapping_cardinality_is_checked_before_candidate_replacement():
    source, donor = _inputs((0, 1))
    candidate = _candidate(source, donor)
    with pytest.raises(ValueError):
        candidate.with_donor_chemistry(donor, (), build_default_component_library())
    assert len(candidate.sidechain_choices) == 1

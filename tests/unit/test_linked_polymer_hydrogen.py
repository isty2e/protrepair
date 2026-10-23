"""Fixed polymer H inventory at unexplained external relationships."""

from dataclasses import replace

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_bonded_structure,
    chain_payload,
    residue_payload,
)
from tests.support.request_builders import whole_structure_requested_goals

from protrepair.api import process_structure
from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.component.semantics import HydrogenSemantics
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.preparation import PolymerMicrostatePreparation
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.scope import AtomSetScope
from protrepair.state import HydrogenApplicabilityState, HydrogenCoverageState
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.structure import AtomRef, ProteinStructure, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.structure.topology import (
    AtomTopology,
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.hydrogen import (
    add_hydrogens,
    cleanup_residual_hydrogen_clashes,
)
from protrepair.transformer.completion.hydrogen.core import materialize_hydrogens_core
from protrepair.transformer.completion.hydrogen.topology import (
    resolve_fixed_hydrogen_topology,
)
from protrepair.transformer.continuous.binding import (
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.readiness import (
    ContinuousRelaxationReadinessPolicy,
    derive_atom_scope_continuous_relaxation_facts,
)
from protrepair.transformer.local import LocalScopeSpec
from protrepair.transformer.refinement.spec import RepairRefinementSpec
from protrepair.workflow.contracts import LigandPolicy, StructureIngressOptions
from protrepair.workflow.contracts.request import (
    RequestedGoalSet,
    WorkflowTransformRequests,
)
from protrepair.workflow.planning.completion.transformer_candidates import (
    plan_hydrogen_completion_transformers,
)

SERINE = ResidueId("A", 1)
PARTNER = ResidueId("L", 9)


def linked_serine(
    relationship: BondRelationshipType = BondRelationshipType.COVALENT,
    *,
    element: str = "C",
    isotope: str | None = None,
    parent: str = "OG",
    component: str = "SER",
    missing: frozenset[str] = frozenset(),
) -> ProteinStructure:
    partner_name = "MG" if element == "MG" else "X"
    atoms = (
        atom_payload("N", "N", Vec3(-1.2, 0.1, 0.0)),
        atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
        atom_payload("C", "C", Vec3(0.6, -1.3, 0.1)),
        atom_payload("O", "O", Vec3(0.1, -2.3, 0.5)),
        atom_payload("OXT", "O", Vec3(1.9, -1.3, -0.3)),
        atom_payload("CB", "C", Vec3(0.5, 1.4, 0.3)),
        atom_payload("OG", "O", Vec3(1.8, 1.7, 0.1)),
    )
    if isotope is not None:
        atoms += (atom_payload("HG", isotope, Vec3(2.1, 2.5, 0.5), b_factor=23.0),)
    atoms = tuple(atom for atom in atoms if atom[0].name not in missing)
    structure = build_bonded_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id=component, residue_id=SERINE, atoms=atoms
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="MG" if element == "MG" else "LIG",
                residue_id=PARTNER,
                is_hetero=True,
                atoms=(atom_payload(partner_name, element, Vec3(2.2, 2.6, 0.4)),),
            ),
        ),
        source_format=FileFormat.MMCIF,
    )
    bond = TopologyBond(
        structure.constitution.atom_index(AtomRef(SERINE, parent)),
        structure.constitution.atom_index(AtomRef(PARTNER, partner_name)),
        relationship_type=relationship,
        order=None if relationship is BondRelationshipType.UNKNOWN else 1,
        provenance=BondProvenance.SOURCE_EXPLICIT,
    )
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(*structure.topology.bonds, bond),
        ),
        provenance=structure.provenance,
    )


@pytest.mark.parametrize("element", ("C", "S", "B", "MG"))
@pytest.mark.parametrize(
    "relationship", (BondRelationshipType.COVALENT, BondRelationshipType.UNKNOWN)
)
def test_external_occupancy_suppresses_only_unresolved_parent_h(
    element: str, relationship: BondRelationshipType
) -> None:
    source = linked_serine(relationship, element=element)
    library = build_default_component_library()
    prepared = PolymerMicrostatePreparation(PolymerMicrostateContext(source), library)
    context = prepared.fixed_hydrogen_context(SERINE)
    assert context is not None and context.unresolved_parent_names == {"OG"}
    assert "HG" not in context.hydrogen_atom_names()

    result = materialize_hydrogens_core(source, component_library=library)
    residue = result.structure.constitution.residue_or_ligand(SERINE)
    assert residue is not None
    assert not residue.has_atom_site("HG")
    assert {"HA", "HB1", "HB2"} <= set(residue.atom_site_names())
    assert any(issue.residue_id == SERINE for issue in result.issues)
    facts = StructureChemistryObservation(result.structure).residue_completion_facts(
        SERINE
    )
    assert facts.hydrogen_applicability_state is HydrogenApplicabilityState.APPLICABLE
    assert facts.hydrogen_coverage_state is HydrogenCoverageState.PARTIAL


@pytest.mark.parametrize(
    "relationship",
    (BondRelationshipType.METAL_COORDINATION, BondRelationshipType.HYDROGEN_BOND),
)
def test_typed_noncovalent_relationship_does_not_consume_static_h_inventory(
    relationship: BondRelationshipType,
) -> None:
    source = linked_serine(relationship, element="MG")
    result = materialize_hydrogens_core(source)
    residue = result.structure.constitution.residue_or_ligand(SERINE)
    assert residue is not None and residue.has_atom_site("HG")


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
def test_unresolved_source_h_is_preserved_not_approved_or_rotated(isotope: str) -> None:
    source = linked_serine(BondRelationshipType.UNKNOWN, isotope=isotope)
    source = read_structure_string(
        write_structure_string(source, FileFormat.MMCIF), FileFormat.MMCIF
    )
    index = source.constitution.atom_index(AtomRef(SERINE, "HG"))
    result = add_hydrogens(source)
    current_index = result.structure.constitution.atom_index(AtomRef(SERINE, "HG"))
    assert result.structure.constitution.atom_site_at(current_index) == (
        source.constitution.atom_site_at(index)
    )
    assert result.structure.geometry.atom_geometry(current_index) == (
        source.geometry.atom_geometry(index)
    )
    cleaned, _ = cleanup_residual_hydrogen_clashes(
        result.structure, component_library=build_default_component_library()
    )
    assert cleaned.geometry.position(current_index) == source.geometry.position(index)
    facts = StructureChemistryObservation(cleaned).residue_completion_facts(SERINE)
    assert facts.hydrogen_coverage_state is HydrogenCoverageState.PARTIAL


def test_topology_only_does_not_invent_bond_at_unresolved_parent() -> None:
    source = linked_serine(isotope="D")
    index = source.constitution.atom_index(AtomRef(SERINE, "HG"))
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=tuple(
                bond for bond in source.topology.bonds if not bond.involves(index)
            ),
        ),
        provenance=source.provenance,
    )
    result = resolve_fixed_hydrogen_topology(source, build_default_component_library())
    assert not any(bond.involves(index) for bond in result.structure.topology.bonds)
    assert result.structure.geometry is source.geometry
    assert any(issue.residue_id == SERINE for issue in result.issues)


def test_rule_applies_to_static_carbon_parent_without_component_recipe() -> None:
    source = linked_serine(parent="CA")
    result = materialize_hydrogens_core(source)
    residue = result.structure.constitution.residue_or_ligand(SERINE)
    assert residue is not None
    assert not residue.has_atom_site("HA")
    assert residue.has_atom_site("HG")


def test_whole_workflow_reports_unresolved_parent_even_without_new_h_work() -> None:
    source = materialize_hydrogens_core(linked_serine()).structure
    result = process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert not result.requested_goals_fully_satisfied()
    assert any(
        issue.residue_id == SERINE and "OG" in issue.message for issue in result.issues
    )
    residue = result.structure.constitution.residue_or_ligand(SERINE)
    assert residue is not None and not residue.has_atom_site("HG")


@pytest.mark.parametrize("typed", (False, True))
def test_pdb_conect_metal_is_not_promoted_to_typed_coordination(typed: bool) -> None:
    source = linked_serine(
        BondRelationshipType.METAL_COORDINATION
        if typed
        else BondRelationshipType.UNKNOWN,
        element="MG",
    )
    pdb = write_structure_string(source, FileFormat.PDB)
    if not typed:
        records = [line for line in pdb.splitlines() if not line.startswith("LINK")]
        parent_serial = next(
            int(line[6:11])
            for line in records
            if line.startswith("ATOM") and line[12:16].strip() == "OG"
        )
        partner_serial = next(
            int(line[6:11]) for line in records if line.startswith("HETATM")
        )
        pdb = (
            "\n".join(line for line in records if line != "END")
            + f"\nCONECT{parent_serial:5d}{partner_serial:5d}\nEND\n"
        )
    reread = read_structure_string(pdb, FileFormat.PDB)
    bond = reread.topology.bond_between(
        reread.constitution.atom_index(AtomRef(SERINE, "OG")),
        reread.constitution.atom_index(AtomRef(PARTNER, "MG")),
    )
    assert bond is not None and bond.relationship_type is (
        BondRelationshipType.METAL_COORDINATION
        if typed
        else BondRelationshipType.UNKNOWN
    )
    result = materialize_hydrogens_core(reread)
    residue = result.structure.constitution.residue_or_ligand(SERINE)
    assert residue is not None and residue.has_atom_site("HG") is typed
    if not typed:
        issue = next(issue for issue in result.issues if issue.residue_id == SERINE)
        assert "untyped connection record" in issue.message
        assert "LINK" in issue.message and "_struct_conn" in issue.message


def test_unresolved_only_does_not_repeatedly_propose_hydrogen_completion() -> None:
    structure = materialize_hydrogens_core(linked_serine()).structure
    outcome = plan_hydrogen_completion_transformers(
        structure,
        requested_goals=RequestedGoalSet(
            whole_structure_requested_goals(HydrogenCoverageState.COMPLETE)
        ),
        component_library=build_default_component_library(),
    )
    assert not outcome.transformers


def test_heavy_only_workflow_does_not_add_unrequested_h_diagnostic() -> None:
    source = linked_serine()
    result = process_structure(
        source, ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP)
    )
    assert not any(
        issue.residue_id == SERINE and "linked polymer H" in issue.message
        for issue in result.issues
    )


def test_multiple_unresolved_parents_do_not_hide_independent_fixed_h() -> None:
    source = linked_serine()
    extra = TopologyBond(
        source.constitution.atom_index(AtomRef(SERINE, "CA")),
        source.constitution.atom_index(AtomRef(PARTNER, "X")),
        relationship_type=BondRelationshipType.UNKNOWN,
        provenance=BondProvenance.SOURCE_EXPLICIT,
    )
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(*source.topology.bonds, extra),
        ),
        provenance=source.provenance,
    )
    preparation = PolymerMicrostatePreparation(
        PolymerMicrostateContext(source), build_default_component_library()
    )
    fixed = preparation.fixed_hydrogen_context(SERINE)
    assert fixed is not None and fixed.unresolved_parent_names == {"CA", "OG"}
    result = materialize_hydrogens_core(source)
    residue = result.structure.constitution.residue_or_ligand(SERINE)
    assert residue is not None
    assert not residue.has_atom_site("HA") and not residue.has_atom_site("HG")
    assert {"HB1", "HB2"} <= set(residue.atom_site_names())


def test_graph_edit_reentry_releases_only_the_removed_occupancy() -> None:
    source = linked_serine()
    first = materialize_hydrogens_core(source).structure
    unlinked = ProteinStructure.from_payload(
        constitution=first.constitution,
        geometry=first.geometry,
        topology=StructureTopology(
            constitution=first.constitution,
            atom_topologies=first.topology.atom_topologies,
            bonds=tuple(
                bond
                for bond in first.topology.bonds
                if first.constitution.atom_ref_at(bond.atom_index_1).residue_id
                == first.constitution.atom_ref_at(bond.atom_index_2).residue_id
            ),
        ),
        provenance=first.provenance,
    )
    second = materialize_hydrogens_core(unlinked).structure
    residue = second.constitution.residue_or_ligand(SERINE)
    assert residue is not None and residue.has_atom_site("HG")
    assert (
        StructureChemistryObservation(second)
        .residue_completion_facts(SERINE)
        .hydrogen_coverage_state
        is HydrogenCoverageState.COMPLETE
    )


@pytest.mark.parametrize("with_h", (False, True))
def test_unknown_only_custom_inventory_is_not_resolved_zero_h(with_h: bool) -> None:
    source = linked_serine(
        component="XSR", isotope="D" if with_h else None, missing=frozenset(("N",))
    )
    template = build_default_component_library().require("SER")
    custom = replace(
        template,
        definition=replace(template.definition, component_id="XSR"),
        backbone_family_component_id="XSR",
        hydrogen_semantics=HydrogenSemantics(
            plan_with_backbone=((("HG",), "class4", ("CA", "CB", "OG")),)
        ),
    )
    library = ComponentLibrary(templates={"XSR": custom})
    observation = StructureChemistryObservation(source, component_library=library)
    facts = observation.residue_completion_facts(SERINE)
    assert facts.hydrogen_applicability_state is HydrogenApplicabilityState.APPLICABLE
    assert facts.hydrogen_coverage_state is (
        HydrogenCoverageState.PARTIAL if with_h else HydrogenCoverageState.NONE
    )
    assert not observation.hydrogen_expectation_model.expected_hydrogen_atom_names_by_residue.get(
        SERINE
    )


def test_positive_source_charge_does_not_authorize_isolated_h_inventory() -> None:
    source = linked_serine(isotope="H")
    index = source.constitution.atom_index(AtomRef(SERINE, "OG"))
    charges = list(source.topology.atom_topologies)
    charges[index.value] = AtomTopology(formal_charge=1)
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=tuple(charges),
            bonds=source.topology.bonds,
        ),
        provenance=source.provenance,
    )
    result = materialize_hydrogens_core(source, rebuild_existing=True).structure
    current = result.constitution.atom_index(AtomRef(SERINE, "OG"))
    assert result.topology.formal_charge(current) == 1
    hydrogen = result.constitution.atom_index(AtomRef(SERINE, "HG"))
    assert result.geometry.atom_geometry(hydrogen) == source.geometry.atom_geometry(
        source.constitution.atom_index(AtomRef(SERINE, "HG"))
    )
    assert (
        StructureChemistryObservation(result)
        .residue_completion_facts(SERINE)
        .hydrogen_coverage_state
        is HydrogenCoverageState.PARTIAL
    )


def test_local_refinement_readiness_rejects_unresolved_fixed_parent() -> None:
    source = materialize_hydrogens_core(linked_serine()).structure
    facts = derive_atom_scope_continuous_relaxation_facts(
        ProteinStructureSnapshot.from_structure(source),
        AtomSetScope(atom_refs=(AtomRef(SERINE, "OG"),)),
    )
    assessment = ContinuousRelaxationReadinessPolicy().assess_atom_scope_facts(facts)
    assert not assessment.is_ready()
    assert facts.continuous_region_readiness_facts.chemistry_readiness_facts.hydrogen_coverage_state.needs_hydrogenation()


def test_explicit_local_prerequisite_reports_current_unresolved_h() -> None:
    source = materialize_hydrogens_core(linked_serine()).structure
    result = process_structure(
        source,
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        transform_requests=WorkflowTransformRequests(
            repair_refinement=RepairRefinementSpec(
                scope_spec=LocalScopeSpec.from_residues((SERINE,)),
                binding=RecommendedContinuousRelaxationBinding(),
            )
        ),
    )
    assert any(
        issue.residue_id == SERINE and "linked polymer H" in issue.message
        for issue in result.issues
    )
    assert not result.relaxation_executions


def test_cleanup_preserves_h_when_its_template_parent_is_missing() -> None:
    source = linked_serine(isotope="D").without_atom_refs((AtomRef(SERINE, "OG"),))
    index = source.constitution.atom_index(AtomRef(SERINE, "HG"))
    cleaned, _ = cleanup_residual_hydrogen_clashes(
        source, component_library=build_default_component_library()
    )
    assert cleaned.geometry.atom_geometry(index) == source.geometry.atom_geometry(index)
    fixed = PolymerMicrostatePreparation(
        PolymerMicrostateContext(source), build_default_component_library()
    ).fixed_hydrogen_context(SERINE)
    assert fixed is not None and not fixed.unresolved_parent_names

"""Unit tests for local refinement planning."""

from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    CanonicalAtomPayload,
    CanonicalResiduePayload,
    atom_payload,
    chain_payload,
    residue_payload,
)
from tests.support.canonical_builders import (
    build_structure as build_canonical_structure,
)
from tests.support.refinement_contract import build_continuous_relaxation_problem
from tests.support.request_builders import ingress_options
from tests.support.whole_structure_sources import WHOLE_STRUCTURE_CORPUS_SOURCES

from protrepair.chemistry import (
    BondDefinition,
    ChemicalComponentDefinition,
    ComponentLibrary,
    ResidueTemplate,
    build_default_component_library,
)
from protrepair.chemistry.nonstandard.registry import build_bundled_nonstandard_registry
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.errors import RefinementError
from protrepair.geometry import Vec3
from protrepair.io import read_structure
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import (
    AtomRef,
    ResidueId,
)
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.continuous.support import LocalBondPlanningSupportMode
from protrepair.transformer.local import LocalScopeSpec
from protrepair.workflow.contracts import LigandPolicy


def test_atomwise_plan_promotes_same_residue_and_radius_context() -> None:
    """Atomwise selections should keep same-residue scaffold and nearby context."""

    structure = build_toy_structure()
    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_atoms(
            (AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),)
        ),
        context_radius_angstrom=2.5,
        component_library=build_toy_component_library(),
    )

    assert plan.region.movable_atom_indices == (
        structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "H1")),
    )
    assert plan.region.fixed_context_atom_indices == (
        structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "C1")),
        structure.constitution.atom_index(AtomRef(ResidueId("L", 1), "O1")),
    )


def test_residuewise_plan_expands_whole_residue_and_peptide_context() -> None:
    """Residuewise selections should include adjacent peptide neighbors as context."""

    structure = build_peptide_structure()
    component_library = build_standard_component_library()
    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_residues((ResidueId(chain_id="A", seq_num=1),)),
        context_radius_angstrom=0.0,
        component_library=component_library,
    )

    assert set(plan.region.movable_atom_indices) == {
        structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "N")),
        structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "CA")),
        structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "C")),
        structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "O")),
    }
    assert structure.constitution.atom_index(AtomRef(ResidueId("A", 2), "N")) in set(
        plan.region.fixed_context_atom_indices
    )
    assert any(
        bond.atom_index_1
        == structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "C"))
        and bond.atom_index_2
        == structure.constitution.atom_index(AtomRef(ResidueId("A", 2), "N"))
        for bond in plan.bonds
    )


def test_residuewise_plan_can_target_ligands_and_promote_nearby_polymer_context() -> (
    None
):
    """Residuewise selection should work for ligand residues, not just polymers."""

    structure = build_toy_structure()
    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_residues((ResidueId(chain_id="L", seq_num=1),)),
        context_radius_angstrom=2.5,
        component_library=build_toy_component_library(),
    )

    assert plan.region.movable_atom_indices == (
        structure.constitution.atom_index(AtomRef(ResidueId("L", 1), "O1")),
    )
    assert set(plan.region.fixed_context_atom_indices) == {
        structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "C1")),
        structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "H1")),
    }


@pytest.mark.parametrize(
    "relationship_type",
    (
        BondRelationshipType.COVALENT,
        BondRelationshipType.DISULFIDE,
    ),
)
def test_source_explicit_topology_bond_seeds_refinement_context(
    relationship_type: BondRelationshipType,
) -> None:
    """Source-explicit covalent-like topology bonds seed local bond planning."""

    structure = with_toy_source_link(
        build_toy_structure(),
        relationship_type=relationship_type,
    )
    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_residues((ResidueId(chain_id="A", seq_num=1),)),
        context_radius_angstrom=0.0,
        component_library=build_toy_component_library(),
    )
    movable_index_set = set(plan.region.movable_atom_indices)
    fixed_context_index_set = set(plan.region.fixed_context_atom_indices)
    source_link_atom_1 = structure.constitution.atom_index(
        AtomRef(ResidueId("A", 1), "C1")
    )
    source_link_atom_2 = structure.constitution.atom_index(
        AtomRef(ResidueId("L", 1), "O1")
    )

    assert source_link_atom_1 in movable_index_set
    assert source_link_atom_2 in fixed_context_index_set
    assert any(
        {bond.atom_index_1, bond.atom_index_2}
        == {source_link_atom_1, source_link_atom_2}
        for bond in plan.bonds
    )


@pytest.mark.parametrize(
    "relationship_type",
    (
        BondRelationshipType.HYDROGEN_BOND,
        BondRelationshipType.METAL_COORDINATION,
        BondRelationshipType.UNKNOWN,
    ),
)
def test_source_explicit_non_covalent_topology_bond_is_not_planned(
    relationship_type: BondRelationshipType,
) -> None:
    """Non-covalent topology truth must not become a covalent FF planned bond."""

    structure = with_toy_source_link(
        build_toy_structure(),
        relationship_type=relationship_type,
        source_id=f"toy-{relationship_type.value}",
    )
    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_residues((ResidueId(chain_id="A", seq_num=1),)),
        context_radius_angstrom=2.5,
        component_library=build_toy_component_library(),
    )
    movable_atom_index = structure.constitution.atom_index(
        AtomRef(ResidueId("A", 1), "C1")
    )
    context_atom_index = structure.constitution.atom_index(
        AtomRef(ResidueId("L", 1), "O1")
    )

    assert structure.topology.bonds
    assert context_atom_index in set(plan.region.fixed_context_atom_indices)
    assert not any(
        {bond.atom_index_1, bond.atom_index_2}
        == {movable_atom_index, context_atom_index}
        for bond in plan.bonds
    )


def test_topology_local_bond_seeds_refinement_planned_bonds() -> None:
    """Local FF planning should consume topology-owned residue-local bonds."""

    structure = with_toy_local_topology_bond(build_toy_structure())
    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_residues((ResidueId(chain_id="A", seq_num=1),)),
        context_radius_angstrom=0.0,
        component_library=ComponentLibrary(
            templates={
                "MOV": ResidueTemplate(
                    definition=ChemicalComponentDefinition(
                        component_id="MOV",
                        atom_names=("C1", "H1"),
                        bonds=(),
                    )
                ),
                "OBS": ResidueTemplate(
                    definition=ChemicalComponentDefinition(
                        component_id="OBS",
                        atom_names=("O1",),
                    )
                ),
            }
        ),
    )
    left_atom_index = structure.constitution.atom_index(
        AtomRef(ResidueId("A", 1), "C1")
    )
    right_atom_index = structure.constitution.atom_index(
        AtomRef(ResidueId("A", 1), "H1")
    )

    assert any(
        {bond.atom_index_1, bond.atom_index_2} == {left_atom_index, right_atom_index}
        for bond in plan.bonds
    )


def test_atomwise_selection_normalizes_to_structure_order_across_residues() -> None:
    """Atomwise input order should not leak into the canonical movable atom order."""

    structure = build_peptide_structure()
    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_atoms(
            (
                AtomRef(ResidueId(chain_id="A", seq_num=2), "N"),
                AtomRef(ResidueId(chain_id="A", seq_num=1), "O"),
            )
        ),
        context_radius_angstrom=0.0,
        component_library=build_standard_component_library(),
    )

    assert plan.region.movable_atom_indices == (
        structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "O")),
        structure.constitution.atom_index(AtomRef(ResidueId("A", 2), "N")),
    )


def test_radius_context_promotes_whole_neighbor_residue() -> None:
    """One nearby atom should promote the whole neighbor residue into context."""

    structure = build_multi_atom_context_structure()
    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_atoms(
            (AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),)
        ),
        context_radius_angstrom=2.0,
        component_library=build_multi_atom_context_component_library(),
    )

    fixed_context_atom_indices = set(plan.region.fixed_context_atom_indices)
    assert (
        structure.constitution.atom_index(AtomRef(ResidueId("L", 1), "O1"))
        in fixed_context_atom_indices
    )
    assert (
        structure.constitution.atom_index(AtomRef(ResidueId("L", 1), "C2"))
        in fixed_context_atom_indices
    )


def test_atomwise_plan_excludes_geometry_only_disulfide_at_zero_radius() -> None:
    """Geometry evidence must not become a hidden continuous-execution bond."""

    structure = build_disulfide_structure()
    assert not any(
        bond.relationship_type is BondRelationshipType.DISULFIDE
        for bond in structure.topology.bonds
    )

    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_atoms(
            (AtomRef(ResidueId(chain_id="A", seq_num=1), "SG"),)
        ),
        context_radius_angstrom=0.0,
        component_library=build_standard_component_library(),
    )

    fixed_context_atom_indices = set(plan.region.fixed_context_atom_indices)
    assert (
        structure.constitution.atom_index(AtomRef(ResidueId("B", 1), "SG"))
        not in fixed_context_atom_indices
    )
    assert not any(
        {bond.atom_index_1, bond.atom_index_2}
        == {
            structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "SG")),
            structure.constitution.atom_index(AtomRef(ResidueId("B", 1), "SG")),
        }
        for bond in plan.bonds
    )


def test_atomwise_plan_includes_canonical_disulfide_partner_at_zero_radius() -> None:
    """Canonical disulfides should enter continuous fixed context and bonds."""

    structure = with_disulfide_topology(build_disulfide_structure())
    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_atoms(
            (AtomRef(ResidueId(chain_id="A", seq_num=1), "SG"),)
        ),
        context_radius_angstrom=0.0,
        component_library=build_standard_component_library(),
    )

    left_sg = structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "SG"))
    right_sg = structure.constitution.atom_index(AtomRef(ResidueId("B", 1), "SG"))
    assert right_sg in set(plan.region.fixed_context_atom_indices)
    assert any(
        {bond.atom_index_1, bond.atom_index_2} == {left_sg, right_sg}
        for bond in plan.bonds
    )


def test_ambiguous_disulfide_candidates_do_not_force_arbitrary_context() -> None:
    """Ambiguous sulfur neighborhoods should not invent one bonded context partner."""

    structure = build_ambiguous_disulfide_structure()
    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_atoms(
            (AtomRef(ResidueId(chain_id="A", seq_num=1), "SG"),)
        ),
        context_radius_angstrom=0.0,
        component_library=build_standard_component_library(),
    )

    fixed_context_atom_indices = set(plan.region.fixed_context_atom_indices)
    assert (
        structure.constitution.atom_index(AtomRef(ResidueId("B", 1), "SG"))
        not in fixed_context_atom_indices
    )
    assert (
        structure.constitution.atom_index(AtomRef(ResidueId("C", 1), "SG"))
        not in fixed_context_atom_indices
    )


def test_adjacent_residues_in_different_chains_do_not_form_peptide_context() -> None:
    """Peptide-bond context detection must not jump across chain boundaries."""

    structure = build_cross_chain_adjacent_structure()
    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_residues((ResidueId(chain_id="A", seq_num=1),)),
        context_radius_angstrom=0.0,
        component_library=build_standard_component_library(),
    )

    assert all(
        structure.constitution.residue_site_at(
            structure.constitution.residue_index_for_atom_index(atom_index)
        ).residue_id.chain_id
        != "B"
        for atom_index in plan.region.fixed_context_atom_indices
    )
    assert not any(
        structure.constitution.residue_site_at(
            structure.constitution.residue_index_for_atom_index(bond.atom_index_1)
        ).residue_id.chain_id
        != structure.constitution.residue_site_at(
            structure.constitution.residue_index_for_atom_index(bond.atom_index_2)
        ).residue_id.chain_id
        for bond in plan.bonds
    )


def test_numbering_gaps_do_not_form_implicit_peptide_context() -> None:
    """Peptide-bond context detection must not bridge missing sequence spans."""

    structure = build_same_chain_numbering_gap_structure()
    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_residues(
            (ResidueId(chain_id="A", seq_num=615),)
        ),
        context_radius_angstrom=0.0,
        component_library=build_standard_component_library(),
    )

    assert all(
        structure.constitution.residue_site_at(
            structure.constitution.residue_index_for_atom_index(atom_index)
        ).residue_id.seq_num
        != 581
        for atom_index in plan.region.fixed_context_atom_indices
    )
    assert not any(
        {
            structure.constitution.atom_ref_at(bond.atom_index_1).display_token(),
            structure.constitution.atom_ref_at(bond.atom_index_2).display_token(),
        }
        == {"A:581.C", "A:615.N"}
        for bond in plan.bonds
    )


def test_plan_handles_real_ptr_fixture_with_mixed_insertion_codes() -> None:
    """Real PTR fixtures with insertion codes should still produce one plan."""

    structure = read_structure(
        Path("tests/fixtures/pdb/refinement/4jfx_ptr8_local.pdb"),
        policy=ingress_options(
            ligand_policy=LigandPolicy.KEEP
        ).structure_normalization_policy(),
    )
    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_residues((ResidueId("P", 8),)),
        context_radius_angstrom=4.0,
        component_library=build_default_component_library(),
    )

    assert any(
        structure.constitution.residue_site_at(
            structure.constitution.residue_index_for_atom_index(atom_index)
        ).residue_id
        == ResidueId("P", 8)
        for atom_index in plan.region.movable_atom_indices
    )
    assert plan.bonds


def test_plan_rejects_missing_template_for_promoted_context_residue() -> None:
    """Disconnected multi-atom context should still fail without passive support."""

    structure = build_multi_atom_context_structure()
    with pytest.raises(RefinementError, match="CTX"):
        build_continuous_relaxation_problem(
            structure,
            scope_spec=LocalScopeSpec.from_atoms(
                (AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),)
            ),
            context_radius_angstrom=2.0,
            component_library=ComponentLibrary(
                templates={
                    "MOV": ResidueTemplate(
                        definition=ChemicalComponentDefinition(
                            component_id="MOV",
                            atom_names=("C1", "H1"),
                            bonds=(BondDefinition("C1", "H1"),),
                        )
                    )
                }
            ),
        )


def test_plan_allows_template_less_single_center_passive_context() -> None:
    """Single-center context residues should not require a component template."""

    structure = build_test_structure(
        chain_id="A",
        residues=(
            build_residue_payload(
                component_id="MOV",
                residue_id=ResidueId(chain_id="A", seq_num=1),
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("H1", "H", Vec3(1.0, 0.0, 0.0)),
                ),
            ),
        ),
        ligands=(
            build_residue_payload(
                component_id="ZNX",
                residue_id=ResidueId(chain_id="L", seq_num=1),
                atoms=(atom_payload("ZN1", "Zn", Vec3(2.0, 0.0, 0.0)),),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )

    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_atoms(
            (AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),)
        ),
        context_radius_angstrom=2.0,
        component_library=ComponentLibrary(
            templates={
                "MOV": ResidueTemplate(
                    definition=ChemicalComponentDefinition(
                        component_id="MOV",
                        atom_names=("C1", "H1"),
                        bonds=(BondDefinition("C1", "H1"),),
                    )
                )
            }
        ),
    )

    assert structure.constitution.atom_index(AtomRef(ResidueId("L", 1), "ZN1")) in (
        plan.region.fixed_context_atom_indices
    )
    support_by_residue_index = plan.region.local_bond_planning_support_by_residue_index(
        ComponentLibrary(
            templates={
                "MOV": ResidueTemplate(
                    definition=ChemicalComponentDefinition(
                        component_id="MOV",
                        atom_names=("C1", "H1"),
                        bonds=(BondDefinition("C1", "H1"),),
                    )
                )
            }
        )
    )
    zinc_residue_index = structure.constitution.residue_index(ResidueId("L", 1))
    assert support_by_residue_index[zinc_residue_index].mode is (
        LocalBondPlanningSupportMode.NONBONDING_PASSIVE_CONTEXT
    )


def test_plan_rejects_selected_single_atom_inorganic_as_not_applicable() -> None:
    """Selected metals/ions should fail via explicit nonbonding policy."""

    structure = read_structure(
        Path("tests/fixtures/pdb/refinement/5xm5_metal_sites_local.pdb"),
        policy=ingress_options(
            ligand_policy=LigandPolicy.KEEP
        ).structure_normalization_policy(),
    )

    with pytest.raises(
        RefinementError,
        match=(
            "single-atom metal/ion residues as nonbonding fixed context, "
            "not movable chemistry"
        ),
    ):
        build_continuous_relaxation_problem(
            structure,
            scope_spec=LocalScopeSpec.from_residues(
                (ResidueId("A", 201), ResidueId("B", 202))
            ),
            context_radius_angstrom=4.0,
            component_library=build_default_component_library(),
        )


def test_plan_allows_6nbb_zinc_as_nonbonding_fixed_context() -> None:
    """6NBB zinc context should be explicit nonbonding passive chemistry."""

    structure = read_structure(
        WHOLE_STRUCTURE_CORPUS_SOURCES["6nbb-whole-structure"].output_path
    )
    component_library = build_default_component_library()
    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_atoms((AtomRef(ResidueId("A", 401), "C5N"),)),
        context_radius_angstrom=4.2,
        component_library=component_library,
    )
    zinc_residue_index = structure.constitution.residue_index(ResidueId("A", 402))
    support_by_residue_index = plan.region.local_bond_planning_support_by_residue_index(
        component_library
    )

    assert zinc_residue_index in plan.region.included_residue_indices
    assert support_by_residue_index[zinc_residue_index].mode is (
        LocalBondPlanningSupportMode.NONBONDING_PASSIVE_CONTEXT
    )
    assert not any(
        zinc_residue_index
        in {
            plan.region.snapshot.structure.constitution.residue_index_for_atom_index(
                bond.atom_index_1
            ),
            plan.region.snapshot.structure.constitution.residue_index_for_atom_index(
                bond.atom_index_2
            ),
        }
        for bond in plan.bonds
    )


def test_plan_allows_template_less_connected_retained_non_polymer_context() -> None:
    """Connected hetero context may use RDKit fallback topology without a template."""

    structure = build_test_structure(
        chain_id="A",
        residues=(
            build_residue_payload(
                component_id="MOV",
                residue_id=ResidueId(chain_id="A", seq_num=1),
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("H1", "H", Vec3(1.0, 0.0, 0.0)),
                ),
            ),
        ),
        ligands=(
            build_residue_payload(
                component_id="LIG",
                residue_id=ResidueId(chain_id="L", seq_num=1),
                atoms=(
                    atom_payload("C1", "C", Vec3(2.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(3.4, 0.0, 0.0)),
                    atom_payload("H1", "H", Vec3(1.5, 0.9, 0.0)),
                    atom_payload("H2", "H", Vec3(1.5, -0.9, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )

    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_atoms(
            (AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),)
        ),
        context_radius_angstrom=3.0,
        component_library=ComponentLibrary(
            templates={
                "MOV": ResidueTemplate(
                    definition=ChemicalComponentDefinition(
                        component_id="MOV",
                        atom_names=("C1", "H1"),
                        bonds=(BondDefinition("C1", "H1"),),
                    )
                )
            }
        ),
    )

    assert any(
        {
            plan.region.snapshot.structure.constitution.atom_index(
                AtomRef(ResidueId("L", 1), "C1")
            ),
            plan.region.snapshot.structure.constitution.atom_index(
                AtomRef(ResidueId("L", 1), "O1")
            ),
        }
        == {bond.atom_index_1, bond.atom_index_2}
        for bond in plan.bonds
    )


def test_zero_radius_isolated_selection_keeps_no_external_context() -> None:
    """A truly isolated movable atom should not accumulate external fixed context."""

    structure = build_test_structure(
        chain_id="A",
        residues=(
            build_residue_payload(
                component_id="MOV",
                residue_id=ResidueId(chain_id="A", seq_num=1),
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("H1", "H", Vec3(1.0, 0.0, 0.0)),
                ),
            ),
        ),
        ligands=(
            build_residue_payload(
                component_id="OBS",
                residue_id=ResidueId(chain_id="L", seq_num=1),
                atoms=(atom_payload("O1", "O", Vec3(10.0, 0.0, 0.0)),),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )
    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_atoms(
            (AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),)
        ),
        context_radius_angstrom=0.0,
        component_library=build_toy_component_library(),
    )

    assert plan.region.fixed_context_atom_indices == (
        structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "C1")),
    )


def test_refinement_plan_infers_standard_hydrogen_attachments() -> None:
    """Standard component templates should still yield hydrogen bonds in the plan."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    structure = build_test_structure(
        chain_id="A",
        residues=(
            build_residue_payload(
                component_id="SER",
                residue_id=residue_id,
                atoms=(
                    atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("CA", "C", Vec3(1.45, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(2.40, 1.20, 0.0)),
                    atom_payload("O", "O", Vec3(3.55, 1.10, 0.0)),
                    atom_payload("CB", "C", Vec3(1.95, -1.20, 0.0)),
                    atom_payload("OG", "O", Vec3(3.30, -1.35, 0.0)),
                    atom_payload("HG", "H", Vec3(3.95, -0.75, 0.0)),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_atoms((AtomRef(residue_id, "HG"),)),
        context_radius_angstrom=0.0,
        component_library=build_standard_component_library(),
    )

    assert any(
        {bond.atom_index_1, bond.atom_index_2}
        == {
            plan.region.snapshot.structure.constitution.atom_index(
                AtomRef(residue_id, "OG")
            ),
            plan.region.snapshot.structure.constitution.atom_index(
                AtomRef(residue_id, "HG")
            ),
        }
        for bond in plan.bonds
    )


def test_refinement_plan_prefers_template_hydrogen_attachment_over_nearest_anchor() -> (
    None
):
    """Template hydrogen bonds should override misleading nearest-heavy geometry."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    plan = build_continuous_relaxation_problem(
        build_test_structure(
            chain_id="A",
            residues=(
                build_residue_payload(
                    component_id="LEU",
                    residue_id=residue_id,
                    atoms=(
                        atom_payload("N", "N", Vec3(-1.0, 0.0, 0.0)),
                        atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                        atom_payload("C", "C", Vec3(1.5, 0.0, 0.0)),
                        atom_payload("O", "O", Vec3(2.5, 0.2, 0.0)),
                        atom_payload("CB", "C", Vec3(0.0, 1.0, 0.0)),
                        atom_payload("CG", "C", Vec3(1.0, 1.0, 0.0)),
                        atom_payload("CD1", "C", Vec3(2.0, 1.8, 0.0)),
                        atom_payload("CD2", "C", Vec3(2.0, 0.2, 0.0)),
                        atom_payload("HB1", "H", Vec3(0.96, 1.05, 0.0)),
                    ),
                ),
            ),
            source_format=FileFormat.PDB,
        ),
        scope_spec=LocalScopeSpec.from_atoms((AtomRef(residue_id, "HB1"),)),
        context_radius_angstrom=0.0,
        component_library=build_standard_component_library(),
    )

    assert any(
        {bond.atom_index_1, bond.atom_index_2}
        == {
            plan.region.snapshot.structure.constitution.atom_index(
                AtomRef(residue_id, "CB")
            ),
            plan.region.snapshot.structure.constitution.atom_index(
                AtomRef(residue_id, "HB1")
            ),
        }
        for bond in plan.bonds
    )
    assert not any(
        {bond.atom_index_1, bond.atom_index_2}
        == {
            plan.region.snapshot.structure.constitution.atom_index(
                AtomRef(residue_id, "CG")
            ),
            plan.region.snapshot.structure.constitution.atom_index(
                AtomRef(residue_id, "HB1")
            ),
        }
        for bond in plan.bonds
    )


def test_refinement_plan_does_not_infer_hydrogen_attachment_beyond_threshold() -> None:
    """Very distant hydrogens should not be spuriously bonded into the plan."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    plan = build_continuous_relaxation_problem(
        build_test_structure(
            chain_id="A",
            residues=(
                build_residue_payload(
                    component_id="SER",
                    residue_id=residue_id,
                    atoms=(
                        atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                        atom_payload("CA", "C", Vec3(1.45, 0.0, 0.0)),
                        atom_payload("C", "C", Vec3(2.40, 1.20, 0.0)),
                        atom_payload("O", "O", Vec3(3.55, 1.10, 0.0)),
                        atom_payload("CB", "C", Vec3(1.95, -1.20, 0.0)),
                        atom_payload("OG", "O", Vec3(3.30, -1.35, 0.0)),
                        atom_payload("HG", "H", Vec3(6.20, -1.35, 0.0)),
                    ),
                ),
            ),
            source_format=FileFormat.PDB,
        ),
        scope_spec=LocalScopeSpec.from_atoms((AtomRef(residue_id, "HG"),)),
        context_radius_angstrom=0.0,
        component_library=build_standard_component_library(),
    )

    assert not any(
        {bond.atom_index_1, bond.atom_index_2}
        == {
            plan.region.snapshot.structure.constitution.atom_index(
                AtomRef(residue_id, "OG")
            ),
            plan.region.snapshot.structure.constitution.atom_index(
                AtomRef(residue_id, "HG")
            ),
        }
        for bond in plan.bonds
    )


def test_plan_supports_bundled_nonstandard_templates_in_default_library() -> None:
    """Bundled nonstandard component templates should participate in planning."""

    bundled_record = build_bundled_nonstandard_registry().get("MSE")
    assert bundled_record is not None

    residue_id = ResidueId(chain_id="A", seq_num=1)
    plan = build_continuous_relaxation_problem(
        build_test_structure(
            chain_id="A",
            residues=(
                build_residue_payload(
                    component_id="MSE",
                    residue_id=residue_id,
                    atoms=tuple(
                        atom_payload(
                            atom.atom_name,
                            atom.element,
                            Vec3(float(index), 0.0, 0.0),
                            formal_charge=(
                                atom.formal_charge if atom.formal_charge != 0 else None
                            ),
                        )
                        for index, atom in enumerate(bundled_record.heavy_atoms())
                    ),
                ),
            ),
            source_format=FileFormat.PDB,
        ),
        scope_spec=LocalScopeSpec.from_residues((residue_id,)),
        context_radius_angstrom=0.0,
        component_library=build_default_component_library(),
    )

    assert set(plan.region.movable_atom_indices) == {
        plan.region.snapshot.structure.constitution.atom_index(
            AtomRef(residue_id, atom.atom_name)
        )
        for atom in bundled_record.heavy_atoms()
    }
    assert any(
        {
            plan.region.snapshot.structure.constitution.atom_site_at(
                bond.atom_index_1
            ).name,
            plan.region.snapshot.structure.constitution.atom_site_at(
                bond.atom_index_2
            ).name,
        }
        == {"SE", "CE"}
        for bond in plan.bonds
    )


def test_plan_resolves_standard_alias_templates() -> None:
    """Standard alias component ids should refine against their canonical template."""

    residue_id = ResidueId(chain_id="A", seq_num=1)
    plan = build_continuous_relaxation_problem(
        build_test_structure(
            chain_id="A",
            residues=(
                build_residue_payload(
                    component_id="HSE",
                    residue_id=residue_id,
                    atoms=(
                        atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                        atom_payload("CA", "C", Vec3(1.45, 0.0, 0.0)),
                        atom_payload("C", "C", Vec3(2.25, 1.25, 0.0)),
                        atom_payload("O", "O", Vec3(3.45, 1.25, 0.0)),
                        atom_payload("CB", "C", Vec3(1.85, -1.25, 0.0)),
                        atom_payload("CG", "C", Vec3(3.10, -1.65, 0.0)),
                        atom_payload("ND1", "N", Vec3(4.05, -0.75, 0.0)),
                        atom_payload("CD2", "C", Vec3(3.55, -3.00, 0.0)),
                        atom_payload("CE1", "C", Vec3(5.05, -1.55, 0.0)),
                        atom_payload("NE2", "N", Vec3(4.70, -2.85, 0.0)),
                    ),
                ),
            ),
            source_format=FileFormat.PDB,
        ),
        scope_spec=LocalScopeSpec.from_residues((residue_id,)),
        context_radius_angstrom=0.0,
        component_library=build_standard_component_library(),
    )

    assert any(
        {
            plan.region.snapshot.structure.constitution.atom_site_at(
                bond.atom_index_1
            ).name,
            plan.region.snapshot.structure.constitution.atom_site_at(
                bond.atom_index_2
            ).name,
        }
        == {"CG", "ND1"}
        for bond in plan.bonds
    )


def test_plan_rejects_atomless_selected_residues() -> None:
    """Residues that normalize to an empty atom domain should fail clearly."""

    structure = build_test_structure(
        chain_id="A",
        residues=(
            build_residue_payload(
                component_id="GLY",
                residue_id=ResidueId(chain_id="A", seq_num=1),
                atoms=(),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    with pytest.raises(ValueError, match="at least one atom index"):
        build_continuous_relaxation_problem(
            structure,
            scope_spec=LocalScopeSpec.from_residues(
                (ResidueId(chain_id="A", seq_num=1),)
            ),
            context_radius_angstrom=0.0,
            component_library=build_standard_component_library(),
        )


def build_toy_component_library() -> ComponentLibrary:
    """Return one tiny component library suitable for refinement planning tests."""

    mov_template = ResidueTemplate(
        definition=ChemicalComponentDefinition(
            component_id="MOV",
            atom_names=("C1", "H1"),
            bonds=(BondDefinition("C1", "H1"),),
        )
    )
    obs_template = ResidueTemplate(
        definition=ChemicalComponentDefinition(
            component_id="OBS",
            atom_names=("O1",),
        )
    )
    return ComponentLibrary(
        templates={
            "MOV": mov_template,
            "OBS": obs_template,
        }
    )


def build_toy_structure() -> ProteinStructure:
    """Return one tiny structure with one movable residue and one nearby ligand."""

    return build_test_structure(
        chain_id="A",
        residues=(
            build_residue_payload(
                component_id="MOV",
                residue_id=ResidueId(chain_id="A", seq_num=1),
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("H1", "H", Vec3(1.0, 0.0, 0.0)),
                ),
            ),
        ),
        ligands=(
            build_residue_payload(
                component_id="OBS",
                residue_id=ResidueId(chain_id="L", seq_num=1),
                atoms=(atom_payload("O1", "O", Vec3(2.0, 0.0, 0.0)),),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )


def with_toy_source_link(
    structure: ProteinStructure,
    *,
    relationship_type: BondRelationshipType = BondRelationshipType.COVALENT,
    source_id: str = "toy-link",
) -> ProteinStructure:
    """Return the toy structure with one source-explicit MOV-LIG link."""

    atom_ref_1 = AtomRef(ResidueId("A", 1), "C1")
    atom_ref_2 = AtomRef(ResidueId("L", 1), "O1")
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    atom_index_1=structure.constitution.atom_index(atom_ref_1),
                    atom_index_2=structure.constitution.atom_index(atom_ref_2),
                    relationship_type=relationship_type,
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                    source_metadata=SourceBondMetadata(
                        record_type=SourceBondRecordType.PDB_LINK,
                        source_id=source_id,
                    ),
                ),
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def with_toy_local_topology_bond(structure: ProteinStructure) -> ProteinStructure:
    """Return the toy structure with one residue-local covalent topology bond."""

    atom_ref_1 = AtomRef(ResidueId("A", 1), "C1")
    atom_ref_2 = AtomRef(ResidueId("A", 1), "H1")
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(
                TopologyBond(
                    atom_index_1=structure.constitution.atom_index(atom_ref_1),
                    atom_index_2=structure.constitution.atom_index(atom_ref_2),
                    relationship_type=BondRelationshipType.COVALENT,
                    provenance=BondProvenance.TEMPLATE_RESOLVED,
                ),
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def build_peptide_structure() -> ProteinStructure:
    """Return a two-residue peptide fragment with a present peptide bond."""

    return build_test_structure(
        chain_id="A",
        residues=(
            build_residue_payload(
                component_id="GLY",
                residue_id=ResidueId(chain_id="A", seq_num=1),
                atoms=(
                    atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("CA", "C", Vec3(1.45, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(2.25, 1.25, 0.0)),
                    atom_payload("O", "O", Vec3(3.45, 1.25, 0.0)),
                ),
            ),
            build_residue_payload(
                component_id="GLY",
                residue_id=ResidueId(chain_id="A", seq_num=2),
                atoms=(
                    atom_payload("N", "N", Vec3(2.95, 2.35, 0.0)),
                    atom_payload("CA", "C", Vec3(3.85, 3.45, 0.0)),
                    atom_payload("C", "C", Vec3(5.15, 3.00, 0.0)),
                    atom_payload("O", "O", Vec3(6.15, 3.65, 0.0)),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )


def build_multi_atom_context_component_library() -> ComponentLibrary:
    """Return one tiny component library with a multi-atom context residue."""

    mov_template = ResidueTemplate(
        definition=ChemicalComponentDefinition(
            component_id="MOV",
            atom_names=("C1", "H1"),
            bonds=(BondDefinition("C1", "H1"),),
        )
    )
    ctx_template = ResidueTemplate(
        definition=ChemicalComponentDefinition(
            component_id="CTX",
            atom_names=("O1", "C2"),
            bonds=(BondDefinition("O1", "C2"),),
        )
    )
    return ComponentLibrary(templates={"MOV": mov_template, "CTX": ctx_template})


def build_multi_atom_context_structure() -> ProteinStructure:
    """Return one local environment with one multi-atom context residue."""

    return build_test_structure(
        chain_id="A",
        residues=(
            build_residue_payload(
                component_id="MOV",
                residue_id=ResidueId(chain_id="A", seq_num=1),
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("H1", "H", Vec3(1.0, 0.0, 0.0)),
                ),
            ),
        ),
        ligands=(
            build_residue_payload(
                component_id="CTX",
                residue_id=ResidueId(chain_id="L", seq_num=1),
                atoms=(
                    atom_payload("O1", "O", Vec3(2.0, 0.0, 0.0)),
                    atom_payload("C2", "C", Vec3(8.0, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )


def build_disulfide_structure() -> ProteinStructure:
    """Return two cysteines arranged as one likely disulfide pair."""

    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue_payload(
                        component_id="CYS",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.45, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.25, 1.25, 0.0)),
                            atom_payload("O", "O", Vec3(3.45, 1.25, 0.0)),
                            atom_payload("CB", "C", Vec3(1.85, -1.25, 0.0)),
                            atom_payload("SG", "S", Vec3(3.55, -1.45, 0.0)),
                        ),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    build_residue_payload(
                        component_id="CYS",
                        residue_id=ResidueId(chain_id="B", seq_num=1),
                        atoms=(
                            atom_payload("N", "N", Vec3(2.95, 2.35, 0.0)),
                            atom_payload("CA", "C", Vec3(3.85, 3.45, 0.0)),
                            atom_payload("C", "C", Vec3(5.15, 3.00, 0.0)),
                            atom_payload("O", "O", Vec3(6.15, 3.65, 0.0)),
                            atom_payload("CB", "C", Vec3(4.05, 4.85, 0.0)),
                            atom_payload("SG", "S", Vec3(3.75, 0.65, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )


def with_disulfide_topology(structure: ProteinStructure) -> ProteinStructure:
    """Return a structure with one evidence-resolved canonical disulfide."""

    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(
                *structure.topology.bonds,
                TopologyBond(
                    atom_index_1=structure.constitution.atom_index(
                        AtomRef(ResidueId("A", 1), "SG")
                    ),
                    atom_index_2=structure.constitution.atom_index(
                        AtomRef(ResidueId("B", 1), "SG")
                    ),
                    relationship_type=BondRelationshipType.DISULFIDE,
                    provenance=BondProvenance.EVIDENCE_RESOLVED,
                ),
            ),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def build_ambiguous_disulfide_structure() -> ProteinStructure:
    """Return three cysteines where one sulfur has two nearby candidates."""

    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue_payload(
                        component_id="CYS",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.45, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.25, 1.25, 0.0)),
                            atom_payload("O", "O", Vec3(3.45, 1.25, 0.0)),
                            atom_payload("CB", "C", Vec3(1.85, -1.25, 0.0)),
                            atom_payload("SG", "S", Vec3(3.55, -1.45, 0.0)),
                        ),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    build_residue_payload(
                        component_id="CYS",
                        residue_id=ResidueId(chain_id="B", seq_num=1),
                        atoms=(
                            atom_payload("N", "N", Vec3(2.95, 2.35, 0.0)),
                            atom_payload("CA", "C", Vec3(3.85, 3.45, 0.0)),
                            atom_payload("C", "C", Vec3(5.15, 3.00, 0.0)),
                            atom_payload("O", "O", Vec3(6.15, 3.65, 0.0)),
                            atom_payload("CB", "C", Vec3(4.05, 4.85, 0.0)),
                            atom_payload("SG", "S", Vec3(3.75, 0.65, 0.0)),
                        ),
                    ),
                ),
            ),
            chain_payload(
                "C",
                (
                    build_residue_payload(
                        component_id="CYS",
                        residue_id=ResidueId(chain_id="C", seq_num=1),
                        atoms=(
                            atom_payload("N", "N", Vec3(5.95, 2.35, 0.0)),
                            atom_payload("CA", "C", Vec3(6.85, 3.45, 0.0)),
                            atom_payload("C", "C", Vec3(8.15, 3.00, 0.0)),
                            atom_payload("O", "O", Vec3(9.15, 3.65, 0.0)),
                            atom_payload("CB", "C", Vec3(6.55, 0.85, 0.0)),
                            atom_payload("SG", "S", Vec3(3.95, -3.05, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
    )


def build_cross_chain_adjacent_structure() -> ProteinStructure:
    """Return adjacent sequence numbers split across two chains."""

    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.45, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.25, 1.25, 0.0)),
                            atom_payload("O", "O", Vec3(3.45, 1.25, 0.0)),
                        ),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    build_residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId(chain_id="B", seq_num=2),
                        atoms=(
                            atom_payload("N", "N", Vec3(2.95, 2.35, 0.0)),
                            atom_payload("CA", "C", Vec3(3.85, 3.45, 0.0)),
                            atom_payload("C", "C", Vec3(5.15, 3.00, 0.0)),
                            atom_payload("O", "O", Vec3(6.15, 3.65, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(),
        source_format=FileFormat.PDB,
    )


def build_same_chain_numbering_gap_structure() -> ProteinStructure:
    """Return adjacent slots separated by a missing same-chain sequence span."""

    return build_test_structure(
        chain_id="A",
        residues=(
            build_residue_payload(
                component_id="GLY",
                residue_id=ResidueId(chain_id="A", seq_num=581),
                atoms=(
                    atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("CA", "C", Vec3(1.45, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(2.25, 1.25, 0.0)),
                    atom_payload("O", "O", Vec3(3.45, 1.25, 0.0)),
                ),
            ),
            build_residue_payload(
                component_id="GLY",
                residue_id=ResidueId(chain_id="A", seq_num=615),
                atoms=(
                    atom_payload("N", "N", Vec3(10.0, 0.0, 0.0)),
                    atom_payload("CA", "C", Vec3(11.45, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(12.25, 1.25, 0.0)),
                    atom_payload("O", "O", Vec3(13.45, 1.25, 0.0)),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )


def build_residue_payload(
    *,
    component_id: str,
    residue_id: ResidueId,
    atoms: tuple[CanonicalAtomPayload, ...],
    is_hetero: bool = False,
) -> CanonicalResiduePayload:
    """Build one canonical residue payload for refinement planning tests."""

    return residue_payload(
        component_id=component_id,
        residue_id=residue_id,
        atoms=atoms,
        is_hetero=is_hetero,
    )


def build_test_structure(
    *,
    chain_id: str,
    residues: tuple[CanonicalResiduePayload, ...],
    ligands: tuple[CanonicalResiduePayload, ...] = (),
    source_format: FileFormat,
) -> ProteinStructure:
    """Build one canonical structure from one polymer chain plus optional ligands."""

    return build_canonical_structure(
        chains=(chain_payload(chain_id, residues),),
        ligands=ligands,
        source_format=source_format,
    )

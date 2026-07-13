from math import isclose

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import (
    UnknownElementRadiusError,
    build_default_component_library,
)
from protrepair.diagnostics.clashes import (
    detect_clashes_from_context,
    prepare_clash_detection_basis,
    prepare_clash_detection_context,
)
from protrepair.diagnostics.near_covalent import (
    detect_near_covalent_contacts_from_context,
)
from protrepair.diagnostics.parser_readability import (
    RDKitProximityBondCluster,
    RDKitProximityBondWitness,
)
from protrepair.geometry import Vec3
from protrepair.io import FileFormat
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.discrete import parser_witness_pre_untangle as parser_module
from protrepair.transformer.discrete import (
    parser_witness_pre_untangle_materialization as parser_materialization_module,
)
from protrepair.transformer.discrete import (
    parser_witness_pre_untangle_scoring as parser_scoring_module,
)
from protrepair.transformer.discrete.parser_witness_pre_untangle import (
    _deduplicated_sidechain_root_rotation_plans,
    build_parser_witness_pre_untangle_candidate,
    parser_witness_pre_untangle_score,
)
from protrepair.transformer.local import (
    LocalScopeSpec,
    atom_input_from_local_scope_spec,
)


def test_parser_witness_pre_untangle_rotates_sidechain_without_moving_axis() -> None:
    tyr_id = ResidueId("A", 1)
    obstacle_id = ResidueId("B", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="TYR",
                        residue_id=tyr_id,
                        atoms=(
                            atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.5, 0.0, 0.0)),
                            atom_payload("CG", "C", Vec3(2.4, 0.5, 0.0)),
                            atom_payload("CD1", "C", Vec3(3.0, 1.2, 0.0)),
                            atom_payload("CD2", "C", Vec3(3.0, -0.2, 0.0)),
                            atom_payload("CE1", "C", Vec3(3.8, 1.1, 0.0)),
                            atom_payload("CE2", "C", Vec3(3.8, -0.1, 0.0)),
                            atom_payload("CZ", "C", Vec3(4.2, 0.5, 0.0)),
                            atom_payload("OH", "O", Vec3(5.0, 0.5, 0.0)),
                        ),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=obstacle_id,
                        atoms=(atom_payload("C", "C", Vec3(5.0, 0.5, 0.3)),),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    cluster = RDKitProximityBondCluster(
        residue_ids=(tyr_id, obstacle_id),
        bonds=(
            RDKitProximityBondWitness(
                atom_ref_1=AtomRef(tyr_id, "OH"),
                atom_ref_2=AtomRef(obstacle_id, "C"),
                element_1="O",
                element_2="C",
                is_known_component_bond=False,
            ),
        ),
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    scope_spec = LocalScopeSpec.from_residues(cluster.residue_ids)
    component_library = build_default_component_library()
    atom_input = atom_input_from_local_scope_spec(
        snapshot,
        scope_spec,
        component_library=component_library,
    )

    candidate = build_parser_witness_pre_untangle_candidate(
        snapshot,
        atom_input,
        cluster,
        component_library=component_library,
    )

    assert candidate is not None
    assert candidate.score < parser_witness_pre_untangle_score(structure, cluster)
    assert (
        candidate.structure.residue_geometry(
            candidate.structure.constitution.residue_index(tyr_id)
        ).position("CA")
        == structure.residue_geometry(structure.constitution.residue_index(tyr_id))
        .position("CA")
    )
    assert (
        candidate.structure.residue_geometry(
            candidate.structure.constitution.residue_index(tyr_id)
        ).position("CB")
        == structure.residue_geometry(structure.constitution.residue_index(tyr_id))
        .position("CB")
    )
    before_tyr_geometry = structure.residue_geometry(
        structure.constitution.residue_index(tyr_id)
    )
    after_tyr_geometry = candidate.structure.residue_geometry(
        candidate.structure.constitution.residue_index(tyr_id)
    )
    assert after_tyr_geometry.position("OH").distance_to(
        structure.residue_geometry(structure.constitution.residue_index(obstacle_id))
        .position("C")
    ) > before_tyr_geometry.position("OH").distance_to(
        structure.residue_geometry(structure.constitution.residue_index(obstacle_id))
        .position("C")
    )
    assert isclose(
        after_tyr_geometry.position("CG").distance_to(
            after_tyr_geometry.position("CD1")
        ),
        before_tyr_geometry.position("CG").distance_to(
            before_tyr_geometry.position("CD1")
        ),
    )


def test_parser_witness_pre_untangle_score_uses_element_aware_clearance() -> None:
    """False-proximity scoring should not use a carbon-only fixed cutoff."""

    cys_id = ResidueId("A", 1)
    ala_id = ResidueId("B", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="CYS",
                        residue_id=cys_id,
                        atoms=(atom_payload("SG", "S", Vec3(0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ala_id,
                        atoms=(atom_payload("CB", "C", Vec3(1.98, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    cluster = RDKitProximityBondCluster(
        residue_ids=(cys_id, ala_id),
        bonds=(
            RDKitProximityBondWitness(
                atom_ref_1=AtomRef(cys_id, "SG"),
                atom_ref_2=AtomRef(ala_id, "CB"),
                element_1="S",
                element_2="C",
                is_known_component_bond=False,
            ),
        ),
    )

    score = parser_witness_pre_untangle_score(structure, cluster)

    assert score.unresolved_contact_count == 1
    assert score.total_overlap_angstrom > 0.0


def test_parser_witness_scoring_reports_unknown_radius_elements_once() -> None:
    """Witness scoring should aggregate unknown covalent radii before ranking."""

    cys_id = ResidueId("A", 1)
    ala_id = ResidueId("B", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="CYS",
                        residue_id=cys_id,
                        atoms=(atom_payload("SG", "S", Vec3(0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ala_id,
                        atoms=(atom_payload("CB", "C", Vec3(1.98, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    cluster = RDKitProximityBondCluster(
        residue_ids=(cys_id, ala_id),
        bonds=(
            RDKitProximityBondWitness(
                atom_ref_1=AtomRef(cys_id, "SG"),
                atom_ref_2=AtomRef(ala_id, "CB"),
                element_1="XX",
                element_2="C1",
                is_known_component_bond=False,
            ),
            RDKitProximityBondWitness(
                atom_ref_1=AtomRef(cys_id, "SG"),
                atom_ref_2=AtomRef(ala_id, "CB"),
                element_1="XX",
                element_2="C",
                is_known_component_bond=False,
            ),
        ),
    )

    with pytest.raises(UnknownElementRadiusError) as error_info:
        parser_scoring_module.parser_witness_scoring_context(structure, cluster)

    message = str(error_info.value)
    assert (
        "parser-witness pre-untangle scoring has unresolved covalent radius"
        in message
    )
    assert message.count("XX") == 1
    assert "C1" in message


def test_parser_witness_pre_untangle_deduplicates_rotation_plans_per_residue(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Repeated witness endpoints in one residue should not duplicate CA-CB plans."""

    tyr_id = ResidueId("A", 1)
    other_id = ResidueId("B", 1)
    component_library = build_default_component_library()
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="TYR",
                        residue_id=tyr_id,
                        atoms=(
                            atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.5, 0.0, 0.0)),
                            atom_payload("CG", "C", Vec3(2.4, 0.5, 0.0)),
                            atom_payload("CD1", "C", Vec3(3.0, 1.2, 0.0)),
                            atom_payload("CD2", "C", Vec3(3.0, -0.2, 0.0)),
                            atom_payload("CE1", "C", Vec3(3.8, 1.1, 0.0)),
                            atom_payload("CE2", "C", Vec3(3.8, -0.1, 0.0)),
                            atom_payload("CZ", "C", Vec3(4.2, 0.5, 0.0)),
                            atom_payload("OH", "O", Vec3(5.0, 0.5, 0.0)),
                        ),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=other_id,
                        atoms=(atom_payload("C", "C", Vec3(5.0, 0.5, 0.3)),),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    cluster = RDKitProximityBondCluster(
        residue_ids=(tyr_id, other_id),
        bonds=(
            RDKitProximityBondWitness(
                atom_ref_1=AtomRef(tyr_id, "OH"),
                atom_ref_2=AtomRef(other_id, "C"),
                element_1="O",
                element_2="C",
                is_known_component_bond=False,
            ),
            RDKitProximityBondWitness(
                atom_ref_1=AtomRef(tyr_id, "CZ"),
                atom_ref_2=AtomRef(other_id, "C"),
                element_1="C",
                element_2="C",
                is_known_component_bond=False,
            ),
        ),
    )
    counted_atom_refs: list[str] = []
    module_path = (
        "protrepair.transformer.discrete.parser_witness_pre_untangle_materialization"
    )
    original_plan_builder = parser_materialization_module.sidechain_root_rotation_plan

    def counted_plan_builder(*args, **kwargs):
        atom_ref = args[1]
        counted_atom_refs.append(atom_ref.display_token())
        return original_plan_builder(*args, **kwargs)

    monkeypatch.setattr(
        f"{module_path}.sidechain_root_rotation_plan",
        counted_plan_builder,
    )

    plans = _deduplicated_sidechain_root_rotation_plans(
        structure,
        cluster,
        component_library=component_library,
    )

    assert len(plans) == 1
    assert len([token for token in counted_atom_refs if token.startswith("A:1.")]) == 1


def test_parser_extra_heavy_proximity_bond_count_uses_count_only_probe(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-untangle ranking should not rebuild parser witness clusters just to count."""

    calls: list[str] = []

    def fake_count_only_probe(*_args, **_kwargs) -> int:
        calls.append("count")
        return 7

    monkeypatch.setattr(
        parser_scoring_module,
        "measure_rdkit_no_conect_extra_heavy_proximity_bond_count",
        fake_count_only_probe,
    )
    monkeypatch.setattr(
        parser_scoring_module,
        "rdkit_no_conect_extra_proximity_bond_clusters",
        lambda *_args, **_kwargs: pytest.fail(
            "cluster materialization should not be used for count-only ranking"
        ),
        raising=False,
    )

    assert (
        parser_module._parser_extra_heavy_proximity_bond_count(
            build_structure(chains=(), source_format=FileFormat.PDB),
            component_library=build_default_component_library(),
        )
        == 7
    )
    assert calls == ["count"]


def test_transform_reuses_accepted_candidate_parser_burden_count(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Accepted pre-untangle candidates should carry the measured parser count."""

    tyr_id = ResidueId("A", 1)
    obstacle_id = ResidueId("B", 1)
    component_library = build_default_component_library()
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="TYR",
                        residue_id=tyr_id,
                        atoms=(
                            atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.5, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=obstacle_id,
                        atoms=(atom_payload("C", "C", Vec3(5.0, 0.5, 0.3)),),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    cluster = RDKitProximityBondCluster(
        residue_ids=(tyr_id, obstacle_id),
        bonds=(
            RDKitProximityBondWitness(
                atom_ref_1=AtomRef(tyr_id, "CB"),
                atom_ref_2=AtomRef(obstacle_id, "C"),
                element_1="C",
                element_2="C",
                is_known_component_bond=False,
            ),
        ),
    )
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    atom_input = atom_input_from_local_scope_spec(
        snapshot,
        LocalScopeSpec.from_residues(cluster.residue_ids),
        component_library=component_library,
    )
    observed_baseline_counts: list[int | None] = []
    observed_baseline_ranks: list[
        parser_module._ParserWitnessPreUntangleCandidateRank | None
    ] = []
    observed_pdb_block_projectors: list[object | None] = []
    observed_contact_basis_caches: list[
        parser_module._ParserWitnessContactBasisCache
    ] = []
    observed_contact_assessment_bases: list[
        parser_scoring_module.ParserWitnessContactAssessmentBasis
    ] = []
    sentinel_pdb_block_projector = object()
    accepted_score = parser_module.ParserWitnessPreUntangleScore(
        unresolved_contact_count=0,
        total_overlap_angstrom=0.0,
        worst_overlap_angstrom=0.0,
    )
    accepted_rank = parser_module._ParserWitnessPreUntangleCandidateRank(
        parser_extra_heavy_proximity_bond_count=3,
        focus_near_covalent_contact_count=0,
        focus_total_near_covalent_overlap_angstrom=0.0,
        focus_clash_count=0,
        target_score=accepted_score,
        absolute_angle_degrees=30,
        angle_degrees=30,
    )

    def fake_candidate_builder(*_args, **kwargs):
        observed_baseline_counts.append(
            kwargs["baseline_extra_heavy_proximity_bond_count"]
        )
        observed_baseline_ranks.append(kwargs["baseline_rank"])
        observed_pdb_block_projectors.append(kwargs["pdb_block_projector"])
        contact_basis_cache = kwargs["contact_basis_cache"]
        observed_contact_basis_caches.append(contact_basis_cache)
        observed_contact_assessment_bases.append(
            contact_basis_cache.basis_for(structure, cluster)
        )
        if len(observed_baseline_counts) > 1:
            return None

        return parser_module._RankedRotatedSidechainCandidate(
            candidate=parser_module.ParserWitnessPreUntangleCandidate(
                structure=structure,
                moved_atom_indices=(),
                score=accepted_score,
                parser_extra_heavy_proximity_bond_count=3,
            ),
            payload=_placeholder_rotated_payload(
                structure,
                tyr_id,
                score=accepted_score,
            ),
            rank=accepted_rank,
        )

    monkeypatch.setattr(
        parser_module,
        "rdkit_no_conect_extra_proximity_bond_clusters",
        lambda *_args, **_kwargs: (cluster, cluster),
    )
    monkeypatch.setattr(
        parser_module,
        "_build_ranked_parser_witness_pre_untangle_candidate",
        fake_candidate_builder,
    )
    monkeypatch.setattr(
        parser_module,
        "prepare_rdkit_no_conect_pdb_block_projector",
        lambda *_args, **_kwargs: sentinel_pdb_block_projector,
    )
    monkeypatch.setattr(
        parser_module,
        "_parser_extra_heavy_proximity_bond_count",
        lambda *_args, **_kwargs: pytest.fail(
            "accepted candidate count should not be recomputed"
        ),
    )

    parser_module.ParserWitnessPreUntangleTransformer(component_library).transform(
        ProteinTransformationContext(
            source_snapshot=snapshot,
            atom_input=atom_input,
            supporting_structures=(),
        )
    )

    assert observed_baseline_counts == [2, 3, 3]
    assert observed_baseline_ranks[0] is None
    assert observed_pdb_block_projectors == [
        sentinel_pdb_block_projector,
        sentinel_pdb_block_projector,
        sentinel_pdb_block_projector,
    ]
    assert observed_contact_basis_caches[0] is observed_contact_basis_caches[1]
    assert observed_contact_basis_caches[0] is observed_contact_basis_caches[2]
    assert observed_contact_assessment_bases[0] == observed_contact_assessment_bases[1]
    assert (
        observed_contact_assessment_bases[0].near_covalent_basis
        is observed_contact_assessment_bases[2].near_covalent_basis
    )
    assert (
        observed_contact_assessment_bases[0].clash_basis
        is observed_contact_assessment_bases[2].clash_basis
    )
    expected_baseline_rank = parser_module._ParserWitnessPreUntangleCandidateRank(
        parser_extra_heavy_proximity_bond_count=3,
        focus_near_covalent_contact_count=0,
        focus_total_near_covalent_overlap_angstrom=0.0,
        focus_clash_count=0,
        target_score=accepted_score,
        absolute_angle_degrees=0,
        angle_degrees=0,
    )
    assert observed_baseline_ranks[1] == expected_baseline_rank
    assert observed_baseline_ranks[2] is None


def test_exhaustive_pre_untangle_defers_full_rank_to_minimum_parser_burden(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parser-burden reduction should avoid ranking non-selectable candidates."""

    tyr_id = ResidueId("A", 1)
    obstacle_id = ResidueId("B", 1)
    component_library = build_default_component_library()
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="TYR",
                        residue_id=tyr_id,
                        atoms=(
                            atom_payload("CA", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.5, 0.0, 0.0)),
                            atom_payload("CG", "C", Vec3(2.4, 0.5, 0.0)),
                            atom_payload("OH", "O", Vec3(3.2, 0.5, 0.0)),
                        ),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=obstacle_id,
                        atoms=(atom_payload("C", "C", Vec3(3.2, 0.5, 0.2)),),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    cluster = RDKitProximityBondCluster(
        residue_ids=(tyr_id, obstacle_id),
        bonds=(
            RDKitProximityBondWitness(
                atom_ref_1=AtomRef(tyr_id, "OH"),
                atom_ref_2=AtomRef(obstacle_id, "C"),
                element_1="O",
                element_2="C",
                is_known_component_bond=False,
            ),
        ),
    )
    tyr_site = structure.constitution.residue_or_ligand(tyr_id)
    assert tyr_site is not None
    plan = parser_module._SidechainRootRotationPlan(
        residue_site=tyr_site,
        residue_geometry=structure.residue_geometry(
            structure.constitution.residue_index(tyr_id)
        ),
        residue_index=structure.constitution.residue_index(tyr_id),
        axis_atom_names=("CA", "CB"),
        rotating_atom_names=frozenset({"CG", "OH"}),
        formal_charge_by_atom_name=(),
    )
    baseline_score = parser_module.ParserWitnessPreUntangleScore(
        unresolved_contact_count=3,
        total_overlap_angstrom=3.0,
        worst_overlap_angstrom=1.0,
    )
    improved_score = parser_module.ParserWitnessPreUntangleScore(
        unresolved_contact_count=1,
        total_overlap_angstrom=1.0,
        worst_overlap_angstrom=1.0,
    )
    baseline_rank = parser_module._ParserWitnessPreUntangleCandidateRank(
        parser_extra_heavy_proximity_bond_count=4,
        focus_near_covalent_contact_count=0,
        focus_total_near_covalent_overlap_angstrom=0.0,
        focus_clash_count=0,
        target_score=baseline_score,
        absolute_angle_degrees=0,
        angle_degrees=0,
    )
    parser_counts = iter((3, 1, 2))
    ranked_parser_counts: list[int] = []

    def fake_payload_builder(*_args, **kwargs):
        return _placeholder_rotated_payload(
            structure,
            tyr_id,
            score=improved_score,
            angle_degrees=kwargs["angle_degrees"],
        )

    def fake_rank_builder(*_args, **kwargs):
        parser_count = kwargs["parser_extra_heavy_proximity_bond_count"]
        ranked_parser_counts.append(parser_count)
        return parser_module._ParserWitnessPreUntangleCandidateRank(
            parser_extra_heavy_proximity_bond_count=parser_count,
            focus_near_covalent_contact_count=0,
            focus_total_near_covalent_overlap_angstrom=0.0,
            focus_clash_count=0,
            target_score=improved_score,
            absolute_angle_degrees=abs(kwargs["angle_degrees"]),
            angle_degrees=kwargs["angle_degrees"],
        )

    monkeypatch.setattr(
        parser_module,
        "_build_rotated_sidechain_payload",
        fake_payload_builder,
    )
    monkeypatch.setattr(
        parser_module,
        "_parser_extra_heavy_proximity_bond_count",
        lambda *_args, **_kwargs: next(parser_counts),
    )
    monkeypatch.setattr(
        parser_module,
        "_parser_witness_pre_untangle_candidate_rank",
        fake_rank_builder,
    )

    candidates = parser_module._ranked_parser_witness_pre_untangle_candidates(
        structure,
        plans=(plan,),
        angle_degrees_group=(-30, 30, 60),
        exhaustive=True,
        scoring_context=parser_module._parser_witness_scoring_context(
            structure,
            cluster,
        ),
        cluster=cluster,
        baseline_score=baseline_score,
        baseline_rank=baseline_rank,
        component_library=component_library,
        clash_basis=None,
        known_bond_lookup=None,
        pdb_block_projector=None,
    )

    assert len(candidates) == 1
    assert candidates[0].rank.parser_extra_heavy_proximity_bond_count == 1
    assert ranked_parser_counts == [1]


def test_parser_extra_heavy_proximity_bond_count_accepts_known_bond_lookup(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-untangle parser counts should reuse coordinate-invariant topology facts."""

    component_library = build_default_component_library()
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    known_bond_lookup = parser_module.prepare_rdkit_no_conect_known_bond_lookup(
        structure,
        component_library=component_library,
    )
    observed_known_bond_lookups: list[parser_module.RDKitKnownBondLookup | None] = []

    def fake_measure(*_args, **kwargs):
        observed_known_bond_lookups.append(kwargs["known_bond_lookup"])
        return 9

    monkeypatch.setattr(
        parser_scoring_module,
        "measure_rdkit_no_conect_extra_heavy_proximity_bond_count",
        fake_measure,
    )

    assert (
        parser_module._parser_extra_heavy_proximity_bond_count(
            structure,
            component_library=component_library,
            known_bond_lookup=known_bond_lookup,
        )
        == 9
    )
    assert observed_known_bond_lookups == [known_bond_lookup]


def test_parser_extra_heavy_proximity_bond_count_accepts_pdb_projector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Pre-untangle parser counts should reuse coordinate-only PDB projection."""

    component_library = build_default_component_library()
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    pdb_block_projector = parser_module.prepare_rdkit_no_conect_pdb_block_projector(
        structure,
    )
    observed_pdb_block_projectors: list[
        parser_module.RDKitNoConectPDBBlockProjector | None
    ] = []

    def fake_measure(*_args, **kwargs):
        observed_pdb_block_projectors.append(kwargs["pdb_block_projector"])
        return 9

    monkeypatch.setattr(
        parser_scoring_module,
        "measure_rdkit_no_conect_extra_heavy_proximity_bond_count",
        fake_measure,
    )

    assert (
        parser_module._parser_extra_heavy_proximity_bond_count(
            structure,
            component_library=component_library,
            pdb_block_projector=pdb_block_projector,
        )
        == 9
    )
    assert observed_pdb_block_projectors == [pdb_block_projector]


@pytest.mark.parametrize("angle_degrees", (-30, 0, 30))
def test_parser_witness_contact_basis_preserves_candidate_rank(
    angle_degrees: int,
) -> None:
    """Prepared contact facts must preserve the exact candidate rank tuple."""

    structure, cluster = _parser_witness_contact_fixture()
    component_library = build_default_component_library()
    target_score = parser_witness_pre_untangle_score(structure, cluster)
    focus_residue_ids = frozenset(cluster.residue_ids)
    clash_context = prepare_clash_detection_context(
        structure,
        component_library=component_library,
    )
    focus_clashes = detect_clashes_from_context(
        clash_context,
        focus_residue_ids=focus_residue_ids,
    ).clashes
    near_covalent_contacts = detect_near_covalent_contacts_from_context(
        structure,
        clash_context,
        focus_residue_ids=focus_residue_ids,
    )
    expected_rank = parser_module._ParserWitnessPreUntangleCandidateRank(
        parser_extra_heavy_proximity_bond_count=1,
        focus_near_covalent_contact_count=len(near_covalent_contacts),
        focus_total_near_covalent_overlap_angstrom=sum(
            contact.overlap_angstrom for contact in near_covalent_contacts
        ),
        focus_clash_count=len(focus_clashes),
        target_score=target_score,
        absolute_angle_degrees=abs(angle_degrees),
        angle_degrees=angle_degrees,
    )
    contact_assessment_basis = (
        parser_scoring_module.ParserWitnessContactAssessmentBasis.prepare(
            structure,
            cluster,
            component_library=component_library,
        )
    )

    actual_rank = parser_scoring_module.parser_witness_pre_untangle_candidate_rank(
        structure,
        cluster,
        target_score=target_score,
        angle_degrees=angle_degrees,
        component_library=component_library,
        contact_assessment_basis=contact_assessment_basis,
        parser_extra_heavy_proximity_bond_count=1,
    )

    assert actual_rank == expected_rank


def test_parser_witness_contact_basis_rejects_mismatched_cluster_and_basis() -> None:
    """Prepared rank facts must remain bound to one focus and clash basis."""

    structure, cluster = _parser_witness_contact_fixture()
    component_library = build_default_component_library()
    clash_basis = prepare_clash_detection_basis(
        structure,
        component_library=component_library,
    )
    contact_assessment_basis = (
        parser_scoring_module.ParserWitnessContactAssessmentBasis.prepare(
            structure,
            cluster,
            component_library=component_library,
            clash_basis=clash_basis,
        )
    )
    target_score = parser_witness_pre_untangle_score(structure, cluster)
    mismatched_cluster = RDKitProximityBondCluster(
        residue_ids=(cluster.residue_ids[0],),
        bonds=cluster.bonds,
    )

    with pytest.raises(ValueError, match="matching cluster"):
        parser_scoring_module.parser_witness_pre_untangle_candidate_rank(
            structure,
            mismatched_cluster,
            target_score=target_score,
            angle_degrees=30,
            component_library=component_library,
            contact_assessment_basis=contact_assessment_basis,
            parser_extra_heavy_proximity_bond_count=1,
        )

    independent_clash_basis = prepare_clash_detection_basis(
        structure,
        component_library=component_library,
    )
    with pytest.raises(ValueError, match="conflicts with clash basis"):
        parser_scoring_module.parser_witness_pre_untangle_candidate_rank(
            structure,
            cluster,
            target_score=target_score,
            angle_degrees=30,
            component_library=component_library,
            clash_basis=independent_clash_basis,
            contact_assessment_basis=contact_assessment_basis,
            parser_extra_heavy_proximity_bond_count=1,
        )


def test_parser_witness_search_session_prepares_contact_basis_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One cluster search should reuse coordinate-independent contact facts."""

    structure, cluster = _parser_witness_contact_fixture()
    component_library = build_default_component_library()
    preparation_count = 0
    original_prepare = parser_scoring_module.prepare_near_covalent_contact_basis

    def counted_prepare(*args, **kwargs):
        nonlocal preparation_count
        preparation_count += 1
        return original_prepare(*args, **kwargs)

    monkeypatch.setattr(
        parser_scoring_module,
        "prepare_near_covalent_contact_basis",
        counted_prepare,
    )
    session = parser_module._ParserWitnessPreUntangleSearchSession(
        snapshot=ProteinStructureSnapshot.from_structure(structure),
        atom_input=None,
        cluster=cluster,
        component_library=component_library,
    )

    first_basis = session.resolved_contact_assessment_basis()
    second_basis = session.resolved_contact_assessment_basis()

    assert second_basis is first_basis
    assert preparation_count == 1


def test_transform_does_not_prepare_contact_basis_without_rotation_plan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-rotatable witness cluster remains a no-op before radius preparation."""

    structure, cluster = _parser_witness_contact_fixture()
    component_library = build_default_component_library()
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    atom_input = atom_input_from_local_scope_spec(
        snapshot,
        LocalScopeSpec.from_residues(cluster.residue_ids),
        component_library=component_library,
    )
    monkeypatch.setattr(
        parser_module,
        "rdkit_no_conect_extra_proximity_bond_clusters",
        lambda *_args, **_kwargs: (cluster,),
    )
    monkeypatch.setattr(
        parser_module,
        "prepare_rdkit_no_conect_known_bond_lookup",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        parser_module,
        "prepare_rdkit_no_conect_pdb_block_projector",
        lambda *_args, **_kwargs: None,
    )
    monkeypatch.setattr(
        parser_module,
        "prepare_clash_detection_basis",
        lambda *_args, **_kwargs: pytest.fail(
            "contact facts must stay lazy until a rotation plan exists"
        ),
    )

    result = parser_module.ParserWitnessPreUntangleTransformer(
        component_library
    ).transform(
        ProteinTransformationContext(
            source_snapshot=snapshot,
            atom_input=atom_input,
            supporting_structures=(),
        )
    )

    assert result is snapshot


def _parser_witness_contact_fixture(
) -> tuple[ProteinStructure, RDKitProximityBondCluster]:
    """Return a compact two-residue parser-witness contact fixture."""

    left_id = ResidueId("A", 1)
    right_id = ResidueId("B", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=left_id,
                        atoms=(atom_payload("CB", "C", Vec3(0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=right_id,
                        atoms=(atom_payload("CB", "C", Vec3(1.5, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    return structure, RDKitProximityBondCluster(
        residue_ids=(left_id, right_id),
        bonds=(
            RDKitProximityBondWitness(
                atom_ref_1=AtomRef(left_id, "CB"),
                atom_ref_2=AtomRef(right_id, "CB"),
                element_1="C",
                element_2="C",
                is_known_component_bond=False,
            ),
        ),
    )


def _placeholder_rotated_payload(
    structure: ProteinStructure,
    residue_id: ResidueId,
    *,
    score: parser_module.ParserWitnessPreUntangleScore,
    angle_degrees: int = 0,
) -> parser_module._RotatedSidechainPayload:
    """Return a minimal rotated payload for pre-untangle internal tests."""

    residue_site = structure.constitution.residue_or_ligand(residue_id)
    assert residue_site is not None
    residue_index = structure.constitution.residue_index(residue_id)
    residue_geometry = structure.residue_geometry(residue_index)
    return parser_module._RotatedSidechainPayload(
        plan=parser_module._SidechainRootRotationPlan(
            residue_site=residue_site,
            residue_geometry=residue_geometry,
            residue_index=residue_index,
            axis_atom_names=("CA", "CB"),
            rotating_atom_names=frozenset(),
            formal_charge_by_atom_name=(),
        ),
        angle_degrees=angle_degrees,
        residue_geometry=residue_geometry,
        moved_atom_names=(),
        score=score,
    )

"""Parser-witness repair candidate construction tests."""

import pytest
from tests.support.canonical_builders import (
    CanonicalResiduePayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.diagnostics.parser_readability import (
    RDKitProximityBondCluster,
    RDKitProximityBondWitness,
)
from protrepair.diagnostics.parser_topology import (
    ambiguous_disulfide_parser_witness_blocker_issues,
    ambiguous_disulfide_parser_witness_blockers,
)
from protrepair.geometry import Vec3
from protrepair.io import FileFormat
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.transformer.local import LocalScopeLowering, LocalScopeSpec
from protrepair.transformer.refinement import parser_witness as parser_witness_module
from protrepair.transformer.refinement.parser_witness import (
    ParserWitnessRepairBudget,
    ParserWitnessRepairExclusionReason,
    parser_witness_repair_candidates,
    parser_witness_repair_exclusions,
)


def test_parser_witness_budget_scales_pass_limit_from_initial_burden() -> None:
    """Parser-witness pass count should scale with initial parser burden."""

    budget = ParserWitnessRepairBudget(
        max_passes=64,
        base_passes=3,
        extra_heavy_bonds_per_pass=1,
    )

    assert budget.pass_limit_for_initial_extra_heavy_bond_count(1) == 3
    assert budget.pass_limit_for_initial_extra_heavy_bond_count(3) == 3
    assert budget.pass_limit_for_initial_extra_heavy_bond_count(4) == 4
    assert budget.pass_limit_for_initial_extra_heavy_bond_count(44) == 44
    assert budget.pass_limit_for_initial_extra_heavy_bond_count(100) == 64


def test_parser_witness_budget_keeps_max_passes_as_absolute_cap() -> None:
    """Explicit max_passes remains a hard cap even below the base pass target."""

    budget = ParserWitnessRepairBudget(
        max_passes=2,
        base_passes=3,
        extra_heavy_bonds_per_pass=1,
    )

    assert budget.pass_limit_for_initial_extra_heavy_bond_count(10) == 2


def test_parser_witness_budget_validates_scaled_pass_settings() -> None:
    """Invalid scaled parser-witness budgets should fail at construction/use."""

    with pytest.raises(ValueError, match="base_passes"):
        ParserWitnessRepairBudget(base_passes=0)

    with pytest.raises(ValueError, match="extra_heavy_bonds_per_pass"):
        ParserWitnessRepairBudget(extra_heavy_bonds_per_pass=0)

    with pytest.raises(ValueError, match="non-negative"):
        ParserWitnessRepairBudget().pass_limit_for_initial_extra_heavy_bond_count(-1)


def test_parser_witness_sidechain_only_polymer_cluster_uses_sidechain_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Side-chain parser contacts should not make polymer backbones movable."""

    structure = _sidechain_polymer_structure()
    left_residue_id = ResidueId("A", 1)
    right_residue_id = ResidueId("A", 2)
    cluster = RDKitProximityBondCluster(
        residue_ids=(left_residue_id, right_residue_id),
        bonds=(
            RDKitProximityBondWitness(
                atom_ref_1=AtomRef(left_residue_id, "CB"),
                atom_ref_2=AtomRef(right_residue_id, "SG"),
                element_1="C",
                element_2="S",
                is_known_component_bond=False,
            ),
        ),
    )
    _stub_parser_witness_clusters(monkeypatch, (cluster,))

    candidates = parser_witness_repair_candidates(
        structure,
        component_library=build_default_component_library(),
    )

    assert len(candidates) == 1
    assert candidates[0].repair_refinement.scope_spec == (
        LocalScopeSpec.from_residue_sidechains(
            (left_residue_id, right_residue_id)
        )
    )
    assert candidates[0].repair_refinement.config.max_iterations == 50
    assert candidates[0].repair_refinement.execution_scope_spec is None


def test_parser_witness_backbone_cluster_keeps_whole_residue_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backbone parser contacts require whole-residue motion semantics."""

    structure = _backbone_polymer_structure()
    left_residue_id = ResidueId("A", 1)
    right_residue_id = ResidueId("A", 2)
    cluster = RDKitProximityBondCluster(
        residue_ids=(left_residue_id, right_residue_id),
        bonds=(
            RDKitProximityBondWitness(
                atom_ref_1=AtomRef(left_residue_id, "N"),
                atom_ref_2=AtomRef(right_residue_id, "OD1"),
                element_1="N",
                element_2="O",
                is_known_component_bond=False,
            ),
        ),
    )
    _stub_parser_witness_clusters(monkeypatch, (cluster,))

    candidates = parser_witness_repair_candidates(
        structure,
        component_library=build_default_component_library(),
    )

    assert len(candidates) == 1
    assert candidates[0].repair_refinement.scope_spec == (
        LocalScopeSpec.from_residues((left_residue_id, right_residue_id))
    )
    assert candidates[0].repair_refinement.config.max_iterations == 20


def test_parser_witness_retained_non_polymer_cluster_keeps_whole_residue_scope(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Polymer side-chain lowering must not be projected onto ligand atoms."""

    structure = _retained_non_polymer_structure()
    residue_id = ResidueId("L", 1)
    cluster = RDKitProximityBondCluster(
        residue_ids=(residue_id,),
        bonds=(
            RDKitProximityBondWitness(
                atom_ref_1=AtomRef(residue_id, "C1"),
                atom_ref_2=AtomRef(residue_id, "O1"),
                element_1="C",
                element_2="O",
                is_known_component_bond=False,
            ),
        ),
    )
    _stub_parser_witness_clusters(monkeypatch, (cluster,))

    candidates = parser_witness_repair_candidates(
        structure,
        component_library=build_default_component_library(),
    )

    assert len(candidates) == 1
    assert (
        candidates[0].repair_refinement.scope_spec.lowering
        is LocalScopeLowering.RESIDUE_ATOMS
    )


def test_parser_witness_skips_ambiguous_disulfide_blocker_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Ambiguous disulfide residuals should not be ordinary FF candidates."""

    structure = _ambiguous_disulfide_structure()
    ambiguous_cluster = _ambiguous_disulfide_parser_cluster()
    sidechain_cluster = _sidechain_parser_cluster(
        ResidueId("A", 4),
        ResidueId("A", 5),
    )
    _stub_parser_witness_clusters(
        monkeypatch,
        (ambiguous_cluster, sidechain_cluster),
    )

    candidates = parser_witness_repair_candidates(
        structure,
        component_library=build_default_component_library(),
    )

    assert len(candidates) == 1
    assert candidates[0].cluster == sidechain_cluster
    assert candidates[0].repair_refinement.scope_spec == (
        LocalScopeSpec.from_residue_sidechains(
            (ResidueId("A", 4), ResidueId("A", 5))
        )
    )


def test_parser_witness_exclusions_explain_ambiguous_disulfide_blocker(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate filtering should expose typed ambiguous-disulfide reasons."""

    structure = _ambiguous_disulfide_structure()
    ambiguous_cluster = _ambiguous_disulfide_parser_cluster()
    sidechain_cluster = _sidechain_parser_cluster(
        ResidueId("A", 4),
        ResidueId("A", 5),
    )
    _stub_parser_witness_clusters(
        monkeypatch,
        (ambiguous_cluster, sidechain_cluster),
    )

    exclusions = parser_witness_repair_exclusions(
        structure,
        component_library=build_default_component_library(),
    )

    assert len(exclusions) == 1
    assert exclusions[0].cluster == ambiguous_cluster
    assert (
        exclusions[0].reason
        is ParserWitnessRepairExclusionReason.AMBIGUOUS_DISULFIDE_TOPOLOGY
    )
    assert exclusions[0].display_token() == (
        "ambiguous_disulfide_topology:A:1,A:2"
    )


def test_parser_witness_exclusions_explain_oversized_cluster(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Candidate filtering should expose cluster-size budget reasons."""

    structure = _sidechain_polymer_structure()
    oversized_cluster = RDKitProximityBondCluster(
        residue_ids=tuple(ResidueId("A", seq_num) for seq_num in range(1, 8)),
        bonds=(
            RDKitProximityBondWitness(
                atom_ref_1=AtomRef(ResidueId("A", 1), "CB"),
                atom_ref_2=AtomRef(ResidueId("A", 7), "CB"),
                element_1="C",
                element_2="C",
                is_known_component_bond=False,
            ),
        ),
    )
    _stub_parser_witness_clusters(monkeypatch, (oversized_cluster,))

    exclusions = parser_witness_repair_exclusions(
        structure,
        component_library=build_default_component_library(),
        budget=ParserWitnessRepairBudget(max_cluster_residues=6),
    )
    candidates = parser_witness_repair_candidates(
        structure,
        component_library=build_default_component_library(),
        budget=ParserWitnessRepairBudget(max_cluster_residues=6),
    )

    assert candidates == ()
    assert len(exclusions) == 1
    assert exclusions[0].cluster == oversized_cluster
    assert exclusions[0].reason is ParserWitnessRepairExclusionReason.CLUSTER_TOO_LARGE


def test_parser_witness_keeps_non_ambiguous_disulfide_neighbor_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Unique disulfide neighborhoods should not be filtered as ambiguous."""

    structure = _likely_disulfide_structure()
    cluster = RDKitProximityBondCluster(
        residue_ids=(ResidueId("A", 1), ResidueId("A", 2)),
        bonds=(
            RDKitProximityBondWitness(
                atom_ref_1=AtomRef(ResidueId("A", 1), "SG"),
                atom_ref_2=AtomRef(ResidueId("A", 2), "CB"),
                element_1="S",
                element_2="C",
                is_known_component_bond=False,
            ),
        ),
    )
    _stub_parser_witness_clusters(monkeypatch, (cluster,))

    candidates = parser_witness_repair_candidates(
        structure,
        component_library=build_default_component_library(),
    )

    assert len(candidates) == 1
    assert candidates[0].cluster == cluster


def test_ambiguous_disulfide_parser_witness_issue_explains_blocker() -> None:
    """Parser/topology cross-diagnostic should explain skipped CYS witnesses."""

    structure = _ambiguous_disulfide_structure()

    issues = ambiguous_disulfide_parser_witness_blocker_issues(
        structure,
        clusters=(_ambiguous_disulfide_parser_cluster(),),
    )

    assert len(issues) == 1
    assert issues[0].kind is ValidationIssueKind.AMBIGUOUS_DISULFIDE
    assert issues[0].scope.targets_residue(ResidueId("A", 1))
    assert issues[0].scope.targets_residue(ResidueId("A", 2))
    assert issues[0].scope.targets_residue(ResidueId("A", 3))
    assert "ordinary parser-witness local FF repair was skipped" in issues[0].message
    assert "A:1.SG-A:2.CB" in issues[0].message


def test_ambiguous_disulfide_parser_witness_blocker_is_structured() -> None:
    """Parser/topology blockers should be reusable outside issue projection."""

    structure = _ambiguous_disulfide_structure()

    blockers = ambiguous_disulfide_parser_witness_blockers(
        structure,
        clusters=(_ambiguous_disulfide_parser_cluster(),),
    )

    assert len(blockers) == 1
    assert blockers[0].possible_disulfide_residue_ids == (
        ResidueId("A", 1),
        ResidueId("A", 2),
        ResidueId("A", 3),
    )
    assert tuple(witness.display_token() for witness in blockers[0].witnesses) == (
        "A:1.SG-A:2.CB",
    )


def _stub_parser_witness_clusters(
    monkeypatch: pytest.MonkeyPatch,
    clusters: tuple[RDKitProximityBondCluster, ...],
) -> None:
    """Patch RDKit parser witnesses with deterministic test clusters."""

    def fake_rdkit_no_conect_extra_proximity_bond_clusters(
        structure: ProteinStructure,
        *,
        component_library: ComponentLibrary | None = None,
    ) -> tuple[RDKitProximityBondCluster, ...]:
        del structure
        del component_library
        return clusters

    monkeypatch.setattr(
        parser_witness_module,
        "rdkit_no_conect_extra_proximity_bond_clusters",
        fake_rdkit_no_conect_extra_proximity_bond_clusters,
    )


def _sidechain_polymer_structure() -> ProteinStructure:
    """Return one polymer structure with side-chain atoms on both residues."""

    return build_structure(
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
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.0, 1.0, 0.0)),
                        ),
                    ),
                    residue_payload(
                        component_id="CYS",
                        residue_id=ResidueId("A", 2),
                        atoms=(
                            atom_payload("N", "N", Vec3(4.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(5.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(6.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(7.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(5.0, 1.0, 0.0)),
                            atom_payload("SG", "S", Vec3(5.0, 2.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="parser-witness-sidechain-scope",
    )


def _backbone_polymer_structure() -> ProteinStructure:
    """Return one polymer structure with a backbone/contact atom pair."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="GLY",
                        residue_id=ResidueId("A", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                        ),
                    ),
                    residue_payload(
                        component_id="ASP",
                        residue_id=ResidueId("A", 2),
                        atoms=(
                            atom_payload("N", "N", Vec3(4.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(5.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(6.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(7.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(5.0, 1.0, 0.0)),
                            atom_payload("CG", "C", Vec3(5.0, 2.0, 0.0)),
                            atom_payload("OD1", "O", Vec3(4.0, 2.5, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="parser-witness-backbone-scope",
    )


def _retained_non_polymer_structure() -> ProteinStructure:
    """Return one retained non-polymer structure for scope classification."""

    return build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="LIG",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("O1", "O", Vec3(1.0, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="parser-witness-retained-non-polymer-scope",
    )


def _ambiguous_disulfide_structure() -> ProteinStructure:
    """Return one CYS triad with ambiguous disulfide-like SG distances."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _cys_residue(
                        1,
                        Vec3(0.0, 0.0, 0.0),
                        cb_position=Vec3(0.0, 1.5, 0.0),
                    ),
                    _cys_residue(
                        2,
                        Vec3(2.0, 0.0, 0.0),
                        cb_position=Vec3(2.0, 1.5, 0.0),
                    ),
                    _cys_residue(
                        3,
                        Vec3(0.0, 2.0, 0.0),
                        cb_position=Vec3(0.0, 3.5, 0.0),
                    ),
                    _cys_residue(
                        4,
                        Vec3(8.0, 0.0, 0.0),
                        cb_position=Vec3(8.0, 1.5, 0.0),
                    ),
                    _cys_residue(
                        5,
                        Vec3(10.0, 0.0, 0.0),
                        cb_position=Vec3(10.0, 1.5, 0.0),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="parser-witness-ambiguous-disulfide",
    )


def _likely_disulfide_structure() -> ProteinStructure:
    """Return one unique disulfide-like CYS pair."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _cys_residue(
                        1,
                        Vec3(0.0, 0.0, 0.0),
                        cb_position=Vec3(0.0, 1.5, 0.0),
                    ),
                    _cys_residue(
                        2,
                        Vec3(2.0, 0.0, 0.0),
                        cb_position=Vec3(2.0, 1.5, 0.0),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="parser-witness-likely-disulfide",
    )


def _ambiguous_disulfide_parser_cluster() -> RDKitProximityBondCluster:
    """Return one SG-to-CB parser witness inside an ambiguous CYS triad."""

    return RDKitProximityBondCluster(
        residue_ids=(ResidueId("A", 1), ResidueId("A", 2)),
        bonds=(
            RDKitProximityBondWitness(
                atom_ref_1=AtomRef(ResidueId("A", 1), "SG"),
                atom_ref_2=AtomRef(ResidueId("A", 2), "CB"),
                element_1="S",
                element_2="C",
                is_known_component_bond=False,
            ),
        ),
    )


def _sidechain_parser_cluster(
    left_residue_id: ResidueId,
    right_residue_id: ResidueId,
) -> RDKitProximityBondCluster:
    """Return one ordinary side-chain parser witness cluster."""

    return RDKitProximityBondCluster(
        residue_ids=(left_residue_id, right_residue_id),
        bonds=(
            RDKitProximityBondWitness(
                atom_ref_1=AtomRef(left_residue_id, "CB"),
                atom_ref_2=AtomRef(right_residue_id, "SG"),
                element_1="C",
                element_2="S",
                is_known_component_bond=False,
            ),
        ),
    )


def _cys_residue(
    seq_num: int,
    sg_position: Vec3,
    *,
    cb_position: Vec3,
) -> CanonicalResiduePayload:
    """Return one compact CYS residue payload for parser-witness tests."""

    return residue_payload(
        component_id="CYS",
        residue_id=ResidueId("A", seq_num),
        atoms=(
            atom_payload("N", "N", Vec3(float(seq_num), -3.0, 0.0)),
            atom_payload("CA", "C", Vec3(float(seq_num), -2.0, 0.0)),
            atom_payload("C", "C", Vec3(float(seq_num), -1.0, 0.0)),
            atom_payload("O", "O", Vec3(float(seq_num), -0.5, 0.0)),
            atom_payload("CB", "C", cb_position),
            atom_payload("SG", "S", sg_position),
        ),
    )

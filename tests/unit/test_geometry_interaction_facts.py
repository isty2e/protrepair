"""Unit tests for intrinsic-geometry and interaction fact owners."""

from math import sqrt
from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    CanonicalAtomPayload,
    CanonicalResiduePayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.refinement_benchmarks import load_case_structure
from tests.support.refinement_cases import REFINEMENT_BENCHMARK_CASES
from typing_extensions import assert_type

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.diagnostics import ClashPolicy, ClashReport, detect_clashes
from protrepair.diagnostics.clash_topology_rules import ClashTopologyAtomSite
from protrepair.diagnostics.parser_readability import (
    RDKitNoConectSanitizeReadabilityMetrics,
)
from protrepair.geometry import Vec3
from protrepair.io import FileFormat, read_structure
from protrepair.state import (
    ClashObservationMode,
    ClashPresenceState,
    OrientationCorrectionEligibilityState,
    ParserCompatibilityProfile,
    ParserCompatibilityState,
    StereochemistryState,
    StructureInteractionFacts,
    StructureIntrinsicGeometryFacts,
    StructureParserCompatibilityFacts,
)
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.completion.hydrogen.rotatable import (
    build_rotatable_hydrogen_search,
    rotatable_hydrogen_placement_spec,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.workflow.contracts import StructureIngressOptions
from protrepair.workflow.planning.intrinsic_geometry import (
    derive_structure_intrinsic_geometry_facts,
)


def test_intrinsic_geometry_facts_detect_synthetic_protein_self_clash() -> None:
    """Intrinsic facts should isolate polymer self-clash burden."""

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
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.0, 1.0, 0.0)),
                        ),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("B", 1),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.2, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.2, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.2, 0.0, 0.0)),
                            atom_payload("O", "O", Vec3(3.2, 0.0, 0.0)),
                            atom_payload("CB", "C", Vec3(1.2, 1.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="intrinsic-self-clash",
    )

    facts = StructureIntrinsicGeometryFacts.from_structure(structure)

    assert facts.protein_self_clash_state is ClashPresenceState.PRESENT
    assert facts.protein_self_clash_count > 0
    assert (
        facts.protein_self_clash_observation_mode
        is ClashObservationMode.HEAVY_ATOM_LOWER_BOUND
    )
    assert facts.observed_heavy_atom_self_clash_count == facts.protein_self_clash_count
    assert facts.observed_hydrogen_inclusive_self_clash_count is None
    assert (
        facts.orientation_correction_eligibility_state
        is OrientationCorrectionEligibilityState.NOT_ELIGIBLE
    )


def test_clash_topology_atom_site_protocol_exposes_canonical_residue_id() -> None:
    """Topology clash rules should require canonical residue identity."""

    def assert_protocol_contract(site: ClashTopologyAtomSite) -> None:
        assert_type(site.residue_id, ResidueId)

    assert callable(assert_protocol_contract)


def test_parser_compatibility_facts_preserve_rdkit_profile_observation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Parser compatibility facts should stay separate from chemistry readiness."""

    structure = build_structure(
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
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="parser-compatibility-facts",
    )

    def _measure_parser_metrics(
        *_args: object,
        **_kwargs: object,
    ) -> RDKitNoConectSanitizeReadabilityMetrics:
        return RDKitNoConectSanitizeReadabilityMetrics(
            sanitize_readable=False,
            extra_proximity_bond_count=3,
            extra_heavy_proximity_bond_count=2,
        )

    monkeypatch.setattr(
        "protrepair.state.structure_parser.measure_rdkit_no_conect_sanitize_readability_metrics",
        _measure_parser_metrics,
    )

    facts = StructureParserCompatibilityFacts.from_structure(structure)

    assert facts.carrier is structure
    assert facts.profile is ParserCompatibilityProfile.RDKIT_NO_CONECT_SANITIZE
    assert facts.compatibility_state is ParserCompatibilityState.INCOMPATIBLE
    assert facts.extra_proximity_bond_count == 3
    assert facts.extra_heavy_proximity_bond_count == 2
    assert facts.has_parser_visible_proximity_burden()


def test_intrinsic_geometry_facts_detect_ligand_free_orientation_pathology() -> None:
    """Intrinsic facts should surface ligand-free axis-rotation pathologies."""

    structure = load_case_structure(REFINEMENT_BENCHMARK_CASES["3g8l-asn182"])

    facts = derive_structure_intrinsic_geometry_facts(
        structure,
        component_library=build_default_component_library(),
    )

    assert (
        facts.orientation_correction_eligibility_state
        is OrientationCorrectionEligibilityState.ELIGIBLE
    )


def test_intrinsic_geometry_facts_detect_stereochemistry_violation() -> None:
    """Intrinsic facts should preserve stereochemistry diagnostics."""

    structure = read_structure(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    residue_id = ResidueId("A", 30)
    residue_index = structure.constitution.residue_index(residue_id)
    residue_site = structure.constitution.chain("A").residue(residue_id)
    residue_geometry = structure.geometry.residue_geometry(
        constitution=structure.constitution,
        residue_index=residue_index,
    )
    inverted = structure.with_updated_residue_facets(
        residue_site,
        residue_geometry=residue_geometry.with_atom_geometries(
            (
                ("OG1", residue_geometry.atom_geometry("CG2")),
                ("CG2", residue_geometry.atom_geometry("OG1")),
            )
        ),
        formal_charge_by_atom_name=structure.topology.residue_formal_charge_by_atom_name(
            constitution=structure.constitution,
            residue_index=residue_index,
        ),
    )
    focused_structure = _focused_chain_structure(
        inverted,
        chain_id="A",
        residue_ids={residue_id},
        source_name="inverted-thr",
    )

    facts = StructureIntrinsicGeometryFacts.from_structure(
        focused_structure,
        component_library=build_default_component_library(),
    )

    assert facts.stereochemistry_state is StereochemistryState.VIOLATED


def test_intrinsic_geometry_facts_reuse_all_atom_clashes_for_heavy_projection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Hydrogen-complete intrinsic facts should not rescan heavy-only clashes."""

    library = build_default_component_library()
    hydrogenated = add_hydrogens(
        _build_serine_window_structure(),
        component_library=library,
    ).structure
    observed_policies: list[ClashPolicy | None] = []

    def _detect_clashes_spy(
        structure: ProteinStructure,
        *,
        component_library: ComponentLibrary,
        policy: ClashPolicy | None = None,
    ) -> ClashReport:
        observed_policies.append(policy)
        return detect_clashes(
            structure,
            component_library=component_library,
            policy=policy,
        )

    monkeypatch.setattr(
        "protrepair.state.structure_geometry.detect_clashes",
        _detect_clashes_spy,
    )

    facts = StructureIntrinsicGeometryFacts.from_structure(
        hydrogenated,
        component_library=library,
    )

    assert len(observed_policies) == 1
    assert observed_policies[0] == ClashPolicy(
        include_hydrogens=True,
        include_ligands=False,
    )
    assert (
        facts.protein_self_clash_observation_mode
        is ClashObservationMode.ALL_ATOM_COMPLETE
    )
    assert facts.observed_hydrogen_inclusive_self_clash_count is not None
    assert (
        facts.observed_heavy_atom_self_clash_count
        <= facts.observed_hydrogen_inclusive_self_clash_count
    )


def test_interaction_facts_use_one_canonical_burden_with_observation_detail() -> None:
    """Interaction facts should expose one burden plus heavy/all-atom detail."""

    library = build_default_component_library()
    heavy_only = build_structure(
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
                ),
            ),
        ),
        ligands=(
            build_ligand_residue(
                "L",
                1,
                (atom_payload("C1", "C", Vec3(1.2, 1.0, 0.0)),),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="heavy-ligand-clash",
    )

    heavy_facts = StructureInteractionFacts.from_structure(
        heavy_only,
        component_library=library,
    )

    assert heavy_facts.ligand_aware_clash_state is ClashPresenceState.PRESENT
    assert heavy_facts.ligand_aware_clash_count > 0
    assert heavy_facts.ligand_aware_worst_overlap_angstrom > 0.0
    assert heavy_facts.ligand_aware_total_overlap_angstrom > 0.0
    assert (
        heavy_facts.ligand_aware_clash_observation_mode
        is ClashObservationMode.HEAVY_ATOM_LOWER_BOUND
    )
    assert (
        heavy_facts.observed_hydrogen_inclusive_ligand_worst_overlap_angstrom
        >= heavy_facts.observed_heavy_atom_ligand_worst_overlap_angstrom
    )
    assert (
        heavy_facts.observed_hydrogen_inclusive_ligand_total_overlap_angstrom
        >= heavy_facts.observed_heavy_atom_ligand_total_overlap_angstrom
    )
    assert (
        heavy_facts.observed_heavy_atom_ligand_clash_count
        == heavy_facts.ligand_aware_clash_count
    )

    protein_only = _build_serine_window_structure()
    hydrogenated = add_hydrogens(
        protein_only,
        component_library=library,
    ).structure
    residue = _residue_payload_from_structure(hydrogenated, ResidueId("A", 17))
    ligand_payload = build_ligand_residue(
        "L",
        1,
        (_resolvable_ligand_atom(residue, component_library=library),),
    )
    hydrogen_only_holo = hydrogenated.with_ligand_facets(
        ligand_sites=(ligand_payload[0],),
        ligand_geometries=(ligand_payload[1],),
        ligand_formal_charge_payloads=(ligand_payload[2],),
    )

    hydrogen_facts = StructureInteractionFacts.from_structure(
        hydrogen_only_holo,
        component_library=library,
    )

    assert hydrogen_facts.ligand_aware_clash_state is ClashPresenceState.PRESENT
    assert (
        hydrogen_facts.ligand_aware_clash_observation_mode
        is ClashObservationMode.ALL_ATOM_COMPLETE
    )
    assert (
        hydrogen_facts.ligand_aware_clash_count
        == hydrogen_facts.observed_hydrogen_inclusive_ligand_clash_count
    )
    assert (
        hydrogen_facts.observed_hydrogen_inclusive_ligand_clash_count
        > hydrogen_facts.observed_heavy_atom_ligand_clash_count
    )
    assert (
        hydrogen_facts.observed_hydrogen_inclusive_ligand_worst_overlap_angstrom
        >= hydrogen_facts.observed_heavy_atom_ligand_worst_overlap_angstrom
    )
    assert (
        hydrogen_facts.observed_hydrogen_inclusive_ligand_total_overlap_angstrom
        > hydrogen_facts.observed_heavy_atom_ligand_total_overlap_angstrom
    )


def _focused_chain_structure(
    structure: ProteinStructure,
    *,
    chain_id: str,
    residue_ids: set[ResidueId],
    source_name: str,
) -> ProteinStructure:
    """Return one focused structure over selected residues in one chain."""

    residue_payloads: list[CanonicalResiduePayload] = []
    for residue_site in structure.constitution.chain(chain_id).residues:
        if residue_site.residue_id not in residue_ids:
            continue

        residue_index = structure.constitution.residue_index(residue_site.residue_id)
        residue_payloads.append(
            (
                residue_site,
                structure.geometry.residue_geometry(
                    constitution=structure.constitution,
                    residue_index=residue_index,
                ),
                structure.topology.residue_formal_charge_by_atom_name(
                    constitution=structure.constitution,
                    residue_index=residue_index,
                ),
            )
        )

    return build_structure(
        chains=(chain_payload(chain_id, tuple(residue_payloads)),),
        source_format=structure.provenance.ingress.source_format,
        source_name=source_name,
    )


def _build_serine_window_structure() -> ProteinStructure:
    """Return one real heavy-only chain window centered on Ser17."""

    structure = read_structure(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    residue_id = ResidueId("A", 17)
    chain_site = structure.chain_site("A")
    center_index = next(
        index
        for index, residue_site in enumerate(chain_site.residues)
        if residue_site.residue_id == residue_id
    )
    window_residue_sites = chain_site.residues[
        max(0, center_index - 1) : center_index + 2
    ]
    window_payloads = tuple(
        _canonical_residue_payload_from_structure(
            structure,
            residue_site.residue_id,
        )
        for residue_site in window_residue_sites
    )
    return build_structure(
        chains=(chain_payload("A", window_payloads),),
        source_format=structure.provenance.ingress.source_format,
        source_name="pdb1afc-ser-window",
    )


def build_ligand_residue(
    chain_id: str,
    seq_num: int,
    atoms: tuple[CanonicalAtomPayload, ...],
) -> CanonicalResiduePayload:
    """Return one ligand residue from prebuilt atoms."""

    return residue_payload(
        component_id="LIG",
        residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
        atoms=atoms,
        is_hetero=True,
    )


def _resolvable_ligand_atom(
    residue: CompletionResiduePayload,
    *,
    component_library: ComponentLibrary,
) -> CanonicalAtomPayload:
    """Return one ligand atom that clashes with the current HG only."""

    template = component_library.require(residue.component_id)
    spec = rotatable_hydrogen_placement_spec(template.hydrogen_semantics)
    assert spec is not None

    search = build_rotatable_hydrogen_search(residue, spec=spec)
    assert search is not None

    hydrogen = residue.atom_geometry(spec.hydrogen_atom_name).position
    unique_candidates: list[Vec3] = []
    for candidate in search.candidate_positions():
        candidate_position = Vec3.from_iterable(candidate)
        if all(
            candidate_position.distance_to(existing) > 1e-6
            for existing in unique_candidates
        ):
            unique_candidates.append(candidate_position)

    alternative = max(
        unique_candidates,
        key=lambda candidate_position: candidate_position.distance_to(hydrogen),
    )
    offset_x = hydrogen.x - alternative.x
    offset_y = hydrogen.y - alternative.y
    offset_z = hydrogen.z - alternative.z
    norm = sqrt((offset_x * offset_x) + (offset_y * offset_y) + (offset_z * offset_z))
    return atom_payload(
        "C1",
        "C",
        Vec3(
            hydrogen.x + (offset_x / norm * 0.2),
            hydrogen.y + (offset_y / norm * 0.2),
            hydrogen.z + (offset_z / norm * 0.2),
        ),
    )


def _residue_payload_from_structure(
    structure: ProteinStructure,
    residue_id: ResidueId,
) -> CompletionResiduePayload:
    """Return one completion payload projected from structure facets."""

    residue_index = structure.constitution.residue_index(residue_id)
    residue_site = structure.constitution.residue_or_ligand(residue_id)
    assert residue_site is not None
    return CompletionResiduePayload(
        residue_site=residue_site,
        residue_geometry=structure.geometry.residue_geometry(
            constitution=structure.constitution,
            residue_index=residue_index,
        ),
        formal_charge_by_atom_name=structure.topology.residue_formal_charge_by_atom_name(
            constitution=structure.constitution,
            residue_index=residue_index,
        ),
    )


def _canonical_residue_payload_from_structure(
    structure: ProteinStructure,
    residue_id: ResidueId,
) -> CanonicalResiduePayload:
    """Return one canonical residue payload projected from structure facets."""

    residue_index = structure.constitution.residue_index(residue_id)
    residue_site = structure.constitution.residue_or_ligand(residue_id)
    assert residue_site is not None
    return (
        residue_site,
        structure.geometry.residue_geometry(
            constitution=structure.constitution,
            residue_index=residue_index,
        ),
        structure.topology.residue_formal_charge_by_atom_name(
            constitution=structure.constitution,
            residue_index=residue_index,
        ),
    )

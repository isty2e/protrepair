"""Steric-clash diagnostics over canonical repaired structures."""

from dataclasses import dataclass, replace
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

from protrepair.chemistry import UnknownElementRadiusError
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.diagnostics import (
    ClashPolicy,
    EventScopeKind,
    ValidationIssueKind,
    detect_clashes,
    detect_clashes_involving_residues,
)
from protrepair.diagnostics.clash_pair_generation import (
    ContactDomain,
    PreparedAtomSitePairIndex,
    iter_candidate_atom_site_pairs,
    pair_can_be_rejected_before_distance,
)
from protrepair.diagnostics.clashes import (
    bind_clash_detection_context,
    bind_clash_detection_frame,
    detect_clashes_from_context,
    has_clashes_in_context,
    prepare_clash_detection_basis,
    prepare_clash_detection_context,
    prepare_projected_clash_detection_context,
)
from protrepair.geometry import Vec3
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.heavy import repair_heavy_atoms


@dataclass(frozen=True, slots=True)
class _PairPolicy:
    include_ligands: bool = False
    include_hydrogen_hydrogen: bool = False


@dataclass(frozen=True, slots=True)
class _PairSite:
    residue_id: ResidueId
    domain: ContactDomain
    is_hydrogen_atom: bool = False
    grid_cell: tuple[int, int, int] = (0, 0, 0)


def test_contact_domain_normalizes_boundary_aliases_only_at_ingress() -> None:
    """Raw provider terms should normalize before pair-generation hot logic."""

    assert ContactDomain.normalize("polymer") is ContactDomain.POLYMER
    assert ContactDomain.normalize("ligand") is ContactDomain.RETAINED_NON_POLYMER
    assert (
        ContactDomain.normalize("retained-non-polymer")
        is ContactDomain.RETAINED_NON_POLYMER
    )
    assert ContactDomain.normalize("unknown") is ContactDomain.UNKNOWN
    assert ContactDomain.normalize("not-applicable") is ContactDomain.NOT_APPLICABLE

    with pytest.raises(TypeError, match="ContactDomain or string"):
        ContactDomain.normalize(7)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="unsupported contact domain"):
        ContactDomain.normalize("cofactor-ish")


def test_clash_pair_generation_filters_typed_contact_domains() -> None:
    """Ligand-disabled pair filtering should use typed contact domains."""

    policy = _PairPolicy(include_ligands=False)
    polymer_a = _PairSite(ResidueId("A", 1), ContactDomain.POLYMER)
    polymer_b = _PairSite(ResidueId("A", 2), ContactDomain.POLYMER)
    retained = _PairSite(ResidueId("L", 1), ContactDomain.RETAINED_NON_POLYMER)
    retained_b = _PairSite(ResidueId("L", 2), ContactDomain.RETAINED_NON_POLYMER)
    unknown = _PairSite(ResidueId("U", 1), ContactDomain.UNKNOWN)
    not_applicable = _PairSite(ResidueId("N", 1), ContactDomain.NOT_APPLICABLE)

    assert not pair_can_be_rejected_before_distance(
        polymer_a,
        polymer_b,
        policy=policy,
    )
    assert pair_can_be_rejected_before_distance(polymer_a, retained, policy=policy)
    assert pair_can_be_rejected_before_distance(retained, retained_b, policy=policy)
    assert not pair_can_be_rejected_before_distance(polymer_a, unknown, policy=policy)
    assert not pair_can_be_rejected_before_distance(
        unknown,
        not_applicable,
        policy=policy,
    )


def test_clash_pair_generation_keeps_ligand_pairs_when_policy_includes_them() -> None:
    """The contact-domain contract should not reject retained pairs when enabled."""

    policy = _PairPolicy(include_ligands=True)
    polymer = _PairSite(ResidueId("A", 1), ContactDomain.POLYMER)
    retained = _PairSite(ResidueId("L", 1), ContactDomain.RETAINED_NON_POLYMER)

    assert not pair_can_be_rejected_before_distance(polymer, retained, policy=policy)


def test_clash_pair_generation_still_applies_non_domain_fast_rejections() -> None:
    """Typed domain filtering must not remove existing same-residue/H-H pruning."""

    policy = _PairPolicy(include_ligands=True, include_hydrogen_hydrogen=False)
    same_residue_left = _PairSite(ResidueId("A", 1), ContactDomain.POLYMER)
    same_residue_right = _PairSite(ResidueId("A", 1), ContactDomain.POLYMER)
    hydrogen_left = _PairSite(
        ResidueId("A", 2),
        ContactDomain.UNKNOWN,
        is_hydrogen_atom=True,
    )
    hydrogen_right = _PairSite(
        ResidueId("B", 2),
        ContactDomain.NOT_APPLICABLE,
        is_hydrogen_atom=True,
    )

    assert pair_can_be_rejected_before_distance(
        same_residue_left,
        same_residue_right,
        policy=policy,
    )
    assert pair_can_be_rejected_before_distance(
        hydrogen_left,
        hydrogen_right,
        policy=policy,
    )


@pytest.mark.parametrize(
    ("focus_residue_ids", "policy"),
    (
        (None, _PairPolicy()),
        (frozenset({ResidueId("A", 1)}), _PairPolicy()),
        (
            frozenset({ResidueId("A", 2), ResidueId("L", 1)}),
            _PairPolicy(include_ligands=True),
        ),
        (frozenset(), _PairPolicy(include_ligands=True)),
        (None, _PairPolicy(include_hydrogen_hydrogen=True)),
        (
            frozenset({ResidueId("H", 1), ResidueId("H", 2)}),
            _PairPolicy(include_hydrogen_hydrogen=True),
        ),
    ),
)
def test_prepared_pair_index_preserves_streaming_pair_order(
    focus_residue_ids: frozenset[ResidueId] | None,
    policy: _PairPolicy,
) -> None:
    """Prepared spatial reuse must preserve exact streaming pair order."""

    atom_sites = (
        _PairSite(ResidueId("A", 1), ContactDomain.POLYMER, grid_cell=(0, 0, 0)),
        _PairSite(ResidueId("A", 2), ContactDomain.POLYMER, grid_cell=(0, 0, 0)),
        _PairSite(
            ResidueId("L", 1),
            ContactDomain.RETAINED_NON_POLYMER,
            grid_cell=(1, 0, 0),
        ),
        _PairSite(
            ResidueId("H", 1),
            ContactDomain.POLYMER,
            is_hydrogen_atom=True,
            grid_cell=(1, 0, 0),
        ),
        _PairSite(
            ResidueId("H", 2),
            ContactDomain.POLYMER,
            is_hydrogen_atom=True,
            grid_cell=(1, 1, 0),
        ),
    )
    prepared_pair_index = PreparedAtomSitePairIndex(
        atom_sites=atom_sites,
        focus_residue_ids=focus_residue_ids,
    )

    expected_pairs = tuple(
        iter_candidate_atom_site_pairs(
            atom_sites,
            focus_residue_ids=focus_residue_ids,
            policy=policy,
        )
    )
    actual_pairs = tuple(prepared_pair_index.candidate_pairs(policy=policy))

    assert actual_pairs == expected_pairs


def test_clash_pair_generation_has_no_stringly_domain_coercion() -> None:
    """Hot pair filtering should not use getattr/string fallback domain coercion."""

    source = Path("src/protrepair/diagnostics/clash_pair_generation.py").read_text()

    assert "_domain_value" not in source
    assert "getattr(" not in source
    assert '== "ligand"' not in source


@pytest.mark.parametrize(
    "invalid_tolerance", (float("nan"), float("inf"), -float("inf"))
)
def test_clash_policy_rejects_nonfinite_heavy_overlap_tolerance(
    invalid_tolerance: float,
) -> None:
    """Heavy-atom overlap tolerance must define a finite distance threshold."""

    with pytest.raises(ValueError, match="heavy overlap tolerance must be finite"):
        ClashPolicy(heavy_overlap_tolerance_angstrom=invalid_tolerance)


@pytest.mark.parametrize(
    "invalid_tolerance", (float("nan"), float("inf"), -float("inf"))
)
def test_clash_policy_rejects_nonfinite_hydrogen_overlap_tolerance(
    invalid_tolerance: float,
) -> None:
    """Hydrogen overlap tolerance must define a finite distance threshold."""

    with pytest.raises(ValueError, match="hydrogen overlap tolerance must be finite"):
        ClashPolicy(hydrogen_overlap_tolerance_angstrom=invalid_tolerance)


def test_clash_policy_clamps_nonpositive_pair_distance_threshold() -> None:
    """An overlap tolerance at least as large as the radius sum admits no pair."""

    policy = ClashPolicy(heavy_overlap_tolerance_angstrom=4.0)

    assert (
        policy.allowed_distance_angstrom(
            left_van_der_waals_radius_angstrom=1.7,
            right_van_der_waals_radius_angstrom=1.7,
            left_is_hydrogen=False,
            right_is_hydrogen=False,
        )
        == 0.0
    )


def test_clash_policy_preserves_positional_threshold_contract() -> None:
    """The first positional value remains the heavy-overlap threshold."""

    policy = ClashPolicy(0.5)

    assert policy.heavy_overlap_tolerance_angstrom == 0.5
    assert policy.include_hydrogens is True


@pytest.mark.parametrize(
    ("left_element", "right_element", "policy"),
    (
        ("C", "C", ClashPolicy(heavy_overlap_tolerance_angstrom=10.0)),
        ("H", "C", ClashPolicy(hydrogen_overlap_tolerance_angstrom=10.0)),
    ),
)
def test_nonpositive_pair_threshold_cannot_create_zero_distance_clash(
    left_element: str,
    right_element: str,
    policy: ClashPolicy,
) -> None:
    """Squaring a nonpositive threshold must not revive an impossible clash."""

    structure = build_two_atom_contact_structure(
        left_element=left_element,
        right_element=right_element,
        distance_angstrom=0.0,
    )
    context = prepare_clash_detection_context(
        structure,
        component_library=build_standard_component_library(),
        policy=policy,
    )

    assert (
        context.allowed_distance_by_element_pair[(left_element, right_element)] == 0.0
    )
    assert (
        context.allowed_distance_by_element_pair[(right_element, left_element)] == 0.0
    )
    assert context.candidate_cell_size_angstrom >= 1.0
    assert context.detect_clashes().is_empty()


def test_positive_pair_threshold_still_reports_zero_distance_clash() -> None:
    """The nonpositive guard must not suppress a genuine zero-distance overlap."""

    report = detect_clashes(
        build_two_atom_contact_structure(
            left_element="C",
            right_element="C",
            distance_angstrom=0.0,
        ),
        component_library=build_standard_component_library(),
    )

    assert len(report.clashes) == 1
    assert report.clashes[0].distance_angstrom == 0.0
    assert report.clashes[0].allowed_distance_angstrom > 0.0


def test_pair_distance_threshold_boundary_is_exclusive() -> None:
    """A pair at the threshold is clear while one just inside still clashes."""

    policy = ClashPolicy(heavy_overlap_tolerance_angstrom=0.0)
    threshold = policy.allowed_distance_angstrom(
        left_van_der_waals_radius_angstrom=1.7,
        right_van_der_waals_radius_angstrom=1.7,
        left_is_hydrogen=False,
        right_is_hydrogen=False,
    )
    component_library = build_standard_component_library()

    at_threshold = detect_clashes(
        build_two_atom_contact_structure(
            left_element="C",
            right_element="C",
            distance_angstrom=threshold,
        ),
        component_library=component_library,
        policy=policy,
    )
    just_inside = detect_clashes(
        build_two_atom_contact_structure(
            left_element="C",
            right_element="C",
            distance_angstrom=threshold - 1e-6,
        ),
        component_library=component_library,
        policy=policy,
    )

    assert at_threshold.is_empty()
    assert len(just_inside.clashes) == 1


def test_very_large_finite_tolerance_produces_inert_threshold() -> None:
    """A large finite policy value should remain valid without numeric revival."""

    policy = ClashPolicy(heavy_overlap_tolerance_angstrom=1e308)

    assert (
        policy.allowed_distance_angstrom(
            left_van_der_waals_radius_angstrom=1.7,
            right_van_der_waals_radius_angstrom=1.7,
            left_is_hydrogen=False,
            right_is_hydrogen=False,
        )
        == 0.0
    )


def test_hydrogen_isotope_aliases_share_pair_distance_threshold() -> None:
    """H, D, and T should select the same hydrogen overlap tolerance."""

    component_library = build_standard_component_library()
    thresholds = []
    for hydrogen_element in ("H", "D", "T"):
        context = prepare_clash_detection_context(
            build_two_atom_contact_structure(
                left_element=hydrogen_element,
                right_element="C",
                distance_angstrom=2.0,
            ),
            component_library=component_library,
        )
        thresholds.append(
            context.allowed_distance_by_element_pair[(hydrogen_element, "C")]
        )

    assert thresholds == pytest.approx((2.0, 2.0, 2.0))


def test_empty_clash_basis_uses_minimum_candidate_cell_size() -> None:
    """A structure with no eligible atoms should produce an inert clash basis."""

    structure = build_structure(chains=(), source_format=FileFormat.PDB)

    basis = prepare_clash_detection_basis(
        structure,
        component_library=build_standard_component_library(),
    )

    assert basis.atom_site_bases == ()
    assert basis.candidate_cell_size_angstrom == 1.0
    assert (
        bind_clash_detection_context(structure, basis=basis).detect_clashes().is_empty()
    )


@pytest.mark.parametrize("cell_size", (float("nan"), float("inf"), 0.5))
def test_clash_basis_rejects_unsafe_cell_size_override(cell_size: float) -> None:
    """A shared grid must never undercut the clash basis search radius."""

    structure = build_two_atom_contact_structure(
        left_element="C",
        right_element="C",
        distance_angstrom=1.0,
    )
    basis = prepare_clash_detection_basis(
        structure,
        component_library=build_standard_component_library(),
    )

    with pytest.raises(ValueError, match="finite candidate cell size"):
        bind_clash_detection_frame(
            structure,
            basis=basis,
            candidate_cell_size_angstrom=cell_size,
        )


def test_prepared_clash_artifacts_cannot_be_reassembled_with_replace() -> None:
    """Cached facets must only be assembled by their preparation factories."""

    structure = build_two_atom_contact_structure(
        left_element="C",
        right_element="C",
        distance_angstrom=1.0,
    )
    basis = prepare_clash_detection_basis(
        structure,
        component_library=build_standard_component_library(),
    )
    context = basis.bind_context(structure)

    with pytest.raises(ValueError, match="created by a preparation factory"):
        replace(basis, candidate_cell_size_angstrom=0.5)
    with pytest.raises(ValueError, match="created by a preparation factory"):
        replace(
            context,
            candidate_cell_size_angstrom=0.5,
        )


def test_clash_basis_accepts_larger_shared_cell_size() -> None:
    """A broader shared grid should preserve exact clash results."""

    structure = build_two_atom_contact_structure(
        left_element="C",
        right_element="C",
        distance_angstrom=1.0,
    )
    basis = prepare_clash_detection_basis(
        structure,
        component_library=build_standard_component_library(),
    )
    shared_cell_size_angstrom = basis.candidate_cell_size_angstrom + 1.0

    context = bind_clash_detection_context(
        structure,
        basis=basis,
        candidate_cell_size_angstrom=shared_cell_size_angstrom,
    )

    assert context.candidate_cell_size_angstrom == shared_cell_size_angstrom
    assert all(
        atom_site.grid_cell_size_angstrom == shared_cell_size_angstrom
        for atom_site in context.atom_sites
    )
    assert context.detect_clashes() == basis.bind_context(structure).detect_clashes()


def test_empty_projected_clash_context_uses_minimum_candidate_cell_size() -> None:
    """The projected preparation path should share empty-envelope semantics."""

    structure = build_structure(chains=(), source_format=FileFormat.PDB)

    context = prepare_projected_clash_detection_context(
        structure,
        residue_ids=(),
        component_library=build_standard_component_library(),
    )

    assert context.atom_sites == ()
    assert context.candidate_cell_size_angstrom == 1.0
    assert context.detect_clashes().is_empty()


def test_clash_context_uses_explicit_common_metal_vdw_radii() -> None:
    """Common retained metals should not enter clash diagnostics as carbon fallback."""

    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="ION",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom("FE", "Fe", Vec3(0.0, 0.0, 0.0)),
                    atom("ZN", "Zn", Vec3(6.0, 0.0, 0.0)),
                    atom("MG", "Mg", Vec3(12.0, 0.0, 0.0)),
                    atom("CA1", "Ca", Vec3(18.0, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="common-metal-radius-cache",
    )

    context = prepare_clash_detection_context(
        structure,
        component_library=build_standard_component_library(),
        policy=ClashPolicy(include_ligands=True),
    )

    assert context.van_der_waals_radius("FE") == 2.05
    assert context.van_der_waals_radius("ZN") == 2.10
    assert context.van_der_waals_radius("MG") == 2.20
    assert context.van_der_waals_radius("CA") == 2.40


def test_clash_detection_cell_size_covers_maximum_radius_pair() -> None:
    """Spatial hashing should not miss broad-radius pairs in non-adjacent old cells."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue(
                        "CSX",
                        "A",
                        1,
                        (atom("CS1", "CS", Vec3(3.99, 0.0, 0.0)),),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    build_residue(
                        "CSX",
                        "B",
                        1,
                        (atom("CS1", "CS", Vec3(8.10, 0.0, 0.0)),),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="broad-radius-grid-boundary-clash",
    )
    component_library = build_standard_component_library()

    context = prepare_clash_detection_context(
        structure,
        component_library=component_library,
    )
    report = context.detect_clashes()
    focused_report = context.detect_clashes(
        focus_residue_ids=frozenset((ResidueId("A", 1),))
    )
    projected_context = prepare_projected_clash_detection_context(
        structure,
        residue_ids=(ResidueId("A", 1), ResidueId("B", 1)),
        component_library=component_library,
    )

    assert context.candidate_cell_size_angstrom > 4.0
    assert len(report.clashes) == 1
    assert focused_report == report
    assert projected_context.detect_clashes() == report
    assert report.clashes[0].allowed_distance_angstrom > 4.0
    assert report.clashes[0].distance_angstrom == pytest.approx(4.11)


def test_clash_context_reports_unresolved_element_radii_once() -> None:
    """Clash preparation should aggregate unknown radii before pair iteration."""

    structure = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="UNK",
                residue_id=ResidueId("L", 1),
                atoms=(
                    atom("XX1", "XX", Vec3(0.0, 0.0, 0.0)),
                    atom("C1", "C1", Vec3(1.0, 0.0, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="unknown-radius-clash-context",
    )

    with pytest.raises(UnknownElementRadiusError) as error_info:
        prepare_clash_detection_context(
            structure,
            component_library=build_standard_component_library(),
            policy=ClashPolicy(include_ligands=True),
        )

    error_message = str(error_info.value)
    assert "clash detection basis has unresolved van_der_waals radius" in (
        error_message
    )
    assert "C1" in error_message
    assert "XX" in error_message


def test_standard_component_templates_expose_heavy_bond_hops() -> None:
    """Standard residue templates should expose heavy-atom bond topology."""

    template = build_standard_component_library().require("SER")

    assert template.bond_hop_distance("OG", "CB") == 1
    assert template.bond_hop_distance("OG", "CA") == 2
    assert template.bond_hop_distance("OG", "N") == 3


def test_detect_clashes_reports_same_residue_hydrogen_vs_nonbonded_heavy_atom() -> None:
    """Same-residue hydrogen clashes should survive bond-hop filtering."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue(
                        "SER",
                        "A",
                        1,
                        (
                            atom("N", "N", Vec3(-10.0, 0.0, 0.0)),
                            atom("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom("C", "C", Vec3(11.0, 0.0, 0.0)),
                            atom("O", "O", Vec3(12.0, 0.0, 0.0)),
                            atom("CB", "C", Vec3(2.0, 0.0, 0.0)),
                            atom("OG", "O", Vec3(0.55, 0.0, 0.0)),
                            atom("HG", "H", Vec3(0.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="same-residue-hydrogen-clash",
    )

    report = detect_clashes(
        structure,
        component_library=build_standard_component_library(),
    )

    assert len(report.clashes) == 1
    clash = report.clashes[0]
    assert {clash.left_atom_name, clash.right_atom_name} == {"CA", "HG"}
    assert clash.left_residue_id == clash.right_residue_id
    assert report.to_issues()[0].kind is ValidationIssueKind.STERIC_CLASH
    assert report.to_issues()[0].scope.kind is EventScopeKind.RESIDUE


def test_prepared_clash_context_matches_public_clash_detection() -> None:
    """Prepared clash contexts should cache computation without changing semantics."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue(
                        "SER",
                        "A",
                        1,
                        (
                            atom("N", "N", Vec3(-10.0, 0.0, 0.0)),
                            atom("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom("C", "C", Vec3(11.0, 0.0, 0.0)),
                            atom("O", "O", Vec3(12.0, 0.0, 0.0)),
                            atom("CB", "C", Vec3(2.0, 0.0, 0.0)),
                            atom("OG", "O", Vec3(0.55, 0.0, 0.0)),
                            atom("HG", "H", Vec3(0.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="prepared-clash-context",
    )
    component_library = build_standard_component_library()
    residue_ids = frozenset((ResidueId("A", 1),))

    context = prepare_clash_detection_context(
        structure,
        component_library=component_library,
    )

    assert context.has_clashes()
    assert has_clashes_in_context(context)
    assert context.detect_clashes() == detect_clashes(
        structure,
        component_library=component_library,
    )
    assert detect_clashes_from_context(context) == detect_clashes(
        structure,
        component_library=component_library,
    )
    prepared_pair_index = PreparedAtomSitePairIndex(
        atom_sites=context.atom_sites,
        focus_residue_ids=None,
    )
    assert detect_clashes_from_context(
        context,
        prepared_pair_index=prepared_pair_index,
    ) == detect_clashes(
        structure,
        component_library=component_library,
    )
    assert context.detect_clashes(focus_residue_ids=residue_ids) == (
        detect_clashes_involving_residues(
            structure,
            residue_ids=residue_ids,
            component_library=component_library,
        )
    )
    assert detect_clashes_from_context(
        context,
        focus_residue_ids=residue_ids,
    ) == detect_clashes_involving_residues(
        structure,
        residue_ids=residue_ids,
        component_library=component_library,
    )


def test_clash_detection_basis_rebinds_coordinate_only_structures() -> None:
    """Reusable clash bases should preserve reports across coordinate frames."""

    residue_id = ResidueId("A", 1)
    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue(
                        "SER",
                        "A",
                        1,
                        (
                            atom("N", "N", Vec3(-10.0, 0.0, 0.0)),
                            atom("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom("C", "C", Vec3(11.0, 0.0, 0.0)),
                            atom("O", "O", Vec3(12.0, 0.0, 0.0)),
                            atom("CB", "C", Vec3(2.0, 0.0, 0.0)),
                            atom("OG", "O", Vec3(0.55, 0.0, 0.0)),
                            atom("HG", "H", Vec3(0.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="basis-rebind-clash-context",
    )
    component_library = build_standard_component_library()
    basis = prepare_clash_detection_basis(
        structure,
        component_library=component_library,
    )
    moved_residue_geometry = structure.residue_geometry(
        structure.constitution.residue_index(residue_id)
    ).with_atom_geometry(
        "HG",
        AtomGeometry(position=Vec3(20.0, 0.0, 0.0)),
    )
    moved_structure = structure.with_updated_residue_geometries(
        ((residue_id, moved_residue_geometry),)
    )

    assert detect_clashes_from_context(
        bind_clash_detection_context(structure, basis=basis)
    ) == detect_clashes(structure, component_library=component_library)
    assert basis.bind_context(structure).detect_clashes() == detect_clashes(
        structure,
        component_library=component_library,
    )
    assert detect_clashes_from_context(
        bind_clash_detection_context(moved_structure, basis=basis)
    ) == detect_clashes(moved_structure, component_library=component_library)
    assert basis.bind_context(moved_structure).detect_clashes() == detect_clashes(
        moved_structure,
        component_library=component_library,
    )

    original_frame = bind_clash_detection_frame(structure, basis=basis)
    moved_frame = bind_clash_detection_frame(moved_structure, basis=basis)
    assert original_frame == basis.bind_frame(structure)
    assert moved_frame == basis.bind_frame(moved_structure)
    assert len(original_frame.atom_sites) == len(moved_frame.atom_sites)
    original_hg_site = next(
        atom_site
        for atom_site in original_frame.atom_sites
        if atom_site.atom_name == "HG"
    )
    moved_hg_site = next(
        atom_site
        for atom_site in moved_frame.atom_sites
        if atom_site.atom_name == "HG"
    )
    assert original_hg_site.context.template is moved_hg_site.context.template
    assert original_hg_site.geometry != moved_hg_site.geometry


def test_clash_detection_basis_rejects_same_address_replacement_constitution() -> None:
    """Cached chemistry facts must remain tied to the exact constitution."""

    source_structure = build_two_atom_contact_structure(
        left_element="C",
        right_element="C",
        distance_angstrom=1.0,
    )
    replacement_structure = build_two_atom_contact_structure(
        left_element="N",
        right_element="O",
        distance_angstrom=1.0,
    )
    assert (
        source_structure.constitution.address_space_key
        == replacement_structure.constitution.address_space_key
    )
    basis = prepare_clash_detection_basis(
        source_structure,
        component_library=build_standard_component_library(),
    )

    assert not basis.is_compatible_with(replacement_structure)
    with pytest.raises(ValueError, match="immutable constitution"):
        basis.bind_context(replacement_structure)


def test_detect_clashes_ignores_same_residue_two_hop_hydrogen_contacts() -> None:
    """Hydrogen pairs within two bond hops should be ignored."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue(
                        "SER",
                        "A",
                        1,
                        (
                            atom("N", "N", Vec3(-10.0, 0.0, 0.0)),
                            atom("CA", "C", Vec3(-9.0, 0.0, 0.0)),
                            atom("C", "C", Vec3(-8.0, 0.0, 0.0)),
                            atom("O", "O", Vec3(-7.0, 0.0, 0.0)),
                            atom("CB", "C", Vec3(1.5, 0.0, 0.0)),
                            atom("OG", "O", Vec3(1.6, 0.0, 0.0)),
                            atom("HG", "H", Vec3(1.5, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="same-residue-two-hop-contact",
    )

    report = detect_clashes(
        structure,
        component_library=build_standard_component_library(),
    )

    assert report.is_empty()


def test_detect_clashes_ignores_same_residue_heavy_atom_pairs() -> None:
    """Same-residue heavy-heavy contacts should not become clash warnings."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue(
                        "SER",
                        "A",
                        1,
                        (
                            atom("N", "N", Vec3(-10.0, 0.0, 0.0)),
                            atom("CA", "C", Vec3(0.0, 0.0, 0.0)),
                            atom("C", "C", Vec3(10.0, 0.0, 0.0)),
                            atom("O", "O", Vec3(11.0, 0.0, 0.0)),
                            atom("CB", "C", Vec3(0.2, 0.0, 0.0)),
                            atom("OG", "O", Vec3(1.5, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="same-residue-heavy-contact",
    )

    report = detect_clashes(
        structure,
        component_library=build_standard_component_library(),
    )

    assert report.is_empty()


def test_detect_clashes_ignores_adjacent_peptide_bond_pairs() -> None:
    """Adjacent peptide C-N contacts should not be reported as steric clashes."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue(
                        "GLY",
                        "A",
                        1,
                        (
                            atom("N", "N", Vec3(-5.0, 0.0, 0.0)),
                            atom("CA", "C", Vec3(-4.0, 0.0, 0.0)),
                            atom("C", "C", Vec3(0.0, 0.0, 0.0)),
                            atom("O", "O", Vec3(1.2, 0.0, 0.0)),
                        ),
                    ),
                    build_residue(
                        "GLY",
                        "A",
                        2,
                        (
                            atom("N", "N", Vec3(1.33, 0.0, 0.0)),
                            atom("CA", "C", Vec3(6.0, 0.0, 0.0)),
                            atom("C", "C", Vec3(7.0, 0.0, 0.0)),
                            atom("O", "O", Vec3(8.2, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="adjacent-peptide-bond",
    )

    report = detect_clashes(
        structure,
        component_library=build_standard_component_library(),
    )

    assert report.is_empty()


def test_detect_clashes_reports_slot_neighbor_without_peptide_geometry() -> None:
    """Chain-slot neighbors are not ignored when C-N geometry is implausible."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue(
                        "GLY",
                        "A",
                        1,
                        (
                            atom("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom("C", "C", Vec3(10.0, 0.0, 0.0)),
                        ),
                    ),
                    build_residue(
                        "GLY",
                        "A",
                        2,
                        (
                            atom("N", "N", Vec3(20.0, 0.0, 0.0)),
                            atom("CA", "C", Vec3(10.2, 0.0, 0.0)),
                            atom("C", "C", Vec3(21.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="implausible-peptide-slot-neighbor-clash",
    )

    report = detect_clashes(
        structure,
        component_library=build_standard_component_library(),
    )

    assert any(
        clash.left_residue_id != clash.right_residue_id
        and {clash.left_atom_name, clash.right_atom_name} == {"C", "CA"}
        for clash in report.clashes
    )


def test_detect_clashes_projects_inter_residue_pairs_as_pair_scope() -> None:
    """Inter-residue clashes should carry residue-pair provenance."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue(
                        "ALA",
                        "A",
                        1,
                        (
                            atom("N", "N", Vec3(-5.0, 0.0, 0.0)),
                            atom("CA", "C", Vec3(-4.0, 0.0, 0.0)),
                            atom("C", "C", Vec3(-3.0, 0.0, 0.0)),
                            atom("O", "O", Vec3(-2.0, 0.0, 0.0)),
                            atom("CB", "C", Vec3(0.0, 0.0, 0.0)),
                        ),
                    ),
                    build_residue(
                        "ALA",
                        "A",
                        3,
                        (
                            atom("N", "N", Vec3(4.0, 0.0, 0.0)),
                            atom("CA", "C", Vec3(5.0, 0.0, 0.0)),
                            atom("C", "C", Vec3(6.0, 0.0, 0.0)),
                            atom("O", "O", Vec3(7.0, 0.0, 0.0)),
                            atom("CB", "C", Vec3(0.3, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="inter-residue-heavy-clash",
    )

    report = detect_clashes(
        structure,
        component_library=build_standard_component_library(),
    )

    assert report.clashes
    issue = report.to_issues()[0]
    assert issue.kind is ValidationIssueKind.STERIC_CLASH
    assert issue.scope.kind is EventScopeKind.RESIDUE_PAIR
    assert issue.residue_id is None
    assert issue.scope.targets_residue(ResidueId("A", 1))
    assert issue.scope.targets_residue(ResidueId("A", 3))


def test_detect_clashes_involving_residues_ignores_irrelevant_pairs() -> None:
    """Focused clash detection should drop clashes outside the requested residues."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue(
                        "ALA",
                        "A",
                        1,
                        (
                            atom("N", "N", Vec3(-5.0, 0.0, 0.0)),
                            atom("CA", "C", Vec3(-4.0, 0.0, 0.0)),
                            atom("C", "C", Vec3(-3.0, 0.0, 0.0)),
                            atom("O", "O", Vec3(-2.0, 0.0, 0.0)),
                            atom("CB", "C", Vec3(0.0, 0.0, 0.0)),
                        ),
                    ),
                    build_residue(
                        "ALA",
                        "A",
                        3,
                        (
                            atom("N", "N", Vec3(4.0, 0.0, 0.0)),
                            atom("CA", "C", Vec3(5.0, 0.0, 0.0)),
                            atom("C", "C", Vec3(6.0, 0.0, 0.0)),
                            atom("O", "O", Vec3(7.0, 0.0, 0.0)),
                            atom("CB", "C", Vec3(0.3, 0.0, 0.0)),
                        ),
                    ),
                    build_residue(
                        "ALA",
                        "A",
                        5,
                        (
                            atom("N", "N", Vec3(15.0, 0.0, 0.0)),
                            atom("CA", "C", Vec3(16.0, 0.0, 0.0)),
                            atom("C", "C", Vec3(17.0, 0.0, 0.0)),
                            atom("O", "O", Vec3(18.0, 0.0, 0.0)),
                            atom("CB", "C", Vec3(20.0, 0.0, 0.0)),
                        ),
                    ),
                    build_residue(
                        "ALA",
                        "A",
                        7,
                        (
                            atom("N", "N", Vec3(24.0, 0.0, 0.0)),
                            atom("CA", "C", Vec3(25.0, 0.0, 0.0)),
                            atom("C", "C", Vec3(26.0, 0.0, 0.0)),
                            atom("O", "O", Vec3(27.0, 0.0, 0.0)),
                            atom("CB", "C", Vec3(20.3, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="focused-inter-residue-heavy-clash",
    )

    full_report = detect_clashes(
        structure,
        component_library=build_standard_component_library(),
    )
    focused_report = detect_clashes_involving_residues(
        structure,
        residue_ids=frozenset({ResidueId("A", 1), ResidueId("A", 3)}),
        component_library=build_standard_component_library(),
    )
    focus_residue_ids = frozenset({ResidueId("A", 1), ResidueId("A", 3)})
    filtered_full_clashes = tuple(
        clash
        for clash in full_report.clashes
        if (
            clash.left_residue_id in focus_residue_ids
            or clash.right_residue_id in focus_residue_ids
        )
    )

    full_residue_pairs = {
        (clash.left_residue_id, clash.right_residue_id) for clash in full_report.clashes
    }
    focused_residue_pairs = {
        (clash.left_residue_id, clash.right_residue_id)
        for clash in focused_report.clashes
    }

    assert full_residue_pairs == {
        (ResidueId("A", 1), ResidueId("A", 3)),
        (ResidueId("A", 5), ResidueId("A", 7)),
    }
    assert focused_residue_pairs == {
        (ResidueId("A", 1), ResidueId("A", 3)),
    }
    assert focused_report.clashes == filtered_full_clashes


def test_detect_clashes_ignores_probable_hydrogen_bonds() -> None:
    """Inter-residue donor-H...acceptor pairs should not become clash warnings."""

    structure = build_hydrogen_bond_candidate_structure(
        donor_position=Vec3(3.0, 0.0, 0.0),
        hydrogen_position=Vec3(1.8, 0.0, 0.0),
        acceptor_position=Vec3(0.0, 0.0, 0.0),
        source_name="linear-hydrogen-bond-candidate",
    )

    report = detect_clashes(
        structure,
        component_library=build_standard_component_library(),
    )

    assert report.is_empty()


def test_detect_clashes_reports_acute_angle_hydrogen_bond_candidates() -> None:
    """Acute donor-H...acceptor contacts should remain steric clashes."""

    structure = build_hydrogen_bond_candidate_structure(
        donor_position=Vec3(1.8, 1.2, 0.0),
        hydrogen_position=Vec3(1.8, 0.0, 0.0),
        acceptor_position=Vec3(0.0, 0.0, 0.0),
        source_name="acute-angle-hydrogen-bond-candidate",
    )

    report = detect_clashes(
        structure,
        component_library=build_standard_component_library(),
        policy=ClashPolicy(heavy_overlap_tolerance_angstrom=2.0),
    )

    assert len(report.clashes) == 1
    assert {report.clashes[0].left_atom_name, report.clashes[0].right_atom_name} == {
        "H",
        "O",
    }


def test_detect_clashes_reports_no_anchor_hydrogen_bond_candidates() -> None:
    """Contacts without a donor anchor should not be suppressed as H-bonds."""

    structure = build_hydrogen_bond_candidate_structure(
        donor_position=Vec3(5.0, 0.0, 0.0),
        hydrogen_position=Vec3(1.8, 0.0, 0.0),
        acceptor_position=Vec3(0.0, 0.0, 0.0),
        source_name="no-anchor-hydrogen-bond-candidate",
    )

    report = detect_clashes(
        structure,
        component_library=build_standard_component_library(),
        policy=ClashPolicy(heavy_overlap_tolerance_angstrom=2.0),
    )

    assert len(report.clashes) == 1
    assert {report.clashes[0].left_atom_name, report.clashes[0].right_atom_name} == {
        "H",
        "O",
    }


def test_detect_clashes_reports_degenerate_angle_hydrogen_bond_candidates() -> None:
    """A zero-length donor-H vector should not suppress or crash clash checks."""

    structure = build_hydrogen_bond_candidate_structure(
        donor_position=Vec3(1.8, 0.0, 0.0),
        hydrogen_position=Vec3(1.8, 0.0, 0.0),
        acceptor_position=Vec3(0.0, 0.0, 0.0),
        source_name="degenerate-angle-hydrogen-bond-candidate",
    )

    report = detect_clashes(
        structure,
        component_library=build_standard_component_library(),
        policy=ClashPolicy(heavy_overlap_tolerance_angstrom=2.0),
    )

    assert len(report.clashes) == 1
    assert {report.clashes[0].left_atom_name, report.clashes[0].right_atom_name} == {
        "H",
        "O",
    }


def test_detect_clashes_reports_near_degenerate_hbond_angle_candidates() -> None:
    """Near-zero donor-H vectors should follow the same undefined-angle path."""

    structure = build_hydrogen_bond_candidate_structure(
        donor_position=Vec3(1.8 + 1e-14, 0.0, 0.0),
        hydrogen_position=Vec3(1.8, 0.0, 0.0),
        acceptor_position=Vec3(0.0, 0.0, 0.0),
        source_name="nearly-degenerate-angle-hydrogen-bond-candidate",
    )

    report = detect_clashes(
        structure,
        component_library=build_standard_component_library(),
        policy=ClashPolicy(heavy_overlap_tolerance_angstrom=2.0),
    )

    assert len(report.clashes) == 1
    assert {report.clashes[0].left_atom_name, report.clashes[0].right_atom_name} == {
        "H",
        "O",
    }


def test_detect_clashes_reports_non_acceptor_hydrogen_bond_candidates() -> None:
    """Valid distance and angle should not suppress a non-acceptor contact."""

    structure = build_hydrogen_bond_candidate_structure(
        donor_position=Vec3(3.0, 0.0, 0.0),
        hydrogen_position=Vec3(1.8, 0.0, 0.0),
        acceptor_position=Vec3(0.0, 0.0, 0.0),
        acceptor_atom_name="C1",
        acceptor_element="C",
        source_name="non-acceptor-hydrogen-bond-candidate",
    )

    report = detect_clashes(
        structure,
        component_library=build_standard_component_library(),
        policy=ClashPolicy(heavy_overlap_tolerance_angstrom=2.0),
    )

    assert len(report.clashes) == 1
    assert {report.clashes[0].left_atom_name, report.clashes[0].right_atom_name} == {
        "C1",
        "H",
    }


def test_detect_clashes_reports_too_short_hydrogen_bond_candidates() -> None:
    """Too-short H...acceptor contacts should stay clashes, even when linear."""

    structure = build_hydrogen_bond_candidate_structure(
        donor_position=Vec3(2.7, 0.0, 0.0),
        hydrogen_position=Vec3(1.5, 0.0, 0.0),
        acceptor_position=Vec3(0.0, 0.0, 0.0),
        source_name="too-short-hydrogen-bond-candidate",
    )

    report = detect_clashes(
        structure,
        component_library=build_standard_component_library(),
        policy=ClashPolicy(heavy_overlap_tolerance_angstrom=2.0),
    )

    assert len(report.clashes) == 1
    assert {report.clashes[0].left_atom_name, report.clashes[0].right_atom_name} == {
        "H",
        "O",
    }


def test_repair_heavy_atoms_supports_explicit_clash_reporting() -> None:
    """Heavy repair results should feed explicit clash diagnostics."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue(
                        "ALA",
                        "A",
                        1,
                        (
                            atom("N", "N", Vec3(-2.0, 0.0, 0.0)),
                            atom("CA", "C", Vec3(-1.0, 1.0, 0.0)),
                            atom("C", "C", Vec3(0.0, 0.0, 0.0)),
                            atom("O", "O", Vec3(0.7, -1.0, 0.0)),
                            atom("CB", "C", Vec3(0.0, 2.0, 0.0)),
                        ),
                    ),
                    build_residue(
                        "ALA",
                        "A",
                        2,
                        (
                            atom("N", "N", Vec3(1.3, 0.2, 0.0)),
                            atom("CA", "C", Vec3(2.3, 1.1, 0.0)),
                            atom("C", "C", Vec3(3.5, 0.2, 0.0)),
                            atom("O", "O", Vec3(4.1, -0.8, 0.0)),
                            atom("CB", "C", Vec3(0.2, 2.1, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="heavy-clash-warning",
    )

    result = repair_heavy_atoms(structure)
    report = detect_clashes(
        result.structure,
        component_library=build_standard_component_library(),
    )

    assert any(
        issue.kind is ValidationIssueKind.STERIC_CLASH for issue in report.to_issues()
    )


def build_hydrogen_bond_candidate_structure(
    *,
    donor_position: Vec3,
    hydrogen_position: Vec3,
    acceptor_position: Vec3,
    source_name: str,
    acceptor_atom_name: str = "O",
    acceptor_element: str = "O",
) -> ProteinStructure:
    """Build a two-chain minimal donor-H...acceptor clash fixture."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue(
                        "GLY",
                        "A",
                        1,
                        (
                            atom(
                                acceptor_atom_name,
                                acceptor_element,
                                acceptor_position,
                            ),
                        ),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    build_residue(
                        "GLY",
                        "B",
                        1,
                        (
                            atom("N", "N", donor_position),
                            atom("H", "H", hydrogen_position),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name=source_name,
    )


def build_two_atom_contact_structure(
    *,
    left_element: str,
    right_element: str,
    distance_angstrom: float,
) -> ProteinStructure:
    """Build one cross-chain two-atom contact at a controlled separation."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    build_residue(
                        "UNX",
                        "A",
                        1,
                        (atom("L1", left_element, Vec3(0.0, 0.0, 0.0)),),
                    ),
                ),
            ),
            chain_payload(
                "B",
                (
                    build_residue(
                        "UNY",
                        "B",
                        1,
                        (
                            atom(
                                "R1",
                                right_element,
                                Vec3(distance_angstrom, 0.0, 0.0),
                            ),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="controlled-two-atom-contact",
    )


def build_residue(
    component_id: str,
    chain_id: str,
    seq_num: int,
    atoms: tuple[CanonicalAtomPayload, ...],
) -> CanonicalResiduePayload:
    """Build one canonical residue for clash-diagnostic tests."""

    return residue_payload(
        component_id=component_id,
        residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
        atoms=atoms,
    )


def atom(name: str, element: str, position: Vec3) -> CanonicalAtomPayload:
    """Build one canonical atom payload for clash-diagnostic tests."""

    return atom_payload(name, element, position)

"""Steric-clash diagnostics over canonical repaired structures."""

from dataclasses import dataclass
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

from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.diagnostics import (
    ClashPolicy,
    EventScopeKind,
    ValidationIssueKind,
    bind_clash_detection_context,
    bind_clash_detection_frame,
    detect_clashes,
    detect_clashes_from_context,
    detect_clashes_involving_residues,
    has_clashes_in_context,
    prepare_clash_detection_basis,
    prepare_clash_detection_context,
)
from protrepair.diagnostics.clash_pair_generation import (
    ContactDomain,
    pair_can_be_rejected_before_distance,
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


def test_clash_pair_generation_has_no_stringly_domain_coercion() -> None:
    """Hot pair filtering should not use getattr/string fallback domain coercion."""

    source = Path("src/protrepair/diagnostics/clash_pair_generation.py").read_text()

    assert "_domain_value" not in source
    assert "getattr(" not in source
    assert '== "ligand"' not in source


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

    assert context.van_der_waals_radius("FE") == 2.00
    assert context.van_der_waals_radius("ZN") == 1.39
    assert context.van_der_waals_radius("MG") == 1.73
    assert context.van_der_waals_radius("CA") == 2.31


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

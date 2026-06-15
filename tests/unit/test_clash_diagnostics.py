"""Steric-clash diagnostics over canonical repaired structures."""

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
from protrepair.geometry import Vec3
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.heavy import repair_heavy_atoms


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
                            atom("C", "C", Vec3(-3.0, 0.0, 0.0)),
                            atom("O", "O", Vec3(0.0, 0.0, 0.0)),
                        ),
                    ),
                    build_residue(
                        "GLY",
                        "A",
                        2,
                        (
                            atom("N", "N", Vec3(3.0, 0.0, 0.0)),
                            atom("H", "H", Vec3(1.8, 0.0, 0.0)),
                            atom("CA", "C", Vec3(4.0, 0.0, 0.0)),
                            atom("C", "C", Vec3(5.0, 0.0, 0.0)),
                            atom("O", "O", Vec3(6.2, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="hydrogen-bond-candidate",
    )

    report = detect_clashes(
        structure,
        component_library=build_standard_component_library(),
    )

    assert report.is_empty()


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

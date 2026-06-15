"""Residual rotatable-hydrogen cleanup tests."""

from math import sqrt
from pathlib import Path

from tests.support.canonical_builders import (
    CanonicalAtomPayload,
    CanonicalResiduePayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry import ComponentLibrary
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.diagnostics import (
    ClashPolicy,
    ValidationIssueKind,
    detect_clashes,
)
from protrepair.geometry import Vec3
from protrepair.io import read_structure
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import (
    AtomRef,
    ResidueId,
)
from protrepair.transformer.completion.hydrogen import (
    add_hydrogens,
    cleanup_residual_hydrogen_clashes,
)
from protrepair.transformer.completion.hydrogen.rotatable import (
    build_rotatable_hydrogen_search,
    rotatable_hydrogen_placement_spec,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.workflow.contracts import StructureIngressOptions

SERINE_RESIDUE_ID = ResidueId(chain_id="A", seq_num=17)


def test_cleanup_residual_hydrogen_clashes_reorients_ligand_clashing_serine_hydrogen(
) -> None:
    """Cleanup should reorient one rotatable hydrogen away from a ligand clash."""

    library = build_standard_component_library()
    protein_only = build_serine_window_structure()
    hydrogenated = add_hydrogens(
        protein_only,
        component_library=library,
    ).structure
    original_residue = residue_payload_from_structure(
        hydrogenated,
        SERINE_RESIDUE_ID,
    )
    original_hydrogen = original_residue.atom_geometry("HG").position
    ligand = build_ligand_residue(
        "L",
        1,
        (resolvable_ligand_atom(original_residue, component_library=library),),
    )
    hydrogenated_holo = hydrogenated.with_ligand_facets(
        ligand_sites=(ligand[0],),
        ligand_geometries=(ligand[1],),
        ligand_formal_charge_payloads=(ligand[2],),
    )

    initial_report = detect_clashes(
        hydrogenated_holo,
        component_library=library,
        policy=ClashPolicy(include_ligands=True),
    )
    assert any(
        clash.left_atom_name == "HG" or clash.right_atom_name == "HG"
        for clash in initial_report.clashes
    )

    cleaned_structure, issues = cleanup_residual_hydrogen_clashes(
        hydrogenated_holo,
        component_library=library,
    )
    cleaned_hydrogen = cleaned_structure.geometry.atom_geometry(
        cleaned_structure.constitution.atom_index(
            residue_atom_ref(SERINE_RESIDUE_ID, "HG")
        )
    )
    final_report = detect_clashes(
        cleaned_structure,
        component_library=library,
        policy=ClashPolicy(include_ligands=True),
    )

    assert cleaned_hydrogen is not None
    assert cleaned_hydrogen.position != original_hydrogen
    assert not any(
        clash.left_atom_name == "HG" or clash.right_atom_name == "HG"
        for clash in final_report.clashes
    )
    assert issues == ()


def test_cleanup_residual_hydrogen_clashes_reports_unresolved_targets() -> None:
    """Cleanup should keep warning-level issues when every orientation still clashes."""

    library = build_standard_component_library()
    protein_only = build_serine_window_structure()
    hydrogenated = add_hydrogens(
        protein_only,
        component_library=library,
    ).structure
    residue = residue_payload_from_structure(hydrogenated, SERINE_RESIDUE_ID)
    template = library.require("SER")
    spec = rotatable_hydrogen_placement_spec(template.hydrogen_semantics)
    assert spec is not None

    search = build_rotatable_hydrogen_search(residue, spec=spec)
    assert search is not None

    ligand_atoms = tuple(
        atom_payload(
            f"C{index}",
            "C",
            Vec3.from_iterable(candidate).with_offset(0.05, 0.0, 0.0),
        )
        for index, candidate in enumerate(search.candidate_positions(), start=1)
    )
    unresolved_ligand = build_ligand_residue("L", 1, ligand_atoms)
    hydrogenated_holo = hydrogenated.with_ligand_facets(
        ligand_sites=(unresolved_ligand[0],),
        ligand_geometries=(unresolved_ligand[1],),
        ligand_formal_charge_payloads=(unresolved_ligand[2],),
    )

    cleaned_structure, issues = cleanup_residual_hydrogen_clashes(
        hydrogenated_holo,
        component_library=library,
    )
    final_report = detect_clashes(
        cleaned_structure,
        component_library=library,
        policy=ClashPolicy(include_ligands=True),
    )

    assert any(
        clash.left_atom_name == "HG" or clash.right_atom_name == "HG"
        for clash in final_report.clashes
    )
    assert any(issue.kind is ValidationIssueKind.STERIC_CLASH for issue in issues)


def test_add_hydrogens_runs_residual_cleanup_for_ligand_clashes() -> None:
    """End-to-end hydrogenation should apply the cleanup pass automatically."""

    library = build_standard_component_library()
    protein_only = build_serine_window_structure()
    baseline = add_hydrogens(
        protein_only,
        component_library=library,
    ).structure
    cleanup_ligand = build_ligand_residue(
        "L",
        1,
        (
            resolvable_ligand_atom(
                residue_payload_from_structure(
                    baseline,
                    SERINE_RESIDUE_ID,
                ),
                component_library=library,
            ),
        ),
    )
    holo = protein_only.with_ligand_facets(
        ligand_sites=(cleanup_ligand[0],),
        ligand_geometries=(cleanup_ligand[1],),
        ligand_formal_charge_payloads=(cleanup_ligand[2],),
    )

    result = add_hydrogens(holo, component_library=library)
    final_report = detect_clashes(
        result.structure,
        component_library=library,
        policy=ClashPolicy(include_ligands=True),
    )

    assert not any(
        clash.left_atom_name == "HG" or clash.right_atom_name == "HG"
        for clash in final_report.clashes
    )
    assert not result.has_warnings()


def build_serine_window_structure() -> ProteinStructure:
    """Return a real heavy-only chain window centered on one serine residue."""

    structure = read_structure(
        Path("tests/fixtures/corpus/pdb1afc.ent"),
        policy=StructureIngressOptions().structure_normalization_policy(),
    )
    chain_site = structure.chain_site("A")
    center_index = next(
        index
        for index, residue_site in enumerate(chain_site.residues)
        if residue_site.residue_id == SERINE_RESIDUE_ID
    )
    window_residue_sites = chain_site.residues[
        max(0, center_index - 1) : center_index + 2
    ]
    window_payloads = tuple(
        canonical_residue_payload_from_structure(
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


def resolvable_ligand_atom(
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


def residue_payload_from_structure(
    structure: ProteinStructure,
    residue_id: ResidueId,
) -> CompletionResiduePayload:
    """Return one completion payload projected from canonical structure facets."""

    residue_geometry = structure.geometry.residue_geometry(
        constitution=structure.constitution,
        residue_index=structure.constitution.residue_index(residue_id),
    )
    residue_site = structure.constitution.residue_or_ligand(residue_id)
    assert residue_site is not None
    return CompletionResiduePayload(
        residue_site=residue_site,
        residue_geometry=residue_geometry,
        formal_charge_by_atom_name=structure.topology.residue_formal_charge_by_atom_name(
            constitution=structure.constitution,
            residue_index=structure.constitution.residue_index(residue_id),
        ),
    )


def canonical_residue_payload_from_structure(
    structure: ProteinStructure,
    residue_id: ResidueId,
) -> CanonicalResiduePayload:
    """Return one canonical residue payload projected from structure facets."""

    residue_geometry = structure.geometry.residue_geometry(
        constitution=structure.constitution,
        residue_index=structure.constitution.residue_index(residue_id),
    )
    residue_site = structure.constitution.residue_or_ligand(residue_id)
    assert residue_site is not None
    return (
        residue_site,
        residue_geometry,
        structure.topology.residue_formal_charge_by_atom_name(
            constitution=structure.constitution,
            residue_index=structure.constitution.residue_index(residue_id),
        ),
    )


def residue_atom_ref(residue_id: ResidueId, atom_name: str) -> AtomRef:
    """Return one canonical atom reference for test lookups."""

    return AtomRef(residue_id=residue_id, atom_name=atom_name)

"""Residual rotatable-hydrogen cleanup tests."""

from math import sqrt
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
    build_bonded_structure as build_structure,
)

import protrepair.transformer.completion.hydrogen.repair as hydrogen_repair
from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.chemistry.retained_non_polymer.registry import (
    build_bundled_retained_non_polymer_registry,
)
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
from protrepair.transformer.continuous.binding_policy import (
    ManualContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationConfig,
    ContinuousRelaxationForceField,
)
from protrepair.transformer.refinement.directive import RepairLocalRefinementDirective
from protrepair.transformer.result import TransformationResult
from protrepair.workflow.contracts import StructureIngressOptions

SERINE_RESIDUE_ID = ResidueId(chain_id="A", seq_num=17)


@pytest.mark.parametrize("fixed", (False, True))
def test_cleanup_reorients_only_movable_ligand_clashing_serine_hydrogen(
    fixed: bool,
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
        fixed_hydrogen_atom_refs=(
            frozenset((AtomRef(SERINE_RESIDUE_ID, "HG"),)) if fixed else frozenset()
        ),
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
    if fixed:
        assert cleaned_structure is hydrogenated_holo
        assert cleaned_hydrogen.position == original_hydrogen
        assert final_report == initial_report
        return

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


@pytest.mark.parametrize("symbol", ("H", "D", "T"))
@pytest.mark.parametrize("stage", ("none", "prerequisites", "moved"))
@pytest.mark.parametrize("targeted", (False, True))
def test_outer_hydrogen_cleanup_preserves_input_ligand_h_after_other_stages(
    symbol, stage, targeted, monkeypatch
):
    library = build_default_component_library()
    source = retained_serine_clash_structure(symbol)
    ref = AtomRef(ResidueId("L", 1), "HG")
    original = source.geometry.atom_geometry(source.constitution.atom_index(ref))
    unrestricted, _ = cleanup_residual_hydrogen_clashes(
        source, component_library=library
    )
    assert (
        unrestricted.geometry.atom_geometry(unrestricted.constitution.atom_index(ref))
        != original
    )
    expected = original
    if stage == "moved":
        apply_stage = hydrogen_repair.apply_repair_stage_local_refinement
        expected = original.with_position(
            original.position.with_offset(0.02, 0.01, 0.0)
        )

        def moved_stage(result, **kwargs):
            staged = apply_stage(result, **kwargs)
            current = staged.structure
            # A prior geometry-changing stage is authoritative for its own pose.
            moved = ProteinStructure.from_payload(
                constitution=current.constitution,
                geometry=current.geometry.with_atom_geometry(
                    current.constitution.atom_index(ref), expected
                ),
                topology=current.topology,
                provenance=current.provenance,
                polymer_blueprint=current.polymer_blueprint,
            )
            return TransformationResult(
                structure=moved, repairs=staged.repairs, issues=staged.issues
            )

        monkeypatch.setattr(
            hydrogen_repair, "apply_repair_stage_local_refinement", moved_stage
        )

    directive = (
        None
        if stage == "none"
        else RepairLocalRefinementDirective.from_residue_ids(
            (ref.residue_id,),
            binding=ManualContinuousRelaxationBinding(
                ContinuousRelaxationForceField.UFF
            ),
            config=ContinuousRelaxationConfig(max_iterations=1),
        )
    )
    result = add_hydrogens(
        source,
        component_library=library,
        prepare_heavy_atoms=False,
        target_residue_ids=frozenset((ref.residue_id,)) if targeted else None,
        local_refinement=directive,
    )
    if stage != "none":
        residue = result.structure.constitution.residue_or_ligand(ref.residue_id)
        assert residue is not None and residue.has_atom_site("HA")
        assert any(
            issue.kind is ValidationIssueKind.REFINEMENT_REJECTED
            for issue in result.issues
        )
    index = result.structure.constitution.atom_index(ref)
    assert result.structure.geometry.atom_geometry(index) == expected
    assert result.structure.constitution.atom_site_at(index).element == symbol


def test_outer_cleanup_can_still_move_h_generated_by_retained_prerequisites(
    monkeypatch,
):
    reference = retained_serine_clash_structure("H")
    source = retained_serine_clash_structure("H", include_source_hydrogen=False)
    ref = AtomRef(ResidueId("L", 1), "HG")
    clashing = reference.geometry.atom_geometry(reference.constitution.atom_index(ref))
    apply_stage = hydrogen_repair.apply_repair_stage_local_refinement

    def placed_into_clash(result, **kwargs):
        staged = apply_stage(result, **kwargs)
        current = staged.structure
        index = current.constitution.atom_index(ref)
        moved = ProteinStructure.from_payload(
            constitution=current.constitution,
            geometry=current.geometry.with_atom_geometry(
                index,
                current.geometry.atom_geometry(index).with_position(clashing.position),
            ),
            topology=current.topology,
            provenance=current.provenance,
            polymer_blueprint=current.polymer_blueprint,
        )
        return TransformationResult(
            structure=moved, repairs=staged.repairs, issues=staged.issues
        )

    monkeypatch.setattr(
        hydrogen_repair, "apply_repair_stage_local_refinement", placed_into_clash
    )
    result = add_hydrogens(
        source,
        prepare_heavy_atoms=False,
        local_refinement=RepairLocalRefinementDirective.from_residue_ids(
            (ref.residue_id,),
            binding=ManualContinuousRelaxationBinding(
                ContinuousRelaxationForceField.UFF
            ),
            config=ContinuousRelaxationConfig(max_iterations=1),
        ),
    )
    new = result.structure.geometry.atom_geometry(
        result.structure.constitution.atom_index(ref)
    )
    assert new.position != clashing.position


def retained_serine_clash_structure(
    symbol: str, *, include_source_hydrogen: bool = True
) -> ProteinStructure:
    library = build_default_component_library()
    prepared = add_hydrogens(build_serine_window_structure()).structure
    residue = residue_payload_from_structure(prepared, SERINE_RESIDUE_ID)
    point = resolvable_ligand_atom(residue, component_library=library)[1].position
    record = build_bundled_retained_non_polymer_registry().get("SO4")
    assert record is not None
    sulfate = record.to_idealized_component()
    positions = {
        a.atom_name: Vec3.coerce(a.ideal_position)
        for a in sulfate.heavy_atoms()
        if a.ideal_position is not None
    }
    assert len(positions) == len(sulfate.heavy_atoms())
    shift = point.to_array() - positions["O1"].to_array()
    return build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="SER",
                residue_id=ResidueId("L", 1),
                is_hetero=True,
                atoms=tuple(
                    atom_payload(
                        a.name,
                        symbol if a.name == "HG" else a.element,
                        residue.position(a.name),
                        formal_charge=residue.formal_charge(a.name),
                        occupancy=0.4 if a.name == "HG" else 1.0,
                        b_factor=17.0 if a.name == "HG" else 0.0,
                    )
                    for a in residue.atom_sites
                    if not a.is_hydrogen()
                    or (include_source_hydrogen and a.name == "HG")
                ),
            ),
            residue_payload(
                component_id="SO4",
                residue_id=ResidueId("Z", 1),
                is_hetero=True,
                atoms=tuple(
                    atom_payload(
                        a.atom_name,
                        a.element,
                        Vec3.from_iterable(positions[a.atom_name].to_array() + shift),
                        formal_charge=library.require(
                            "SO4"
                        ).definition.formal_charges.get(a.atom_name, 0),
                    )
                    for a in sulfate.heavy_atoms()
                ),
            ),
        ),
        source_format=prepared.provenance.ingress.source_format,
    )


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

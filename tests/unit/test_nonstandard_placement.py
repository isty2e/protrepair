"""Nonstandard placement tests over bundled ideal-geometry assets."""

from typing import get_args

import numpy as np
import pytest
from tests.support.canonical_builders import (
    atom_payload,
    chain_payload,
)
from tests.support.canonical_builders import (
    build_structure as build_canonical_structure,
)
from tests.support.canonical_builders import (
    completion_payload as build_completion_payload,
)

from protrepair.chemistry import IdealGeometryHeavyAtomSemantics, ResidueTemplate
from protrepair.chemistry.nonstandard.registry import (
    NonstandardComponentRecord,
    build_bundled_nonstandard_registry,
)
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.diagnostics import (
    ValidationIssueKind,
)
from protrepair.geometry import Vec3
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import (
    ResidueId,
)
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import (
    ResidueIndex,
)
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.completion.atom.rigid_frame import (
    RigidFramePlacementTransformer,
)
from protrepair.transformer.completion.heavy import repair_heavy_atoms
from protrepair.transformer.completion.heavy.reconstruction import (
    build_component_reconstruction_plan,
)
from protrepair.transformer.completion.shared import (
    LocalFramePlacementDirective,
    MseBridgePlacementDirective,
    OrderedAtomPatch,
    ResidueCompletionSite,
    ResidueFramePlacementDirective,
    RigidComponentPlacementDirective,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.binding_policy import (
    ManualContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationConfig,
    ContinuousRelaxationForceField,
)
from protrepair.transformer.refinement.directive import RepairLocalRefinementDirective
from protrepair.workflow.contracts import OrphanFragmentPolicy

ROTATION = np.asarray(
    (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)
TRANSLATION = np.asarray((4.5, -2.0, 1.75), dtype=np.float64)


def test_rigid_component_placement_directive_is_closed_variant_union() -> None:
    """Rigid placement directives should expose a closed concrete variant set."""

    assert get_args(RigidComponentPlacementDirective) == (
        LocalFramePlacementDirective,
        ResidueFramePlacementDirective,
        MseBridgePlacementDirective,
    )


def materialize_patch_on_residue(
    patch: OrderedAtomPatch,
    residue: CompletionResiduePayload,
) -> CompletionResiduePayload:
    """Return one patched residue payload projected from a canonical residue."""

    residue_site, residue_geometry, formal_charge_by_atom_name = (
        patch.materialize_on_payload(
            residue.residue_site,
            residue_geometry=residue.residue_geometry,
            formal_charge_by_atom_name=residue.formal_charge_by_atom_name,
        )
    )
    return CompletionResiduePayload(
        residue_site=residue_site,
        residue_geometry=residue_geometry,
        formal_charge_by_atom_name=formal_charge_by_atom_name,
    )


def structure_from_payloads(
    residues: tuple[CompletionResiduePayload, ...],
    *,
    ligands: tuple[CompletionResiduePayload, ...] = (),
    source_name: str | None = None,
) -> ProteinStructure:
    """Build one canonical structure from completion payloads."""

    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                tuple(
                    (
                        residue.residue_site,
                        residue.residue_geometry,
                        residue.formal_charge_by_atom_name,
                    )
                    for residue in residues
                ),
            ),
        ),
        ligands=tuple(
            (
                ligand.residue_site,
                ligand.residue_geometry,
                ligand.formal_charge_by_atom_name,
            )
            for ligand in ligands
        ),
        source_format=FileFormat.PDB,
        source_name=source_name,
    )


def test_place_component_atoms_places_supported_hyp_leaf_from_local_frame() -> None:
    """Bundled ideal geometry should place HYP OD1 from the retained local frame."""

    record = require_record("HYP")
    residue = build_hyp_residue(missing_atom_names=("OD1",))
    plan = build_component_reconstruction_plan(
        residue,
        record.to_idealized_component(),
        template=canonical_template_for(record),
        parent_template=build_standard_component_library().require("PRO"),
    )

    repaired_patch = rigid_component_patch(
        original_residue=residue,
        residue=residue,
        semantics=heavy_atom_semantics_for(record),
        plan=plan,
        target_atom_names=plan.reconstruction_atom_names,
    )

    assert repaired_patch is not None
    repaired_residue = materialize_patch_on_residue(repaired_patch, residue)
    expected_position = transformed_ideal_position(record, "OD1")
    placed_position = repaired_residue.position("OD1")
    assert placed_position.x == pytest.approx(expected_position.x, abs=1e-6)
    assert placed_position.y == pytest.approx(expected_position.y, abs=1e-6)
    assert placed_position.z == pytest.approx(expected_position.z, abs=1e-6)


def test_place_component_atoms_rejects_unsupported_hyp_scope() -> None:
    """HYP placements beyond the accepted first-wave scope should return ``None``."""

    record = require_record("HYP")
    residue = build_hyp_residue(missing_atom_names=("CD", "OD1"))
    plan = build_component_reconstruction_plan(
        residue,
        record.to_idealized_component(),
        template=canonical_template_for(record),
        parent_template=build_standard_component_library().require("PRO"),
    )

    repaired_patch = rigid_component_patch(
        original_residue=residue,
        residue=residue,
        semantics=heavy_atom_semantics_for(record),
        plan=plan,
        target_atom_names=plan.reconstruction_atom_names,
    )

    assert repaired_patch is None


def test_repair_heavy_atoms_repairs_supported_nonstandard_hyp() -> None:
    """Heavy repair should enable the first bundled nonstandard allowlist residue."""

    residue = build_hyp_residue(missing_atom_names=("OD1",))
    structure = structure_from_payloads(
        (residue,),
        source_name="hyp-supported-heavy-repair",
    )

    result = repair_heavy_atoms(structure)
    repaired_residue = result.structure.chain_site("A").residues[0]

    assert repaired_residue.component_id == "HYP"
    assert repaired_residue.has_atom_site("OD1")
    assert not any(
        issue.kind is ValidationIssueKind.UNSUPPORTED_TEMPLATE_REPAIR
        for issue in result.issues
    )
    assert any(event.atom_names == ("OD1",) for event in result.repairs)


def test_repair_heavy_atoms_leaves_unsupported_hyp_gap_unchanged() -> None:
    """Unsupported HYP gap shapes should stay on the structured diagnostic path."""

    residue = build_hyp_residue(missing_atom_names=("CD", "OD1"))
    structure = structure_from_payloads(
        (residue,),
        source_name="hyp-unsupported-heavy-repair",
    )

    result = repair_heavy_atoms(structure)
    repaired_residue = result.structure.chain_site("A").residues[0]

    assert repaired_residue.atom_site_names() == residue.atom_names()
    assert any(
        issue.kind is ValidationIssueKind.UNSUPPORTED_TEMPLATE_REPAIR
        and issue.residue_id == residue.residue_id
        for issue in result.issues
    )


@pytest.mark.parametrize(
    ("component_id", "missing_atom_names"),
    (
        ("CSO", ("OD",)),
        ("SEP", ("P", "O1P", "O2P", "O3P")),
        ("TPO", ("P", "O1P", "O2P", "O3P")),
        ("PTR", ("P", "O1P", "O2P", "O3P")),
    ),
)
def test_place_component_atoms_places_supported_with_refinement_seed(
    component_id: str,
    missing_atom_names: tuple[str, ...],
) -> None:
    """Retained-fragment seeding should place refinement-required subtrees."""

    record = require_record(component_id)
    residue = build_nonstandard_residue(
        component_id=component_id,
        missing_atom_names=missing_atom_names,
    )
    plan = build_component_reconstruction_plan(
        residue,
        record.to_idealized_component(),
        template=canonical_template_for(record),
        parent_template=parent_template_for(record),
    )

    repaired_patch = rigid_component_patch(
        original_residue=residue,
        residue=residue,
        semantics=heavy_atom_semantics_for(record),
        plan=plan,
        target_atom_names=plan.reconstruction_atom_names,
    )

    assert repaired_patch is not None
    repaired_residue = materialize_patch_on_residue(repaired_patch, residue)
    for atom_name in missing_atom_names:
        expected_position = transformed_ideal_position(record, atom_name)
        placed_position = repaired_residue.position(atom_name)
        assert placed_position.x == pytest.approx(expected_position.x, abs=1e-6)
        assert placed_position.y == pytest.approx(expected_position.y, abs=1e-6)
        assert placed_position.z == pytest.approx(expected_position.z, abs=1e-6)


@pytest.mark.parametrize(
    ("component_id", "missing_atom_names"),
    (
        ("CSO", ("OD",)),
        ("SEP", ("P", "O1P", "O2P", "O3P")),
        ("TPO", ("P", "O1P", "O2P", "O3P")),
        ("PTR", ("P", "O1P", "O2P", "O3P")),
    ),
)
def test_repair_heavy_atoms_requires_explicit_local_refinement_for_supported_bucket(
    component_id: str,
    missing_atom_names: tuple[str, ...],
) -> None:
    """Supported-with-refinement bundled residues should stay unchanged without one."""

    residue = build_nonstandard_residue(
        component_id=component_id,
        missing_atom_names=missing_atom_names,
    )
    structure = structure_from_payloads(
        (residue,),
        source_name=f"{component_id.lower()}-missing-heavy-repair",
    )

    result = repair_heavy_atoms(structure)
    repaired_residue = result.structure.chain_site("A").residues[0]

    assert (
        repaired_residue.without_atom_sites(("OXT",)).atom_site_names()
        == residue.atom_names()
    )
    assert any(
        issue.kind is ValidationIssueKind.UNSUPPORTED_TEMPLATE_REPAIR
        and issue.residue_id == residue.residue_id
        and "explicit local refinement" in issue.message
        for issue in result.issues
    )


@pytest.mark.parametrize(
    ("component_id", "missing_atom_names"),
    (
        ("CSO", ("OD",)),
        ("SEP", ("P", "O1P", "O2P", "O3P")),
        ("TPO", ("P", "O1P", "O2P", "O3P")),
        ("PTR", ("P", "O1P", "O2P", "O3P")),
    ),
)
def test_repair_heavy_atoms_repairs_supported_with_refinement_nonstandard_when_selected(
    monkeypatch: pytest.MonkeyPatch,
    component_id: str,
    missing_atom_names: tuple[str, ...],
) -> None:
    """Explicit local refinement should unlock supported-with-refinement residues."""

    residue = build_nonstandard_residue(
        component_id=component_id,
        missing_atom_names=missing_atom_names,
    )
    structure = structure_from_payloads(
        (residue,),
        source_name=f"{component_id.lower()}-refinement-heavy-repair",
    )
    local_refinement = RepairLocalRefinementDirective.from_residue_ids(
        (residue.residue_id,),
        config=ContinuousRelaxationConfig(),
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.UFF),
    )
    execute_call_count = 0

    def fake_refine_local_region(*args, **kwargs):
        nonlocal execute_call_count
        execute_call_count += 1
        raise AssertionError("heavy-only repair should not bind any force field")

    monkeypatch.setattr(
        "protrepair.transformer.refinement.repair_stage.execute_local_transformation",
        fake_refine_local_region,
    )

    result = repair_heavy_atoms(structure, local_refinement=local_refinement)
    repaired_residue = result.structure.chain_site("A").residues[0]

    for atom_name in missing_atom_names:
        assert repaired_residue.has_atom_site(atom_name)

    assert not any(
        issue.kind is ValidationIssueKind.UNSUPPORTED_TEMPLATE_REPAIR
        for issue in result.issues
    )
    assert execute_call_count == 0
    assert any(
        issue.kind is ValidationIssueKind.REFINEMENT_REJECTED for issue in result.issues
    )


def test_repair_heavy_atoms_requires_targeted_refinement_for_supported_bucket(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit refinement must target the residue before seed-only repair."""

    target_residue = build_nonstandard_residue(
        component_id="SEP",
        missing_atom_names=("P", "O1P", "O2P", "O3P"),
    )
    context_residue = build_ala_residue(seq_num=2)
    structure = structure_from_payloads(
        (target_residue, context_residue),
        source_name="sep-untargeted-refinement-heavy-repair",
    )
    local_refinement = RepairLocalRefinementDirective.from_residue_ids(
        (context_residue.residue_id,),
        config=ContinuousRelaxationConfig(),
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.UFF),
    )

    monkeypatch.setattr(
        "protrepair.transformer.refinement.repair_stage.execute_local_transformation",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("heavy-only repair should not bind any force field")
        ),
    )

    result = repair_heavy_atoms(structure, local_refinement=local_refinement)
    repaired_residue = result.structure.chain_site("A").residues[0]

    assert (
        repaired_residue.without_atom_sites(("OXT",)).atom_site_names()
        == target_residue.atom_names()
    )
    assert any(
        issue.kind is ValidationIssueKind.UNSUPPORTED_TEMPLATE_REPAIR
        and issue.residue_id == target_residue.residue_id
        for issue in result.issues
    )


def test_place_mse_bridge_atoms_can_bridge_from_salvaged_ce_anchor() -> None:
    """MSE bridge repair should place SE from CB/CG/CE local geometry."""

    record = require_record("MSE")
    original_residue = build_mse_residue(missing_atom_names=("SE",))
    rebuilt_residue = original_residue.without_atom_sites(("CE",))

    repaired_patch = rigid_component_patch(
        original_residue=original_residue,
        residue=rebuilt_residue,
        semantics=heavy_atom_semantics_for(record),
        plan=build_component_reconstruction_plan(
            rebuilt_residue,
            record.to_idealized_component(),
            template=canonical_template_for(record),
            parent_template=parent_template_for(record),
            orphan_fragment_policy=OrphanFragmentPolicy.SALVAGE_WHEN_SAFE,
        ),
        target_atom_names=("SE", "CE"),
        orphan_fragment_policy=OrphanFragmentPolicy.SALVAGE_WHEN_SAFE,
    )

    assert repaired_patch is not None
    repaired_residue = materialize_patch_on_residue(repaired_patch, rebuilt_residue)
    assert repaired_residue.position("CE") == original_residue.position("CE")
    expected_position = transformed_ideal_position(record, "SE")
    placed_position = repaired_residue.position("SE")
    assert placed_position.x == pytest.approx(expected_position.x, abs=1e-6)
    assert placed_position.y == pytest.approx(expected_position.y, abs=1e-6)
    assert placed_position.z == pytest.approx(expected_position.z, abs=1e-6)


def test_repair_heavy_atoms_can_fill_mse_from_reference() -> None:
    """Reference-guided MSE repair should restore missing SE and CE atoms."""

    residue = build_mse_residue(missing_atom_names=("SE", "CE"))
    reference_residue = build_mse_residue(missing_atom_names=())
    structure = structure_from_payloads(
        (residue,),
        source_name="mse-reference-heavy-repair",
    )
    reference_structure = structure_from_payloads(
        (reference_residue,),
        source_name="mse-reference-heavy-repair-ref",
    )

    result = repair_heavy_atoms(
        structure,
        reference_structure=reference_structure,
    )
    repaired_residue = result.structure.chain_site("A").residues[0]
    repaired_geometry = result.structure.geometry.residue_geometry(
        constitution=result.structure.constitution,
        residue_index=result.structure.constitution.residue_index(
            repaired_residue.residue_id
        ),
    )

    assert repaired_residue.has_atom_site("SE")
    assert repaired_residue.has_atom_site("CE")
    assert repaired_geometry.position("SE") == reference_residue.position("SE")
    assert repaired_geometry.position("CE") == reference_residue.position("CE")
    assert not any(
        issue.kind is ValidationIssueKind.UNSUPPORTED_TEMPLATE_REPAIR
        for issue in result.issues
    )


def test_repair_heavy_atoms_can_salvage_mse_bridge_without_reference() -> None:
    """Safe-orphan policy should allow MSE bridge repair from original CE geometry."""

    residue = build_mse_residue(missing_atom_names=("SE",))
    structure = structure_from_payloads(
        (residue,),
        source_name="mse-salvage-heavy-repair",
    )

    result = repair_heavy_atoms(
        structure,
        orphan_fragment_policy=OrphanFragmentPolicy.SALVAGE_WHEN_SAFE,
    )
    repaired_residue = result.structure.chain_site("A").residues[0]
    repaired_geometry = result.structure.geometry.residue_geometry(
        constitution=result.structure.constitution,
        residue_index=result.structure.constitution.residue_index(
            repaired_residue.residue_id
        ),
    )

    assert repaired_residue.has_atom_site("SE")
    assert repaired_residue.has_atom_site("CE")
    assert repaired_geometry.position("CE") == residue.position("CE")
    expected_position = transformed_ideal_position(require_record("MSE"), "SE")
    placed_position = repaired_geometry.position("SE")
    assert placed_position.x == pytest.approx(expected_position.x, abs=1e-6)
    assert placed_position.y == pytest.approx(expected_position.y, abs=1e-6)
    assert placed_position.z == pytest.approx(expected_position.z, abs=1e-6)
    assert not any(
        issue.kind is ValidationIssueKind.UNSUPPORTED_TEMPLATE_REPAIR
        for issue in result.issues
    )


def test_repair_heavy_atoms_leaves_mse_unsupported_without_reference_or_salvage() -> (
    None
):
    """Default rebuild policy should still reject MSE bridge repair without anchors."""

    residue = build_mse_residue(missing_atom_names=("SE",))
    structure = structure_from_payloads(
        (residue,),
        source_name="mse-unsupported-heavy-repair",
    )

    result = repair_heavy_atoms(structure)
    repaired_residue = result.structure.chain_site("A").residues[0]

    assert not repaired_residue.has_atom_site("SE")
    assert not repaired_residue.has_atom_site("CE")
    assert any(
        issue.kind is ValidationIssueKind.UNSUPPORTED_TEMPLATE_REPAIR
        and issue.residue_id == residue.residue_id
        for issue in result.issues
    )


def require_record(component_id: str) -> NonstandardComponentRecord:
    """Return one bundled nonstandard record required by the test fixture."""

    record = build_bundled_nonstandard_registry().get(component_id)
    assert record is not None
    return record


def canonical_template_for(record: NonstandardComponentRecord) -> ResidueTemplate:
    """Return the canonical template projected from one bundled record."""

    template = record.to_template()
    assert isinstance(template.heavy_atom_semantics, IdealGeometryHeavyAtomSemantics)
    return template


def heavy_atom_semantics_for(
    record: NonstandardComponentRecord,
) -> IdealGeometryHeavyAtomSemantics:
    """Return the canonical ideal-geometry heavy-atom semantics for one record."""

    semantics = canonical_template_for(record).heavy_atom_semantics
    assert isinstance(semantics, IdealGeometryHeavyAtomSemantics)
    return semantics


def rigid_component_patch(
    *,
    original_residue: CompletionResiduePayload,
    residue: CompletionResiduePayload,
    semantics: IdealGeometryHeavyAtomSemantics,
    plan,
    target_atom_names: tuple[str, ...],
    orphan_fragment_policy: OrphanFragmentPolicy = OrphanFragmentPolicy.REBUILD,
):
    """Return the canonical rigid-frame patch chosen by one reconstruction plan."""

    directive = plan.rigid_component_placement_directive(
        original_residue=original_residue,
        residue=residue,
        semantics=semantics,
        target_atom_names=target_atom_names,
        orphan_fragment_policy=orphan_fragment_policy,
    )
    if directive is None:
        return None

    site = ResidueCompletionSite(
        residue_index=ResidueIndex(0),
        template=plan.template,
        original_payload=original_residue,
    )
    snapshot = ProteinStructureSnapshot.from_structure(
        structure_from_payloads(
            (residue,),
            source_name="rigid-component-patch-test",
        )
    )
    transformer = RigidFramePlacementTransformer(site, directive)
    atom_input = site.atom_input(snapshot)
    context = ProteinTransformationContext.from_snapshot_atom_input(
        snapshot,
        atom_input,
    )
    if not transformer.is_applicable(context):
        return None

    repaired_snapshot = transformer.transform(context)
    repaired_residue = site.payload(repaired_snapshot)
    assert repaired_residue is not None
    return OrderedAtomPatch.from_residue_payload(
        repaired_residue.residue_site,
        residue_geometry=repaired_residue.residue_geometry,
    )


def build_hyp_residue(
    *,
    missing_atom_names: tuple[str, ...],
) -> CompletionResiduePayload:
    """Build one transformed HYP residue with a chosen heavy-atom gap."""

    record = require_record("HYP")
    missing_atom_name_set = frozenset(missing_atom_names)
    atoms = tuple(
        atom_payload(
            atom.atom_name,
            atom.element,
            transformed_ideal_position(record, atom.atom_name),
        )
        for atom in record.heavy_atoms()
        if atom.atom_name not in missing_atom_name_set
    )
    return build_completion_payload(
        component_id="HYP",
        residue_id=ResidueId(chain_id="A", seq_num=1),
        atoms=atoms,
    )


def build_mse_residue(
    *,
    missing_atom_names: tuple[str, ...],
) -> CompletionResiduePayload:
    """Build one transformed MSE residue with a chosen heavy-atom gap."""

    record = require_record("MSE")
    missing_atom_name_set = frozenset(missing_atom_names)
    atoms = tuple(
        atom_payload(
            atom.atom_name,
            atom.element,
            transformed_ideal_position(record, atom.atom_name),
        )
        for atom in record.heavy_atoms()
        if atom.atom_name not in missing_atom_name_set and atom.atom_name != "OXT"
    )
    return build_completion_payload(
        component_id="MSE",
        residue_id=ResidueId(chain_id="A", seq_num=1),
        atoms=atoms,
    )


def build_nonstandard_residue(
    *,
    component_id: str,
    missing_atom_names: tuple[str, ...],
) -> CompletionResiduePayload:
    """Build one transformed bundled residue with a chosen heavy-atom gap."""

    record = require_record(component_id)
    missing_atom_name_set = frozenset(missing_atom_names)
    atoms = tuple(
        atom_payload(
            atom.atom_name,
            atom.element,
            transformed_ideal_position(record, atom.atom_name),
        )
        for atom in record.heavy_atoms()
        if atom.atom_name not in missing_atom_name_set and atom.atom_name != "OXT"
    )
    return build_completion_payload(
        component_id=component_id,
        residue_id=ResidueId(chain_id="A", seq_num=1),
        atoms=atoms,
    )


def parent_template_for(
    record: NonstandardComponentRecord,
) -> ResidueTemplate | None:
    """Return the optional parent template for one bundled record."""

    if record.parent_standard_id is None:
        return None

    return build_standard_component_library().get(record.parent_standard_id)


def build_ala_residue(*, seq_num: int) -> CompletionResiduePayload:
    """Build one complete alanine residue for untargeted refinement tests."""

    return build_completion_payload(
        component_id="ALA",
        residue_id=ResidueId(chain_id="A", seq_num=seq_num),
        atoms=(
            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
            atom_payload("CA", "C", Vec3(1.45, 0.0, 0.0)),
            atom_payload("C", "C", Vec3(2.40, 1.20, 0.0)),
            atom_payload("O", "O", Vec3(2.10, 2.35, 0.0)),
            atom_payload("CB", "C", Vec3(1.80, -0.75, 1.25)),
        ),
    )


def transformed_ideal_position(
    record: NonstandardComponentRecord,
    atom_name: str,
) -> Vec3:
    """Return one bundled ideal coordinate projected into the fixture frame."""

    ideal_position = atom_ideal_position(record, atom_name)
    transformed_point = np.asarray(ideal_position, dtype=np.float64) @ ROTATION
    transformed_point = transformed_point + TRANSLATION
    return Vec3.from_iterable(transformed_point)


def atom_ideal_position(
    record: NonstandardComponentRecord,
    atom_name: str,
) -> tuple[float, float, float]:
    """Return one required ideal position from a bundled nonstandard record."""

    normalized_atom_name = atom_name.strip().upper()
    for atom in record.atoms:
        if atom.atom_name != normalized_atom_name:
            continue

        assert atom.ideal_position is not None
        return atom.ideal_position

    raise AssertionError(f"missing bundled atom {normalized_atom_name}")

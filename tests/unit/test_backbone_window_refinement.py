"""Tests for backbone-window refinement execution contracts."""

import pytest
from tests.support.canonical_builders import (
    CanonicalResiduePayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.refinement_benchmarks import resolve_fixture_path
from tests.support.refinement_cases import EXPLORATORY_REFINEMENT_FIXTURE_SOURCES
from tests.support.request_builders import ingress_options

from protrepair.diagnostics import IssueSeverity, ValidationIssue, ValidationIssueKind
from protrepair.geometry import Vec3
from protrepair.io import read_structure
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.artifacts import RegionTransformationResult
from protrepair.transformer.artifacts.patch import StructureDelta
from protrepair.transformer.atom_input import AtomInputRealization
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationProfile,
    ContinuousRelaxationSettings,
)
from protrepair.transformer.refinement.backbone_window import (
    atom_input_from_backbone_window_refinement_spec,
    execute_backbone_window_refinement,
)
from protrepair.transformer.refinement.spec import BackboneWindowRefinementSpec
from protrepair.workflow.contracts import LigandPolicy


def test_backbone_window_spec_lowers_to_residuewise_backbone_atom_input() -> None:
    """Backbone-window lowering should preserve residue-window semantics."""

    structure = _two_residue_backbone_structure()
    window_spec = BackboneWindowRefinementSpec(
        residue_ids=(
            ResidueId("A", 10),
            ResidueId("A", 11),
        )
    )

    atom_input = atom_input_from_backbone_window_refinement_spec(
        ProteinStructureSnapshot.from_structure(structure),
        window_spec,
    )

    assert atom_input.is_residuewise()
    assert atom_input.realizes_residue_backbones()
    assert atom_input.realization is AtomInputRealization.RESIDUE_BACKBONE_ATOMS
    assert atom_input.referenced_residue_ids() == window_spec.residue_ids
    assert tuple(
        structure.constitution.atom_ref_at(atom_index).atom_name
        for atom_index in atom_input.atom_indices
    ) == ("N", "CA", "C", "O", "N", "CA", "C", "O")


def test_backbone_window_executor_uses_backbone_atom_input(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Executor should route a backbone-window atom input to local relaxation."""

    structure = _two_residue_backbone_structure()
    settings = ContinuousRelaxationSettings(
        profile=ContinuousRelaxationProfile.RDKIT_UFF,
        max_iterations=5,
    )
    captured_contexts: list[ProteinTransformationContext] = []

    def fake_execute_local_transformation(
        context: ProteinTransformationContext,
        *,
        spec: ContinuousRelaxationSettings,
        component_library=None,
        restraint_library=None,
    ) -> RegionTransformationResult:
        del component_library
        del restraint_library
        assert spec is settings
        captured_contexts.append(context)
        return RegionTransformationResult(
            refined_structure=context.source_snapshot.structure,
            delta=StructureDelta(
                before_constitution=context.source_snapshot.structure.constitution,
                after_constitution=context.source_snapshot.structure.constitution,
            ),
            issues=(),
            backend_name="fake",
        )

    monkeypatch.setattr(
        "protrepair.transformer.refinement.backbone_window.execute_local_transformation",
        fake_execute_local_transformation,
    )

    result = execute_backbone_window_refinement(
        structure,
        BackboneWindowRefinementSpec(
            residue_ids=(
                ResidueId("A", 10),
                ResidueId("A", 11),
            )
        ),
        spec=settings,
    )

    assert result.backend_name == "fake"
    assert len(captured_contexts) == 1
    assert captured_contexts[0].atom_input.realizes_residue_backbones()


def test_backbone_window_executor_contextualizes_operator_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Backbone-window no-op rejection should name the insufficient operator."""

    structure = _two_residue_backbone_structure()
    settings = ContinuousRelaxationSettings(
        profile=ContinuousRelaxationProfile.RDKIT_UFF,
        max_iterations=5,
    )

    def fake_execute_local_transformation(
        context: ProteinTransformationContext,
        *,
        spec: ContinuousRelaxationSettings,
        component_library=None,
        restraint_library=None,
    ) -> RegionTransformationResult:
        del spec
        del component_library
        del restraint_library
        return RegionTransformationResult(
            refined_structure=context.source_snapshot.structure,
            delta=StructureDelta(
                before_constitution=context.source_snapshot.structure.constitution,
                after_constitution=context.source_snapshot.structure.constitution,
            ),
            issues=(
                ValidationIssue(
                    kind=ValidationIssueKind.REFINEMENT_REJECTED,
                    severity=IssueSeverity.INFO,
                    message="selected-region quality regressed",
                ),
            ),
            backend_name="fake",
        )

    monkeypatch.setattr(
        "protrepair.transformer.refinement.backbone_window.execute_local_transformation",
        fake_execute_local_transformation,
    )

    result = execute_backbone_window_refinement(
        structure,
        BackboneWindowRefinementSpec(
            residue_ids=(
                ResidueId("A", 10),
                ResidueId("A", 11),
            )
        ),
        spec=settings,
    )

    assert result.moved_atom_count() == 0
    assert result.issues[0].kind is ValidationIssueKind.REFINEMENT_REJECTED
    assert result.issues[0].message.startswith(
        "backbone-window refinement was not sufficient: "
    )


@pytest.mark.parametrize(
    "case_id",
    (
        "3j6b-terminal-helix-misthread",
        "3j9e-loop-backbone-error",
        "7s9d-prestin-segment",
    ),
)
def test_exploratory_backbone_window_fixtures_lower_to_operator_input(
    case_id: str,
) -> None:
    """Exploratory backbone-window fixtures should enter the new operator path."""

    source = EXPLORATORY_REFINEMENT_FIXTURE_SOURCES[case_id]
    structure = read_structure(
        resolve_fixture_path(source.output_path),
        policy=ingress_options(
            ligand_policy=LigandPolicy.KEEP
        ).structure_normalization_policy(),
    )

    atom_input = atom_input_from_backbone_window_refinement_spec(
        ProteinStructureSnapshot.from_structure(structure),
        BackboneWindowRefinementSpec(residue_ids=source.seed_residue_ids),
    )

    assert atom_input.realizes_residue_backbones()
    assert atom_input.referenced_residue_ids() == source.seed_residue_ids


def test_backbone_window_lowering_rejects_non_polymer_or_incomplete_backbone() -> None:
    """Backbone-window execution should reject non-window chemistry early."""

    structure = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ALA",
                        residue_id=ResidueId("A", 10),
                        atoms=(
                            atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                        ),
                    ),
                    _backbone_residue_payload("ALA", ResidueId("A", 11), x_offset=3.0),
                ),
            ),
        ),
        ligands=(
            _backbone_residue_payload(
                "ZN",
                ResidueId("A", 201),
                x_offset=6.0,
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )

    with pytest.raises(ValueError, match="present requested backbone atoms"):
        atom_input_from_backbone_window_refinement_spec(
            ProteinStructureSnapshot.from_structure(structure),
            BackboneWindowRefinementSpec(
                residue_ids=(ResidueId("A", 10), ResidueId("A", 11))
            ),
        )

    with pytest.raises(ValueError, match="polymer residues"):
        atom_input_from_backbone_window_refinement_spec(
            ProteinStructureSnapshot.from_structure(structure),
            BackboneWindowRefinementSpec(
                residue_ids=(ResidueId("A", 11), ResidueId("A", 201))
            ),
        )


def _two_residue_backbone_structure() -> ProteinStructure:
    """Return a compact two-residue polymer structure with backbone atoms."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    _backbone_residue_payload("ALA", ResidueId("A", 10), x_offset=0.0),
                    _backbone_residue_payload("GLY", ResidueId("A", 11), x_offset=3.0),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )


def _backbone_residue_payload(
    component_id: str,
    residue_id: ResidueId,
    *,
    x_offset: float,
    is_hetero: bool = False,
) -> CanonicalResiduePayload:
    """Return a residue payload with canonical peptide backbone atoms."""

    return residue_payload(
        component_id=component_id,
        residue_id=residue_id,
        is_hetero=is_hetero,
        atoms=(
            atom_payload("N", "N", Vec3(x_offset, 0.0, 0.0)),
            atom_payload("CA", "C", Vec3(x_offset + 1.0, 0.0, 0.0)),
            atom_payload("C", "C", Vec3(x_offset + 2.0, 0.0, 0.0)),
            atom_payload("O", "O", Vec3(x_offset + 2.5, 0.0, 0.0)),
        ),
    )

"""Unit tests for canonical scoped-state contracts and adapters."""

import importlib

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.geometry import Vec3
from protrepair.scope import (
    AbsentResidueSpanScope,
    AnchorAtomPairScope,
    AtomSetScope,
    AtomToChainSetScopeCoarsening,
    AtomToResidueSetScopeCoarsening,
    ChainSetScope,
    CompositeScope,
    ResidueSetScope,
    ResidueToChainSetScopeCoarsening,
    ScopeKind,
    ScopeToWholeStructureCoarsening,
    WholeStructureScope,
    scope_refines,
    scopes_overlap,
)
from protrepair.scope.observed_atom_scope_lowering import OBSERVED_ATOM_SCOPE_LOWERING
from protrepair.state import (
    CarrierScopedState,
    HydrogenCoverageState,
    ScopedState,
)
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.snapshot import (
    ProteinStructureSnapshot,
)
from protrepair.transformer.atom_input import AtomInput, AtomInputBasis
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.local import LocalScopeSpec
from protrepair.workflow.planning.completion import (
    WorkflowAnchorAtomPairExecutionScope,
    WorkflowCompositeExecutionScope,
    WorkflowResidueSetExecutionScope,
)


def test_scope_package_root_remains_semantic_only_surface() -> None:
    """The scope package root should not re-export structure-aware lowering owners."""

    scope_module = importlib.import_module("protrepair.scope")

    assert not hasattr(scope_module, "OBSERVED_ATOM_SCOPE_LOWERING")


def _single_residue_structure() -> ProteinStructure:
    """Return one canonical single-residue structure for scope tests."""

    return build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        component_id="GLY",
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
        source_name="state-scope-context",
    )


def test_canonical_state_scopes_normalize_chain_and_composite_inputs() -> None:
    """Canonical state scopes should deduplicate and validate their members."""

    chain_scope = ChainSetScope(chain_ids=(" A ", "B", "A"))
    residue_scope = ResidueSetScope(
        residue_ids=(
            ResidueId(chain_id="A", seq_num=1),
            ResidueId(chain_id="A", seq_num=1),
        )
    )
    atom_scope = AtomSetScope(
        atom_refs=(
            AtomRef(ResidueId(chain_id="A", seq_num=1), " CA "),
            AtomRef(ResidueId(chain_id="A", seq_num=1), "CA"),
        ),
    )
    composite_scope = CompositeScope(
        scopes=(WholeStructureScope(), residue_scope, atom_scope, residue_scope)
    )

    assert chain_scope.kind is ScopeKind.CHAIN_SET
    assert chain_scope.chain_ids == ("A", "B")
    assert residue_scope.kind is ScopeKind.RESIDUE_SET
    assert atom_scope.kind is ScopeKind.ATOM_SET
    assert composite_scope.kind is ScopeKind.COMPOSITE
    assert composite_scope.scopes == (
        WholeStructureScope(),
        residue_scope,
        atom_scope,
    )

    with pytest.raises(ValueError, match="blank chain ids"):
        ChainSetScope(chain_ids=(" ",))

    with pytest.raises(
        ValueError,
        match="at least two distinct scopes",
    ):
        CompositeScope(scopes=(WholeStructureScope(),))


def test_local_scope_specs_and_semantic_scopes_project_into_canonical_state_scopes(
) -> None:
    """Boundary local scope specs should align with canonical semantic scopes."""

    residue_selection = LocalScopeSpec.from_residues(
        (ResidueId(chain_id="A", seq_num=10),)
    )
    atom_selection = LocalScopeSpec.from_atoms(
        (AtomRef(ResidueId(chain_id="A", seq_num=10), "CA"),)
    )
    attached_hydrogen_selection = (
        LocalScopeSpec.from_atoms_with_attached_hydrogens(
            (AtomRef(ResidueId(chain_id="A", seq_num=10), "CA"),)
        )
    )
    absent_scope = AbsentResidueSpanScope(
        preceding_residue_id=ResidueId(chain_id="A", seq_num=10),
        following_residue_id=ResidueId(chain_id="A", seq_num=13),
        absent_residue_ids=(
            ResidueId(chain_id="A", seq_num=11),
            ResidueId(chain_id="A", seq_num=12),
        ),
    )
    anchor_scope = AnchorAtomPairScope(
        left_anchor_atom_ref=AtomRef(ResidueId(chain_id="A", seq_num=10), "C"),
        right_anchor_atom_ref=AtomRef(ResidueId(chain_id="A", seq_num=13), "N"),
    )
    composite_scope = CompositeScope(
        scopes=(residue_selection.as_scope(), absent_scope, anchor_scope)
    )

    assert residue_selection.as_scope() == ResidueSetScope(
        residue_ids=(ResidueId(chain_id="A", seq_num=10),)
    )
    assert atom_selection.as_scope() == AtomSetScope(
        atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=10), "CA"),)
    )
    assert attached_hydrogen_selection.as_scope() == AtomSetScope(
        atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=10), "CA"),)
    )
    assert absent_scope == AbsentResidueSpanScope(
        preceding_residue_id=ResidueId(chain_id="A", seq_num=10),
        following_residue_id=ResidueId(chain_id="A", seq_num=13),
        absent_residue_ids=(
            ResidueId(chain_id="A", seq_num=11),
            ResidueId(chain_id="A", seq_num=12),
        ),
    )
    assert anchor_scope == AnchorAtomPairScope(
        left_anchor_atom_ref=AtomRef(ResidueId(chain_id="A", seq_num=10), "C"),
        right_anchor_atom_ref=AtomRef(ResidueId(chain_id="A", seq_num=13), "N"),
    )
    assert composite_scope == CompositeScope(
        scopes=(
            ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=10),)),
            AbsentResidueSpanScope(
                preceding_residue_id=ResidueId(chain_id="A", seq_num=10),
                following_residue_id=ResidueId(chain_id="A", seq_num=13),
                absent_residue_ids=(
                    ResidueId(chain_id="A", seq_num=11),
                    ResidueId(chain_id="A", seq_num=12),
                ),
            ),
            AnchorAtomPairScope(
                left_anchor_atom_ref=AtomRef(
                    ResidueId(chain_id="A", seq_num=10),
                    "C",
                ),
                right_anchor_atom_ref=AtomRef(
                    ResidueId(chain_id="A", seq_num=13),
                    "N",
                ),
            ),
        )
    )


def test_workflow_execution_scopes_project_into_canonical_state_scopes() -> None:
    """Workflow execution scopes should share the canonical state-scope basis."""

    residue_scope = WorkflowResidueSetExecutionScope(
        residue_ids=(
            ResidueId(chain_id="A", seq_num=10),
            ResidueId(chain_id="A", seq_num=11),
        )
    )
    anchor_scope = WorkflowAnchorAtomPairExecutionScope(
        left_anchor_atom_ref=AtomRef(ResidueId(chain_id="A", seq_num=10), "C"),
        right_anchor_atom_ref=AtomRef(ResidueId(chain_id="A", seq_num=11), "N"),
    )
    composite_scope = WorkflowCompositeExecutionScope(
        scopes=(residue_scope, anchor_scope)
    )

    assert residue_scope.as_scope() == (
        ResidueSetScope(
            residue_ids=(
                ResidueId(chain_id="A", seq_num=10),
                ResidueId(chain_id="A", seq_num=11),
            )
        )
    )
    assert anchor_scope.as_scope() == (
        AnchorAtomPairScope(
            left_anchor_atom_ref=AtomRef(ResidueId(chain_id="A", seq_num=10), "C"),
            right_anchor_atom_ref=AtomRef(ResidueId(chain_id="A", seq_num=11), "N"),
        )
    )
    assert composite_scope.as_scope() == (
        CompositeScope(
            scopes=(
                ResidueSetScope(
                    residue_ids=(
                        ResidueId(chain_id="A", seq_num=10),
                        ResidueId(chain_id="A", seq_num=11),
                    )
                ),
                AnchorAtomPairScope(
                    left_anchor_atom_ref=AtomRef(
                        ResidueId(chain_id="A", seq_num=10),
                        "C",
                    ),
                    right_anchor_atom_ref=AtomRef(
                        ResidueId(chain_id="A", seq_num=11),
                        "N",
                    ),
                ),
            )
        )
    )


def test_atom_domain_and_transformer_context_project_source_scope() -> None:
    """Transformer internals should project active domains into state scopes."""

    structure = _single_residue_structure()
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    residuewise_scope = ResidueSetScope(
        residue_ids=(ResidueId(chain_id="A", seq_num=1),)
    )
    residuewise_domain = AtomInput(
        atom_indices=tuple(
            snapshot.structure.constitution.atom_index(atom_ref)
            for atom_ref in (
                AtomRef(ResidueId(chain_id="A", seq_num=1), "N"),
                AtomRef(ResidueId(chain_id="A", seq_num=1), "CA"),
                AtomRef(ResidueId(chain_id="A", seq_num=1), "C"),
            )
        ),
        basis=AtomInputBasis.RESIDUEWISE,
        selected_scope=residuewise_scope,
    )
    atomwise_scope = AtomSetScope(
        atom_refs=(
            AtomRef(ResidueId(chain_id="A", seq_num=1), "CA"),
            AtomRef(ResidueId(chain_id="A", seq_num=1), "C"),
        )
    )
    atomwise_domain = AtomInput(
        atom_indices=tuple(
            snapshot.structure.constitution.atom_index(atom_ref)
            for atom_ref in atomwise_scope.atom_refs
        ),
        basis=AtomInputBasis.ATOMWISE,
        selected_scope=atomwise_scope,
    )
    context = ProteinTransformationContext(
        source_snapshot=snapshot,
        atom_input=atomwise_domain,
    )

    assert residuewise_domain.as_scope() == ResidueSetScope(
        residue_ids=(ResidueId(chain_id="A", seq_num=1),)
    )
    assert atomwise_domain.as_scope() == AtomSetScope(
        atom_refs=(
            AtomRef(ResidueId(chain_id="A", seq_num=1), "CA"),
            AtomRef(ResidueId(chain_id="A", seq_num=1), "C"),
        )
    )
    assert context.source_scope() == AtomSetScope(
        atom_refs=(
            AtomRef(ResidueId(chain_id="A", seq_num=1), "CA"),
            AtomRef(ResidueId(chain_id="A", seq_num=1), "C"),
        )
    )


def test_scoped_state_represents_state_values_over_explicit_scope() -> None:
    """Scoped state should bind one state value to one explicit canonical scope."""

    scoped_state = ScopedState(
        scope=ResidueSetScope(
            residue_ids=(ResidueId(chain_id="A", seq_num=5),)
        ),
        value=HydrogenCoverageState.COMPLETE,
    )

    assert scoped_state.scope.kind is ScopeKind.RESIDUE_SET
    assert scoped_state.value is HydrogenCoverageState.COMPLETE


def test_structure_endpointd_state_binds_explicit_carrier_and_roundtrips() -> None:
    """Carrier-scoped state should express the full state basis and bridge back."""

    snapshot = ProteinStructureSnapshot.from_structure(_single_residue_structure())
    scoped_state = ScopedState(
        scope=ResidueSetScope(
            residue_ids=(ResidueId(chain_id="A", seq_num=1),)
        ),
        value=HydrogenCoverageState.COMPLETE,
    )
    carrier_state = scoped_state.for_carrier(snapshot)

    assert carrier_state == CarrierScopedState(
        carrier=snapshot,
        scope=ResidueSetScope(
            residue_ids=(ResidueId(chain_id="A", seq_num=1),)
        ),
        value=HydrogenCoverageState.COMPLETE,
    )
    assert carrier_state.without_carrier() == scoped_state


def test_scope_coarsenings_cover_canonical_containment_paths() -> None:
    """Pure scope coarsenings should cover canonical atom/residue/chain paths."""

    atom_scope = AtomSetScope(
        atom_refs=(
            AtomRef(ResidueId(chain_id="A", seq_num=1), "N"),
            AtomRef(ResidueId(chain_id="A", seq_num=2), "CA"),
        )
    )
    residue_scope = ResidueSetScope(
        residue_ids=(
            ResidueId(chain_id="A", seq_num=1),
            ResidueId(chain_id="B", seq_num=2),
        )
    )

    assert AtomToResidueSetScopeCoarsening().coarsen(atom_scope) == ResidueSetScope(
        residue_ids=(
            ResidueId(chain_id="A", seq_num=1),
            ResidueId(chain_id="A", seq_num=2),
        )
    )
    assert ResidueToChainSetScopeCoarsening().coarsen(residue_scope) == ChainSetScope(
        chain_ids=("A", "B")
    )
    assert AtomToChainSetScopeCoarsening().coarsen(atom_scope) == ChainSetScope(
        chain_ids=("A",)
    )
    assert ScopeToWholeStructureCoarsening().coarsen(
        atom_scope
    ) == WholeStructureScope()


def test_absent_span_scope_relations_allow_same_anchor_subset_refinement() -> None:
    """Absent-span scopes should refine and overlap by same-anchor residue subsets."""

    coarse_scope = AbsentResidueSpanScope(
        preceding_residue_id=ResidueId(chain_id="A", seq_num=10),
        following_residue_id=ResidueId(chain_id="A", seq_num=14),
        absent_residue_ids=(
            ResidueId(chain_id="A", seq_num=11),
            ResidueId(chain_id="A", seq_num=12),
            ResidueId(chain_id="A", seq_num=13),
        ),
    )
    finer_scope = AbsentResidueSpanScope(
        preceding_residue_id=ResidueId(chain_id="A", seq_num=10),
        following_residue_id=ResidueId(chain_id="A", seq_num=14),
        absent_residue_ids=(
            ResidueId(chain_id="A", seq_num=12),
        ),
    )
    disjoint_scope = AbsentResidueSpanScope(
        preceding_residue_id=ResidueId(chain_id="A", seq_num=20),
        following_residue_id=ResidueId(chain_id="A", seq_num=24),
        absent_residue_ids=(
            ResidueId(chain_id="A", seq_num=21),
        ),
    )

    assert scope_refines(finer_scope, coarse_scope)
    assert not scope_refines(coarse_scope, finer_scope)
    assert scopes_overlap(finer_scope, coarse_scope)
    assert not scopes_overlap(disjoint_scope, coarse_scope)


def test_only_supported_scope_subset_lowers_into_observed_atom_scopes() -> None:
    """Only supported semantic scopes should lower into observed atom scopes."""

    snapshot = ProteinStructureSnapshot.from_structure(_single_residue_structure())

    whole_scope = WholeStructureScope()
    chain_scope = ChainSetScope(chain_ids=("A",))
    residue_scope = ResidueSetScope(
        residue_ids=(ResidueId(chain_id="A", seq_num=1),)
    )
    atom_scope = AtomSetScope(
        atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=1), "CA"),)
    )
    absent_scope = AbsentResidueSpanScope(
        preceding_residue_id=ResidueId(chain_id="A", seq_num=1),
        absent_residue_ids=(ResidueId(chain_id="A", seq_num=2),),
    )
    anchor_scope = AnchorAtomPairScope(
        left_anchor_atom_ref=AtomRef(ResidueId(chain_id="A", seq_num=1), "N"),
        right_anchor_atom_ref=AtomRef(ResidueId(chain_id="A", seq_num=1), "C"),
    )

    assert OBSERVED_ATOM_SCOPE_LOWERING.supports(whole_scope)
    assert OBSERVED_ATOM_SCOPE_LOWERING.supports(chain_scope)
    assert OBSERVED_ATOM_SCOPE_LOWERING.supports(residue_scope)
    assert OBSERVED_ATOM_SCOPE_LOWERING.supports(atom_scope)
    assert not OBSERVED_ATOM_SCOPE_LOWERING.supports(absent_scope)
    assert not OBSERVED_ATOM_SCOPE_LOWERING.supports(anchor_scope)

    assert OBSERVED_ATOM_SCOPE_LOWERING.lower(
        whole_scope,
        carrier=snapshot,
    ) == AtomSetScope(
        atom_refs=(
            AtomRef(ResidueId(chain_id="A", seq_num=1), "N"),
            AtomRef(ResidueId(chain_id="A", seq_num=1), "CA"),
            AtomRef(ResidueId(chain_id="A", seq_num=1), "C"),
            AtomRef(ResidueId(chain_id="A", seq_num=1), "O"),
        )
    )
    assert OBSERVED_ATOM_SCOPE_LOWERING.lower(
        chain_scope,
        carrier=snapshot,
    ) == AtomSetScope(
        atom_refs=(
            AtomRef(ResidueId(chain_id="A", seq_num=1), "N"),
            AtomRef(ResidueId(chain_id="A", seq_num=1), "CA"),
            AtomRef(ResidueId(chain_id="A", seq_num=1), "C"),
            AtomRef(ResidueId(chain_id="A", seq_num=1), "O"),
        )
    )
    assert OBSERVED_ATOM_SCOPE_LOWERING.lower(
        residue_scope,
        carrier=snapshot,
    ) == AtomSetScope(
        atom_refs=(
            AtomRef(ResidueId(chain_id="A", seq_num=1), "N"),
            AtomRef(ResidueId(chain_id="A", seq_num=1), "CA"),
            AtomRef(ResidueId(chain_id="A", seq_num=1), "C"),
            AtomRef(ResidueId(chain_id="A", seq_num=1), "O"),
        )
    )
    assert OBSERVED_ATOM_SCOPE_LOWERING.lower(
        atom_scope,
        carrier=snapshot,
    ) == AtomSetScope(
        atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=1), "CA"),)
    )

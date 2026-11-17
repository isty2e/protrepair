"""Unit tests for canonical scoped-state contracts and adapters."""

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
    Scope,
    ScopeKind,
    ScopeToWholeStructureCoarsening,
    WholeStructureScope,
    scope_refines,
    scopes_overlap,
)
from protrepair.scope.observed_atom_scope_lowering import OBSERVED_ATOM_SCOPE_LOWERING
from protrepair.state import (
    BackboneHeavyAtomCompletenessState,
    CarrierScopedState,
    HydrogenCoverageState,
    ScopedState,
    SidechainHeavyAtomCompletenessState,
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


def test_local_and_semantic_scopes_project_into_canonical_state_scopes() -> None:
    """Boundary local scope specs should align with canonical semantic scopes."""

    residue_selection = LocalScopeSpec.from_residues(
        (ResidueId(chain_id="A", seq_num=10),)
    )
    atom_selection = LocalScopeSpec.from_atoms(
        (AtomRef(ResidueId(chain_id="A", seq_num=10), "CA"),)
    )
    attached_hydrogen_selection = LocalScopeSpec.from_atoms_with_attached_hydrogens(
        (AtomRef(ResidueId(chain_id="A", seq_num=10), "CA"),)
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
        scope=ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=5),)),
        value=HydrogenCoverageState.COMPLETE,
    )

    assert scoped_state.scope.kind is ScopeKind.RESIDUE_SET
    assert scoped_state.value is HydrogenCoverageState.COMPLETE


def test_structure_endpointd_state_binds_explicit_carrier_and_roundtrips() -> None:
    """Carrier-scoped state should express the full state basis and bridge back."""

    snapshot = ProteinStructureSnapshot.from_structure(_single_residue_structure())
    scoped_state = ScopedState(
        scope=ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),)),
        value=HydrogenCoverageState.COMPLETE,
    )
    carrier_state = scoped_state.for_carrier(snapshot)

    assert carrier_state == CarrierScopedState(
        carrier=snapshot,
        scope=ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),)),
        value=HydrogenCoverageState.COMPLETE,
    )
    assert carrier_state.without_carrier() == scoped_state


@pytest.mark.parametrize("bound", (False, True))
@pytest.mark.parametrize(
    "scope", (WholeStructureScope(), ResidueSetScope((ResidueId("A", 1),)))
)
@pytest.mark.parametrize(
    ("left_value", "right_value"),
    (
        (BackboneHeavyAtomCompletenessState.COMPLETE, HydrogenCoverageState.COMPLETE),
        (SidechainHeavyAtomCompletenessState.COMPLETE, HydrogenCoverageState.COMPLETE),
        (
            BackboneHeavyAtomCompletenessState.COMPLETE,
            SidechainHeavyAtomCompletenessState.COMPLETE,
        ),
        (
            BackboneHeavyAtomCompletenessState.INCOMPLETE,
            SidechainHeavyAtomCompletenessState.INCOMPLETE,
        ),
        ("complete", HydrogenCoverageState.COMPLETE),
        (1, 1.0),
        (1, True),
        (1.0, True),
    ),
)
def test_scoped_state_identity_keeps_concrete_value_types(
    left_value: str | float,
    right_value: str | float,
    scope: Scope,
    bound: bool,
) -> None:
    left = ScopedState(scope, left_value)
    right = ScopedState(scope, right_value)
    repeated = ScopedState(scope, left_value)
    if bound:
        left = left.for_carrier(("carrier",))
        right = right.for_carrier(("carrier",))
        repeated = repeated.for_carrier(("carrier",))

    assert left == repeated
    assert repeated == left
    assert hash(left) == hash(repeated)
    assert left != right
    assert right != left
    assert len({left, right, repeated}) == 2
    values = {left: "left", right: "right"}
    assert values[repeated] == "left"
    assert values[right] == "right"


@pytest.mark.parametrize("value", (HydrogenCoverageState.COMPLETE, "complete", 1, None))
def test_scoped_state_round_trip_preserves_value_type(value: str | int | None) -> None:
    original = ScopedState(WholeStructureScope(), value)
    bound = original.for_carrier(("carrier",))
    restored = bound.without_carrier()

    assert type(restored.value) is type(value)
    assert restored == original
    assert hash(restored) == hash(original)


def test_scoped_state_equality_keeps_carrier_identity_shortcut() -> None:
    class EqualityTrap:
        def __eq__(self, other: object) -> bool:
            raise AssertionError("same-object carrier equality must not be called")

    carrier = EqualityTrap()
    left = CarrierScopedState(
        carrier, WholeStructureScope(), HydrogenCoverageState.COMPLETE
    )
    right = CarrierScopedState(
        carrier, WholeStructureScope(), HydrogenCoverageState.COMPLETE
    )

    assert left == right
    assert right == left


def test_scoped_state_equality_preserves_scope_and_carrier_comparison() -> None:
    value = HydrogenCoverageState.COMPLETE
    whole = ScopedState(WholeStructureScope(), value)
    local = ScopedState(ResidueSetScope((ResidueId("A", 1),)), value)

    assert whole != local
    assert whole.for_carrier(["a"]) == whole.for_carrier(["a"])
    assert whole.for_carrier(["a"]) != whole.for_carrier(["b"])
    assert whole.for_carrier(["a"]) != local.for_carrier(["a"])


@pytest.mark.parametrize("bound", (False, True))
def test_scoped_state_rejects_other_wrapper_classes_in_equality(bound: bool) -> None:
    value = ScopedState(WholeStructureScope(), HydrogenCoverageState.COMPLETE)
    subject = value.for_carrier(("carrier",)) if bound else value
    other = value if bound else value.for_carrier(("carrier",))

    assert subject.__eq__(other) is NotImplemented
    assert subject.__eq__(object()) is NotImplemented
    assert subject != other


def test_scoped_state_equality_keeps_exact_wrapper_class() -> None:
    class DerivedState(ScopedState[str]):
        pass

    base = ScopedState(WholeStructureScope(), "complete")
    derived = DerivedState(WholeStructureScope(), "complete")

    assert base.__eq__(derived) is NotImplemented
    assert derived.__eq__(base) is NotImplemented
    assert base != derived
    assert derived == DerivedState(WholeStructureScope(), "complete")


def test_scoped_state_hash_keeps_unhashable_payload_failures() -> None:
    value = ScopedState(WholeStructureScope(), ["complete"])
    with pytest.raises(TypeError):
        hash(value)
    with pytest.raises(TypeError):
        hash(value.for_carrier(("carrier",)))
    with pytest.raises(TypeError):
        hash(ScopedState(WholeStructureScope(), "complete").for_carrier([]))


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
    assert (
        ScopeToWholeStructureCoarsening().coarsen(atom_scope) == WholeStructureScope()
    )


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
        absent_residue_ids=(ResidueId(chain_id="A", seq_num=12),),
    )
    disjoint_scope = AbsentResidueSpanScope(
        preceding_residue_id=ResidueId(chain_id="A", seq_num=20),
        following_residue_id=ResidueId(chain_id="A", seq_num=24),
        absent_residue_ids=(ResidueId(chain_id="A", seq_num=21),),
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
    residue_scope = ResidueSetScope(residue_ids=(ResidueId(chain_id="A", seq_num=1),))
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
    ) == AtomSetScope(atom_refs=(AtomRef(ResidueId(chain_id="A", seq_num=1), "CA"),))

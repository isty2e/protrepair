"""Regression tests for transformer package facades."""


def test_transformer_contract_facade_exposes_base_contracts() -> None:
    """Transformer contract facade should expose base contracts eagerly."""

    from protrepair.transformer.contracts import (
        AtomInput,
        ProteinTransformationContext,
        ProteinTransformer,
        TransformationResult,
    )

    assert AtomInput.__name__ == "AtomInput"
    assert ProteinTransformationContext.__name__ == "ProteinTransformationContext"
    assert ProteinTransformer.__name__ == "ProteinTransformer"
    assert TransformationResult.__name__ == "TransformationResult"


def test_workflow_action_facade_exposes_workflow_visible_transformers() -> None:
    """Workflow action facade should expose concrete workflow transformers."""

    from protrepair.workflow.actions import (
        BackboneWindowRefinementTransformer,
        CommittedPackingTransformer,
        ExternalSpanReconstructionTransformer,
        HeavyAtomCompletionTransformer,
        HydrogenCompletionTransformer,
        LocalRefinementTransformer,
        RetainedNonPolymerHydrogenCompletionTransformer,
        StereochemistryCorrectionTransformer,
        TerminalAugmentationTransformer,
    )

    assert (
        BackboneWindowRefinementTransformer.__name__
        == "BackboneWindowRefinementTransformer"
    )
    assert CommittedPackingTransformer.__name__ == "CommittedPackingTransformer"
    assert (
        ExternalSpanReconstructionTransformer.__name__
        == "ExternalSpanReconstructionTransformer"
    )
    assert HeavyAtomCompletionTransformer.__name__ == "HeavyAtomCompletionTransformer"
    assert HydrogenCompletionTransformer.__name__ == "HydrogenCompletionTransformer"
    assert LocalRefinementTransformer.__name__ == "LocalRefinementTransformer"
    assert (
        RetainedNonPolymerHydrogenCompletionTransformer.__name__
        == "RetainedNonPolymerHydrogenCompletionTransformer"
    )
    assert (
        StereochemistryCorrectionTransformer.__name__
        == "StereochemistryCorrectionTransformer"
    )
    assert TerminalAugmentationTransformer.__name__ == "TerminalAugmentationTransformer"


def test_refinement_contract_facade_exposes_workflow_independent_contracts() -> None:
    """Refinement contract facade should avoid workflow-visible action imports."""

    from protrepair.transformer.refinement.contracts import (
        BackboneWindowRefinementSpec,
        LocalRefinementRequest,
        RepairLocalRefinementDirective,
        RepairRefinementSpec,
        execute_local_transformation,
    )

    assert BackboneWindowRefinementSpec.__name__ == "BackboneWindowRefinementSpec"
    assert LocalRefinementRequest.__name__ == "LocalRefinementRequest"
    assert RepairLocalRefinementDirective.__name__ == "RepairLocalRefinementDirective"
    assert RepairRefinementSpec.__name__ == "RepairRefinementSpec"
    assert execute_local_transformation.__name__ == "execute_local_transformation"

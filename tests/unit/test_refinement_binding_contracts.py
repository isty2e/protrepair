"""Tests for refinement directive and binding-policy contracts."""

from math import nan

import pytest
from tests.support.refinement_type_fixtures import ready_atom_scope_facts

from protrepair.errors import RefinementError
from protrepair.scope import AbsentResidueSpanScope
from protrepair.state import HydrogenCoverageState, TopologyAvailabilityState
from protrepair.structure.labels import ResidueId
from protrepair.transformer.atom_input import AtomInputBasis
from protrepair.transformer.continuous.binding_policy import (
    ContinuousRelaxationBackendCapabilities,
    ContinuousRelaxationBindingReason,
    RecommendedContinuousRelaxationBinding,
    decide_continuous_relaxation_binding,
)
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationConfig,
    ContinuousRelaxationForceField,
    ContinuousRelaxationProfile,
)
from protrepair.transformer.local import (
    DirectRegionTransformationSpec,
    LocalScopeLowering,
    LocalScopeSpec,
)
from protrepair.transformer.refinement.spec import (
    BackboneWindowRefinementSpec,
    RefinementOperatorFamily,
)


def test_refinement_spec_validates_scalar_configuration() -> None:
    """Direct specs should normalize once and reject invalid scalar config."""

    residue_selection = LocalScopeSpec.from_residues(
        (ResidueId(chain_id="A", seq_num=1),)
    )
    spec = DirectRegionTransformationSpec(
        scope_spec=residue_selection,
        force_field=ContinuousRelaxationForceField.UFF,
        config=ContinuousRelaxationConfig(
            backend_name=" RDKIT ",
            context_radius_angstrom=5.0,
            max_iterations=25,
        ),
    )

    assert spec.config.backend_name == "rdkit"
    assert isinstance(spec.scope_spec, LocalScopeSpec)
    assert spec.scope_spec.referenced_residue_ids() == (
        ResidueId(chain_id="A", seq_num=1),
    )
    assert spec.scope_spec.lowering is LocalScopeLowering.RESIDUE_ATOMS
    assert spec.config == ContinuousRelaxationConfig(
        backend_name="rdkit",
        context_radius_angstrom=5.0,
        max_iterations=25,
    )

    mmff_spec = DirectRegionTransformationSpec(
        scope_spec=residue_selection,
        force_field=ContinuousRelaxationForceField.MMFF,
    )
    assert mmff_spec.force_field is ContinuousRelaxationForceField.MMFF

    with pytest.raises(ValueError, match="non-negative"):
        DirectRegionTransformationSpec(
            scope_spec=residue_selection,
            force_field=ContinuousRelaxationForceField.UFF,
            config=ContinuousRelaxationConfig(context_radius_angstrom=-1.0),
        )

    with pytest.raises(ValueError, match="finite"):
        DirectRegionTransformationSpec(
            scope_spec=residue_selection,
            force_field=ContinuousRelaxationForceField.UFF,
            config=ContinuousRelaxationConfig(context_radius_angstrom=nan),
        )


def test_direct_region_transformation_rejects_non_scope_specs() -> None:
    """Direct refinement boundary should remain honest about local scope specs."""

    scope_spec = LocalScopeSpec.from_residues((ResidueId(chain_id="A", seq_num=1),))

    with pytest.raises(TypeError, match="LocalScopeSpec scope_spec"):
        DirectRegionTransformationSpec(
            scope_spec=AbsentResidueSpanScope(
                preceding_residue_id=ResidueId(chain_id="A", seq_num=10),
                following_residue_id=ResidueId(chain_id="A", seq_num=13),
            ),  # type: ignore[arg-type]
            force_field=ContinuousRelaxationForceField.UFF,
        )

    with pytest.raises(TypeError, match="finite number"):
        DirectRegionTransformationSpec(
            scope_spec=scope_spec,
            force_field=ContinuousRelaxationForceField.UFF,
            config=ContinuousRelaxationConfig(context_radius_angstrom=True),
        )

    with pytest.raises(ValueError, match="positive"):
        DirectRegionTransformationSpec(
            scope_spec=scope_spec,
            force_field=ContinuousRelaxationForceField.UFF,
            config=ContinuousRelaxationConfig(max_iterations=0),
        )

    with pytest.raises(TypeError, match="integer"):
        DirectRegionTransformationSpec(
            scope_spec=scope_spec,
            force_field=ContinuousRelaxationForceField.UFF,
            config=ContinuousRelaxationConfig(max_iterations=True),
        )

    with pytest.raises(TypeError, match="integer"):
        DirectRegionTransformationSpec(
            scope_spec=scope_spec,
            force_field=ContinuousRelaxationForceField.UFF,
            config=ContinuousRelaxationConfig(
                max_iterations=1.5,  # pyright: ignore[reportArgumentType]
            ),
        )

    with pytest.raises(TypeError, match="ContinuousRelaxationForceField"):
        DirectRegionTransformationSpec(
            scope_spec=scope_spec,
            force_field="uff",  # type: ignore[arg-type]
        )


def test_backbone_window_refinement_spec_is_not_a_direct_ff_scope() -> None:
    """Backbone-window correction should be a distinct operator contract."""

    residue_ids = (
        ResidueId(chain_id="A", seq_num=10),
        ResidueId(chain_id="A", seq_num=11),
        ResidueId(chain_id="A", seq_num=12),
    )
    spec = BackboneWindowRefinementSpec(
        residue_ids=(residue_ids[0], residue_ids[1], residue_ids[1], residue_ids[2]),
        movable_atom_names=("n", "CA", "C", "O", "O"),
    )

    assert spec.operator_family is RefinementOperatorFamily.BACKBONE_WINDOW_REFINEMENT
    assert spec.residue_ids == residue_ids
    assert spec.movable_atom_names == ("N", "CA", "C", "O")
    assert spec.as_residue_scope().residue_ids == residue_ids

    with pytest.raises(TypeError, match="LocalScopeSpec scope_spec"):
        DirectRegionTransformationSpec(
            scope_spec=spec,  # type: ignore[arg-type]
            force_field=ContinuousRelaxationForceField.UFF,
        )


def test_backbone_window_refinement_spec_requires_ordered_chain_window() -> None:
    """Backbone-window contracts should reject non-window scope shapes."""

    with pytest.raises(ValueError, match="at least two residues"):
        BackboneWindowRefinementSpec(
            residue_ids=(ResidueId(chain_id="A", seq_num=10),)
        )

    with pytest.raises(ValueError, match="one ordered chain window"):
        BackboneWindowRefinementSpec(
            residue_ids=(
                ResidueId(chain_id="A", seq_num=10),
                ResidueId(chain_id="B", seq_num=11),
            )
        )

    with pytest.raises(ValueError, match="chain order"):
        BackboneWindowRefinementSpec(
            residue_ids=(
                ResidueId(chain_id="A", seq_num=11),
                ResidueId(chain_id="A", seq_num=10),
            )
        )

    with pytest.raises(ValueError, match="polymer backbone atoms"):
        BackboneWindowRefinementSpec(
            residue_ids=(
                ResidueId(chain_id="A", seq_num=10),
                ResidueId(chain_id="A", seq_num=11),
            ),
            movable_atom_names=("CB",),
        )


def test_backend_capabilities_choose_preferred_force_field() -> None:
    """Recommended binding should honor declared backend profile preference."""

    decision = ContinuousRelaxationBackendCapabilities(
        backend_name="rdkit",
        supported_profiles=frozenset(
            {
                ContinuousRelaxationProfile.RDKIT_UFF,
                ContinuousRelaxationProfile.RDKIT_MMFF,
            }
        ),
        preferred_profile=ContinuousRelaxationProfile.RDKIT_MMFF,
    ).recommended_binding_decision(
        ContinuousRelaxationConfig(backend_name="rdkit"),
        atom_scope_facts=ready_atom_scope_facts(),
    )

    assert decision.settings.backend_name == "rdkit"
    assert decision.settings.profile is ContinuousRelaxationProfile.RDKIT_MMFF
    assert decision.settings.force_field is ContinuousRelaxationForceField.MMFF
    assert (
        decision.reason
        is ContinuousRelaxationBindingReason.BACKEND_PREFERRED_PROFILE
    )


def test_backend_capabilities_fall_back_to_only_supported_force_field() -> None:
    """Recommended binding should use the only supported profile."""

    decision = ContinuousRelaxationBackendCapabilities(
        backend_name="rdkit",
        supported_profiles=frozenset({ContinuousRelaxationProfile.RDKIT_UFF}),
    ).recommended_binding_decision(
        ContinuousRelaxationConfig(backend_name="rdkit"),
        atom_scope_facts=ready_atom_scope_facts(),
    )

    assert decision.settings.profile is ContinuousRelaxationProfile.RDKIT_UFF
    assert decision.settings.force_field is ContinuousRelaxationForceField.UFF
    assert (
        decision.reason is ContinuousRelaxationBindingReason.ONLY_SUPPORTED_PROFILE
    )


def test_backend_capabilities_reject_ambiguous_recommended_binding() -> None:
    """Recommended binding should reject multi-profile ambiguity."""

    with pytest.raises(RefinementError, match="ambiguous"):
        ContinuousRelaxationBackendCapabilities(
            backend_name="rdkit",
            supported_profiles=frozenset(
                {
                    ContinuousRelaxationProfile.RDKIT_UFF,
                    ContinuousRelaxationProfile.RDKIT_MMFF,
                }
            ),
        ).recommended_binding_decision(
            ContinuousRelaxationConfig(backend_name="rdkit"),
            atom_scope_facts=ready_atom_scope_facts(),
        )


def test_recommended_binding_prefers_mmff_for_hydrogenated_residuewise_rdkit() -> None:
    """RDKit recommendation should prefer MMFF for hydrogenated residuewise use."""

    decision = decide_continuous_relaxation_binding(
        RecommendedContinuousRelaxationBinding(),
        ContinuousRelaxationConfig(backend_name="rdkit"),
        atom_scope_facts=ready_atom_scope_facts(),
        atom_input_basis=AtomInputBasis.RESIDUEWISE,
    )

    assert decision.settings.force_field is ContinuousRelaxationForceField.MMFF
    assert (
        decision.reason is ContinuousRelaxationBindingReason.HYDROGENATED_DOMAIN_POLICY
    )


def test_config_bind_accepts_explicit_rdkit_builtin_profile() -> None:
    """Explicit profiles should bind directly when the runtime matches config."""

    settings = ContinuousRelaxationConfig(backend_name="rdkit").bind(
        ContinuousRelaxationProfile.RDKIT_UFF
    )

    assert settings.profile is ContinuousRelaxationProfile.RDKIT_UFF
    assert settings.backend_name == "rdkit"
    assert settings.force_field is ContinuousRelaxationForceField.UFF


def test_recommended_binding_prefers_mmff_for_atomwise_rdkit_domains() -> None:
    """RDKit recommendation should also start from MMFF for atomwise domains."""

    decision = decide_continuous_relaxation_binding(
        RecommendedContinuousRelaxationBinding(),
        ContinuousRelaxationConfig(backend_name="rdkit"),
        atom_scope_facts=ready_atom_scope_facts(),
        atom_input_basis=AtomInputBasis.ATOMWISE,
    )

    assert decision.settings.force_field is ContinuousRelaxationForceField.MMFF
    assert (
        decision.reason is ContinuousRelaxationBindingReason.HYDROGENATED_DOMAIN_POLICY
    )


def test_recommended_binding_prefers_uff_for_non_hydrogenated_residuewise_rdkit() -> (
    None
):
    """RDKit recommendation should reject domains without realized hydrogens."""

    with pytest.raises(
        RefinementError,
        match="requires hydrogens to be fully realized",
    ):
        decide_continuous_relaxation_binding(
            RecommendedContinuousRelaxationBinding(),
            ContinuousRelaxationConfig(backend_name="rdkit"),
            atom_scope_facts=ready_atom_scope_facts(
                hydrogen_coverage_state=HydrogenCoverageState.NONE,
                hydrogen_atom_count=0,
                hydrogen_topology_state=TopologyAvailabilityState.NOT_APPLICABLE,
            ),
            atom_input_basis=AtomInputBasis.RESIDUEWISE,
        )

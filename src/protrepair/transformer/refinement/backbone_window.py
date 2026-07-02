"""Backbone-window refinement executor."""

from dataclasses import dataclass, replace

from protrepair.chemistry import ComponentLibrary, RestraintLibrary
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.artifacts import RegionTransformationResult
from protrepair.transformer.atom_input import (
    AtomInput,
    AtomInputBasis,
    AtomInputRealization,
)
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.settings import ContinuousRelaxationSettings
from protrepair.transformer.refinement.local_pipeline.runtime import (
    execute_local_transformation,
)
from protrepair.transformer.refinement.spec import BackboneWindowRefinementSpec

OPTIONAL_BACKBONE_WINDOW_ATOM_NAMES = frozenset({"OXT"})


@dataclass(frozen=True, slots=True)
class BackboneWindowRefinementExecutor:
    """Executor for ordered same-chain backbone-window refinement specs."""

    spec: ContinuousRelaxationSettings
    component_library: ComponentLibrary | None = None
    restraint_library: RestraintLibrary | None = None

    def transform(
        self,
        structure: ProteinStructure,
        window_spec: BackboneWindowRefinementSpec,
    ) -> RegionTransformationResult:
        """Run backbone-window refinement over one source structure."""

        return execute_backbone_window_refinement(
            structure,
            window_spec,
            spec=self.spec,
            component_library=self.component_library,
            restraint_library=self.restraint_library,
        )


def execute_backbone_window_refinement(
    structure: ProteinStructure,
    window_spec: BackboneWindowRefinementSpec,
    *,
    spec: ContinuousRelaxationSettings,
    component_library: ComponentLibrary | None = None,
    restraint_library: RestraintLibrary | None = None,
) -> RegionTransformationResult:
    """Execute one backbone-window refinement over an ordered residue window."""

    if not isinstance(structure, ProteinStructure):
        raise TypeError("backbone-window refinement requires a ProteinStructure")
    if not isinstance(window_spec, BackboneWindowRefinementSpec):
        raise TypeError(
            "backbone-window refinement requires a BackboneWindowRefinementSpec"
        )
    if not isinstance(spec, ContinuousRelaxationSettings):
        raise TypeError(
            "backbone-window refinement requires ContinuousRelaxationSettings"
        )

    snapshot = ProteinStructureSnapshot.from_structure(structure)
    atom_input = atom_input_from_backbone_window_refinement_spec(
        snapshot,
        window_spec,
    )
    result = execute_local_transformation(
        ProteinTransformationContext.from_snapshot_atom_input(
            snapshot,
            atom_input,
        ),
        spec=spec,
        component_library=component_library,
        restraint_library=restraint_library,
    )
    return _with_backbone_window_rejection_context(result)


def atom_input_from_backbone_window_refinement_spec(
    snapshot: ProteinStructureSnapshot,
    window_spec: BackboneWindowRefinementSpec,
) -> AtomInput:
    """Lower one backbone-window contract into residuewise movable backbone atoms."""

    if not isinstance(window_spec, BackboneWindowRefinementSpec):
        raise TypeError(
            "backbone-window atom input requires a BackboneWindowRefinementSpec"
        )

    constitution = snapshot.structure.constitution
    selected_atom_indices = []
    missing_atom_tokens = []
    for residue_id in window_spec.residue_ids:
        residue_site = constitution.residue_or_ligand(residue_id)
        if residue_site is None:
            raise ValueError(
                "backbone-window refinement references unknown residue "
                f"{residue_id.display_token()}"
            )
        if residue_site.is_hetero:
            raise ValueError(
                "backbone-window refinement requires polymer residues, got "
                f"{residue_id.display_token()}"
            )

        residue_index = constitution.residue_index(residue_id)
        for atom_name in window_spec.movable_atom_names:
            if not residue_site.has_atom_site(atom_name):
                if atom_name in OPTIONAL_BACKBONE_WINDOW_ATOM_NAMES:
                    continue

                missing_atom_tokens.append(f"{residue_id.display_token()}.{atom_name}")
                continue

            selected_atom_indices.append(
                constitution.atom_index_in_residue(residue_index, atom_name)
            )

    if missing_atom_tokens:
        raise ValueError(
            "backbone-window refinement requires present requested backbone atoms: "
            + ", ".join(missing_atom_tokens)
        )

    return AtomInput(
        atom_indices=tuple(selected_atom_indices),
        basis=AtomInputBasis.RESIDUEWISE,
        realization=AtomInputRealization.RESIDUE_BACKBONE_ATOMS,
        selected_scope=window_spec.as_residue_scope(),
    )


def _with_backbone_window_rejection_context(
    result: RegionTransformationResult,
) -> RegionTransformationResult:
    if result.moved_atom_count() != 0:
        return result
    if not any(
        issue.kind is ValidationIssueKind.REFINEMENT_REJECTED
        for issue in result.issues
    ):
        return result

    return replace(
        result,
        issues=tuple(
            _with_backbone_window_issue_message(issue) for issue in result.issues
        ),
    )


def _with_backbone_window_issue_message(issue: ValidationIssue) -> ValidationIssue:
    if issue.kind is not ValidationIssueKind.REFINEMENT_REJECTED:
        return issue

    prefix = "backbone-window refinement was not sufficient: "
    if issue.message.startswith(prefix):
        return issue

    return replace(issue, message=prefix + issue.message)

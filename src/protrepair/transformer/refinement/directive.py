"""Local-refinement directives and current-state-bound execution artifacts."""

from dataclasses import dataclass

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.scope import AtomSetScope, ResidueSetScope
from protrepair.state.domain import AtomScopeStateFacts
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.atom_input import AtomInput
from protrepair.transformer.continuous.binding import (
    ContinuousRelaxationBinding,
    ContinuousRelaxationBindingDecision,
    ManualContinuousRelaxationBinding,
    RecommendedContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.binding_policy import (
    decide_continuous_relaxation_binding,
)
from protrepair.transformer.continuous.readiness import (
    derive_atom_scope_continuous_relaxation_facts,
    require_atom_scope_continuous_relaxation_execution,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationConfig
from protrepair.transformer.local.models import LocalScopeSpec


@dataclass(frozen=True, slots=True)
class BoundRepairLocalRefinementExecution:
    """Current-state-bound repair-stage refinement execution artifact."""

    atom_input: AtomInput
    atom_scope_facts: AtomScopeStateFacts
    binding_decision: ContinuousRelaxationBindingDecision


@dataclass(frozen=True, slots=True)
class RepairLocalRefinementDirective:
    """Semantic repair-stage refinement directive for workflow internals."""

    config: ContinuousRelaxationConfig
    binding: ContinuousRelaxationBinding
    scope_spec: LocalScopeSpec
    execution_scope_spec: LocalScopeSpec | None = None

    def __post_init__(self) -> None:
        if not isinstance(
            self.binding,
            (
                ManualContinuousRelaxationBinding,
                RecommendedContinuousRelaxationBinding,
            ),
        ):
            raise TypeError(
                "repair local refinement binding must be a manual or recommended "
                "continuous-relaxation binding value"
            )
        if not isinstance(self.scope_spec, LocalScopeSpec):
            raise TypeError(
                "repair local refinement directives require a LocalScopeSpec value"
            )
        if self.execution_scope_spec is None:
            return

        if not isinstance(self.execution_scope_spec, LocalScopeSpec):
            raise TypeError(
                "repair local refinement execution_scope_spec must be a "
                "LocalScopeSpec value when present"
            )
        if not self.scope_spec.is_residuewise():
            raise TypeError(
                "repair local refinement execution_scope_spec is supported only "
                "for residuewise semantic scopes"
            )
        if not self.execution_scope_spec.is_residuewise():
            raise TypeError(
                "repair local refinement execution_scope_spec must stay "
                "residuewise"
            )

        semantic_residue_ids = set(self.scope_spec.referenced_residue_ids())
        execution_residue_ids = set(
            self.execution_scope_spec.referenced_residue_ids()
        )
        if not semantic_residue_ids <= execution_residue_ids:
            raise ValueError(
                "repair local refinement execution_scope_spec must cover the "
                "semantic scope residue ids"
            )

    @classmethod
    def from_atom_input(
        cls,
        atom_input: AtomInput,
        *,
        binding: ContinuousRelaxationBinding,
        config: ContinuousRelaxationConfig | None = None,
    ) -> "RepairLocalRefinementDirective":
        """Build one canonical directive from one normalized atom domain."""

        return cls(
            config=ContinuousRelaxationConfig() if config is None else config,
            binding=binding,
            scope_spec=_local_scope_spec_from_atom_input(atom_input),
            execution_scope_spec=None,
        )

    @classmethod
    def from_residue_ids(
        cls,
        residue_ids: tuple[ResidueId, ...],
        *,
        binding: ContinuousRelaxationBinding,
        config: ContinuousRelaxationConfig | None = None,
    ) -> "RepairLocalRefinementDirective":
        """Build one residuewise directive from canonical residue identifiers."""

        return cls(
            config=ContinuousRelaxationConfig() if config is None else config,
            binding=binding,
            scope_spec=LocalScopeSpec.from_residues(residue_ids),
            execution_scope_spec=None,
        )

    @classmethod
    def from_atom_refs(
        cls,
        atom_refs: tuple[AtomRef, ...],
        *,
        binding: ContinuousRelaxationBinding,
        config: ContinuousRelaxationConfig | None = None,
    ) -> "RepairLocalRefinementDirective":
        """Build one atomwise directive from canonical atom identifiers."""

        return cls(
            config=ContinuousRelaxationConfig() if config is None else config,
            binding=binding,
            scope_spec=LocalScopeSpec.from_atoms(atom_refs),
            execution_scope_spec=None,
        )

    @property
    def selected_scope(self) -> ResidueSetScope | AtomSetScope:
        """Return the semantic scope targeted by this directive."""

        return self.scope_spec.scope

    def resolved_execution_scope_spec(self) -> LocalScopeSpec:
        """Return the execution-local scope spec to lower against one snapshot."""

        return (
            self.scope_spec
            if self.execution_scope_spec is None
            else self.execution_scope_spec
        )

    def targets_residue(self, residue_id: ResidueId) -> bool:
        """Return whether one residue falls inside the refinement directive."""

        selected_scope = self.scope_spec.scope
        if isinstance(selected_scope, ResidueSetScope):
            return residue_id in selected_scope.residue_ids

        return any(
            atom_ref.residue_id == residue_id
            for atom_ref in selected_scope.atom_refs
        )

    def single_focus_residue_id(self) -> ResidueId | None:
        """Return the sole selected residue id when the directive has one focus."""

        selected_scope = self.scope_spec.scope
        if isinstance(selected_scope, ResidueSetScope):
            if len(selected_scope.residue_ids) != 1:
                return None

            return selected_scope.residue_ids[0]

        ordered_residue_ids: list[ResidueId] = []
        seen_residue_ids: set[ResidueId] = set()
        for atom_ref in selected_scope.atom_refs:
            if atom_ref.residue_id in seen_residue_ids:
                continue

            ordered_residue_ids.append(atom_ref.residue_id)
            seen_residue_ids.add(atom_ref.residue_id)

        if len(ordered_residue_ids) != 1:
            return None

        return ordered_residue_ids[0]

    def resolve_atom_input(
        self,
        snapshot: ProteinStructureSnapshot,
    ) -> AtomInput:
        """Project the directive onto one concrete snapshot as an atom domain."""

        return self.resolved_execution_scope_spec().lower_to_atom_input(snapshot)

    def bind_execution(
        self,
        snapshot: ProteinStructureSnapshot,
        *,
        component_library: ComponentLibrary | None = None,
        allow_retained_non_polymer_rdkit_fallback: bool = True,
        retained_non_polymer_chemistry_evidence: tuple[
            RetainedNonPolymerChemistryEvidence,
            ...,
        ] = (),
    ) -> BoundRepairLocalRefinementExecution:
        """Bind one legal current-state execution from this directive."""

        atom_input = self.resolve_atom_input(snapshot)
        active_component_library = (
            build_default_component_library()
            if component_library is None
            else component_library
        )
        atom_scope = atom_input.observed_atom_scope(snapshot)
        atom_scope_facts = derive_atom_scope_continuous_relaxation_facts(
            snapshot,
            atom_scope,
            component_library=active_component_library,
            context_radius_angstrom=self.config.context_radius_angstrom,
            allow_retained_non_polymer_rdkit_fallback=(
                allow_retained_non_polymer_rdkit_fallback
            ),
            retained_non_polymer_chemistry_evidence=(
                retained_non_polymer_chemistry_evidence
            ),
        )

        require_atom_scope_continuous_relaxation_execution(atom_scope_facts)
        binding_decision = decide_continuous_relaxation_binding(
            self.binding,
            self.config,
            atom_scope_facts=atom_scope_facts,
            atom_input_basis=atom_input.basis,
        )
        return BoundRepairLocalRefinementExecution(
            atom_input=atom_input,
            atom_scope_facts=atom_scope_facts,
            binding_decision=binding_decision,
        )


def _local_scope_spec_from_atom_input(
    atom_input: AtomInput,
) -> LocalScopeSpec:
    """Return the semantic local scope spec represented by one atom input."""

    selected_scope = atom_input.as_scope()
    if isinstance(selected_scope, ResidueSetScope):
        if atom_input.realizes_residue_sidechains():
            return LocalScopeSpec.from_residue_sidechains(selected_scope.residue_ids)

        return LocalScopeSpec.from_residues(selected_scope.residue_ids)

    if isinstance(selected_scope, AtomSetScope):
        return LocalScopeSpec.from_atoms(selected_scope.atom_refs)

    raise TypeError("repair local refinement atom input requires a local scope")

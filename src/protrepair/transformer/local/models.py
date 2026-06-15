"""Boundary request models for selected-region transformations."""

from collections.abc import Iterable
from dataclasses import dataclass, field
from enum import Enum
from typing import TYPE_CHECKING, TypeAlias

from protrepair.relation import (
    ExternalCorrespondenceEvidence,
    SupportingStructureAuthorityGrantSpec,
    SupportingStructureCorrespondenceSpec,
)
from protrepair.relation.evidence import (
    AnchorDistanceConstraint,
    DensityEvidence,
    SymmetryContactEvidence,
)
from protrepair.relation.supporting_role import SupportingStructureRole
from protrepair.scope import (
    AbsentResidueSpanScope,
    AnchorAtomPairScope,
    AtomSetScope,
    CompositeScope,
    ResidueSetScope,
    Scope,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationConfig,
    ContinuousRelaxationForceField,
)

if TYPE_CHECKING:
    from protrepair.chemistry import ComponentLibrary
    from protrepair.structure.snapshot import ProteinStructureSnapshot
    from protrepair.transformer.atom_input import AtomInput


class LocalScopeLowering(str, Enum):
    """Closed lowering semantics for local refinement over canonical scope."""

    RESIDUE_ATOMS = "residue_atoms"
    RESIDUE_SIDECHAIN_ATOMS = "residue_sidechain_atoms"
    EXACT_ATOMS = "exact_atoms"
    ATTACHED_PRESENT_HYDROGENS = "attached_present_hydrogens"


@dataclass(frozen=True, slots=True)
class LocalScopeSpec:
    """Canonical local scope request built from semantic scope plus lowering."""

    scope: ResidueSetScope | AtomSetScope
    lowering: LocalScopeLowering

    def __post_init__(self) -> None:
        if not isinstance(self.scope, (ResidueSetScope, AtomSetScope)):
            raise TypeError("local scope specs require a residue-set or atom-set scope")
        if not isinstance(self.lowering, LocalScopeLowering):
            raise TypeError("local scope specs require a LocalScopeLowering value")

        if self.lowering in {
            LocalScopeLowering.RESIDUE_ATOMS,
            LocalScopeLowering.RESIDUE_SIDECHAIN_ATOMS,
        }:
            if not isinstance(self.scope, ResidueSetScope):
                raise TypeError("residue-atom lowering requires a ResidueSetScope")
            return

        if not isinstance(self.scope, AtomSetScope):
            raise TypeError("atom-local lowerings require an AtomSetScope")

    @classmethod
    def from_residues(
        cls,
        residue_ids: Iterable[ResidueId],
    ) -> "LocalScopeSpec":
        """Construct one whole-residue local scope request from residue ids."""

        return cls(
            scope=ResidueSetScope(residue_ids=tuple(residue_ids)),
            lowering=LocalScopeLowering.RESIDUE_ATOMS,
        )

    @classmethod
    def from_residue_sidechains(
        cls,
        residue_ids: Iterable[ResidueId],
    ) -> "LocalScopeSpec":
        """Construct one sidechain-local scope request from residue ids."""

        return cls(
            scope=ResidueSetScope(residue_ids=tuple(residue_ids)),
            lowering=LocalScopeLowering.RESIDUE_SIDECHAIN_ATOMS,
        )

    @classmethod
    def from_atoms(
        cls,
        atom_refs: Iterable[AtomRef],
    ) -> "LocalScopeSpec":
        """Construct one exact atomwise local scope request from atom refs."""

        return cls(
            scope=AtomSetScope(atom_refs=tuple(atom_refs)),
            lowering=LocalScopeLowering.EXACT_ATOMS,
        )

    @classmethod
    def from_atoms_with_attached_hydrogens(
        cls,
        atom_refs: Iterable[AtomRef],
    ) -> "LocalScopeSpec":
        """Construct one atom-focus request closed over attached hydrogens."""

        return cls(
            scope=AtomSetScope(atom_refs=tuple(atom_refs)),
            lowering=LocalScopeLowering.ATTACHED_PRESENT_HYDROGENS,
        )

    def is_residuewise(self) -> bool:
        """Return whether this scope request lowers residuewise."""

        return self.lowering in {
            LocalScopeLowering.RESIDUE_ATOMS,
            LocalScopeLowering.RESIDUE_SIDECHAIN_ATOMS,
        }

    def is_atomwise(self) -> bool:
        """Return whether this scope request lowers atomwise."""

        return not self.is_residuewise()

    def includes_attached_hydrogens(self) -> bool:
        """Return whether lowering closes over attached present hydrogens."""

        return self.lowering is LocalScopeLowering.ATTACHED_PRESENT_HYDROGENS

    def lowers_to_residue_sidechains(self) -> bool:
        """Return whether residuewise lowering targets side-chain atoms only."""

        return self.lowering is LocalScopeLowering.RESIDUE_SIDECHAIN_ATOMS

    def referenced_residue_ids(self) -> tuple[ResidueId, ...]:
        """Return referenced residue ids in first-seen semantic order."""

        if isinstance(self.scope, ResidueSetScope):
            return self.scope.residue_ids

        return _referenced_residue_ids_from_atom_refs(self.scope.atom_refs)

    def as_scope(self) -> ResidueSetScope | AtomSetScope:
        """Return the semantic scope carried by this local scope request."""

        return self.scope

    def lower_to_atom_input(
        self,
        snapshot: "ProteinStructureSnapshot",
        *,
        component_library: "ComponentLibrary | None" = None,
    ) -> "AtomInput":
        """Lower this local scope request into one atom-input domain."""

        from protrepair.transformer.local.lowering import (
            atom_input_from_local_scope_spec,
        )

        return atom_input_from_local_scope_spec(
            snapshot,
            self,
            component_library=component_library,
        )


def _referenced_residue_ids_from_atom_refs(
    atom_refs: tuple[AtomRef, ...],
) -> tuple[ResidueId, ...]:
    """Return referenced residue ids in first-seen atom order."""

    ordered_residue_ids: list[ResidueId] = []
    seen_residue_ids: set[ResidueId] = set()
    for atom_ref in atom_refs:
        if atom_ref.residue_id not in seen_residue_ids:
            ordered_residue_ids.append(atom_ref.residue_id)
            seen_residue_ids.add(atom_ref.residue_id)

    return tuple(ordered_residue_ids)


LocalScope: TypeAlias = (
    ResidueSetScope
    | AtomSetScope
    | AbsentResidueSpanScope
    | AnchorAtomPairScope
    | CompositeScope
)


def _require_local_scope(
    scope: Scope,
    *,
    field_name: str,
) -> LocalScope:
    """Return one validated local semantic scope."""

    if isinstance(scope, CompositeScope):
        for child_scope in scope.scopes:
            _require_local_scope(child_scope, field_name=field_name)
        return scope

    if isinstance(
        scope,
        (
            ResidueSetScope,
            AtomSetScope,
            AbsentResidueSpanScope,
            AnchorAtomPairScope,
        ),
    ):
        return scope

    raise TypeError(
        f"{field_name} must be a residue-set, atom-set, absent-span, "
        "anchor-pair, or composite scope"
    )


@dataclass(frozen=True, slots=True)
class SupportingStructureSpec:
    """Boundary supporting-structure input for contextual local transforms."""

    role: SupportingStructureRole
    structure: ProteinStructure
    scope: LocalScope | None = None
    correspondences: tuple[SupportingStructureCorrespondenceSpec, ...] = ()
    authority_grants: tuple[SupportingStructureAuthorityGrantSpec, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.role, SupportingStructureRole):
            raise TypeError(
                "supporting structure specs require a SupportingStructureRole value"
            )
        if not isinstance(self.structure, ProteinStructure):
            raise TypeError(
                "supporting structure specs require a ProteinStructure value"
            )
        if self.scope is not None:
            object.__setattr__(
                self,
                "scope",
                _require_local_scope(self.scope, field_name="scope"),
            )

        correspondences = tuple(dict.fromkeys(self.correspondences))
        for correspondence in correspondences:
            if not isinstance(
                correspondence,
                SupportingStructureCorrespondenceSpec,
            ):
                raise TypeError(
                    "supporting structure specs require "
                    "SupportingStructureCorrespondenceSpec values"
                )

        authority_grants = tuple(self.authority_grants)
        for authority_grant in authority_grants:
            if not isinstance(
                authority_grant,
                SupportingStructureAuthorityGrantSpec,
            ):
                raise TypeError(
                    "supporting structure specs require "
                    "SupportingStructureAuthorityGrantSpec values"
                )
            if authority_grant.correspondence not in correspondences:
                correspondences = (
                    *correspondences,
                    authority_grant.correspondence,
                )

        object.__setattr__(self, "correspondences", correspondences)
        object.__setattr__(self, "authority_grants", authority_grants)


@dataclass(frozen=True, slots=True)
class LocalTransformationContextSpec:
    """Boundary context for transforms that depend on supporting structures."""

    supporting_structures: tuple[SupportingStructureSpec, ...] = ()
    external_evidence: tuple[
        DensityEvidence | SymmetryContactEvidence | ExternalCorrespondenceEvidence,
        ...,
    ] = ()
    external_constraints: tuple[AnchorDistanceConstraint, ...] = ()

    def __post_init__(self) -> None:
        supporting_structures = tuple(self.supporting_structures)
        for supporting_structure in supporting_structures:
            if not isinstance(supporting_structure, SupportingStructureSpec):
                raise TypeError(
                    "local transformation context requires SupportingStructureSpec "
                    "values"
                )

        external_evidence = tuple(dict.fromkeys(self.external_evidence))
        for evidence in external_evidence:
            if not isinstance(
                evidence,
                (
                    DensityEvidence,
                    SymmetryContactEvidence,
                    ExternalCorrespondenceEvidence,
                ),
            ):
                raise TypeError(
                    "local transformation context requires canonical external "
                    "evidence values"
                )

        external_constraints = tuple(dict.fromkeys(self.external_constraints))
        for constraint in external_constraints:
            if not isinstance(constraint, AnchorDistanceConstraint):
                raise TypeError(
                    "local transformation context requires canonical external "
                    "constraint values"
                )

        object.__setattr__(self, "supporting_structures", supporting_structures)
        object.__setattr__(self, "external_evidence", external_evidence)
        object.__setattr__(self, "external_constraints", external_constraints)

    def is_source_only(self) -> bool:
        """Return whether the context contains no external transform inputs."""

        return (
            not self.supporting_structures
            and not self.external_evidence
            and not self.external_constraints
        )


@dataclass(frozen=True, slots=True)
class DirectRegionTransformationSpec:
    """Public direct-API request for one selected-region transformation.

    Direct region transformation requires an explicit force-field choice.
    Hydrogen-less or topology-invalid current domains are still rejected before
    execution, regardless of the requested force field. Additional supporting
    structures are modeled on the boundary request, but current direct execution
    remains limited to source-only contexts.
    """

    scope_spec: LocalScopeSpec
    force_field: ContinuousRelaxationForceField
    config: ContinuousRelaxationConfig = field(
        default_factory=ContinuousRelaxationConfig
    )
    context: LocalTransformationContextSpec = field(
        default_factory=LocalTransformationContextSpec
    )

    def __post_init__(self) -> None:
        if not isinstance(self.scope_spec, LocalScopeSpec):
            raise TypeError(
                "direct region transformation requires a LocalScopeSpec scope_spec"
            )
        if not isinstance(self.force_field, ContinuousRelaxationForceField):
            raise TypeError(
                "direct region transformation requires a "
                "ContinuousRelaxationForceField value"
            )
        if not isinstance(self.config, ContinuousRelaxationConfig):
            raise TypeError(
                "direct region transformation requires "
                "ContinuousRelaxationConfig config"
            )
        if not isinstance(self.context, LocalTransformationContextSpec):
            raise TypeError(
                "direct region transformation requires "
                "LocalTransformationContextSpec context"
            )

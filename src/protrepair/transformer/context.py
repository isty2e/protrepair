"""Canonical transformation-context models over source and supporting structures."""

from dataclasses import dataclass

from protrepair.relation import (
    ExternalCorrespondenceEvidence,
    SupportingStructureAuthorityAspect,
    SupportingStructureAuthorityGrant,
    SupportingStructureCorrespondence,
)
from protrepair.relation import evidence as relation_evidence
from protrepair.relation.supporting_role import SupportingStructureRole
from protrepair.scope import (
    AbsentResidueSpanScope,
    AnchorAtomPairScope,
    AtomSetScope,
    CompositeScope,
    ResidueSetScope,
    Scope,
    scope_refines,
)
from protrepair.structure.endpoint import StructureEndpoint
from protrepair.structure.provenance import StructureProvenanceOrigin
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.atom_input import AtomInput


def _structure_endpoint_is_grounded_in_snapshot(
    structure_endpoint: StructureEndpoint,
    snapshot: ProteinStructureSnapshot,
) -> bool:
    """Return whether one structure endpoint is grounded in one snapshot."""

    scope = structure_endpoint.scope
    if isinstance(scope, ResidueSetScope):
        return all(
            snapshot.structure.constitution.residue_or_ligand(residue_id) is not None
            for residue_id in scope.residue_ids
        )
    if isinstance(scope, AtomSetScope):
        return all(
            snapshot.structure.constitution.resolve_atom_site(atom_ref) is not None
            for atom_ref in scope.atom_refs
        )
    if isinstance(scope, AbsentResidueSpanScope):
        return (
            (
                scope.preceding_residue_id is None
                or snapshot.structure.constitution.residue_or_ligand(
                    scope.preceding_residue_id
                )
                is not None
            )
            and (
                scope.following_residue_id is None
                or snapshot.structure.constitution.residue_or_ligand(
                    scope.following_residue_id
                )
                is not None
            )
            and all(
                snapshot.structure.constitution.residue_or_ligand(residue_id) is None
                for residue_id in scope.absent_residue_ids
            )
        )
    if isinstance(scope, AnchorAtomPairScope):
        return (
            snapshot.structure.constitution.resolve_atom_site(
                scope.left_anchor_atom_ref
            )
            is not None
            and snapshot.structure.constitution.resolve_atom_site(
                scope.right_anchor_atom_ref
            )
            is not None
        )
    if isinstance(scope, CompositeScope):
        return all(
            _structure_endpoint_is_grounded_in_snapshot(
                StructureEndpoint(
                    carrier_handle=structure_endpoint.carrier_handle,
                    scope=member_scope,
                    realization_selector=None,
                ),
                snapshot,
            )
            for member_scope in scope.scopes
        )

    raise TypeError("unsupported scope for structure-endpoint grounding")


def _require_grounded_structure_endpoint(
    structure_endpoint: StructureEndpoint,
    snapshot: ProteinStructureSnapshot,
    *,
    field_name: str,
    snapshot_name: str,
) -> None:
    """Raise when one structure endpoint does not ground in one snapshot."""

    if _structure_endpoint_is_grounded_in_snapshot(structure_endpoint, snapshot):
        return

    raise ValueError(
        f"{field_name} must ground in the active {snapshot_name} snapshot"
    )


def _context_endpoint_covers(
    covering_endpoint: StructureEndpoint,
    covered_endpoint: StructureEndpoint,
) -> bool:
    """Return whether one context-local endpoint covers another endpoint."""

    if (
        covering_endpoint.carrier_handle.kind
        is not covered_endpoint.carrier_handle.kind
    ):
        return False
    if (
        covering_endpoint.realization_selector is not None
        and covered_endpoint.realization_selector
        != covering_endpoint.realization_selector
    ):
        return False

    return scope_refines(covered_endpoint.scope, covering_endpoint.scope)


def _covering_correspondences(
    supporting_structure: "SupportingStructureContext",
    *,
    source_structure_endpoint: StructureEndpoint,
    supporting_structure_endpoint: StructureEndpoint,
) -> tuple[SupportingStructureCorrespondence, ...]:
    """Return correspondences that cover one source/supporting reference pair."""

    return tuple(
        correspondence
        for correspondence in supporting_structure.correspondences
        if _context_endpoint_covers(
            correspondence.source_structure_endpoint,
            source_structure_endpoint,
        )
        and _context_endpoint_covers(
            correspondence.supporting_structure_endpoint,
            supporting_structure_endpoint,
        )
    )


def _has_authority_for_reference_pair(
    supporting_structure: "SupportingStructureContext",
    *,
    aspect: SupportingStructureAuthorityAspect,
    source_structure_endpoint: StructureEndpoint,
    supporting_structure_endpoint: StructureEndpoint,
) -> bool:
    """Return whether one support structure authorizes one reference pair."""

    return any(
        authority_grant.grants(aspect)
        and _context_endpoint_covers(
            authority_grant.correspondence.source_structure_endpoint,
            source_structure_endpoint,
        )
        and _context_endpoint_covers(
            authority_grant.correspondence.supporting_structure_endpoint,
            supporting_structure_endpoint,
        )
        for authority_grant in supporting_structure.authority_grants
    )


@dataclass(frozen=True, slots=True)
class SupportingStructureContext:
    """One additional structure plus its semantic role and optional focus."""

    role: SupportingStructureRole
    snapshot: ProteinStructureSnapshot
    atom_input: AtomInput | None = None
    correspondences: tuple[
        SupportingStructureCorrespondence,
        ...,
    ] = ()
    authority_grants: tuple[
        SupportingStructureAuthorityGrant,
        ...,
    ] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.role, SupportingStructureRole):
            raise TypeError(
                "supporting structure contexts require a SupportingStructureRole value"
            )
        if not isinstance(self.snapshot, ProteinStructureSnapshot):
            raise TypeError(
                "supporting structure contexts require a ProteinStructureSnapshot"
            )
        if self.atom_input is not None and not isinstance(
            self.atom_input,
            AtomInput,
        ):
            raise TypeError(
                "supporting structure contexts require atom_input to be an "
                "AtomInput or None"
            )

        correspondences = tuple(dict.fromkeys(self.correspondences))
        for correspondence in correspondences:
            if not isinstance(
                correspondence,
                SupportingStructureCorrespondence,
            ):
                raise TypeError(
                    "supporting structure contexts require "
                    "SupportingStructureCorrespondence values"
                )
            _require_grounded_structure_endpoint(
                correspondence.supporting_structure_endpoint,
                self.snapshot,
                field_name="supporting_structure_endpoint",
                snapshot_name="supporting",
            )
            for mapping in correspondence.mappings:
                _require_grounded_structure_endpoint(
                    mapping.supporting_structure_endpoint,
                    self.snapshot,
                    field_name="supporting_structure_endpoint",
                    snapshot_name="supporting",
                )

        authority_grants = tuple(self.authority_grants)
        for authority_grant in authority_grants:
            if not isinstance(
                authority_grant,
                SupportingStructureAuthorityGrant,
            ):
                raise TypeError(
                    "supporting structure contexts require "
                    "SupportingStructureAuthorityGrant values"
                )
            if authority_grant.correspondence not in correspondences:
                correspondences = (
                    *correspondences,
                    authority_grant.correspondence,
                )

        object.__setattr__(self, "correspondences", correspondences)
        object.__setattr__(self, "authority_grants", authority_grants)

    def authoritative_grants(
        self,
        aspect: SupportingStructureAuthorityAspect,
    ) -> tuple[SupportingStructureAuthorityGrant, ...]:
        """Return authority grants that authorize one specific aspect."""

        return tuple(
            authority_grant
            for authority_grant in self.authority_grants
            if authority_grant.grants(aspect)
        )

    def has_authority_for(
        self,
        aspect: SupportingStructureAuthorityAspect,
    ) -> bool:
        """Return whether this support structure is authoritative for one aspect."""

        return bool(self.authoritative_grants(aspect))


@dataclass(frozen=True, slots=True)
class ProteinTransformationContext:
    """Canonical source-plus-supporting-structures transformer input."""

    source_snapshot: ProteinStructureSnapshot
    atom_input: AtomInput
    supporting_structures: tuple[SupportingStructureContext, ...] = ()
    external_evidence: tuple[
        relation_evidence.TransformationExternalEvidence,
        ...,
    ] = ()
    external_constraints: tuple[
        relation_evidence.TransformationExternalConstraint,
        ...,
    ] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source_snapshot, ProteinStructureSnapshot):
            raise TypeError(
                "protein transformation contexts require a ProteinStructureSnapshot"
            )
        if not isinstance(self.atom_input, AtomInput):
            raise TypeError(
                "protein transformation contexts require an AtomInput"
            )
        supporting_structures = tuple(self.supporting_structures)
        for supporting_structure in supporting_structures:
            if not isinstance(supporting_structure, SupportingStructureContext):
                raise TypeError(
                    "protein transformation contexts require "
                    "SupportingStructureContext values for "
                    "supporting_structures"
                )
            for correspondence in supporting_structure.correspondences:
                _require_grounded_structure_endpoint(
                    correspondence.source_structure_endpoint,
                    self.source_snapshot,
                    field_name="source_structure_endpoint",
                    snapshot_name="source",
                )
                for mapping in correspondence.mappings:
                    _require_grounded_structure_endpoint(
                        mapping.source_structure_endpoint,
                        self.source_snapshot,
                        field_name="source_structure_endpoint",
                        snapshot_name="source",
                    )

        external_evidence = tuple(dict.fromkeys(self.external_evidence))
        for evidence in external_evidence:
            if not isinstance(
                evidence,
                (
                    relation_evidence.DensityEvidence,
                    relation_evidence.SymmetryContactEvidence,
                    ExternalCorrespondenceEvidence,
                ),
            ):
                raise TypeError(
                    "protein transformation contexts require canonical "
                    "external evidence values"
                )
            if isinstance(
                evidence,
                (
                    relation_evidence.DensityEvidence,
                    relation_evidence.SymmetryContactEvidence,
                ),
            ):
                _require_grounded_structure_endpoint(
                    evidence.target_structure_endpoint,
                    self.source_snapshot,
                    field_name=(
                        f"{type(evidence).__name__} "
                        "target_structure_endpoint"
                    ),
                    snapshot_name="source",
                )
            else:
                self._validate_external_correspondence_evidence(
                    evidence,
                    supporting_structures,
                )

        external_constraints = tuple(dict.fromkeys(self.external_constraints))
        for constraint in external_constraints:
            if not isinstance(
                constraint,
                relation_evidence.AnchorDistanceConstraint,
            ):
                raise TypeError(
                    "protein transformation contexts require canonical "
                    "external constraint values"
                )
            self._validate_external_constraint(
                constraint,
                supporting_structures,
            )

        object.__setattr__(self, "supporting_structures", supporting_structures)
        object.__setattr__(self, "external_evidence", external_evidence)
        object.__setattr__(self, "external_constraints", external_constraints)

    def source_scope(self) -> Scope:
        """Return the semantic scope covered by the active source domain."""

        return self.atom_input.as_scope()

    @classmethod
    def from_snapshot_atom_input(
        cls,
        snapshot: ProteinStructureSnapshot,
        atom_input: AtomInput,
    ) -> "ProteinTransformationContext":
        """Build one source-only transformation context from a snapshot/domain pair."""

        return cls(
            source_snapshot=snapshot,
            atom_input=atom_input,
        )

    def is_source_only(self) -> bool:
        """Return whether this context contains no external transform inputs."""

        return (
            not self.supporting_structures
            and not self.external_evidence
            and not self.external_constraints
        )

    def supporting_structures_of_role(
        self,
        role: SupportingStructureRole,
    ) -> tuple[SupportingStructureContext, ...]:
        """Return supporting structures with one specific semantic role."""

        return tuple(
            supporting_structure
            for supporting_structure in self.supporting_structures
            if supporting_structure.role is role
        )

    def supporting_structures_authoritative_for(
        self,
        aspect: SupportingStructureAuthorityAspect,
    ) -> tuple[SupportingStructureContext, ...]:
        """Return supporting structures that are authoritative for one aspect."""

        return tuple(
            supporting_structure
            for supporting_structure in self.supporting_structures
            if supporting_structure.has_authority_for(aspect)
        )

    def _resolve_supporting_structure_for_origin(
        self,
        origin: StructureProvenanceOrigin,
    ) -> SupportingStructureContext:
        """Return the active supporting structure referenced by one origin."""

        supporting_structures = self.supporting_structures
        if not supporting_structures:
            raise ValueError(
                "supporting provenance origins require at least one active "
                "supporting structure"
            )

        carrier_token = origin.structure_endpoint.carrier_handle.token
        if carrier_token is None:
            if len(supporting_structures) != 1:
                raise ValueError(
                    "supporting provenance origins require a unique carrier token "
                    "when multiple active supporting structures exist"
                )

            return supporting_structures[0]

        if carrier_token.startswith("supporting-"):
            suffix = carrier_token.removeprefix("supporting-")
            if suffix.isdigit():
                supporting_index = int(suffix)
                if supporting_index < len(supporting_structures):
                    return supporting_structures[supporting_index]
                raise ValueError(
                    "supporting provenance origin carrier token must resolve "
                    "inside the active supporting structures"
                )

        matching_supporting_structures = tuple(
            supporting_structure
            for supporting_structure in supporting_structures
            if (
                supporting_structure.snapshot.structure.provenance.ingress.source_name
                == carrier_token
            )
        )
        if len(matching_supporting_structures) == 1:
            return matching_supporting_structures[0]
        if not matching_supporting_structures:
            raise ValueError(
                "supporting provenance origin carrier token must match one active "
                "supporting structure"
            )
        raise ValueError(
            "supporting provenance origin carrier token must resolve uniquely "
            "among active supporting structures"
        )

    def _validate_origin_grounding(
        self,
        origin: StructureProvenanceOrigin,
        *,
        field_name: str,
    ) -> SupportingStructureContext | None:
        """Validate one provenance origin against the active context."""

        if origin.is_source():
            _require_grounded_structure_endpoint(
                origin.structure_endpoint,
                self.source_snapshot,
                field_name=field_name,
                snapshot_name="source",
            )
            return None

        supporting_structure = self._resolve_supporting_structure_for_origin(origin)
        _require_grounded_structure_endpoint(
            origin.structure_endpoint,
            supporting_structure.snapshot,
            field_name=field_name,
            snapshot_name="supporting",
        )
        return supporting_structure

    def _validate_external_correspondence_evidence(
        self,
        evidence: ExternalCorrespondenceEvidence,
        supporting_structures: tuple[SupportingStructureContext, ...],
    ) -> None:
        """Validate one external correspondence evidence value."""

        del supporting_structures
        if evidence.source_origin.is_supporting():
            raise ValueError(
                "external correspondence evidence source_origin must point into "
                "the active source snapshot"
            )

        self._validate_origin_grounding(
            evidence.source_origin,
            field_name="external correspondence evidence source_origin",
        )
        supporting_structure = self._validate_origin_grounding(
            evidence.counterpart_origin,
            field_name="external correspondence evidence counterpart_origin",
        )
        if supporting_structure is None:
            return

        if _covering_correspondences(
            supporting_structure,
            source_structure_endpoint=evidence.source_origin.structure_endpoint,
            supporting_structure_endpoint=evidence.counterpart_origin.structure_endpoint,
        ):
            return

        raise ValueError(
            "external correspondence evidence counterpart_origin must be covered "
            "by a declared supporting correspondence"
        )

    def _validate_external_constraint(
        self,
        constraint: relation_evidence.AnchorDistanceConstraint,
        supporting_structures: tuple[SupportingStructureContext, ...],
    ) -> None:
        """Validate one external anchor-distance constraint value."""

        del supporting_structures
        left_support = self._validate_origin_grounding(
            constraint.left_anchor_origin,
            field_name="anchor-distance constraint left_anchor_origin",
        )
        right_support = self._validate_origin_grounding(
            constraint.right_anchor_origin,
            field_name="anchor-distance constraint right_anchor_origin",
        )
        if left_support is None and right_support is None:
            return
        if left_support is not None and right_support is not None:
            raise ValueError(
                "anchor-distance constraints that reference supporting "
                "structures require at least one source-grounded anchor origin"
            )

        source_origin = (
            constraint.right_anchor_origin
            if left_support is not None
            else constraint.left_anchor_origin
        )
        supporting_origin = (
            constraint.left_anchor_origin
            if left_support is not None
            else constraint.right_anchor_origin
        )
        supporting_structure = (
            left_support if left_support is not None else right_support
        )
        assert supporting_structure is not None

        if _has_authority_for_reference_pair(
            supporting_structure,
            aspect=SupportingStructureAuthorityAspect.COORDINATES,
            source_structure_endpoint=source_origin.structure_endpoint,
            supporting_structure_endpoint=supporting_origin.structure_endpoint,
        ):
            return

        raise ValueError(
            "anchor-distance constraints with supporting anchors require a "
            "declared coordinate authority grant"
        )

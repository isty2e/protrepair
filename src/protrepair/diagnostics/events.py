"""Structured events and validation results for the redesigned ProtRepair package."""

from dataclasses import dataclass, field
from enum import Enum

from protrepair.diagnostics.kinds import (
    IssueSeverity,
    RepairEventKind,
    ValidationIssueKind,
)
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import StructureProvenanceOrigin


class EventScopeKind(str, Enum):
    """Closed provenance scope variants for repairs and validation issues."""

    STRUCTURE = "structure"
    RESIDUE = "residue"
    RESIDUE_PAIR = "residue_pair"
    RESIDUE_SPAN = "residue_span"
    RESIDUE_SET = "residue_set"


@dataclass(frozen=True, slots=True)
class EventScope:
    """Canonical repair or issue scope over residues or the whole structure."""

    kind: EventScopeKind
    residue_ids: tuple[ResidueId, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, EventScopeKind):
            raise TypeError("event scope kind must be an EventScopeKind value")

        raw_residue_ids = tuple(self.residue_ids)
        for residue_id in raw_residue_ids:
            if not isinstance(residue_id, ResidueId):
                raise TypeError(
                    "event scope residue_ids must contain ResidueId values"
                )

        residue_ids = tuple(dict.fromkeys(raw_residue_ids))
        if self.kind is EventScopeKind.STRUCTURE:
            if residue_ids:
                raise ValueError(
                    "structure event scope must not carry residue ids"
                )
        elif self.kind is EventScopeKind.RESIDUE:
            if len(residue_ids) != 1:
                raise ValueError(
                    "residue event scope requires exactly one residue id"
                )
        elif self.kind is EventScopeKind.RESIDUE_PAIR:
            if len(residue_ids) != 2:
                raise ValueError(
                    "residue-pair event scope requires exactly two residue ids"
                )
        elif self.kind is EventScopeKind.RESIDUE_SPAN:
            if len(residue_ids) < 2:
                raise ValueError(
                    "residue-span event scope requires at least two residue ids"
                )
        elif self.kind is EventScopeKind.RESIDUE_SET:
            if not residue_ids:
                raise ValueError(
                    "residue-set event scope requires at least one residue id"
                )

        object.__setattr__(self, "residue_ids", residue_ids)

    @classmethod
    def for_structure(cls) -> "EventScope":
        """Return one structure-wide provenance scope."""

        return cls(EventScopeKind.STRUCTURE)

    @classmethod
    def for_residue(cls, residue_id: ResidueId) -> "EventScope":
        """Return one single-residue provenance scope."""

        return cls(EventScopeKind.RESIDUE, (residue_id,))

    @classmethod
    def for_residue_pair(
        cls,
        left_residue_id: ResidueId,
        right_residue_id: ResidueId,
    ) -> "EventScope":
        """Return one residue-pair provenance scope."""

        return cls(
            EventScopeKind.RESIDUE_PAIR,
            (left_residue_id, right_residue_id),
        )

    @classmethod
    def for_residue_span(
        cls,
        residue_ids: tuple[ResidueId, ...],
    ) -> "EventScope":
        """Return one residue-span provenance scope."""

        return cls(EventScopeKind.RESIDUE_SPAN, residue_ids)

    @classmethod
    def for_residue_set(
        cls,
        residue_ids: tuple[ResidueId, ...],
    ) -> "EventScope":
        """Return one unordered residue-set provenance scope."""

        return cls(EventScopeKind.RESIDUE_SET, residue_ids)

    def single_residue_id(self) -> ResidueId | None:
        """Return the sole residue id when this scope is residue-local."""

        if self.kind is not EventScopeKind.RESIDUE:
            return None

        return self.residue_ids[0]

    def targets_residue(self, residue_id: ResidueId) -> bool:
        """Return whether one residue falls inside this provenance scope."""

        return residue_id in self.residue_ids


@dataclass(frozen=True, slots=True)
class ResidueAtomImpact:
    """One residue-local atom impact embedded inside a broader event scope."""

    residue_id: ResidueId
    component_id: str | None = None
    atom_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.residue_id, ResidueId):
            raise TypeError("residue atom impacts require a ResidueId")

        component_id = self.component_id
        if component_id is not None:
            if not isinstance(component_id, str):
                raise TypeError("residue atom impact component_id must be a string")
            component_id = component_id.strip() or None

        atom_names: list[str] = []
        for atom_name in self.atom_names:
            if not isinstance(atom_name, str):
                raise TypeError("residue atom impact atom_names must contain strings")
            atom_names.append(atom_name.strip().upper())

        object.__setattr__(self, "atom_names", tuple(atom_names))
        object.__setattr__(self, "component_id", component_id)


@dataclass(frozen=True, slots=True)
class RepairEvent:
    """Structured record of a successful repair or normalization event."""

    kind: RepairEventKind
    scope: EventScope
    residue_impacts: tuple[ResidueAtomImpact, ...] = ()
    provenance_origins: tuple[StructureProvenanceOrigin, ...] = ()
    details: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.kind, RepairEventKind):
            raise TypeError("repair events require a RepairEventKind value")
        if not isinstance(self.scope, EventScope):
            raise TypeError("repair events require an EventScope value")

        residue_impacts = tuple(self.residue_impacts)
        for residue_impact in residue_impacts:
            if not isinstance(residue_impact, ResidueAtomImpact):
                raise TypeError(
                    "repair events require ResidueAtomImpact values"
                )

        provenance_origins = tuple(dict.fromkeys(self.provenance_origins))
        if self.scope.kind is not EventScopeKind.STRUCTURE:
            scope_residue_ids = set(self.scope.residue_ids)
            for residue_impact in residue_impacts:
                if residue_impact.residue_id not in scope_residue_ids:
                    raise ValueError(
                        "repair event residue impacts must fall inside the event "
                        "scope"
                    )
        for provenance_origin in provenance_origins:
            if not isinstance(provenance_origin, StructureProvenanceOrigin):
                raise TypeError(
                    "repair event provenance origins must be "
                    "StructureProvenanceOrigin values"
                )

        object.__setattr__(self, "residue_impacts", residue_impacts)
        object.__setattr__(self, "provenance_origins", provenance_origins)

    @classmethod
    def for_residue(
        cls,
        *,
        kind: RepairEventKind,
        residue_id: ResidueId,
        component_id: str,
        atom_names: tuple[str, ...],
        provenance_origins: tuple[StructureProvenanceOrigin, ...] = (),
        details: str | None = None,
    ) -> "RepairEvent":
        """Return one canonical single-residue repair event."""

        return cls(
            kind=kind,
            scope=EventScope.for_residue(residue_id),
            residue_impacts=(
                ()
                if component_id is None and not atom_names
                else (
                    ResidueAtomImpact(
                        residue_id=residue_id,
                        component_id=component_id,
                        atom_names=atom_names,
                    ),
                )
            ),
            provenance_origins=provenance_origins,
            details=details,
        )

    @classmethod
    def for_residue_span(
        cls,
        *,
        kind: RepairEventKind,
        residue_ids: tuple[ResidueId, ...],
        residue_impacts: tuple[ResidueAtomImpact, ...],
        provenance_origins: tuple[StructureProvenanceOrigin, ...] = (),
        details: str | None = None,
    ) -> "RepairEvent":
        """Return one canonical residue-span repair event."""

        return cls(
            kind=kind,
            scope=EventScope.for_residue_span(residue_ids),
            residue_impacts=residue_impacts,
            provenance_origins=provenance_origins,
            details=details,
        )

    @property
    def residue_id(self) -> ResidueId | None:
        """Return the sole residue id when this event is residue-local."""

        return self.scope.single_residue_id()

    @property
    def component_id(self) -> str | None:
        """Return the single impacted component id when this event is residue-local."""

        if len(self.residue_impacts) != 1:
            return None

        return self.residue_impacts[0].component_id

    @property
    def atom_names(self) -> tuple[str, ...]:
        """Return impacted atom names when this event is residue-local."""

        if len(self.residue_impacts) != 1:
            return ()

        return self.residue_impacts[0].atom_names

    def affects_atom(self, atom_name: str) -> bool:
        """Return whether the event references a specific atom."""

        normalized_atom_name = atom_name.strip().upper()
        return any(
            normalized_atom_name in residue_impact.atom_names
            for residue_impact in self.residue_impacts
        )

    def supporting_provenance_origins(self) -> tuple[StructureProvenanceOrigin, ...]:
        """Return provenance origins that point into supporting structures."""

        return tuple(
            provenance_origin
            for provenance_origin in self.provenance_origins
            if provenance_origin.is_supporting()
        )


@dataclass(frozen=True, slots=True)
class ValidationIssue:
    """Structured validation issue attached to a residue or structure."""

    kind: ValidationIssueKind
    severity: IssueSeverity
    message: str
    scope: EventScope = field(default_factory=EventScope.for_structure)
    provenance_origins: tuple[StructureProvenanceOrigin, ...] = ()
    residue_impacts: tuple[ResidueAtomImpact, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.kind, ValidationIssueKind):
            raise TypeError(
                "validation issues require a ValidationIssueKind value"
            )
        if not isinstance(self.severity, IssueSeverity):
            raise TypeError("validation issues require an IssueSeverity value")
        if not isinstance(self.scope, EventScope):
            raise TypeError("validation issues require an EventScope value")
        residue_impacts = tuple(self.residue_impacts)
        for residue_impact in residue_impacts:
            if not isinstance(residue_impact, ResidueAtomImpact):
                raise TypeError(
                    "validation issues require ResidueAtomImpact values"
                )
        if self.scope.kind is not EventScopeKind.STRUCTURE:
            scope_residue_ids = set(self.scope.residue_ids)
            for residue_impact in residue_impacts:
                if residue_impact.residue_id not in scope_residue_ids:
                    raise ValueError(
                        "validation issue residue impacts must fall inside the "
                        "event scope"
                    )
        provenance_origins = tuple(dict.fromkeys(self.provenance_origins))
        for provenance_origin in provenance_origins:
            if not isinstance(provenance_origin, StructureProvenanceOrigin):
                raise TypeError(
                    "validation issue provenance origins must be "
                    "StructureProvenanceOrigin values"
                )
        object.__setattr__(self, "residue_impacts", residue_impacts)
        object.__setattr__(self, "provenance_origins", provenance_origins)

    @classmethod
    def for_residue(
        cls,
        *,
        kind: ValidationIssueKind,
        severity: IssueSeverity,
        message: str,
        residue_id: ResidueId,
        provenance_origins: tuple[StructureProvenanceOrigin, ...] = (),
        component_id: str | None = None,
        atom_names: tuple[str, ...] = (),
    ) -> "ValidationIssue":
        """Return one residue-local validation issue."""

        return cls(
            kind=kind,
            severity=severity,
            message=message,
            scope=EventScope.for_residue(residue_id),
            provenance_origins=provenance_origins,
            residue_impacts=(
                ResidueAtomImpact(
                    residue_id=residue_id,
                    component_id=component_id,
                    atom_names=atom_names,
                ),
            ),
        )

    @property
    def residue_id(self) -> ResidueId | None:
        """Return the sole residue id when this issue is residue-local."""

        return self.scope.single_residue_id()

    @property
    def component_id(self) -> str | None:
        """Return the single impacted component id when residue-local."""

        if len(self.residue_impacts) != 1:
            return None

        return self.residue_impacts[0].component_id

    @property
    def atom_names(self) -> tuple[str, ...]:
        """Return impacted atom names when this issue is residue-local."""

        if len(self.residue_impacts) != 1:
            return ()

        return self.residue_impacts[0].atom_names

    def affects_atom(self, atom_name: str) -> bool:
        """Return whether the issue references a specific atom."""

        normalized_atom_name = atom_name.strip().upper()
        return any(
            normalized_atom_name in residue_impact.atom_names
            for residue_impact in self.residue_impacts
        )

    def is_error(self) -> bool:
        """Return whether the issue is error-severity."""

        return self.severity is IssueSeverity.ERROR

    def is_warning(self) -> bool:
        """Return whether the issue is warning-severity."""

        return self.severity is IssueSeverity.WARNING

    def supporting_provenance_origins(self) -> tuple[StructureProvenanceOrigin, ...]:
        """Return provenance origins that point into supporting structures."""

        return tuple(
            provenance_origin
            for provenance_origin in self.provenance_origins
            if provenance_origin.is_supporting()
        )

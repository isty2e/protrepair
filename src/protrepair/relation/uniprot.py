"""Relation-owned structure-to-UniProt correspondence entities."""

from dataclasses import dataclass
from enum import IntEnum

from protrepair.relation.endpoint import StructureEndpoint
from protrepair.relation.uniprot_reference import UniProtSequenceReference
from protrepair.scope import (
    AbsentResidueSpanScope,
    CompositeScope,
    ResidueSetScope,
)
from protrepair.structure.labels import ResidueId


class UniProtCorrespondenceConfidence(IntEnum):
    """Ordered confidence levels for structure-to-UniProt correspondence."""

    LOW = 1
    MEDIUM = 2
    HIGH = 3
    EXACT = 4


@dataclass(frozen=True, slots=True)
class StructureUniProtCorrespondence:
    """One structure polymer region mapped onto one UniProt reference."""

    structure_endpoint: StructureEndpoint
    uniprot_reference: UniProtSequenceReference
    confidence: UniProtCorrespondenceConfidence
    sequence_identity: float | None = None
    sequence_coverage: float | None = None
    evidence_label: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.structure_endpoint, StructureEndpoint):
            raise TypeError(
                "structure-UniProt correspondence requires a StructureEndpoint"
            )
        _require_polymer_region_endpoint(self.structure_endpoint)
        if not isinstance(self.uniprot_reference, UniProtSequenceReference):
            raise TypeError(
                "structure-UniProt correspondence requires a UniProtSequenceReference"
            )
        if not isinstance(self.confidence, UniProtCorrespondenceConfidence):
            raise TypeError(
                "structure-UniProt correspondence requires a confidence value"
            )

        if self.sequence_identity is not None:
            _require_unit_interval(
                self.sequence_identity,
                field_name="sequence_identity",
            )
        if self.sequence_coverage is not None:
            _require_unit_interval(
                self.sequence_coverage,
                field_name="sequence_coverage",
            )

        evidence_label = self.evidence_label
        if evidence_label is not None:
            evidence_label = evidence_label.strip() or None
        object.__setattr__(self, "evidence_label", evidence_label)

    def chain_id(self) -> str:
        """Return the single chain covered by this correspondence."""

        return _single_chain_id(self.structure_endpoint)


@dataclass(frozen=True, slots=True)
class StructureUniProtCorrespondenceSet:
    """Candidate UniProt correspondences for one structure polymer region."""

    structure_endpoint: StructureEndpoint
    candidates: tuple[StructureUniProtCorrespondence, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.structure_endpoint, StructureEndpoint):
            raise TypeError(
                "structure-UniProt correspondence set requires a StructureEndpoint"
            )
        _require_polymer_region_endpoint(self.structure_endpoint)

        candidates = tuple(dict.fromkeys(self.candidates))
        if not candidates:
            raise ValueError(
                "structure-UniProt correspondence set requires at least one candidate"
            )

        seen_references: set[UniProtSequenceReference] = set()
        for candidate in candidates:
            if not isinstance(candidate, StructureUniProtCorrespondence):
                raise TypeError(
                    "structure-UniProt correspondence set candidates must be "
                    "StructureUniProtCorrespondence values"
                )
            if candidate.structure_endpoint != self.structure_endpoint:
                raise ValueError(
                    "structure-UniProt correspondence set candidates must share "
                    "the same structure endpoint"
                )
            if candidate.uniprot_reference in seen_references:
                raise ValueError(
                    "structure-UniProt correspondence set candidates must not "
                    "repeat the same UniProt reference"
                )
            seen_references.add(candidate.uniprot_reference)

        object.__setattr__(self, "candidates", candidates)

    def chain_id(self) -> str:
        """Return the single chain covered by this correspondence set."""

        return _single_chain_id(self.structure_endpoint)

    def is_ambiguous(self) -> bool:
        """Return whether multiple UniProt candidates remain in play."""

        return len(self.candidates) > 1

    def best_candidates(self) -> tuple[StructureUniProtCorrespondence, ...]:
        """Return highest-confidence candidates in stable first-seen order."""

        best_confidence = max(candidate.confidence for candidate in self.candidates)
        return tuple(
            candidate
            for candidate in self.candidates
            if candidate.confidence is best_confidence
        )

    def primary_candidate(self) -> StructureUniProtCorrespondence | None:
        """Return one unique best candidate when the set is unambiguous."""

        best_candidates = self.best_candidates()
        if len(best_candidates) != 1:
            return None

        return best_candidates[0]


def _require_unit_interval(
    value: float,
    *,
    field_name: str,
) -> None:
    """Raise when one confidence-like value falls outside [0, 1]."""

    if not 0.0 <= value <= 1.0:
        raise ValueError(f"{field_name} must lie in [0, 1]")


def _require_polymer_region_endpoint(endpoint: StructureEndpoint) -> None:
    """Raise when one structure endpoint is not a single-chain polymer region."""

    _single_chain_id(endpoint)


def _single_chain_id(endpoint: StructureEndpoint) -> str:
    """Return the single chain covered by one polymer-region endpoint."""

    chain_ids = {
        residue_id.chain_id
        for residue_id in _referenced_residue_ids(endpoint)
    }
    if not chain_ids:
        raise ValueError(
            "structure-UniProt correspondence requires at least one residue anchor"
        )
    if len(chain_ids) != 1:
        raise ValueError(
            "structure-UniProt correspondence requires a single-chain polymer region"
        )

    return next(iter(chain_ids))


def _referenced_residue_ids(
    endpoint: StructureEndpoint,
) -> tuple[ResidueId, ...]:
    """Return residue ids touched by one structure endpoint."""

    scope = endpoint.scope
    if isinstance(scope, ResidueSetScope):
        return scope.residue_ids

    if isinstance(scope, AbsentResidueSpanScope):
        residue_ids: list[ResidueId] = []
        if scope.preceding_residue_id is not None:
            residue_ids.append(scope.preceding_residue_id)
        if scope.following_residue_id is not None:
            residue_ids.append(scope.following_residue_id)
        residue_ids.extend(scope.absent_residue_ids)
        return tuple(dict.fromkeys(residue_ids))

    if isinstance(scope, CompositeScope):
        composite_residue_ids: list[ResidueId] = []
        for member_scope in scope.scopes:
            composite_residue_ids.extend(
                _referenced_residue_ids(
                    StructureEndpoint(
                        carrier_handle=endpoint.carrier_handle,
                        scope=member_scope,
                        realization_selector=endpoint.realization_selector,
                    )
                )
            )
        return tuple(dict.fromkeys(composite_residue_ids))

    raise ValueError(
        "structure-UniProt correspondence only supports residue-set, "
        "absent-residue-span, or flat composite endpoints"
    )


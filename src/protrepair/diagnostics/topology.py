"""Topology diagnostics for cis peptides and disulfide assignments.

Criteria:
- cis peptide: wwPDB CISPEP definition, omega = 0 +/- 30 degrees
- likely disulfide: unique SG-SG candidate within 2.5 A
- ambiguous disulfide: any sulfur with multiple SG partners within 3.0 A

The 3.0 A ambiguity cutoff matches HTMD's permissive disulfide candidate search.
"""

from collections import defaultdict
from dataclasses import dataclass
from math import isfinite

from protrepair.diagnostics.events import EventScope, ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.geometry import InternalCoordinateFrame
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.disulfide import (
    disulfide_bonded_cysteine_residue_ids,
)
from protrepair.structure.labels import AtomRef, ResidueId

CIS_PEPTIDE_ABS_OMEGA_MAX_DEGREES = 30.0
LIKELY_DISULFIDE_DISTANCE_MAX_ANGSTROM = 2.5
AMBIGUOUS_DISULFIDE_DISTANCE_MAX_ANGSTROM = 3.0


@dataclass(frozen=True, slots=True)
class CisPeptideFinding:
    """One adjacent residue pair whose omega angle is cis-like."""

    left_residue_id: ResidueId
    right_residue_id: ResidueId
    omega_degrees: float


@dataclass(frozen=True, slots=True)
class LikelyDisulfideBond:
    """One unique cysteine sulfur pair that looks like a likely disulfide."""

    left_residue_id: ResidueId
    right_residue_id: ResidueId
    sg_distance_angstrom: float

    def __post_init__(self) -> None:
        if self.left_residue_id == self.right_residue_id:
            raise ValueError("likely disulfide evidence requires two residues")
        if self.right_residue_id < self.left_residue_id:
            left_residue_id = self.left_residue_id
            object.__setattr__(self, "left_residue_id", self.right_residue_id)
            object.__setattr__(self, "right_residue_id", left_residue_id)
        if (
            not isfinite(self.sg_distance_angstrom)
            or self.sg_distance_angstrom <= 0.0
        ):
            raise ValueError(
                "likely disulfide evidence requires a finite positive distance"
            )

    def residue_pair(self) -> tuple[ResidueId, ResidueId]:
        """Return the canonically ordered candidate residue pair."""

        return (self.left_residue_id, self.right_residue_id)


@dataclass(frozen=True, slots=True)
class DisulfideCandidate:
    """One candidate sulfur partner for an ambiguous cysteine."""

    residue_id: ResidueId
    sg_distance_angstrom: float

    def __post_init__(self) -> None:
        if (
            not isfinite(self.sg_distance_angstrom)
            or self.sg_distance_angstrom <= 0.0
        ):
            raise ValueError(
                "disulfide candidates require a finite positive distance"
            )


@dataclass(frozen=True, slots=True)
class AmbiguousDisulfideFinding:
    """One cysteine sulfur with multiple competing disulfide partners."""

    residue_id: ResidueId
    candidates: tuple[DisulfideCandidate, ...]

    def __post_init__(self) -> None:
        candidates = tuple(self.candidates)
        if len(candidates) < 2:
            raise ValueError(
                "ambiguous disulfide findings require multiple candidates"
            )
        candidate_residue_ids = tuple(
            candidate.residue_id for candidate in candidates
        )
        if self.residue_id in candidate_residue_ids:
            raise ValueError("ambiguous disulfide candidates must be distinct")
        if len(set(candidate_residue_ids)) != len(candidate_residue_ids):
            raise ValueError("ambiguous disulfide candidates must not repeat")
        object.__setattr__(self, "candidates", candidates)


@dataclass(frozen=True, slots=True)
class TopologyReport:
    """Structured cis-peptide and disulfide topology findings."""

    cis_peptides: tuple[CisPeptideFinding, ...]
    likely_disulfides: tuple[LikelyDisulfideBond, ...]
    ambiguous_disulfides: tuple[AmbiguousDisulfideFinding, ...]

    def __post_init__(self) -> None:
        object.__setattr__(self, "cis_peptides", tuple(self.cis_peptides))
        object.__setattr__(self, "likely_disulfides", tuple(self.likely_disulfides))
        object.__setattr__(
            self,
            "ambiguous_disulfides",
            tuple(self.ambiguous_disulfides),
        )

    def is_empty(self) -> bool:
        """Return whether the report contains no findings."""

        return not (
            self.cis_peptides
            or self.likely_disulfides
            or self.ambiguous_disulfides
        )

    def to_issues(self) -> tuple[ValidationIssue, ...]:
        """Project warning-grade topology findings into validation issues."""

        issues: list[ValidationIssue] = []
        for finding in self.cis_peptides:
            issues.append(
                ValidationIssue(
                    kind=ValidationIssueKind.CIS_PEPTIDE,
                    severity=IssueSeverity.WARNING,
                    scope=EventScope.for_residue_pair(
                        finding.left_residue_id,
                        finding.right_residue_id,
                    ),
                    message=(
                        f"{finding.left_residue_id.display_token()}-"
                        f"{finding.right_residue_id.display_token()} has "
                        f"cis-like omega {finding.omega_degrees:.1f} deg"
                    ),
                )
            )

        for finding in self.ambiguous_disulfides:
            candidate_text = ", ".join(
                f"{candidate.residue_id.display_token()} "
                f"({candidate.sg_distance_angstrom:.2f} A)"
                for candidate in finding.candidates
            )
            issues.append(
                ValidationIssue(
                    kind=ValidationIssueKind.AMBIGUOUS_DISULFIDE,
                    severity=IssueSeverity.WARNING,
                    scope=EventScope.for_residue_set(
                        (
                            finding.residue_id,
                            *tuple(
                                candidate.residue_id
                                for candidate in finding.candidates
                            ),
                        )
                    ),
                    message=(
                        f"{finding.residue_id.display_token()} has multiple possible "
                        f"disulfide partners: {candidate_text}"
                    ),
                )
            )

        return tuple(issues)


def detect_topology(structure: ProteinStructure) -> TopologyReport:
    """Return cis-peptide and disulfide topology findings for one structure."""

    cis_peptides = detect_cis_peptides(structure)
    likely_disulfides, ambiguous_disulfides = detect_disulfide_topology(structure)
    return TopologyReport(
        cis_peptides=cis_peptides,
        likely_disulfides=likely_disulfides,
        ambiguous_disulfides=ambiguous_disulfides,
    )


def detect_cis_peptides(
    structure: ProteinStructure,
) -> tuple[CisPeptideFinding, ...]:
    """Return adjacent same-chain residue pairs with cis-like omega angles."""

    findings: list[CisPeptideFinding] = []
    for chain in structure.constitution.chains:
        for left_residue, right_residue in zip(
            chain.residues,
            chain.residues[1:],
            strict=False,
        ):
            if not has_omega_atoms(left_residue, right_residue):
                continue

            left_geometry = structure.residue_geometry(
                structure.constitution.residue_index(left_residue.residue_id)
            )
            right_geometry = structure.residue_geometry(
                structure.constitution.residue_index(right_residue.residue_id)
            )
            omega_degrees = InternalCoordinateFrame.torsion(
                left_geometry.position("CA"),
                left_geometry.position("C"),
                right_geometry.position("N"),
                right_geometry.position("CA"),
            )
            if abs(omega_degrees) > CIS_PEPTIDE_ABS_OMEGA_MAX_DEGREES:
                continue

            findings.append(
                CisPeptideFinding(
                    left_residue_id=left_residue.residue_id,
                    right_residue_id=right_residue.residue_id,
                    omega_degrees=omega_degrees,
                )
            )

    return tuple(findings)


def detect_disulfide_topology(
    structure: ProteinStructure,
) -> tuple[tuple[LikelyDisulfideBond, ...], tuple[AmbiguousDisulfideFinding, ...]]:
    """Return raw geometric disulfide evidence across all CYS SG sites."""

    return _classify_disulfide_evidence(
        structure,
        cysteine_sites=cysteine_sulfur_sites(structure),
    )


def detect_unassigned_disulfide_evidence(
    structure: ProteinStructure,
) -> tuple[tuple[LikelyDisulfideBond, ...], tuple[AmbiguousDisulfideFinding, ...]]:
    """Return planning evidence over canonically unassigned CYS SG sites."""

    assigned_residue_ids = disulfide_bonded_cysteine_residue_ids(structure)
    return _classify_disulfide_evidence(
        structure,
        cysteine_sites=tuple(
            site
            for site in cysteine_sulfur_sites(structure)
            if site.residue_id not in assigned_residue_ids
        ),
    )


def _classify_disulfide_evidence(
    structure: ProteinStructure,
    *,
    cysteine_sites: tuple[ResidueSite, ...],
) -> tuple[tuple[LikelyDisulfideBond, ...], tuple[AmbiguousDisulfideFinding, ...]]:
    """Classify geometric disulfide evidence over one explicit CYS projection."""

    candidate_pairs: list[tuple[int, int, float]] = []
    adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for left_index, left_site in enumerate(cysteine_sites):
        for right_index in range(left_index + 1, len(cysteine_sites)):
            right_site = cysteine_sites[right_index]
            left_geometry = structure.geometry.atom_geometry(
                structure.constitution.atom_index(
                    AtomRef(residue_id=left_site.residue_id, atom_name="SG")
                )
            )
            right_geometry = structure.geometry.atom_geometry(
                structure.constitution.atom_index(
                    AtomRef(residue_id=right_site.residue_id, atom_name="SG")
                )
            )
            pair_distance = left_geometry.distance_to(right_geometry)
            if (
                pair_distance <= 0.0
                or pair_distance > AMBIGUOUS_DISULFIDE_DISTANCE_MAX_ANGSTROM
            ):
                continue

            candidate_pairs.append((left_index, right_index, pair_distance))
            adjacency[left_index].append((right_index, pair_distance))
            adjacency[right_index].append((left_index, pair_distance))

    ambiguous_indices = {
        index for index, neighbors in adjacency.items() if len(neighbors) > 1
    }
    ambiguous_findings = tuple(
        AmbiguousDisulfideFinding(
            residue_id=cysteine_sites[index].residue_id,
            candidates=tuple(
                sorted(
                    (
                        DisulfideCandidate(
                            residue_id=cysteine_sites[neighbor_index].residue_id,
                            sg_distance_angstrom=pair_distance,
                        )
                        for neighbor_index, pair_distance in adjacency[index]
                    ),
                    key=lambda candidate: residue_sort_key(candidate.residue_id),
                )
            ),
        )
        for index in sorted(
            ambiguous_indices,
            key=lambda candidate_index: residue_sort_key(
                cysteine_sites[candidate_index].residue_id
            ),
        )
    )

    likely_bonds: list[LikelyDisulfideBond] = []
    for left_index, right_index, pair_distance in candidate_pairs:
        if pair_distance > LIKELY_DISULFIDE_DISTANCE_MAX_ANGSTROM:
            continue

        if left_index in ambiguous_indices or right_index in ambiguous_indices:
            continue

        likely_bonds.append(
            LikelyDisulfideBond(
                left_residue_id=cysteine_sites[left_index].residue_id,
                right_residue_id=cysteine_sites[right_index].residue_id,
                sg_distance_angstrom=pair_distance,
            )
        )

    return tuple(likely_bonds), ambiguous_findings


def cysteine_sulfur_sites(structure: ProteinStructure) -> tuple[ResidueSite, ...]:
    """Return cysteine residues with an SG atom across one structure."""

    return tuple(
        residue
        for residue in structure.constitution.iter_residues(include_ligands=True)
        if residue.component_id == "CYS" and residue.has_atom_site("SG")
    )


def has_omega_atoms(left_residue: ResidueSite, right_residue: ResidueSite) -> bool:
    """Return whether one residue pair has the atoms needed for omega."""

    return (
        left_residue.has_atom_site("CA")
        and left_residue.has_atom_site("C")
        and right_residue.has_atom_site("N")
        and right_residue.has_atom_site("CA")
    )


def residue_sort_key(residue_id: ResidueId) -> tuple[str, int, str]:
    """Return a stable residue ordering key."""

    return (
        residue_id.chain_id,
        residue_id.seq_num,
        residue_id.insertion_code or "",
    )

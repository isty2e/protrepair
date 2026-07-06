"""Source-microstate evidence diagnostics and decision projection."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.inference.retained_non_polymer_fallback import (
    retained_non_polymer_rdkit_fallback_heavy_bond_definitions,
)
from protrepair.chemistry.single_atom_inorganic import is_single_atom_inorganic_residue
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.errors import RdkitUnavailableError
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import ResidueGeometry


class MicrostateStructuralRole(str, Enum):
    """Closed structural-role axis for microstate adjudication."""

    STANDARD_POLYMER = "standard_polymer"
    RETAINED_NON_POLYMER = "retained_non_polymer"
    SINGLE_ATOM_INORGANIC = "single_atom_inorganic"
    UNCLASSIFIED_RESIDUE = "unclassified_residue"


class MicrostateChemistrySupportMode(str, Enum):
    """Closed chemistry-support axis for microstate adjudication."""

    STANDARD_COMPONENT_TEMPLATE = "standard_component_template"
    CURATED_COMPONENT_TEMPLATE = "curated_component_template"
    TEMPLATELESS_RDKIT_FALLBACK = "templateless_rdkit_fallback"
    NONE = "none"


class MicrostateApplicability(str, Enum):
    """Closed applicability axis for microstate adjudication."""

    APPLICABLE = "applicable"
    NOT_APPLICABLE = "not_applicable"


class MicrostateDecision(str, Enum):
    """Closed decision outcomes for one adjudicated residue."""

    PRESERVE_SOURCE = "preserve_source"
    ADJUDICATE = "adjudicate"
    AMBIGUOUS = "ambiguous"
    NOT_APPLICABLE = "not_applicable"


class MicrostateDecisionReason(str, Enum):
    """Reasons that justify one microstate decision."""

    FAMILY_NOT_SUPPORTED = "family_not_supported"
    EXPLICIT_HYDROGEN_EVIDENCE = "explicit_hydrogen_evidence"
    NO_RELEVANT_MOTIF = "no_relevant_motif"
    NO_CONTRADICTION_DETECTED = "no_contradiction_detected"
    HARD_IMPOSSIBILITY = "hard_impossibility"
    INSUFFICIENT_GEOMETRY_SUPPORT = "insufficient_geometry_support"
    TEMPLATE_SOURCE_CHARGE_CONTRADICTION = "template_source_charge_contradiction"
    TEMPLATE_GEOMETRY_BOND_CONTRADICTION = "template_geometry_bond_contradiction"
    SOURCE_CHARGE_GEOMETRY_CONTRADICTION = "source_charge_geometry_contradiction"


@dataclass(frozen=True, slots=True)
class CarboxylateLikeMotifEvidence:
    """Evidence collected for one carbon-with-two-oxygens motif."""

    carbon_atom_name: str
    oxygen_atom_names: tuple[str, str]
    source_double_negative: bool
    distance_to_oxygen_1_angstrom: float | None
    distance_to_oxygen_2_angstrom: float | None
    geometry_supports_delocalized_microstate: bool


@dataclass(frozen=True, slots=True)
class MicrostateClassification:
    """Orthogonal residue classification for microstate adjudication."""

    structural_role: MicrostateStructuralRole
    chemistry_support_mode: MicrostateChemistrySupportMode
    applicability: MicrostateApplicability = MicrostateApplicability.APPLICABLE

    def uses_standard_polymer_policy(self) -> bool:
        """Return whether this classification uses standard-polymer adjudication."""

        return (
            self.applicability is MicrostateApplicability.APPLICABLE
            and self.structural_role is MicrostateStructuralRole.STANDARD_POLYMER
        )

    def uses_curated_retained_policy(self) -> bool:
        """Return whether this classification uses curated retained chemistry."""

        return (
            self.applicability is MicrostateApplicability.APPLICABLE
            and self.structural_role is MicrostateStructuralRole.RETAINED_NON_POLYMER
            and self.chemistry_support_mode
            is MicrostateChemistrySupportMode.CURATED_COMPONENT_TEMPLATE
        )

    def uses_unknown_retained_policy(self) -> bool:
        """Return whether this classification uses fallback retained chemistry."""

        return (
            self.applicability is MicrostateApplicability.APPLICABLE
            and self.chemistry_support_mode
            is MicrostateChemistrySupportMode.TEMPLATELESS_RDKIT_FALLBACK
        )


@dataclass(frozen=True, slots=True)
class MicrostateEvidence:
    """Collected residue-local evidence before family policy adjudication."""

    residue_site: ResidueSite
    classification: MicrostateClassification
    has_explicit_hydrogen_evidence: bool
    source_nonzero_formal_charges: Mapping[str, int]
    carboxylate_like_motifs: tuple[CarboxylateLikeMotifEvidence, ...] = ()
    template_charge_mismatch_descriptors: tuple[str, ...] = ()
    template_geometry_bond_mismatch_descriptors: tuple[str, ...] = ()
    unknown_charge_geometry_contradiction_descriptors: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class MicrostateDecisionRecord:
    """One residue-local microstate decision plus mutation payload."""

    residue_site: ResidueSite
    classification: MicrostateClassification
    decision: MicrostateDecision
    reasons: tuple[MicrostateDecisionReason, ...]
    demoted_atom_names: tuple[str, ...] = ()
    issue_details: tuple[str, ...] = ()


_CARBOXYLATE_DISTANCE_SIMILARITY_TOLERANCE_ANGSTROM = 0.12
_CARBOXYLATE_DISTANCE_UPPER_BOUND_ANGSTROM = 1.35


def collect_microstate_evidence(
    residue_site: ResidueSite,
    *,
    residue_geometry: ResidueGeometry,
    source_formal_charge_by_atom_name: Mapping[str, int | None],
    standard_component_library: ComponentLibrary,
    component_library: ComponentLibrary,
) -> MicrostateEvidence:
    """Collect one residue-local evidence record for later adjudication."""

    classification = _microstate_classification_for_residue(
        residue_site,
        standard_component_library=standard_component_library,
        component_library=component_library,
    )
    source_nonzero_formal_charges = _normalized_nonzero_charge_payload(
        source_formal_charge_by_atom_name
    )
    template_charge_mismatch_descriptors: tuple[str, ...] = ()
    template_geometry_bond_mismatch_descriptors: tuple[str, ...] = ()
    unknown_charge_geometry_contradiction_descriptors: tuple[str, ...] = ()
    if classification.uses_curated_retained_policy():
        template = component_library.get(residue_site.component_id)
        if template is not None:
            template_charge_mismatch_descriptors = _curated_template_charge_mismatches(
                source_nonzero_formal_charges,
                template_nonzero_formal_charges=_normalized_nonzero_charge_payload(
                    template.definition.formal_charges
                ),
            )
            template_geometry_bond_mismatch_descriptors = (
                _geometry_bond_mismatch_descriptors(
                    residue_site,
                    residue_geometry=residue_geometry,
                    template_bonds=template.definition.bonds,
                    charged_atom_names=frozenset(source_nonzero_formal_charges),
                )
            )
    if classification.uses_unknown_retained_policy():
        unknown_charge_geometry_contradiction_descriptors = (
            _unknown_charge_geometry_contradiction_descriptors(
                residue_site,
                residue_geometry=residue_geometry,
                source_nonzero_formal_charges=source_nonzero_formal_charges,
            )
        )

    carboxylate_like_motifs = (
        _collect_standard_polymer_carboxylate_like_motif_evidence(
            residue_site,
            residue_geometry=residue_geometry,
            source_formal_charge_by_atom_name=source_nonzero_formal_charges,
            component_library=component_library,
        )
        if classification.uses_standard_polymer_policy()
        else ()
    )
    return MicrostateEvidence(
        residue_site=residue_site,
        classification=classification,
        has_explicit_hydrogen_evidence=_residue_contains_explicit_hydrogen(
            residue_site
        ),
        source_nonzero_formal_charges=source_nonzero_formal_charges,
        carboxylate_like_motifs=carboxylate_like_motifs,
        template_charge_mismatch_descriptors=template_charge_mismatch_descriptors,
        template_geometry_bond_mismatch_descriptors=(
            template_geometry_bond_mismatch_descriptors
        ),
        unknown_charge_geometry_contradiction_descriptors=(
            unknown_charge_geometry_contradiction_descriptors
        ),
    )


def adjudicate_microstate_evidence(
    evidence: MicrostateEvidence,
) -> MicrostateDecisionRecord:
    """Adjudicate one residue-local evidence record into a closed decision."""

    if evidence.classification.uses_standard_polymer_policy():
        return _adjudicate_standard_polymer_microstate(evidence)

    if evidence.classification.uses_curated_retained_policy():
        return _adjudicate_curated_retained_non_polymer_microstate(evidence)

    if evidence.classification.uses_unknown_retained_policy():
        return _adjudicate_unknown_retained_non_polymer_microstate(evidence)

    if evidence.classification.applicability is MicrostateApplicability.NOT_APPLICABLE:
        return _adjudicate_metal_or_ion_microstate(evidence)

    return MicrostateDecisionRecord(
        residue_site=evidence.residue_site,
        classification=evidence.classification,
        decision=MicrostateDecision.PRESERVE_SOURCE,
        reasons=(MicrostateDecisionReason.FAMILY_NOT_SUPPORTED,),
    )


def validation_issue_from_microstate_decision(
    decision_record: MicrostateDecisionRecord,
) -> ValidationIssue | None:
    """Project one adjudication decision into one typed validation issue."""

    if (
        decision_record.classification.uses_standard_polymer_policy()
        and decision_record.decision is MicrostateDecision.ADJUDICATE
        and decision_record.issue_details
    ):
        residue_token = decision_record.residue_site.residue_id.display_token()
        return ValidationIssue.for_residue(
            kind=ValidationIssueKind.CHEMISTRY_CONTRADICTION,
            severity=IssueSeverity.WARNING,
            message=(
                f"{residue_token} carries one "
                "chemically impossible standard-polymer carboxylate-like charge "
                "annotation without explicit hydrogen evidence; "
                + "; ".join(decision_record.issue_details)
                + ". Source atom charges were demoted for canonical handling."
            ),
            residue_id=decision_record.residue_site.residue_id,
        )

    if (
        decision_record.classification.uses_curated_retained_policy()
        and decision_record.decision is MicrostateDecision.AMBIGUOUS
        and decision_record.issue_details
    ):
        residue_token = decision_record.residue_site.residue_id.display_token()
        return ValidationIssue.for_residue(
            kind=ValidationIssueKind.CHEMISTRY_CONTRADICTION,
            severity=IssueSeverity.WARNING,
            message=(
                f"{residue_token} carries "
                "contradictory chemistry evidence for template-backed hydrogenation; "
                + "; ".join(decision_record.issue_details)
                + ". Hydrogen placement follows template chemistry."
            ),
            residue_id=decision_record.residue_site.residue_id,
        )

    if (
        decision_record.classification.uses_unknown_retained_policy()
        and decision_record.decision is MicrostateDecision.AMBIGUOUS
        and decision_record.issue_details
    ):
        residue_token = decision_record.residue_site.residue_id.display_token()
        return ValidationIssue.for_residue(
            kind=ValidationIssueKind.CHEMISTRY_CONTRADICTION,
            severity=IssueSeverity.WARNING,
            message=(
                f"{residue_token} carries contradictory source-charge and "
                "geometry evidence for unknown retained non-polymer chemistry; "
                + "; ".join(decision_record.issue_details)
                + ". No automatic microstate adjudication was applied."
            ),
            residue_id=decision_record.residue_site.residue_id,
        )

    return None


def _adjudicate_standard_polymer_microstate(
    evidence: MicrostateEvidence,
) -> MicrostateDecisionRecord:
    """Adjudicate one standard-polymer residue using hard-constraint rules."""

    if evidence.has_explicit_hydrogen_evidence:
        return MicrostateDecisionRecord(
            residue_site=evidence.residue_site,
            classification=evidence.classification,
            decision=MicrostateDecision.PRESERVE_SOURCE,
            reasons=(MicrostateDecisionReason.EXPLICIT_HYDROGEN_EVIDENCE,),
        )

    if not evidence.carboxylate_like_motifs:
        return MicrostateDecisionRecord(
            residue_site=evidence.residue_site,
            classification=evidence.classification,
            decision=MicrostateDecision.PRESERVE_SOURCE,
            reasons=(MicrostateDecisionReason.NO_RELEVANT_MOTIF,),
        )

    demoted_atom_names: list[str] = []
    issue_details: list[str] = []
    insufficient_geometry_support = False
    for motif in evidence.carboxylate_like_motifs:
        if not motif.source_double_negative:
            continue

        if not motif.geometry_supports_delocalized_microstate:
            insufficient_geometry_support = True
            continue

        demoted_atom_names.extend(motif.oxygen_atom_names)
        issue_details.append(
            f"{motif.oxygen_atom_names[0]}/{motif.oxygen_atom_names[1]} were "
            "both annotated as -1 while "
            f"{motif.carbon_atom_name}-{motif.oxygen_atom_names[0]}="
            f"{motif.distance_to_oxygen_1_angstrom:.3f} Å and "
            f"{motif.carbon_atom_name}-{motif.oxygen_atom_names[1]}="
            f"{motif.distance_to_oxygen_2_angstrom:.3f} Å support one "
            "delocalized carboxylate-like microstate"
        )

    if demoted_atom_names:
        return MicrostateDecisionRecord(
            residue_site=evidence.residue_site,
            classification=evidence.classification,
            decision=MicrostateDecision.ADJUDICATE,
            reasons=(MicrostateDecisionReason.HARD_IMPOSSIBILITY,),
            demoted_atom_names=tuple(dict.fromkeys(demoted_atom_names)),
            issue_details=tuple(issue_details),
        )

    if insufficient_geometry_support:
        return MicrostateDecisionRecord(
            residue_site=evidence.residue_site,
            classification=evidence.classification,
            decision=MicrostateDecision.PRESERVE_SOURCE,
            reasons=(MicrostateDecisionReason.INSUFFICIENT_GEOMETRY_SUPPORT,),
        )

    return MicrostateDecisionRecord(
        residue_site=evidence.residue_site,
        classification=evidence.classification,
        decision=MicrostateDecision.PRESERVE_SOURCE,
        reasons=(MicrostateDecisionReason.NO_RELEVANT_MOTIF,),
    )


def _adjudicate_curated_retained_non_polymer_microstate(
    evidence: MicrostateEvidence,
) -> MicrostateDecisionRecord:
    """Adjudicate one curated retained non-polymer residue conservatively."""

    if evidence.has_explicit_hydrogen_evidence:
        return MicrostateDecisionRecord(
            residue_site=evidence.residue_site,
            classification=evidence.classification,
            decision=MicrostateDecision.PRESERVE_SOURCE,
            reasons=(MicrostateDecisionReason.EXPLICIT_HYDROGEN_EVIDENCE,),
        )

    reasons: list[MicrostateDecisionReason] = []
    issue_details: list[str] = []
    if evidence.template_charge_mismatch_descriptors:
        reasons.append(MicrostateDecisionReason.TEMPLATE_SOURCE_CHARGE_CONTRADICTION)
        issue_details.append(
            "source atom charges disagree with template formal charges: "
            + ", ".join(evidence.template_charge_mismatch_descriptors)
        )
    if evidence.template_geometry_bond_mismatch_descriptors:
        reasons.append(MicrostateDecisionReason.TEMPLATE_GEOMETRY_BOND_CONTRADICTION)
        issue_details.append(
            "geometry-inferred bonding around charged atoms disagrees with the "
            "template bond graph: "
            + ", ".join(evidence.template_geometry_bond_mismatch_descriptors)
        )
    if reasons:
        return MicrostateDecisionRecord(
            residue_site=evidence.residue_site,
            classification=evidence.classification,
            decision=MicrostateDecision.AMBIGUOUS,
            reasons=tuple(reasons),
            issue_details=tuple(issue_details),
        )

    return MicrostateDecisionRecord(
        residue_site=evidence.residue_site,
        classification=evidence.classification,
        decision=MicrostateDecision.PRESERVE_SOURCE,
        reasons=(MicrostateDecisionReason.NO_CONTRADICTION_DETECTED,),
    )


def _adjudicate_unknown_retained_non_polymer_microstate(
    evidence: MicrostateEvidence,
) -> MicrostateDecisionRecord:
    """Adjudicate one unknown retained non-polymer residue conservatively."""

    if evidence.has_explicit_hydrogen_evidence:
        return MicrostateDecisionRecord(
            residue_site=evidence.residue_site,
            classification=evidence.classification,
            decision=MicrostateDecision.PRESERVE_SOURCE,
            reasons=(MicrostateDecisionReason.EXPLICIT_HYDROGEN_EVIDENCE,),
        )

    if evidence.unknown_charge_geometry_contradiction_descriptors:
        return MicrostateDecisionRecord(
            residue_site=evidence.residue_site,
            classification=evidence.classification,
            decision=MicrostateDecision.AMBIGUOUS,
            reasons=(MicrostateDecisionReason.SOURCE_CHARGE_GEOMETRY_CONTRADICTION,),
            issue_details=(
                "source-charge-aware chemistry inference disagrees with "
                "geometry-backed inference: "
                + ", ".join(evidence.unknown_charge_geometry_contradiction_descriptors),
            ),
        )

    return MicrostateDecisionRecord(
        residue_site=evidence.residue_site,
        classification=evidence.classification,
        decision=MicrostateDecision.PRESERVE_SOURCE,
        reasons=(MicrostateDecisionReason.NO_CONTRADICTION_DETECTED,),
    )


def _adjudicate_metal_or_ion_microstate(
    evidence: MicrostateEvidence,
) -> MicrostateDecisionRecord:
    """Adjudicate one single-atom inorganic species conservatively."""

    return MicrostateDecisionRecord(
        residue_site=evidence.residue_site,
        classification=evidence.classification,
        decision=MicrostateDecision.PRESERVE_SOURCE,
        reasons=(MicrostateDecisionReason.NO_CONTRADICTION_DETECTED,),
    )


def _microstate_classification_for_residue(
    residue_site: ResidueSite,
    *,
    standard_component_library: ComponentLibrary,
    component_library: ComponentLibrary,
) -> MicrostateClassification:
    """Return the orthogonal microstate classification for one residue."""

    if residue_site.is_hetero:
        if _is_known_single_atom_inorganic_species(residue_site):
            return MicrostateClassification(
                structural_role=MicrostateStructuralRole.SINGLE_ATOM_INORGANIC,
                chemistry_support_mode=MicrostateChemistrySupportMode.NONE,
                applicability=MicrostateApplicability.NOT_APPLICABLE,
            )

        if component_library.has(residue_site.component_id):
            return MicrostateClassification(
                structural_role=MicrostateStructuralRole.RETAINED_NON_POLYMER,
                chemistry_support_mode=(
                    MicrostateChemistrySupportMode.CURATED_COMPONENT_TEMPLATE
                ),
            )

        return MicrostateClassification(
            structural_role=MicrostateStructuralRole.RETAINED_NON_POLYMER,
            chemistry_support_mode=(
                MicrostateChemistrySupportMode.TEMPLATELESS_RDKIT_FALLBACK
            ),
        )

    if standard_component_library.has(residue_site.component_id):
        return MicrostateClassification(
            structural_role=MicrostateStructuralRole.STANDARD_POLYMER,
            chemistry_support_mode=(
                MicrostateChemistrySupportMode.STANDARD_COMPONENT_TEMPLATE
            ),
        )

    return MicrostateClassification(
        structural_role=MicrostateStructuralRole.UNCLASSIFIED_RESIDUE,
        chemistry_support_mode=(
            MicrostateChemistrySupportMode.TEMPLATELESS_RDKIT_FALLBACK
        ),
    )


def _residue_contains_explicit_hydrogen(residue_site: ResidueSite) -> bool:
    """Return whether one residue already carries explicit hydrogen evidence."""

    return any(atom_site.element == "H" for atom_site in residue_site.atom_sites)


def _is_known_single_atom_inorganic_species(residue_site: ResidueSite) -> bool:
    """Return whether one residue should be treated as a single-atom ion/metal."""

    return is_single_atom_inorganic_residue(residue_site)


def _collect_standard_polymer_carboxylate_like_motif_evidence(
    residue_site: ResidueSite,
    *,
    residue_geometry: ResidueGeometry,
    source_formal_charge_by_atom_name: Mapping[str, int],
    component_library: ComponentLibrary,
) -> tuple[CarboxylateLikeMotifEvidence, ...]:
    """Collect carboxylate-like motif evidence for one standard-polymer residue."""

    template = component_library.get(residue_site.component_id)
    if template is None:
        return ()

    atom_element_by_name = {
        atom_site.name: atom_site.element for atom_site in residue_site.atom_sites
    }
    motif_evidences: list[CarboxylateLikeMotifEvidence] = []
    for atom_site in residue_site.atom_sites:
        if atom_site.element != "C":
            continue

        oxygen_neighbor_atom_names = tuple(
            sorted(
                neighbor_atom_name
                for neighbor_atom_name in template.definition.bonded_atom_names(
                    atom_site.name
                )
                if atom_element_by_name.get(neighbor_atom_name) == "O"
            )
        )
        if len(oxygen_neighbor_atom_names) != 2:
            continue

        supported_distances = _supported_carboxylate_distances(
            residue_geometry,
            carbon_atom_name=atom_site.name,
            oxygen_atom_names=oxygen_neighbor_atom_names,
        )
        motif_evidences.append(
            CarboxylateLikeMotifEvidence(
                carbon_atom_name=atom_site.name,
                oxygen_atom_names=oxygen_neighbor_atom_names,
                source_double_negative=all(
                    source_formal_charge_by_atom_name.get(atom_name) == -1
                    for atom_name in oxygen_neighbor_atom_names
                ),
                distance_to_oxygen_1_angstrom=(
                    None if supported_distances is None else supported_distances[0]
                ),
                distance_to_oxygen_2_angstrom=(
                    None if supported_distances is None else supported_distances[1]
                ),
                geometry_supports_delocalized_microstate=(
                    supported_distances is not None
                ),
            )
        )

    return tuple(motif_evidences)


def _supported_carboxylate_distances(
    residue_geometry: ResidueGeometry,
    *,
    carbon_atom_name: str,
    oxygen_atom_names: tuple[str, str],
) -> tuple[float, float] | None:
    """Return motif distances when geometry supports one shared microstate."""

    try:
        carbon_position = residue_geometry.atom_geometry(carbon_atom_name).position
        oxygen_positions = tuple(
            residue_geometry.atom_geometry(atom_name).position
            for atom_name in oxygen_atom_names
        )
    except KeyError:
        return None

    distance_1 = carbon_position.distance_to(oxygen_positions[0])
    distance_2 = carbon_position.distance_to(oxygen_positions[1])
    if max(distance_1, distance_2) > _CARBOXYLATE_DISTANCE_UPPER_BOUND_ANGSTROM:
        return None

    if (
        abs(distance_1 - distance_2)
        > _CARBOXYLATE_DISTANCE_SIMILARITY_TOLERANCE_ANGSTROM
    ):
        return None

    return (distance_1, distance_2)


def _normalized_nonzero_charge_payload(
    charge_payload: Mapping[str, int | None],
) -> dict[str, int]:
    """Return nonzero formal charges keyed by normalized atom name."""

    return {
        atom_name.strip().upper(): int(formal_charge)
        for atom_name, formal_charge in charge_payload.items()
        if formal_charge not in (None, 0)
    }


def _curated_template_charge_mismatches(
    source_nonzero_formal_charges: Mapping[str, int],
    *,
    template_nonzero_formal_charges: Mapping[str, int],
) -> tuple[str, ...]:
    """Return compact charge mismatch descriptors for one curated component."""

    return tuple(
        _charge_mismatch_descriptor(
            atom_name,
            source_charge=source_nonzero_formal_charges.get(atom_name, 0),
            template_charge=template_nonzero_formal_charges.get(atom_name, 0),
        )
        for atom_name in sorted(
            set(source_nonzero_formal_charges) | set(template_nonzero_formal_charges)
        )
        if source_nonzero_formal_charges.get(atom_name, 0)
        != template_nonzero_formal_charges.get(atom_name, 0)
    )


def _charge_mismatch_descriptor(
    atom_name: str,
    *,
    source_charge: int,
    template_charge: int,
) -> str:
    """Return one compact charge mismatch descriptor."""

    return f"{atom_name} source {source_charge:+d} vs template {template_charge:+d}"


def _geometry_bond_mismatch_descriptors(
    residue_site: ResidueSite,
    *,
    residue_geometry: ResidueGeometry,
    template_bonds: tuple[BondDefinition, ...],
    charged_atom_names: frozenset[str],
) -> tuple[str, ...]:
    """Return bond mismatch descriptors touching charged atoms."""

    if not template_bonds or not charged_atom_names:
        return ()

    try:
        inferred_bonds = retained_non_polymer_rdkit_fallback_heavy_bond_definitions(
            residue_site,
            residue_geometry,
            formal_charge_by_atom_name=None,
        )
    except Exception:
        return ()

    template_bonds_by_key = _bond_payload_by_key(template_bonds)
    inferred_bonds_by_key = _bond_payload_by_key(inferred_bonds)
    return _bond_mismatch_descriptors(
        reference_bonds_by_key=template_bonds_by_key,
        reference_label="template",
        observed_bonds_by_key=inferred_bonds_by_key,
        observed_label="inferred",
        charged_atom_names=charged_atom_names,
    )


def _unknown_charge_geometry_contradiction_descriptors(
    residue_site: ResidueSite,
    *,
    residue_geometry: ResidueGeometry,
    source_nonzero_formal_charges: Mapping[str, int],
) -> tuple[str, ...]:
    """Return contradiction descriptors for unknown retained non-polymers."""

    if not source_nonzero_formal_charges:
        return ()

    try:
        geometry_bonds = retained_non_polymer_rdkit_fallback_heavy_bond_definitions(
            residue_site,
            residue_geometry,
            formal_charge_by_atom_name=None,
        )
    except RdkitUnavailableError:
        return ()
    except Exception:
        return (
            "geometry-backed RDKit inference fails for charged atoms "
            + ", ".join(sorted(source_nonzero_formal_charges)),
        )

    charged_atom_names = frozenset(source_nonzero_formal_charges)
    try:
        source_charge_bonds = (
            retained_non_polymer_rdkit_fallback_heavy_bond_definitions(
                residue_site,
                residue_geometry,
                formal_charge_by_atom_name=source_nonzero_formal_charges,
            )
        )
    except RdkitUnavailableError:
        return ()
    except Exception:
        return (
            "source-charge-aware RDKit sanitization fails for charged atoms "
            + ", ".join(sorted(charged_atom_names)),
        )

    return _bond_mismatch_descriptors(
        reference_bonds_by_key=_bond_payload_by_key(geometry_bonds),
        reference_label="geometry-backed",
        observed_bonds_by_key=_bond_payload_by_key(source_charge_bonds),
        observed_label="source-charge-aware",
        charged_atom_names=charged_atom_names,
    )


def _bond_mismatch_descriptors(
    *,
    reference_bonds_by_key: Mapping[tuple[str, str], tuple[int, bool]],
    reference_label: str,
    observed_bonds_by_key: Mapping[tuple[str, str], tuple[int, bool]],
    observed_label: str,
    charged_atom_names: frozenset[str],
) -> tuple[str, ...]:
    """Return mismatch descriptors touching charged atoms."""

    mismatches: list[str] = []
    for bond_key in sorted(set(reference_bonds_by_key) | set(observed_bonds_by_key)):
        atom_name_1, atom_name_2 = bond_key
        if (
            atom_name_1 not in charged_atom_names
            and atom_name_2 not in charged_atom_names
        ):
            continue

        reference_payload = reference_bonds_by_key.get(bond_key)
        observed_payload = observed_bonds_by_key.get(bond_key)
        if reference_payload == observed_payload:
            continue

        mismatches.append(
            f"{atom_name_1}-{atom_name_2} {reference_label} "
            f"{_bond_payload_descriptor(reference_payload)} vs {observed_label} "
            f"{_bond_payload_descriptor(observed_payload)}"
        )

    return tuple(mismatches)


def _bond_payload_by_key(
    bonds: tuple[BondDefinition, ...],
) -> dict[tuple[str, str], tuple[int, bool]]:
    """Return bond payloads keyed by sorted undirected atom pairs."""

    def normalized_bond_key(bond: BondDefinition) -> tuple[str, str]:
        atom_name_1 = bond.atom_name_1
        atom_name_2 = bond.atom_name_2
        if atom_name_1 <= atom_name_2:
            return (atom_name_1, atom_name_2)

        return (atom_name_2, atom_name_1)

    return {
        normalized_bond_key(bond): (
            int(bond.order),
            bool(bond.aromatic),
        )
        for bond in bonds
    }


def _bond_payload_descriptor(payload: tuple[int, bool] | None) -> str:
    """Return one short bond payload descriptor."""

    if payload is None:
        return "absent"

    order, aromatic = payload
    if aromatic:
        return f"order {order} aromatic"

    return f"order {order}"

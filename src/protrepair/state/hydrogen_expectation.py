"""Hydrogen expectation read-models over observed structures."""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.inference.retained_non_polymer_evidence import (
    retained_non_polymer_evidence_expected_hydrogen_atom_names,
)
from protrepair.chemistry.inference.retained_non_polymer_fallback import (
    preferred_retained_non_polymer_expected_hydrogen_atom_names,
    retained_non_polymer_rdkit_fallback_expected_hydrogen_atom_names,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
    evidence_by_residue_id,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ChainSite, ResidueSite
from protrepair.structure.labels import AtomRef, ResidueId

DISULFIDE_HYDROGEN_SUPPRESSION_DISTANCE_ANGSTROM = 3.0


class RetainedNonPolymerChemistryEvidenceSource(str, Enum):
    """Evidence source for one retained non-polymer hydrogen expectation axis."""

    TEMPLATE = "template"
    EXTERNAL_OVERRIDE = "external_override"
    RDKIT_FALLBACK = "rdkit_fallback"
    UNRESOLVED = "unresolved"

    def is_resolved(self) -> bool:
        """Return whether this axis has concrete chemistry evidence."""

        return self is not RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED


@dataclass(frozen=True, slots=True)
class RetainedNonPolymerHydrogenExpectationResolution:
    """Resolved hydrogen expectation policy for one retained non-polymer residue."""

    source: RetainedNonPolymerChemistryEvidenceSource
    expected_hydrogen_atom_names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if (
            self.source is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
            and self.expected_hydrogen_atom_names
        ):
            raise ValueError(
                "unresolved retained non-polymer hydrogen expectation must not "
                "carry expected hydrogen names"
            )


@dataclass(frozen=True, slots=True)
class StructureHydrogenExpectationModel:
    """Hydrogen expectation policy derived for one observed structure."""

    expected_hydrogen_atom_names_by_residue: Mapping[ResidueId, tuple[str, ...]]
    retained_non_polymer_resolution_by_residue_id: Mapping[
        ResidueId,
        RetainedNonPolymerHydrogenExpectationResolution,
    ]

    def resolution_for_retained_non_polymer(
        self,
        residue_id: ResidueId,
    ) -> RetainedNonPolymerHydrogenExpectationResolution:
        """Return retained-non-polymer expectation resolution for one residue."""

        return self.retained_non_polymer_resolution_by_residue_id.get(
            residue_id,
            RetainedNonPolymerHydrogenExpectationResolution(
                source=RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
            ),
        )


def derive_structure_hydrogen_expectation_model(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence,
        ...,
    ] = (),
) -> StructureHydrogenExpectationModel:
    """Derive structure-level hydrogen expectation policy."""

    expected_hydrogen_atom_names_by_residue = _polymer_expected_hydrogen_atom_names(
        structure,
        component_library=component_library,
    )
    evidence_map = evidence_by_residue_id(retained_non_polymer_chemistry_evidence)
    retained_non_polymer_resolution_by_residue_id: dict[
        ResidueId,
        RetainedNonPolymerHydrogenExpectationResolution,
    ] = {}
    for ligand in structure.constitution.ligands:
        resolution = resolve_retained_non_polymer_hydrogen_expectation(
            structure,
            ligand,
            component_library=component_library,
            evidence=evidence_map.get(ligand.residue_id),
        )
        retained_non_polymer_resolution_by_residue_id[ligand.residue_id] = resolution
        if resolution.expected_hydrogen_atom_names:
            expected_hydrogen_atom_names_by_residue[ligand.residue_id] = (
                resolution.expected_hydrogen_atom_names
            )

    return StructureHydrogenExpectationModel(
        expected_hydrogen_atom_names_by_residue=MappingProxyType(
            expected_hydrogen_atom_names_by_residue
        ),
        retained_non_polymer_resolution_by_residue_id=MappingProxyType(
            retained_non_polymer_resolution_by_residue_id
        ),
    )


def resolve_retained_non_polymer_hydrogen_expectation(
    structure: ProteinStructure,
    residue: ResidueSite,
    *,
    component_library: ComponentLibrary,
    evidence: RetainedNonPolymerChemistryEvidence | None = None,
) -> RetainedNonPolymerHydrogenExpectationResolution:
    """Resolve hydrogen expectation policy for one retained non-polymer residue."""

    residue_index = structure.constitution.residue_index(residue.residue_id)
    residue_geometry = structure.residue_geometry(residue_index)
    present_hydrogen_atom_names = tuple(
        atom_site.name for atom_site in residue.atom_sites if atom_site.element == "H"
    )
    if evidence is not None:
        evidence_expected_hydrogen_atom_names = (
            retained_non_polymer_evidence_expected_hydrogen_atom_names(evidence)
        )
        return RetainedNonPolymerHydrogenExpectationResolution(
            source=RetainedNonPolymerChemistryEvidenceSource.EXTERNAL_OVERRIDE,
            expected_hydrogen_atom_names=_preferred_expected_hydrogen_atom_names(
                expected_hydrogen_atom_names=evidence_expected_hydrogen_atom_names,
                present_hydrogen_atom_names=present_hydrogen_atom_names,
            ),
        )

    template = component_library.get(residue.component_id)
    if template is not None and template.can_add_hydrogens():
        return RetainedNonPolymerHydrogenExpectationResolution(
            source=RetainedNonPolymerChemistryEvidenceSource.TEMPLATE,
            expected_hydrogen_atom_names=_preferred_expected_hydrogen_atom_names(
                expected_hydrogen_atom_names=(
                    _materializable_retained_non_polymer_template_hydrogen_atom_names(
                        residue,
                        template=template,
                    )
                ),
                present_hydrogen_atom_names=present_hydrogen_atom_names,
            ),
        )

    try:
        fallback_expected_hydrogen_atom_names = (
            retained_non_polymer_rdkit_fallback_expected_hydrogen_atom_names(
                residue,
                residue_geometry,
                formal_charge_by_atom_name=dict(
                    structure.residue_formal_charge_by_atom_name(residue_index)
                ),
            )
        )
    except (RdkitUnavailableError, RuntimeError, ValueError):
        return RetainedNonPolymerHydrogenExpectationResolution(
            source=RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
        )

    return RetainedNonPolymerHydrogenExpectationResolution(
        source=RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK,
        expected_hydrogen_atom_names=fallback_expected_hydrogen_atom_names,
    )


def _polymer_expected_hydrogen_atom_names(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
) -> dict[ResidueId, tuple[str, ...]]:
    """Return chain-aware expected hydrogens for polymer residues."""

    expected_hydrogen_atom_names_by_residue: dict[ResidueId, list[str]] = {}
    disulfide_bonded_residue_ids = _disulfide_bonded_cysteine_residue_ids(structure)
    for chain in structure.constitution.chains:
        _extend_chain_expected_hydrogens(
            chain,
            component_library=component_library,
            disulfide_bonded_residue_ids=disulfide_bonded_residue_ids,
            expected_hydrogen_atom_names_by_residue=(
                expected_hydrogen_atom_names_by_residue
            ),
        )

    return {
        residue_id: tuple(atom_names)
        for residue_id, atom_names in expected_hydrogen_atom_names_by_residue.items()
    }


def _extend_chain_expected_hydrogens(
    chain: ChainSite,
    *,
    component_library: ComponentLibrary,
    disulfide_bonded_residue_ids: frozenset[ResidueId],
    expected_hydrogen_atom_names_by_residue: dict[ResidueId, list[str]],
) -> None:
    """Accumulate chain-aware expected hydrogens onto one residue map."""

    templates = tuple(
        component_library.get(residue.component_id) for residue in chain.residues
    )
    for residue, template in zip(chain.residues, templates, strict=True):
        if template is None or not template.can_add_hydrogens():
            continue

        expected_hydrogen_atom_names = list(template.expected_hydrogen_atom_names())
        if (
            residue.component_id == "CYS"
            and residue.residue_id in disulfide_bonded_residue_ids
        ):
            expected_hydrogen_atom_names = [
                atom_name
                for atom_name in expected_hydrogen_atom_names
                if atom_name != "HG"
            ]
        if expected_hydrogen_atom_names:
            expected_hydrogen_atom_names_by_residue[residue.residue_id] = (
                expected_hydrogen_atom_names
            )

    if chain.residues:
        first_residue = chain.residues[0]
        first_template = templates[0]
        if (
            first_template is not None
            and first_template.can_add_hydrogens()
            and _supports_peptide_backbone_hydrogens(first_residue)
        ):
            backbone_family_component_id = first_template.backbone_family_component_id
            _append_expected_hydrogen_atom_names(
                expected_hydrogen_atom_names_by_residue,
                first_residue.residue_id,
                ("H1", "H2")
                if backbone_family_component_id == "PRO"
                else ("H1", "H2", "H3"),
            )

    for residue_index, residue in enumerate(chain.residues[:-1]):
        template = templates[residue_index]
        next_template = templates[residue_index + 1]
        if template is None or next_template is None:
            continue
        if not template.can_add_hydrogens() or not next_template.can_add_hydrogens():
            continue
        if not _supports_peptide_backbone_hydrogens(
            residue
        ) or not _supports_peptide_backbone_hydrogens(
            chain.residues[residue_index + 1]
        ):
            continue

        next_backbone_family_component_id = next_template.backbone_family_component_id
        if next_backbone_family_component_id == "PRO":
            continue

        _append_expected_hydrogen_atom_names(
            expected_hydrogen_atom_names_by_residue,
            chain.residues[residue_index + 1].residue_id,
            ("H",),
        )


def _preferred_expected_hydrogen_atom_names(
    *,
    expected_hydrogen_atom_names: tuple[str, ...],
    present_hydrogen_atom_names: tuple[str, ...],
) -> tuple[str, ...]:
    """Normalize expected hydrogen names while preserving present names when sane."""

    if not expected_hydrogen_atom_names:
        return ()

    return preferred_retained_non_polymer_expected_hydrogen_atom_names(
        inferred_hydrogen_count=len(expected_hydrogen_atom_names),
        present_hydrogen_atom_names=present_hydrogen_atom_names,
    )


def _materializable_retained_non_polymer_template_hydrogen_atom_names(
    residue: ResidueSite,
    *,
    template: ResidueTemplate,
) -> tuple[str, ...]:
    """Return template hydrogens whose heavy anchors exist in the retained surface."""

    expected_hydrogen_atom_names = tuple(template.expected_hydrogen_atom_names())
    anchor_by_hydrogen_name = template.template_hydrogen_anchor_by_name(
        expected_hydrogen_atom_names
    )
    present_atom_names = residue.atom_site_names()
    return tuple(
        hydrogen_atom_name
        for hydrogen_atom_name in expected_hydrogen_atom_names
        if anchor_by_hydrogen_name.get(hydrogen_atom_name) in present_atom_names
    )


def _append_expected_hydrogen_atom_names(
    expected_hydrogen_atom_names_by_residue: dict[ResidueId, list[str]],
    residue_id: ResidueId,
    hydrogen_atom_names: tuple[str, ...],
) -> None:
    """Append unique expected hydrogen names for one residue id."""

    ordered_hydrogen_atom_names = (
        expected_hydrogen_atom_names_by_residue.setdefault(residue_id, [])
    )
    seen_hydrogen_atom_names = set(ordered_hydrogen_atom_names)
    for hydrogen_atom_name in hydrogen_atom_names:
        if hydrogen_atom_name in seen_hydrogen_atom_names:
            continue

        ordered_hydrogen_atom_names.append(hydrogen_atom_name)
        seen_hydrogen_atom_names.add(hydrogen_atom_name)


def _disulfide_bonded_cysteine_residue_ids(
    structure: ProteinStructure,
) -> frozenset[ResidueId]:
    """Return cysteine residues whose SG is close enough to suppress HG."""

    cysteine_sg_atom_refs = tuple(
        AtomRef(residue_site.residue_id, "SG")
        for chain in structure.constitution.chains
        for residue_site in chain.residues
        if residue_site.component_id == "CYS" and residue_site.has_atom_site("SG")
    )
    disulfide_bonded_residue_ids: set[ResidueId] = set()
    for left_index, left_atom_ref in enumerate(cysteine_sg_atom_refs):
        left_position = structure.geometry.position(
            structure.constitution.atom_index(left_atom_ref)
        )
        for right_atom_ref in cysteine_sg_atom_refs[left_index + 1 :]:
            right_position = structure.geometry.position(
                structure.constitution.atom_index(right_atom_ref)
            )
            if (
                left_position.distance_to(right_position)
                > DISULFIDE_HYDROGEN_SUPPRESSION_DISTANCE_ANGSTROM
            ):
                continue

            disulfide_bonded_residue_ids.add(left_atom_ref.residue_id)
            disulfide_bonded_residue_ids.add(right_atom_ref.residue_id)

    return frozenset(disulfide_bonded_residue_ids)


def _supports_peptide_backbone_hydrogens(
    residue: ResidueSite,
) -> bool:
    """Return whether one residue can participate in peptide-H placement."""

    return all(residue.has_atom_site(atom_name) for atom_name in ("N", "CA", "C"))

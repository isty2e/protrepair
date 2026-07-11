"""Hydrogen expectation read-models over observed structures."""

from collections.abc import Mapping
from dataclasses import dataclass
from types import MappingProxyType

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
    evidence_by_residue_id,
)
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryEvidenceSource,
    RetainedNonPolymerChemistryResolution,
    resolve_retained_non_polymer_chemistry,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ChainSite, ResidueSite
from protrepair.structure.disulfide import disulfide_bonded_cysteine_residue_ids
from protrepair.structure.labels import ResidueId


@dataclass(frozen=True, slots=True)
class StructureHydrogenExpectationModel:
    """Hydrogen expectation policy derived for one observed structure."""

    expected_hydrogen_atom_names_by_residue: Mapping[ResidueId, tuple[str, ...]]
    retained_non_polymer_resolution_by_residue_id: Mapping[
        ResidueId,
        RetainedNonPolymerChemistryResolution,
    ]

    def resolution_for_retained_non_polymer(
        self,
        residue_id: ResidueId,
    ) -> RetainedNonPolymerChemistryResolution:
        """Return retained-non-polymer expectation resolution for one residue."""

        return self.retained_non_polymer_resolution_by_residue_id.get(
            residue_id,
            RetainedNonPolymerChemistryResolution(
                source=RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
            ),
        )


def derive_structure_hydrogen_expectation_model(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    allow_retained_non_polymer_rdkit_fallback: bool = True,
    retained_non_polymer_chemistry_evidence: tuple[
        RetainedNonPolymerChemistryEvidence,
        ...,
    ] = (),
) -> StructureHydrogenExpectationModel:
    """Derive structure-level hydrogen expectation policy."""

    disulfide_residue_ids = disulfide_bonded_cysteine_residue_ids(structure)
    expected_hydrogen_atom_names_by_residue = _polymer_expected_hydrogen_atom_names(
        structure,
        component_library=component_library,
        disulfide_bonded_residue_ids=disulfide_residue_ids,
    )
    evidence_map = evidence_by_residue_id(retained_non_polymer_chemistry_evidence)
    retained_non_polymer_resolution_by_residue_id: dict[
        ResidueId,
        RetainedNonPolymerChemistryResolution,
    ] = {}
    for ligand in structure.constitution.ligands:
        resolution = resolve_retained_non_polymer_chemistry(
            structure,
            ligand,
            component_library=component_library,
            evidence=evidence_map.get(ligand.residue_id),
            allow_rdkit_fallback=allow_retained_non_polymer_rdkit_fallback,
            disulfide_residue_ids=disulfide_residue_ids,
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


def _polymer_expected_hydrogen_atom_names(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary,
    disulfide_bonded_residue_ids: frozenset[ResidueId],
) -> dict[ResidueId, tuple[str, ...]]:
    """Return chain-aware expected hydrogens for polymer residues."""

    expected_hydrogen_atom_names_by_residue: dict[ResidueId, list[str]] = {}
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


def _supports_peptide_backbone_hydrogens(
    residue: ResidueSite,
) -> bool:
    """Return whether one residue can participate in peptide-H placement."""

    return all(residue.has_atom_site(atom_name) for atom_name in ("N", "CA", "C"))

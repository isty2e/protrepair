"""Retained non-polymer chemistry resolution facts."""

from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, replace
from enum import Enum
from itertools import combinations

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.component.semantics import IdealGeometryHydrogenSemantics
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.component.topology import (
    template_heavy_bond_definitions_for_present_atoms,
    template_hydrogen_bond_definitions_for_names,
)
from protrepair.chemistry.inference.retained_non_polymer_evidence import (
    require_retained_non_polymer_evidence_stereochemistry,
    retained_non_polymer_evidence_expected_hydrogen_atom_names,
    retained_non_polymer_evidence_external_bonds,
    retained_non_polymer_evidence_formal_charge_requirements,
    retained_non_polymer_evidence_heavy_atom_elements,
    retained_non_polymer_evidence_heavy_bond_definitions,
    retained_non_polymer_evidence_hydrogen_bond_definitions,
)
from protrepair.chemistry.inference.retained_non_polymer_fallback import (
    infer_retained_non_polymer_rdkit_fallback,
    preferred_retained_non_polymer_expected_hydrogen_atom_names,
)
from protrepair.chemistry.retained_non_polymer.covalent_context import (
    RetainedCovalentContext,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.chemistry.retained_non_polymer.template_binding import (
    RetainedTemplateBinding,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.disulfide import disulfide_bonded_cysteine_residue_ids
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import BondRelationshipType


class RetainedNonPolymerChemistryEvidenceSource(str, Enum):
    """Chemistry source for one retained non-polymer resolution axis."""

    TEMPLATE = "template"
    EXTERNAL_OVERRIDE = "external_override"
    RDKIT_FALLBACK = "rdkit_fallback"
    UNRESOLVED = "unresolved"

    def is_resolved(self) -> bool:
        """Return whether this axis has concrete chemistry evidence."""

        return self is not RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED


@dataclass(frozen=True, slots=True)
class RetainedNonPolymerChemistryResolution:
    """Selected chemistry facts bound to one retained residue and its context."""

    source: RetainedNonPolymerChemistryEvidenceSource
    expected_hydrogen_atom_names: tuple[str, ...] = ()
    expected_heavy_atom_names: tuple[str, ...] = ()
    heavy_bond_definitions: tuple[BondDefinition, ...] = ()
    hydrogen_bond_definitions: tuple[BondDefinition, ...] = ()
    heavy_atom_elements: tuple[str, ...] = ()
    formal_charge_requirements: tuple[tuple[str, int], ...] = ()
    external_bonds: tuple[tuple[AtomRef, AtomRef], ...] = ()
    failure_reason: str = ""
    rdkit_backend_version: str | None = None
    hydrogen_name_projection_candidate_count: int = 0
    hydrogen_name_projection_candidate_limit: int = 0

    def has_expected_heavy_scaffold(self, residue: ResidueSite) -> bool:
        """Check the selected product inventory instead of its isolated precursor.

        Parameters
        ----------
        residue : ResidueSite
            Current retained atom inventory.

        Returns
        -------
        bool
            The complete selected heavy inventory is present without extra names
            and agrees with any recorded element identities.
        """
        present = {atom.name for atom in residue.atom_sites if not atom.is_hydrogen()}
        if present != set(self.expected_heavy_atom_names):
            return False
        return not self.heavy_atom_elements or all(
            residue.atom_site(name).element == element
            for name, element in zip(
                self.expected_heavy_atom_names, self.heavy_atom_elements, strict=True
            )
        )

    def has_external_bond_topology(self, structure: ProteinStructure) -> bool:
        """Check that admitted external chemistry has been realized.

        Parameters
        ----------
        structure : ProteinStructure
            Current canonical graph, observed without modification.

        Returns
        -------
        bool
            Whether every required attachment is covalent-like, single and
            non-aromatic. A deposited LINK without order is insufficient.
        """
        for first, second in self.external_bonds:
            indices = tuple(
                structure.constitution.resolve_atom_index(ref)
                for ref in (first, second)
            )
            if indices[0] is None or indices[1] is None:
                return False
            bond = structure.topology.bond_between(indices[0], indices[1])
            if (
                bond is None
                or bond.relationship_type
                not in (BondRelationshipType.COVALENT, BondRelationshipType.DISULFIDE)
                or bond.order != 1
                or bond.aromatic
            ):
                return False
        return True

    def has_formal_charge_topology(
        self, structure: ProteinStructure, residue: ResidueSite
    ) -> bool:
        """Compare realized atom charges with the resolved chemistry.

        Parameters
        ----------
        structure : ProteinStructure
            Current canonical topology, observed without mutation.
        residue : ResidueSite
            Retained residue being checked.

        Returns
        -------
        bool
            Whether all selected charges are realized. Explicit SMILES requires
            an assignment even for zero; a neutral template permits an unknown
            charge but not a contradictory known charge.
        """
        residue_index = structure.constitution.residue_index(residue.residue_id)
        for name, charge in self.formal_charge_requirements:
            if not residue.has_atom_site(name):
                return False
            realized = structure.topology.formal_charge(
                structure.constitution.atom_index_in_residue(residue_index, name)
            )
            if realized == charge:
                continue
            if (
                realized is None
                and charge == 0
                and self.source is RetainedNonPolymerChemistryEvidenceSource.TEMPLATE
            ):
                continue
            return False
        return True

    def has_hydrogen_topology(
        self, structure: ProteinStructure, residue: ResidueSite
    ) -> bool:
        """Compare H attachments without treating source names as chemistry.

        Parameters
        ----------
        structure : ProteinStructure
            Current canonical graph, observed without mutation.
        residue : ResidueSite
            Retained residue on that graph.

        Returns
        -------
        bool
            Whether each present H has exactly one resolved local heavy anchor
            and the anchor multiplicities match admitted chemistry. Equivalent H
            names may differ, but matching a total H count alone is insufficient.
        """
        if not self.source.is_resolved():
            return False
        heavy_names = {
            atom.name for atom in residue.atom_sites if not atom.is_hydrogen()
        }
        expected_anchors: Counter[str] = Counter()
        for bond in self.hydrogen_bond_definitions:
            heavy_endpoints = heavy_names.intersection(
                (bond.atom_name_1, bond.atom_name_2)
            )
            if len(heavy_endpoints) != 1:
                return False
            expected_anchors.update(heavy_endpoints)
        hydrogens = tuple(
            atom.name for atom in residue.atom_sites if atom.is_hydrogen()
        )
        if len(hydrogens) != len(self.expected_hydrogen_atom_names):
            return False
        if sum(expected_anchors.values()) != len(hydrogens):
            return False

        residue_index = structure.constitution.residue_index(residue.residue_id)
        hydrogen_indices = tuple(
            structure.constitution.atom_index_in_residue(residue_index, name)
            for name in hydrogens
        )
        for first, second in combinations(hydrogen_indices, 2):
            bond = structure.topology.bond_between(first, second)
            if bond is not None and bond.relationship_type in {
                BondRelationshipType.UNKNOWN,
                BondRelationshipType.COVALENT,
                BondRelationshipType.DISULFIDE,
            }:
                return False

        observed_anchors: Counter[str] = Counter()
        for hydrogen_index in hydrogen_indices:
            anchors = []
            for heavy_name in heavy_names:
                heavy_index = structure.constitution.atom_index_in_residue(
                    residue_index, heavy_name
                )
                bond = structure.topology.bond_between(hydrogen_index, heavy_index)
                if bond is None or bond.relationship_type not in {
                    BondRelationshipType.UNKNOWN,
                    BondRelationshipType.COVALENT,
                    BondRelationshipType.DISULFIDE,
                }:
                    continue
                if (
                    bond.relationship_type is not BondRelationshipType.COVALENT
                    or bond.order != 1
                    or bond.aromatic
                ):
                    return False
                anchors.append(heavy_name)
            if len(anchors) != 1:
                return False
            observed_anchors.update(anchors)
        return observed_anchors == expected_anchors

    def __post_init__(self) -> None:
        rdkit_backend_version = self.rdkit_backend_version
        if rdkit_backend_version is not None:
            rdkit_backend_version = rdkit_backend_version.strip() or None
            object.__setattr__(
                self,
                "rdkit_backend_version",
                rdkit_backend_version,
            )

        resolved = self.source.is_resolved()
        if (
            self.source is RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED
            and self.expected_hydrogen_atom_names
        ):
            raise ValueError(
                "unresolved retained non-polymer hydrogen expectation must not "
                "carry expected hydrogen names"
            )
        if not resolved and (
            self.expected_heavy_atom_names
            or self.heavy_bond_definitions
            or self.hydrogen_bond_definitions
            or self.heavy_atom_elements
            or self.formal_charge_requirements
            or self.external_bonds
        ):
            raise ValueError(
                "unresolved retained non-polymer chemistry must not carry "
                "resolved topology or element facts"
            )
        if (
            self.expected_heavy_atom_names
            and self.heavy_atom_elements
            and len(self.expected_heavy_atom_names) != len(self.heavy_atom_elements)
        ):
            raise ValueError("resolved heavy atom names and elements must align")
        if resolved and self.failure_reason:
            raise ValueError(
                "resolved retained non-polymer chemistry must not carry a failure "
                "reason"
            )
        if (
            self.source is RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
            and rdkit_backend_version is None
        ):
            raise ValueError(
                "RDKit fallback chemistry must carry an RDKit backend version"
            )
        if (
            self.source is not RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
            and rdkit_backend_version is not None
        ):
            raise ValueError(
                "RDKit backend version is only valid for RDKit fallback chemistry"
            )
        if self.hydrogen_name_projection_candidate_count < 0:
            raise ValueError(
                "hydrogen_name_projection_candidate_count must not be negative"
            )
        if self.hydrogen_name_projection_candidate_limit < 0:
            raise ValueError(
                "hydrogen_name_projection_candidate_limit must not be negative"
            )
        if (
            self.hydrogen_name_projection_candidate_count
            and not self.hydrogen_name_projection_candidate_limit
        ):
            raise ValueError(
                "hydrogen_name_projection_candidate_limit is required when "
                "candidate_count is present"
            )
        if (
            self.source is not RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK
            and (
                self.hydrogen_name_projection_candidate_count
                or self.hydrogen_name_projection_candidate_limit
            )
        ):
            raise ValueError(
                "hydrogen name projection diagnostics are only valid for RDKit "
                "fallback chemistry"
            )

    def without_hydrogens_anchored_to(
        self,
        anchor_atom_names: Collection[str],
    ) -> "RetainedNonPolymerChemistryResolution":
        """Return chemistry facts without H atoms bonded to selected anchors."""

        normalized_anchor_names = {
            atom_name.strip().upper() for atom_name in anchor_atom_names
        }
        excluded_hydrogen_names = {
            bonded_atom_name
            for bond_definition in self.hydrogen_bond_definitions
            for anchor_atom_name, bonded_atom_name in (
                (bond_definition.atom_name_1, bond_definition.atom_name_2),
                (bond_definition.atom_name_2, bond_definition.atom_name_1),
            )
            if anchor_atom_name in normalized_anchor_names
        }
        if not excluded_hydrogen_names:
            return self

        return replace(
            self,
            expected_hydrogen_atom_names=tuple(
                atom_name
                for atom_name in self.expected_hydrogen_atom_names
                if atom_name not in excluded_hydrogen_names
            ),
            hydrogen_bond_definitions=tuple(
                bond_definition
                for bond_definition in self.hydrogen_bond_definitions
                if bond_definition.atom_name_1 not in excluded_hydrogen_names
                and bond_definition.atom_name_2 not in excluded_hydrogen_names
            ),
        )


def resolve_retained_non_polymer_chemistry(
    structure: ProteinStructure,
    residue: ResidueSite,
    *,
    component_library: ComponentLibrary,
    evidence: RetainedNonPolymerChemistryEvidence | None = None,
    allow_rdkit_fallback: bool = True,
    disulfide_residue_ids: frozenset[ResidueId] | None = None,
) -> RetainedNonPolymerChemistryResolution:
    """Resolve selected chemistry against source and current connected context."""

    resolution = _resolve_retained_non_polymer_chemistry_source(
        structure,
        residue,
        component_library=component_library,
        evidence=evidence,
        allow_rdkit_fallback=allow_rdkit_fallback,
    )
    active_disulfide_residue_ids = (
        disulfide_bonded_cysteine_residue_ids(structure)
        if disulfide_residue_ids is None
        else disulfide_residue_ids
    )
    if residue.residue_id not in active_disulfide_residue_ids:
        return resolution

    return resolution.without_hydrogens_anchored_to(("SG",))


def _resolve_retained_non_polymer_chemistry_source(
    structure: ProteinStructure,
    residue: ResidueSite,
    *,
    component_library: ComponentLibrary,
    evidence: RetainedNonPolymerChemistryEvidence | None = None,
    allow_rdkit_fallback: bool = True,
) -> RetainedNonPolymerChemistryResolution:
    """Resolve retained chemistry from its template, override, or fallback source."""

    residue_index = structure.constitution.residue_index(residue.residue_id)
    residue_geometry = structure.residue_geometry(residue_index)
    present_hydrogen_atom_names = tuple(
        atom_site.name for atom_site in residue.atom_sites if atom_site.is_hydrogen()
    )
    if evidence is not None:
        try:
            external_bonds = retained_non_polymer_evidence_external_bonds(
                evidence, structure
            )
            require_retained_non_polymer_evidence_stereochemistry(
                evidence,
                residue_geometry,
                hydrogen_attachments=tuple(
                    structure.topology.residue_hydrogen_anchor_by_name(
                        constitution=structure.constitution,
                        residue_index=residue_index,
                    ).items()
                ),
            )
            evidence_expected_hydrogen_atom_names = (
                retained_non_polymer_evidence_expected_hydrogen_atom_names(evidence)
            )
            heavy_bond_definitions = (
                retained_non_polymer_evidence_heavy_bond_definitions(evidence)
            )
            hydrogen_bond_definitions = (
                retained_non_polymer_evidence_hydrogen_bond_definitions(evidence)
            )
            heavy_atom_elements = retained_non_polymer_evidence_heavy_atom_elements(
                evidence
            )
            charge_requirements = (
                retained_non_polymer_evidence_formal_charge_requirements(evidence)
            )
        except (RuntimeError, ValueError) as error:
            return RetainedNonPolymerChemistryResolution(
                source=RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED,
                failure_reason=str(error),
            )
        return RetainedNonPolymerChemistryResolution(
            source=RetainedNonPolymerChemistryEvidenceSource.EXTERNAL_OVERRIDE,
            expected_hydrogen_atom_names=_preferred_expected_hydrogen_atom_names(
                expected_hydrogen_atom_names=evidence_expected_hydrogen_atom_names,
                present_hydrogen_atom_names=present_hydrogen_atom_names,
            ),
            heavy_bond_definitions=heavy_bond_definitions,
            hydrogen_bond_definitions=hydrogen_bond_definitions,
            heavy_atom_elements=heavy_atom_elements,
            formal_charge_requirements=charge_requirements,
            external_bonds=external_bonds,
        )

    template = component_library.get(residue.component_id)
    if template is not None and template.can_add_hydrogens():
        try:
            binding = RetainedTemplateBinding(
                structure, residue, template, component_library
            )
        except ValueError as error:
            return RetainedNonPolymerChemistryResolution(
                source=RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED,
                failure_reason=str(error),
            )
        template = binding.template
        expected_heavy_names = template.expected_heavy_atom_names()
        heavy_elements = ()
        if isinstance(template.hydrogen_semantics, IdealGeometryHydrogenSemantics):
            elements_by_name = {
                atom.atom_name: atom.element
                for atom in template.hydrogen_semantics.component.heavy_atoms()
            }
            heavy_elements = tuple(
                elements_by_name[name] for name in expected_heavy_names
            )
        expected_hydrogen_atom_names = _preferred_expected_hydrogen_atom_names(
            expected_hydrogen_atom_names=(
                _materializable_retained_non_polymer_template_hydrogen_atom_names(
                    residue,
                    template=template,
                )
            ),
            present_hydrogen_atom_names=present_hydrogen_atom_names,
        )
        return RetainedNonPolymerChemistryResolution(
            source=RetainedNonPolymerChemistryEvidenceSource.TEMPLATE,
            expected_hydrogen_atom_names=expected_hydrogen_atom_names,
            expected_heavy_atom_names=expected_heavy_names,
            heavy_atom_elements=heavy_elements,
            heavy_bond_definitions=template_heavy_bond_definitions_for_present_atoms(
                residue,
                template=template,
            ),
            hydrogen_bond_definitions=template_hydrogen_bond_definitions_for_names(
                template,
                hydrogen_atom_names=_materializable_retained_non_polymer_template_hydrogen_atom_names(
                    residue, template=template
                ),
            ),
            formal_charge_requirements=template.heavy_atom_charge_requirements(),
            external_bonds=binding.external_bonds,
        )

    if not allow_rdkit_fallback:
        return RetainedNonPolymerChemistryResolution(
            source=RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED,
            failure_reason="RDKit fallback is disabled",
        )

    try:
        RetainedCovalentContext(structure, residue.residue_id).require_admitted()
        fallback_inference_result = infer_retained_non_polymer_rdkit_fallback(
            residue,
            residue_geometry,
            formal_charge_by_atom_name=dict(
                structure.residue_formal_charge_by_atom_name(residue_index)
            ),
            hydrogen_position_by_name={
                atom_site.name: residue_geometry.position(atom_site.name)
                for atom_site in residue.atom_sites
                if atom_site.is_hydrogen()
            },
            hydrogen_anchor_by_name=structure.topology.residue_hydrogen_anchor_by_name(
                constitution=structure.constitution,
                residue_index=residue_index,
            ),
        )
    except (RuntimeError, ValueError) as error:
        return RetainedNonPolymerChemistryResolution(
            source=RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED,
            failure_reason=str(error),
        )

    return RetainedNonPolymerChemistryResolution(
        source=RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK,
        rdkit_backend_version=fallback_inference_result.rdkit_backend_version,
        expected_hydrogen_atom_names=(fallback_inference_result.hydrogen_atom_names),
        heavy_bond_definitions=fallback_inference_result.heavy_bond_definitions,
        hydrogen_bond_definitions=(fallback_inference_result.hydrogen_bond_definitions),
        hydrogen_name_projection_candidate_count=(
            fallback_inference_result.hydrogen_name_projection_candidate_count
        ),
        hydrogen_name_projection_candidate_limit=(
            fallback_inference_result.hydrogen_name_projection_candidate_limit
        ),
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

"""Retained non-polymer chemistry resolution facts."""

from collections import Counter
from collections.abc import Collection
from dataclasses import dataclass, replace
from enum import Enum
from itertools import combinations
from math import isfinite

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.idealized import IdealizedComponent
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.component.semantics import IdealGeometryHydrogenSemantics
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.component.topology import (
    template_heavy_bond_definitions_for_present_atoms,
    template_hydrogen_bond_definitions_for_names,
)
from protrepair.chemistry.inference.idealized_stereochemistry import (
    component_reference_hydrogen_bond_lengths,
    require_compatible_component_pose,
    require_compatible_component_reference,
)
from protrepair.chemistry.inference.retained_non_polymer_evidence import (
    require_retained_non_polymer_evidence_stereochemistry,
    retained_non_polymer_evidence_coordination_bonds,
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
from protrepair.chemistry.retained_non_polymer.coordination import (
    coordinated_component_preparation,
)
from protrepair.chemistry.retained_non_polymer.covalent_context import (
    RetainedCovalentContext,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.chemistry.retained_non_polymer.hydrogen_correspondence import (
    source_hydrogen_bond_length_is_compatible,
)
from protrepair.chemistry.retained_non_polymer.template_binding import (
    RetainedTemplateBinding,
)
from protrepair.chemistry.single_atom_inorganic import RetainedAtomicSite
from protrepair.errors import BondOrderConstraintError, RdkitUnavailableError
from protrepair.state.structure_axes import HydrogenCoverageState
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.disulfide import disulfide_bonded_cysteine_residue_ids
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import BondProvenance, BondRelationshipType


class RetainedNonPolymerChemistryEvidenceSource(str, Enum):
    """Chemistry source for one retained non-polymer resolution axis."""

    TEMPLATE = "template"
    EXTERNAL_OVERRIDE = "external_override"
    COORDINATION_PREPARATION = "coordination_preparation"
    ATOMIC_INVENTORY = "atomic_inventory"
    RDKIT_FALLBACK = "rdkit_fallback"
    UNRESOLVED = "unresolved"

    def is_resolved(self) -> bool:
        """Return whether this axis has concrete chemistry evidence."""

        return self is not RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED


class RetainedNonPolymerChemistryFailureKind(str, Enum):
    """Why an admitted retained-chemistry interpretation is unavailable."""

    NO_ADMITTED_PREPARATION = "no_admitted_preparation"
    CONTEXT_UNRESOLVED = "context_unresolved"
    EVIDENCE_REJECTED = "evidence_rejected"
    FALLBACK_DISABLED = "fallback_disabled"
    INFERENCE_FAILED = "inference_failed"
    REFERENCE_INVALID = "reference_invalid"
    REFERENCE_STATE_NOT_COVERED = "reference_state_not_covered"
    HYDROGEN_PARENT_UNRESOLVED = "hydrogen_parent_unresolved"
    SOURCE_CONFLICT = "source_conflict"
    BACKEND_UNAVAILABLE = "backend_unavailable"


@dataclass(frozen=True, slots=True)
class RetainedNonPolymerChemistryFailure:
    """Typed cause and human-readable detail for unresolved retained chemistry.

    Parameters
    ----------
    kind : RetainedNonPolymerChemistryFailureKind
        Machine-readable failure category.
    detail : str
        Explanation of the unavailable or rejected interpretation.
    """

    kind: RetainedNonPolymerChemistryFailureKind
    detail: str


class ReferenceHydrogenParentInterpretation(str, Enum):
    """Parent correspondence without an asserted connectivity relationship."""

    REFERENCE_NAME = "reference_name"
    COMPATIBLE_GEOMETRY = "compatible_geometry"


@dataclass(frozen=True, slots=True)
class RetainedReferenceHydrogenInventory:
    """Comparable current H inventory without preparation authority.

    Parameters
    ----------
    reference : IdealizedComponent
        Active chemical reference, independent of H placement support.
    residue_id : ResidueId
        Stable occurrence identity for these observations.
    hydrogen_parents : tuple[tuple[str, str, BondProvenance or ReferenceHydrogenParentInterpretation], ...]
        Current H name, heavy parent and either connection provenance or
        ReferenceHydrogenParentInterpretation. Naming and compatible geometry
        interpretations do not assert a source bond.

    Raises
    ------
    ValueError
        H identities repeat, parents are unknown, or multiplicities exceed the
        reference. Reference/pose conformance is checked by the observation owner;
        this value is not a transferable chemistry-validity token.
    TypeError
        A parent interpretation has no canonical evidence basis.
    """

    reference: IdealizedComponent
    residue_id: ResidueId
    hydrogen_parents: tuple[
        tuple[str, str, BondProvenance | ReferenceHydrogenParentInterpretation], ...
    ]

    def __post_init__(self) -> None:
        names = tuple(name for name, _, _ in self.hydrogen_parents)
        expected = Counter(
            self.reference.hydrogen_anchor_atom_name(atom.atom_name)
            for atom in self.reference.hydrogen_atoms()
        )
        observed = Counter(parent for _, parent, _ in self.hydrogen_parents)
        if None in expected or len(set(names)) != len(names):
            raise ValueError(
                "reference H inventory needs unique identities and parents"
            )
        if any(count > expected[parent] for parent, count in observed.items()):
            raise ValueError("observed H parents do not match the reference state")
        if set(names).intersection(self.reference.heavy_atom_names()):
            raise ValueError("reference H identities collide with heavy atoms")
        if any(
            not isinstance(
                basis, (BondProvenance, ReferenceHydrogenParentInterpretation)
            )
            for _, _, basis in self.hydrogen_parents
        ):
            raise TypeError(
                "reference H interpretation needs a canonical evidence basis"
            )

    def coverage_state(self) -> HydrogenCoverageState:
        """Return coverage of the validated reference inventory.

        Returns
        -------
        HydrogenCoverageState
            Complete, empty or proper-subset H coverage under this reference,
            without implying graph realization or generation support.
        """
        expected_count = len(self.reference.hydrogen_atoms())
        if len(self.hydrogen_parents) == expected_count:
            return HydrogenCoverageState.COMPLETE
        if not self.hydrogen_parents:
            return HydrogenCoverageState.NONE
        return HydrogenCoverageState.PARTIAL


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
    context_bonds: tuple[tuple[AtomRef, AtomRef], ...] = ()
    coordination_bonds: tuple[tuple[AtomRef, AtomRef], ...] = ()
    failure: RetainedNonPolymerChemistryFailure | None = None
    reference_requires_hydrogens: bool | None = None
    reference_hydrogen_inventory: RetainedReferenceHydrogenInventory | None = None
    rdkit_backend_version: str | None = None
    hydrogen_name_projection_candidate_count: int = 0
    hydrogen_name_projection_candidate_limit: int = 0

    @property
    def failure_reason(self) -> str:
        """Return the readable detail of the typed failure, if any."""

        return "" if self.failure is None else self.failure.detail

    @classmethod
    def from_unprepared_reference(
        cls, template: ResidueTemplate | None
    ) -> "RetainedNonPolymerChemistryResolution | None":
        """Refuse inference when the active reference lacks H preparation.

        Parameters
        ----------
        template : ResidueTemplate or None
            Active component definition, including caller-supplied replacements.

        Returns
        -------
        RetainedNonPolymerChemistryResolution or None
            Unresolved reference-backed requirement, or None when this guard
            does not apply. A reference is not itself a preparation capability.
        """
        if template is None or not template.has_unprepared_hydrogen_reference():
            return None
        reference = template.reference_component
        if reference is None:
            return None
        return cls(
            source=RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED,
            reference_requires_hydrogens=bool(reference.hydrogen_atoms()),
            failure=RetainedNonPolymerChemistryFailure(
                RetainedNonPolymerChemistryFailureKind.NO_ADMITTED_PREPARATION,
                "the active component reference has no admitted preparation; "
                "geometry-based chemistry inference is blocked; provide compatible "
                "explicit chemistry or a hydrogen-capable template",
            ),
        )

    @classmethod
    def observe_unprepared_reference(
        cls,
        template: ResidueTemplate | None,
        structure: ProteinStructure,
        residue: ResidueSite,
    ) -> "RetainedNonPolymerChemistryResolution | None":
        """Observe a reference inventory without resolving preparation.

        Parameters
        ----------
        template : ResidueTemplate or None
            Active component authority, including custom replacements.
        structure : ProteinStructure
            Exact snapshot supplying current and original observations.
        residue : ResidueSite
            Retained occurrence in that snapshot.

        Returns
        -------
        RetainedNonPolymerChemistryResolution or None
            Guarded preparation with a comparable inventory or typed cause;
            None when the reference guard does not apply.
        """
        guarded = cls.from_unprepared_reference(template)
        if guarded is None or template is None or template.reference_component is None:
            return guarded
        assessment = _assess_reference_hydrogens(
            structure, residue, template.reference_component
        )
        if isinstance(assessment, RetainedNonPolymerChemistryFailure):
            if not any(atom.is_hydrogen() for atom in residue.atom_sites):
                return guarded
            return replace(guarded, failure=assessment)
        return replace(guarded, reference_hydrogen_inventory=assessment)

    def has_complete_reference_hydrogens(self) -> bool:
        """Return whether read-only reference comparison established complete H.

        Returns
        -------
        bool
            Reference-verified completion, without generation or topology rights.
        """
        inventory = self.reference_hydrogen_inventory
        return (
            inventory is not None
            and inventory.coverage_state() is HydrogenCoverageState.COMPLETE
        )

    def can_generate_hydrogens(self) -> bool:
        """Return whether the selected chemistry admits H generation.

        Returns
        -------
        bool
            A resolved generation source exists. This does not promise that
            native coordinate placement will succeed.
        """

        return self.source.is_resolved()

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

    def has_context_bond_topology(self, structure: ProteinStructure) -> bool:
        """Check admitted attachments and partner-local product bonds.

        Parameters
        ----------
        structure : ProteinStructure
            Current canonical graph, observed without modification.

        Returns
        -------
        bool
            Whether every required bond is single and non-aromatic. External
            attachments may be covalent or disulfide; partner-local bonds must
            be covalent. Unresolved and missing bonds are not conformance.
        """
        for first, second in (*self.external_bonds, *self.context_bonds):
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
                or (
                    (first, second) in self.context_bonds
                    and bond.relationship_type is not BondRelationshipType.COVALENT
                )
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

    def has_coordination_topology(self, structure: ProteinStructure) -> bool:
        """Check every required typed physical coordination relationship.

        Parameters
        ----------
        structure : ProteinStructure
            Current graph, whose atom slots may differ from the evidence snapshot.

        Returns
        -------
        bool
            Whether all stable endpoint pairs exist as non-aromatic metal
            coordination without a covalent order. Covalent scaffold completeness
            does not substitute for this relationship check.
        """
        for first, second in self.coordination_bonds:
            indices = tuple(
                structure.constitution.resolve_atom_index(ref)
                for ref in (first, second)
            )
            if indices[0] is None or indices[1] is None:
                return False
            bond = structure.topology.bond_between(indices[0], indices[1])
            if (
                bond is None
                or bond.relationship_type is not BondRelationshipType.METAL_COORDINATION
                or bond.order is not None
                or bond.aromatic
            ):
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
        inventory = self.reference_hydrogen_inventory
        if inventory is not None and (
            resolved
            or self.reference_requires_hydrogens is None
            or self.reference_requires_hydrogens
            != bool(inventory.reference.hydrogen_atoms())
        ):
            raise ValueError("reference inventory must not grant preparation authority")
        if (
            self.source is RetainedNonPolymerChemistryEvidenceSource.ATOMIC_INVENTORY
            and (
                len(self.expected_heavy_atom_names) != 1
                or len(self.heavy_atom_elements) != 1
                or self.expected_hydrogen_atom_names
                or self.heavy_bond_definitions
                or self.hydrogen_bond_definitions
                or self.formal_charge_requirements
                or self.external_bonds
                or self.context_bonds
            )
        ):
            raise ValueError(
                "atomic inventory must describe one atom without molecular chemistry"
            )
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
            or self.context_bonds
            or self.coordination_bonds
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
        if resolved and self.failure is not None:
            raise ValueError(
                "resolved retained non-polymer chemistry must not carry a failure "
                "reason"
            )
        if resolved and self.reference_requires_hydrogens is not None:
            raise ValueError(
                "unprepared reference requirements cannot accompany resolved chemistry"
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
    if not resolution.source.is_resolved():
        reference = RetainedNonPolymerChemistryResolution.from_unprepared_reference(
            component_library.get(residue.component_id)
        )
        if reference is not None:
            resolution = replace(
                resolution,
                reference_requires_hydrogens=reference.reference_requires_hydrogens,
                failure=resolution.failure or reference.failure,
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
    preparation = None
    if evidence is None:
        try:
            preparation = coordinated_component_preparation(
                structure, residue, component_library
            )
        except (RuntimeError, ValueError) as error:
            return RetainedNonPolymerChemistryResolution(
                source=RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED,
                failure=RetainedNonPolymerChemistryFailure(
                    RetainedNonPolymerChemistryFailureKind.CONTEXT_UNRESOLVED,
                    str(error),
                ),
            )
        if preparation is not None:
            evidence = preparation.hydrogenation_template

    if evidence is not None:
        try:
            external_bonds = retained_non_polymer_evidence_external_bonds(
                evidence, structure
            )
            coordination_bonds = retained_non_polymer_evidence_coordination_bonds(
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
            heavy_bonds = retained_non_polymer_evidence_heavy_bond_definitions(
                evidence,
                bond_order_constraints=structure.topology.residue_source_heavy_bond_orders(
                    constitution=structure.constitution, residue_index=residue_index
                ),
            )
            hydrogen_bond_definitions = (
                retained_non_polymer_evidence_hydrogen_bond_definitions(evidence)
            )
            heavy_atom_elements = retained_non_polymer_evidence_heavy_atom_elements(
                evidence
            )
            charge_requirements = (
                preparation.formal_charge_requirements()
                if preparation is not None
                else retained_non_polymer_evidence_formal_charge_requirements(evidence)
            )
        except (RuntimeError, ValueError) as error:
            return RetainedNonPolymerChemistryResolution(
                source=RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED,
                failure=RetainedNonPolymerChemistryFailure(
                    RetainedNonPolymerChemistryFailureKind.EVIDENCE_REJECTED,
                    str(error),
                ),
            )
        return RetainedNonPolymerChemistryResolution(
            source=(
                RetainedNonPolymerChemistryEvidenceSource.COORDINATION_PREPARATION
                if preparation is not None
                else RetainedNonPolymerChemistryEvidenceSource.EXTERNAL_OVERRIDE
            ),
            expected_heavy_atom_names=evidence.heavy_atom_names,
            expected_hydrogen_atom_names=_preferred_expected_hydrogen_atom_names(
                expected_hydrogen_atom_names=evidence_expected_hydrogen_atom_names,
                present_hydrogen_atom_names=present_hydrogen_atom_names,
            ),
            heavy_bond_definitions=heavy_bonds,
            hydrogen_bond_definitions=hydrogen_bond_definitions,
            heavy_atom_elements=heavy_atom_elements,
            formal_charge_requirements=charge_requirements,
            external_bonds=external_bonds,
            coordination_bonds=coordination_bonds,
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
                failure=RetainedNonPolymerChemistryFailure(
                    RetainedNonPolymerChemistryFailureKind.CONTEXT_UNRESOLVED,
                    str(error),
                ),
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
            coordination_bonds=binding.coordination_bonds,
            context_bonds=()
            if binding.michael is None
            else binding.michael.context_bonds(),
        )

    atomic_site = RetainedAtomicSite(residue)
    try:
        atomic_site.require_unambiguous_hydrogenation()
        if atomic_site.is_monatomic_ion():
            coordination_bonds = atomic_site.require_source_context(structure)
            atom = residue.atom_sites[0]
            return RetainedNonPolymerChemistryResolution(
                source=RetainedNonPolymerChemistryEvidenceSource.ATOMIC_INVENTORY,
                expected_heavy_atom_names=(atom.name,),
                heavy_atom_elements=(atom.element,),
                coordination_bonds=coordination_bonds,
            )
    except ValueError as error:
        return RetainedNonPolymerChemistryResolution(
            source=RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED,
            failure=RetainedNonPolymerChemistryFailure(
                RetainedNonPolymerChemistryFailureKind.CONTEXT_UNRESOLVED,
                str(error),
            ),
        )

    guarded = RetainedNonPolymerChemistryResolution.observe_unprepared_reference(
        template, structure, residue
    )
    if guarded is not None:
        return guarded

    if not allow_rdkit_fallback:
        return RetainedNonPolymerChemistryResolution(
            source=RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED,
            failure=RetainedNonPolymerChemistryFailure(
                RetainedNonPolymerChemistryFailureKind.FALLBACK_DISABLED,
                "RDKit fallback is disabled",
            ),
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
            bond_order_constraints=structure.topology.residue_source_heavy_bond_orders(
                constitution=structure.constitution, residue_index=residue_index
            ),
            hydrogen_anchor_by_name=structure.topology.residue_hydrogen_anchor_by_name(
                constitution=structure.constitution,
                residue_index=residue_index,
            ),
        )
        coordination_bonds = fallback_inference_result.coordination_bonds(
            structure, residue.residue_id
        )
    except (RuntimeError, ValueError) as error:
        return RetainedNonPolymerChemistryResolution(
            source=RetainedNonPolymerChemistryEvidenceSource.UNRESOLVED,
            failure=RetainedNonPolymerChemistryFailure(
                RetainedNonPolymerChemistryFailureKind.INFERENCE_FAILED,
                str(error),
            ),
        )

    return RetainedNonPolymerChemistryResolution(
        source=RetainedNonPolymerChemistryEvidenceSource.RDKIT_FALLBACK,
        coordination_bonds=coordination_bonds,
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


def _assess_reference_hydrogens(
    structure: ProteinStructure,
    residue: ResidueSite,
    reference: IdealizedComponent,
) -> RetainedReferenceHydrogenInventory | RetainedNonPolymerChemistryFailure:
    try:
        require_compatible_component_reference(reference)
    except RdkitUnavailableError as error:
        return RetainedNonPolymerChemistryFailure(
            RetainedNonPolymerChemistryFailureKind.BACKEND_UNAVAILABLE, str(error)
        )
    except (ValueError, RuntimeError) as error:
        return RetainedNonPolymerChemistryFailure(
            RetainedNonPolymerChemistryFailureKind.REFERENCE_INVALID, str(error)
        )

    expected_heavy = {atom.atom_name: atom.element for atom in reference.heavy_atoms()}
    current_heavy = {
        atom.name: atom.element for atom in residue.atom_sites if not atom.is_hydrogen()
    }
    if current_heavy != expected_heavy:
        return RetainedNonPolymerChemistryFailure(
            RetainedNonPolymerChemistryFailureKind.REFERENCE_STATE_NOT_COVERED,
            "reference H validation requires the complete named heavy scaffold "
            "with matching elements",
        )

    constitution = structure.constitution
    incident = structure.topology.incident_bonds_by_atom()
    current_bonds = tuple(
        dict.fromkeys(
            bond
            for atom in residue.atom_sites
            for bond in incident.get(
                constitution.atom_index(AtomRef(residue.residue_id, atom.name)), ()
            )
        )
    )
    snapshots = [(constitution, structure.topology, current_bonds)]
    observation = structure.provenance.ingress.observation
    if observation is not None:
        original = observation.constitution.residue_or_ligand(residue.residue_id)
        if original is not None and (
            original.component_id != residue.component_id
            or any(
                not residue.has_atom_site(atom.name)
                or residue.atom_site(atom.name).element_identity
                != atom.element_identity
                for atom in original.atom_sites
            )
        ):
            return RetainedNonPolymerChemistryFailure(
                RetainedNonPolymerChemistryFailureKind.SOURCE_CONFLICT,
                "reference H validation cannot hide changed original atom identities",
            )
        if any(
            endpoint.atom_ref.residue_id == residue.residue_id
            for connection in observation.excluded_coordination_connections
            for endpoint in connection.endpoint_pair()
        ):
            return RetainedNonPolymerChemistryFailure(
                RetainedNonPolymerChemistryFailureKind.CONTEXT_UNRESOLVED,
                "read-only reference H validation cannot hide excluded coordination",
            )
        snapshots.append(
            (
                observation.constitution,
                observation.topology,
                observation.bonds_for_residue(residue.residue_id),
            )
        )

    reference_pairs = {
        frozenset((bond.atom_name_1, bond.atom_name_2))
        for bond in reference.heavy_bonds()
    }
    parents: dict[str, tuple[str, BondProvenance]] = {}
    orders: dict[tuple[str, str], int] = {}
    for snapshot_constitution, topology, bonds in snapshots:
        snapshot_residue = snapshot_constitution.residue_or_ligand(residue.residue_id)
        if snapshot_residue is None:
            continue
        for atom in snapshot_residue.atom_sites:
            charge = topology.formal_charge(
                snapshot_constitution.atom_index(AtomRef(residue.residue_id, atom.name))
            )
            expected = reference.atom(atom.name)
            expected_charge = (
                0
                if atom.is_hydrogen()
                else (None if expected is None else expected.formal_charge)
            )
            if charge is not None and charge != expected_charge:
                return RetainedNonPolymerChemistryFailure(
                    RetainedNonPolymerChemistryFailureKind.REFERENCE_STATE_NOT_COVERED,
                    f"known charge at {residue.residue_id.display_token()}.{atom.name} "
                    "differs from the represented reference state",
                )
        for bond in bonds:
            if bond.relationship_type is BondRelationshipType.HYDROGEN_BOND:
                continue
            first, second = (
                snapshot_constitution.atom_ref_at(index)
                for index in bond.endpoint_pair()
            )
            if (
                first.residue_id != residue.residue_id
                or second.residue_id != residue.residue_id
                or bond.relationship_type is BondRelationshipType.METAL_COORDINATION
            ):
                return RetainedNonPolymerChemistryFailure(
                    RetainedNonPolymerChemistryFailureKind.CONTEXT_UNRESOLVED,
                    "read-only reference H validation requires an isolated molecular "
                    f"context: {first.display_token()}--{second.display_token()}",
                )
            atoms = tuple(
                snapshot_constitution.atom_site_at(i) for i in bond.endpoint_pair()
            )
            if any(atom.is_hydrogen() for atom in atoms):
                if (
                    sum(atom.is_hydrogen() for atom in atoms) != 1
                    or bond.aromatic
                    or bond.order not in (None, 1)
                    or bond.relationship_type
                    not in {BondRelationshipType.COVALENT, BondRelationshipType.UNKNOWN}
                    or (
                        (
                            bond.order is None
                            or bond.relationship_type is BondRelationshipType.UNKNOWN
                        )
                        and bond.provenance is not BondProvenance.SOURCE_EXPLICIT
                    )
                ):
                    return RetainedNonPolymerChemistryFailure(
                        RetainedNonPolymerChemistryFailureKind.HYDROGEN_PARENT_UNRESOLVED,
                        "reference H validation requires one local heavy parent per H",
                    )
                hydrogen, parent = (
                    (first, second) if atoms[0].is_hydrogen() else (second, first)
                )
                previous = parents.get(hydrogen.atom_name)
                if previous is not None and previous[0] != parent.atom_name:
                    return RetainedNonPolymerChemistryFailure(
                        RetainedNonPolymerChemistryFailureKind.HYDROGEN_PARENT_UNRESOLVED,
                        f"conflicting H parents for {hydrogen.display_token()}",
                    )
                parents[hydrogen.atom_name] = (parent.atom_name, bond.provenance)
                continue
            if frozenset((first.atom_name, second.atom_name)) not in reference_pairs:
                return RetainedNonPolymerChemistryFailure(
                    RetainedNonPolymerChemistryFailureKind.REFERENCE_STATE_NOT_COVERED,
                    "current/source heavy connectivity differs from the reference",
                )
            if bond.order is not None:
                pair = (
                    (first.atom_name, second.atom_name)
                    if first.atom_name < second.atom_name
                    else (second.atom_name, first.atom_name)
                )
                if pair in orders and orders[pair] != bond.order:
                    return RetainedNonPolymerChemistryFailure(
                        RetainedNonPolymerChemistryFailureKind.SOURCE_CONFLICT,
                        "current and original heavy bond orders disagree",
                    )
                orders[pair] = bond.order

    geometry = structure.residue_geometry(
        constitution.residue_index(residue.residue_id)
    )
    expected_parent_counts = Counter(
        reference.hydrogen_anchor_atom_name(atom.atom_name)
        for atom in reference.hydrogen_atoms()
    )
    prepared_lengths: tuple[tuple[str, float], ...] | None = None
    interpreted = []
    for atom in residue.atom_sites:
        if not atom.is_hydrogen():
            continue
        connection = parents.get(atom.name)
        basis: BondProvenance | ReferenceHydrogenParentInterpretation
        if connection is None:
            position = geometry.position(atom.name)
            distances = {
                name: position.distance_to(geometry.position(name))
                for name in current_heavy
            }
            if any(not isfinite(distance) for distance in distances.values()):
                return RetainedNonPolymerChemistryFailure(
                    RetainedNonPolymerChemistryFailureKind.HYDROGEN_PARENT_UNRESOLVED,
                    "reference H parent comparison requires finite local distances "
                    f"for {residue.residue_id.display_token()}.{atom.name}",
                )
            nearest_distance = min(distances.values(), default=float("inf"))
            named_parent = reference.hydrogen_anchor_atom_name(atom.name)
            if (
                named_parent in distances
                and distances[named_parent] == nearest_distance
            ):
                parent = named_parent
                basis = ReferenceHydrogenParentInterpretation.REFERENCE_NAME
            else:
                closest = tuple(
                    name
                    for name, distance in distances.items()
                    if distance == nearest_distance
                )
                if len(closest) != 1:
                    return RetainedNonPolymerChemistryFailure(
                        RetainedNonPolymerChemistryFailureKind.HYDROGEN_PARENT_UNRESOLVED,
                        "no unambiguous local heavy parent for "
                        f"{residue.residue_id.display_token()}.{atom.name}",
                    )
                parent = closest[0]
                if not expected_parent_counts[parent]:
                    return RetainedNonPolymerChemistryFailure(
                        RetainedNonPolymerChemistryFailureKind.REFERENCE_STATE_NOT_COVERED,
                        f"observed H at {residue.residue_id.display_token()}.{parent} "
                        "is outside the represented reference state",
                    )
                if prepared_lengths is None:
                    try:
                        prepared_lengths = component_reference_hydrogen_bond_lengths(
                            reference, geometry, bond_order_constraints=orders
                        )
                    except RdkitUnavailableError as error:
                        return RetainedNonPolymerChemistryFailure(
                            RetainedNonPolymerChemistryFailureKind.BACKEND_UNAVAILABLE,
                            str(error),
                        )
                    except BondOrderConstraintError as error:
                        return RetainedNonPolymerChemistryFailure(
                            RetainedNonPolymerChemistryFailureKind.SOURCE_CONFLICT,
                            str(error),
                        )
                    except (ValueError, RuntimeError) as error:
                        return RetainedNonPolymerChemistryFailure(
                            RetainedNonPolymerChemistryFailureKind.HYDROGEN_PARENT_UNRESOLVED,
                            f"reference H correspondence is unavailable: {error}",
                        )
                if not any(
                    source_hydrogen_bond_length_is_compatible(nearest_distance, length)
                    for anchor, length in prepared_lengths
                    if anchor == parent
                ):
                    return RetainedNonPolymerChemistryFailure(
                        RetainedNonPolymerChemistryFailureKind.HYDROGEN_PARENT_UNRESOLVED,
                        f"source H {residue.residue_id.display_token()}.{atom.name} "
                        "has no compatible prepared-reference parent distance",
                    )
                basis = ReferenceHydrogenParentInterpretation.COMPATIBLE_GEOMETRY
        else:
            parent, basis = connection
        interpreted.append((atom.name, parent, basis))
    try:
        inventory = RetainedReferenceHydrogenInventory(
            reference, residue.residue_id, tuple(interpreted)
        )
    except ValueError as error:
        return RetainedNonPolymerChemistryFailure(
            RetainedNonPolymerChemistryFailureKind.REFERENCE_STATE_NOT_COVERED,
            str(error),
        )
    try:
        require_compatible_component_pose(
            reference,
            geometry,
            hydrogen_attachments=tuple(
                (name, parent) for name, parent, _ in interpreted
            ),
            bond_order_constraints=orders,
        )
    except RdkitUnavailableError as error:
        return RetainedNonPolymerChemistryFailure(
            RetainedNonPolymerChemistryFailureKind.BACKEND_UNAVAILABLE, str(error)
        )
    except (ValueError, RuntimeError) as error:
        return RetainedNonPolymerChemistryFailure(
            RetainedNonPolymerChemistryFailureKind.SOURCE_CONFLICT, str(error)
        )
    return inventory


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

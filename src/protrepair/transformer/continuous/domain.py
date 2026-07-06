"""Canonical region/problem models for continuous local-relaxation execution."""

from collections.abc import Mapping
from dataclasses import dataclass

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.errors import RefinementError
from protrepair.state.retained_non_polymer_chemistry import (
    RetainedNonPolymerChemistryResolution,
)
from protrepair.structure.constitution import AtomSite as ConstitutionAtomSite
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import AtomGeometry, ResidueGeometry
from protrepair.structure.labels import ResidueId
from protrepair.structure.slots import AtomIndex, ResidueIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.atom_input import AtomInput
from protrepair.transformer.continuous.bonds import (
    PlannedBond,
    directly_bonded_context_residue_indices,
    inter_residue_bonds,
    plan_continuous_region_bonds,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationSettings
from protrepair.transformer.continuous.spatial import radius_context_residue_indices
from protrepair.transformer.continuous.support import (
    LocalBondPlanningSupportResolution,
    resolve_local_bond_planning_support,
)


@dataclass(frozen=True, slots=True)
class ContinuousRelaxationRegion:
    """Canonical included region for one continuous-relaxation problem."""

    snapshot: ProteinStructureSnapshot
    atom_input: AtomInput
    movable_atom_indices: tuple[AtomIndex, ...]
    fixed_context_atom_indices: tuple[AtomIndex, ...]
    included_residue_indices: tuple[ResidueIndex, ...]
    inter_residue_bonds: tuple[PlannedBond, ...]

    @classmethod
    def from_inputs(
        cls,
        snapshot: ProteinStructureSnapshot,
        atom_input: AtomInput,
        *,
        context_radius_angstrom: float,
    ) -> "ContinuousRelaxationRegion":
        """Build one canonical included region from canonical inputs."""

        constitution = snapshot.structure.constitution
        ordered_atom_indices = tuple(
            AtomIndex(atom_index) for atom_index in range(len(constitution.atom_slots))
        )
        movable_atom_index_set = set(atom_input.atom_indices)
        movable_atom_indices = tuple(
            atom_index
            for atom_index in ordered_atom_indices
            if atom_index in movable_atom_index_set
        )
        if not movable_atom_indices:
            raise RefinementError(
                "continuous relaxation requires at least one movable atom"
            )

        planned_inter_residue_bonds = inter_residue_bonds(snapshot)
        selected_residue_index_set = {
            constitution.residue_index_for_atom_index(atom_index)
            for atom_index in movable_atom_indices
        }
        included_residue_index_set = (
            set(selected_residue_index_set)
            | radius_context_residue_indices(
                snapshot,
                movable_atom_index_set=movable_atom_index_set,
                selected_residue_index_set=selected_residue_index_set,
                context_radius_angstrom=context_radius_angstrom,
            )
            | directly_bonded_context_residue_indices(
                constitution,
                planned_inter_residue_bonds,
                selected_residue_index_set=selected_residue_index_set,
            )
        )
        included_residue_indices = tuple(
            ResidueIndex(residue_index)
            for residue_index, residue_site in enumerate(
                snapshot.structure.constitution.residue_slots
            )
            if ResidueIndex(residue_index) in included_residue_index_set
        )
        fixed_context_atom_indices = tuple(
            atom_index
            for atom_index in ordered_atom_indices
            if constitution.residue_index_for_atom_index(atom_index)
            in included_residue_index_set
            and atom_index not in movable_atom_index_set
        )
        return cls(
            snapshot=snapshot,
            atom_input=atom_input,
            movable_atom_indices=movable_atom_indices,
            fixed_context_atom_indices=fixed_context_atom_indices,
            included_residue_indices=included_residue_indices,
            inter_residue_bonds=planned_inter_residue_bonds,
        )

    def included_atom_indices(self) -> tuple[AtomIndex, ...]:
        """Return movable atoms followed by fixed context atoms."""

        return (*self.movable_atom_indices, *self.fixed_context_atom_indices)

    def atom_site(self, atom_index: AtomIndex) -> ConstitutionAtomSite:
        """Return one constitution atom site referenced by the included region."""

        return self.snapshot.structure.constitution.atom_site_at(atom_index)

    def atom_geometry(self, atom_index: AtomIndex) -> AtomGeometry:
        """Return one atom-geometry payload referenced by the included region."""

        return self.snapshot.structure.geometry.atom_geometry(atom_index)

    def formal_charge(self, atom_index: AtomIndex) -> int | None:
        """Return formal-charge payload for one included atom slot."""

        return self.snapshot.structure.topology.formal_charge(atom_index)

    def residue_site(self, residue_index: ResidueIndex) -> ResidueSite:
        """Return one included residue site by slot index."""

        return self.snapshot.structure.constitution.residue_site_at(residue_index)

    def residue_geometry(self, residue_index: ResidueIndex) -> ResidueGeometry:
        """Return one included residue geometry by slot index."""

        return self.snapshot.structure.residue_geometry(residue_index)

    def included_residue_sites(self) -> tuple[ResidueSite, ...]:
        """Return included residue sites in structure order."""

        return tuple(
            self.residue_site(residue_index)
            for residue_index in self.included_residue_indices
        )

    def require_local_bond_planning_support(
        self,
        component_library: ComponentLibrary,
        *,
        allow_retained_non_polymer_rdkit_fallback: bool = True,
        retained_non_polymer_chemistry_evidence: tuple[
            RetainedNonPolymerChemistryEvidence,
            ...,
        ] = (),
        retained_non_polymer_chemistry_resolution_by_residue_id: Mapping[
            ResidueId,
            RetainedNonPolymerChemistryResolution,
        ]
        | None = None,
    ) -> dict[ResidueIndex, LocalBondPlanningSupportResolution]:
        """Raise when any included residue lacks editable or passive bond support."""

        support_by_residue_index = self.local_bond_planning_support_by_residue_index(
            component_library,
            allow_retained_non_polymer_rdkit_fallback=(
                allow_retained_non_polymer_rdkit_fallback
            ),
            retained_non_polymer_chemistry_evidence=(
                retained_non_polymer_chemistry_evidence
            ),
            retained_non_polymer_chemistry_resolution_by_residue_id=(
                retained_non_polymer_chemistry_resolution_by_residue_id
            ),
        )

        blocker_messages = sorted(
            {
                support_resolution.blocker_message
                for support_resolution in support_by_residue_index.values()
                if not support_resolution.supports_local_bond_planning()
                and support_resolution.blocker_message
            }
        )
        if blocker_messages:
            raise RefinementError("; ".join(blocker_messages))

        missing_component_tokens = sorted(
            {
                f"{residue_site.component_id} "
                f"({residue_site.residue_id.display_token()})"
                for residue_index, support_resolution in (
                    support_by_residue_index.items()
                )
                if not support_resolution.supports_local_bond_planning()
                for residue_site in (self.residue_site(residue_index),)
            }
        )
        if missing_component_tokens:
            raise RefinementError(
                "continuous relaxation requires local bond-planning support for all "
                "included residues: "
                f"{', '.join(missing_component_tokens)}"
            )

        return support_by_residue_index

    def local_bond_planning_support_by_residue_index(
        self,
        component_library: ComponentLibrary,
        *,
        allow_retained_non_polymer_rdkit_fallback: bool = True,
        retained_non_polymer_chemistry_evidence: tuple[
            RetainedNonPolymerChemistryEvidence,
            ...,
        ] = (),
        retained_non_polymer_chemistry_resolution_by_residue_id: Mapping[
            ResidueId,
            RetainedNonPolymerChemistryResolution,
        ]
        | None = None,
    ) -> dict[ResidueIndex, LocalBondPlanningSupportResolution]:
        """Resolve local bond-planning support for each included residue."""

        return {
            residue_index: resolve_local_bond_planning_support(
                self.snapshot,
                residue_index,
                self.residue_site(residue_index),
                movable_atom_indices=self.movable_atom_indices,
                component_library=component_library,
                allow_retained_non_polymer_rdkit_fallback=(
                    allow_retained_non_polymer_rdkit_fallback
                ),
                retained_non_polymer_chemistry_evidence=(
                    retained_non_polymer_chemistry_evidence
                ),
                retained_non_polymer_chemistry_resolution_by_residue_id=(
                    retained_non_polymer_chemistry_resolution_by_residue_id
                ),
            )
            for residue_index in self.included_residue_indices
        }


@dataclass(frozen=True, slots=True)
class ContinuousRelaxationProblem:
    """Resolved continuous-relaxation problem over one included region."""

    region: ContinuousRelaxationRegion
    spec: ContinuousRelaxationSettings
    bonds: tuple[PlannedBond, ...]

    @classmethod
    def from_inputs(
        cls,
        snapshot: ProteinStructureSnapshot,
        atom_input: AtomInput,
        *,
        spec: ContinuousRelaxationSettings,
        component_library: ComponentLibrary,
        allow_retained_non_polymer_rdkit_fallback: bool = True,
        retained_non_polymer_chemistry_evidence: tuple[
            RetainedNonPolymerChemistryEvidence,
            ...,
        ] = (),
        retained_non_polymer_chemistry_resolution_by_residue_id: Mapping[
            ResidueId,
            RetainedNonPolymerChemistryResolution,
        ]
        | None = None,
    ) -> "ContinuousRelaxationProblem":
        """Build one continuous-relaxation problem from canonical inputs."""

        region = ContinuousRelaxationRegion.from_inputs(
            snapshot,
            atom_input,
            context_radius_angstrom=spec.context_radius_angstrom,
        )
        support_by_residue_index = region.require_local_bond_planning_support(
            component_library,
            allow_retained_non_polymer_rdkit_fallback=(
                allow_retained_non_polymer_rdkit_fallback
            ),
            retained_non_polymer_chemistry_evidence=(
                retained_non_polymer_chemistry_evidence
            ),
            retained_non_polymer_chemistry_resolution_by_residue_id=(
                retained_non_polymer_chemistry_resolution_by_residue_id
            ),
        )
        return cls(
            region=region,
            spec=spec,
            bonds=plan_continuous_region_bonds(
                region,
                component_library,
                support_by_residue_index=support_by_residue_index,
            ),
        )

"""Rich domain models for residue-local completion transforms."""

from collections.abc import Collection, Mapping
from dataclasses import dataclass
from typing import TypeAlias

from protrepair.chemistry import IdealizedComponent, ResidueTemplate
from protrepair.scope import ResidueSetScope
from protrepair.structure.constitution import AtomSite, ResidueSite
from protrepair.structure.geometry import AtomGeometry, ResidueGeometry
from protrepair.structure.labels import ResidueId
from protrepair.structure.slots import ResidueIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.atom_input import AtomInput, AtomInputBasis
from protrepair.transformer.completion.policies import OrphanFragmentPolicy
from protrepair.transformer.completion.shared.patch import (
    OrderedAtomPatch,
)
from protrepair.transformer.completion.targets import (
    BACKBONE_ATOM_NAMES,
    TERMINAL_EXCLUDED_ATOM_NAMES,
    missing_atoms_for_completion,
)


@dataclass(frozen=True, slots=True)
class CompletionResiduePayload:
    """Facet-native residue payload for completion execution paths."""

    residue_site: ResidueSite
    residue_geometry: ResidueGeometry
    formal_charge_by_atom_name: tuple[tuple[str, int | None], ...] = ()

    @property
    def component_id(self) -> str:
        """Return the canonical residue component identifier."""

        return self.residue_site.component_id

    @property
    def residue_id(self) -> ResidueId:
        """Return the canonical residue identifier."""

        return self.residue_site.residue_id

    @property
    def is_hetero(self) -> bool:
        """Return whether this residue payload is hetero."""

        return self.residue_site.is_hetero

    @property
    def atom_sites(self) -> tuple[AtomSite, ...]:
        """Return atom sites in residue order."""

        return self.residue_site.atom_sites

    def atom_names(self) -> tuple[str, ...]:
        """Return atom names in residue order."""

        return self.residue_site.atom_site_names()

    def has_atom(self, atom_name: str) -> bool:
        """Return whether this payload contains the requested atom."""

        return self.residue_site.has_atom_site(atom_name)

    def has_atom_site(self, atom_name: str) -> bool:
        """Return whether this payload contains the requested atom site."""

        return self.residue_site.has_atom_site(atom_name)

    def atom_site(self, atom_name: str) -> AtomSite:
        """Return the requested residue-local atom site."""

        return self.residue_site.atom_site(atom_name)

    def atom_geometry(self, atom_name: str) -> AtomGeometry:
        """Return the requested residue-local atom geometry."""

        return self.residue_geometry.atom_geometry(atom_name)

    def position(self, atom_name: str):
        """Return the requested residue-local atom position."""

        return self.residue_geometry.position(atom_name)

    def formal_charge(self, atom_name: str) -> int | None:
        """Return formal-charge payload for one residue-local atom."""

        return dict(self.formal_charge_by_atom_name).get(atom_name.strip().upper())

    def with_component_id(self, component_id: str) -> "CompletionResiduePayload":
        """Return a copy with an updated residue component identifier."""

        return type(self)(
            residue_site=self.residue_site.with_component_id(component_id),
            residue_geometry=self.residue_geometry,
            formal_charge_by_atom_name=self.formal_charge_by_atom_name,
        )

    def renamed_atoms(
        self,
        atom_names_by_current_name: Mapping[str, str],
    ) -> "CompletionResiduePayload":
        """Return a copy with selected atom names renamed across all facets."""

        normalized_name_map = {
            atom_name.strip().upper(): renamed_atom_name.strip().upper()
            for atom_name, renamed_atom_name in atom_names_by_current_name.items()
        }
        if not normalized_name_map:
            return self

        renamed_residue_site = ResidueSite(
            component_id=self.component_id,
            residue_id=self.residue_id,
            atom_sites=tuple(
                atom_site.with_name(
                    normalized_name_map.get(atom_site.name, atom_site.name)
                )
                for atom_site in self.atom_sites
            ),
            is_hetero=self.is_hetero,
        )
        renamed_residue_geometry = ResidueGeometry(
            atoms_by_name={
                normalized_name_map.get(atom_name, atom_name): atom_geometry
                for atom_name, atom_geometry in (
                    self.residue_geometry.atoms_by_name.items()
                )
            },
        )
        renamed_formal_charge_by_atom_name = tuple(
            (
                normalized_name_map.get(atom_name, atom_name),
                formal_charge,
            )
            for atom_name, formal_charge in self.formal_charge_by_atom_name
        )
        return type(self)(
            residue_site=renamed_residue_site,
            residue_geometry=renamed_residue_geometry,
            formal_charge_by_atom_name=renamed_formal_charge_by_atom_name,
        )

    def without_atom_sites(
        self,
        atom_names: Collection[str],
    ) -> "CompletionResiduePayload":
        """Return a copy without the requested atom payload."""

        normalized_atom_names = {atom_name.strip().upper() for atom_name in atom_names}
        return type(self)(
            residue_site=self.residue_site.without_atom_sites(normalized_atom_names),
            residue_geometry=self.residue_geometry.without_atoms(normalized_atom_names),
            formal_charge_by_atom_name=tuple(
                (atom_name, formal_charge)
                for atom_name, formal_charge in self.formal_charge_by_atom_name
                if atom_name not in normalized_atom_names
            ),
        )

    def reordered(
        self,
        ordered_atom_names: Collection[str],
    ) -> "CompletionResiduePayload":
        """Return a copy projected into the requested atom order."""

        return type(self)(
            residue_site=self.residue_site.reordered_atom_sites(
                tuple(ordered_atom_names)
            ),
            residue_geometry=self.residue_geometry,
            formal_charge_by_atom_name=tuple(
                (atom_name, formal_charge)
                for atom_name, formal_charge in self.formal_charge_by_atom_name
                if atom_name
                in self.residue_site.reordered_atom_sites(
                    tuple(ordered_atom_names)
                ).atom_site_names()
            ),
        )

    def with_atom_payload(
        self,
        atom_site: AtomSite,
        *,
        atom_geometry: AtomGeometry,
        formal_charge: int | None = None,
    ) -> "CompletionResiduePayload":
        """Return a copy with one atom payload added or replaced."""

        updated_residue_site = self.residue_site.with_atom_site(atom_site)
        updated_residue_geometry = self.residue_geometry.with_atom_geometry(
            atom_site.name,
            atom_geometry,
        )
        updated_formal_charge_by_atom_name = {
            atom_name: existing_formal_charge
            for atom_name, existing_formal_charge in self.formal_charge_by_atom_name
        }
        if formal_charge is None:
            updated_formal_charge_by_atom_name.pop(atom_site.name, None)
        else:
            updated_formal_charge_by_atom_name[atom_site.name] = formal_charge

        return type(self)(
            residue_site=updated_residue_site,
            residue_geometry=updated_residue_geometry,
            formal_charge_by_atom_name=tuple(
                (
                    current_atom_name,
                    updated_formal_charge_by_atom_name[current_atom_name],
                )
                for current_atom_name in updated_residue_site.atom_site_names()
                if current_atom_name in updated_formal_charge_by_atom_name
            ),
        )

    def apply_patch(
        self,
        patch: OrderedAtomPatch,
    ) -> "CompletionResiduePayload":
        """Return a copy with one ordered patch materialized onto this payload."""

        residue_site, residue_geometry, formal_charge_by_atom_name = (
            patch.materialize_on_payload(
                self.residue_site,
                residue_geometry=self.residue_geometry,
                formal_charge_by_atom_name=self.formal_charge_by_atom_name,
            )
        )
        return type(self)(
            residue_site=residue_site,
            residue_geometry=residue_geometry,
            formal_charge_by_atom_name=formal_charge_by_atom_name,
        )


@dataclass(frozen=True, slots=True)
class ResidueBackboneNeighborhood:
    """One residue-local backbone neighborhood for coordinate completion."""

    previous_residue_index: ResidueIndex | None
    next_residue_index: ResidueIndex | None

    @classmethod
    def from_linear_residue_slots(
        cls,
        residue_index: ResidueIndex,
        *,
        residue_count: int,
    ) -> "ResidueBackboneNeighborhood":
        """Build a non-wrapping neighborhood for one linear residue sequence."""

        if residue_count < 0:
            raise ValueError("residue_count must be non-negative")
        if residue_index.value >= residue_count:
            raise ValueError("residue_index must be within residue_count")

        previous_residue_index = (
            ResidueIndex(residue_index.value - 1)
            if residue_index.value > 0
            else None
        )
        next_residue_index = (
            ResidueIndex(residue_index.value + 1)
            if residue_index.value + 1 < residue_count
            else None
        )
        return cls(
            previous_residue_index=previous_residue_index,
            next_residue_index=next_residue_index,
        )

    def resolve(
        self,
        snapshot: ProteinStructureSnapshot,
    ) -> tuple[CompletionResiduePayload | None, CompletionResiduePayload | None]:
        """Resolve the previous and next residues from one snapshot."""

        return (
            (
                None
                if self.previous_residue_index is None
                else resolve_completion_residue_payload(
                    snapshot,
                    self.previous_residue_index,
                )
            ),
            (
                None
                if self.next_residue_index is None
                else resolve_completion_residue_payload(
                    snapshot,
                    self.next_residue_index,
                )
            ),
        )


@dataclass(frozen=True, slots=True)
class LocalFramePlacementDirective:
    """Place idealized atoms by growing local frames layer by layer."""

    idealized_component: IdealizedComponent
    reconstruction_layers: tuple[tuple[str, ...], ...]
    target_atom_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ResidueFramePlacementDirective:
    """Place one reconstruction subtree from a retained residue frame."""

    idealized_component: IdealizedComponent
    retained_fragment_atom_names: tuple[str, ...]
    reconstruction_layers: tuple[tuple[str, ...], ...]
    target_atom_names: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MseBridgePlacementDirective:
    """Repair the MSE selenium bridge from the retained sulfur bridge frame."""

    original_payload: CompletionResiduePayload
    idealized_component: IdealizedComponent
    target_atom_names: tuple[str, ...]
    orphan_fragment_policy: OrphanFragmentPolicy


RigidComponentPlacementDirective: TypeAlias = (
    LocalFramePlacementDirective
    | ResidueFramePlacementDirective
    | MseBridgePlacementDirective
)
"""Closed directive variants for rigid-frame component atom placement."""


@dataclass(frozen=True, slots=True)
class ResidueCompletionSite:
    """Canonical residue-local completion site with dynamic snapshot behavior."""

    residue_index: ResidueIndex
    template: ResidueTemplate
    original_payload: CompletionResiduePayload
    reference_payload: CompletionResiduePayload | None = None
    neighborhood: ResidueBackboneNeighborhood | None = None

    def is_selected_by(
        self,
        snapshot: ProteinStructureSnapshot,
        atom_input: AtomInput,
    ) -> bool:
        """Return whether the selected domain includes this completion site."""

        selected_atom_index_set = frozenset(atom_input.atom_indices)
        return any(
            atom_index in selected_atom_index_set
            for atom_index in (
                snapshot.structure.constitution.atom_indices_for_residue_index(
                    self.residue_index
                )
            )
        )

    def payload(
        self,
        snapshot: ProteinStructureSnapshot,
    ) -> CompletionResiduePayload | None:
        """Resolve the current residue payload from one snapshot."""

        return resolve_completion_residue_payload(snapshot, self.residue_index)

    def missing_atom_names(
        self,
        snapshot: ProteinStructureSnapshot,
    ) -> tuple[str, ...]:
        """Return the currently missing completion targets for this site."""

        payload = self.payload(snapshot)
        if payload is None:
            return ()

        return missing_atoms_for_completion(payload.residue_site, self.template)

    def atom_input(
        self,
        snapshot: ProteinStructureSnapshot,
    ) -> AtomInput:
        """Derive one residuewise atom domain for this site from one snapshot."""

        payload = self.payload(snapshot)
        if payload is None:
            raise ValueError(
                "completion site residue "
                f"{self.original_payload.residue_id.display_token()} is missing"
            )

        return AtomInput(
            atom_indices=tuple(
                snapshot.structure.constitution.atom_indices_for_residue_index(
                    self.residue_index
                )
            ),
            basis=AtomInputBasis.RESIDUEWISE,
            selected_scope=ResidueSetScope(residue_ids=(payload.residue_id,)),
        )

    def neighbor_payloads(
        self,
        snapshot: ProteinStructureSnapshot,
    ) -> tuple[CompletionResiduePayload | None, CompletionResiduePayload | None]:
        """Resolve the backbone-neighbor payloads for one snapshot."""

        if self.neighborhood is None:
            return None, None

        return self.neighborhood.resolve(snapshot)

    def reference_graft_patch(
        self,
        snapshot: ProteinStructureSnapshot,
    ) -> OrderedAtomPatch | None:
        """Return one residue-local reference graft patch when available."""

        payload = self.payload(snapshot)
        if payload is None or self.reference_payload is None:
            return None

        if self.reference_payload.component_id != payload.component_id:
            return None

        guided_atom_names = tuple(
            atom_name
            for atom_name in self.missing_atom_names(snapshot)
            if atom_name not in BACKBONE_ATOM_NAMES
            and atom_name not in TERMINAL_EXCLUDED_ATOM_NAMES
            and self.reference_payload.has_atom_site(atom_name)
        )
        if not guided_atom_names:
            return None

        return OrderedAtomPatch.from_residue_payload(
            ResidueSite(
                component_id=self.reference_payload.component_id,
                residue_id=payload.residue_id,
                atom_sites=tuple(
                    self.reference_payload.atom_site(atom_name)
                    for atom_name in guided_atom_names
                ),
                is_hetero=self.reference_payload.is_hetero,
            ),
            residue_geometry=ResidueGeometry(
                atoms_by_name={
                    atom_name: self.reference_payload.atom_geometry(atom_name)
                    for atom_name in guided_atom_names
                },
            ),
        )

    def apply_patch(
        self,
        snapshot: ProteinStructureSnapshot,
        patch: OrderedAtomPatch,
    ) -> ProteinStructureSnapshot:
        """Materialize one completion patch onto this site within one snapshot."""

        return patch.apply_to_snapshot(snapshot, self.residue_index)

    def replace_residue(
        self,
        snapshot: ProteinStructureSnapshot,
        residue_payload: CompletionResiduePayload,
    ) -> ProteinStructureSnapshot:
        """Replace this site residue within one snapshot and re-derive state."""

        return snapshot.with_updated_residue_facets(
            residue_payload.residue_site,
            residue_geometry=residue_payload.residue_geometry,
            formal_charge_by_atom_name=residue_payload.formal_charge_by_atom_name,
        )


def resolve_completion_residue_payload(
    snapshot: ProteinStructureSnapshot,
    residue_index: ResidueIndex,
) -> CompletionResiduePayload | None:
    """Return one completion payload for the requested residue slot if present."""

    if residue_index.value >= len(snapshot.structure.constitution.residue_slots):
        return None

    residue_site = snapshot.structure.constitution.residue_site_at(residue_index)
    residue_geometry = snapshot.structure.residue_geometry(residue_index)
    return CompletionResiduePayload(
        residue_site=residue_site,
        residue_geometry=residue_geometry,
        formal_charge_by_atom_name=(
            snapshot.structure.residue_formal_charge_by_atom_name(residue_index)
        ),
    )

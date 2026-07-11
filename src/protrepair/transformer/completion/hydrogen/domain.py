"""Rich domain models for residue-local hydrogen completion."""

from collections.abc import Sequence
from dataclasses import dataclass

from protrepair.chemistry import ResidueTemplate
from protrepair.geometry import Vec3
from protrepair.scope import ResidueSetScope
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.labels import ResidueId
from protrepair.structure.slots import ResidueIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.atom_input import AtomInput, AtomInputBasis
from protrepair.transformer.completion.hydrogen.rotatable import (
    RotatableHydrogenEnvironment,
    RotatableHydrogenSearch,
    build_rotatable_hydrogen_environments,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch


@dataclass(frozen=True, slots=True)
class HydrogenCompletionEnvironment:
    """Chain-local environment shared by residue-local hydrogen sites."""

    rotatable_environments: tuple[RotatableHydrogenEnvironment, ...]
    disulfide_bonded_residue_ids: frozenset[ResidueId]

    def __post_init__(self) -> None:
        residue_ids = tuple(
            environment.residue_id for environment in self.rotatable_environments
        )
        if len(residue_ids) != len(set(residue_ids)):
            raise ValueError(
                "hydrogen completion environments must not repeat residue identities"
            )
        disulfide_residue_ids = frozenset(self.disulfide_bonded_residue_ids)
        if not disulfide_residue_ids.issubset(residue_ids):
            raise ValueError(
                "hydrogen completion disulfide ids must belong to the environment"
            )
        object.__setattr__(
            self,
            "disulfide_bonded_residue_ids",
            disulfide_residue_ids,
        )

    @classmethod
    def from_payloads(
        cls,
        residues: tuple[CompletionResiduePayload, ...],
        *,
        templates: Sequence[ResidueTemplate | None],
        disulfide_bonded_residue_ids: frozenset[ResidueId],
    ) -> "HydrogenCompletionEnvironment":
        """Build the shared hydrogen environment for one chain residue tuple."""

        return cls(
            rotatable_environments=build_rotatable_hydrogen_environments(
                residues=residues,
                templates=templates,
            ),
            disulfide_bonded_residue_ids=disulfide_bonded_residue_ids,
        )

    def optimize_rotatable(
        self,
        residue_index: ResidueIndex,
        search: RotatableHydrogenSearch,
    ) -> Vec3:
        """Return the optimized rotatable-hydrogen coordinate for one residue."""

        return search.optimized_coordinate(
            self.rotatable_environments[residue_index.value]
        )

    def is_disulfide_bonded(
        self,
        residue_index: ResidueIndex,
    ) -> bool:
        """Return whether one environment residue has canonical disulfide topology."""

        residue_id = self.rotatable_environments[residue_index.value].residue_id
        return residue_id in self.disulfide_bonded_residue_ids


@dataclass(frozen=True, slots=True)
class HydrogenResidueSite:
    """Canonical residue-local site for hydrogen placement and propagation."""

    residue_index: ResidueIndex
    template: ResidueTemplate
    environment: HydrogenCompletionEnvironment
    backbone_family_component_id: str | None = None
    next_residue_index: ResidueIndex | None = None

    def is_selected_by(
        self,
        snapshot: ProteinStructureSnapshot,
        atom_input: AtomInput,
    ) -> bool:
        """Return whether the selected domain includes this hydrogen site."""

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

        residue_constitution = self.residue_constitution(snapshot)
        residue_geometry = self.residue_geometry(snapshot)
        if residue_constitution is None or residue_geometry is None:
            return None

        return CompletionResiduePayload(
            residue_site=residue_constitution,
            residue_geometry=residue_geometry,
            formal_charge_by_atom_name=(
                snapshot.structure.residue_formal_charge_by_atom_name(
                    self.residue_index
                )
            ),
        )

    def residue_constitution(
        self,
        snapshot: ProteinStructureSnapshot,
    ) -> ResidueSite | None:
        """Resolve the current residue constitution from one snapshot."""

        if self.residue_index.value >= len(
            snapshot.structure.constitution.residue_slots
        ):
            return None

        return snapshot.structure.constitution.residue_site_at(self.residue_index)

    def residue_geometry(
        self,
        snapshot: ProteinStructureSnapshot,
    ) -> ResidueGeometry | None:
        """Resolve the current residue geometry from one snapshot."""

        if self.residue_constitution(snapshot) is None:
            return None

        return snapshot.structure.residue_geometry(self.residue_index)

    def patch(
        self,
        snapshot: ProteinStructureSnapshot,
    ) -> OrderedAtomPatch:
        """Project the current residue into one canonical completion patch."""

        residue_constitution = self.residue_constitution(snapshot)
        residue_geometry = self.residue_geometry(snapshot)
        if residue_constitution is None or residue_geometry is None:
            raise ValueError(
                "hydrogen site residue "
                f"{self.template.component_id}@{self.residue_index.value} is missing"
            )

        return OrderedAtomPatch.from_residue_payload(
            residue_constitution,
            residue_geometry=residue_geometry,
        )

    def includes_backbone_hydrogen(self) -> bool:
        """Return whether this site may propagate one backbone hydrogen."""

        return self.next_residue_index is not None

    def atom_input(
        self,
        snapshot: ProteinStructureSnapshot,
    ) -> AtomInput:
        """Derive the canonical atom domain touched by this residue-local site."""

        residue_constitution = self.residue_constitution(snapshot)
        if residue_constitution is None:
            raise ValueError(
                "hydrogen site residue "
                f"{self.template.component_id}@{self.residue_index.value} is missing"
            )

        return AtomInput(
            atom_indices=tuple(
                snapshot.structure.constitution.atom_indices_for_residue_index(
                    self.residue_index
                )
            ),
            basis=AtomInputBasis.RESIDUEWISE,
            selected_scope=ResidueSetScope(
                residue_ids=(residue_constitution.residue_id,)
            ),
        )

    def optimize_rotatable(
        self,
        search: RotatableHydrogenSearch,
    ) -> Vec3:
        """Return the optimized rotatable-hydrogen coordinate for this site."""

        return self.environment.optimize_rotatable(self.residue_index, search)

    def is_disulfide_bonded(self) -> bool:
        """Return whether this site has canonical disulfide topology."""

        return self.environment.is_disulfide_bonded(self.residue_index)

    def apply_patch(
        self,
        snapshot: ProteinStructureSnapshot,
        patch: OrderedAtomPatch,
    ) -> ProteinStructureSnapshot:
        """Materialize one hydrogen patch on the current residue."""

        if self.residue_constitution(snapshot) is None:
            return snapshot

        return patch.apply_to_snapshot(snapshot, self.residue_index)

"""Canonical domain models for side-chain packing execution."""

from dataclasses import dataclass

from protrepair.diagnostics.events import ValidationIssue
from protrepair.errors import PackingError, ResidueNotFoundError
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ChainSite, ResidueSite
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.labels import ResidueId
from protrepair.transformer.packing.spec import PackingMode, PackingScope, PackingSpec

_MEANINGFUL_HEAVY_ATOM_DISPLACEMENT_ANGSTROM = 0.05


@dataclass(frozen=True, slots=True)
class PackingCapabilities:
    """Declared capabilities of one side-chain packing backend."""

    supports_full_structure_packing: bool
    supports_local_packing: bool
    supports_partial_sequence: bool
    supports_refinement: bool
    supports_noncanonical_components: bool
    deterministic_given_same_inputs: bool

    def supports_spec(self, spec: PackingSpec) -> bool:
        """Return whether these capabilities support one packing spec."""

        if spec.scope is PackingScope.FULL and not self.supports_full_structure_packing:
            return False

        if spec.scope is PackingScope.LOCAL and not self.supports_local_packing:
            return False

        if spec.mode is PackingMode.REFINE and not self.supports_refinement:
            return False

        if spec.has_sequence_override() and not self.supports_partial_sequence:
            return False

        return True

    def require_support_for(self, spec: PackingSpec) -> None:
        """Raise when a packing spec exceeds these capabilities."""

        if spec.scope is PackingScope.FULL and not self.supports_full_structure_packing:
            raise ValueError("packing backend does not support full-structure packing")

        if spec.scope is PackingScope.LOCAL and not self.supports_local_packing:
            raise ValueError("packing backend does not support local packing")

        if spec.mode is PackingMode.REFINE and not self.supports_refinement:
            raise ValueError("packing backend does not support refinement mode")

        if spec.has_sequence_override() and not self.supports_partial_sequence:
            raise ValueError("packing backend does not support sequence overrides")


@dataclass(frozen=True, slots=True)
class PackingPlan:
    """Resolved canonical packing plan over one structure and packing spec."""

    structure: ProteinStructure
    spec: PackingSpec
    polymer_residue_ids: tuple[ResidueId, ...]

    @classmethod
    def from_inputs(
        cls,
        structure: ProteinStructure,
        spec: PackingSpec,
    ) -> "PackingPlan":
        """Build one canonical packing plan from a structure and spec."""

        polymer_residue_ids = tuple(
            residue.residue_id
            for chain in structure.constitution.chains
            for residue in chain.residues
        )
        return cls(
            structure=structure,
            spec=spec,
            polymer_residue_ids=polymer_residue_ids,
        )

    def __post_init__(self) -> None:
        polymer_residue_ids = _normalize_residue_id_tuple(self.polymer_residue_ids)
        if polymer_residue_ids is None:
            raise ValueError("packing plan requires at least one polymer residue")

        polymer_residue_id_set = set(polymer_residue_ids)
        for residue_id in self.spec.referenced_residue_ids():
            if residue_id not in polymer_residue_id_set:
                if self.structure.constitution.residue_or_ligand(residue_id) is None:
                    raise ResidueNotFoundError(
                        f"structure has no residue {residue_id.display_token()}"
                    )

        if (
            self.spec.target_sequence is not None
            and self.spec.mutable_residue_ids is not None
            and len(self.spec.target_sequence) != len(self.spec.mutable_residue_ids)
        ):
            raise ValueError(
                "packing target_sequence length must match mutable_residue_ids"
            )

        object.__setattr__(self, "polymer_residue_ids", polymer_residue_ids)

    def assert_supported_by(self, capabilities: PackingCapabilities) -> None:
        """Raise when one capability set cannot satisfy this plan."""

        capabilities.require_support_for(self.spec)

    def polymer_chain_sites(self) -> tuple[ChainSite, ...]:
        """Return polymer chain sites in structure order."""

        return self.structure.constitution.chains

    def polymer_residue_sites(self) -> tuple[ResidueSite, ...]:
        """Return polymer residue sites in structure order."""

        return tuple(
            residue
            for chain in self.polymer_chain_sites()
            for residue in chain.residues
        )

    def selected_residue_ids(self) -> tuple[ResidueId, ...]:
        """Return residue identifiers explicitly targeted by the plan."""

        if self.spec.mutable_residue_ids is not None:
            return self.spec.mutable_residue_ids

        return self.polymer_residue_ids

    def fixed_residue_ids(self) -> tuple[ResidueId, ...]:
        """Return residue identifiers that must remain fixed during packing."""

        ordered_residue_ids: list[ResidueId] = []
        seen_residue_ids: set[ResidueId] = set()
        selected_residue_id_set = set(self.selected_residue_ids())

        if self.spec.scope is PackingScope.LOCAL:
            for residue_id in self.polymer_residue_ids:
                if residue_id not in selected_residue_id_set:
                    ordered_residue_ids.append(residue_id)
                    seen_residue_ids.add(residue_id)

        if self.spec.frozen_residue_ids is not None:
            for residue_id in self.spec.frozen_residue_ids:
                if residue_id not in seen_residue_ids:
                    ordered_residue_ids.append(residue_id)
                    seen_residue_ids.add(residue_id)

        return tuple(ordered_residue_ids)

    def selected_residue_count(self) -> int:
        """Return the number of explicitly targeted residues."""

        return len(self.selected_residue_ids())

    def residue_site(self, residue_id: ResidueId) -> ResidueSite:
        """Return one polymer residue site from the plan structure."""

        residue_site = self.structure.constitution.residue_or_ligand(residue_id)
        if residue_site is None:
            raise ResidueNotFoundError(
                f"structure has no residue {residue_id.display_token()}"
            )

        if residue_id.chain_id in self.structure.constitution.chain_ids():
            return self.structure.constitution.chain(residue_id.chain_id).residue(
                residue_id
            )

        raise PackingError(
            f"packing plan references non-polymer residue {residue_id.display_token()}"
        )

    def selected_residue_sites(self) -> tuple[ResidueSite, ...]:
        """Return explicitly targeted residue sites in selection order."""

        return tuple(
            self.residue_site(residue_id)
            for residue_id in self.selected_residue_ids()
        )

    def changed_residue_ids_after(
        self,
        packed_structure: ProteinStructure,
    ) -> tuple[ResidueId, ...]:
        """Return residue identifiers with semantically meaningful packing changes."""

        packed_residue_sites = tuple(
            residue
            for chain in packed_structure.constitution.chains
            for residue in chain.residues
        )
        original_residue_sites = self.polymer_residue_sites()
        if len(packed_residue_sites) != len(original_residue_sites):
            raise PackingError(
                "packed structure changed the number of polymer residues unexpectedly"
            )

        changed_residue_ids: list[ResidueId] = []
        for original_residue_site, packed_residue_site in zip(
            original_residue_sites,
            packed_residue_sites,
            strict=True,
        ):
            if original_residue_site.residue_id != packed_residue_site.residue_id:
                raise PackingError(
                    "packed structure changed residue identifiers or order unexpectedly"
                )

            residue_id = packed_residue_site.residue_id
            if _has_semantic_packing_change(
                original_structure=self.structure,
                packed_structure=packed_structure,
                original_residue_site=original_residue_site,
                packed_residue_site=packed_residue_site,
            ):
                changed_residue_ids.append(residue_id)

        return tuple(changed_residue_ids)


def _has_semantic_packing_change(
    *,
    original_structure: ProteinStructure,
    packed_structure: ProteinStructure,
    original_residue_site: ResidueSite,
    packed_residue_site: ResidueSite,
) -> bool:
    """Return whether one residue changed in packing-owned heavy-atom state."""

    if _heavy_atom_site_signature(original_residue_site) != _heavy_atom_site_signature(
        packed_residue_site
    ):
        return True

    residue_id = original_residue_site.residue_id
    original_residue_index = original_structure.constitution.residue_index(residue_id)
    packed_residue_index = packed_structure.constitution.residue_index(residue_id)
    original_geometry = original_structure.geometry.residue_geometry(
        constitution=original_structure.constitution,
        residue_index=original_residue_index,
    )
    packed_geometry = packed_structure.geometry.residue_geometry(
        constitution=packed_structure.constitution,
        residue_index=packed_residue_index,
    )

    if _heavy_atom_coordinates_moved_meaningfully(
        original_residue_site=original_residue_site,
        packed_residue_site=packed_residue_site,
        original_geometry=original_geometry,
        packed_geometry=packed_geometry,
    ):
        return True

    return _heavy_atom_formal_charge_by_name(
        structure=original_structure,
        residue_site=original_residue_site,
    ) != _heavy_atom_formal_charge_by_name(
        structure=packed_structure,
        residue_site=packed_residue_site,
    )


def _heavy_atom_site_signature(
    residue_site: ResidueSite,
) -> tuple[str, bool, tuple[tuple[str, str], ...]]:
    """Return residue identity and non-hydrogen atom sites relevant to packing."""

    return (
        residue_site.component_id,
        residue_site.is_hetero,
        tuple(
            (atom_site.name, atom_site.element)
            for atom_site in residue_site.atom_sites
            if not atom_site.is_hydrogen()
        ),
    )


def _heavy_atom_coordinates_moved_meaningfully(
    *,
    original_residue_site: ResidueSite,
    packed_residue_site: ResidueSite,
    original_geometry: ResidueGeometry,
    packed_geometry: ResidueGeometry,
) -> bool:
    """Return whether retained heavy atoms moved beyond backend output noise."""

    packed_heavy_atom_names = {
        atom_site.name
        for atom_site in packed_residue_site.atom_sites
        if not atom_site.is_hydrogen()
    }
    for atom_site in original_residue_site.atom_sites:
        if atom_site.is_hydrogen() or atom_site.name not in packed_heavy_atom_names:
            continue

        distance = original_geometry.atom_geometry(atom_site.name).distance_to(
            packed_geometry.atom_geometry(atom_site.name)
        )
        if distance > _MEANINGFUL_HEAVY_ATOM_DISPLACEMENT_ANGSTROM:
            return True

    return False


def _heavy_atom_formal_charge_by_name(
    *,
    structure: ProteinStructure,
    residue_site: ResidueSite,
) -> tuple[tuple[str, int | None], ...]:
    """Return residue formal charges restricted to non-hydrogen atom sites."""

    heavy_atom_names = {
        atom_site.name
        for atom_site in residue_site.atom_sites
        if not atom_site.is_hydrogen()
    }
    residue_formal_charge_by_atom_name = (
        structure.topology.residue_formal_charge_by_atom_name(
            constitution=structure.constitution,
            residue_index=structure.constitution.residue_index(residue_site.residue_id),
        )
    )
    return tuple(
        (atom_name, formal_charge)
        for atom_name, formal_charge in residue_formal_charge_by_atom_name
        if atom_name in heavy_atom_names
    )


@dataclass(frozen=True, slots=True)
class PackingResult:
    """Structured result from one side-chain packing backend."""

    packed_structure: ProteinStructure
    changed_residue_ids: tuple[ResidueId, ...]
    issues: tuple[ValidationIssue, ...]
    backend_name: str
    backend_version: str | None = None

    def __post_init__(self) -> None:
        backend_name = self.backend_name.strip().lower()
        if not backend_name:
            raise ValueError("packing result backend_name must not be blank")

        backend_version = self.backend_version
        if backend_version is not None:
            backend_version = backend_version.strip() or None

        changed_residue_ids = _normalize_residue_id_tuple(self.changed_residue_ids)
        issues = tuple(self.issues)

        if changed_residue_ids is not None:
            for residue_id in changed_residue_ids:
                if (
                    self.packed_structure.constitution.residue_or_ligand(residue_id)
                    is None
                ):
                    raise ResidueNotFoundError(
                        f"structure has no residue {residue_id.display_token()}"
                    )

        object.__setattr__(self, "backend_name", backend_name)
        object.__setattr__(
            self,
            "changed_residue_ids",
            () if changed_residue_ids is None else changed_residue_ids,
        )
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "backend_version", backend_version)

    def changed_residue_count(self) -> int:
        """Return the number of residues changed by the backend."""

        return len(self.changed_residue_ids)

    def changed_residue(self, residue_id: ResidueId) -> bool:
        """Return whether one residue identifier was changed."""

        return residue_id in self.changed_residue_ids

    def has_issues(self) -> bool:
        """Return whether the backend reported any validation issue."""

        return bool(self.issues)


def _normalize_residue_id_tuple(
    residue_ids: tuple[ResidueId, ...] | None,
) -> tuple[ResidueId, ...] | None:
    """Normalize one optional residue-identifier tuple."""

    if residue_ids is None:
        return None

    normalized_residue_ids: list[ResidueId] = []
    seen_residue_ids: set[ResidueId] = set()
    for residue_id in residue_ids:
        if residue_id not in seen_residue_ids:
            normalized_residue_ids.append(residue_id)
            seen_residue_ids.add(residue_id)

    return tuple(normalized_residue_ids) or None

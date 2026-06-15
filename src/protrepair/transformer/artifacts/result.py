"""Transformer output results over refined structures and slot deltas."""

from dataclasses import dataclass

from protrepair.diagnostics.events import ValidationIssue
from protrepair.structure.aggregate import ProteinStructure
from protrepair.transformer.artifacts.patch import StructureDelta


@dataclass(frozen=True, slots=True)
class RegionTransformationResult:
    """Structured result from one selected-region transformation."""

    refined_structure: ProteinStructure
    delta: StructureDelta
    issues: tuple[ValidationIssue, ...]
    backend_name: str
    backend_version: str | None = None

    def __post_init__(self) -> None:
        backend_name = self.backend_name.strip().lower()
        if not backend_name:
            raise ValueError(
                "region transformation result backend_name must not be blank"
            )

        backend_version = self.backend_version
        if backend_version is not None:
            backend_version = backend_version.strip() or None

        if not isinstance(self.delta, StructureDelta):
            raise TypeError(
                "region transformation result delta must be a StructureDelta"
            )
        if self.delta.after_constitution != self.refined_structure.constitution:
            raise ValueError(
                "region transformation result delta after_constitution must match "
                "the refined structure constitution"
            )

        issues = tuple(self.issues)
        for rewrite in self.delta.residue_identity_rewrites:
            residue_site = self.refined_structure.constitution.residue_site_at(
                rewrite.after_residue_index
            )
            if residue_site.component_id != rewrite.current_component_id:
                raise ValueError(
                    "region transformation result residue identity rewrite for "
                    f"slot {rewrite.after_residue_index.value} expects component "
                    f"{rewrite.current_component_id}, but refined structure has "
                    f"{residue_site.component_id}"
                )
        for rewrite in self.delta.residue_topology_rewrites:
            for atom_index in rewrite.affected_atom_indices:
                if (
                    self.refined_structure.constitution.residue_index_for_atom_index(
                        atom_index
                    )
                    != rewrite.after_residue_index
                ):
                    raise ValueError(
                        "region transformation result residue topology rewrite "
                        "references an atom outside the rewritten residue"
                    )
        for rewrite in self.delta.atom_element_rewrites:
            atom_site = self.refined_structure.constitution.atom_site_at(
                rewrite.after_atom_index
            )
            if atom_site.element != rewrite.current_element:
                raise ValueError(
                    "region transformation result atom element rewrite for "
                    f"slot {rewrite.after_atom_index.value} expects element "
                    f"{rewrite.current_element}, but refined structure has "
                    f"{atom_site.element}"
                )
        for rewrite in self.delta.atom_formal_charge_rewrites:
            if (
                self.refined_structure.topology.formal_charge(
                    rewrite.after_atom_index
                )
                != rewrite.current_formal_charge
            ):
                raise ValueError(
                    "region transformation result atom formal-charge rewrite for "
                    f"slot {rewrite.after_atom_index.value} expects charge "
                    f"{rewrite.current_formal_charge}, but refined structure has "
                    f"{self.refined_structure.topology.formal_charge(rewrite.after_atom_index)}"
                )

        object.__setattr__(self, "backend_name", backend_name)
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "backend_version", backend_version)

    def moved_atom_count(self) -> int:
        """Return the number of atoms whose coordinates were updated."""

        return self.delta.moved_atom_count()

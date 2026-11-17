"""Coverage and residuals for the reference-backed bonded geometry goal."""

from collections.abc import Collection
from dataclasses import dataclass
from itertools import pairwise
from math import fsum

from protrepair.chemistry import ComponentLibrary, build_default_restraint_library
from protrepair.chemistry.restraint.structure import StructureRestraints
from protrepair.chemistry.restraint.template import (
    AngleRestraintTarget,
    BondRestraintTarget,
)
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.chemistry.standard.restraints import build_standard_restraint_library
from protrepair.diagnostics.geometry import (
    HeavyGeometryReport,
    bond_angle_degrees,
    detect_heavy_geometry,
)
from protrepair.state.structure_axes import BondedGeometryState
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import BondRelationshipType

BondedTerm = tuple[AtomRef, ...]


@dataclass(frozen=True, slots=True)
class BondedGeometryObservation:
    """Assess stock reference terms without treating missing evidence as clear."""

    assessed_references: frozenset[
        tuple[BondedTerm, BondRestraintTarget | AngleRestraintTarget]
    ]
    unassessed_residue_ids: tuple[ResidueId, ...]
    qualified_residue_ids: frozenset[ResidueId]
    residual_excesses: tuple[tuple[BondedTerm, float], ...]
    report: HeavyGeometryReport

    @classmethod
    def from_structure(
        cls,
        structure: ProteinStructure,
        *,
        component_library: ComponentLibrary,
        residue_ids: Collection[ResidueId] | None = None,
    ) -> "BondedGeometryObservation":
        """Bind qualified coverage to this snapshot and its applicable references."""

        standard = build_standard_component_library()
        stock_restraints = build_standard_restraint_library()
        library = build_default_restraint_library()
        references = StructureRestraints(structure, component_library, library)
        assessed: dict[BondedTerm, BondRestraintTarget | AngleRestraintTarget] = {}
        unassessed: list[ResidueId] = []
        qualified: set[ResidueId] = set()
        for residue in structure.constitution.iter_residues():
            rid = residue.residue_id
            if residue_ids is not None and rid not in residue_ids:
                continue
            stock = standard.get(residue.component_id)
            template, unresolved = references.bound_targets(rid)
            if (
                stock is None
                or component_library.get(residue.component_id) != stock
                or library.get(residue.component_id)
                != stock_restraints.get(residue.component_id)
                or library.component_references.get(
                    library.normalize_component_id(residue.component_id)
                )
                or template is None
            ):
                unassessed.append(rid)
                continue

            expected = (
                *(
                    (tuple(sorted((target.atom_name_1, target.atom_name_2))), target)
                    for target in template.bond_targets
                ),
                *(
                    ((left, center, right), target)
                    for target in template.angle_targets
                    for center, left, right in (target.canonical_triplet(),)
                ),
            )
            geometry = structure.residue_geometry(
                structure.constitution.residue_index(rid)
            )
            measured = 0
            for names, target in expected:
                if unresolved.intersection(names):
                    continue
                refs = tuple(AtomRef(rid, name) for name in names)
                if not _has_realized_heavy_term(structure, refs):
                    continue
                if (
                    len(names) == 3
                    and bond_angle_degrees(
                        geometry,
                        atom_name_1=names[0],
                        center_atom_name=names[1],
                        atom_name_2=names[2],
                    )
                    is None
                ):
                    continue
                assessed[refs] = target
                measured += 1
            if unresolved or not expected or measured != len(expected):
                unassessed.append(rid)
            else:
                qualified.add(rid)

        report = detect_heavy_geometry(
            structure,
            component_library=component_library,
            restraint_library=library,
            residue_ids=residue_ids,
        )
        residuals: list[tuple[BondedTerm, float]] = []
        for bond in report.bond_length_outliers:
            names = sorted((bond.atom_name_1, bond.atom_name_2))
            refs = tuple(AtomRef(bond.residue_id, name) for name in names)
            if bond.restraint_backed and refs in assessed:
                residuals.append(
                    (
                        refs,
                        (bond.deviation_angstrom() - bond.tolerance_angstrom)
                        / bond.tolerance_angstrom,
                    )
                )
        for angle in report.bond_angle_outliers:
            left, right = sorted((angle.atom_name_1, angle.atom_name_2))
            refs = (
                AtomRef(angle.residue_id, left),
                AtomRef(angle.residue_id, angle.center_atom_name),
                AtomRef(angle.residue_id, right),
            )
            if angle.restraint_backed and refs in assessed:
                half_width = (
                    angle.maximum_angle_degrees - angle.minimum_angle_degrees
                ) / 2
                residuals.append((refs, angle.deviation_degrees() / half_width))

        return cls(
            assessed_references=frozenset(assessed.items()),
            unassessed_residue_ids=tuple(unassessed),
            qualified_residue_ids=frozenset(qualified),
            residual_excesses=tuple(residuals),
            report=report,
        )

    def observed_state(self) -> BondedGeometryState | None:
        """Keep known outliers distinct from incomplete or empty coverage."""

        if self.residual_excesses:
            return BondedGeometryState.OUTLIERS
        if self.assessed_references and not self.unassessed_residue_ids:
            return BondedGeometryState.WITHIN_REFERENCE
        return None

    def total_excess(self) -> float:
        """Sum normalized excess without mixing bond lengths and angle units."""

        return fsum(value for _, value in self.residual_excesses)


def _has_realized_heavy_term(structure: ProteinStructure, refs: BondedTerm) -> bool:
    constitution = structure.constitution
    indices = []
    for ref in refs:
        index = constitution.resolve_atom_index(ref)
        if index is None or constitution.atom_site_at(index).is_hydrogen():
            return False
        indices.append(index)

    return all(
        bond is not None
        and bond.relationship_type is BondRelationshipType.COVALENT
        and bond.order is not None
        for left, right in pairwise(indices)
        for bond in (structure.topology.bond_between(left, right),)
    )

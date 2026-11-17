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
from protrepair.scope import ChainSetScope, ResidueSetScope, Scope, WholeStructureScope
from protrepair.state.structure_axes import BondedGeometryState
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import StructureConstitution
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import BondRelationshipType

BondedTerm = tuple[AtomRef, ...]
BONDED_GEOMETRY_SCOPE_TYPES = (WholeStructureScope, ChainSetScope, ResidueSetScope)


@dataclass(frozen=True, slots=True)
class BondedGeometrySelection:
    """Resolve polymer targets without discarding explicitly requested identities."""

    residue_ids: frozenset[ResidueId]
    missing_chain_ids: tuple[str, ...]

    @classmethod
    def from_scopes(
        cls,
        constitution: StructureConstitution,
        scopes: tuple[Scope, ...],
    ) -> "BondedGeometrySelection":
        """Union the requested scopes, including gaps in the current snapshot."""

        select_all = False
        chain_ids: set[str] = set()
        residue_ids: set[ResidueId] = set()
        for scope in scopes:
            if isinstance(scope, WholeStructureScope):
                select_all = True
            elif isinstance(scope, ChainSetScope):
                chain_ids.update(scope.chain_ids)
            elif isinstance(scope, ResidueSetScope):
                residue_ids.update(scope.residue_ids)
            else:
                raise TypeError("unsupported bonded-geometry goal scope")

        found_chains: set[str] = set()
        for chain in constitution.chains:
            if chain.residues and (select_all or chain.chain_id in chain_ids):
                found_chains.add(chain.chain_id)
                residue_ids.update(residue.residue_id for residue in chain.residues)
        return cls(frozenset(residue_ids), tuple(sorted(chain_ids - found_chains)))

    def includes_term(self, term: BondedTerm) -> bool:
        """Select an intra-residue term by its polymer residue identity."""

        return term[0].residue_id in self.residue_ids


@dataclass(frozen=True, slots=True)
class ScopedBondedGeometryObservation:
    """Coverage and residual excess for the requested polymer terms only."""

    assessed_term_count: int
    unassessed_residue_ids: tuple[ResidueId, ...]
    missing_chain_ids: tuple[str, ...]
    residual_excesses: tuple[tuple[BondedTerm, float], ...]

    def observed_state(self) -> BondedGeometryState | None:
        """Keep known outliers distinct from incomplete or empty coverage."""

        if self.residual_excesses:
            return BondedGeometryState.OUTLIERS
        if (
            self.assessed_term_count
            and not self.unassessed_residue_ids
            and not self.missing_chain_ids
        ):
            return BondedGeometryState.WITHIN_TOLERANCE
        return None

    def branch_preference(self) -> tuple[int, float]:
        """Prefer better coverage, then less normalized excess."""

        return (
            len(self.unassessed_residue_ids) + len(self.missing_chain_ids),
            fsum(value for _, value in self.residual_excesses),
        )


@dataclass(frozen=True, slots=True)
class BondedGeometryObservation:
    """Assess stock reference terms without treating missing evidence as clear."""

    assessed_references: frozenset[
        tuple[BondedTerm, BondRestraintTarget | AngleRestraintTarget]
    ]
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
            if not unresolved and expected and measured == len(expected):
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
            qualified_residue_ids=frozenset(qualified),
            residual_excesses=tuple(residuals),
            report=report,
        )

    def for_selection(
        self, selection: BondedGeometrySelection
    ) -> ScopedBondedGeometryObservation:
        """Project the observations without narrowing candidate motion or safety."""

        return ScopedBondedGeometryObservation(
            assessed_term_count=sum(
                selection.includes_term(term) for term, _ in self.assessed_references
            ),
            unassessed_residue_ids=tuple(
                sorted(selection.residue_ids - self.qualified_residue_ids)
            ),
            missing_chain_ids=selection.missing_chain_ids,
            residual_excesses=tuple(
                (term, excess)
                for term, excess in self.residual_excesses
                if selection.includes_term(term)
            ),
        )


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

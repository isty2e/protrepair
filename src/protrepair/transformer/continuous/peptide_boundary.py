"""Calculation-only support for ordinary peptide cuts, never source repair."""

from collections import Counter, defaultdict
from dataclasses import dataclass

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.microstate.catalog import PeptideLinkage, PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.errors import RefinementError
from protrepair.structure.labels import AtomRef
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import BondRelationshipType
from protrepair.transformer.continuous.bonds import PlannedBond
from protrepair.transformer.continuous.domain import ContinuousRelaxationRegion


@dataclass(frozen=True, slots=True)
class PeptideBoundary:
    """One missing calculation bond and its optional canonical source partner.

    Parameters
    ----------
    endpoint : AtomRef
        Included backbone N or C needing one single peptide bond.
    source_partner : AtomRef or None
        Omitted canonical partner, or None for an internal source crop.
    supplemental_hydrogen : bool
        The unresolved internal N also needs a calculation-only N-H attachment.
    source_partner_hydrogen : AtomRef or None
        Validated source H anchor for an omitted N, including named attachments
        not yet materialized as source bonds.
    """

    endpoint: AtomRef
    source_partner: AtomRef | None
    supplemental_hydrogen: bool = False
    source_partner_hydrogen: AtomRef | None = None


@dataclass(frozen=True, slots=True)
class PeptideBoundaryPlan:
    """Validated peptide cuts for one immutable continuous problem.

    Parameters
    ----------
    region : ContinuousRelaxationRegion
        Original source graph, included context and movable scope.
    bonds : tuple[PlannedBond, ...]
        Validated calculation edges in source atom indices.
    boundaries : tuple[PeptideBoundary, ...]
        Cuts validated by ``from_region`` against source and current chemistry.
    """

    region: ContinuousRelaxationRegion
    bonds: tuple[PlannedBond, ...]
    boundaries: tuple[PeptideBoundary, ...]

    @classmethod
    def from_region(
        cls,
        region: ContinuousRelaxationRegion,
        bonds: tuple[PlannedBond, ...],
        component_library: ComponentLibrary,
    ) -> "PeptideBoundaryPlan":
        """Identify supported cuts without treating missing links as free termini.

        Parameters
        ----------
        region : ContinuousRelaxationRegion
            Source-indexed region to complete.
        bonds : tuple[PlannedBond, ...]
            Resolved edges, independent of force-field and optimizer selection.
        component_library : ComponentLibrary
            Active chemistry; custom site chemistry is not replaced by defaults.

        Returns
        -------
        PeptideBoundaryPlan
            No boundaries when the included graph is already closed.

        Raises
        ------
        RefinementError
            A cut is non-peptide, ambiguous, or incompatible with source chemistry.
        """
        source = region.snapshot.structure
        constitution = source.constitution
        included = frozenset(region.included_atom_indices())
        polymer_count = len(constitution.residue_slots) - len(constitution.ligands)
        neighbors: dict[AtomIndex, list[tuple[AtomIndex, PlannedBond]]] = defaultdict(
            list
        )
        # Planned edges include validated H attachments, but never override a
        # canonical edge. Unresolved relationships must not disappear at a cut.
        edges = {bond.sort_key(): bond for bond in bonds}
        unsupported_indices: set[AtomIndex] = set()
        for bond in source.topology.bonds:
            if bond.relationship_type is BondRelationshipType.HYDROGEN_BOND:
                continue
            left, right = bond.endpoint_pair()
            if bond.relationship_type is not BondRelationshipType.COVALENT:
                unsupported_indices.update((left, right))
            if (left in included) != (right in included):
                if bond.relationship_type is not BondRelationshipType.COVALENT:
                    raise RefinementError(
                        "unsupported non-peptide calculation boundary"
                    )
            if bond.relationship_type is BondRelationshipType.COVALENT:
                edges[(left.value, right.value)] = PlannedBond(
                    left, right, bond.order, bond.aromatic
                )
        for bond in edges.values():
            for index, other in (
                (bond.atom_index_1, bond.atom_index_2),
                (bond.atom_index_2, bond.atom_index_1),
            ):
                neighbors[index].append((other, bond))

        context = PolymerMicrostateContext(source)
        boundaries = []
        for index in sorted(included, key=lambda item: item.value):
            ref = constitution.atom_ref_at(index)
            residue_index = constitution.residue_index_for_atom_index(index)
            outside = [
                (other, bond)
                for other, bond in neighbors[index]
                if other not in included
            ]
            if residue_index.value >= polymer_count or ref.atom_name not in {"N", "C"}:
                if outside:
                    raise RefinementError(f"non-peptide calculation cut at {ref}")
                continue
            local = [
                (other, bond) for other, bond in neighbors[index] if other in included
            ]
            expected = (
                3 + (source.topology.formal_charge(index) or 0)
                if ref.atom_name == "N"
                else 4
            )
            if all(bond.order is not None for _, bond in local):
                if (
                    sum(bond.order or 0 for _, bond in local) == expected
                    and not outside
                ):
                    continue
            if len(outside) > 1:
                raise RefinementError(f"multiple omitted partners at {ref}")
            partner = constitution.atom_ref_at(outside[0][0]) if outside else None
            partner_hydrogen = None
            if outside:
                other, bond = outside[0]
                if (
                    partner is None
                    or partner.residue_id == ref.residue_id
                    or {partner.atom_name, ref.atom_name} != {"C", "N"}
                    or constitution.residue_index_for_atom_index(other).value
                    >= polymer_count
                    or bond.order != 1
                    or bond.aromatic
                ):
                    raise RefinementError(f"unsupported peptide partner at {ref}")
                partner_hydrogen = cls._source_partner_hydrogen(
                    context, partner, component_library
                )
            else:
                chain = constitution.chain(ref.residue_id.chain_id)
                end = chain.residues[0 if ref.atom_name == "N" else -1]
                if end.residue_id == ref.residue_id:
                    raise RefinementError(
                        f"unrealized free-terminal chemistry at {ref}"
                    )
                if any(
                    constitution.atom_ref_at(other).residue_id != ref.residue_id
                    for other, _ in neighbors[index]
                ):
                    raise RefinementError(
                        f"conflicting external boundary chemistry at {ref}"
                    )

            site = constitution.residue_site_at(residue_index)
            template = component_library.get(site.component_id)
            if template is None:
                raise RefinementError(
                    f"unsupported peptide boundary component at {ref}"
                )
            kind = (
                PolymerChemicalSite.BACKBONE_N
                if ref.atom_name == "N"
                else PolymerChemicalSite.BACKBONE_C
            )
            linked_site = PolymerMicrostateSite(template, kind, PeptideLinkage.LINKED)
            resolution = context.resolve(ref.residue_id, linked_site)
            graph = resolution.graph
            if graph is None:
                raise RefinementError(
                    f"unsupported peptide boundary chemistry at {ref}: "
                    f"{resolution.details}"
                )
            if source.topology.formal_charge(index) not in (None, 0):
                raise RefinementError(f"charged peptide boundary at {ref}")
            if ref.atom_name == "C" and site.has_atom_site("OXT"):
                raise RefinementError(
                    f"terminal oxygen contradicts peptide cap at {ref}"
                )

            # Check the linked catalog with precisely one absent external bond.
            # Unlike source realization, this does not assert that bond exists.
            site_indices = {
                constitution.atom_index(AtomRef(ref.residue_id, atom.name))
                for atom in graph.atoms
            }
            if site_indices.intersection(unsupported_indices):
                raise RefinementError(
                    f"unsupported current peptide relationship at {ref}"
                )
            for attachment in resolution.observed_hydrogens:
                h = constitution.resolve_atom_index(attachment.hydrogen)
                parent = constitution.resolve_atom_index(attachment.parent)
                if (
                    h is None
                    or parent is None
                    or not any(
                        other == parent and bond.order == 1
                        for other, bond in neighbors[h]
                    )
                ):
                    raise RefinementError(f"source H attachment changed at {ref}")
            supplemental_hydrogen = False
            for atom in graph.atoms:
                atom_index = constitution.atom_index(AtomRef(ref.residue_id, atom.name))
                local_neighbors = [
                    (other, bond)
                    for other, bond in neighbors[atom_index]
                    if other in included
                ]
                hs = [
                    other
                    for other, _ in local_neighbors
                    if constitution.atom_site_at(other).is_hydrogen()
                ]
                missing_order = int(atom.name == ref.atom_name)
                modeled_h = (
                    partner is None
                    and atom.name == "N"
                    and atom.hydrogens == 1
                    and not hs
                    and not resolution.observed_hydrogens
                )
                supplemental_hydrogen = supplemental_hydrogen or modeled_h
                if (
                    (source.topology.formal_charge(atom_index) or 0) != atom.charge
                    or len(hs) + int(modeled_h) != atom.hydrogens
                    or any(
                        bond.order is None or bond.aromatic
                        for _, bond in local_neighbors
                    )
                    or sum(bond.order or 0 for _, bond in local_neighbors)
                    + missing_order
                    + int(modeled_h)
                    != atom.closed_shell_valence()
                    or any(
                        len(neighbors[h]) != 1
                        or source.topology.formal_charge(h) not in (None, 0)
                        for h in hs
                    )
                ):
                    raise RefinementError(
                        f"unrealized peptide boundary H/charge/valence at {ref}"
                    )
                observed = Counter(
                    entry.parent.atom_name for entry in resolution.observed_hydrogens
                )
                if observed[atom.name] > len(hs):
                    raise RefinementError(
                        f"source H contradicts peptide boundary at {ref}"
                    )
            for bond in graph.bonds:
                first = constitution.atom_index(
                    AtomRef(ref.residue_id, bond.atom_name_1)
                )
                second = constitution.atom_index(
                    AtomRef(ref.residue_id, bond.atom_name_2)
                )
                actual = next(
                    (b for other, b in neighbors[first] if other == second), None
                )
                if actual is None or actual.order != bond.order or actual.aromatic:
                    raise RefinementError(f"unrealized peptide boundary bond at {ref}")
            boundaries.append(
                PeptideBoundary(ref, partner, supplemental_hydrogen, partner_hydrogen)
            )
        return cls(region, bonds, tuple(boundaries))

    @staticmethod
    def _source_partner_hydrogen(
        context: PolymerMicrostateContext,
        partner: AtomRef,
        component_library: ComponentLibrary,
    ) -> AtomRef | None:
        source = context.source
        constitution = source.constitution
        residue = constitution.residue_or_ligand(partner.residue_id)
        assert residue is not None
        template = component_library.get(residue.component_id)
        if template is None:
            raise RefinementError(f"unsupported source cap partner at {partner}")
        site = PolymerMicrostateSite(
            template,
            PolymerChemicalSite.BACKBONE_N
            if partner.atom_name == "N"
            else PolymerChemicalSite.BACKBONE_C,
            PeptideLinkage.LINKED,
        )
        resolution = context.resolve(partner.residue_id, site)
        graph = resolution.graph
        if graph is None:
            raise RefinementError(f"unresolved source cap partner at {partner}")
        try:
            attachments = context.hydrogen_attachments(
                partner.residue_id, site, resolution
            )
        except ValueError as error:
            raise RefinementError(
                f"unsupported source cap partner at {partner}"
            ) from error
        # Omitted residues need valid anchors, not a source-topology repair.
        # The context validates named H attachment without requiring an N-H edge.
        counts = Counter(attachments.values())
        if any(
            counts[atom.name] != atom.hydrogens
            or (
                source.topology.formal_charge(
                    constitution.atom_index(AtomRef(partner.residue_id, atom.name))
                )
                or 0
            )
            != atom.charge
            for atom in graph.atoms
        ):
            raise RefinementError(f"unrealized source cap partner at {partner}")
        for bond in graph.bonds:
            actual = source.topology.bond_between(
                constitution.atom_index(AtomRef(partner.residue_id, bond.atom_name_1)),
                constitution.atom_index(AtomRef(partner.residue_id, bond.atom_name_2)),
            )
            if (
                actual is None
                or actual.relationship_type is not BondRelationshipType.COVALENT
                or actual.order != bond.order
                or actual.aromatic
            ):
                raise RefinementError(
                    f"unrealized source cap partner bond at {partner}"
                )
        return next(
            (
                AtomRef(partner.residue_id, name)
                for name, parent in attachments.items()
                if parent == "N"
            ),
            None,
        )

    def modeled_source_endpoints(self) -> tuple[AtomRef, ...]:
        """Return internal source gaps supported without claiming source repair.

        Returns
        -------
        tuple[AtomRef, ...]
            Only endpoints without an existing canonical partner.
        """
        return tuple(b.endpoint for b in self.boundaries if b.source_partner is None)

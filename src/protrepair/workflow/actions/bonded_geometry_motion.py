"""Resolve explicit movable atoms for a bonded-geometry attempt."""

from dataclasses import dataclass, field

from protrepair.scope import AtomSetScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.structure.topology import BondRelationshipType, TopologyBond


@dataclass(frozen=True, slots=True)
class BondedGeometryMotion:
    """Source-safe motion admission for qualified polymer residues."""

    structure: ProteinStructure
    qualified_residue_ids: frozenset[ResidueId]
    _unresolved_hydrogen_parents: dict[ResidueId, bool] = field(
        default_factory=dict, init=False, compare=False, repr=False
    )

    def group_for(self, heavy: AtomRef) -> frozenset[AtomRef]:
        """Return one heavy atom with its eligible attached H, or no motion."""

        if heavy.residue_id not in self.qualified_residue_ids:
            return frozenset()
        constitution = self.structure.constitution
        index = constitution.resolve_atom_index(heavy)
        if index is None or constitution.atom_site_at(index).is_hydrogen():
            return frozenset()
        residue = constitution.residue_or_ligand(heavy.residue_id)
        assert residue is not None
        if self._has_unresolved_hydrogen_parents(residue):
            return frozenset()
        attached: set[AtomRef] = {heavy}
        for other, bond in self._neighbors(index):
            other_ref = constitution.atom_ref_at(other)
            if other_ref.residue_id != heavy.residue_id and (
                bond.relationship_type is not BondRelationshipType.HYDROGEN_BOND
                and not is_canonical_peptide_connection(heavy, other_ref, bond)
            ):
                return frozenset()
            site = constitution.atom_site_at(other)
            if not site.is_hydrogen():
                continue
            if bond.relationship_type is not BondRelationshipType.COVALENT:
                continue
            if bond.order != 1 or not self._hydrogen_is_generated(residue, other_ref):
                return frozenset()
            attached.add(other_ref)
        return frozenset(attached)

    def allows(self, scope: AtomSetScope) -> bool:
        """Check exact motion groups before lowering a planned atom scope."""

        refs = frozenset(scope.atom_refs)
        admitted = frozenset(atom for ref in refs for atom in self.group_for(ref))
        return bool(admitted) and refs == admitted

    def covalent_frontier(self, refs: frozenset[AtomRef]) -> tuple[AtomRef, ...]:
        """Find ordered heavy neighbors on real, supported covalent edges."""

        constitution = self.structure.constitution
        frontier: set[AtomRef] = set()
        for ref in refs:
            index = constitution.atom_index(ref)
            if constitution.atom_site_at(index).is_hydrogen():
                continue
            for other, bond in self._neighbors(index):
                other_ref = constitution.atom_ref_at(other)
                if (
                    other_ref not in refs
                    and not constitution.atom_site_at(other).is_hydrogen()
                    and bond.relationship_type is BondRelationshipType.COVALENT
                    and bond.order is not None
                    and (
                        ref.residue_id == other_ref.residue_id
                        or is_canonical_peptide_connection(ref, other_ref, bond)
                    )
                ):
                    frontier.add(other_ref)
        return tuple(sorted(frontier))

    def heavy_contact_partner(self, ref: AtomRef) -> AtomRef | None:
        """Map a contact to an admissible heavy atom, with H custody intact."""

        constitution = self.structure.constitution
        index = constitution.resolve_atom_index(ref)
        if index is None:
            return None
        if not constitution.atom_site_at(index).is_hydrogen():
            return ref if self.group_for(ref) else None
        parents = tuple(
            constitution.atom_ref_at(other)
            for other, bond in self._neighbors(index)
            if bond.relationship_type is BondRelationshipType.COVALENT
            and not constitution.atom_site_at(other).is_hydrogen()
        )
        if len(parents) != 1 or ref not in self.group_for(parents[0]):
            return None
        return parents[0]

    def _neighbors(
        self, index: AtomIndex
    ) -> tuple[tuple[AtomIndex, TopologyBond], ...]:
        return tuple(
            (second if first == index else first, bond)
            for bond in self.structure.topology.incident_bonds_by_atom().get(index, ())
            for first, second in (bond.endpoint_pair(),)
        )

    def _has_unresolved_hydrogen_parents(self, residue: ResidueSite) -> bool:
        rid = residue.residue_id
        if rid not in self._unresolved_hydrogen_parents:
            constitution = self.structure.constitution
            parents = self.structure.topology.residue_hydrogen_anchor_by_name(
                constitution=constitution, residue_index=constitution.residue_index(rid)
            )
            self._unresolved_hydrogen_parents[rid] = any(
                atom.is_hydrogen() and atom.name not in parents
                for atom in residue.atom_sites
            )
        return self._unresolved_hydrogen_parents[rid]

    def _hydrogen_is_generated(self, residue: ResidueSite, ref: AtomRef) -> bool:
        source = self.structure.provenance.ingress.observation
        if source is None:
            return False
        original = source.constitution.residue_or_ligand(residue.residue_id)
        if (
            original is None
            or original.component_id != residue.component_id
            or original.has_atom_site(ref.atom_name)
            or not source.heavy_atoms_are_preserved_in(residue)
        ):
            return False

        current = self.structure.constitution
        for original_h in source.hydrogen_atoms(residue.residue_id):
            old_index = source.constitution.atom_index(original_h)
            new_index = current.resolve_atom_index(original_h)
            if (
                new_index is None
                or current.atom_site_at(new_index).element_identity
                != source.constitution.atom_site_at(old_index).element_identity
                or self.structure.geometry.atom_geometry(new_index).altloc
                != source.geometry.atom_geometry(old_index).altloc
            ):
                return False
        return (
            current.atom_site_at(
                current.atom_index(ref)
            ).element_identity.isotope_mass_number
            is None
        )


def is_canonical_peptide_connection(
    left: AtomRef, right: AtomRef, bond: TopologyBond
) -> bool:
    """Recognize an actual peptide edge, never ordinal or proximity adjacency."""

    return (
        left.residue_id != right.residue_id
        and left.residue_id.chain_id == right.residue_id.chain_id
        and {left.atom_name, right.atom_name} == {"C", "N"}
        and bond.relationship_type is BondRelationshipType.COVALENT
        and bond.order == 1
        and not bond.aromatic
    )

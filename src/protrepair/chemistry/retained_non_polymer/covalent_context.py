"""Covalent attachments that a retained chemistry application must explain."""

from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING

from protrepair.structure.disulfide import disulfide_atom_ref_pair
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import BondRelationshipType, TopologyBond

if TYPE_CHECKING:
    from protrepair.structure.aggregate import ProteinStructure


@dataclass(frozen=True, slots=True, init=False)
class RetainedCovalentContext:
    """Bind external covalent or unresolved edges to their canonical endpoints.

    Parameters
    ----------
    structure : ProteinStructure
        Snapshot supplying current relationships, independent of provenance.
    residue_id : ResidueId
        Residue whose application must account for every attachment.

    Notes
    -----
    Typed hydrogen bonds and metal coordination are not covalent substitutions.
    Unknown relationships cannot be assumed to be either noncovalent or admitted.
    """

    attachments: tuple[tuple[TopologyBond, AtomRef, AtomRef], ...]
    _structure: "ProteinStructure" = field(repr=False)

    def __init__(self, structure: "ProteinStructure", residue_id: ResidueId) -> None:
        object.__setattr__(self, "_structure", structure)
        constitution = structure.constitution
        if constitution.residue_or_ligand(residue_id) is None:
            object.__setattr__(self, "attachments", ())
            return
        residue_index = constitution.residue_index(residue_id)
        attachments = []
        for bond in structure.topology.inter_residue_bonds_for_residue(
            constitution=constitution, residue_index=residue_index
        ):
            if bond.relationship_type in (
                BondRelationshipType.HYDROGEN_BOND,
                BondRelationshipType.METAL_COORDINATION,
            ):
                continue
            first, second = bond.endpoint_pair()
            inside_index, outside_index = (
                (first, second)
                if constitution.residue_index_for_atom_index(first) == residue_index
                else (second, first)
            )
            attachments.append(
                (
                    bond,
                    constitution.atom_ref_at(inside_index),
                    constitution.atom_ref_at(outside_index),
                )
            )
        object.__setattr__(self, "attachments", tuple(attachments))

    def has_exclusive_attachment(self, inside: AtomRef, outside: AtomRef) -> bool:
        """Check that one atom has exactly the given external covalent partner.

        Parameters
        ----------
        inside, outside : AtomRef
            Endpoint in this context and its required partner. Unknown edges
            occupy the same boundary; typed noncovalent edges do not.

        Returns
        -------
        bool
            Whether the endpoint has that partner and no other attachment.
        """
        return tuple(
            partner for _, endpoint, partner in self.attachments if endpoint == inside
        ) == (outside,)

    def require_neutral_disulfide_pair(self) -> tuple[AtomRef, AtomRef]:
        """Validate both endpoints of one canonical neutral disulfide.

        Returns
        -------
        tuple[AtomRef, AtomRef]
            Canonically ordered CYS SG endpoints with neutral source charges
            and one external covalent attachment per sulfur.

        Raises
        ------
        ValueError
            The context is not a single compatible disulfide, either endpoint
            has a conflicting charge, or the partner sulfur has another link.
        """
        if len(self.attachments) != 1:
            raise ValueError(
                "disulfide chemistry requires exactly one external attachment"
            )
        bond, inside, outside = self.attachments[0]
        pair = disulfide_atom_ref_pair(self._structure.constitution, bond)
        if pair is None:
            raise ValueError(
                "selected chemistry cannot represent external "
                f"{bond.relationship_type.value} relationship "
                f"{inside.display_token()}--{outside.display_token()}"
            )
        if bond.order not in (None, 1) or bond.aromatic:
            raise ValueError("disulfide chemistry requires a non-aromatic single link")
        bond.with_resolved_chemistry(replace(bond, order=1))

        for endpoint in (inside, outside):
            index = self._structure.constitution.atom_index(endpoint)
            if self._structure.constitution.atom_site_at(index).element != "S":
                raise ValueError(
                    "disulfide chemistry requires sulfur at both SG endpoints"
                )
            if self._structure.topology.formal_charge(index) not in (None, 0):
                raise ValueError(
                    "disulfide chemistry conflicts with a source sulfur charge"
                )
        partner = RetainedCovalentContext(self._structure, outside.residue_id)
        if not partner.has_exclusive_attachment(outside, inside):
            raise ValueError("disulfide chemistry requires one attachment per sulfur")
        return pair

    def require_admitted(
        self, endpoint_pairs: tuple[tuple[AtomRef, AtomRef], ...] = ()
    ) -> None:
        """Require every attachment to belong to the selected chemical grammar.

        Parameters
        ----------
        endpoint_pairs : tuple[tuple[AtomRef, AtomRef], ...]
            Already validated contextual links; empty requires isolation.

        Raises
        ------
        ValueError
            An external edge is unexplained by the selected application.
        """
        admitted = {frozenset(pair) for pair in endpoint_pairs}
        for bond, inside, outside in self.attachments:
            if frozenset((inside, outside)) not in admitted:
                raise ValueError(
                    "selected chemistry cannot represent external "
                    f"{bond.relationship_type.value} relationship "
                    f"{inside.display_token()}--{outside.display_token()}"
                )

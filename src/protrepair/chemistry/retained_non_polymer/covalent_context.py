"""Covalent attachments that a retained chemistry application must explain."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

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
        Retained residue whose application must account for every attachment.

    Notes
    -----
    Typed hydrogen bonds and metal coordination are not covalent substitutions.
    Unknown relationships cannot be assumed to be either noncovalent or admitted.
    """

    attachments: tuple[tuple[TopologyBond, AtomRef, AtomRef], ...]

    def __init__(self, structure: "ProteinStructure", residue_id: ResidueId) -> None:
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

"""Named, isolated orthophosphate models from the wwPDB Chemical Component Dictionary."""

from collections import Counter
from dataclasses import dataclass
from types import MappingProxyType

from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import (
    ResidueRole,
    ResidueSite,
    StructureConstitution,
)
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import BondRelationshipType, StructureTopology

_HEAVY_NAMES = ("P", "O1", "O2", "O3", "O4")
_PROTONATED_OXYGENS = MappingProxyType({"PO4": (), "PI": ("O4",), "2HP": ("O3", "O4")})


def is_orthophosphate_component(component_id: str) -> bool:
    """Recognize the three deposited inorganic orthophosphate component IDs."""
    return component_id in _PROTONATED_OXYGENS


def _require_source_isolation(
    structure: ProteinStructure, residue_id: ResidueId
) -> None:
    """Require a selected orthophosphate graph to account for its source partners.

    Parameters
    ----------
    structure : ProteinStructure
        Selected carrier with the original source observation, if available.
    residue_id : ResidueId
        Occurrence being prepared with explicit chemistry evidence.

    Raises
    ------
    ValueError
        A source declaration connects this phosphate to another residue. The
        isolated model cannot account for the partner even if it was dropped.
    """
    observation = structure.provenance.ingress.observation
    residue = structure.constitution.residue_or_ligand(residue_id)
    if (
        observation is None
        or residue is None
        or not is_orthophosphate_component(residue.component_id)
    ):
        return
    for link in (
        *observation.unresolved_non_polymer_links,
        *observation.excluded_coordination_connections,
    ):
        if residue_id not in (
            link.endpoint_1.atom_ref.residue_id,
            link.endpoint_2.atom_ref.residue_id,
        ):
            continue
        kept = link.lone_retained_endpoint(
            observation.constitution, observation.geometry
        )
        if kept is not None and kept.atom_ref.residue_id == residue_id:
            raise ValueError(
                "orthophosphate has an unrepresented "
                f"{link.source_metadata.reported_relationship_type.value} partner "
                f"at {kept.atom_ref.display_token()}"
            )

    for bond in observation.bonds_for_residue(residue_id):
        if bond.relationship_type is BondRelationshipType.HYDROGEN_BOND:
            continue
        a, b = (observation.constitution.atom_ref_at(i) for i in bond.endpoint_pair())
        if a.residue_id != b.residue_id:
            raise ValueError(
                f"orthophosphate has a declared {bond.relationship_type.value} partner: {a.display_token()}--{b.display_token()}"
            )


@dataclass(frozen=True, slots=True, init=False)
class DepositedOrthophosphate(RetainedNonPolymerChemistryEvidence):
    """Bind the selected CCD graph to one retained occurrence by atom name.

    Parameters
    ----------
    residue_id : ResidueId
        Retained site to prepare.
    component_id : str
        Deposited CCD identifier: PO4, PI or 2HP.
    """

    component_id: str

    def __init__(self, residue_id: ResidueId, component_id: str) -> None:
        protonated = _PROTONATED_OXYGENS[component_id]
        o2, o3, o4 = (
            "O" if name in protonated else "[O-]" for name in _HEAVY_NAMES[2:]
        )
        RetainedNonPolymerChemistryEvidence.__init__(
            self, residue_id, f"P(=O)({o2})({o3}){o4}", _HEAVY_NAMES
        )
        object.__setattr__(self, "component_id", component_id)

    @classmethod
    def for_residue(cls, residue: ResidueSite) -> "DepositedOrthophosphate | None":
        """Bind a named model for validation rather than skipping a malformed site.

        Parameters
        ----------
        residue : ResidueSite
            Occurrence selected for non-polymer retention.

        Returns
        -------
        DepositedOrthophosphate or None
            The model named by the component, or None for other identifiers.
        """
        if not is_orthophosphate_component(residue.component_id):
            return None
        return cls(residue.residue_id, residue.component_id)


@dataclass(frozen=True, slots=True)
class SelectedOrthophosphateGraph:
    """Named heavy atoms, charges, bonds and H capacities of the selected graph."""

    residue_id: ResidueId
    component_id: str
    elements: dict[str, str]
    formal_charges: dict[str, int]
    bond_orders: dict[frozenset[str], int]
    hydrogen_capacity: dict[str, int]

    def hydrogen_parents(self, structure: ProteinStructure) -> dict[str, str]:
        """Check current and original chemistry before binding observed hydrogen slots.

        Parameters
        ----------
        structure : ProteinStructure
            Current carrier with its original observation, when available.

        Returns
        -------
        dict[str, str]
            Present H/D/T names and their source-compatible oxygen parents.

        Raises
        ------
        ValueError
            The scaffold, isotope inventory, charges, attachments or named
            parent sites conflict with the selected model.
        """
        _require_source_isolation(structure, self.residue_id)
        current = self._snapshot_parents(structure.constitution, structure.topology)
        observation = structure.provenance.ingress.observation
        if observation is None:
            return current

        original = self._snapshot_parents(
            observation.constitution, observation.topology
        )
        for name, parent in original.items():
            ref = AtomRef(self.residue_id, name)
            if current.get(name) != parent or structure.constitution.resolve_atom_site(
                ref
            ) != observation.constitution.resolve_atom_site(ref):
                raise ValueError(
                    "orthophosphate model cannot replace or reparent source H/D/T"
                )
        return current

    def _snapshot_parents(
        self, constitution: StructureConstitution, topology: StructureTopology
    ) -> dict[str, str]:
        residue = constitution.residue_or_ligand(self.residue_id)
        if (
            residue is None
            or residue.component_id != self.component_id
            or constitution.role_at(constitution.residue_index(self.residue_id))
            is not ResidueRole.LIGAND
            or {
                (atom.name, atom.element)
                for atom in residue.atom_sites
                if not atom.is_hydrogen()
            }
            != set(self.elements.items())
        ):
            raise ValueError(
                f"selected {self.component_id} graph does not match the retained heavy atoms"
            )

        parents: dict[str, str] = {}
        for atom in residue.atom_sites:
            index = constitution.atom_index(AtomRef(self.residue_id, atom.name))
            if topology.formal_charge(index) not in (
                None,
                self.formal_charges.get(atom.name, 0),
            ):
                raise ValueError(
                    f"selected {self.component_id} graph conflicts with formal charge at {atom.name}"
                )

            attachments: set[str] = set()
            for bond in topology.incident_bonds_by_atom().get(index, ()):
                if bond.relationship_type is BondRelationshipType.HYDROGEN_BOND:
                    continue
                other = (
                    bond.atom_index_2
                    if index == bond.atom_index_1
                    else bond.atom_index_1
                )
                partner = constitution.atom_ref_at(other)
                if (
                    partner.residue_id != self.residue_id
                    or bond.relationship_type
                    not in (BondRelationshipType.COVALENT, BondRelationshipType.UNKNOWN)
                ):
                    raise ValueError(
                        f"selected {self.component_id} graph does not support this {bond.relationship_type.value} attachment at {atom.name}"
                    )
                if atom.is_hydrogen():
                    attachments.add(partner.atom_name)
                    order = 1
                elif constitution.atom_site_at(other).is_hydrogen():
                    continue  # The hydrogen endpoint owns this check.
                else:
                    pair = frozenset((atom.name, partner.atom_name))
                    if pair not in self.bond_orders:
                        raise ValueError(
                            f"unexpected bond at {atom.name} in selected {self.component_id}"
                        )
                    order = self.bond_orders[pair]
                if bond.aromatic or bond.order not in (None, order):
                    raise ValueError(
                        f"selected {self.component_id} graph conflicts with the bond order at {atom.name}"
                    )

            if atom.is_hydrogen():
                named = atom.name[1:] if atom.name[:1] in ("H", "D", "T") else ""
                if self.elements.get(named) == "O":
                    attachments.add(named)
                if len(attachments) != 1 or not attachments.issubset(
                    self.hydrogen_capacity
                ):
                    raise ValueError(
                        f"selected {self.component_id} graph cannot preserve the oxygen parent of {atom.name}"
                    )
                parents[atom.name] = attachments.pop()
        if any(
            count > self.hydrogen_capacity[parent]
            for parent, count in Counter(parents.values()).items()
        ):
            raise ValueError(
                f"selected {self.component_id} graph cannot preserve all source H/D/T at their oxygen parents"
            )
        return parents

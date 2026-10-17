"""Mapped finite coordination hypotheses with graph-qualified chemical roles."""

from dataclasses import dataclass, field

from protrepair.chemistry.inference.coordination_donor import NitrogenCoordinationDonor
from protrepair.chemistry.inference.rdkit_graph import (
    AROMATIC_N_COORDINATION_METALS,
    native_bond_projection_from_molecule,
)
from protrepair.chemistry.inference.retained_non_polymer_evidence import (
    template_without_hydrogens,
)
from protrepair.chemistry.retained_non_polymer.evidence import (
    RetainedNonPolymerChemistryEvidence,
)
from protrepair.structure.labels import ResidueId


@dataclass(frozen=True, slots=True)
class CoordinationMicrostates:
    """Validate a finite mapped family without selecting its source microstate.

    Parameters
    ----------
    heavy_atom_names : tuple[str, ...]
        Names in the same heavy-atom traversal order for every hypothesis.
    smiles_states : tuple[str, ...]
        Source-backed preparation alternatives, in declared preference order.
    provenance : str
        Attribution and scope of the supplied chemical assumptions.

    Raises
    ------
    ValueError
        States do not share a connected named scaffold, stereo or typed
        coordination, or have no admitted coordination role.
    RdkitUnavailableError
        Native graph interpretation is unavailable.

    Notes
    -----
    Registration supplies chemical knowledge, not a pKa or metal-state
    prediction. Graph roles qualify processing; component identifiers do not.
    Stereo-bearing states must also share their named stereo-defining references.
    RDKit radical flags do not describe a metal's physical spin state.
    """

    heavy_atom_names: tuple[str, ...]
    smiles_states: tuple[str, ...]
    provenance: str
    metal_atom_names: frozenset[str] = field(init=False)
    has_internal_coordination: bool = field(init=False)
    _donors_by_state: tuple[tuple[tuple[str, NitrogenCoordinationDonor], ...], ...] = (
        field(init=False, repr=False)
    )
    _charges_by_state: tuple[tuple[int, ...], ...] = field(init=False, repr=False)

    def __post_init__(self) -> None:
        names = tuple(name.strip().upper() for name in self.heavy_atom_names)
        states = tuple(smiles.strip() for smiles in self.smiles_states)
        if (
            not names
            or any(not name for name in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("coordination family requires unique nonblank atom names")
        if not states or any(not smiles for smiles in states):
            raise ValueError("coordination family requires nonblank microstates")
        if len(set(states)) != len(states):
            raise ValueError("coordination family must not repeat a microstate")
        provenance = self.provenance.strip()
        if not provenance:
            raise ValueError("coordination family requires chemical provenance")

        templates = tuple(template_without_hydrogens(smiles) for smiles in states)
        signatures = []
        donors = []
        charges = []
        for template in templates:
            if template.GetNumAtoms() != len(names):
                raise ValueError("coordination atom mapping does not match its graph")
            if any(atom.GetNumRadicalElectrons() for atom in template.GetAtoms()):
                raise ValueError(
                    "coordination microstates do not support explicit radicals"
                )
            projection = native_bond_projection_from_molecule(
                template, atom_names_by_index=dict(enumerate(names))
            )
            covalent = frozenset(
                tuple(sorted((bond.atom_name_1, bond.atom_name_2)))
                for bond in projection.covalent_bonds
            )
            coordination = frozenset(
                (bond.donor_atom_name, bond.acceptor_atom_name)
                for bond in projection.coordination_bonds
            )
            elements = tuple(atom.GetSymbol().upper() for atom in template.GetAtoms())
            stereo = (
                tuple(
                    (
                        int(atom.GetChiralTag()),
                        tuple(names[n.GetIdx()] for n in atom.GetNeighbors())
                        if atom.GetChiralTag()
                        else (),
                    )
                    for atom in template.GetAtoms()
                ),
                tuple(
                    sorted(
                        (
                            tuple(
                                sorted(
                                    (
                                        names[bond.GetBeginAtomIdx()],
                                        names[bond.GetEndAtomIdx()],
                                    )
                                )
                            ),
                            int(bond.GetStereo()),
                            tuple(names[i] for i in bond.GetStereoAtoms()),
                        )
                        for bond in template.GetBonds()
                    )
                ),
            )
            signatures.append((elements, covalent, coordination, stereo))
            adjacency: dict[str, set[str]] = {name: set() for name in names}
            for first, second in (*covalent, *coordination):
                adjacency[first].add(second)
                adjacency[second].add(first)
            seen = {names[0]}
            pending = [names[0]]
            while pending:
                for neighbor in adjacency[pending.pop()] - seen:
                    seen.add(neighbor)
                    pending.append(neighbor)
            if len(seen) != len(names):
                raise ValueError("coordination family requires a connected scaffold")
            accepted = {}
            for index, name in enumerate(names):
                try:
                    role = NitrogenCoordinationDonor.from_molecule(template, index)
                except ValueError:
                    continue
                accepted[name] = role
            if any(
                donor not in accepted
                or not accepted[donor].admits_metal(elements[names.index(acceptor)])
                for donor, acceptor in coordination
            ):
                raise ValueError(
                    "internal coordination has an incompatible nitrogen donor/metal"
                )
            donors.append(tuple(accepted.items()))
            charges.append(
                tuple(atom.GetFormalCharge() for atom in template.GetAtoms())
            )
        if any(signature != signatures[0] for signature in signatures[1:]):
            raise ValueError(
                "coordination microstates change the mapped heavy scaffold"
            )
        elements, _, coordination, _ = signatures[0]
        metals = frozenset(
            name
            for name, element in zip(names, elements, strict=True)
            if element in AROMATIC_N_COORDINATION_METALS
        )
        if metals != frozenset(acceptor for _, acceptor in coordination):
            raise ValueError(
                "metal roles require typed internal coordination acceptors"
            )
        if not coordination and not any(donors):
            raise ValueError("coordination family has no admitted nitrogen donor")
        object.__setattr__(self, "heavy_atom_names", names)
        object.__setattr__(self, "smiles_states", states)
        object.__setattr__(self, "provenance", provenance)
        object.__setattr__(self, "metal_atom_names", metals)
        object.__setattr__(self, "has_internal_coordination", bool(coordination))
        object.__setattr__(self, "_donors_by_state", tuple(donors))
        object.__setattr__(self, "_charges_by_state", tuple(charges))

    def admits_donor(self, atom_name: str, *, metal_element: str | None = None) -> bool:
        """Return whether an alternative supplies the mapped donor/metal role.

        Parameters
        ----------
        atom_name : str
            Canonical source endpoint, not an assumed atom label.
        metal_element : str or None, optional
            Canonical partner element, or None to query donor capability alone.

        Returns
        -------
        bool
            True only for an admitted graph-qualified donor and optional partner.
        """
        return any(
            name == atom_name
            and (metal_element is None or role.admits_metal(metal_element))
            for donors in self._donors_by_state
            for name, role in donors
        )

    def hypotheses(
        self, residue_id: ResidueId, *, donor_atom_name: str | None = None
    ) -> tuple[RetainedNonPolymerChemistryEvidence, ...]:
        """Bind evidence to a residue and its declared external donor.

        Parameters
        ----------
        residue_id : ResidueId
            Current source identity.
        donor_atom_name : str or None, optional
            Required single nitrogen donor, or None for internal metal chemistry.

        Returns
        -------
        tuple[RetainedNonPolymerChemistryEvidence, ...]
            Complete mapped alternatives, neutral donor before anionic donor.
            Equal priorities preserve the catalogue's declared ordering.

        Raises
        ------
        ValueError
            An external-donor family has no matching typed donor endpoint.
        """
        selected = list(range(len(self.smiles_states)))
        if not self.has_internal_coordination:
            if donor_atom_name is None or not self.admits_donor(donor_atom_name):
                raise ValueError("coordination family requires an admitted donor")
            selected = [
                i
                for i in selected
                if any(name == donor_atom_name for name, _ in self._donors_by_state[i])
            ]
            donor_index = self.heavy_atom_names.index(donor_atom_name)
            selected.sort(key=lambda i: self._charges_by_state[i][donor_index] == -1)
        return tuple(
            RetainedNonPolymerChemistryEvidence(
                residue_id, self.smiles_states[index], self.heavy_atom_names
            )
            for index in selected
        )

"""Bind retained amino-acid occurrences to the shared site catalogs."""

from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import product
from math import floor, isfinite

from protrepair.chemistry.component.defaults import build_default_component_library
from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.inference.idealized_stereochemistry import (
    require_compatible_component_pose,
)
from protrepair.chemistry.microstate.catalog import (
    PeptideLinkage,
    PolymerChemicalSite,
    standard_microstate_candidates,
)
from protrepair.chemistry.microstate.context import AminoAcidMicrostateContext
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.resolution import MicrostateConstraints
from protrepair.chemistry.retained_non_polymer.covalent_context import (
    RetainedCovalentContext,
)
from protrepair.geometry import Vec3
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.peptide import PEPTIDE_CN_DISTANCE_MAX_ANGSTROM
from protrepair.structure.topology import BondRelationshipType, TopologyBond


@dataclass(frozen=True, slots=True, eq=False)
class RetainedAminoAcidContext(AminoAcidMicrostateContext):
    """Constrain retained sites by declared chemistry, not inferred connectivity.

    Parameters
    ----------
    source : ProteinStructure
        Immutable source/current snapshot. Membership is never rewritten.
    """

    _compatible_scaffolds: dict[ResidueId, ResidueTemplate] = field(
        default_factory=dict, init=False, repr=False, compare=False
    )
    _peptide_endpoint_cells: dict[tuple[int, int, int], tuple[AtomRef, ...]] | None = (
        field(default=None, init=False, repr=False, compare=False)
    )

    def _has_peptide_proximity(self, endpoint: AtomRef) -> bool:
        index = self.source.constitution.resolve_atom_index(endpoint)
        if index is None:
            return False
        if self.source.constitution.atom_site_at(index).element != endpoint.atom_name:
            return False
        position = self.source.geometry.position(index)
        if not all(isfinite(value) for value in position):
            return True
        cells = self._peptide_endpoint_cells
        if cells is None:
            indexed = defaultdict(list)
            for other_index, other_position in self.source.geometry.iter_positions():
                atom = self.source.constitution.atom_site_at(other_index)
                if (
                    atom.name not in {"C", "N"}
                    or atom.element != atom.name
                    or not all(isfinite(value) for value in other_position)
                ):
                    continue
                indexed[_peptide_endpoint_cell(other_position)].append(
                    self.source.constitution.atom_ref_at(other_index)
                )
            cells = {key: tuple(refs) for key, refs in indexed.items()}
            object.__setattr__(self, "_peptide_endpoint_cells", cells)

        center = _peptide_endpoint_cell(position)
        opposite = "C" if endpoint.atom_name == "N" else "N"
        for offset in product((-1, 0, 1), repeat=3):
            key = (center[0] + offset[0], center[1] + offset[1], center[2] + offset[2])
            for other in cells.get(key, ()):
                if (
                    other.residue_id == endpoint.residue_id
                    or other.atom_name != opposite
                ):
                    continue
                other_position = self.source.geometry.position(
                    self.source.constitution.atom_index(other)
                )
                if (
                    position.distance_to(other_position)
                    <= PEPTIDE_CN_DISTANCE_MAX_ANGSTROM
                ):
                    return True
        return False

    def site_requirements(
        self, residue_id: ResidueId, site: PolymerMicrostateSite
    ) -> MicrostateConstraints:
        """Combine physical feasibility and current explicit declarations.

        Parameters
        ----------
        residue_id : ResidueId
            Retained occurrence in this snapshot.
        site : PolymerMicrostateSite
            Coupled chemistry selected by retained admission.

        Returns
        -------
        MicrostateConstraints
            Known charges, attached-H lower bounds and source bond orders.
            These requirements do not manufacture an original observation.

        Raises
        ------
        ValueError
            Current H or physical relationships cannot be interpreted.
        """
        physical = self.coordination_requirements(residue_id, site)
        self.require_compatible_scaffold(residue_id, site.template)
        catalog = site.candidates()
        if catalog is None:
            return physical
        names = {atom.name for atom in catalog.candidates[0].atoms}
        saved = next(
            (
                choice.graph
                for choice in self._choices_by_residue.get(residue_id, ())
                if names == {atom.name for atom in choice.graph.atoms}
                and choice.component_id == site.template.component_id
            ),
            None,
        )
        constitution = self.source.constitution
        charges = tuple(
            (name, charge)
            for name in sorted(names)
            if (index := constitution.resolve_atom_index(AtomRef(residue_id, name)))
            is not None
            and (charge := self.source.topology.formal_charge(index)) is not None
            and (saved is None or saved.atom(name).charge != charge)
        )
        parents = self.hydrogen_parents(residue_id, site)
        counts = Counter(parent for parent in parents.values() if parent in names)
        if saved is not None:
            counts = Counter(
                {
                    parent: count
                    for parent, count in counts.items()
                    if count > saved.atom(parent).hydrogens
                }
            )
        bonds = tuple(
            BondDefinition(first, second, order=order)
            for (
                first,
                second,
            ), order in self.source.topology.residue_source_heavy_bond_orders(
                constitution=constitution,
                residue_index=constitution.residue_index(residue_id),
            ).items()
            if first in names and second in names
        )
        return MicrostateConstraints(
            charges=charges,
            hydrogens=physical.hydrogens,
            minimum_hydrogens=tuple(counts.items()),
            bonds=bonds,
        )

    def existing_hydrogen_preferences(
        self, residue_id: ResidueId, site: PolymerMicrostateSite
    ) -> tuple[MicrostateConstraints, ...]:
        """Prefer a valid existing H inventory without claiming exact evidence.

        Parameters
        ----------
        residue_id : ResidueId
            Retained occurrence in this snapshot.
        site : PolymerMicrostateSite
            Catalogued chemical site whose source parents are interpreted.

        Returns
        -------
        tuple[MicrostateConstraints, ...]
            One preference matching the H-bearing site's parent counts, or
            empty for absent, uninterpretable or nonmatching inventories.
            Hard chemistry requirements still filter candidates first.
            H from an applied choice is not
            promoted into fresh source evidence.
        """
        catalog = site.candidates()
        if catalog is None:
            return ()
        names = {atom.name for atom in catalog.candidates[0].atoms}
        try:
            parents = self.hydrogen_parents(residue_id, site, original=True)
            if not any(
                any(atom.name in names for atom in choice.graph.atoms)
                for choice in self._choices_by_residue.get(residue_id, ())
            ):
                parents.update(self.hydrogen_parents(residue_id, site))
        except ValueError:
            return ()
        counts = Counter(parent for parent in parents.values() if parent in names)
        if not counts:
            return ()
        preference = MicrostateConstraints(
            hydrogens=tuple((name, counts[name]) for name in sorted(names))
        )
        return (
            (preference,)
            if any(
                preference.violations(graph).is_empty() for graph in catalog.candidates
            )
            else ()
        )

    def require_compatible_scaffold(
        self, residue_id: ResidueId, template: ResidueTemplate
    ) -> None:
        """Reject body conflicts without choosing terminal or standard-site states.

        Parameters
        ----------
        residue_id : ResidueId
            Retained occurrence in this snapshot.
        template : ResidueTemplate
            Stock H-capable amino-acid template.

        Raises
        ------
        ValueError
            Reported body orders, declared charges or reference stereo conflict.
        RdkitUnavailableError
            Native reference stereo validation is unavailable.
        """
        constitution = self.source.constitution
        index = constitution.residue_index(residue_id)
        if index.value < len(constitution.residue_slots) - len(constitution.ligands):
            raise ValueError(
                "retained amino-acid chemistry requires retained membership"
            )
        if self._compatible_scaffolds.get(residue_id) is template:
            return
        if constitution.residue_site_at(index).component_id != template.component_id:
            raise ValueError(
                "source component alias needs explicit chemical reconciliation"
            )
        observation = self.source.provenance.ingress.observation
        if observation is not None:
            current_external = {
                frozenset(
                    constitution.atom_ref_at(i) for i in bond.endpoint_pair()
                ): bond.relationship_type
                for atom in constitution.residue_site_at(index).atom_sites
                if not atom.is_hydrogen()
                for bond in self.external_heavy_bonds(AtomRef(residue_id, atom.name))
            }
            for bond in observation.bonds_for_residue(residue_id):
                refs = tuple(
                    observation.constitution.atom_ref_at(i)
                    for i in bond.endpoint_pair()
                )
                if (
                    refs[0].residue_id != refs[1].residue_id
                    and bond.relationship_type
                    in (
                        BondRelationshipType.COVALENT,
                        BondRelationshipType.DISULFIDE,
                        BondRelationshipType.UNKNOWN,
                        BondRelationshipType.METAL_COORDINATION,
                    )
                    and not any(
                        observation.constitution.atom_site_at(i).is_hydrogen()
                        for i in bond.endpoint_pair()
                    )
                    and (
                        current_external.get(frozenset(refs))
                        in (None, BondRelationshipType.HYDROGEN_BOND)
                        or (
                            bond.relationship_type
                            in (
                                BondRelationshipType.COVALENT,
                                BondRelationshipType.DISULFIDE,
                            )
                            and current_external.get(frozenset(refs))
                            not in (
                                BondRelationshipType.COVALENT,
                                BondRelationshipType.DISULFIDE,
                            )
                        )
                        or (
                            bond.relationship_type
                            is BondRelationshipType.METAL_COORDINATION
                            and current_external.get(frozenset(refs))
                            is not BondRelationshipType.METAL_COORDINATION
                        )
                    )
                ):
                    raise ValueError(
                        "original external amino-acid relationship is missing from current context"
                    )
        groups = [{"N"}, {"C", "O", "OXT"}]
        catalog = standard_microstate_candidates(
            template.component_id, PolymerChemicalSite.SIDECHAIN
        )
        if catalog is not None:
            groups.append({atom.name for atom in catalog.candidates[0].atoms})
        body_orders = {
            tuple(sorted((bond.atom_name_1, bond.atom_name_2))): bond.order
            for bond in template.definition.bonds
        }
        for definition in template.definition.bonds:
            pair = {definition.atom_name_1, definition.atom_name_2}
            if any(pair <= group for group in groups):
                continue
            first = constitution.resolve_atom_index(
                AtomRef(residue_id, definition.atom_name_1)
            )
            second = constitution.resolve_atom_index(
                AtomRef(residue_id, definition.atom_name_2)
            )
            if first is None or second is None:
                continue
            existing = self.source.topology.bond_between(first, second)
            if existing is not None:
                existing.with_resolved_chemistry(
                    TopologyBond(
                        first,
                        second,
                        order=definition.order,
                        aromatic=definition.aromatic,
                    )
                )
        for pair, order in self.source.topology.residue_source_heavy_bond_orders(
            constitution=constitution,
            residue_index=index,
        ).items():
            pair = tuple(sorted(pair))
            if (
                not any(set(pair) <= group for group in groups)
                and body_orders.get(pair) != order
            ):
                raise ValueError(
                    f"source bond order conflicts with amino-acid body at {pair}"
                )
        controlled = set().union(*groups)
        for name, charge in template.heavy_atom_charge_requirements():
            if name in controlled:
                continue
            atom_index = constitution.resolve_atom_index(AtomRef(residue_id, name))
            if atom_index is not None and self.source.topology.formal_charge(
                atom_index
            ) not in (None, charge):
                raise ValueError(
                    f"source charge conflicts with amino-acid body at {name}"
                )
        reference = template.reference_component
        if reference is not None:
            require_compatible_component_pose(
                reference,
                self.source.residue_geometry(index),
                hydrogen_attachments=tuple(
                    self.hydrogen_parents(
                        residue_id,
                        PolymerMicrostateSite(template, PolymerChemicalSite.SIDECHAIN),
                    ).items()
                ),
                bond_order_constraints={},
                allow_partial=True,
            )
        self._compatible_scaffolds[residue_id] = template

    def admitted_sites(
        self, residue_id: ResidueId, template: ResidueTemplate
    ) -> tuple[PolymerMicrostateSite, ...]:
        """Bind stock H-capable amino acids without guessing external products.

        Parameters
        ----------
        residue_id : ResidueId
            Retained occurrence in this snapshot.
        template : ResidueTemplate
            Active component definition; custom replacements retain their owner.

        Returns
        -------
        tuple[PolymerMicrostateSite, ...]
            Standard side-chain site and free/unknown terminal sites. Empty
            means this shell does not own preparation, not H inapplicability.

        Raises
        ------
        ValueError
            The occurrence is absent or is polymer.
        """
        constitution = self.source.constitution
        index = constitution.residue_index(residue_id)
        if index.value < len(constitution.residue_slots) - len(constitution.ligands):
            raise ValueError("retained amino-acid sites require retained membership")
        if (
            not template.can_add_hydrogens()
            or template != build_default_component_library().get(template.component_id)
            or standard_microstate_candidates(
                template.backbone_family_component_id,
                PolymerChemicalSite.BACKBONE_N,
                PeptideLinkage.FREE,
            )
            is None
        ):
            return ()
        sites = []
        covalent_sulfur = (
            template.component_id == "CYS"
            and any(
                bond.relationship_type
                not in (
                    BondRelationshipType.HYDROGEN_BOND,
                    BondRelationshipType.METAL_COORDINATION,
                )
                for bond in self.external_heavy_bonds(AtomRef(residue_id, "SG"))
            )
            if constitution.resolve_atom_index(AtomRef(residue_id, "SG")) is not None
            else False
        )
        if covalent_sulfur:
            try:
                RetainedCovalentContext(
                    self.source, residue_id
                ).require_neutral_disulfide_pair()
            except ValueError:
                return ()
        sulfur_index = constitution.resolve_atom_index(AtomRef(residue_id, "SG"))
        variable_sulfur = template.component_id != "CYS" or (
            sulfur_index is not None
            and (
                self.source.topology.formal_charge(sulfur_index) not in (None, 0)
                or any(
                    bond.relationship_type is BondRelationshipType.METAL_COORDINATION
                    for bond in self.external_heavy_bonds(AtomRef(residue_id, "SG"))
                )
                or (
                    (original := self.source.provenance.ingress.observation) is not None
                    and any(
                        bond.relationship_type is not BondRelationshipType.HYDROGEN_BOND
                        and any(
                            original.constitution.atom_ref_at(i).residue_id
                            != residue_id
                            for i in bond.endpoint_pair()
                        )
                        for bond in original.bonds_for_atom(AtomRef(residue_id, "SG"))
                    )
                )
            )
        )
        if (
            standard_microstate_candidates(
                template.component_id, PolymerChemicalSite.SIDECHAIN
            )
            is not None
            and not covalent_sulfur
            and variable_sulfur
        ):
            sites.append(PolymerMicrostateSite(template, PolymerChemicalSite.SIDECHAIN))
        observation = self.source.provenance.ingress.observation
        for kind, names in (
            (PolymerChemicalSite.BACKBONE_N, ("N",)),
            (PolymerChemicalSite.BACKBONE_C, ("C", "O", "OXT")),
        ):
            unexplained = False
            for name in names:
                ref = AtomRef(residue_id, name)
                if constitution.resolve_atom_index(ref) is not None:
                    unexplained |= any(
                        bond.relationship_type is not BondRelationshipType.HYDROGEN_BOND
                        for bond in self.external_heavy_bonds(ref)
                    )
                if observation is not None:
                    unexplained |= any(
                        bond.relationship_type is not BondRelationshipType.HYDROGEN_BOND
                        and any(
                            observation.constitution.atom_ref_at(i).residue_id
                            != residue_id
                            and not observation.constitution.atom_site_at(
                                i
                            ).is_hydrogen()
                            for i in bond.endpoint_pair()
                        )
                        for bond in observation.bonds_for_atom(ref)
                    )
            if not unexplained:
                unexplained = self._has_peptide_proximity(
                    AtomRef(
                        residue_id,
                        "N" if kind is PolymerChemicalSite.BACKBONE_N else "C",
                    )
                )
            sites.append(
                PolymerMicrostateSite(
                    template,
                    kind,
                    PeptideLinkage.UNKNOWN if unexplained else PeptideLinkage.FREE,
                    free_terminal_assumption=not unexplained,
                )
            )
        return tuple(sites)


def _peptide_endpoint_cell(position: Vec3) -> tuple[int, int, int]:
    return (
        floor(position.x / PEPTIDE_CN_DISTANCE_MAX_ANGSTROM),
        floor(position.y / PEPTIDE_CN_DISTANCE_MAX_ANGSTROM),
        floor(position.z / PEPTIDE_CN_DISTANCE_MAX_ANGSTROM),
    )

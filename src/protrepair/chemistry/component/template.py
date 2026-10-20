"""Residue-template aggregates over chemistry graphs and completion semantics."""

from collections import deque
from collections.abc import Collection, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

from protrepair.chemistry.component.graph import (
    ChemicalComponentDefinition,
    ForceFieldAtomParams,
    TetrahedralCenterSemantics,
)
from protrepair.chemistry.component.semantics import (
    HeavyAtomSemantics,
    HydrogenSemantics,
    IdealGeometryHeavyAtomSemantics,
    IdealGeometryHydrogenSemantics,
)
from protrepair.structure.constitution import ResidueSite


@dataclass(frozen=True, slots=True)
class ResidueTemplate:
    """Residue template combining chemistry graph and completion semantics."""

    definition: ChemicalComponentDefinition
    lineage_parent_component_id: str | None = None
    backbone_family_component_id: str = ""
    forcefield_parameters: Mapping[str, ForceFieldAtomParams] = field(
        default_factory=dict
    )
    preferred_atom_order: tuple[str, ...] = ()
    heavy_atom_semantics: (
        HeavyAtomSemantics | IdealGeometryHeavyAtomSemantics | None
    ) = None
    hydrogen_semantics: HydrogenSemantics | IdealGeometryHydrogenSemantics | None = None
    tetrahedral_centers: tuple[TetrahedralCenterSemantics, ...] = ()
    _polymer_hydrogen_parent_aliases: Mapping[str, str] | None = field(
        default=None, init=False, compare=False, hash=False, repr=False
    )

    def __post_init__(self) -> None:
        normalized_params = {
            atom_name.strip().upper(): params
            for atom_name, params in self.forcefield_parameters.items()
        }
        preferred_atom_order = tuple(
            atom_name.strip().upper() for atom_name in self.preferred_atom_order
        )
        lineage_parent_component_id = self.lineage_parent_component_id
        if lineage_parent_component_id is not None:
            lineage_parent_component_id = (
                lineage_parent_component_id.strip().upper() or None
            )

        backbone_family_component_id = (
            self.backbone_family_component_id.strip().upper()
            if self.backbone_family_component_id.strip()
            else self.definition.component_id
        )

        object.__setattr__(
            self,
            "lineage_parent_component_id",
            lineage_parent_component_id,
        )
        object.__setattr__(
            self,
            "backbone_family_component_id",
            backbone_family_component_id,
        )
        object.__setattr__(
            self,
            "forcefield_parameters",
            MappingProxyType(normalized_params),
        )
        object.__setattr__(self, "preferred_atom_order", preferred_atom_order)
        object.__setattr__(
            self,
            "tetrahedral_centers",
            tuple(self.tetrahedral_centers),
        )

    def ordered_atom_names(self) -> tuple[str, ...]:
        """Return the preferred atom order for serialization or comparison."""

        return self.preferred_atom_order or self.definition.expected_atom_names()

    @property
    def component_id(self) -> str:
        """Return the canonical component identifier for the template."""

        return self.definition.component_id

    @property
    def aliases(self) -> tuple[str, ...]:
        """Return all aliases accepted for this template."""

        return self.definition.aliases

    def expected_atom_names(self) -> tuple[str, ...]:
        """Return the expected canonical atom order for the component."""

        return self.definition.expected_atom_names()

    def expected_heavy_atom_names(self) -> tuple[str, ...]:
        """Return the complete component heavy inventory, including ligand OXT.

        Returns
        -------
        tuple[str, ...]
            Declared heavy atoms independent of polymer context or which atoms
            a placement program can reconstruct.
        """

        if isinstance(
            self.heavy_atom_semantics,
            IdealGeometryHeavyAtomSemantics,
        ):
            return self.heavy_atom_semantics.component.heavy_atom_names()

        if isinstance(self.hydrogen_semantics, IdealGeometryHydrogenSemantics):
            return self.hydrogen_semantics.component.heavy_atom_names()

        hydrogen_names = frozenset(self.expected_hydrogen_atom_names())
        return tuple(
            atom_name
            for atom_name in self.expected_atom_names()
            if atom_name not in hydrogen_names
        )

    def expected_polymer_heavy_atom_names(self) -> tuple[str, ...]:
        """Project polymer scaffold atoms without a free-terminal oxygen.

        Returns
        -------
        tuple[str, ...]
            Component heavy inventory excluding OXT. A selected free-carbonyl
            microstate separately requires its terminal oxygen; retained
            components must use the complete component inventory instead.
        """
        return tuple(name for name in self.expected_heavy_atom_names() if name != "OXT")

    def expected_hydrogen_atom_names(self) -> tuple[str, ...]:
        """Return expected hydrogen atoms for hydrogen-completeness workflows."""

        hydrogen_semantics = self.hydrogen_semantics
        if hydrogen_semantics is None:
            return ()

        return hydrogen_semantics.expected_hydrogen_atom_names()

    def heavy_atom_charge_requirements(self) -> tuple[tuple[str, int], ...]:
        """Return the component's declared heavy-atom charge assignment.

        Returns
        -------
        tuple[tuple[str, int], ...]
            Heavy atom names and integral formal charges, including neutral atoms
            in a template with charge declarations. Empty when it declares none;
            that absence is not evidence overriding a source microstate.
        """
        if not self.definition.formal_charges:
            return ()
        return tuple(
            (name, self.definition.formal_charges.get(name, 0))
            for name in self.expected_heavy_atom_names()
        )

    def has_compatible_heavy_scaffold(self, residue: ResidueSite) -> bool:
        """Check present heavy atoms without certifying completeness.

        Parameters
        ----------
        residue : ResidueSite
            Materialized residue being compared with this template.

        Returns
        -------
        bool
            Whether present heavy atoms are a subset of the expected inventory
            with compatible ideal-component elements, when available.
        """
        present_names = {
            atom.name for atom in residue.atom_sites if not atom.is_hydrogen()
        }
        if not present_names.issubset(self.expected_heavy_atom_names()):
            return False
        semantics = self.hydrogen_semantics
        return not isinstance(semantics, IdealGeometryHydrogenSemantics) or all(
            atom.atom_name not in present_names
            or residue.atom_site(atom.atom_name).element == atom.element
            for atom in semantics.component.heavy_atoms()
        )

    def applicable_heavy_atom_charge_requirements(
        self, residue: ResidueSite
    ) -> tuple[tuple[str, int], ...]:
        """Project charge assignments supported by the present local scaffold.

        Parameters
        ----------
        residue : ResidueSite
            Compatible materialized component, possibly missing heavy atoms.

        Returns
        -------
        tuple[tuple[str, int], ...]
            Requirements for present atoms. A nonzero charge also requires all
            declared heavy neighbors. This structural applicability check does
            not certify a fragment's chemistry. Known source charges must still
            be checked against the complete selected requirements.
        """
        present = frozenset(
            atom.name for atom in residue.atom_sites if not atom.is_hydrogen()
        )
        heavy_names = frozenset(self.expected_heavy_atom_names())
        return tuple(
            (name, charge)
            for name, charge in self.heavy_atom_charge_requirements()
            if name in present
            and (
                charge == 0
                or self.definition.bonded_atom_names(name).intersection(heavy_names)
                <= present
            )
        )

    def has_complete_heavy_scaffold(self, residue: ResidueSite) -> bool:
        """Check exact heavy inventory and available element evidence.

        Parameters
        ----------
        residue : ResidueSite
            Materialized residue being compared with this template.

        Returns
        -------
        bool
            Whether all and only the expected heavy atoms are present with
            compatible ideal-component elements, when available.
        """
        return sum(not atom.is_hydrogen() for atom in residue.atom_sites) == len(
            self.expected_heavy_atom_names()
        ) and self.has_compatible_heavy_scaffold(residue)

    def missing_atom_names(
        self,
        present_atom_names: Collection[str],
        *,
        exclude_atom_names: Collection[str] = (),
    ) -> tuple[str, ...]:
        """Return missing template atoms for a given present-atom set."""

        present = {atom_name.strip().upper() for atom_name in present_atom_names}
        excluded = {atom_name.strip().upper() for atom_name in exclude_atom_names}
        return tuple(
            atom_name
            for atom_name in self.expected_atom_names()
            if atom_name not in present and atom_name not in excluded
        )

    def can_repair_heavy_atoms(self) -> bool:
        """Return whether heavy-atom repair semantics are available."""

        return self.heavy_atom_semantics is not None

    def can_add_hydrogens(self) -> bool:
        """Return whether hydrogen-placement semantics are available."""

        return self.hydrogen_semantics is not None

    def template_hydrogen_anchor_by_name(
        self,
        present_atom_names: Collection[str],
    ) -> Mapping[str, str]:
        """Return explicit template-resolved anchors for present hydrogens."""

        normalized_present_atom_names = {
            atom_name.strip().upper() for atom_name in present_atom_names
        }
        hydrogen_semantics = self.hydrogen_semantics
        if hydrogen_semantics is None:
            return MappingProxyType({})

        anchor_by_name = hydrogen_semantics.template_hydrogen_anchor_by_name()
        return MappingProxyType(
            {
                hydrogen_atom_name: anchor_atom_name
                for hydrogen_atom_name, anchor_atom_name in anchor_by_name.items()
                if hydrogen_atom_name in normalized_present_atom_names
            }
        )

    def polymer_hydrogen_parent_aliases(self) -> Mapping[str, str]:
        """Interpret fixed polymer H naming rules, not observed attachments.

        Returns
        -------
        Mapping[str, str]
            Immutable template anchors, terminal aliases and unambiguous carbon
            families. Isotope normalization and actual bonds remain live queries.
            Linked/free backbone N naming is not part of this projection.

        Raises
        ------
        ValueError
            The complete hydrogen grammar has conflicting anchor declarations.
        """
        cached = self._polymer_hydrogen_parent_aliases
        if cached is not None:
            return cached
        aliases = {
            "H": "N",
            "H1": "N",
            "H2": "N",
            "H3": "N",
            "1H": "N",
            "2H": "N",
            "3H": "N",
            "HXT": "OXT",
            "HO": "O",
        }
        if self.component_id == "HIS":
            aliases.update({"HD1": "ND1", "HE2": "NE2"})
        elif self.component_id == "ARG":
            aliases.update({"HH11": "NH1", "HH12": "NH1", "HH21": "NH2", "HH22": "NH2"})
        elif self.component_id == "ASN":
            aliases.update({"HD21": "ND2", "HD22": "ND2"})
        elif self.component_id == "GLN":
            aliases.update({"HE21": "NE2", "HE22": "NE2"})
        elif self.component_id == "ASP":
            aliases.update({"HD1": "OD1", "HD2": "OD2"})
        elif self.component_id == "GLU":
            aliases.update({"HE1": "OE1", "HE2": "OE2"})
        if self.hydrogen_semantics is not None:
            aliases.update(self.hydrogen_semantics.template_hydrogen_anchor_by_name())

        # Ambiguous carbon families stay unresolved; declared anchors win.
        carbon_parents: dict[str, set[str]] = {}
        for atom_name in self.expected_polymer_heavy_atom_names():
            if not atom_name.startswith("C"):
                continue
            stem = "H" + atom_name[1:]
            for name in (
                stem,
                *(f"{stem}{i}" for i in (1, 2, 3)),
                *(f"{i}{stem}" for i in (1, 2, 3)),
            ):
                carbon_parents.setdefault(name, set()).add(atom_name)
        for name, parents in carbon_parents.items():
            if len(parents) == 1:
                aliases.setdefault(name, next(iter(parents)))
        result = MappingProxyType(aliases)
        object.__setattr__(self, "_polymer_hydrogen_parent_aliases", result)
        return result

    def requires_coordinate_hydrogen_topology(
        self,
        present_atom_names: Collection[str],
    ) -> bool:
        """Return whether present hydrogens need coordinate-derived attachment."""

        normalized_present_hydrogen_atom_names = {
            atom_name.strip().upper()
            for atom_name in present_atom_names
            if atom_name.strip()
        }
        if not normalized_present_hydrogen_atom_names:
            return False

        template_resolved_hydrogen_atom_names = set(
            self.template_hydrogen_anchor_by_name(
                normalized_present_hydrogen_atom_names
            ).keys()
        )
        return any(
            hydrogen_atom_name not in template_resolved_hydrogen_atom_names
            for hydrogen_atom_name in normalized_present_hydrogen_atom_names
        )

    def has_tetrahedral_stereochemistry(self) -> bool:
        """Return whether side-chain tetrahedral expectations are defined."""

        return bool(self.tetrahedral_centers)

    def tetrahedral_center(
        self,
        *,
        center_atom_name: str,
        ordered_neighbor_atom_names: tuple[str, str, str],
    ) -> TetrahedralCenterSemantics | None:
        """Return the matching tetrahedral-center semantics when available."""

        for center_semantics in self.tetrahedral_centers:
            if center_semantics.matches(
                center_atom_name=center_atom_name,
                ordered_neighbor_atom_names=ordered_neighbor_atom_names,
            ):
                return center_semantics

        return None

    def correction_subtree_atom_names(
        self,
        center_semantics: TetrahedralCenterSemantics,
    ) -> tuple[str, ...]:
        """Return subtree atoms that should be rebuilt for one center correction."""

        center_atom_name = center_semantics.center_atom_name
        branch_root_atom_names = center_semantics.ordered_neighbor_atom_names[1:]
        atom_names_to_correct: set[str] = set()
        for branch_root_atom_name in branch_root_atom_names:
            queue: deque[str] = deque([branch_root_atom_name])
            visited = {center_atom_name}
            while queue:
                atom_name = queue.popleft()
                if atom_name in visited:
                    continue

                visited.add(atom_name)
                atom_names_to_correct.add(atom_name)
                for neighbor_atom_name in self.definition.bonded_atom_names(atom_name):
                    if neighbor_atom_name not in visited:
                        queue.append(neighbor_atom_name)

        return tuple(
            atom_name
            for atom_name in self.ordered_atom_names()
            if atom_name in atom_names_to_correct
        )

    def has_forcefield_params(self, atom_name: str) -> bool:
        """Return whether force-field parameters exist for a named atom."""

        return atom_name.strip().upper() in self.forcefield_parameters

    def bond_hop_distance(
        self,
        atom_name_1: str,
        atom_name_2: str,
        *,
        max_hops: int | None = None,
    ) -> int | None:
        """Return the heavy-atom bond hop distance between two named atoms."""

        return self.definition.bond_hop_distance(
            atom_name_1,
            atom_name_2,
            max_hops=max_hops,
        )

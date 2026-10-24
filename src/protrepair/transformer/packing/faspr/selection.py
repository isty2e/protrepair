"""Graph-qualified movement selection for fixed-backbone FASPR packing."""

from dataclasses import dataclass, replace

from typing_extensions import Self

from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.structure.labels import ResidueId
from protrepair.structure.topology import BondRelationshipType
from protrepair.transformer.packing.domain import PackingPlan

FIXED_BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})
_LINKED_RELATIONSHIPS = frozenset(
    {
        BondRelationshipType.COVALENT,
        BondRelationshipType.DISULFIDE,
        BondRelationshipType.METAL_COORDINATION,
        BondRelationshipType.UNKNOWN,
    }
)


@dataclass(frozen=True, slots=True)
class FasprPackingSelection:
    """Requested packing partitioned by existing linked-sidechain custody.

    Linked identities cover the whole polymer; protected and incomplete
    identities cover only selected residues not already frozen by the caller.
    """

    plan: PackingPlan
    linked_residue_ids: tuple[ResidueId, ...]
    protected_residue_ids: tuple[ResidueId, ...]
    incomplete_linked_residue_ids: tuple[ResidueId, ...]

    @classmethod
    def from_plan(cls, plan: PackingPlan) -> Self:
        """Resolve protection from canonical links and current heavy inventory."""

        polymer_ids = frozenset(plan.polymer_residue_ids)
        linked_ids: set[ResidueId] = set()
        for bond in plan.structure.topology.bonds:
            if bond.relationship_type not in _LINKED_RELATIONSHIPS:
                continue
            endpoints = bond.endpoint_pair()
            if any(
                plan.structure.constitution.atom_site_at(index).is_hydrogen()
                for index in endpoints
            ):
                continue
            refs = tuple(
                plan.structure.constitution.atom_ref_at(index) for index in endpoints
            )
            if refs[0].residue_id == refs[1].residue_id:
                continue
            for ref in refs:
                if (
                    ref.residue_id in polymer_ids
                    and ref.atom_name not in FIXED_BACKBONE_ATOM_NAMES
                ):
                    linked_ids.add(ref.residue_id)

        fixed_ids = frozenset(plan.fixed_residue_ids())
        library = build_standard_component_library()
        protected: list[ResidueId] = []
        incomplete: list[ResidueId] = []
        for residue_id in plan.selected_residue_ids():
            residue = plan.residue_site(residue_id)
            if residue_id not in linked_ids or residue_id in fixed_ids:
                continue
            template = library.get(residue.component_id)
            complete = template is not None and all(
                residue.has_atom_site(name)
                for name in template.expected_polymer_heavy_atom_names()
            )
            (protected if complete else incomplete).append(residue_id)

        return cls(
            plan,
            tuple(rid for rid in plan.polymer_residue_ids if rid in linked_ids),
            tuple(protected),
            tuple(incomplete),
        )

    def movable_residue_ids(self) -> tuple[ResidueId, ...]:
        """Return selected residues not frozen by the caller or linked custody."""

        fixed = frozenset(self.plan.fixed_residue_ids()) | frozenset(
            self.protected_residue_ids
        )
        return tuple(
            rid for rid in self.plan.selected_residue_ids() if rid not in fixed
        )

    def restricted_plan(self) -> PackingPlan | None:
        """Keep sequence correspondence while removing protected motion targets."""

        if not self.protected_residue_ids:
            return self.plan
        movable = self.movable_residue_ids()
        if not movable:
            return None

        sequence = self.plan.spec.target_sequence
        if sequence is not None:
            replacements = dict(
                zip(self.plan.selected_residue_ids(), sequence, strict=True)
            )
            sequence = "".join(replacements[rid] for rid in movable)
        spec = replace(
            self.plan.spec,
            mutable_residue_ids=movable,
            target_sequence=sequence,
        )
        return PackingPlan.from_inputs(self.plan.structure, spec)

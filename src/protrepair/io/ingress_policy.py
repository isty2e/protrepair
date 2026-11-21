"""Structure ingress normalization policy contracts."""

from dataclasses import dataclass, field
from enum import Enum

from protrepair.errors import StructureNormalizationError
from protrepair.structure.constitution import ResidueRole, ResidueSite
from protrepair.structure.labels import ResidueId


class OccupancyPolicy(str, Enum):
    """Policy for resolving alternate atom occupancies."""

    HIGHEST = "highest"
    LOWEST = "lowest"


class MutationPolicy(str, Enum):
    """Policy for resolving residue-level mutation conflicts."""

    HIGHEST_OCCUPANCY = "highest_occupancy"
    LOWEST_OCCUPANCY = "lowest_occupancy"


class LigandHandling(str, Enum):
    """How one normalization boundary should treat non-water ligands."""

    KEEP = "keep"
    DROP = "drop"
    REJECT = "reject"


class RetentionAction(str, Enum):
    """Final disposition of a selected non-polymer residue at ingress."""

    KEEP = "keep"
    DROP = "drop"
    REJECT = "reject"


@dataclass(frozen=True, slots=True)
class RetentionRule:
    """Override the default for residues matching every supplied selector.

    Parameters
    ----------
    action : RetentionAction
        Action taken unless a subsequent matching rule overrides it.
    role : ResidueRole or None
        WATER or LIGAND (including ions and cofactors).
    component_ids, chain_ids : tuple of str or None
        Alternative identifiers within each selector. Chain IDs are case-sensitive.
    residue_ids : tuple of ResidueId or None
        Exact addresses, including insertion codes.
    """

    action: RetentionAction
    role: ResidueRole | None = field(default=None, kw_only=True)
    component_ids: tuple[str, ...] | None = field(default=None, kw_only=True)
    chain_ids: tuple[str, ...] | None = field(default=None, kw_only=True)
    residue_ids: tuple[ResidueId, ...] | None = field(default=None, kw_only=True)

    def __post_init__(self) -> None:
        if not isinstance(self.action, RetentionAction):
            raise TypeError("action must be a RetentionAction")
        if self.role is not None and not isinstance(self.role, ResidueRole):
            raise TypeError("role must be a ResidueRole")
        if self.role is ResidueRole.POLYMER:
            raise ValueError("non-polymer retention cannot select a polymer role")
        for name, values in (
            ("component_ids", self.component_ids),
            ("chain_ids", self.chain_ids),
        ):
            if values is None:
                continue
            if isinstance(values, str):
                raise TypeError(f"{name} must be a collection, not a single string")
            values = tuple(values)
            if not all(isinstance(v, str) for v in values):
                raise TypeError(f"{name} must contain string identifiers")
            normalized = tuple(
                dict.fromkeys(
                    v.strip().upper() if name == "component_ids" else v.strip()
                    for v in values
                )
            )
            if not normalized or "" in normalized:
                raise ValueError(f"{name} must select at least one nonblank identifier")
            object.__setattr__(self, name, normalized)
        if self.residue_ids is not None:
            residue_ids = tuple(self.residue_ids)
            if not all(isinstance(rid, ResidueId) for rid in residue_ids):
                raise TypeError("residue_ids must contain ResidueId values")
            if not residue_ids:
                raise ValueError("residue_ids must not be empty")
            object.__setattr__(self, "residue_ids", tuple(dict.fromkeys(residue_ids)))
        if all(
            v is None
            for v in (self.role, self.component_ids, self.chain_ids, self.residue_ids)
        ):
            raise ValueError(
                "a retention rule needs a selector; use the role defaults otherwise"
            )

    def matches(self, role: ResidueRole, residue: ResidueSite) -> bool:
        """Match the normalized role and identity of a selected site."""
        return (
            role is not ResidueRole.POLYMER
            and (self.role is None or role is self.role)
            and (
                self.component_ids is None or residue.component_id in self.component_ids
            )
            and (
                self.chain_ids is None or residue.residue_id.chain_id in self.chain_ids
            )
            and (self.residue_ids is None or residue.residue_id in self.residue_ids)
        )


@dataclass(frozen=True, slots=True)
class NonPolymerRetentionPolicy:
    """Select whole non-polymer sites after alternate and variant selection.

    Parameters
    ----------
    water_default, other_default : RetentionAction
        Fallback for waters and all other non-polymers, respectively.
    rules : tuple of RetentionRule, default=()
        Evaluated in caller order; the last matching rule wins.
    """

    water_default: RetentionAction
    other_default: RetentionAction
    rules: tuple[RetentionRule, ...] = ()

    def __post_init__(self) -> None:
        if not all(
            isinstance(a, RetentionAction)
            for a in (self.water_default, self.other_default)
        ):
            raise TypeError("retention defaults must be RetentionAction values")
        rules = tuple(self.rules)
        if not all(isinstance(rule, RetentionRule) for rule in rules):
            raise TypeError("rules must contain RetentionRule values")
        object.__setattr__(self, "rules", rules)

    def keeps(self, role: ResidueRole, residue: ResidueSite) -> bool:
        """Decide retention, raising StructureNormalizationError on a final rejection."""
        if role is ResidueRole.POLYMER:
            return True
        action = next(
            (
                rule.action
                for rule in reversed(self.rules)
                if rule.matches(role, residue)
            ),
            self.water_default if role is ResidueRole.WATER else self.other_default,
        )
        if action is RetentionAction.REJECT:
            raise StructureNormalizationError(
                f"structure normalization rejected {role.value} {residue.component_id} at {residue.residue_id.display_token()}"
            )
        return action is RetentionAction.KEEP


@dataclass(frozen=True, slots=True)
class StructureNormalizationPolicy:
    """Canonical normalization policy for one structure-ingress boundary.

    Parameters
    ----------
    occupancy_policy : OccupancyPolicy, default=OccupancyPolicy.HIGHEST
        How to select alternate atom positions.
    mutation_policy : MutationPolicy, default=MutationPolicy.HIGHEST_OCCUPANCY
        How to select between residue variants at one position.
    ligand_handling : LigandHandling, default=LigandHandling.KEEP
        How to treat non-water ligands.
    selected_chain_ids : tuple of str or None, default=None
        Chains to retain; None selects all chains.
    non_polymer_retention : NonPolymerRetentionPolicy or None, default=None
        Ordered selection rules replacing legacy water and ligand handling.
    omit_zero_occupancy_sidechain_heavy_atoms : bool, default=False
        Treat eligible standard side-chain coordinates explicitly declared zero
        in the source text as missing, including zero-occupancy dependent H/D/T.
    omit_zero_occupancy_polymer_residues : bool, default=False
        Withhold isolated interior standard residues with exclusively declared
        zero source coordinates and observed backbone flanks.

    Raises
    ------
    TypeError
        A policy option has a noncanonical type.
    ValueError
        The chain selection contains a blank identifier.
    """

    occupancy_policy: OccupancyPolicy = OccupancyPolicy.HIGHEST
    mutation_policy: MutationPolicy = MutationPolicy.HIGHEST_OCCUPANCY
    ligand_handling: LigandHandling = LigandHandling.KEEP
    selected_chain_ids: tuple[str, ...] | None = None
    non_polymer_retention: NonPolymerRetentionPolicy | None = field(
        default=None, kw_only=True
    )
    omit_zero_occupancy_sidechain_heavy_atoms: bool = field(default=False, kw_only=True)
    omit_zero_occupancy_polymer_residues: bool = field(default=False, kw_only=True)

    def __post_init__(self) -> None:
        if self.non_polymer_retention is not None:
            if not isinstance(self.non_polymer_retention, NonPolymerRetentionPolicy):
                raise TypeError(
                    "non_polymer_retention must be a NonPolymerRetentionPolicy"
                )
            if self.ligand_handling is not LigandHandling.KEEP:
                raise ValueError("non_polymer_retention replaces ligand_handling")
        if not isinstance(self.occupancy_policy, OccupancyPolicy):
            raise TypeError(
                "structure normalization occupancy_policy must be an "
                "OccupancyPolicy value"
            )

        if not isinstance(self.mutation_policy, MutationPolicy):
            raise TypeError(
                "structure normalization mutation_policy must be a MutationPolicy value"
            )

        if not isinstance(self.ligand_handling, LigandHandling):
            raise TypeError(
                "structure normalization ligand_handling must be a LigandHandling value"
            )

        if not isinstance(self.omit_zero_occupancy_sidechain_heavy_atoms, bool):
            raise TypeError("omit_zero_occupancy_sidechain_heavy_atoms must be a bool")
        if not isinstance(self.omit_zero_occupancy_polymer_residues, bool):
            raise TypeError("omit_zero_occupancy_polymer_residues must be a bool")

        selected_chain_ids = self.selected_chain_ids
        if selected_chain_ids is None:
            return

        normalized_chain_ids: list[str] = []
        seen_chain_ids: set[str] = set()
        for chain_id in selected_chain_ids:
            normalized_chain_id = chain_id.strip()
            if not normalized_chain_id:
                raise ValueError(
                    "structure normalization selected_chain_ids must not contain blanks"
                )

            if normalized_chain_id in seen_chain_ids:
                continue

            normalized_chain_ids.append(normalized_chain_id)
            seen_chain_ids.add(normalized_chain_id)

        object.__setattr__(
            self,
            "selected_chain_ids",
            tuple(normalized_chain_ids) or None,
        )

    def selects_chain(self, chain_id: str) -> bool:
        """Return whether one chain should be normalized by this policy."""

        if self.selected_chain_ids is None:
            return True

        return chain_id.strip() in self.selected_chain_ids

    def keeps_ligands(self) -> bool:
        """Return whether the boundary preserves ligands."""

        return self.ligand_handling is LigandHandling.KEEP

    def drops_ligands(self) -> bool:
        """Return whether the boundary silently drops ligands."""

        return self.ligand_handling is LigandHandling.DROP

    def rejects_ligands(self) -> bool:
        """Return whether the boundary rejects ligand-bearing payloads."""

        return self.ligand_handling is LigandHandling.REJECT


__all__ = [
    "LigandHandling",
    "MutationPolicy",
    "NonPolymerRetentionPolicy",
    "OccupancyPolicy",
    "ResidueRole",
    "RetentionAction",
    "RetentionRule",
    "StructureNormalizationPolicy",
]

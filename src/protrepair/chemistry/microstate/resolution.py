"""Evidence intersection and explicit authority for local microstate graphs."""

from dataclasses import dataclass
from enum import Enum

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.microstate.graph import MicrostateGraph
from protrepair.structure.labels import AtomRef


class HydrogenAttachmentBasis(str, Enum):
    """Distinguish declared covalent attachment from interpreted atom naming."""

    SOURCE_COVALENT = "source_covalent"
    STANDARD_NAME = "standard_name"


@dataclass(frozen=True, slots=True)
class ObservedHydrogenAttachment:
    """Retain an interpreted source-H identity for subsequent atomic application.

    Parameters
    ----------
    hydrogen, parent : AtomRef
        Original atom identities, not indices in a repaired structure.
    basis : HydrogenAttachmentBasis
        Authority for the attachment interpretation, not protonation selection.

    Raises
    ------
    ValueError
        The endpoints coincide.
    TypeError
        The interpretation basis is not a canonical enum member.
    """

    hydrogen: AtomRef
    parent: AtomRef
    basis: HydrogenAttachmentBasis

    def __post_init__(self) -> None:
        if self.hydrogen == self.parent:
            raise ValueError("hydrogen attachment needs distinct endpoints")
        if not isinstance(self.basis, HydrogenAttachmentBasis):
            raise TypeError("hydrogen attachment requires a canonical basis")


@dataclass(frozen=True, slots=True)
class MicrostateConstraints:
    """Own exact declarations and lower bounds without conflating them.

    Parameters
    ----------
    charges : tuple[tuple[str, int], ...]
        Exact formal-charge constraints, including explicit zero.
    hydrogens : tuple[tuple[str, int], ...]
        Exact per-parent H counts, for explicit requests or complete evidence.
    minimum_hydrogens : tuple[tuple[str, int], ...]
        Observed attached H counts. Unobserved H is not negative evidence.
    bonds : tuple[BondDefinition, ...]
        Exact source/request integral bond-order constraints.

    Raises
    ------
    ValueError
        Subjects repeat, names are noncanonical, or H counts are negative.
    TypeError
        Values are not integers.
    """

    charges: tuple[tuple[str, int], ...] = ()
    hydrogens: tuple[tuple[str, int], ...] = ()
    minimum_hydrogens: tuple[tuple[str, int], ...] = ()
    bonds: tuple[BondDefinition, ...] = ()

    def __post_init__(self) -> None:
        for field_name, supplied in (
            ("charges", self.charges),
            ("hydrogens", self.hydrogens),
            ("minimum_hydrogens", self.minimum_hydrogens),
        ):
            values = tuple(sorted(supplied))
            if len({name for name, _ in values}) != len(values):
                raise ValueError("microstate constraint subjects must be unique")
            for name, value in values:
                if not name or name != name.strip().upper():
                    raise ValueError("microstate constraint names must be canonical")
                if type(value) is not int:
                    raise TypeError("microstate constraint values must be integers")
                if field_name != "charges" and value < 0:
                    raise ValueError("hydrogen constraints must be nonnegative")
            object.__setattr__(self, field_name, values)
        canonical_bonds: dict[tuple[str, str], BondDefinition] = {}
        for bond in self.bonds:
            first, second = sorted((bond.atom_name_1, bond.atom_name_2))
            pair = (first, second)
            if pair in canonical_bonds or first == second or bond.aromatic:
                raise ValueError("microstate constraints require unique integral bonds")
            if type(bond.order) is not int:
                raise TypeError("microstate bond-order constraints must be integers")
            canonical_bonds[pair] = BondDefinition(first, second, order=bond.order)
        object.__setattr__(
            self,
            "bonds",
            tuple(canonical_bonds[pair] for pair in sorted(canonical_bonds)),
        )

    def is_empty(self) -> bool:
        """Return whether this value makes no chemical declaration.

        Returns
        -------
        bool
            True when all four constraint collections are empty.
        """
        return not (
            self.charges or self.hydrogens or self.minimum_hydrogens or self.bonds
        )

    def violations(self, graph: MicrostateGraph) -> "MicrostateConstraints":
        """Retain the exact evidence incompatible with one candidate.

        Parameters
        ----------
        graph : MicrostateGraph
            Candidate for the same chemical site.

        Returns
        -------
        MicrostateConstraints
            Unsatisfied constraints; outside-site subjects also remain violations.
        """
        names = {atom.name for atom in graph.atoms}
        return MicrostateConstraints(
            charges=tuple(
                (name, value)
                for name, value in self.charges
                if name not in names or graph.atom(name).charge != value
            ),
            hydrogens=tuple(
                (name, value)
                for name, value in self.hydrogens
                if name not in names or graph.atom(name).hydrogens != value
            ),
            minimum_hydrogens=tuple(
                (name, value)
                for name, value in self.minimum_hydrogens
                if name not in names or graph.atom(name).hydrogens < value
            ),
            bonds=tuple(
                bond
                for bond in self.bonds
                if graph.bond_order(bond.atom_name_1, bond.atom_name_2) != bond.order
            ),
        )


class MicrostateResolutionStatus(str, Enum):
    """Chemical resolution outcomes, not transformer execution statuses."""

    RESOLVED = "resolved"
    AMBIGUOUS = "ambiguous"
    CONFLICT = "conflict"
    INSUFFICIENT = "insufficient"
    UNSUPPORTED = "unsupported"


class MicrostateSelectionBasis(str, Enum):
    """Authority selecting a resolved chemical state."""

    EVIDENCE = "evidence"
    OVERRIDE = "override"
    DEFAULT = "default"


@dataclass(frozen=True, slots=True)
class MicrostateResolution:
    """Expose a graph only when one protonation state is resolved.

    Parameters
    ----------
    status : MicrostateResolutionStatus
        Resolved, ambiguous, conflicting, insufficient or unsupported outcome.
    candidates : tuple[MicrostateGraph, ...]
        Surviving representations in deterministic catalog order.
    basis : MicrostateSelectionBasis
        Governing authority; a default cannot override source evidence.
    superseded_source : MicrostateConstraints
        Source declarations replaced by an explicit site override.
    details : tuple[str, ...]
        Reasons for unavailable evidence or unsupported chemistry.
    observed_hydrogens : tuple[ObservedHydrogenAttachment, ...]
        Interpreted original H identities and parents for this site. They remain
        source evidence even when an explicit override requests a different count.

    Raises
    ------
    ValueError
        Status and candidate multiplicity disagree, source is silently replaced,
        or observed H mappings repeat or refer to another site.
    TypeError
        Status or selection basis is not a canonical enum member.
    """

    status: MicrostateResolutionStatus
    candidates: tuple[MicrostateGraph, ...] = ()
    basis: MicrostateSelectionBasis = MicrostateSelectionBasis.EVIDENCE
    superseded_source: MicrostateConstraints = MicrostateConstraints()
    details: tuple[str, ...] = ()
    observed_hydrogens: tuple[ObservedHydrogenAttachment, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.status, MicrostateResolutionStatus) or not isinstance(
            self.basis, MicrostateSelectionBasis
        ):
            raise TypeError("microstate resolution requires canonical enum values")
        if self.candidates:
            object.__setattr__(
                self, "candidates", MicrostateSite(self.candidates).candidates
            )
        keys = {graph.protonation_key() for graph in self.candidates}
        if (
            (self.status is MicrostateResolutionStatus.RESOLVED and len(keys) != 1)
            or (self.status is MicrostateResolutionStatus.AMBIGUOUS and len(keys) < 2)
            or (
                self.status
                not in {
                    MicrostateResolutionStatus.RESOLVED,
                    MicrostateResolutionStatus.AMBIGUOUS,
                }
                and self.candidates
            )
        ):
            raise ValueError("microstate resolution status disagrees with candidates")
        if not self.superseded_source.is_empty() and (
            self.basis is not MicrostateSelectionBasis.OVERRIDE
            or self.status is not MicrostateResolutionStatus.RESOLVED
        ):
            raise ValueError(
                "only resolved explicit overrides can replace source evidence"
            )
        object.__setattr__(self, "candidates", tuple(self.candidates))
        object.__setattr__(self, "details", tuple(self.details))
        if len({entry.hydrogen for entry in self.observed_hydrogens}) != len(
            self.observed_hydrogens
        ):
            raise ValueError("source hydrogens must have one interpreted parent")
        if self.candidates and (
            len({entry.parent.residue_id for entry in self.observed_hydrogens}) > 1
            or any(
                entry.parent.atom_name
                not in {atom.name for atom in self.candidates[0].atoms}
                for entry in self.observed_hydrogens
            )
        ):
            raise ValueError("source hydrogen parents must belong to this site")
        object.__setattr__(self, "observed_hydrogens", tuple(self.observed_hydrogens))

    @property
    def graph(self) -> MicrostateGraph | None:
        """Return the canonical surviving representation, or no resolved graph.

        Returns
        -------
        MicrostateGraph or None
            None for every unresolved outcome, including multiple protonation states.
        """
        return (
            self.candidates[0]
            if self.status is MicrostateResolutionStatus.RESOLVED
            else None
        )


@dataclass(frozen=True, slots=True)
class MicrostateSite:
    """Resolve alternative local states of one fixed chemical skeleton.

    Parameters
    ----------
    candidates : tuple[MicrostateGraph, ...]
        Valence-valid representations sharing atoms, edges and boundary valence.

    Raises
    ------
    ValueError
        The catalog is empty or mixes distinct chemical skeletons.
    """

    candidates: tuple[MicrostateGraph, ...]

    def __post_init__(self) -> None:
        if not self.candidates:
            raise ValueError("a microstate site needs candidate graphs")
        first = self.candidates[0]
        skeleton = tuple(
            (atom.name, atom.element, atom.boundary_order) for atom in first.atoms
        )
        edges = tuple((bond.atom_name_1, bond.atom_name_2) for bond in first.bonds)
        if any(
            tuple(
                (atom.name, atom.element, atom.boundary_order) for atom in graph.atoms
            )
            != skeleton
            or tuple((bond.atom_name_1, bond.atom_name_2) for bond in graph.bonds)
            != edges
            for graph in self.candidates
        ):
            raise ValueError("microstate site candidates must share one skeleton")
        object.__setattr__(self, "candidates", tuple(dict.fromkeys(self.candidates)))

    def resolve(
        self,
        source: MicrostateConstraints,
        *,
        override: MicrostateConstraints | None = None,
        preferences: tuple[MicrostateConstraints, ...] = (),
    ) -> MicrostateResolution:
        """Intersect evidence before choosing a resonance representation.

        Parameters
        ----------
        source : MicrostateConstraints
            Original declarations; unobserved hydrogen counts are not exact zero.
        override : MicrostateConstraints or None
            Explicit replacement authority over this whole site, not the residue.
        preferences : tuple[MicrostateConstraints, ...]
            Highest-priority first. Each preference narrows the remaining
            source-compatible candidates only if the intersection is nonempty.
            Preferences never restore eliminated candidates or override evidence;
            an explicit override bypasses them.

        Returns
        -------
        MicrostateResolution
            One protonation class, ambiguity, or no supported matching graph.

        Raises
        ------
        ValueError
            An empty override would grant unqualified source-replacement authority.
        """
        source_matches = tuple(
            graph for graph in self.candidates if source.violations(graph).is_empty()
        )
        basis = MicrostateSelectionBasis.EVIDENCE
        candidates = source_matches
        if override is not None:
            if override.is_empty():
                raise ValueError(
                    "microstate override must specify chemical constraints"
                )
            requested = tuple(
                graph
                for graph in self.candidates
                if override.violations(graph).is_empty()
            )
            candidates = (
                tuple(graph for graph in requested if graph in source_matches)
                or requested
            )
            basis = MicrostateSelectionBasis.OVERRIDE
        else:
            for preference in preferences:
                preferred = tuple(
                    graph
                    for graph in candidates
                    if preference.violations(graph).is_empty()
                )
                if preferred and preferred != candidates:
                    candidates = preferred
                    basis = MicrostateSelectionBasis.DEFAULT

        if not candidates:
            return MicrostateResolution(
                MicrostateResolutionStatus.CONFLICT, basis=basis
            )
        if len({graph.protonation_key() for graph in candidates}) > 1:
            return MicrostateResolution(
                MicrostateResolutionStatus.AMBIGUOUS, candidates, basis
            )
        return MicrostateResolution(
            MicrostateResolutionStatus.RESOLVED,
            candidates,
            basis,
            source.violations(candidates[0]),
        )

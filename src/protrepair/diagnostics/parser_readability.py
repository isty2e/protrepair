"""RDKit parser-readability diagnostics over canonical structures."""

from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from types import MappingProxyType
from typing import Protocol

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.io.pdb_projection import (
    RDKitNoConectPDBBlockProjector,
    pdb_without_conect,
    pdb_without_conect_for_parser_probe,
)
from protrepair.io.source_identity import (
    normalize_chain_id,
    normalize_insertion_code,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import (
    BondProvenance,
    TopologyBond,
    is_covalent_like_relationship,
)

try:
    from rdkit import Chem, rdBase
except ImportError:  # pragma: no cover - exercised by optional dependency checks
    Chem = None
    rdBase = None


class _RDKitResidueInfo(Protocol):
    """PDB residue metadata surface consumed from RDKit atoms."""

    def GetChainId(self) -> str:
        """Return the PDB chain identifier."""
        ...

    def GetName(self) -> str:
        """Return the PDB atom name."""
        ...

    def GetResidueNumber(self) -> int:
        """Return the PDB residue sequence number."""
        ...

    def GetInsertionCode(self) -> str:
        """Return the PDB insertion code."""
        ...


class _RDKitBond(Protocol):
    """RDKit bond surface needed for parser witnesses."""

    def GetBeginAtom(self) -> "_RDKitAtom":
        """Return the first atom in this bond."""
        ...

    def GetEndAtom(self) -> "_RDKitAtom":
        """Return the second atom in this bond."""
        ...


class _RDKitAtom(Protocol):
    """RDKit atom surface needed for parser witnesses."""

    def GetPDBResidueInfo(self) -> _RDKitResidueInfo | None:
        """Return PDB residue metadata if present."""
        ...

    def GetBonds(self) -> Sequence[_RDKitBond]:
        """Return bonds adjacent to this atom."""
        ...

    def GetSymbol(self) -> str:
        """Return the atomic element symbol."""
        ...


class _RDKitMol(Protocol):
    """RDKit molecule surface needed for parser witnesses."""

    def GetAtomWithIdx(self, atom_index: int) -> _RDKitAtom:
        """Return an atom by RDKit atom index."""
        ...


@contextmanager
def _expected_rdkit_parser_probe_logs_blocked() -> Iterator[None]:
    """Suppress expected RDKit parser diagnostics during no-CONECT probes."""

    if rdBase is None:
        yield
        return

    with rdBase.BlockLogs():
        yield


@dataclass(frozen=True, slots=True)
class RDKitProximityBondWitness:
    """One RDKit-inferred no-CONECT proximity bond near a parser failure."""

    atom_ref_1: AtomRef
    atom_ref_2: AtomRef
    element_1: str
    element_2: str
    is_known_component_bond: bool

    def __post_init__(self) -> None:
        object.__setattr__(self, "element_1", self.element_1.strip().upper())
        object.__setattr__(self, "element_2", self.element_2.strip().upper())

    def display_token(self) -> str:
        """Return a compact human-readable bond token."""

        return f"{self.atom_ref_1.display_token()}-{self.atom_ref_2.display_token()}"

    def is_heavy_heavy(self) -> bool:
        """Return whether both witness atoms are non-hydrogen atoms."""

        return self.element_1 != "H" and self.element_2 != "H"

    def residue_ids(self) -> tuple[ResidueId, ...]:
        """Return residue ids touched by this witness bond."""

        if self.atom_ref_1.residue_id == self.atom_ref_2.residue_id:
            return (self.atom_ref_1.residue_id,)

        return tuple(
            sorted(
                (self.atom_ref_1.residue_id, self.atom_ref_2.residue_id),
                key=lambda residue_id: residue_id.display_token(),
            )
        )


@dataclass(frozen=True, slots=True)
class RDKitParserProblemWitness:
    """RDKit parser failure enriched with local proximity-bond witnesses."""

    residue_id: ResidueId
    atom_names: tuple[str, ...]
    messages: tuple[str, ...]
    proximity_bonds: tuple[RDKitProximityBondWitness, ...]

    def extra_proximity_bonds(self) -> tuple[RDKitProximityBondWitness, ...]:
        """Return inferred bonds unknown to ProtRepair topology."""

        return tuple(
            witness
            for witness in self.proximity_bonds
            if not witness.is_known_component_bond
        )

    def extra_heavy_proximity_bonds(self) -> tuple[RDKitProximityBondWitness, ...]:
        """Return extra inferred heavy-heavy proximity bonds."""

        return tuple(
            witness
            for witness in self.extra_proximity_bonds()
            if witness.is_heavy_heavy()
        )


@dataclass(frozen=True, slots=True)
class RDKitProximityBondCluster:
    """Connected component of RDKit-inferred extra heavy-heavy proximity bonds."""

    residue_ids: tuple[ResidueId, ...]
    bonds: tuple[RDKitProximityBondWitness, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "residue_ids",
            tuple(
                sorted(
                    self.residue_ids,
                    key=lambda residue_id: residue_id.display_token(),
                )
            ),
        )
        object.__setattr__(
            self,
            "bonds",
            tuple(
                sorted(
                    self.bonds,
                    key=lambda bond: bond.display_token(),
                )
            ),
        )

    def display_token(self) -> str:
        """Return a compact residue-cluster token."""

        return ",".join(residue_id.display_token() for residue_id in self.residue_ids)


@dataclass(frozen=True, slots=True)
class RDKitNoConectSanitizeReadabilityMetrics:
    """Compact no-CONECT RDKit parser-readability burden metrics."""

    sanitize_readable: bool | None
    extra_proximity_bond_count: int = 0
    extra_heavy_proximity_bond_count: int = 0

    def __post_init__(self) -> None:
        if self.extra_proximity_bond_count < 0:
            raise ValueError("extra proximity-bond count must be non-negative")
        if self.extra_heavy_proximity_bond_count < 0:
            raise ValueError("extra heavy proximity-bond count must be non-negative")


@dataclass(frozen=True, slots=True)
class RDKitNoConectParserReadabilityProbe:
    """Single no-CONECT RDKit parser probe with reusable projections."""

    sanitize_readable: bool | None
    residue_problem_witnesses: tuple[RDKitParserProblemWitness, ...]
    structure_messages: tuple[str, ...]

    def metrics(self) -> RDKitNoConectSanitizeReadabilityMetrics:
        """Return compact parser-readability burden metrics for this probe."""

        if not self.residue_problem_witnesses and not self.structure_messages:
            return RDKitNoConectSanitizeReadabilityMetrics(
                sanitize_readable=self.sanitize_readable,
            )

        extra_bonds = _extra_proximity_bonds_from_problem_witnesses(
            self.residue_problem_witnesses
        )
        return RDKitNoConectSanitizeReadabilityMetrics(
            sanitize_readable=False,
            extra_proximity_bond_count=len(extra_bonds),
            extra_heavy_proximity_bond_count=sum(
                1 for bond in extra_bonds if bond.is_heavy_heavy()
            ),
        )

    def issues(self) -> tuple[ValidationIssue, ...]:
        """Return typed parser-readability issues for this probe."""

        if self.structure_messages == ("_pdb_parse_failed_before_sanitize",):
            return (
                ValidationIssue(
                    kind=ValidationIssueKind.PARSER_READABILITY,
                    severity=IssueSeverity.ERROR,
                    message=(
                        "RDKit could not parse the no-CONECT hydrogenated PDB surface "
                        "before sanitize"
                    ),
                ),
            )

        if not self.residue_problem_witnesses and not self.structure_messages:
            return ()

        issues: list[ValidationIssue] = []
        for problem_witness in sorted(
            self.residue_problem_witnesses,
            key=lambda item: item.residue_id.display_token(),
        ):
            atom_names = problem_witness.atom_names
            messages = problem_witness.messages
            message = (
                "no-CONECT RDKit sanitize would fail around atoms "
                f"{', '.join(atom_names)}: {messages[0]}"
            )
            if len(messages) > 1:
                message += (
                    f" (+{len(messages) - 1} additional parser chemistry problems)"
                )
            extra_bond_tokens = tuple(
                witness.display_token()
                for witness in problem_witness.extra_proximity_bonds()
            )
            if extra_bond_tokens:
                message += (
                    "; RDKit proximity witness extra bond(s): "
                    f"{', '.join(extra_bond_tokens[:3])}"
                )
                if len(extra_bond_tokens) > 3:
                    message += f" (+{len(extra_bond_tokens) - 3} more)"

            issues.append(
                ValidationIssue.for_residue(
                    kind=ValidationIssueKind.PARSER_READABILITY,
                    severity=IssueSeverity.ERROR,
                    message=message,
                    residue_id=problem_witness.residue_id,
                )
            )

        if self.structure_messages:
            summary = self.structure_messages
            message = (
                "no-CONECT RDKit sanitize would fail with non-localized parser "
                f"chemistry problems: {summary[0]}"
            )
            if len(summary) > 1:
                message += (
                    f" (+{len(summary) - 1} additional parser chemistry problems)"
                )

            issues.append(
                ValidationIssue(
                    kind=ValidationIssueKind.PARSER_READABILITY,
                    severity=IssueSeverity.ERROR,
                    message=message,
                )
            )

        return tuple(issues)

    def parser_problem_witnesses(self) -> tuple[RDKitParserProblemWitness, ...]:
        """Return residue-local parser problem witnesses in stable order."""

        return tuple(
            sorted(
                self.residue_problem_witnesses,
                key=lambda item: item.residue_id.display_token(),
            )
        )

    def extra_proximity_bond_clusters(self) -> tuple[RDKitProximityBondCluster, ...]:
        """Return connected clusters of extra heavy-heavy parser witnesses."""

        extra_heavy_bonds = tuple(
            bond
            for bond in _extra_proximity_bonds_from_problem_witnesses(
                self.residue_problem_witnesses
            )
            if bond.is_heavy_heavy()
        )
        return _extra_proximity_bond_clusters_from_witnesses(extra_heavy_bonds)


@dataclass(frozen=True, slots=True)
class _RDKitNoConectProblemSurface:
    """Private parsed RDKit problem surface before ProtRepair-specific projections."""

    molecule: _RDKitMol | None
    residue_atom_names: Mapping[ResidueId, frozenset[str]]
    residue_messages: Mapping[ResidueId, frozenset[str]]
    residue_problem_atom_indices: Mapping[ResidueId, frozenset[int]]
    structure_messages: tuple[str, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "residue_atom_names",
            MappingProxyType(dict(self.residue_atom_names)),
        )
        object.__setattr__(
            self,
            "residue_messages",
            MappingProxyType(dict(self.residue_messages)),
        )
        object.__setattr__(
            self,
            "residue_problem_atom_indices",
            MappingProxyType(dict(self.residue_problem_atom_indices)),
        )


@dataclass(frozen=True, slots=True)
class RDKitKnownBondLookup:
    """Coordinate-invariant topology lookup for no-CONECT parser projections."""

    component_library: ComponentLibrary
    component_id_by_residue_id: Mapping[ResidueId, str]
    known_atom_name_pairs_by_residue_id: Mapping[
        ResidueId,
        frozenset[tuple[str, str]],
    ]
    topology_atom_name_pairs_by_residue_id: Mapping[
        ResidueId,
        frozenset[tuple[str, str]],
    ]
    topology_inter_residue_endpoint_pairs: frozenset[tuple[AtomRef, AtomRef]]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "component_id_by_residue_id",
            MappingProxyType(dict(self.component_id_by_residue_id)),
        )
        object.__setattr__(
            self,
            "known_atom_name_pairs_by_residue_id",
            MappingProxyType(dict(self.known_atom_name_pairs_by_residue_id)),
        )
        object.__setattr__(
            self,
            "topology_atom_name_pairs_by_residue_id",
            MappingProxyType(dict(self.topology_atom_name_pairs_by_residue_id)),
        )
        object.__setattr__(
            self,
            "topology_inter_residue_endpoint_pairs",
            frozenset(self.topology_inter_residue_endpoint_pairs),
        )

    def is_known_bond(self, atom_ref_1: AtomRef, atom_ref_2: AtomRef) -> bool:
        """Return whether one RDKit-inferred bond exists in ProtRepair topology."""

        if atom_ref_1.residue_id != atom_ref_2.residue_id:
            endpoint_pair = _atom_ref_endpoint_pair(atom_ref_1, atom_ref_2)
            if endpoint_pair in self.topology_inter_residue_endpoint_pairs:
                return True

            return _is_known_polymer_backbone_link(
                atom_ref_1,
                atom_ref_2,
                component_id_by_residue_id=self.component_id_by_residue_id,
                component_library=self.component_library,
            )

        atom_name_pair = _sorted_atom_name_pair(
            atom_ref_1.atom_name,
            atom_ref_2.atom_name,
        )
        if atom_name_pair in self.topology_atom_name_pairs_by_residue_id.get(
            atom_ref_1.residue_id,
            frozenset(),
        ):
            return True

        return atom_name_pair in self.known_atom_name_pairs_by_residue_id.get(
            atom_ref_1.residue_id,
            frozenset(),
        )


def probe_rdkit_no_conect_parser_readability(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary | None = None,
    pdb_block_projector: RDKitNoConectPDBBlockProjector | None = None,
) -> RDKitNoConectParserReadabilityProbe:
    """Return one reusable no-CONECT RDKit parser-readability probe."""

    if Chem is None or not _structure_contains_hydrogens(structure):
        return RDKitNoConectParserReadabilityProbe(
            sanitize_readable=None,
            residue_problem_witnesses=(),
            structure_messages=(),
        )

    residue_problem_witnesses, structure_messages = _rdkit_no_conect_problem_witnesses(
        structure,
        component_library=component_library,
        pdb_block_projector=pdb_block_projector,
    )
    if not residue_problem_witnesses and not structure_messages:
        return RDKitNoConectParserReadabilityProbe(
            sanitize_readable=True,
            residue_problem_witnesses=(),
            structure_messages=(),
        )

    return RDKitNoConectParserReadabilityProbe(
        sanitize_readable=False,
        residue_problem_witnesses=tuple(residue_problem_witnesses.values()),
        structure_messages=structure_messages,
    )


def measure_rdkit_no_conect_extra_heavy_proximity_bond_count(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary | None = None,
    known_bond_lookup: RDKitKnownBondLookup | None = None,
    pdb_block_projector: RDKitNoConectPDBBlockProjector | None = None,
) -> int:
    """Return the extra heavy-heavy parser witness count without clustering."""

    if Chem is None or not _structure_contains_hydrogens(structure):
        return 0

    problem_surface = _rdkit_no_conect_problem_surface(
        structure,
        pdb_block_projector=pdb_block_projector,
    )
    if (
        problem_surface.molecule is None
        or not problem_surface.residue_problem_atom_indices
    ):
        return 0

    active_known_bond_lookup = known_bond_lookup
    if active_known_bond_lookup is None:
        active_known_bond_lookup = prepare_rdkit_no_conect_known_bond_lookup(
            structure,
            component_library=component_library,
        )

    atom_indices = {
        atom_index
        for atom_indices_for_residue in (
            problem_surface.residue_problem_atom_indices.values()
        )
        for atom_index in atom_indices_for_residue
    }
    extra_heavy_pairs: set[tuple[AtomRef, AtomRef]] = set()
    for atom_index in sorted(atom_indices):
        atom = problem_surface.molecule.GetAtomWithIdx(atom_index)
        for bond in atom.GetBonds():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            if begin_atom.GetSymbol() == "H" or end_atom.GetSymbol() == "H":
                continue

            begin_ref = _atom_ref_from_rdkit_atom(begin_atom)
            end_ref = _atom_ref_from_rdkit_atom(end_atom)
            if begin_ref is None or end_ref is None:
                continue

            left_ref, right_ref = sorted((begin_ref, end_ref))
            pair = (left_ref, right_ref)
            if pair in extra_heavy_pairs:
                continue

            if active_known_bond_lookup.is_known_bond(begin_ref, end_ref):
                continue

            extra_heavy_pairs.add(pair)

    return len(extra_heavy_pairs)


def measure_rdkit_no_conect_sanitize_readability(
    structure: ProteinStructure,
) -> bool | None:
    """Return whether hydrogenated structure geometry survives no-CONECT RDKit sanitize.

    The result stays in the acceptance layer. It must not be promoted into
    canonical state facts because it reflects one parser-visible geometry check,
    not an authoritative structure truth.
    """

    if Chem is None or not _structure_contains_hydrogens(structure):
        return None

    pdb_block = pdb_without_conect(structure)
    with _expected_rdkit_parser_probe_logs_blocked():
        try:
            molecule = Chem.MolFromPDBBlock(
                pdb_block,
                sanitize=True,
                removeHs=False,
                proximityBonding=True,
            )
        except Exception:  # pragma: no cover - backend-specific failure path
            return False

    return molecule is not None


def measure_rdkit_no_conect_sanitize_readability_metrics(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary | None = None,
    pdb_block_projector: RDKitNoConectPDBBlockProjector | None = None,
) -> RDKitNoConectSanitizeReadabilityMetrics:
    """Return no-CONECT RDKit readability plus parser-visible bond burden."""

    return probe_rdkit_no_conect_parser_readability(
        structure,
        component_library=component_library,
        pdb_block_projector=pdb_block_projector,
    ).metrics()


def diagnose_rdkit_no_conect_sanitize_readability(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary | None = None,
    pdb_block_projector: RDKitNoConectPDBBlockProjector | None = None,
) -> tuple[ValidationIssue, ...]:
    """Return typed parser-readability issues over one hydrogenated structure."""

    return probe_rdkit_no_conect_parser_readability(
        structure,
        component_library=component_library,
        pdb_block_projector=pdb_block_projector,
    ).issues()


def rdkit_no_conect_parser_failing_residue_ids(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary | None = None,
    pdb_block_projector: RDKitNoConectPDBBlockProjector | None = None,
) -> tuple[ResidueId, ...]:
    """Return residues that fail no-CONECT RDKit sanitize under proximity bonding."""

    return tuple(
        problem_witness.residue_id
        for problem_witness in probe_rdkit_no_conect_parser_readability(
            structure,
            component_library=component_library,
            pdb_block_projector=pdb_block_projector,
        ).parser_problem_witnesses()
    )


def rdkit_no_conect_parser_problem_witnesses(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary | None = None,
    pdb_block_projector: RDKitNoConectPDBBlockProjector | None = None,
) -> tuple[RDKitParserProblemWitness, ...]:
    """Return residue-local RDKit parser problem witnesses."""

    return probe_rdkit_no_conect_parser_readability(
        structure,
        component_library=component_library,
        pdb_block_projector=pdb_block_projector,
    ).parser_problem_witnesses()


def rdkit_no_conect_extra_proximity_bond_clusters(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary | None = None,
    pdb_block_projector: RDKitNoConectPDBBlockProjector | None = None,
) -> tuple[RDKitProximityBondCluster, ...]:
    """Return clusters of extra heavy-heavy RDKit proximity bonds."""

    return probe_rdkit_no_conect_parser_readability(
        structure,
        component_library=component_library,
        pdb_block_projector=pdb_block_projector,
    ).extra_proximity_bond_clusters()


def prepare_rdkit_no_conect_known_bond_lookup(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary | None = None,
) -> RDKitKnownBondLookup:
    """Return topology facts reused across coordinate-only parser probes."""

    active_component_library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    return RDKitKnownBondLookup(
        component_library=active_component_library,
        component_id_by_residue_id={
            residue_site.residue_id: residue_site.component_id
            for residue_site in structure.constitution.iter_residues(
                include_ligands=True,
            )
        },
        known_atom_name_pairs_by_residue_id={
            residue_site.residue_id: _known_atom_name_pairs_for_residue(
                residue_site.component_id,
                residue_site.atom_site_names(),
                component_library=active_component_library,
            )
            for residue_site in structure.constitution.iter_residues(
                include_ligands=True,
            )
        },
        topology_atom_name_pairs_by_residue_id=(
            _topology_known_atom_name_pairs_by_residue_id(structure)
        ),
        topology_inter_residue_endpoint_pairs=(
            _topology_known_inter_residue_endpoint_pairs(structure)
        ),
    )


def _structure_contains_hydrogens(structure: ProteinStructure) -> bool:
    """Return whether one canonical structure contains any hydrogen atoms."""

    return any(
        atom_site.element == "H"
        for residue_site in structure.constitution.residue_slots
        for atom_site in residue_site.atom_sites
    )


def _atom_ref_from_pdb_atom_line(line: str) -> AtomRef | None:
    """Return the atom reference encoded in one PDB ATOM/HETATM line."""

    if len(line) < 54:
        return None

    try:
        return AtomRef(
            residue_id=ResidueId(
                chain_id=normalize_chain_id(line[21:22]),
                seq_num=int(line[22:26]),
                insertion_code=normalize_insertion_code(line[26:27]),
            ),
            atom_name=line[12:16].strip(),
        )
    except ValueError:
        return None


def _rdkit_no_conect_problem_witnesses(
    structure: ProteinStructure,
    *,
    component_library: ComponentLibrary | None,
    pdb_block_projector: RDKitNoConectPDBBlockProjector | None = None,
) -> tuple[
    dict[ResidueId, RDKitParserProblemWitness],
    tuple[str, ...],
]:
    """Return residue-localized parser chemistry problems and bond witnesses."""

    problem_surface = _rdkit_no_conect_problem_surface(
        structure,
        pdb_block_projector=pdb_block_projector,
    )
    if problem_surface.molecule is None:
        return ({}, problem_surface.structure_messages)
    if not problem_surface.residue_problem_atom_indices:
        return ({}, problem_surface.structure_messages)

    known_bond_lookup = prepare_rdkit_no_conect_known_bond_lookup(
        structure,
        component_library=component_library,
    )
    residue_problem_witnesses = {
        residue_id: RDKitParserProblemWitness(
            residue_id=residue_id,
            atom_names=tuple(sorted(problem_surface.residue_atom_names[residue_id])),
            messages=tuple(sorted(problem_surface.residue_messages[residue_id])),
            proximity_bonds=_problem_proximity_bond_witnesses(
                problem_surface.molecule,
                residue_id=residue_id,
                atom_indices=set(
                    problem_surface.residue_problem_atom_indices[residue_id]
                ),
                known_bond_lookup=known_bond_lookup,
            ),
        )
        for residue_id in problem_surface.residue_atom_names
    }
    return (
        residue_problem_witnesses,
        problem_surface.structure_messages,
    )


def _rdkit_no_conect_problem_surface(
    structure: ProteinStructure,
    *,
    pdb_block_projector: RDKitNoConectPDBBlockProjector | None = None,
) -> _RDKitNoConectProblemSurface:
    """Return parsed RDKit problem data before witness projection."""

    if Chem is None or not _structure_contains_hydrogens(structure):
        return _RDKitNoConectProblemSurface(
            molecule=None,
            residue_atom_names={},
            residue_messages={},
            residue_problem_atom_indices={},
            structure_messages=(),
        )

    pdb_block = pdb_without_conect_for_parser_probe(
        structure,
        pdb_block_projector,
    )
    with _expected_rdkit_parser_probe_logs_blocked():
        molecule = Chem.MolFromPDBBlock(
            pdb_block,
            sanitize=False,
            removeHs=False,
            proximityBonding=True,
        )
        if molecule is None:
            return _RDKitNoConectProblemSurface(
                molecule=None,
                residue_atom_names={},
                residue_messages={},
                residue_problem_atom_indices={},
                structure_messages=("_pdb_parse_failed_before_sanitize",),
            )

        chemistry_problems = Chem.DetectChemistryProblems(molecule)
    if not chemistry_problems:
        return _RDKitNoConectProblemSurface(
            molecule=molecule,
            residue_atom_names={},
            residue_messages={},
            residue_problem_atom_indices={},
            structure_messages=(),
        )

    residue_atom_names: dict[ResidueId, set[str]] = defaultdict(set)
    residue_messages: dict[ResidueId, set[str]] = defaultdict(set)
    residue_problem_atom_indices: dict[ResidueId, set[int]] = defaultdict(set)
    structure_messages: set[str] = set()

    for problem in chemistry_problems:
        message = problem.Message().splitlines()[0].strip()
        atom_index = problem.GetAtomIdx() if hasattr(problem, "GetAtomIdx") else None
        if atom_index is None:
            structure_messages.add(message)
            continue

        atom = molecule.GetAtomWithIdx(atom_index)
        residue_info = atom.GetPDBResidueInfo()
        if residue_info is None:
            structure_messages.add(message)
            continue

        residue_id = ResidueId(
            chain_id=residue_info.GetChainId().strip(),
            seq_num=residue_info.GetResidueNumber(),
            insertion_code=(
                None
                if not residue_info.GetInsertionCode().strip()
                else residue_info.GetInsertionCode().strip()
            ),
        )
        residue_atom_names[residue_id].add(residue_info.GetName().strip())
        residue_messages[residue_id].add(message)
        residue_problem_atom_indices[residue_id].add(atom_index)

    return _RDKitNoConectProblemSurface(
        molecule=molecule,
        residue_atom_names={
            residue_id: frozenset(atom_names)
            for residue_id, atom_names in residue_atom_names.items()
        },
        residue_messages={
            residue_id: frozenset(messages)
            for residue_id, messages in residue_messages.items()
        },
        residue_problem_atom_indices={
            residue_id: frozenset(atom_indices)
            for residue_id, atom_indices in residue_problem_atom_indices.items()
        },
        structure_messages=tuple(sorted(structure_messages)),
    )


def _problem_proximity_bond_witnesses(
    molecule: _RDKitMol,
    *,
    residue_id: ResidueId | None,
    atom_indices: set[int],
    known_bond_lookup: RDKitKnownBondLookup,
) -> tuple[RDKitProximityBondWitness, ...]:
    """Return RDKit-inferred bonds adjacent to the failing atom indices."""

    witnesses: list[RDKitProximityBondWitness] = []
    for atom_index in sorted(atom_indices):
        atom = molecule.GetAtomWithIdx(atom_index)
        for bond in atom.GetBonds():
            begin_atom = bond.GetBeginAtom()
            end_atom = bond.GetEndAtom()
            begin_ref = _atom_ref_from_rdkit_atom(begin_atom)
            end_ref = _atom_ref_from_rdkit_atom(end_atom)
            if begin_ref is None or end_ref is None:
                continue
            if residue_id is not None and residue_id not in (
                begin_ref.residue_id,
                end_ref.residue_id,
            ):
                continue
            witness = RDKitProximityBondWitness(
                atom_ref_1=begin_ref,
                atom_ref_2=end_ref,
                element_1=begin_atom.GetSymbol(),
                element_2=end_atom.GetSymbol(),
                is_known_component_bond=known_bond_lookup.is_known_bond(
                    begin_ref,
                    end_ref,
                ),
            )
            if witness not in witnesses:
                witnesses.append(witness)

    return tuple(witnesses)


def _extra_proximity_bonds_from_problem_witnesses(
    problem_witnesses: Sequence[RDKitParserProblemWitness],
) -> tuple[RDKitProximityBondWitness, ...]:
    """Return unique extra proximity bonds from residue-local problem witnesses."""

    return tuple(
        dict.fromkeys(
            bond
            for witness in problem_witnesses
            for bond in witness.extra_proximity_bonds()
        )
    )


def _extra_proximity_bond_clusters_from_witnesses(
    extra_heavy_bonds: Sequence[RDKitProximityBondWitness],
) -> tuple[RDKitProximityBondCluster, ...]:
    """Return connected clusters of extra heavy-heavy parser witnesses."""

    if not extra_heavy_bonds:
        return ()

    residue_graph: dict[ResidueId, set[ResidueId]] = defaultdict(set)
    bonds_by_residue_id: dict[ResidueId, list[RDKitProximityBondWitness]] = defaultdict(
        list
    )
    for bond in extra_heavy_bonds:
        bond_residue_ids = bond.residue_ids()
        for residue_id in bond_residue_ids:
            bonds_by_residue_id[residue_id].append(bond)

        if len(bond_residue_ids) == 1:
            residue_graph[bond_residue_ids[0]].add(bond_residue_ids[0])
            continue

        left, right = bond_residue_ids
        residue_graph[left].add(right)
        residue_graph[right].add(left)

    clusters: list[RDKitProximityBondCluster] = []
    visited: set[ResidueId] = set()
    for seed in sorted(
        residue_graph,
        key=lambda residue_id: residue_id.display_token(),
    ):
        if seed in visited:
            continue

        stack = [seed]
        component_residue_ids: set[ResidueId] = set()
        component_bonds: set[RDKitProximityBondWitness] = set()
        while stack:
            residue_id = stack.pop()
            if residue_id in visited:
                continue

            visited.add(residue_id)
            component_residue_ids.add(residue_id)
            component_bonds.update(bonds_by_residue_id[residue_id])
            for neighbor in residue_graph[residue_id]:
                if neighbor not in visited:
                    stack.append(neighbor)

        clusters.append(
            RDKitProximityBondCluster(
                residue_ids=tuple(component_residue_ids),
                bonds=tuple(component_bonds),
            )
        )

    return tuple(
        sorted(
            clusters,
            key=lambda cluster: (
                -len(cluster.bonds),
                len(cluster.residue_ids),
                cluster.display_token(),
            ),
        )
    )


def _atom_ref_from_rdkit_atom(atom: _RDKitAtom) -> AtomRef | None:
    """Return a canonical atom reference from RDKit PDB residue metadata."""

    residue_info = atom.GetPDBResidueInfo()
    if residue_info is None:
        return None

    chain_id = residue_info.GetChainId().strip()
    atom_name = residue_info.GetName().strip()
    if not chain_id or not atom_name:
        return None

    insertion_code = residue_info.GetInsertionCode().strip()
    return AtomRef(
        residue_id=ResidueId(
            chain_id=chain_id,
            seq_num=residue_info.GetResidueNumber(),
            insertion_code=None if not insertion_code else insertion_code,
        ),
        atom_name=atom_name,
    )


def _known_atom_name_pairs_for_residue(
    component_id: str,
    atom_names: tuple[str, ...],
    *,
    component_library: ComponentLibrary,
) -> frozenset[tuple[str, str]]:
    """Return known intra-residue atom-name pairs for one residue topology."""

    template = component_library.get(component_id)
    if template is None:
        return frozenset()

    present_atom_names = frozenset(atom_names)
    atom_name_pairs: set[tuple[str, str]] = set()
    for atom_name in present_atom_names:
        atom_name_pairs.update(
            _sorted_atom_name_pair(atom_name, bonded_atom_name)
            for bonded_atom_name in template.definition.bonded_atom_names(atom_name)
            if bonded_atom_name in present_atom_names
        )

    hydrogen_anchor_by_name = template.template_hydrogen_anchor_by_name(
        tuple(sorted(present_atom_names)),
    )
    atom_name_pairs.update(
        _sorted_atom_name_pair(hydrogen_atom_name, anchor_atom_name)
        for hydrogen_atom_name, anchor_atom_name in hydrogen_anchor_by_name.items()
        if hydrogen_atom_name in present_atom_names
        and anchor_atom_name in present_atom_names
    )
    return frozenset(atom_name_pairs)


def _sorted_atom_name_pair(atom_name_1: str, atom_name_2: str) -> tuple[str, str]:
    """Return a deterministic atom-name pair key."""

    if atom_name_1 <= atom_name_2:
        return atom_name_1, atom_name_2

    return atom_name_2, atom_name_1


def _topology_known_atom_name_pairs_by_residue_id(
    structure: ProteinStructure,
) -> Mapping[ResidueId, frozenset[tuple[str, str]]]:
    """Return residue-local parser-known topology bonds by residue id."""

    atom_name_pairs_by_residue_id: dict[
        ResidueId,
        set[tuple[str, str]],
    ] = defaultdict(set)
    for bond in structure.topology.bonds:
        if not _topology_bond_is_parser_known(bond):
            continue

        residue_index_1 = structure.constitution.residue_index_for_atom_index(
            bond.atom_index_1
        )
        residue_index_2 = structure.constitution.residue_index_for_atom_index(
            bond.atom_index_2
        )
        if residue_index_1 != residue_index_2:
            continue

        atom_site_1 = structure.constitution.atom_site_at(bond.atom_index_1)
        atom_site_2 = structure.constitution.atom_site_at(bond.atom_index_2)
        residue_id = structure.constitution.residue_site_at(residue_index_1).residue_id
        atom_name_pairs_by_residue_id[residue_id].add(
            _sorted_atom_name_pair(atom_site_1.name, atom_site_2.name)
        )

    return {
        residue_id: frozenset(atom_name_pairs)
        for residue_id, atom_name_pairs in atom_name_pairs_by_residue_id.items()
    }


def _topology_known_inter_residue_endpoint_pairs(
    structure: ProteinStructure,
) -> frozenset[tuple[AtomRef, AtomRef]]:
    """Return inter-residue parser-known topology bonds in atom-ref address space."""

    endpoint_pairs: set[tuple[AtomRef, AtomRef]] = set()
    for bond in structure.topology.bonds:
        if not _topology_bond_is_parser_known(bond):
            continue

        atom_ref_1 = structure.constitution.atom_ref_at(bond.atom_index_1)
        atom_ref_2 = structure.constitution.atom_ref_at(bond.atom_index_2)
        if atom_ref_1.residue_id == atom_ref_2.residue_id:
            continue

        endpoint_pairs.add(_atom_ref_endpoint_pair(atom_ref_1, atom_ref_2))

    return frozenset(endpoint_pairs)


def _topology_bond_is_parser_known(bond: TopologyBond) -> bool:
    """Return whether RDKit parser burden should treat a bond as known topology."""

    return (
        is_covalent_like_relationship(bond)
        or bond.provenance is BondProvenance.SOURCE_EXPLICIT
    )


def _atom_ref_endpoint_pair(
    atom_ref_1: AtomRef,
    atom_ref_2: AtomRef,
) -> tuple[AtomRef, AtomRef]:
    """Return one canonical atom-ref endpoint pair."""

    left, right = sorted((atom_ref_1, atom_ref_2))
    return left, right


def _is_known_polymer_backbone_link(
    atom_ref_1: AtomRef,
    atom_ref_2: AtomRef,
    *,
    component_id_by_residue_id: Mapping[ResidueId, str],
    component_library: ComponentLibrary,
) -> bool:
    """Return whether one inferred inter-residue bond is a peptide backbone link."""

    left, right = sorted(
        (atom_ref_1, atom_ref_2),
        key=lambda atom_ref: (
            atom_ref.residue_id.chain_id,
            atom_ref.residue_id.seq_num,
            atom_ref.residue_id.insertion_code or "",
            atom_ref.atom_name,
        ),
    )
    if left.residue_id.chain_id != right.residue_id.chain_id:
        return False
    if left.residue_id.insertion_code or right.residue_id.insertion_code:
        return False
    if right.residue_id.seq_num != left.residue_id.seq_num + 1:
        return False
    if (left.atom_name, right.atom_name) != ("C", "N"):
        return False

    left_component_id = component_id_by_residue_id.get(left.residue_id)
    right_component_id = component_id_by_residue_id.get(right.residue_id)
    if left_component_id is None or right_component_id is None:
        return False

    return (
        component_library.get(left_component_id) is not None
        and component_library.get(right_component_id) is not None
    )

"""Observed-to-external protein sequence alignment models."""

from dataclasses import dataclass
from enum import Enum, IntEnum

from protrepair.structure.constitution import ChainSite, ResidueSite
from protrepair.structure.labels import ResidueId
from protrepair.structure.polymer_blueprint import PolymerChainBlueprint

RESIDUE_TOKEN_BY_COMPONENT_ID: dict[str, str] = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "ASP": "D",
    "CYS": "C",
    "GLN": "Q",
    "GLU": "E",
    "GLY": "G",
    "HIS": "H",
    "ILE": "I",
    "LEU": "L",
    "LYS": "K",
    "MET": "M",
    "PHE": "F",
    "PRO": "P",
    "SER": "S",
    "THR": "T",
    "TRP": "W",
    "TYR": "Y",
    "VAL": "V",
}


class SequenceAlignmentRelation(str, Enum):
    """Closed per-column relations for observed-to-reference sequence alignment."""

    MATCH = "match"
    SUBSTITUTION = "substitution"
    REFERENCE_ONLY = "reference_only"
    OBSERVED_ONLY = "observed_only"


class SequenceDifferenceKind(str, Enum):
    """Closed grouped difference kinds derived from one sequence alignment."""

    SUBSTITUTION = "substitution"
    OBSERVED_INSERTION = "observed_insertion"
    INTERNAL_DELETION = "internal_deletion"
    N_TERMINAL_TRUNCATION = "n_terminal_truncation"
    C_TERMINAL_TRUNCATION = "c_terminal_truncation"


class AlignmentMove(IntEnum):
    """Backtrace moves for global sequence alignment."""

    DIAGONAL = 1
    UP = 2
    LEFT = 3


class ReferenceSequenceSourceAuthority(str, Enum):
    """Closed authorities for source-family identities attached to references."""

    UNIPROT = "uniprot"


@dataclass(frozen=True, slots=True)
class ReferenceSequenceSourceFamily:
    """Typed source-family identity for one external reference attachment."""

    authority: ReferenceSequenceSourceAuthority
    identifier: str

    def __post_init__(self) -> None:
        if not isinstance(self.authority, ReferenceSequenceSourceAuthority):
            raise TypeError(
                "reference sequence source families require a "
                "ReferenceSequenceSourceAuthority"
            )

        identifier = self.identifier.strip()
        if not identifier:
            raise ValueError(
                "reference sequence source families require a non-blank identifier"
            )

        object.__setattr__(self, "identifier", identifier)

    @classmethod
    def uniprot(cls, accession: str) -> "ReferenceSequenceSourceFamily":
        """Return one UniProt-backed source-family identity."""

        normalized_accession = accession.strip().upper()
        if not normalized_accession:
            raise ValueError(
                "UniProt source-family identities require a non-blank accession"
            )

        return cls(
            authority=ReferenceSequenceSourceAuthority.UNIPROT,
            identifier=normalized_accession,
        )


@dataclass(frozen=True, slots=True)
class ObservedSequenceResidue:
    """One observed residue projected into one sequence token."""

    residue_id: ResidueId
    component_id: str
    token: str

    def __post_init__(self) -> None:
        if not isinstance(self.residue_id, ResidueId):
            raise TypeError(
                "observed sequence residues require a ResidueId value"
            )

        component_id = self.component_id.strip().upper()
        token = self.token.strip().upper()
        if not component_id:
            raise ValueError("observed sequence residues require a component id")
        if len(token) != 1 or not token.isalpha():
            raise ValueError(
                "observed sequence residues require one alphabetic token"
            )

        object.__setattr__(self, "component_id", component_id)
        object.__setattr__(self, "token", token)


@dataclass(frozen=True, slots=True)
class ReferenceSequenceAttachment:
    """Optional external source attachment for one reference blueprint."""

    label: str
    source_family: ReferenceSequenceSourceFamily | None = None
    is_canonical: bool | None = None

    def __post_init__(self) -> None:
        label = self.label.strip()
        if not label:
            raise ValueError(
                "reference sequence attachments require a non-blank label"
            )

        source_family = self.source_family
        if source_family is not None and not isinstance(
            source_family,
            ReferenceSequenceSourceFamily,
        ):
            raise TypeError(
                "reference sequence attachments require source_family to be a "
                "ReferenceSequenceSourceFamily or None"
            )

        object.__setattr__(self, "label", label)


@dataclass(frozen=True, slots=True)
class ReferenceSequenceCandidate:
    """One canonical reference blueprint plus optional external source attachment."""

    blueprint: PolymerChainBlueprint
    attachment: ReferenceSequenceAttachment | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.blueprint, PolymerChainBlueprint):
            raise TypeError(
                "reference sequence candidates require a PolymerChainBlueprint"
            )
        if self.attachment is not None and not isinstance(
            self.attachment,
            ReferenceSequenceAttachment,
        ):
            raise TypeError(
                "reference sequence candidates require attachment to be a "
                "ReferenceSequenceAttachment or None"
            )


@dataclass(frozen=True, slots=True)
class ObservedChainSequence:
    """Observed polymer chain sequence projected from canonical residues."""

    chain_id: str
    residues: tuple[ObservedSequenceResidue, ...]

    def __post_init__(self) -> None:
        chain_id = self.chain_id.strip()
        if not chain_id:
            raise ValueError("observed chain sequences require a non-blank chain id")

        residues = tuple(self.residues)
        for residue in residues:
            if not isinstance(residue, ObservedSequenceResidue):
                raise TypeError(
                    "observed chain sequences require ObservedSequenceResidue values"
                )
            if residue.residue_id.chain_id != chain_id:
                raise ValueError(
                    "observed chain sequence residues must stay on one chain"
                )

        object.__setattr__(self, "chain_id", chain_id)
        object.__setattr__(self, "residues", residues)

    @classmethod
    def from_chain(cls, chain: ChainSite) -> "ObservedChainSequence":
        """Project one canonical chain into one tokenized observed sequence."""

        if not isinstance(chain, ChainSite):
            raise TypeError("observed chain sequences require a ChainSite value")

        return cls(
            chain_id=chain.chain_id,
            residues=tuple(
                _observed_sequence_residue_from_residue(residue)
                for residue in chain.residues
                if not residue.is_hetero
            ),
        )

    def tokens(self) -> tuple[str, ...]:
        """Return one token sequence in chain order."""

        return tuple(residue.token for residue in self.residues)

    def align_to_candidate(
        self,
        reference_candidate: ReferenceSequenceCandidate,
    ) -> "ObservedSequenceAlignment":
        """Return one global alignment against one canonical reference candidate."""

        if not isinstance(reference_candidate, ReferenceSequenceCandidate):
            raise TypeError(
                "observed sequence alignment requires a ReferenceSequenceCandidate"
            )

        return self.align_to_blueprint(
            reference_candidate.blueprint,
            reference_attachment=reference_candidate.attachment,
        )

    def align_to_candidates(
        self,
        reference_candidates: tuple[ReferenceSequenceCandidate, ...],
    ) -> "ObservedSequenceAlignmentSet":
        """Return alignments against one explicit set of canonical candidates."""

        if not reference_candidates:
            raise ValueError(
                "observed sequence candidate alignment requires at least one "
                "reference candidate"
            )

        return ObservedSequenceAlignmentSet(
            observed_sequence=self,
            alignments=tuple(
                self.align_to_candidate(candidate)
                for candidate in reference_candidates
            ),
        )

    def align_to_blueprint(
        self,
        reference_blueprint: PolymerChainBlueprint,
        *,
        reference_attachment: ReferenceSequenceAttachment | None = None,
    ) -> "ObservedSequenceAlignment":
        """Return one global alignment against a canonical polymer blueprint."""

        if not isinstance(reference_blueprint, PolymerChainBlueprint):
            raise TypeError(
                "observed sequence alignment requires a PolymerChainBlueprint"
            )

        columns, score = _global_alignment_columns(
            observed_residues=self.residues,
            reference_sequence=reference_blueprint.sequence(),
            reference_offset=reference_blueprint.first_sequence_position(),
        )
        return ObservedSequenceAlignment(
            observed_sequence=self,
            reference_blueprint=reference_blueprint,
            reference_attachment=reference_attachment,
            columns=columns,
            score=score,
        )


@dataclass(frozen=True, slots=True)
class SequenceAlignmentColumn:
    """One aligned observed/reference sequence column."""

    relation: SequenceAlignmentRelation
    observed_residue: ObservedSequenceResidue | None = None
    reference_position: int | None = None
    reference_token: str | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.relation, SequenceAlignmentRelation):
            raise TypeError(
                "sequence alignment columns require a SequenceAlignmentRelation"
            )
        if self.observed_residue is not None and not isinstance(
            self.observed_residue,
            ObservedSequenceResidue,
        ):
            raise TypeError(
                "sequence alignment columns require an observed residue or None"
            )
        if self.reference_position is not None and self.reference_position <= 0:
            raise ValueError(
                "sequence alignment reference positions must be positive"
            )

        reference_token = self.reference_token
        if reference_token is not None:
            reference_token = reference_token.strip().upper()
            if len(reference_token) != 1 or not reference_token.isalpha():
                raise ValueError(
                    "sequence alignment reference tokens must be one alphabetic "
                    "character"
                )

        object.__setattr__(self, "reference_token", reference_token)


@dataclass(frozen=True, slots=True)
class SequenceDifference:
    """One grouped difference segment derived from an alignment."""

    kind: SequenceDifferenceKind
    observed_residue_ids: tuple[ResidueId, ...] = ()
    reference_start: int | None = None
    reference_end: int | None = None
    observed_sequence: str = ""
    reference_sequence: str = ""

    def __post_init__(self) -> None:
        if not isinstance(self.kind, SequenceDifferenceKind):
            raise TypeError(
                "sequence differences require a SequenceDifferenceKind"
            )

        observed_sequence = self.observed_sequence.strip().upper()
        reference_sequence = self.reference_sequence.strip().upper()
        if observed_sequence and not observed_sequence.isalpha():
            raise ValueError(
                "sequence differences require alphabetic observed sequences"
            )
        if reference_sequence and not reference_sequence.isalpha():
            raise ValueError(
                "sequence differences require alphabetic reference sequences"
            )
        if self.reference_start is not None and self.reference_start <= 0:
            raise ValueError("sequence difference reference_start must be positive")
        if self.reference_end is not None and self.reference_end <= 0:
            raise ValueError("sequence difference reference_end must be positive")
        if (
            self.reference_start is not None
            and self.reference_end is not None
            and self.reference_start > self.reference_end
        ):
            raise ValueError(
                "sequence difference reference interval must not be reversed"
            )

        object.__setattr__(self, "observed_sequence", observed_sequence)
        object.__setattr__(self, "reference_sequence", reference_sequence)


@dataclass(frozen=True, slots=True)
class ObservedSequenceAlignment:
    """One observed-chain alignment against one canonical reference blueprint."""

    observed_sequence: ObservedChainSequence
    reference_blueprint: PolymerChainBlueprint
    columns: tuple[SequenceAlignmentColumn, ...]
    score: int
    reference_attachment: ReferenceSequenceAttachment | None = None

    def __post_init__(self) -> None:
        if not isinstance(self.observed_sequence, ObservedChainSequence):
            raise TypeError(
                "observed sequence alignments require an ObservedChainSequence"
            )
        if not isinstance(self.reference_blueprint, PolymerChainBlueprint):
            raise TypeError(
                "observed sequence alignments require a PolymerChainBlueprint"
            )
        if self.reference_attachment is not None and not isinstance(
            self.reference_attachment,
            ReferenceSequenceAttachment,
        ):
            raise TypeError(
                "observed sequence alignments require reference_attachment to be a "
                "ReferenceSequenceAttachment or None"
            )

    def match_count(self) -> int:
        """Return the number of exact-match columns."""

        return sum(
            column.relation is SequenceAlignmentRelation.MATCH
            for column in self.columns
        )

    def edit_count(self) -> int:
        """Return the number of non-match columns."""

        return sum(
            column.relation is not SequenceAlignmentRelation.MATCH
            for column in self.columns
        )

    def differences(self) -> tuple[SequenceDifference, ...]:
        """Return grouped sequence differences derived from this alignment."""

        return _group_alignment_differences(self.columns)

    def preference_key(self) -> tuple[int, int, int]:
        """Return a sortable key where smaller tuples are better."""

        return (-self.match_count(), self.edit_count(), -self.score)

    def interpretation(
        self,
        *,
        prefers_isoform: bool = False,
    ) -> "SequenceAlignmentInterpretation":
        """Return one typed interpretation for this alignment."""

        return SequenceAlignmentInterpretation(
            alignment=self,
            prefers_isoform=prefers_isoform,
        )


@dataclass(frozen=True, slots=True)
class SequenceAlignmentInterpretation:
    """One best-effort semantic interpretation of an observed/reference alignment."""

    alignment: ObservedSequenceAlignment
    prefers_isoform: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.alignment, ObservedSequenceAlignment):
            raise TypeError(
                "sequence alignment interpretations require an "
                "ObservedSequenceAlignment"
            )

    def differences(self) -> tuple[SequenceDifference, ...]:
        """Return grouped differences for this interpreted alignment."""

        return self.alignment.differences()

    def is_exact_match(self) -> bool:
        """Return whether the alignment is difference-free."""

        return not self.differences()

    def has_substitutions(self) -> bool:
        """Return whether the alignment contains residue substitutions."""

        return any(
            difference.kind is SequenceDifferenceKind.SUBSTITUTION
            for difference in self.differences()
        )

    def has_insertions(self) -> bool:
        """Return whether the alignment contains observed-only insertions."""

        return any(
            difference.kind is SequenceDifferenceKind.OBSERVED_INSERTION
            for difference in self.differences()
        )

    def has_truncations(self) -> bool:
        """Return whether the alignment contains N- or C-terminal truncation."""

        return any(
            difference.kind
            in {
                SequenceDifferenceKind.N_TERMINAL_TRUNCATION,
                SequenceDifferenceKind.C_TERMINAL_TRUNCATION,
            }
            for difference in self.differences()
        )


@dataclass(frozen=True, slots=True)
class ObservedSequenceAlignmentSet:
    """Candidate alignments of one observed chain against one reference set."""

    observed_sequence: ObservedChainSequence
    alignments: tuple[ObservedSequenceAlignment, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.observed_sequence, ObservedChainSequence):
            raise TypeError(
                "observed sequence alignment sets require an ObservedChainSequence"
            )
        if not self.alignments:
            raise ValueError(
                "observed sequence alignment sets require at least one alignment"
            )

    def best_alignments(self) -> tuple[ObservedSequenceAlignment, ...]:
        """Return best-scoring alignments in stable first-seen order."""

        best_key = min(
            alignment.preference_key()
            for alignment in self.alignments
        )
        return tuple(
            alignment
            for alignment in self.alignments
            if alignment.preference_key() == best_key
        )

    def primary_alignment(self) -> ObservedSequenceAlignment | None:
        """Return one unique best alignment when the family is unambiguous."""

        best_alignments = self.best_alignments()
        if len(best_alignments) != 1:
            return None

        return best_alignments[0]

    def primary_interpretation(self) -> SequenceAlignmentInterpretation | None:
        """Return the interpretation of the unique best alignment when present."""

        primary_alignment = self.primary_alignment()
        if primary_alignment is None:
            return None

        return primary_alignment.interpretation(
            prefers_isoform=(
                primary_alignment.reference_attachment is not None
                and primary_alignment.reference_attachment.is_canonical is False
            )
        )


def _observed_sequence_residue_from_residue(
    residue: ResidueSite,
) -> ObservedSequenceResidue:
    """Project one residue into one observed sequence residue token."""

    token = RESIDUE_TOKEN_BY_COMPONENT_ID.get(residue.component_id, "X")
    return ObservedSequenceResidue(
        residue_id=residue.residue_id,
        component_id=residue.component_id,
        token=token,
    )


def _substitution_score(observed_token: str, reference_token: str) -> int:
    """Return one simple substitution score for global alignment."""

    if observed_token == reference_token:
        return 2
    if observed_token == "X" or reference_token == "X":
        return 0

    return -1


def _global_alignment_columns(
    *,
    observed_residues: tuple[ObservedSequenceResidue, ...],
    reference_sequence: str,
    reference_offset: int,
) -> tuple[tuple[SequenceAlignmentColumn, ...], int]:
    """Return alignment columns and score from one global DP alignment."""

    observed_length = len(observed_residues)
    reference_length = len(reference_sequence)
    scores = [
        [0] * (reference_length + 1)
        for _ in range(observed_length + 1)
    ]
    moves = [
        [AlignmentMove.DIAGONAL] * (reference_length + 1)
        for _ in range(observed_length + 1)
    ]

    for observed_index in range(1, observed_length + 1):
        scores[observed_index][0] = -observed_index
        moves[observed_index][0] = AlignmentMove.UP
    for reference_index in range(1, reference_length + 1):
        scores[0][reference_index] = -reference_index
        moves[0][reference_index] = AlignmentMove.LEFT

    for observed_index in range(1, observed_length + 1):
        for reference_index in range(1, reference_length + 1):
            observed_token = observed_residues[observed_index - 1].token
            reference_token = reference_sequence[reference_index - 1]
            diagonal_score = (
                scores[observed_index - 1][reference_index - 1]
                + _substitution_score(observed_token, reference_token)
            )
            up_score = scores[observed_index - 1][reference_index] - 1
            left_score = scores[observed_index][reference_index - 1] - 1

            if diagonal_score >= up_score and diagonal_score >= left_score:
                scores[observed_index][reference_index] = diagonal_score
                moves[observed_index][reference_index] = AlignmentMove.DIAGONAL
            elif up_score >= left_score:
                scores[observed_index][reference_index] = up_score
                moves[observed_index][reference_index] = AlignmentMove.UP
            else:
                scores[observed_index][reference_index] = left_score
                moves[observed_index][reference_index] = AlignmentMove.LEFT

    observed_index = observed_length
    reference_index = reference_length
    reversed_columns: list[SequenceAlignmentColumn] = []
    while observed_index > 0 or reference_index > 0:
        move = moves[observed_index][reference_index]
        if (
            move is AlignmentMove.DIAGONAL
            and observed_index > 0
            and reference_index > 0
        ):
            observed_residue = observed_residues[observed_index - 1]
            reference_token = reference_sequence[reference_index - 1]
            reversed_columns.append(
                SequenceAlignmentColumn(
                    relation=(
                        SequenceAlignmentRelation.MATCH
                        if observed_residue.token == reference_token
                        else SequenceAlignmentRelation.SUBSTITUTION
                    ),
                    observed_residue=observed_residue,
                    reference_position=reference_offset + reference_index - 1,
                    reference_token=reference_token,
                )
            )
            observed_index -= 1
            reference_index -= 1
            continue

        if move is AlignmentMove.UP and observed_index > 0:
            observed_residue = observed_residues[observed_index - 1]
            reversed_columns.append(
                SequenceAlignmentColumn(
                    relation=SequenceAlignmentRelation.OBSERVED_ONLY,
                    observed_residue=observed_residue,
                )
            )
            observed_index -= 1
            continue

        if reference_index <= 0:
            raise AssertionError("alignment backtrace exhausted reference unexpectedly")

        reference_token = reference_sequence[reference_index - 1]
        reversed_columns.append(
            SequenceAlignmentColumn(
                relation=SequenceAlignmentRelation.REFERENCE_ONLY,
                reference_position=reference_offset + reference_index - 1,
                reference_token=reference_token,
            )
        )
        reference_index -= 1

    reversed_columns.reverse()
    return tuple(reversed_columns), scores[observed_length][reference_length]


def _group_alignment_differences(
    columns: tuple[SequenceAlignmentColumn, ...],
) -> tuple[SequenceDifference, ...]:
    """Return grouped difference segments from one alignment column stream."""

    leading_reference_only_end = 0
    while (
        leading_reference_only_end < len(columns)
        and columns[leading_reference_only_end].relation
        is SequenceAlignmentRelation.REFERENCE_ONLY
    ):
        leading_reference_only_end += 1

    trailing_reference_only_start = len(columns)
    while (
        trailing_reference_only_start > leading_reference_only_end
        and columns[trailing_reference_only_start - 1].relation
        is SequenceAlignmentRelation.REFERENCE_ONLY
    ):
        trailing_reference_only_start -= 1

    differences: list[SequenceDifference] = []
    index = 0
    while index < len(columns):
        column = columns[index]
        if column.relation is SequenceAlignmentRelation.MATCH:
            index += 1
            continue

        run_end = index + 1
        while run_end < len(columns) and columns[run_end].relation is column.relation:
            run_end += 1

        run_columns = columns[index:run_end]
        if column.relation is SequenceAlignmentRelation.SUBSTITUTION:
            kind = SequenceDifferenceKind.SUBSTITUTION
        elif column.relation is SequenceAlignmentRelation.OBSERVED_ONLY:
            kind = SequenceDifferenceKind.OBSERVED_INSERTION
        elif index == 0 and run_end <= leading_reference_only_end:
            kind = SequenceDifferenceKind.N_TERMINAL_TRUNCATION
        elif index >= trailing_reference_only_start:
            kind = SequenceDifferenceKind.C_TERMINAL_TRUNCATION
        else:
            kind = SequenceDifferenceKind.INTERNAL_DELETION

        observed_residue_ids = tuple(
            run_column.observed_residue.residue_id
            for run_column in run_columns
            if run_column.observed_residue is not None
        )
        reference_positions = tuple(
            run_column.reference_position
            for run_column in run_columns
            if run_column.reference_position is not None
        )
        observed_sequence = "".join(
            run_column.observed_residue.token
            for run_column in run_columns
            if run_column.observed_residue is not None
        )
        reference_sequence = "".join(
            run_column.reference_token or ""
            for run_column in run_columns
        )
        differences.append(
            SequenceDifference(
                kind=kind,
                observed_residue_ids=observed_residue_ids,
                reference_start=(
                    None if not reference_positions else reference_positions[0]
                ),
                reference_end=(
                    None if not reference_positions else reference_positions[-1]
                ),
                observed_sequence=observed_sequence,
                reference_sequence=reference_sequence,
            )
        )
        index = run_end

    return tuple(differences)

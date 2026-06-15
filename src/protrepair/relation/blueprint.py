"""Canonical relations between realized structure residues and polymer blueprints."""

from dataclasses import dataclass
from typing import TYPE_CHECKING

from protrepair.relation.sequence_alignment import (
    ObservedChainSequence,
    ObservedSequenceAlignment,
    SequenceAlignmentColumn,
    SequenceAlignmentRelation,
)
from protrepair.structure.labels import ResidueId
from protrepair.structure.polymer_blueprint import PolymerChainBlueprint

if TYPE_CHECKING:
    from protrepair.structure.aggregate import ProteinStructure


@dataclass(frozen=True, slots=True)
class StructureBlueprintResidueMapping:
    """One realized residue mapped onto one canonical blueprint sequence slot."""

    structure_chain_id: str
    blueprint_chain_id: str | None
    residue_id: ResidueId
    sequence_position: int
    relation: SequenceAlignmentRelation

    def __post_init__(self) -> None:
        structure_chain_id = self.structure_chain_id.strip()
        if not structure_chain_id:
            raise ValueError(
                "structure blueprint residue mappings require a non-blank "
                "structure_chain_id"
            )
        if self.residue_id.chain_id != structure_chain_id:
            raise ValueError(
                "structure blueprint residue mappings require residue_id to stay on "
                "the mapped structure chain"
            )
        if self.sequence_position <= 0:
            raise ValueError(
                "structure blueprint residue mappings require positive "
                "sequence_position"
            )
        if self.relation not in {
            SequenceAlignmentRelation.MATCH,
            SequenceAlignmentRelation.SUBSTITUTION,
        }:
            raise ValueError(
                "structure blueprint residue mappings only admit aligned "
                "match/substitution relations"
            )

        blueprint_chain_id = self.blueprint_chain_id
        if blueprint_chain_id is not None:
            blueprint_chain_id = blueprint_chain_id.strip() or None

        object.__setattr__(self, "structure_chain_id", structure_chain_id)
        object.__setattr__(self, "blueprint_chain_id", blueprint_chain_id)


@dataclass(frozen=True, slots=True)
class StructureBlueprintCoverageGap:
    """One blueprint-covered but unrealized residue span for one structure chain."""

    structure_chain_id: str
    blueprint_chain_id: str | None
    absent_sequence_positions: tuple[int, ...]
    preceding_residue_id: ResidueId | None = None
    following_residue_id: ResidueId | None = None

    def __post_init__(self) -> None:
        structure_chain_id = self.structure_chain_id.strip()
        if not structure_chain_id:
            raise ValueError(
                "structure blueprint coverage gaps require a non-blank "
                "structure_chain_id"
            )

        blueprint_chain_id = self.blueprint_chain_id
        if blueprint_chain_id is not None:
            blueprint_chain_id = blueprint_chain_id.strip() or None

        preceding_residue_id = self.preceding_residue_id
        following_residue_id = self.following_residue_id
        if preceding_residue_id is None and following_residue_id is None:
            raise ValueError(
                "structure blueprint coverage gaps require at least one anchor residue"
            )
        for anchor_residue_id in (preceding_residue_id, following_residue_id):
            if (
                anchor_residue_id is not None
                and anchor_residue_id.chain_id != structure_chain_id
            ):
                raise ValueError(
                    "structure blueprint coverage gap anchors must stay on the "
                    "mapped structure chain"
                )
        if (
            preceding_residue_id is not None
            and following_residue_id is not None
            and preceding_residue_id == following_residue_id
        ):
            raise ValueError(
                "structure blueprint coverage gaps require distinct anchors"
            )

        absent_sequence_positions = tuple(self.absent_sequence_positions)
        if not absent_sequence_positions:
            raise ValueError(
                "structure blueprint coverage gaps require at least one absent "
                "sequence position"
            )
        if any(
            sequence_position <= 0 for sequence_position in absent_sequence_positions
        ):
            raise ValueError(
                "structure blueprint coverage gaps require positive sequence positions"
            )
        if len(absent_sequence_positions) != len(set(absent_sequence_positions)):
            raise ValueError(
                "structure blueprint coverage gaps must not repeat sequence positions"
            )
        if absent_sequence_positions != tuple(sorted(absent_sequence_positions)):
            raise ValueError(
                "structure blueprint coverage gaps require increasing sequence "
                "positions"
            )

        object.__setattr__(self, "structure_chain_id", structure_chain_id)
        object.__setattr__(self, "blueprint_chain_id", blueprint_chain_id)
        object.__setattr__(self, "absent_sequence_positions", absent_sequence_positions)

    def is_internal(self) -> bool:
        """Return whether this gap has both flanking anchors."""

        return (
            self.preceding_residue_id is not None
            and self.following_residue_id is not None
        )

    def is_terminal(self) -> bool:
        """Return whether this gap touches one structure terminus."""

        return not self.is_internal()

    def is_prefix_terminal(self) -> bool:
        """Return whether this gap precedes the first realized residue."""

        return self.preceding_residue_id is None

    def is_suffix_terminal(self) -> bool:
        """Return whether this gap follows the last realized residue."""

        return self.following_residue_id is None


@dataclass(frozen=True, slots=True)
class StructureBlueprintCoverage:
    """One chain-local coverage relation between realized residues and a blueprint."""

    structure_chain_id: str
    blueprint: PolymerChainBlueprint
    residue_mappings: tuple[StructureBlueprintResidueMapping, ...]
    coverage_gaps: tuple[StructureBlueprintCoverageGap, ...]

    def __post_init__(self) -> None:
        structure_chain_id = self.structure_chain_id.strip()
        if not structure_chain_id:
            raise ValueError(
                "structure blueprint coverage requires a non-blank structure_chain_id"
            )
        if not isinstance(self.blueprint, PolymerChainBlueprint):
            raise TypeError(
                "structure blueprint coverage requires a PolymerChainBlueprint"
            )

        blueprint_positions = set(self.blueprint.sequence_positions())
        residue_mappings = tuple(self.residue_mappings)
        coverage_gaps = tuple(self.coverage_gaps)
        mapped_positions: set[int] = set()
        gap_positions: set[int] = set()

        for mapping in residue_mappings:
            if not isinstance(mapping, StructureBlueprintResidueMapping):
                raise TypeError(
                    "structure blueprint coverage requires "
                    "StructureBlueprintResidueMapping values"
                )
            if mapping.structure_chain_id != structure_chain_id:
                raise ValueError(
                    "structure blueprint coverage mappings must stay on one "
                    "structure chain"
                )
            if mapping.blueprint_chain_id != self.blueprint.chain_id:
                raise ValueError(
                    "structure blueprint coverage mappings must target one "
                    "blueprint chain"
                )
            if mapping.sequence_position not in blueprint_positions:
                raise ValueError(
                    "structure blueprint coverage mappings must target sequence "
                    "positions present in the blueprint"
                )
            if mapping.sequence_position in mapped_positions:
                raise ValueError(
                    "structure blueprint coverage must not repeat mapped "
                    "sequence positions"
                )
            mapped_positions.add(mapping.sequence_position)

        for gap in coverage_gaps:
            if not isinstance(gap, StructureBlueprintCoverageGap):
                raise TypeError(
                    "structure blueprint coverage requires "
                    "StructureBlueprintCoverageGap values"
                )
            if gap.structure_chain_id != structure_chain_id:
                raise ValueError(
                    "structure blueprint coverage gaps must stay on one structure chain"
                )
            if gap.blueprint_chain_id != self.blueprint.chain_id:
                raise ValueError(
                    "structure blueprint coverage gaps must target one blueprint chain"
                )
            for sequence_position in gap.absent_sequence_positions:
                if sequence_position not in blueprint_positions:
                    raise ValueError(
                        "structure blueprint coverage gaps must target sequence "
                        "positions present in the blueprint"
                    )
                if (
                    sequence_position in mapped_positions
                    or sequence_position in gap_positions
                ):
                    raise ValueError(
                        "structure blueprint coverage must not overlap mapped and "
                        "absent sequence positions"
                    )
                gap_positions.add(sequence_position)

        object.__setattr__(self, "structure_chain_id", structure_chain_id)
        object.__setattr__(self, "residue_mappings", residue_mappings)
        object.__setattr__(self, "coverage_gaps", coverage_gaps)

    @classmethod
    def from_alignment(
        cls,
        alignment: ObservedSequenceAlignment,
    ) -> "StructureBlueprintCoverage":
        """Derive canonical blueprint coverage from one observed/reference alignment."""

        if not isinstance(alignment, ObservedSequenceAlignment):
            raise TypeError(
                "structure blueprint coverage requires an ObservedSequenceAlignment"
            )

        structure_chain_id = alignment.observed_sequence.chain_id
        blueprint = alignment.reference_blueprint
        residue_mappings = tuple(
            StructureBlueprintResidueMapping(
                structure_chain_id=structure_chain_id,
                blueprint_chain_id=blueprint.chain_id,
                residue_id=column.observed_residue.residue_id,
                sequence_position=column.reference_position,
                relation=column.relation,
            )
            for column in alignment.columns
            if (
                column.observed_residue is not None
                and column.reference_position is not None
                and column.relation
                in {
                    SequenceAlignmentRelation.MATCH,
                    SequenceAlignmentRelation.SUBSTITUTION,
                }
            )
        )
        coverage_gaps = tuple(
            _coverage_gap_from_reference_only_run(
                structure_chain_id=structure_chain_id,
                blueprint_chain_id=blueprint.chain_id,
                columns=alignment.columns,
                start_index=start_index,
                end_index=end_index,
            )
            for start_index, end_index in _reference_only_runs(alignment.columns)
        )

        return cls(
            structure_chain_id=structure_chain_id,
            blueprint=blueprint,
            residue_mappings=residue_mappings,
            coverage_gaps=coverage_gaps,
        )

    @classmethod
    def from_structure(
        cls,
        structure: "ProteinStructure",
        chain_id: str,
    ) -> "StructureBlueprintCoverage":
        """Derive blueprint coverage from one structure's attached blueprint."""

        from protrepair.structure.aggregate import ProteinStructure

        if not isinstance(structure, ProteinStructure):
            raise TypeError("structure blueprint coverage requires a ProteinStructure")
        if structure.polymer_blueprint is None:
            raise ValueError(
                "structure blueprint coverage requires the structure to carry a "
                "polymer blueprint"
            )

        observed_sequence = ObservedChainSequence.from_chain(
            structure.constitution.chain(chain_id)
        )
        alignment = observed_sequence.align_to_blueprint(
            structure.polymer_blueprint.chain(chain_id)
        )
        return cls.from_alignment(alignment)


def _reference_only_runs(
    columns: tuple[SequenceAlignmentColumn, ...],
) -> tuple[tuple[int, int], ...]:
    """Return inclusive-exclusive index pairs for reference-only runs."""

    runs: list[tuple[int, int]] = []
    index = 0
    while index < len(columns):
        if columns[index].relation is not SequenceAlignmentRelation.REFERENCE_ONLY:
            index += 1
            continue

        run_end = index + 1
        while (
            run_end < len(columns)
            and columns[run_end].relation is SequenceAlignmentRelation.REFERENCE_ONLY
        ):
            run_end += 1

        runs.append((index, run_end))
        index = run_end

    return tuple(runs)


def _coverage_gap_from_reference_only_run(
    *,
    structure_chain_id: str,
    blueprint_chain_id: str | None,
    columns: tuple[SequenceAlignmentColumn, ...],
    start_index: int,
    end_index: int,
) -> StructureBlueprintCoverageGap:
    """Build one canonical coverage gap from one reference-only alignment run."""

    preceding_column = None if start_index == 0 else columns[start_index - 1]
    following_column = None if end_index == len(columns) else columns[end_index]

    preceding_residue_id = (
        None
        if preceding_column is None or preceding_column.observed_residue is None
        else preceding_column.observed_residue.residue_id
    )
    following_residue_id = (
        None
        if following_column is None or following_column.observed_residue is None
        else following_column.observed_residue.residue_id
    )
    absent_sequence_positions = tuple(
        column.reference_position
        for column in columns[start_index:end_index]
        if column.reference_position is not None
    )

    return StructureBlueprintCoverageGap(
        structure_chain_id=structure_chain_id,
        blueprint_chain_id=blueprint_chain_id,
        preceding_residue_id=preceding_residue_id,
        following_residue_id=following_residue_id,
        absent_sequence_positions=absent_sequence_positions,
    )

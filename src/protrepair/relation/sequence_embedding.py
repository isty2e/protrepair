"""Bounds shared by all exact, order-preserving sequence embeddings."""

from collections import defaultdict
from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from itertools import pairwise


@dataclass(frozen=True, slots=True)
class SequenceEmbedding:
    """One-based position bounds and obligatory internal gaps.

    Parameters
    ----------
    first, last : tuple[int, ...]
        Earliest and latest possible position of each observed monomer.
    gaps_after : tuple[int, ...]
        Zero-based observed offsets at which no embedding permits adjacency.
    """

    first: tuple[int, ...]
    last: tuple[int, ...]
    gaps_after: tuple[int, ...]

    @classmethod
    def match(
        cls, observed: tuple[str, ...], declared: tuple[str, ...]
    ) -> "SequenceEmbedding | None":
        """Find bounds and compulsory gaps without enumerating alignments.

        Parameters
        ----------
        observed, declared : tuple[str, ...]
            Component IDs in original observed and declared sequence order.

        Returns
        -------
        SequenceEmbedding or None
            No result for an empty observation or when any monomer cannot be
            embedded. Repeated tokens need not have a unique alignment.
        """
        if not observed:
            return None
        first = _greedy_positions(observed, enumerate(declared, start=1))
        if first is None:
            return None
        reverse = zip(range(len(declared), 0, -1), reversed(declared), strict=True)
        backward = _greedy_positions(reversed(observed), reverse)
        assert backward is not None
        last = backward[::-1]

        # An adjacent pair can extend a valid prefix and suffix exactly when
        # its left position lies in [first[i], last[i+1]). Both bounds advance.
        by_pair: dict[tuple[str, str], list[int]] = defaultdict(list)
        for pos, pair in enumerate(pairwise(declared), start=1):
            by_pair[pair].append(pos)
        cursors: dict[tuple[str, str], int] = {}
        gaps: list[int] = []
        for i, pair in enumerate(pairwise(observed)):
            positions = by_pair[pair]
            cursor = cursors.get(pair, 0)
            while cursor < len(positions) and positions[cursor] < first[i]:
                cursor += 1
            cursors[pair] = cursor
            if cursor == len(positions) or positions[cursor] >= last[i + 1]:
                gaps.append(i)
        return cls(first, last, tuple(gaps))

    @classmethod
    def labelled(cls, positions: tuple[int, ...]) -> "SequenceEmbedding":
        """Bind verified increasing, one-based source labels.

        Parameters
        ----------
        positions : tuple[int, ...]
            Verified labels in observed order within one entity instance.

        Returns
        -------
        SequenceEmbedding
            Fixed positions and every internal coordinate gap.
        """
        return cls(
            positions,
            positions,
            tuple(i for i, (a, b) in enumerate(pairwise(positions)) if b > a + 1),
        )


def _greedy_positions(
    observed: Iterable[str], positions: Iterator[tuple[int, str]]
) -> tuple[int, ...] | None:
    matched: list[int] = []
    for token in observed:
        for index, candidate in positions:
            if candidate == token:
                matched.append(index)
                break
        else:
            return None
    return tuple(matched)

"""Packing-alphabet models for backend-specific sequence encoding."""

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from types import MappingProxyType

from protrepair.errors import PackingError
from protrepair.structure.constitution import ResidueSite


@dataclass(frozen=True, slots=True)
class PackingAlphabet:
    """Mapping between component identifiers and one-letter packing tokens."""

    token_by_component_id: Mapping[str, str]

    def __post_init__(self) -> None:
        normalized_mapping: dict[str, str] = {}
        for component_id, token in self.token_by_component_id.items():
            normalized_component_id = component_id.strip().upper()
            normalized_token = token.strip().upper()
            if not normalized_component_id:
                raise ValueError("packing alphabet component ids must not be blank")

            if len(normalized_token) != 1 or not normalized_token.isalpha():
                raise ValueError(
                    "packing alphabet tokens must be one alphabetic character"
                )

            normalized_mapping[normalized_component_id] = normalized_token

        object.__setattr__(
            self,
            "token_by_component_id",
            MappingProxyType(normalized_mapping),
        )

    def supports_component(self, component_id: str) -> bool:
        """Return whether the alphabet supports a component identifier."""

        return component_id.strip().upper() in self.token_by_component_id

    def require_token(self, component_id: str) -> str:
        """Return the token for one component identifier or raise."""

        normalized_component_id = component_id.strip().upper()
        token = self.token_by_component_id.get(normalized_component_id)
        if token is None:
            raise PackingError(
                f"packing alphabet does not support component {normalized_component_id}"
            )

        return token

    def sequence_for_residues(
        self,
        residues: Iterable[ResidueSite],
    ) -> tuple[str, ...]:
        """Return one tokenized sequence for residues in order."""

        return tuple(self.require_token(residue.component_id) for residue in residues)

    def components_for_sequence(self, sequence: Iterable[str]) -> tuple[str, ...]:
        """Resolve packing tokens to unambiguous component identifiers.

        Parameters
        ----------
        sequence : Iterable[str]
            Tokens in sequence order. Lowercase fixed-residue tokens are accepted.

        Returns
        -------
        tuple[str, ...]
            Component identifiers in the same order.

        Raises
        ------
        PackingError
            A token is unsupported or names more than one component.
        """

        components_by_token: dict[str, list[str]] = {}
        for component_id, token in self.token_by_component_id.items():
            components_by_token.setdefault(token, []).append(component_id)

        components = []
        for token in sequence:
            normalized_token = token.upper()
            candidates = components_by_token.get(normalized_token, ())
            if not candidates:
                raise PackingError(
                    f"packing alphabet does not support token {normalized_token!r}"
                )
            if len(candidates) != 1:
                raise PackingError(
                    f"packing alphabet token {normalized_token!r} is ambiguous"
                )
            components.append(candidates[0])

        return tuple(components)

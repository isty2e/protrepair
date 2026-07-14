"""Public construction contracts for canonical residue and atom labels."""

from typing import cast

import pytest

from protrepair.scope import (
    AbsentResidueSpanScope,
    AnchorAtomPairScope,
    AtomSetScope,
    ChainSetScope,
    CompositeScope,
    ResidueBoundaryScope,
    ResidueBoundarySide,
    ResidueSetScope,
    WholeStructureScope,
)
from protrepair.sources.chemistry import RetainedNonPolymerChemistryOverride
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.transformer.local import LocalScopeSpec


def test_residue_id_normalizes_valid_string_fields() -> None:
    """ResidueId keeps the existing canonical whitespace normalization."""

    residue_id = ResidueId(chain_id=" A ", seq_num=7, insertion_code=" ")

    assert residue_id.chain_id == "A"
    assert residue_id.seq_num == 7
    assert residue_id.insertion_code is None


def test_residue_id_ordering_handles_blank_and_letter_insertion_codes() -> None:
    """ResidueId ordering should be total over canonical insertion codes."""

    residue_ids = (
        ResidueId(chain_id="A", seq_num=100, insertion_code="A"),
        ResidueId(chain_id="A", seq_num=101),
        ResidueId(chain_id="A", seq_num=100),
    )

    assert tuple(sorted(residue_ids)) == (
        ResidueId(chain_id="A", seq_num=100),
        ResidueId(chain_id="A", seq_num=100, insertion_code="A"),
        ResidueId(chain_id="A", seq_num=101),
    )


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        pytest.param(ResidueId("A", 100), ResidueId("A", 100, "A"), True),
        pytest.param(ResidueId("A", 100, "A"), ResidueId("A", 100, "B"), True),
        pytest.param(ResidueId("A", 100, "A"), ResidueId("A", 100, "Z"), False),
        pytest.param(ResidueId("A", 100, "A"), ResidueId("A", 101), True),
        pytest.param(ResidueId("A", 100), ResidueId("A", 101, "A"), False),
        pytest.param(ResidueId("A", 100), ResidueId("B", 101), False),
    ],
)
def test_residue_id_immediate_precedence_contract(
    left: ResidueId,
    right: ResidueId,
    expected: bool,
) -> None:
    """Residue numbering adjacency should be explicit and insertion-code aware."""

    assert left.immediately_precedes(right) is expected


@pytest.mark.parametrize(
    "chain_id",
    [
        cast(str, 1),
        cast(str, None),
        cast(str, {"chain_id": "A"}),
    ],
)
def test_residue_id_rejects_non_string_chain_ids(chain_id: str) -> None:
    """JSON-like chain payloads must not enter canonical residue identity."""

    with pytest.raises(TypeError, match="chain_id must be a string"):
        ResidueId(chain_id=chain_id, seq_num=7)


@pytest.mark.parametrize(
    "seq_num",
    [
        cast(int, "7"),
        cast(int, 7.0),
        cast(int, True),
    ],
)
def test_residue_id_rejects_non_integer_sequence_numbers(seq_num: int) -> None:
    """Residue sequence numbers are integers, not bools or string tokens."""

    with pytest.raises(TypeError, match="seq_num must be an integer"):
        ResidueId(chain_id="A", seq_num=seq_num)


@pytest.mark.parametrize(
    "insertion_code",
    [
        cast(str | None, 1),
        cast(str | None, False),
        cast(str | None, ["A"]),
    ],
)
def test_residue_id_rejects_non_string_insertion_codes(
    insertion_code: str | None,
) -> None:
    """Insertion code is an optional string axis, not an arbitrary payload."""

    with pytest.raises(TypeError, match="insertion_code must be a string or None"):
        ResidueId(chain_id="A", seq_num=7, insertion_code=insertion_code)


def test_atom_ref_normalizes_valid_atom_name() -> None:
    """AtomRef keeps the existing canonical atom-name normalization."""

    atom_ref = AtomRef(ResidueId(chain_id="A", seq_num=7), " ca ")

    assert atom_ref.atom_name == "CA"


def test_atom_ref_rejects_json_like_residue_payload() -> None:
    """AtomRef must point at canonical ResidueId, not a nested dict."""

    with pytest.raises(TypeError, match="residue_id must be a ResidueId"):
        AtomRef(
            residue_id=cast(ResidueId, {"chain_id": "A", "seq_num": 7}),
            atom_name="CA",
        )


@pytest.mark.parametrize(
    "atom_name",
    [
        cast(str, 1),
        cast(str, None),
        cast(str, {"atom_name": "CA"}),
    ],
)
def test_atom_ref_rejects_non_string_atom_names(atom_name: str) -> None:
    """Atom names are canonical strings, not raw external objects."""

    with pytest.raises(TypeError, match="atom_name must be a string"):
        AtomRef(ResidueId(chain_id="A", seq_num=7), atom_name)


def test_public_scopes_reject_json_like_residue_payloads() -> None:
    """Public scopes should fail at construction, not later display/lowering."""

    malformed_residue = cast(ResidueId, {"chain_id": "A", "seq_num": 7})

    with pytest.raises(TypeError, match="residue_ids must contain ResidueId values"):
        ResidueSetScope(residue_ids=(malformed_residue,))

    with pytest.raises(TypeError, match="residue_id must be a ResidueId"):
        ResidueBoundaryScope(
            residue_id=malformed_residue,
            side=ResidueBoundarySide.N_TERMINUS,
        )

    with pytest.raises(
        TypeError,
        match="preceding_residue_id must be a ResidueId or None",
    ):
        AbsentResidueSpanScope(preceding_residue_id=malformed_residue)

    with pytest.raises(TypeError, match="residue_ids must contain ResidueId values"):
        LocalScopeSpec.from_residues((malformed_residue,))


def test_public_scopes_reject_json_like_atom_payloads() -> None:
    """Atom scopes should not accept dict-shaped atom references."""

    malformed_atom = cast(
        AtomRef,
        {"residue_id": {"chain_id": "A", "seq_num": 7}, "atom_name": "CA"},
    )

    with pytest.raises(TypeError, match="atom_refs must contain AtomRef values"):
        AtomSetScope(atom_refs=(malformed_atom,))

    with pytest.raises(TypeError, match="left_anchor_atom_ref must be an AtomRef"):
        AnchorAtomPairScope(
            left_anchor_atom_ref=malformed_atom,
            right_anchor_atom_ref=AtomRef(ResidueId("A", 8), "N"),
        )

    with pytest.raises(TypeError, match="atom_refs must contain AtomRef values"):
        LocalScopeSpec.from_atoms((malformed_atom,))


def test_public_scope_collection_inputs_accept_common_iterables() -> None:
    """Public scope constructors should accept common collection inputs."""

    residue_a = ResidueId(chain_id="A", seq_num=1)
    residue_b = ResidueId(chain_id="A", seq_num=2)
    atom_a = AtomRef(residue_a, "CA")
    atom_b = AtomRef(residue_b, "N")

    chain_scope = ChainSetScope(chain_ids=[" B ", "A", "A"])
    residue_scope = ResidueSetScope(residue_ids=[residue_a, residue_b, residue_a])
    atom_scope = AtomSetScope(atom_refs=[atom_a, atom_b, atom_a])
    absent_span_scope = AbsentResidueSpanScope(
        preceding_residue_id=residue_a,
        absent_residue_ids=[residue_b, residue_b],
    )
    composite_scope = CompositeScope(scopes=[WholeStructureScope(), residue_scope])

    assert chain_scope.chain_ids == ("B", "A")
    assert residue_scope.residue_ids == (residue_a, residue_b)
    assert atom_scope.atom_refs == (atom_a, atom_b)
    assert absent_span_scope.absent_residue_ids == (residue_b,)
    assert composite_scope.scopes == (WholeStructureScope(), residue_scope)


def test_chain_set_scope_rejects_string_as_collection_input() -> None:
    """A chain-id string is one chain id, not a collection of chain ids."""

    with pytest.raises(TypeError, match="chain_ids"):
        ChainSetScope(chain_ids="AB")


def test_public_scope_collection_inputs_accept_one_shot_generators() -> None:
    """Public scope constructors should canonicalize generator inputs once."""

    residue_a = ResidueId(chain_id="A", seq_num=1)
    residue_b = ResidueId(chain_id="A", seq_num=2)
    atom_a = AtomRef(residue_a, "CA")
    atom_b = AtomRef(residue_b, "N")

    residue_scope = ResidueSetScope(
        residue_ids=(residue_id for residue_id in (residue_a, residue_b, residue_a))
    )
    atom_scope = AtomSetScope(atom_refs=(atom_ref for atom_ref in (atom_a, atom_b)))

    assert residue_scope.residue_ids == (residue_a, residue_b)
    assert atom_scope.atom_refs == (atom_a, atom_b)


def test_retained_non_polymer_override_rejects_json_like_residue_payload() -> None:
    """External chemistry overrides must carry canonical residue identity."""

    with pytest.raises(TypeError, match="residue_id must be a ResidueId"):
        RetainedNonPolymerChemistryOverride(
            residue_id=cast(ResidueId, {"chain_id": "L", "seq_num": 1}),
            smiles="CO",
            heavy_atom_names=("C1", "O1"),
        )

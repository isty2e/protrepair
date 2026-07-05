"""Internal-coordinate atom completion regressions."""

import numpy as np
from tests.support.canonical_builders import atom_payload, chain_payload
from tests.support.canonical_builders import (
    build_structure as build_canonical_structure,
)
from tests.support.canonical_builders import (
    completion_payload as build_completion_payload,
)

from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.geometry import Vec3
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import ResidueIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.completion.atom.backbone import (
    PeptideCarbonylFrame,
    backbone_psi_degrees,
)
from protrepair.transformer.completion.atom.internal_coordinates import (
    InternalCoordinatePlacementTransformer,
)
from protrepair.transformer.completion.shared.domain import (
    CompletionResiduePayload,
    ResidueBackboneNeighborhood,
    ResidueCompletionSite,
)
from protrepair.transformer.context import ProteinTransformationContext

_DEFAULT_O_POSITION = Vec3(0.0, 1.0, 0.0)
_DEFAULT_CB_POSITION = Vec3(1.8, -0.75, 1.25)


def test_backbone_oxygen_uses_current_residue_psi_not_previous_context() -> None:
    """A current-residue carbonyl must not inherit the previous residue psi."""

    previous_residue = _ala_payload(
        seq_num=1,
        n=Vec3(-3.0, 0.0, 0.0),
        ca=Vec3(-2.0, 1.0, 0.0),
        c=Vec3(-1.0, 0.0, 0.0),
    )
    residue = _ala_payload(
        seq_num=2,
        n=Vec3(0.0, 0.0, 0.0),
        ca=Vec3(1.45, 0.0, 0.0),
        c=Vec3(2.40, 1.20, 0.0),
        o=None,
    )
    next_residue = _ala_payload(
        seq_num=3,
        n=Vec3(1.5, 0.0, 1.0),
        ca=Vec3(2.6, 0.4, 1.4),
        c=Vec3(3.4, 1.4, 1.2),
    )
    snapshot = ProteinStructureSnapshot.from_structure(
        _structure_from_residues((previous_residue, residue, next_residue))
    )
    site = _completion_site(
        snapshot,
        residue_index=ResidueIndex(1),
        original_payload=residue,
    )
    transformer = InternalCoordinatePlacementTransformer(site)
    context = _context_for_site(snapshot, site)

    assert transformer.is_applicable(context) is True

    transformed = transformer.transform(context)
    repaired_residue = site.payload(transformed)
    assert repaired_residue is not None
    assert repaired_residue.has_atom_site("O")

    next_nitrogen = next_residue.position("N")
    current_psi = backbone_psi_degrees(
        (
            residue.position("N"),
            residue.position("CA"),
            residue.position("C"),
            next_nitrogen,
        )
    )
    previous_context_psi = backbone_psi_degrees(
        (
            previous_residue.position("N"),
            previous_residue.position("CA"),
            previous_residue.position("C"),
            residue.position("N"),
        )
    )
    assert 0.0 < current_psi < 100.0
    assert previous_context_psi >= 100.0

    expected_position = _oxygen_position(residue, next_nitrogen, current_psi)
    stale_position = _oxygen_position(residue, next_nitrogen, previous_context_psi)
    actual_position = repaired_residue.position("O")

    np.testing.assert_allclose(
        _vec_array(actual_position),
        _vec_array(expected_position),
        atol=1e-12,
    )
    assert np.linalg.norm(
        _vec_array(actual_position) - _vec_array(stale_position)
    ) > 1.0


def test_n_terminal_backbone_oxygen_does_not_require_previous_residue() -> None:
    """The first residue can place O when the next peptide N is available."""

    residue = _ala_payload(
        seq_num=1,
        n=Vec3(0.0, 0.0, 0.0),
        ca=Vec3(1.45, 0.0, 0.0),
        c=Vec3(2.40, 1.20, 0.0),
        o=None,
    )
    next_residue = _ala_payload(
        seq_num=2,
        n=Vec3(1.5, 0.0, 1.0),
        ca=Vec3(2.6, 0.4, 1.4),
        c=Vec3(3.4, 1.4, 1.2),
    )
    snapshot = ProteinStructureSnapshot.from_structure(
        _structure_from_residues((residue, next_residue))
    )
    site = _completion_site(
        snapshot,
        residue_index=ResidueIndex(0),
        original_payload=residue,
    )
    transformer = InternalCoordinatePlacementTransformer(site)
    context = _context_for_site(snapshot, site)

    assert transformer.is_applicable(context) is True

    repaired_residue = site.payload(transformer.transform(context))
    assert repaired_residue is not None
    assert repaired_residue.has_atom_site("O")


def test_c_terminal_missing_backbone_oxygen_uses_local_fallback_not_wrap() -> None:
    """The last residue must place O without using the first residue as next N."""

    first_residue = _ala_payload(
        seq_num=1,
        n=Vec3(-3.0, 0.0, 0.0),
        ca=Vec3(-2.0, 1.0, 0.0),
        c=Vec3(-1.0, 0.0, 0.0),
    )
    terminal_residue = _ala_payload(
        seq_num=2,
        n=Vec3(0.0, 0.0, 0.0),
        ca=Vec3(1.45, 0.0, 0.0),
        c=Vec3(2.40, 1.20, 0.0),
        o=None,
    )
    snapshot = ProteinStructureSnapshot.from_structure(
        _structure_from_residues((first_residue, terminal_residue))
    )
    site = _completion_site(
        snapshot,
        residue_index=ResidueIndex(1),
        original_payload=terminal_residue,
    )
    transformer = InternalCoordinatePlacementTransformer(site)
    context = _context_for_site(snapshot, site)

    assert transformer.is_applicable(context) is True

    transformed_residue = site.payload(transformer.transform(context))
    assert transformed_residue is not None
    assert transformed_residue.has_atom_site("O")

    fallback_position = _oxygen_position(
        terminal_residue,
        terminal_residue.position("N"),
        0.0,
    )
    wrapped_context_psi = backbone_psi_degrees(
        (
            first_residue.position("N"),
            first_residue.position("CA"),
            first_residue.position("C"),
            terminal_residue.position("N"),
        )
    )
    wrapped_position = _oxygen_position(
        terminal_residue,
        first_residue.position("N"),
        wrapped_context_psi,
    )
    actual_position = transformed_residue.position("O")
    np.testing.assert_allclose(
        _vec_array(actual_position),
        _vec_array(fallback_position),
        atol=1e-12,
    )
    assert np.linalg.norm(
        _vec_array(actual_position) - _vec_array(wrapped_position)
    ) > 0.5


def test_backbone_oxygen_gap_uses_local_fallback_not_next_residue() -> None:
    """Slot-adjacent residues with a sequence gap are not peptide neighbors."""

    residue = _ala_payload(
        seq_num=1,
        n=Vec3(0.0, 0.0, 0.0),
        ca=Vec3(1.45, 0.0, 0.0),
        c=Vec3(2.40, 1.20, 0.0),
        o=None,
    )
    gapped_next_residue = _ala_payload(
        seq_num=3,
        n=Vec3(1.5, 0.0, 1.0),
        ca=Vec3(2.6, 0.4, 1.4),
        c=Vec3(3.4, 1.4, 1.2),
    )
    snapshot = ProteinStructureSnapshot.from_structure(
        _structure_from_residues((residue, gapped_next_residue))
    )
    site = _completion_site(
        snapshot,
        residue_index=ResidueIndex(0),
        original_payload=residue,
    )
    transformer = InternalCoordinatePlacementTransformer(site)
    context = _context_for_site(snapshot, site)

    assert transformer.is_applicable(context) is True

    transformed_residue = site.payload(transformer.transform(context))
    assert transformed_residue is not None
    assert transformed_residue.has_atom_site("O")

    fallback_position = _oxygen_position(residue, residue.position("N"), 0.0)
    gapped_next_psi = backbone_psi_degrees(
        (
            residue.position("N"),
            residue.position("CA"),
            residue.position("C"),
            gapped_next_residue.position("N"),
        )
    )
    gapped_next_position = _oxygen_position(
        residue,
        gapped_next_residue.position("N"),
        gapped_next_psi,
    )
    actual_position = transformed_residue.position("O")
    np.testing.assert_allclose(
        _vec_array(actual_position),
        _vec_array(fallback_position),
        atol=1e-12,
    )
    assert np.linalg.norm(
        _vec_array(actual_position) - _vec_array(gapped_next_position)
    ) > 0.5


def test_backbone_oxygen_missing_next_nitrogen_uses_local_fallback() -> None:
    """A next residue without N cannot provide psi context."""

    residue = _ala_payload(
        seq_num=1,
        n=Vec3(0.0, 0.0, 0.0),
        ca=Vec3(1.45, 0.0, 0.0),
        c=Vec3(2.40, 1.20, 0.0),
        o=None,
    )
    next_residue_without_n = _ala_payload(
        seq_num=2,
        n=None,
        ca=Vec3(2.6, 0.4, 1.4),
        c=Vec3(3.4, 1.4, 1.2),
    )
    snapshot = ProteinStructureSnapshot.from_structure(
        _structure_from_residues((residue, next_residue_without_n))
    )
    site = _completion_site(
        snapshot,
        residue_index=ResidueIndex(0),
        original_payload=residue,
    )
    transformer = InternalCoordinatePlacementTransformer(site)
    context = _context_for_site(snapshot, site)

    assert transformer.is_applicable(context) is True

    transformed_residue = site.payload(transformer.transform(context))
    assert transformed_residue is not None
    assert transformed_residue.has_atom_site("O")
    np.testing.assert_allclose(
        _vec_array(transformed_residue.position("O")),
        _vec_array(_oxygen_position(residue, residue.position("N"), 0.0)),
        atol=1e-12,
    )


def test_terminal_sidechain_completion_does_not_need_next_peptide_context() -> None:
    """Only missing O needs next peptide context; existing O keeps CB repair local."""

    first_residue = _ala_payload(
        seq_num=1,
        n=Vec3(-3.0, 0.0, 0.0),
        ca=Vec3(-2.0, 1.0, 0.0),
        c=Vec3(-1.0, 0.0, 0.0),
    )
    terminal_residue = _ala_payload(
        seq_num=2,
        n=Vec3(0.0, 0.0, 0.0),
        ca=Vec3(1.45, 0.0, 0.0),
        c=Vec3(2.40, 1.20, 0.0),
        cb=None,
    )
    snapshot = ProteinStructureSnapshot.from_structure(
        _structure_from_residues((first_residue, terminal_residue))
    )
    site = _completion_site(
        snapshot,
        residue_index=ResidueIndex(1),
        original_payload=terminal_residue,
    )
    transformer = InternalCoordinatePlacementTransformer(site)
    context = _context_for_site(snapshot, site)

    assert transformer.is_applicable(context) is True

    transformed_residue = site.payload(transformer.transform(context))
    assert transformed_residue is not None
    assert transformed_residue.has_atom_site("O")
    assert transformed_residue.has_atom_site("CB")


def test_backbone_oxygen_insertion_code_adjacency_uses_local_fallback() -> None:
    """Insertion-code adjacency is not silently treated as peptide-next context."""

    residue = _ala_payload(
        seq_num=1,
        n=Vec3(0.0, 0.0, 0.0),
        ca=Vec3(1.45, 0.0, 0.0),
        c=Vec3(2.40, 1.20, 0.0),
        o=None,
    )
    insertion_residue = _ala_payload(
        seq_num=2,
        insertion_code="A",
        n=Vec3(1.5, 0.0, 1.0),
        ca=Vec3(2.6, 0.4, 1.4),
        c=Vec3(3.4, 1.4, 1.2),
    )
    snapshot = ProteinStructureSnapshot.from_structure(
        _structure_from_residues((residue, insertion_residue))
    )
    site = _completion_site(
        snapshot,
        residue_index=ResidueIndex(0),
        original_payload=residue,
    )
    transformer = InternalCoordinatePlacementTransformer(site)
    context = _context_for_site(snapshot, site)

    assert transformer.is_applicable(context) is True

    transformed_residue = site.payload(transformer.transform(context))
    assert transformed_residue is not None
    assert transformed_residue.has_atom_site("O")
    np.testing.assert_allclose(
        _vec_array(transformed_residue.position("O")),
        _vec_array(_oxygen_position(residue, residue.position("N"), 0.0)),
        atol=1e-12,
    )


def _ala_payload(
    *,
    seq_num: int,
    n: Vec3 | None,
    ca: Vec3,
    c: Vec3,
    insertion_code: str | None = None,
    o: Vec3 | None = _DEFAULT_O_POSITION,
    cb: Vec3 | None = _DEFAULT_CB_POSITION,
) -> CompletionResiduePayload:
    atoms = []
    if n is not None:
        atoms.append(atom_payload("N", "N", n))
    atoms.append(atom_payload("CA", "C", ca))
    atoms.append(atom_payload("C", "C", c))
    if o is not None:
        atoms.append(atom_payload("O", "O", o))
    if cb is not None:
        atoms.append(atom_payload("CB", "C", cb))

    return build_completion_payload(
        component_id="ALA",
        residue_id=ResidueId(
            chain_id="A",
            seq_num=seq_num,
            insertion_code=insertion_code,
        ),
        atoms=tuple(atoms),
    )


def _structure_from_residues(
    residues: tuple[CompletionResiduePayload, ...],
) -> ProteinStructure:
    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                tuple(
                    (
                        residue.residue_site,
                        residue.residue_geometry,
                        residue.formal_charge_by_atom_name,
                    )
                    for residue in residues
                ),
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="internal-coordinate-completion-test",
    )


def _completion_site(
    snapshot: ProteinStructureSnapshot,
    *,
    residue_index: ResidueIndex,
    original_payload: CompletionResiduePayload,
) -> ResidueCompletionSite:
    return ResidueCompletionSite(
        residue_index=residue_index,
        template=build_standard_component_library().require("ALA"),
        original_payload=original_payload,
        neighborhood=ResidueBackboneNeighborhood.from_linear_residue_slots(
            residue_index,
            residue_count=len(snapshot.structure.constitution.residue_slots),
        ),
    )


def _context_for_site(
    snapshot: ProteinStructureSnapshot,
    site: ResidueCompletionSite,
) -> ProteinTransformationContext:
    return ProteinTransformationContext.from_snapshot_atom_input(
        snapshot,
        site.atom_input(snapshot),
    )


def _oxygen_position(
    residue: CompletionResiduePayload,
    next_nitrogen: Vec3,
    psi_degrees: float,
) -> Vec3:
    return PeptideCarbonylFrame(
        nitrogen=residue.position("N"),
        alpha_carbon=residue.position("CA"),
        carbonyl_carbon=residue.position("C"),
    ).backbone_oxygen(
        psi_degrees=psi_degrees,
        clash_reference=next_nitrogen,
    )


def _vec_array(vector: Vec3) -> np.ndarray:
    return np.asarray(tuple(vector), dtype=np.float64)

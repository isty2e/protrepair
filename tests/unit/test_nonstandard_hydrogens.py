"""Nonstandard hydrogenation tests over bundled ideal-geometry assets."""

import numpy as np
import pytest
from tests.support.canonical_builders import (
    CanonicalResiduePayload,
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry.nonstandard.registry import (
    NonstandardComponentRecord,
    build_bundled_nonstandard_registry,
)
from protrepair.geometry import Vec3
from protrepair.structure.labels import (
    ResidueId,
)
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.artifacts import RegionTransformationResult, StructureDelta
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.continuous.binding_policy import (
    ManualContinuousRelaxationBinding,
)
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationConfig,
    ContinuousRelaxationForceField,
)
from protrepair.transformer.refinement.directive import RepairLocalRefinementDirective

ROTATION = np.asarray(
    (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    ),
    dtype=np.float64,
)
TRANSLATION = np.asarray((4.5, -2.0, 1.75), dtype=np.float64)


def test_add_hydrogens_hydrogenates_hyp_with_proline_like_backbone_rules() -> None:
    """HYP should use template-driven sidechain H placement and PRO-like N-terminus."""

    residue = build_nonstandard_residue("HYP", missing_atom_names=("OD1",))
    structure = build_structure(
        chains=(chain_payload("A", (residue,)),),
        source_format=FileFormat.PDB,
        source_name="hyp-nonstandard-hydrogens",
    )

    result = add_hydrogens(structure)
    hydrogenated = result.structure.constitution.chain("A").residues[0]
    atom_names = set(hydrogenated.atom_site_names())

    assert {"HA", "HB2", "HB3", "HG", "HD22", "HD23", "HD1"} <= atom_names
    assert {"H1", "H2"} <= atom_names
    assert "H3" not in atom_names
    assert "H" not in atom_names


def test_add_hydrogens_unlocks_sep_hydrogenation_after_refinement_gated_heavy_repair(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SEP should hydrogenate once local refinement unlocks the heavy repair path."""

    residue = build_nonstandard_residue(
        "SEP",
        missing_atom_names=("P", "O1P", "O2P", "O3P"),
    )
    structure = build_structure(
        chains=(chain_payload("A", (residue,)),),
        source_format=FileFormat.PDB,
        source_name="sep-nonstandard-hydrogens",
    )
    settings = ContinuousRelaxationConfig().bind(ContinuousRelaxationForceField.UFF)
    local_refinement = RepairLocalRefinementDirective.from_residue_ids(
        (residue[0].residue_id,),
        config=ContinuousRelaxationConfig(),
        binding=ManualContinuousRelaxationBinding(ContinuousRelaxationForceField.UFF),
    )
    execute_call_count = 0

    def fake_execute_local_refinement(
        context: ProteinTransformationContext,
        *,
        spec,
        component_library=None,
    ):
        nonlocal execute_call_count
        execute_call_count += 1
        snapshot = context.source_snapshot
        atom_input = context.atom_input
        repaired_residue = snapshot.structure.constitution.chain("A").residues[0]
        assert repaired_residue.has_atom_site("P")
        assert repaired_residue.has_atom_site("HOP2")
        assert repaired_residue.has_atom_site("HOP3")
        assert atom_input.referenced_residue_ids() == (residue[0].residue_id,)
        assert spec == settings
        assert component_library is not None
        return RegionTransformationResult(
            refined_structure=snapshot.structure,
            delta=StructureDelta(
                before_constitution=snapshot.structure.constitution,
                after_constitution=snapshot.structure.constitution,
            ),
            issues=(),
            backend_name="rdkit",
        )

    monkeypatch.setattr(
        "protrepair.transformer.refinement.repair_stage.execute_local_transformation",
        fake_execute_local_refinement,
    )

    result = add_hydrogens(structure, local_refinement=local_refinement)
    hydrogenated = result.structure.constitution.chain("A").residues[0]

    assert execute_call_count == 1
    assert {"P", "O1P", "O2P", "O3P", "HOP2", "HOP3"} <= set(
        hydrogenated.atom_site_names()
    )


def build_nonstandard_residue(
    component_id: str,
    *,
    missing_atom_names: tuple[str, ...] = (),
    seq_num: int = 1,
) -> CanonicalResiduePayload:
    """Return one transformed bundled nonstandard residue with heavy atoms only."""

    record = require_record(component_id)
    normalized_missing_atom_names = frozenset(
        atom_name.upper() for atom_name in missing_atom_names
    )
    atoms = []
    for template_atom in record.heavy_atoms():
        if template_atom.atom_name in normalized_missing_atom_names:
            continue

        atoms.append(
            atom_payload(
                name=template_atom.atom_name,
                element=template_atom.element,
                position=transformed_ideal_position(record, template_atom.atom_name),
                formal_charge=(
                    None
                    if template_atom.formal_charge == 0
                    else template_atom.formal_charge
                ),
            )
        )

    return residue_payload(
        component_id=component_id,
        residue_id=ResidueId(chain_id="A", seq_num=seq_num),
        atoms=tuple(atoms),
    )


def transformed_ideal_position(
    record: NonstandardComponentRecord,
    atom_name: str,
) -> Vec3:
    """Return one rigidly transformed ideal coordinate for a bundled atom."""

    ideal_position = atom_ideal_position(record, atom_name)
    transformed_point = np.asarray(ideal_position, dtype=np.float64) @ ROTATION
    transformed_point = transformed_point + TRANSLATION
    return Vec3.from_iterable(transformed_point)


def atom_ideal_position(
    record: NonstandardComponentRecord,
    atom_name: str,
) -> tuple[float, float, float]:
    """Return one bundled ideal coordinate by atom name."""

    normalized_atom_name = atom_name.strip().upper()
    for atom in record.atoms:
        if atom.atom_name != normalized_atom_name:
            continue

        assert atom.ideal_position is not None
        return atom.ideal_position

    raise AssertionError(
        f"missing ideal position for {record.component_id} {atom_name}"
    )


def require_record(component_id: str) -> NonstandardComponentRecord:
    """Return one bundled nonstandard record required by the tests."""

    record = build_bundled_nonstandard_registry().get(component_id)
    assert record is not None
    return record

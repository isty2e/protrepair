"""Lysine fallback uses aliphatic terminal bond geometry."""

import numpy as np
import pytest
from tests.support.canonical_builders import atom_payload, completion_payload

from protrepair.chemistry.standard.heavy_atom_programs import (
    STANDARD_HEAVY_ATOM_SEMANTICS,
)
from protrepair.geometry import Vec3
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.atom.internal_coordinates import (
    InternalCoordinateEnvironment,
)


def test_literal_lysine_recipe_has_aliphatic_ce_nz_geometry() -> None:
    residue = completion_payload(
        component_id="LYS",
        residue_id=ResidueId("A", 1),
        atoms=tuple(
            atom_payload(name, element, position)
            for name, element, position in (
                ("N", "N", Vec3(0.0, 0.0, 0.0)),
                ("CA", "C", Vec3(1.45, 0.0, 0.0)),
                ("C", "C", Vec3(2.40, 1.20, 0.0)),
                ("O", "O", Vec3(2.40, 2.40, 0.0)),
                ("CB", "C", Vec3(1.80, -0.75, 1.25)),
            )
        ),
    )
    context = InternalCoordinateEnvironment.from_payloads(
        residue=residue,
        next_peptide_residue=None,
        missing_atom_names=("CG", "CD", "CE", "NZ"),
    )
    STANDARD_HEAVY_ATOM_SEMANTICS["LYS"].program.apply(context)
    cd, ce, nz = (
        context.geometry.position(name).to_array() for name in ("CD", "CE", "NZ")
    )

    assert np.linalg.norm(cd - ce) == pytest.approx(1.520, abs=1e-12)
    assert np.linalg.norm(ce - nz) == pytest.approx(1.489, abs=1e-12)
    angle = np.degrees(
        np.arccos(
            np.dot(cd - ce, nz - ce)
            / (np.linalg.norm(cd - ce) * np.linalg.norm(nz - ce))
        )
    )
    assert angle == pytest.approx(111.900, abs=1e-10)
    assert context.torsion(("CG", "CD", "CE", "NZ")) == pytest.approx(179.6)

"""Native energy and gradients at the calibrated soft-window boundaries."""

from math import cos, radians, sin

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdForceFieldHelpers

from protrepair.chemistry.restraint.template import (
    AngleRestraintTarget,
    BondRestraintTarget,
)
from protrepair.diagnostics.geometry import (
    HeavyGeometryPolicy,
    bond_length_tolerance_angstrom,
)
from protrepair.transformer.continuous import rdkit as backend
from protrepair.transformer.continuous.settings import ContinuousRelaxationForceField


@pytest.mark.parametrize("family", tuple(ContinuousRelaxationForceField))
@pytest.mark.parametrize("esd", (None, 0.005, 0.01, 0.03))
@pytest.mark.parametrize("side", (-1, 1))
def test_native_bond_restraint_penalizes_deviation_beyond_default_window(
    family: ContinuousRelaxationForceField, esd: float | None, side: int
) -> None:
    target = BondRestraintTarget("C1", "C2", 1.5, esd)
    diagnostic_width = bond_length_tolerance_angstrom(
        target=target, policy=HeavyGeometryPolicy()
    )
    width = backend.bond_constraint_tolerance_angstrom(target)
    molecule = Chem.MolFromSmiles("CC")
    conformer = Chem.Conformer(2)
    conformer.SetAtomPosition(0, (0.0, 0.0, 0.0))
    conformer.SetAtomPosition(1, (1.5, 0.0, 0.0))
    molecule.AddConformer(conformer)
    force_field = rdForceFieldHelpers.CreateEmptyForceFieldForMol(molecule)
    backend.add_rdkit_distance_constraint(
        force_field,
        force_field_family=family,
        atom_index_1=0,
        atom_index_2=1,
        minimum_distance=1.5 - width,
        maximum_distance=1.5 + width,
        force_constant=80.0,
    )
    force_field.Initialize()

    for deviation in (0.0, diagnostic_width / 2, diagnostic_width):
        position = (0.0, 0.0, 0.0, 1.5 + side * deviation, 0.0, 0.0)
        assert force_field.CalcEnergy(position) == pytest.approx(0.0, abs=1e-12)

    position = (0.0, 0.0, 0.0, 1.5 + side * (diagnostic_width + 0.005), 0.0, 0.0)
    assert force_field.CalcEnergy(position) > 0.0
    gradient = np.asarray(force_field.CalcGrad(position))
    assert gradient[3] * side > 0.0


@pytest.mark.parametrize("family", tuple(ContinuousRelaxationForceField))
@pytest.mark.parametrize(
    "esd, expected_width", ((None, 5.0), (0.5, 5.0), (1.25, 5.0), (2.0, 8.0))
)
@pytest.mark.parametrize("side", (-1, 1))
def test_native_angle_restraint_keeps_its_independently_calibrated_window(
    family: ContinuousRelaxationForceField,
    esd: float | None,
    expected_width: float,
    side: int,
) -> None:
    target = AngleRestraintTarget("C1", "C2", "C3", 120.0, esd)
    width = backend.angle_constraint_tolerance_degrees(target)
    molecule = Chem.MolFromSmiles("CCC")
    conformer = Chem.Conformer(3)
    conformer.SetAtomPosition(0, (1.5, 0.0, 0.0))
    conformer.SetAtomPosition(1, (0.0, 0.0, 0.0))
    conformer.SetAtomPosition(2, (-0.75, 1.3, 0.0))
    molecule.AddConformer(conformer)
    force_field = rdForceFieldHelpers.CreateEmptyForceFieldForMol(molecule)
    backend.add_rdkit_angle_constraint(
        force_field,
        force_field_family=family,
        atom_index_1=0,
        center_atom_index=1,
        atom_index_2=2,
        minimum_angle_degrees=120.0 - width,
        maximum_angle_degrees=120.0 + width,
        force_constant=30.0,
    )
    force_field.Initialize()

    for deviation in (0.0, expected_width / 2, expected_width):
        angle = radians(120.0 + side * deviation)
        position = (
            1.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1.5 * cos(angle),
            1.5 * sin(angle),
            0.0,
        )
        assert force_field.CalcEnergy(position) == pytest.approx(0.0, abs=1e-12)

    angle = radians(120.0 + side * (expected_width + 0.5))
    position = (1.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.5 * cos(angle), 1.5 * sin(angle), 0.0)
    assert force_field.CalcEnergy(position) > 0.0
    gradient = np.asarray(force_field.CalcGrad(position)).reshape(3, 3)
    tangent = np.asarray((-sin(angle), cos(angle), 0.0))
    assert float(gradient[2] @ tangent) * side > 0.0

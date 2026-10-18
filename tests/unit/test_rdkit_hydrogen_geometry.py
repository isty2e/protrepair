from dataclasses import replace

import numpy as np
import pytest
from rdkit.Chem import rdForceFieldHelpers
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    chain_payload,
    residue_payload,
)
from tests.support.correction_state_fixtures import (
    residue_bond_specs,
    with_topology_bonds,
)
from tests.support.refinement_contract import build_continuous_relaxation_problem

from protrepair.chemistry import (
    BondDefinition,
    ChemicalComponentDefinition,
    ComponentLibrary,
    HeavyAtomSemantics,
    ResidueTemplate,
    RestraintLibrary,
)
from protrepair.chemistry.internal_coordinates import InternalCoordinateProgram
from protrepair.errors import RefinementError
from protrepair.geometry import Vec3
from protrepair.io import FileFormat
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.continuous import rdkit as native
from protrepair.transformer.continuous.settings import ContinuousRelaxationForceField
from protrepair.transformer.continuous.shared import apply_position_updates
from protrepair.transformer.local import LocalScopeSpec


def _problem(element, isotope, moving, method):
    rid = ResidueId("A", 1)
    directions = np.array([(1, 0, 0), (-1, 1, 1), (-1, -1, 1), (-1, 0, -1)])
    directions = directions / np.linalg.norm(directions, axis=1)[:, None]
    count = {"C": 4, "N": 3, "O": 2, "S": 2}[element]
    length = {"C": 1.09, "N": 1.01, "O": 0.98, "S": 1.34}[element]
    names = tuple(f"H{i + 1}" for i in range(count))
    source = build_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="TST",
                        residue_id=rid,
                        atoms=(
                            atom_payload("X", element, Vec3(0, 0, 0)),
                            *(
                                atom_payload(
                                    name, isotope, Vec3.from_iterable(d * length)
                                )
                                for name, d in zip(names, directions, strict=False)
                            ),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    source = with_topology_bonds(
        source, *residue_bond_specs(rid, tuple(("X", name) for name in names))
    )
    library = ComponentLibrary(
        templates={
            "TST": ResidueTemplate(
                definition=ChemicalComponentDefinition(
                    component_id="TST",
                    atom_names=("X", *names),
                    bonds=tuple(BondDefinition("X", name) for name in names),
                ),
                heavy_atom_semantics=HeavyAtomSemantics(
                    program=InternalCoordinateProgram.backbone_only(),
                    atom_order=("X",),
                ),
            )
        }
    )
    selected = {"hydrogen": ("H1",), "parent": ("X",), "both": ("X", "H1")}[moving]
    return build_continuous_relaxation_problem(
        source,
        LocalScopeSpec.from_atoms(tuple(AtomRef(rid, name) for name in selected)),
        component_library=library,
        force_field=method,
        context_radius_angstrom=0,
        max_iterations=7,
    )


def _inject_endpoint(monkeypatch, problem, *, endpoint_distance):
    def field(molecule, **kwargs):
        mapping = kwargs["rdkit_atom_index_by_structure_atom_index"]
        source = problem.region.snapshot.structure
        parent = source.constitution.atom_index(AtomRef(ResidueId("A", 1), "X"))
        hydrogen = source.constitution.atom_index(AtomRef(ResidueId("A", 1), "H1"))
        initial = molecule.GetConformer().GetPositions().copy()
        a, h = mapping[parent], mapping[hydrogen]

        def positions(length):
            result = initial.copy()
            if hydrogen in problem.region.movable_atom_indices:
                result[h] = result[a] + (length, 0, 0)
            else:
                result[a] = result[h] - (length, 0, 0)
            return result

        class Frames:
            def Initialize(self):
                pass

            def Minimize(self, *, maxIts):
                assert maxIts == 7
                for i, point in enumerate(positions(endpoint_distance)):
                    molecule.GetConformer().SetAtomPosition(i, tuple(point))
                return 1

        return Frames()

    monkeypatch.setattr(native, "build_rdkit_force_field", field)


@pytest.mark.parametrize("element", ["C", "N", "O", "S"])
@pytest.mark.parametrize("isotope", ["H", "D", "T"])
@pytest.mark.parametrize("moving", ["hydrogen", "parent", "both"])
@pytest.mark.parametrize("method", list(ContinuousRelaxationForceField))
def test_native_guard_checks_canonical_xh_pairs_without_template_h_targets(
    monkeypatch, element, isotope, moving, method
):
    problem = _problem(element, isotope, moving, method)
    source = problem.region.snapshot.structure
    _inject_endpoint(monkeypatch, problem, endpoint_distance=0.5)
    result = native.RdkitContinuousRelaxationBackend().relax(
        problem, restraint_library=RestraintLibrary({})
    )
    assert result.refined_structure is source
    assert not result.delta.moved_atoms
    assert len(result.issues) == 1
    assert f"native {method.value.upper()} X-H" in result.issues[0].message
    assert len(result.relaxation_executions) == 1
    assert result.relaxation_executions[0].settings is problem.spec
    assert result.relaxation_executions[0].minimizer_status == 1


@pytest.mark.parametrize("method", list(ContinuousRelaxationForceField))
@pytest.mark.parametrize(
    "initial, final, accepted",
    [
        (0.65, 0.65, True),
        (0.65, 0.8, True),
        (0.65, 0.6, False),
        (1.09, 0.97, True),
        (1.09, 1.6, False),
    ],
)
def test_inherited_distortion_can_stay_or_improve_and_xray_h_is_not_rejected(
    monkeypatch, method, initial, final, accepted
):
    problem = _problem("C", "H", "hydrogen", method)
    source = problem.region.snapshot.structure
    h = source.constitution.atom_index(AtomRef(ResidueId("A", 1), "H1"))
    source = apply_position_updates(source, moved_positions={h: Vec3(initial, 0, 0)})
    problem = replace(
        problem,
        region=replace(
            problem.region, snapshot=ProteinStructureSnapshot.from_structure(source)
        ),
    )
    _inject_endpoint(monkeypatch, problem, endpoint_distance=final)
    result = native.RdkitContinuousRelaxationBackend().relax(
        problem, restraint_library=RestraintLibrary({})
    )
    assert bool(result.issues) is not accepted
    assert result.refined_structure.geometry.position(h).x == pytest.approx(
        final if accepted else initial
    )


@pytest.mark.parametrize(
    "offset, accepted",
    [
        (0.199, True),
        (0.2, True),
        (0.201, False),
        (-0.199, True),
        (-0.2, True),
        (-0.201, False),
    ],
)
def test_coarse_native_h_tolerance_boundary(monkeypatch, offset, accepted):
    problem = _problem("C", "H", "hydrogen", ContinuousRelaxationForceField.UFF)
    molecule, mapping = native.build_rdkit_molecule(problem)
    constitution = problem.region.snapshot.structure.constitution
    a, h = (
        mapping[constitution.atom_index(AtomRef(ResidueId("A", 1), name))]
        for name in ("X", "H1")
    )
    _, target = rdForceFieldHelpers.GetUFFBondStretchParams(molecule, a, h)
    _inject_endpoint(monkeypatch, problem, endpoint_distance=target + offset)
    result = native.RdkitContinuousRelaxationBackend().relax(
        problem, restraint_library=RestraintLibrary({})
    )
    assert bool(result.issues) is not accepted


def test_fixed_inherited_h_distortion_does_not_block_other_atom_repair(monkeypatch):
    problem = _problem("C", "H", "hydrogen", ContinuousRelaxationForceField.UFF)
    source = problem.region.snapshot.structure
    fixed_h = source.constitution.atom_index(AtomRef(ResidueId("A", 1), "H2"))
    source = apply_position_updates(source, moved_positions={fixed_h: Vec3(0, 0.5, 0)})
    problem = replace(
        problem,
        region=replace(
            problem.region, snapshot=ProteinStructureSnapshot.from_structure(source)
        ),
    )
    _inject_endpoint(monkeypatch, problem, endpoint_distance=1.1)
    result = native.RdkitContinuousRelaxationBackend().relax(
        problem, restraint_library=RestraintLibrary({})
    )
    assert not result.issues
    assert result.refined_structure.geometry.position(fixed_h) == Vec3(0, 0.5, 0)


@pytest.mark.parametrize("method", list(ContinuousRelaxationForceField))
@pytest.mark.parametrize("target", [None, float("nan"), float("inf"), 0.0, -1.0])
def test_missing_or_invalid_native_target_is_not_silent_admission(
    monkeypatch, method, target
):
    problem = _problem("C", "H", "hydrogen", method)
    _inject_endpoint(monkeypatch, problem, endpoint_distance=0.683)

    class Parameters:
        def GetMMFFBondStretchParams(self, *args):
            return None if target is None else (0, 1.0, target)

    if method is ContinuousRelaxationForceField.UFF:
        monkeypatch.setattr(
            rdForceFieldHelpers,
            "GetUFFBondStretchParams",
            lambda *args: None if target is None else (1.0, target),
        )
    else:
        monkeypatch.setattr(
            rdForceFieldHelpers,
            "MMFFGetMoleculeProperties",
            lambda *args, **kwargs: Parameters(),
        )
    with pytest.raises(RefinementError, match="hydrogen bond target"):
        native.RdkitContinuousRelaxationBackend().relax(
            problem, restraint_library=RestraintLibrary({})
        )


@pytest.mark.parametrize("element", ["C", "N", "O", "S"])
@pytest.mark.parametrize("method", list(ContinuousRelaxationForceField))
def test_native_minimization_keeps_supported_xh_bonds_usable(element, method):
    problem = _problem(element, "H", "hydrogen", method)
    result = native.RdkitContinuousRelaxationBackend().relax(
        problem, restraint_library=RestraintLibrary({})
    )
    assert not result.issues
    source = problem.region.snapshot.structure
    assert result.refined_structure.topology is source.topology
    for index in problem.region.fixed_context_atom_indices:
        assert result.refined_structure.geometry.position(
            index
        ) == source.geometry.position(index)


@pytest.mark.parametrize("method", list(ContinuousRelaxationForceField))
def test_native_target_lookup_exception_preserves_refinement_error_boundary(
    monkeypatch, method
):
    problem = _problem("C", "H", "hydrogen", method)
    _inject_endpoint(monkeypatch, problem, endpoint_distance=0.683)

    def fail(*args, **kwargs):
        raise RuntimeError("native parameter lookup failed")

    monkeypatch.setattr(
        rdForceFieldHelpers,
        "GetUFFBondStretchParams"
        if method is ContinuousRelaxationForceField.UFF
        else "MMFFGetMoleculeProperties",
        fail,
    )
    with pytest.raises(RefinementError, match="hydrogen bond target") as caught:
        native.RdkitContinuousRelaxationBackend().relax(
            problem, restraint_library=RestraintLibrary({})
        )
    assert isinstance(caught.value.__cause__, RuntimeError)

"""Unit tests for the RDKit local refinement specialization."""

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    chain_payload,
    residue_payload,
)
from tests.support.canonical_builders import (
    build_structure as build_canonical_structure,
)
from tests.support.refinement_benchmarks import load_case_structure
from tests.support.refinement_cases import REFINEMENT_BENCHMARK_CASES
from tests.support.refinement_contract import (
    build_continuous_relaxation_problem,
)

import protrepair.transformer.continuous.rdkit as continuous_rdkit
from protrepair.chemistry import (
    BondDefinition,
    ChemicalComponentDefinition,
    ComponentLibrary,
    HeavyAtomSemantics,
    HydrogenSemantics,
    ResidueTemplate,
    RestraintLibrary,
    build_default_component_library,
    build_default_restraint_library,
)
from protrepair.chemistry.internal_coordinates import InternalCoordinateProgram
from protrepair.chemistry.restraint.template import (
    AngleRestraintTarget,
    BondRestraintTarget,
    ResidueRestraintTemplate,
)
from protrepair.diagnostics import ValidationIssueKind
from protrepair.errors import RdkitUnavailableError, RefinementError
from protrepair.geometry import Vec3
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import (
    AtomRef,
    ResidueId,
)
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import AtomIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.artifacts import (
    MovedAtomDelta,
    StructureDelta,
)
from protrepair.transformer.continuous.settings import (
    ContinuousRelaxationConfig,
    ContinuousRelaxationForceField,
)
from protrepair.transformer.local import (
    DirectRegionTransformationSpec,
    LocalScopeSpec,
    atom_input_from_local_scope_spec,
    transform_local_region,
)

RDKIT_AVAILABLE = (
    continuous_rdkit.Chem is not None
    and continuous_rdkit.rdBase is not None
    and continuous_rdkit.rdForceFieldHelpers is not None
)


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_refine_local_region_moves_selected_atom_and_keeps_context_fixed() -> None:
    """RDKit refinement should move selected atoms while keeping context fixed."""

    structure = build_toy_structure()
    refined = transform_local_region(
        structure,
        DirectRegionTransformationSpec(
            scope_spec=LocalScopeSpec.from_atoms(
                (AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),)
            ),
            force_field=ContinuousRelaxationForceField.UFF,
            config=ContinuousRelaxationConfig(
                context_radius_angstrom=2.5,
                max_iterations=200,
            ),
        ),
        component_library=build_toy_component_library(),
    )

    original_hydrogen = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "H1"))
    )
    refined_hydrogen = refined.refined_structure.geometry.atom_geometry(
        refined.refined_structure.constitution.atom_index(
            AtomRef(ResidueId("A", 1), "H1")
        )
    )
    original_anchor = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "C1"))
    )
    refined_anchor = refined.refined_structure.geometry.atom_geometry(
        refined.refined_structure.constitution.atom_index(
            AtomRef(ResidueId("A", 1), "C1")
        )
    )
    original_observer = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(ResidueId("L", 1), "O1"))
    )
    refined_observer = refined.refined_structure.geometry.atom_geometry(
        refined.refined_structure.constitution.atom_index(
            AtomRef(ResidueId("L", 1), "O1")
        )
    )

    assert original_hydrogen is not None
    assert refined_hydrogen is not None
    assert original_anchor is not None
    assert refined_anchor is not None
    assert original_observer is not None
    assert refined_observer is not None

    assert refined.delta.moved_atoms == (
        MovedAtomDelta(
            before_atom_index=structure.constitution.atom_index(
                AtomRef(ResidueId(chain_id="A", seq_num=1), "H1")
            ),
            after_atom_index=refined.refined_structure.constitution.atom_index(
                AtomRef(ResidueId(chain_id="A", seq_num=1), "H1")
            ),
        ),
    )
    assert refined_hydrogen.distance_to(
        refined_observer
    ) > original_hydrogen.distance_to(original_observer)
    assert refined_anchor.position == original_anchor.position
    assert refined_observer.position == original_observer.position


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_refine_local_region_supports_mmff_on_benchmark_fixture() -> None:
    """MMFF should run on one chemistry-valid literature-backed benchmark fixture."""

    case = REFINEMENT_BENCHMARK_CASES["1bkr-thr101"]
    structure = load_case_structure(case)
    problem = build_continuous_relaxation_problem(
        structure,
        LocalScopeSpec.from_atoms(case.focus_atom_refs),
        force_field=ContinuousRelaxationForceField.MMFF,
        context_radius_angstrom=3.0,
        max_iterations=100,
        component_library=build_default_component_library(),
    )
    refined = continuous_rdkit.RdkitContinuousRelaxationBackend().relax(
        problem,
        restraint_library=build_default_restraint_library(),
    )

    assert refined.backend_name == "rdkit"
    assert refined.moved_atom_count() > 0


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_refine_local_region_moves_ligand_atoms_against_polymer_context() -> None:
    """Ligand atom selections should optimize while nearby polymer atoms stay fixed."""

    structure = build_toy_structure()
    refined = transform_local_region(
        structure,
        DirectRegionTransformationSpec(
            scope_spec=LocalScopeSpec.from_atoms(
                (AtomRef(ResidueId(chain_id="L", seq_num=1), "O1"),)
            ),
            force_field=ContinuousRelaxationForceField.UFF,
            config=ContinuousRelaxationConfig(
                context_radius_angstrom=2.5,
                max_iterations=200,
            ),
        ),
        component_library=build_toy_component_library(),
    )

    original_observer = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(ResidueId("L", 1), "O1"))
    )
    refined_observer = refined.refined_structure.geometry.atom_geometry(
        refined.refined_structure.constitution.atom_index(
            AtomRef(ResidueId("L", 1), "O1")
        )
    )
    original_anchor = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "C1"))
    )
    refined_anchor = refined.refined_structure.geometry.atom_geometry(
        refined.refined_structure.constitution.atom_index(
            AtomRef(ResidueId("A", 1), "C1")
        )
    )

    assert original_observer is not None
    assert refined_observer is not None
    assert original_anchor is not None
    assert refined_anchor is not None

    assert refined.delta.moved_atoms == (
        MovedAtomDelta(
            before_atom_index=structure.constitution.atom_index(
                AtomRef(ResidueId(chain_id="L", seq_num=1), "O1")
            ),
            after_atom_index=refined.refined_structure.constitution.atom_index(
                AtomRef(ResidueId(chain_id="L", seq_num=1), "O1")
            ),
        ),
    )
    assert refined_observer.position != original_observer.position
    assert refined_anchor.position == original_anchor.position


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_rdkit_backend_discards_catastrophic_bond_distortion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RDKit outputs that tear restraint-backed bonds should not materialize."""

    structure = build_toy_structure()
    target_atom_index = structure.constitution.atom_index(
        AtomRef(ResidueId("A", 1), "H1")
    )
    problem = build_continuous_relaxation_problem(
        structure,
        LocalScopeSpec.from_atoms((AtomRef(ResidueId("A", 1), "H1"),)),
        force_field=ContinuousRelaxationForceField.UFF,
        context_radius_angstrom=2.5,
        max_iterations=1,
        component_library=build_toy_component_library(),
    )
    rdkit_target_atom_index = problem.region.included_atom_indices().index(
        target_atom_index
    )
    restraint_library = RestraintLibrary(
        templates={
            "MOV": ResidueRestraintTemplate(
                component_id="MOV",
                bond_targets=(
                    BondRestraintTarget(
                        atom_name_1="C1",
                        atom_name_2="H1",
                        target_distance_angstrom=1.09,
                        esd_angstrom=0.01,
                    ),
                ),
                angle_targets=(),
            ),
        }
    )
    original_conformer_position = continuous_rdkit.conformer_position

    def far_hydrogen_position(conformer, atom_index: int) -> Vec3:
        if atom_index == rdkit_target_atom_index:
            return Vec3(8.0, 0.0, 0.0)

        return original_conformer_position(conformer, atom_index)

    monkeypatch.setattr(
        continuous_rdkit,
        "conformer_position",
        far_hydrogen_position,
    )

    result = continuous_rdkit.RdkitContinuousRelaxationBackend().relax(
        problem,
        restraint_library=restraint_library,
    )

    assert result.refined_structure == structure
    assert result.delta.moved_atoms == ()
    assert any(
        issue.kind is ValidationIssueKind.REFINEMENT_REJECTED
        for issue in result.issues
    )


def test_refine_local_region_raises_rdkit_unavailable_error_when_missing_dependency(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The seam should surface one clear error when RDKit is unavailable."""

    monkeypatch.setattr(continuous_rdkit, "Chem", None)
    monkeypatch.setattr(continuous_rdkit, "rdBase", None)
    monkeypatch.setattr(continuous_rdkit, "rdForceFieldHelpers", None)

    with pytest.raises(RdkitUnavailableError, match="optional rdkit dependency"):
        transform_local_region(
            build_toy_structure(),
            DirectRegionTransformationSpec(
                scope_spec=LocalScopeSpec.from_atoms(
                    (AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),)
                ),
                force_field=ContinuousRelaxationForceField.UFF,
                config=ContinuousRelaxationConfig(context_radius_angstrom=2.5),
            ),
            component_library=build_toy_component_library(),
        )


def test_rdkit_element_symbol_normalizes_multicharacter_elements() -> None:
    """RDKit element construction should normalize two-character symbols."""

    assert continuous_rdkit._rdkit_element_symbol("FE") == "Fe"
    assert continuous_rdkit._rdkit_element_symbol("CL") == "Cl"
    assert continuous_rdkit._rdkit_element_symbol("ZN") == "Zn"
    assert continuous_rdkit._rdkit_element_symbol("C") == "C"


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_refine_local_region_wraps_rdkit_sanitize_failures() -> None:
    """Invalid local chemistry should surface as one canonical refinement error."""

    assert continuous_rdkit.Chem is not None

    original_sanitize = continuous_rdkit.Chem.SanitizeMol

    def raise_sanitize_error(*args, **kwargs):
        raise RuntimeError("boom")

    continuous_rdkit.Chem.SanitizeMol = raise_sanitize_error
    try:
        with pytest.raises(RefinementError, match="sanitize"):
            transform_local_region(
                build_toy_structure(),
                DirectRegionTransformationSpec(
                    scope_spec=LocalScopeSpec.from_atoms(
                        (AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),)
                    ),
                    force_field=ContinuousRelaxationForceField.UFF,
                    config=ContinuousRelaxationConfig(context_radius_angstrom=2.5),
                ),
                component_library=build_toy_component_library(),
            )
    finally:
        continuous_rdkit.Chem.SanitizeMol = original_sanitize


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_refine_local_region_wraps_natural_uff_parameterization_failures() -> None:
    """Chemically impossible local graphs should still surface one refinement error."""

    with pytest.raises(RefinementError, match="could not parameterize"):
        transform_local_region(
            build_invalid_valence_structure(),
            DirectRegionTransformationSpec(
                scope_spec=LocalScopeSpec.from_atoms(
                    (AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),)
                ),
                force_field=ContinuousRelaxationForceField.UFF,
                config=ContinuousRelaxationConfig(context_radius_angstrom=0.0),
            ),
            component_library=build_invalid_valence_component_library(),
        )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_refine_local_region_wraps_uff_parameterization_failures(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RDKit UFF parameterization errors should not leak backend exceptions."""

    assert continuous_rdkit.rdForceFieldHelpers is not None

    def raise_parameterization_error(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr(
        continuous_rdkit.rdForceFieldHelpers,
        "UFFGetMoleculeForceField",
        raise_parameterization_error,
    )

    with pytest.raises(RefinementError, match="could not parameterize"):
        transform_local_region(
            build_toy_structure(),
            DirectRegionTransformationSpec(
                scope_spec=LocalScopeSpec.from_atoms(
                    (AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),)
                ),
                force_field=ContinuousRelaxationForceField.UFF,
                config=ContinuousRelaxationConfig(context_radius_angstrom=2.5),
            ),
            component_library=build_toy_component_library(),
        )


def test_sidechain_local_selection_drops_position_constraint_cap() -> None:
    """Sidechain-local residue lowering should not add a position cap."""

    case = REFINEMENT_BENCHMARK_CASES["1bkr-thr101"]
    structure = load_case_structure(case)
    snapshot = ProteinStructureSnapshot.from_structure(structure)
    atom_input = atom_input_from_local_scope_spec(
        snapshot,
        LocalScopeSpec.from_residue_sidechains(case.focus_residue_ids),
    )

    assert atom_input.realizes_residue_sidechains()
    assert continuous_rdkit.position_constraint_max_displacement(atom_input) is None


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_refine_local_region_raises_when_uff_builder_returns_none(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A null force-field object should surface one canonical refinement error."""

    assert continuous_rdkit.rdForceFieldHelpers is not None

    monkeypatch.setattr(
        continuous_rdkit.rdForceFieldHelpers,
        "UFFGetMoleculeForceField",
        lambda *args, **kwargs: None,
    )

    with pytest.raises(RefinementError, match="could not build a force field"):
        transform_local_region(
            build_toy_structure(),
            DirectRegionTransformationSpec(
                scope_spec=LocalScopeSpec.from_atoms(
                    (AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),)
                ),
                force_field=ContinuousRelaxationForceField.UFF,
                config=ContinuousRelaxationConfig(context_radius_angstrom=2.5),
            ),
            component_library=build_toy_component_library(),
        )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_refine_local_region_rejects_unsupported_planned_bond_orders() -> None:
    """Illegal bond orders should fail before backend execution leaks weird state."""

    with pytest.raises(RefinementError, match="unsupported planned bond order"):
        transform_local_region(
            build_invalid_bond_order_structure(),
            DirectRegionTransformationSpec(
                scope_spec=LocalScopeSpec.from_residues(
                    (ResidueId(chain_id="A", seq_num=1),)
                ),
                force_field=ContinuousRelaxationForceField.UFF,
                config=ContinuousRelaxationConfig(context_radius_angstrom=0.0),
            ),
            component_library=build_invalid_bond_order_component_library(),
        )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_build_rdkit_force_field_adds_position_constraints_for_movable_atoms(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """RDKit should tether movable atoms and restraint-backed fixed anchors."""

    structure = build_toy_structure()
    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_atoms(
            (AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),)
        ),
        context_radius_angstrom=2.5,
        component_library=build_toy_component_library(),
    )
    molecule, rdkit_atom_index_by_structure_atom_index = (
        continuous_rdkit.build_rdkit_molecule(plan)
    )

    class FakeForceField:
        def __init__(self) -> None:
            self.fixed_points: list[int] = []
            self.position_constraints: list[tuple[int, float, float]] = []
            self.distance_constraints: list[
                tuple[int, int, bool, float, float, float]
            ] = []
            self.angle_constraints: list[
                tuple[int, int, int, bool, float, float, float]
            ] = []

        def AddFixedPoint(self, atom_index: int) -> None:
            self.fixed_points.append(atom_index)

        def UFFAddPositionConstraint(
            self,
            atom_index: int,
            max_displacement: float,
            force_constant: float,
        ) -> None:
            self.position_constraints.append(
                (atom_index, max_displacement, force_constant)
            )

        def UFFAddDistanceConstraint(
            self,
            atom_index_1: int,
            atom_index_2: int,
            relative: bool,
            minimum_distance: float,
            maximum_distance: float,
            force_constant: float,
        ) -> None:
            self.distance_constraints.append(
                (
                    atom_index_1,
                    atom_index_2,
                    relative,
                    minimum_distance,
                    maximum_distance,
                    force_constant,
                )
            )

        def UFFAddAngleConstraint(
            self,
            atom_index_1: int,
            center_atom_index: int,
            atom_index_2: int,
            relative: bool,
            minimum_angle_degrees: float,
            maximum_angle_degrees: float,
            force_constant: float,
        ) -> None:
            self.angle_constraints.append(
                (
                    atom_index_1,
                    center_atom_index,
                    atom_index_2,
                    relative,
                    minimum_angle_degrees,
                    maximum_angle_degrees,
                    force_constant,
                )
            )

    fake_force_field = FakeForceField()
    assert continuous_rdkit.rdForceFieldHelpers is not None
    restraint_library = RestraintLibrary(
        templates={
            "MOV": ResidueRestraintTemplate(
                component_id="MOV",
                bond_targets=(
                    BondRestraintTarget(
                        atom_name_1="C1",
                        atom_name_2="H1",
                        target_distance_angstrom=1.09,
                        esd_angstrom=0.01,
                    ),
                ),
                angle_targets=(),
            ),
        }
    )
    monkeypatch.setattr(
        continuous_rdkit.rdForceFieldHelpers,
        "UFFGetMoleculeForceField",
        lambda *args, **kwargs: fake_force_field,
    )

    continuous_rdkit.build_rdkit_force_field(
        molecule,
        problem=plan,
        restraint_library=restraint_library,
        rdkit_atom_index_by_structure_atom_index=(
            rdkit_atom_index_by_structure_atom_index
        ),
    )

    assert fake_force_field.position_constraints == [
        (
            rdkit_atom_index_by_structure_atom_index[
                structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "H1"))
            ],
            continuous_rdkit.ATOMWISE_POSITION_CONSTRAINT_MAX_DISPLACEMENT_ANGSTROM,
            continuous_rdkit.POSITION_CONSTRAINT_FORCE_CONSTANT,
        )
    ]
    assert fake_force_field.distance_constraints == [
        (
            rdkit_atom_index_by_structure_atom_index[
                structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "C1"))
            ],
            rdkit_atom_index_by_structure_atom_index[
                structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "H1"))
            ],
            False,
            pytest.approx(1.04),
            pytest.approx(1.14),
            continuous_rdkit.BOND_CONSTRAINT_FORCE_CONSTANT,
        )
    ]
    assert fake_force_field.angle_constraints == []
    assert fake_force_field.fixed_points == [
        rdkit_atom_index_by_structure_atom_index[
            structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "C1"))
        ],
        rdkit_atom_index_by_structure_atom_index[
            structure.constitution.atom_index(AtomRef(ResidueId("L", 1), "O1"))
        ],
        rdkit_atom_index_by_structure_atom_index[
            structure.constitution.atom_index(AtomRef(ResidueId("L", 1), "H2"))
        ],
        rdkit_atom_index_by_structure_atom_index[
            structure.constitution.atom_index(AtomRef(ResidueId("L", 1), "H3"))
        ],
    ]


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_build_rdkit_force_field_adds_geometry_constraints_for_residuewise_selection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Residuewise selections should add residue-local bond and angle constraints."""

    structure = build_small_angle_structure()
    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_residues((ResidueId(chain_id="A", seq_num=1),)),
        context_radius_angstrom=0.0,
        component_library=build_small_angle_component_library(),
    )
    molecule, rdkit_atom_index_by_structure_atom_index = (
        continuous_rdkit.build_rdkit_molecule(plan)
    )

    class FakeForceField:
        def __init__(self) -> None:
            self.fixed_points: list[int] = []
            self.position_constraints: list[tuple[int, float, float]] = []
            self.distance_constraints: list[
                tuple[int, int, bool, float, float, float]
            ] = []
            self.angle_constraints: list[
                tuple[int, int, int, bool, float, float, float]
            ] = []

        def AddFixedPoint(self, atom_index: int) -> None:
            self.fixed_points.append(atom_index)

        def UFFAddPositionConstraint(
            self,
            atom_index: int,
            max_displacement: float,
            force_constant: float,
        ) -> None:
            self.position_constraints.append(
                (atom_index, max_displacement, force_constant)
            )

        def UFFAddDistanceConstraint(
            self,
            atom_index_1: int,
            atom_index_2: int,
            relative: bool,
            minimum_distance: float,
            maximum_distance: float,
            force_constant: float,
        ) -> None:
            self.distance_constraints.append(
                (
                    atom_index_1,
                    atom_index_2,
                    relative,
                    minimum_distance,
                    maximum_distance,
                    force_constant,
                )
            )

        def UFFAddAngleConstraint(
            self,
            atom_index_1: int,
            center_atom_index: int,
            atom_index_2: int,
            relative: bool,
            minimum_angle_degrees: float,
            maximum_angle_degrees: float,
            force_constant: float,
        ) -> None:
            self.angle_constraints.append(
                (
                    atom_index_1,
                    center_atom_index,
                    atom_index_2,
                    relative,
                    minimum_angle_degrees,
                    maximum_angle_degrees,
                    force_constant,
                )
            )

    fake_force_field = FakeForceField()
    assert continuous_rdkit.rdForceFieldHelpers is not None
    restraint_library = RestraintLibrary(
        templates={
            "ANG": ResidueRestraintTemplate(
                component_id="ANG",
                bond_targets=(
                    BondRestraintTarget(
                        atom_name_1="A1",
                        atom_name_2="A2",
                        target_distance_angstrom=1.20,
                        esd_angstrom=0.01,
                    ),
                    BondRestraintTarget(
                        atom_name_1="A2",
                        atom_name_2="A3",
                        target_distance_angstrom=1.30,
                        esd_angstrom=0.02,
                    ),
                ),
                angle_targets=(
                    AngleRestraintTarget(
                        atom_name_1="A1",
                        center_atom_name="A2",
                        atom_name_2="A3",
                        target_angle_degrees=120.0,
                        esd_degrees=1.0,
                    ),
                ),
            ),
        }
    )
    monkeypatch.setattr(
        continuous_rdkit.rdForceFieldHelpers,
        "UFFGetMoleculeForceField",
        lambda *args, **kwargs: fake_force_field,
    )

    continuous_rdkit.build_rdkit_force_field(
        molecule,
        problem=plan,
        restraint_library=restraint_library,
        rdkit_atom_index_by_structure_atom_index=(
            rdkit_atom_index_by_structure_atom_index
        ),
    )

    assert len(fake_force_field.position_constraints) == 3
    assert fake_force_field.distance_constraints == [
        (
            rdkit_atom_index_by_structure_atom_index[
                structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "A1"))
            ],
            rdkit_atom_index_by_structure_atom_index[
                structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "A2"))
            ],
            False,
            pytest.approx(1.15),
            pytest.approx(1.25),
            continuous_rdkit.BOND_CONSTRAINT_FORCE_CONSTANT,
        ),
        (
            rdkit_atom_index_by_structure_atom_index[
                structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "A2"))
            ],
            rdkit_atom_index_by_structure_atom_index[
                structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "A3"))
            ],
            False,
            pytest.approx(1.22),
            pytest.approx(1.38),
            continuous_rdkit.BOND_CONSTRAINT_FORCE_CONSTANT,
        ),
    ]
    assert fake_force_field.angle_constraints == [
        (
            rdkit_atom_index_by_structure_atom_index[
                structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "A1"))
            ],
            rdkit_atom_index_by_structure_atom_index[
                structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "A2"))
            ],
            rdkit_atom_index_by_structure_atom_index[
                structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "A3"))
            ],
            False,
            pytest.approx(115.0),
            pytest.approx(125.0),
            continuous_rdkit.ANGLE_CONSTRAINT_FORCE_CONSTANT,
        )
    ]
    assert fake_force_field.fixed_points == []


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_build_rdkit_force_field_uses_mmff_builder_and_constraints(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """MMFF specs should use the MMFF builder path and MMFF constraint calls."""

    structure = build_small_angle_structure()
    plan = build_continuous_relaxation_problem(
        structure,
        scope_spec=LocalScopeSpec.from_residues((ResidueId(chain_id="A", seq_num=1),)),
        force_field=ContinuousRelaxationForceField.MMFF,
        context_radius_angstrom=0.0,
        component_library=build_small_angle_component_library(),
    )
    molecule, rdkit_atom_index_by_structure_atom_index = (
        continuous_rdkit.build_rdkit_molecule(plan)
    )

    class FakeForceField:
        def __init__(self) -> None:
            self.fixed_points: list[int] = []
            self.position_constraints: list[tuple[int, float, float]] = []
            self.distance_constraints: list[
                tuple[int, int, bool, float, float, float]
            ] = []
            self.angle_constraints: list[
                tuple[int, int, int, bool, float, float, float]
            ] = []

        def AddFixedPoint(self, atom_index: int) -> None:
            self.fixed_points.append(atom_index)

        def MMFFAddPositionConstraint(
            self,
            atom_index: int,
            max_displacement: float,
            force_constant: float,
        ) -> None:
            self.position_constraints.append(
                (atom_index, max_displacement, force_constant)
            )

        def MMFFAddDistanceConstraint(
            self,
            atom_index_1: int,
            atom_index_2: int,
            relative: bool,
            minimum_distance: float,
            maximum_distance: float,
            force_constant: float,
        ) -> None:
            self.distance_constraints.append(
                (
                    atom_index_1,
                    atom_index_2,
                    relative,
                    minimum_distance,
                    maximum_distance,
                    force_constant,
                )
            )

        def MMFFAddAngleConstraint(
            self,
            atom_index_1: int,
            center_atom_index: int,
            atom_index_2: int,
            relative: bool,
            minimum_angle_degrees: float,
            maximum_angle_degrees: float,
            force_constant: float,
        ) -> None:
            self.angle_constraints.append(
                (
                    atom_index_1,
                    center_atom_index,
                    atom_index_2,
                    relative,
                    minimum_angle_degrees,
                    maximum_angle_degrees,
                    force_constant,
                )
            )

    fake_force_field = FakeForceField()
    assert continuous_rdkit.rdForceFieldHelpers is not None
    restraint_library = RestraintLibrary(
        templates={
            "ANG": ResidueRestraintTemplate(
                component_id="ANG",
                bond_targets=(
                    BondRestraintTarget(
                        atom_name_1="A1",
                        atom_name_2="A2",
                        target_distance_angstrom=1.20,
                        esd_angstrom=0.01,
                    ),
                    BondRestraintTarget(
                        atom_name_1="A2",
                        atom_name_2="A3",
                        target_distance_angstrom=1.30,
                        esd_angstrom=0.02,
                    ),
                ),
                angle_targets=(
                    AngleRestraintTarget(
                        atom_name_1="A1",
                        center_atom_name="A2",
                        atom_name_2="A3",
                        target_angle_degrees=120.0,
                        esd_degrees=1.0,
                    ),
                ),
            ),
        }
    )
    monkeypatch.setattr(
        continuous_rdkit.rdForceFieldHelpers,
        "MMFFHasAllMoleculeParams",
        lambda *args, **kwargs: True,
    )
    monkeypatch.setattr(
        continuous_rdkit.rdForceFieldHelpers,
        "MMFFGetMoleculeProperties",
        lambda *args, **kwargs: object(),
    )
    monkeypatch.setattr(
        continuous_rdkit.rdForceFieldHelpers,
        "MMFFGetMoleculeForceField",
        lambda *args, **kwargs: fake_force_field,
    )

    continuous_rdkit.build_rdkit_force_field(
        molecule,
        problem=plan,
        restraint_library=restraint_library,
        rdkit_atom_index_by_structure_atom_index=(
            rdkit_atom_index_by_structure_atom_index
        ),
    )

    assert len(fake_force_field.position_constraints) == 3
    assert fake_force_field.distance_constraints == [
        (
            rdkit_atom_index_by_structure_atom_index[
                structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "A1"))
            ],
            rdkit_atom_index_by_structure_atom_index[
                structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "A2"))
            ],
            False,
            pytest.approx(1.15),
            pytest.approx(1.25),
            continuous_rdkit.BOND_CONSTRAINT_FORCE_CONSTANT,
        ),
        (
            rdkit_atom_index_by_structure_atom_index[
                structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "A2"))
            ],
            rdkit_atom_index_by_structure_atom_index[
                structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "A3"))
            ],
            False,
            pytest.approx(1.22),
            pytest.approx(1.38),
            continuous_rdkit.BOND_CONSTRAINT_FORCE_CONSTANT,
        ),
    ]
    assert fake_force_field.angle_constraints == [
        (
            rdkit_atom_index_by_structure_atom_index[
                structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "A1"))
            ],
            rdkit_atom_index_by_structure_atom_index[
                structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "A2"))
            ],
            rdkit_atom_index_by_structure_atom_index[
                structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "A3"))
            ],
            False,
            pytest.approx(115.0),
            pytest.approx(125.0),
            continuous_rdkit.ANGLE_CONSTRAINT_FORCE_CONSTANT,
        )
    ]
    assert fake_force_field.fixed_points == []


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_refine_local_region_preserves_atom_metadata_and_structure_order() -> None:
    """Coordinate updates should preserve metadata and canonical structure order."""

    structure = build_toy_structure_with_metadata()
    refined = transform_local_region(
        structure,
        DirectRegionTransformationSpec(
            scope_spec=LocalScopeSpec.from_atoms(
                (AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),)
            ),
            force_field=ContinuousRelaxationForceField.UFF,
            config=ContinuousRelaxationConfig(
                context_radius_angstrom=2.5,
                max_iterations=200,
            ),
        ),
        component_library=build_toy_component_library(),
    )

    original_residue = structure.constitution.residue_or_ligand(ResidueId("A", 1))
    refined_residue = refined.refined_structure.constitution.residue_or_ligand(
        ResidueId("A", 1)
    )
    assert original_residue is not None
    assert refined_residue is not None
    original_hydrogen = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "H1"))
    )
    refined_hydrogen = refined.refined_structure.geometry.atom_geometry(
        refined.refined_structure.constitution.atom_index(
            AtomRef(ResidueId("A", 1), "H1")
        )
    )
    original_observer = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(ResidueId("L", 1), "O1"))
    )
    refined_observer = refined.refined_structure.geometry.atom_geometry(
        refined.refined_structure.constitution.atom_index(
            AtomRef(ResidueId("L", 1), "O1")
        )
    )

    assert tuple(
        residue.residue_id
        for residue in refined.refined_structure.chain_site("A").residues
    ) == tuple(residue.residue_id for residue in structure.chain_site("A").residues)
    assert refined_residue.atom_site_names() == original_residue.atom_site_names()

    assert refined_hydrogen.occupancy == original_hydrogen.occupancy
    assert refined_hydrogen.b_factor == original_hydrogen.b_factor
    assert refined_hydrogen.altloc == original_hydrogen.altloc
    assert refined.refined_structure.topology.formal_charge(
        refined.refined_structure.constitution.atom_index(
            AtomRef(ResidueId("A", 1), "H1")
        )
    ) == structure.topology.formal_charge(
        structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "H1"))
    )

    assert refined_observer.occupancy == original_observer.occupancy
    assert refined_observer.b_factor == original_observer.b_factor
    assert refined_observer.altloc == original_observer.altloc
    assert refined_observer.position == original_observer.position
    assert (
        refined.refined_structure.provenance.ingress.source_name
        == structure.provenance.ingress.source_name
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_refine_local_region_preserves_all_atoms_of_multi_atom_fixed_context() -> None:
    """Whole context residues should remain fixed across promoted context residues."""

    structure = build_multi_atom_context_structure()
    refined = transform_local_region(
        structure,
        DirectRegionTransformationSpec(
            scope_spec=LocalScopeSpec.from_atoms(
                (AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),)
            ),
            force_field=ContinuousRelaxationForceField.UFF,
            config=ContinuousRelaxationConfig(
                context_radius_angstrom=2.0,
                max_iterations=200,
            ),
        ),
        component_library=build_multi_atom_context_component_library(),
    )

    original_context = structure.constitution.ligands[0]
    refined_context = refined.refined_structure.constitution.ligands[0]
    refined_positions = []
    for atom_site in refined_context.atom_sites:
        atom_geometry = refined.refined_structure.geometry.atom_geometry(
            refined.refined_structure.constitution.atom_index(
                AtomRef(refined_context.residue_id, atom_site.name)
            )
        )
        refined_positions.append(atom_geometry.position)

    original_positions = []
    for atom_site in original_context.atom_sites:
        atom_geometry = structure.geometry.atom_geometry(
            structure.constitution.atom_index(
                AtomRef(original_context.residue_id, atom_site.name)
            )
        )
        original_positions.append(atom_geometry.position)

    assert tuple(refined_positions) == tuple(original_positions)


def test_structure_delta_rejects_unknown_moved_atom_slots() -> None:
    """Delta integrity should reject moved-atom slots outside the constitution."""

    structure = build_toy_structure()

    with pytest.raises(ValueError, match="outside the constitution atom slots"):
        StructureDelta(
            before_constitution=structure.constitution,
            after_constitution=structure.constitution,
            moved_atoms=(
                MovedAtomDelta(
                    before_atom_index=AtomIndex(999),
                    after_atom_index=AtomIndex(999),
                ),
            ),
        )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_refine_local_region_canonicalizes_moved_atom_order() -> None:
    """Moved atom refs should be reported in canonical structure order."""

    refined = transform_local_region(
        build_isolated_structure(),
        DirectRegionTransformationSpec(
            scope_spec=LocalScopeSpec.from_atoms(
                (
                    AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),
                    AtomRef(ResidueId(chain_id="A", seq_num=1), "C1"),
                )
            ),
            force_field=ContinuousRelaxationForceField.UFF,
            config=ContinuousRelaxationConfig(
                context_radius_angstrom=0.0,
                max_iterations=200,
            ),
        ),
        component_library=build_toy_component_library(),
    )

    assert refined.delta.moved_atoms == (
        MovedAtomDelta(
            before_atom_index=refined.refined_structure.constitution.atom_index(
                AtomRef(ResidueId(chain_id="A", seq_num=1), "C1")
            ),
            after_atom_index=refined.refined_structure.constitution.atom_index(
                AtomRef(ResidueId(chain_id="A", seq_num=1), "C1")
            ),
        ),
        MovedAtomDelta(
            before_atom_index=refined.refined_structure.constitution.atom_index(
                AtomRef(ResidueId(chain_id="A", seq_num=1), "H1")
            ),
            after_atom_index=refined.refined_structure.constitution.atom_index(
                AtomRef(ResidueId(chain_id="A", seq_num=1), "H1")
            ),
        ),
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_refine_local_region_canonicalizes_mixed_domain_moved_atom_order() -> None:
    """Mixed polymer and ligand selections should report canonical move ordering."""

    refined = transform_local_region(
        build_toy_structure(),
        DirectRegionTransformationSpec(
            scope_spec=LocalScopeSpec.from_atoms(
                (
                    AtomRef(ResidueId(chain_id="L", seq_num=1), "O1"),
                    AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),
                )
            ),
            force_field=ContinuousRelaxationForceField.UFF,
            config=ContinuousRelaxationConfig(
                context_radius_angstrom=2.5,
                max_iterations=200,
            ),
        ),
        component_library=build_toy_component_library(),
    )

    assert refined.delta.moved_atoms == (
        MovedAtomDelta(
            before_atom_index=refined.refined_structure.constitution.atom_index(
                AtomRef(ResidueId(chain_id="A", seq_num=1), "H1")
            ),
            after_atom_index=refined.refined_structure.constitution.atom_index(
                AtomRef(ResidueId(chain_id="A", seq_num=1), "H1")
            ),
        ),
        MovedAtomDelta(
            before_atom_index=refined.refined_structure.constitution.atom_index(
                AtomRef(ResidueId(chain_id="L", seq_num=1), "O1")
            ),
            after_atom_index=refined.refined_structure.constitution.atom_index(
                AtomRef(ResidueId(chain_id="L", seq_num=1), "O1")
            ),
        ),
    )


@pytest.mark.skipif(not RDKIT_AVAILABLE, reason="rdkit is not installed")
def test_refine_local_region_can_run_with_zero_context_and_same_residue_closure() -> (
    None
):
    """Selected atoms should still refine against same-residue fixed context."""

    structure = build_isolated_structure()
    refined = transform_local_region(
        structure,
        DirectRegionTransformationSpec(
            scope_spec=LocalScopeSpec.from_atoms(
                (AtomRef(ResidueId(chain_id="A", seq_num=1), "H1"),)
            ),
            force_field=ContinuousRelaxationForceField.UFF,
            config=ContinuousRelaxationConfig(
                context_radius_angstrom=0.0,
                max_iterations=200,
            ),
        ),
        component_library=build_toy_component_library(),
    )

    original_anchor = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "C1"))
    )
    original_hydrogen = structure.geometry.atom_geometry(
        structure.constitution.atom_index(AtomRef(ResidueId("A", 1), "H1"))
    )
    refined_anchor = refined.refined_structure.geometry.atom_geometry(
        refined.refined_structure.constitution.atom_index(
            AtomRef(ResidueId("A", 1), "C1")
        )
    )
    refined_hydrogen = refined.refined_structure.geometry.atom_geometry(
        refined.refined_structure.constitution.atom_index(
            AtomRef(ResidueId("A", 1), "H1")
        )
    )

    assert original_anchor is not None
    assert original_hydrogen is not None
    assert refined_anchor is not None
    assert refined_hydrogen is not None

    assert refined.delta.moved_atoms == (
        MovedAtomDelta(
            before_atom_index=structure.constitution.atom_index(
                AtomRef(ResidueId(chain_id="A", seq_num=1), "H1")
            ),
            after_atom_index=refined.refined_structure.constitution.atom_index(
                AtomRef(ResidueId(chain_id="A", seq_num=1), "H1")
            ),
        ),
    )
    assert refined_anchor.position == original_anchor.position
    assert refined_hydrogen.position != original_hydrogen.position


def build_toy_component_library() -> ComponentLibrary:
    """Return one tiny component library suitable for RDKit refinement tests."""

    mov_template = ResidueTemplate(
        definition=ChemicalComponentDefinition(
            component_id="MOV",
            atom_names=("C1", "H1"),
            bonds=(BondDefinition("C1", "H1"),),
        ),
        heavy_atom_semantics=HeavyAtomSemantics(
            program=InternalCoordinateProgram.backbone_only(),
            atom_order=("C1",),
        ),
        hydrogen_semantics=HydrogenSemantics(
            plan_with_backbone=((("H1",), "class3", ("C1", "C1", "C1")),),
        ),
    )
    obs_template = ResidueTemplate(
        definition=ChemicalComponentDefinition(
            component_id="OBS",
            atom_names=("O1", "H2", "H3"),
            bonds=(
                BondDefinition("O1", "H2"),
                BondDefinition("O1", "H3"),
            ),
        ),
        heavy_atom_semantics=HeavyAtomSemantics(
            program=InternalCoordinateProgram.backbone_only(),
            atom_order=("O1",),
        ),
        hydrogen_semantics=HydrogenSemantics(
            plan_with_backbone=((("H2", "H3"), "class2", ("O1", "O1", "O1")),),
        ),
    )
    return ComponentLibrary(
        templates={
            "MOV": mov_template,
            "OBS": obs_template,
        }
    )


def build_invalid_valence_component_library() -> ComponentLibrary:
    """Return one component library with one impossible carbon valence."""

    return ComponentLibrary(
        templates={
            "INV": ResidueTemplate(
                definition=ChemicalComponentDefinition(
                    component_id="INV",
                    atom_names=("C1", "H1", "H2", "H3", "H4", "H5"),
                    bonds=(
                        BondDefinition("C1", "H1"),
                        BondDefinition("C1", "H2"),
                        BondDefinition("C1", "H3"),
                        BondDefinition("C1", "H4"),
                        BondDefinition("C1", "H5"),
                    ),
                ),
                heavy_atom_semantics=HeavyAtomSemantics(
                    program=InternalCoordinateProgram.backbone_only(),
                    atom_order=("C1",),
                ),
                hydrogen_semantics=HydrogenSemantics(
                    plan_with_backbone=(
                        (
                            ("H1", "H2", "H3", "H4", "H5"),
                            "class2",
                            ("C1", "C1", "C1"),
                        ),
                    ),
                ),
            )
        }
    )


def build_invalid_bond_order_component_library() -> ComponentLibrary:
    """Return one component library with one unsupported bond order."""

    return ComponentLibrary(
        templates={
            "INV": ResidueTemplate(
                definition=ChemicalComponentDefinition(
                    component_id="INV",
                    atom_names=("C1", "O1"),
                    bonds=(BondDefinition("C1", "O1", order=4),),
                ),
                heavy_atom_semantics=HeavyAtomSemantics(
                    program=InternalCoordinateProgram.backbone_only(),
                    atom_order=("C1", "O1"),
                ),
            )
        }
    )


def build_small_angle_component_library() -> ComponentLibrary:
    """Return one component library for bond/angle-constraint coverage."""

    return ComponentLibrary(
        templates={
            "ANG": ResidueTemplate(
                definition=ChemicalComponentDefinition(
                    component_id="ANG",
                    atom_names=("A1", "A2", "A3"),
                    bonds=(
                        BondDefinition("A1", "A2"),
                        BondDefinition("A2", "A3"),
                    ),
                )
            )
        }
    )


def build_toy_structure() -> ProteinStructure:
    """Return one tiny local environment with one hydrogen clash."""

    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="MOV",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("H1", "H", Vec3(1.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="OBS",
                residue_id=ResidueId(chain_id="L", seq_num=1),
                atoms=(
                    atom_payload("O1", "O", Vec3(1.9, 0.0, 0.0)),
                    atom_payload("H2", "H", Vec3(2.5, 0.75, 0.0)),
                    atom_payload("H3", "H", Vec3(2.5, -0.75, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )


def build_small_angle_structure() -> ProteinStructure:
    """Return one tiny three-atom residue for bond/angle constraint coverage."""

    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="ANG",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("A1", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("A2", "N", Vec3(1.2, 0.0, 0.0)),
                            atom_payload("A3", "O", Vec3(2.0, 0.9, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )


def build_isolated_structure() -> ProteinStructure:
    """Return one one-residue structure that can only see same-residue context."""

    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="MOV",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("H1", "H", Vec3(2.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )


def build_toy_structure_with_metadata() -> ProteinStructure:
    """Return one local environment whose atom metadata should survive refinement."""

    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="MOV",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload(
                                "C1",
                                "C",
                                Vec3(0.0, 0.0, 0.0),
                                occupancy=0.95,
                                b_factor=12.0,
                            ),
                            atom_payload(
                                "H1",
                                "H",
                                Vec3(1.0, 0.0, 0.0),
                                occupancy=0.60,
                                b_factor=17.5,
                                formal_charge=0,
                                altloc="A",
                            ),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="OBS",
                residue_id=ResidueId(chain_id="L", seq_num=1),
                atoms=(
                    atom_payload(
                        "O1",
                        "O",
                        Vec3(1.9, 0.0, 0.0),
                        occupancy=0.80,
                        b_factor=25.0,
                        altloc="B",
                    ),
                    atom_payload("H2", "H", Vec3(2.5, 0.75, 0.0)),
                    atom_payload("H3", "H", Vec3(2.5, -0.75, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
        source_name="toy-local-environment",
    )


def build_invalid_valence_structure() -> ProteinStructure:
    """Return one residue whose explicit bonding graph is chemically impossible."""

    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="INV",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("H1", "H", Vec3(1.0, 0.0, 0.0)),
                            atom_payload("H2", "H", Vec3(-1.0, 0.0, 0.0)),
                            atom_payload("H3", "H", Vec3(0.0, 1.0, 0.0)),
                            atom_payload("H4", "H", Vec3(0.0, -1.0, 0.0)),
                            atom_payload("H5", "H", Vec3(0.0, 0.0, 1.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )


def build_invalid_bond_order_structure() -> ProteinStructure:
    """Return one structure whose template requests one illegal bond order."""

    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="INV",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("O1", "O", Vec3(1.2, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )


def build_multi_atom_context_component_library() -> ComponentLibrary:
    """Return one tiny component library with a multi-atom context residue."""

    return ComponentLibrary(
        templates={
            "MOV": ResidueTemplate(
                definition=ChemicalComponentDefinition(
                    component_id="MOV",
                    atom_names=("C1", "H1"),
                    bonds=(BondDefinition("C1", "H1"),),
                ),
                heavy_atom_semantics=HeavyAtomSemantics(
                    program=InternalCoordinateProgram.backbone_only(),
                    atom_order=("C1",),
                ),
                hydrogen_semantics=HydrogenSemantics(
                    plan_with_backbone=((("H1",), "class3", ("C1", "C1", "C1")),),
                ),
            ),
            "CTX": ResidueTemplate(
                definition=ChemicalComponentDefinition(
                    component_id="CTX",
                    atom_names=("O1", "C2", "HO1", "H21", "H22", "H23"),
                    bonds=(
                        BondDefinition("O1", "C2"),
                        BondDefinition("O1", "HO1"),
                        BondDefinition("C2", "H21"),
                        BondDefinition("C2", "H22"),
                        BondDefinition("C2", "H23"),
                    ),
                ),
                heavy_atom_semantics=HeavyAtomSemantics(
                    program=InternalCoordinateProgram.backbone_only(),
                    atom_order=("O1", "C2"),
                ),
                hydrogen_semantics=HydrogenSemantics(
                    plan_with_backbone=(
                        (("HO1",), "class2", ("O1", "O1", "O1")),
                        (("H21", "H22", "H23"), "class2", ("C2", "C2", "C2")),
                    ),
                ),
            ),
        }
    )


def build_multi_atom_context_structure() -> ProteinStructure:
    """Return one local environment with one promoted multi-atom context residue."""

    return build_canonical_structure(
        chains=(
            chain_payload(
                "A",
                (
                    residue_payload(
                        component_id="MOV",
                        residue_id=ResidueId(chain_id="A", seq_num=1),
                        atoms=(
                            atom_payload("C1", "C", Vec3(0.0, 0.0, 0.0)),
                            atom_payload("H1", "H", Vec3(2.0, 0.0, 0.0)),
                        ),
                    ),
                ),
            ),
        ),
        ligands=(
            residue_payload(
                component_id="CTX",
                residue_id=ResidueId(chain_id="L", seq_num=1),
                atoms=(
                    atom_payload("O1", "O", Vec3(2.2, 0.0, 0.0)),
                    atom_payload("C2", "C", Vec3(8.0, 0.0, 0.0)),
                    atom_payload("HO1", "H", Vec3(2.2, 0.96, 0.0)),
                    atom_payload("H21", "H", Vec3(8.9, 0.0, 0.0)),
                    atom_payload("H22", "H", Vec3(7.7, 0.85, 0.0)),
                    atom_payload("H23", "H", Vec3(7.7, -0.85, 0.0)),
                ),
                is_hetero=True,
            ),
        ),
        source_format=FileFormat.PDB,
    )

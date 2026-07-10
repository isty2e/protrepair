"""RDKit realization of the continuous local-relaxation backend seam."""

from dataclasses import dataclass

try:
    from rdkit import Chem, rdBase
    from rdkit.Chem import rdForceFieldHelpers
except ImportError:  # pragma: no cover - exercised by availability checks
    Chem = None
    rdBase = None
    rdForceFieldHelpers = None

from protrepair.chemistry.restraint.library import RestraintLibrary
from protrepair.chemistry.restraint.template import (
    AngleRestraintTarget,
    BondRestraintTarget,
    ResidueRestraintTemplate,
)
from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.errors import RdkitUnavailableError, RefinementError
from protrepair.geometry import Vec3
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import StructureConstitution
from protrepair.structure.slots import AtomIndex, ResidueIndex
from protrepair.transformer.artifacts import RegionTransformationResult
from protrepair.transformer.artifacts.patch import MovedAtomDelta, StructureDelta
from protrepair.transformer.atom_input import AtomInput
from protrepair.transformer.continuous.domain import (
    ContinuousRelaxationProblem,
    ContinuousRelaxationRegion,
    PlannedBond,
)
from protrepair.transformer.continuous.settings import ContinuousRelaxationForceField
from protrepair.transformer.continuous.shared import apply_position_updates

ATOMWISE_POSITION_CONSTRAINT_MAX_DISPLACEMENT_ANGSTROM = 0.35
# Keep residue-local refinement equally conservative so seeded side-chain
# cleanup does not distort the selected residue backbone.
RESIDUEWISE_POSITION_CONSTRAINT_MAX_DISPLACEMENT_ANGSTROM = 0.35
POSITION_CONSTRAINT_FORCE_CONSTANT = 50.0
BOND_CONSTRAINT_ESD_MULTIPLIER = 4.0
BOND_CONSTRAINT_MIN_TOLERANCE_ANGSTROM = 0.05
BOND_CONSTRAINT_FORCE_CONSTANT = 80.0
BACKBONE_WINDOW_BOND_CONSTRAINT_FORCE_CONSTANT = 500.0
ANGLE_CONSTRAINT_ESD_MULTIPLIER = 4.0
ANGLE_CONSTRAINT_MIN_TOLERANCE_DEGREES = 5.0
ANGLE_CONSTRAINT_FORCE_CONSTANT = 30.0
CATASTROPHIC_BOND_DISTORTION_DEVIATION_ANGSTROM = 0.20
MMFF_VARIANT = "MMFF94s"
PEPTIDE_CN_TARGET_DISTANCE_ANGSTROM = 1.329
PEPTIDE_CN_DISTANCE_TOLERANCE_ANGSTROM = 0.06


@dataclass(frozen=True, slots=True)
class _CatastrophicBondDistortion:
    """One high-confidence bonded geometry failure in an RDKit output."""

    residue_index: ResidueIndex
    component_id: str
    atom_name_1: str
    atom_name_2: str
    observed_distance_angstrom: float
    target_distance_angstrom: float

    def deviation_angstrom(self) -> float:
        """Return the absolute bond-length deviation from the restraint target."""

        return abs(self.observed_distance_angstrom - self.target_distance_angstrom)


@dataclass(frozen=True, slots=True)
class _ChiralityChange:
    """One assigned chiral center changed by an RDKit output."""

    atom_index: AtomIndex
    before_code: str
    after_code: str | None

RDKIT_SANITIZE_FLAGS = 0
if Chem is not None:  # pragma: no branch
    RDKIT_SANITIZE_FLAGS = (
        Chem.SanitizeFlags.SANITIZE_FINDRADICALS
        | Chem.SanitizeFlags.SANITIZE_SETAROMATICITY
        | Chem.SanitizeFlags.SANITIZE_SETCONJUGATION
        | Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION
        | Chem.SanitizeFlags.SANITIZE_SYMMRINGS
    )


@dataclass(frozen=True, slots=True)
class RdkitContinuousRelaxationBackend:
    """RDKit-backed continuous local geometry optimizer."""

    def relax(
        self,
        problem: ContinuousRelaxationProblem,
        *,
        restraint_library: RestraintLibrary,
    ) -> RegionTransformationResult:
        """Execute one canonical continuous-relaxation problem through RDKit."""

        if Chem is None or rdBase is None or rdForceFieldHelpers is None:
            raise RdkitUnavailableError(
                "RDKit-backed continuous relaxation requires the required "
                "rdkit dependency"
            )

        molecule, rdkit_atom_index_by_structure_atom_index = build_rdkit_molecule(
            problem
        )
        force_field = build_rdkit_force_field(
            molecule,
            problem=problem,
            restraint_library=restraint_library,
            rdkit_atom_index_by_structure_atom_index=(
                rdkit_atom_index_by_structure_atom_index
            ),
        )
        force_field.Initialize()
        force_field.Minimize(maxIts=problem.spec.max_iterations)

        moved_positions = {
            atom_index: conformer_position(
                molecule.GetConformer(),
                rdkit_atom_index_by_structure_atom_index[atom_index],
            )
            for atom_index in problem.region.movable_atom_indices
        }
        refined_structure = apply_position_updates(
            problem.region.snapshot.structure,
            moved_positions=moved_positions,
        )
        chirality_changes = _rdkit_chirality_changes(
            problem,
            molecule=molecule,
            rdkit_atom_index_by_structure_atom_index=(
                rdkit_atom_index_by_structure_atom_index
            ),
            refined_structure=refined_structure,
        )
        if chirality_changes:
            return _rejected_chirality_change_result(
                problem,
                chirality_changes=chirality_changes,
                backend_version=rdBase.rdkitVersion,
            )

        catastrophic_distortions = catastrophic_bond_distortions(
            problem,
            refined_structure=refined_structure,
            restraint_library=restraint_library,
        )
        if catastrophic_distortions:
            return rejected_original_structure_result(
                problem,
                catastrophic_distortions=catastrophic_distortions,
                backend_version=rdBase.rdkitVersion,
            )

        return RegionTransformationResult(
            refined_structure=refined_structure,
            delta=StructureDelta(
                before_constitution=problem.region.snapshot.structure.constitution,
                after_constitution=refined_structure.constitution,
                moved_atoms=tuple(
                    MovedAtomDelta(
                        before_atom_index=atom_index,
                        after_atom_index=atom_index,
                    )
                    for atom_index in problem.region.movable_atom_indices
                ),
            ),
            issues=(),
            backend_name="rdkit",
            backend_version=rdBase.rdkitVersion,
        )


def catastrophic_bond_distortions(
    problem: ContinuousRelaxationProblem,
    *,
    refined_structure: ProteinStructure,
    restraint_library: RestraintLibrary,
) -> tuple[_CatastrophicBondDistortion, ...]:
    """Return severe restraint-backed bond distortions introduced by RDKit."""

    constitution = problem.region.snapshot.structure.constitution
    movable_atom_index_set = set(problem.region.movable_atom_indices)
    distortions: list[_CatastrophicBondDistortion] = []
    for residue_index in problem.region.included_residue_indices:
        residue_site = constitution.residue_site_at(residue_index)
        restraint_template = restraint_library.get(residue_site.component_id)
        if restraint_template is None:
            continue

        for target in restraint_template.bond_targets:
            if not residue_site.has_atom_site(target.atom_name_1):
                continue
            if not residue_site.has_atom_site(target.atom_name_2):
                continue

            atom_index_1 = constitution.atom_index_in_residue(
                residue_index,
                target.atom_name_1,
            )
            atom_index_2 = constitution.atom_index_in_residue(
                residue_index,
                target.atom_name_2,
            )
            if (
                atom_index_1 not in movable_atom_index_set
                and atom_index_2 not in movable_atom_index_set
            ):
                continue

            observed_distance = refined_structure.geometry.atom_geometry(
                atom_index_1
            ).distance_to(refined_structure.geometry.atom_geometry(atom_index_2))
            deviation = abs(observed_distance - target.target_distance_angstrom)
            if deviation < CATASTROPHIC_BOND_DISTORTION_DEVIATION_ANGSTROM:
                continue

            distortions.append(
                _CatastrophicBondDistortion(
                    residue_index=residue_index,
                    component_id=residue_site.component_id,
                    atom_name_1=target.atom_name_1,
                    atom_name_2=target.atom_name_2,
                    observed_distance_angstrom=observed_distance,
                    target_distance_angstrom=target.target_distance_angstrom,
                )
            )

    return tuple(
        sorted(
            distortions,
            key=lambda distortion: (
                -distortion.deviation_angstrom(),
                distortion.residue_index.value,
                distortion.atom_name_1,
                distortion.atom_name_2,
            ),
        )
    )


def _rdkit_chirality_changes(
    problem: ContinuousRelaxationProblem,
    *,
    molecule,
    rdkit_atom_index_by_structure_atom_index: dict[AtomIndex, int],
    refined_structure: ProteinStructure,
) -> tuple[_ChiralityChange, ...]:
    """Return assigned 3D chiral centers changed by an RDKit output."""

    before_assignments = _rdkit_chiral_center_assignments_for_structure(
        molecule,
        rdkit_atom_index_by_structure_atom_index=(
            rdkit_atom_index_by_structure_atom_index
        ),
        structure=problem.region.snapshot.structure,
    )
    if not before_assignments:
        return ()

    after_assignments = _rdkit_chiral_center_assignments_for_structure(
        molecule,
        rdkit_atom_index_by_structure_atom_index=(
            rdkit_atom_index_by_structure_atom_index
        ),
        structure=refined_structure,
    )
    changes = [
        _ChiralityChange(
            atom_index=atom_index,
            before_code=before_code,
            after_code=after_assignments.get(atom_index),
        )
        for atom_index, before_code in before_assignments.items()
        if after_assignments.get(atom_index) != before_code
    ]
    return tuple(
        sorted(
            changes,
            key=lambda change: change.atom_index.value,
        )
    )


def _rdkit_chiral_center_assignments_for_structure(
    molecule,
    *,
    rdkit_atom_index_by_structure_atom_index: dict[AtomIndex, int],
    structure: ProteinStructure,
) -> dict[AtomIndex, str]:
    """Return assigned R/S centers for one structure on an RDKit topology."""

    assert Chem is not None

    working_molecule = Chem.Mol(molecule)
    conformer = working_molecule.GetConformer()
    conformer.Set3D(True)
    for (
        structure_atom_index,
        rdkit_atom_index,
    ) in rdkit_atom_index_by_structure_atom_index.items():
        atom_geometry = structure.geometry.atom_geometry(structure_atom_index)
        conformer.SetAtomPosition(rdkit_atom_index, tuple(atom_geometry.position))

    Chem.AssignStereochemistryFrom3D(
        working_molecule,
        confId=conformer.GetId(),
        replaceExistingTags=True,
    )
    structure_atom_index_by_rdkit_atom_index = {
        rdkit_atom_index: structure_atom_index
        for structure_atom_index, rdkit_atom_index in (
            rdkit_atom_index_by_structure_atom_index.items()
        )
    }
    return {
        structure_atom_index_by_rdkit_atom_index[rdkit_atom_index]: str(cip_code)
        for rdkit_atom_index, cip_code in Chem.FindMolChiralCenters(
            working_molecule,
            force=True,
            includeUnassigned=False,
            includeCIP=True,
        )
        if cip_code in {"R", "S"}
    }


def rejected_original_structure_result(
    problem: ContinuousRelaxationProblem,
    *,
    catastrophic_distortions: tuple[_CatastrophicBondDistortion, ...],
    backend_version: str,
) -> RegionTransformationResult:
    """Return an explicit no-op result when RDKit tears bonded geometry."""

    original_structure = problem.region.snapshot.structure
    worst_distortion = catastrophic_distortions[0]
    residue_site = original_structure.constitution.residue_site_at(
        worst_distortion.residue_index
    )
    issue = ValidationIssue.for_residue(
        kind=ValidationIssueKind.REFINEMENT_REJECTED,
        severity=IssueSeverity.WARNING,
        residue_id=residue_site.residue_id,
        message=(
            "RDKit local refinement output was discarded because it produced "
            "catastrophic restraint-backed bond distortion: "
            f"{residue_site.residue_id.display_token()} "
            f"{worst_distortion.component_id} "
            f"{worst_distortion.atom_name_1}-{worst_distortion.atom_name_2} "
            f"{worst_distortion.observed_distance_angstrom:.3f} A vs "
            f"{worst_distortion.target_distance_angstrom:.3f} A target "
            f"({len(catastrophic_distortions)} severe bond distortion(s))"
        ),
    )
    return RegionTransformationResult(
        refined_structure=original_structure,
        delta=StructureDelta(
            before_constitution=original_structure.constitution,
            after_constitution=original_structure.constitution,
        ),
        issues=(issue,),
        backend_name="rdkit",
        backend_version=backend_version,
    )


def _rejected_chirality_change_result(
    problem: ContinuousRelaxationProblem,
    *,
    chirality_changes: tuple[_ChiralityChange, ...],
    backend_version: str,
) -> RegionTransformationResult:
    """Return an explicit no-op result when RDKit changes assigned chirality."""

    original_structure = problem.region.snapshot.structure
    worst_change = chirality_changes[0]
    constitution = original_structure.constitution
    residue_site = constitution.residue_site_at(
        constitution.residue_index_for_atom_index(worst_change.atom_index)
    )
    atom_site = constitution.atom_site_at(worst_change.atom_index)
    after_code = worst_change.after_code or "unassigned"
    issue = ValidationIssue.for_residue(
        kind=ValidationIssueKind.REFINEMENT_REJECTED,
        severity=IssueSeverity.WARNING,
        residue_id=residue_site.residue_id,
        message=(
            "RDKit local refinement output was discarded because it changed "
            "assigned stereochemistry: "
            f"{residue_site.residue_id.display_token()} "
            f"{residue_site.component_id} {atom_site.name} "
            f"{worst_change.before_code}->{after_code} "
            f"({len(chirality_changes)} changed chiral center(s))"
        ),
    )
    return RegionTransformationResult(
        refined_structure=original_structure,
        delta=StructureDelta(
            before_constitution=original_structure.constitution,
            after_constitution=original_structure.constitution,
        ),
        issues=(issue,),
        backend_name="rdkit",
        backend_version=backend_version,
    )


def build_rdkit_molecule(
    problem: ContinuousRelaxationProblem,
):
    """Return one RDKit molecule plus atom-index mapping for one problem."""

    assert Chem is not None

    molecule = Chem.RWMol()
    rdkit_atom_index_by_structure_atom_index: dict[AtomIndex, int] = {}
    conformer = Chem.Conformer(len(problem.region.included_atom_indices()))
    conformer.Set3D(True)
    for atom_index in problem.region.included_atom_indices():
        atom_site = problem.region.atom_site(atom_index)
        atom_geometry = problem.region.atom_geometry(atom_index)
        element_identity = atom_site.element_identity
        rdkit_atom = Chem.Atom(
            _rdkit_element_symbol(element_identity.chemical_symbol)
        )
        if element_identity.isotope_mass_number is not None:
            rdkit_atom.SetIsotope(element_identity.isotope_mass_number)
        rdkit_atom.SetNoImplicit(True)
        formal_charge = problem.region.formal_charge(atom_index)
        if formal_charge is not None:
            rdkit_atom.SetFormalCharge(formal_charge)

        rdkit_atom_index_by_structure_atom_index[atom_index] = molecule.AddAtom(
            rdkit_atom
        )
        conformer.SetAtomPosition(
            rdkit_atom_index_by_structure_atom_index[atom_index],
            tuple(atom_geometry.position),
        )

    for bond in problem.bonds:
        left_index = rdkit_atom_index_by_structure_atom_index[bond.atom_index_1]
        right_index = rdkit_atom_index_by_structure_atom_index[bond.atom_index_2]
        molecule.AddBond(
            left_index,
            right_index,
            rdkit_bond_type(bond),
        )

    molecule.AddConformer(conformer)
    finalized_molecule = molecule.GetMol()
    finalized_molecule.UpdatePropertyCache(strict=False)
    try:
        Chem.SanitizeMol(finalized_molecule, sanitizeOps=RDKIT_SANITIZE_FLAGS)
    except Exception as error:  # pragma: no cover - backend-specific failure path
        raise RefinementError(
            "RDKit could not sanitize the selected continuous-relaxation region"
        ) from error

    return finalized_molecule, rdkit_atom_index_by_structure_atom_index


def _rdkit_element_symbol(element: str) -> str:
    """Return one RDKit-compatible element symbol."""

    normalized_element = element.strip()
    if len(normalized_element) >= 2:
        return normalized_element[0].upper() + normalized_element[1:].lower()

    return normalized_element.upper()


def build_rdkit_force_field(
    molecule,
    *,
    problem: ContinuousRelaxationProblem,
    restraint_library: RestraintLibrary,
    rdkit_atom_index_by_structure_atom_index: dict[AtomIndex, int],
):
    """Return one initialized RDKit force field for a continuous-relaxation run."""

    assert rdForceFieldHelpers is not None

    force_field = construct_rdkit_force_field(
        molecule,
        force_field_family=problem.spec.force_field,
    )

    add_position_constraints(
        force_field,
        atom_input=problem.region.atom_input,
        force_field_family=problem.spec.force_field,
        movable_atom_indices=problem.region.movable_atom_indices,
        rdkit_atom_index_by_structure_atom_index=(
            rdkit_atom_index_by_structure_atom_index
        ),
    )
    add_bond_and_angle_constraints(
        force_field,
        region=problem.region,
        restraint_library=restraint_library,
        force_field_family=problem.spec.force_field,
        rdkit_atom_index_by_structure_atom_index=(
            rdkit_atom_index_by_structure_atom_index
        ),
    )
    add_inter_residue_peptide_constraints(
        force_field,
        region=problem.region,
        force_field_family=problem.spec.force_field,
        rdkit_atom_index_by_structure_atom_index=(
            rdkit_atom_index_by_structure_atom_index
        ),
    )
    for atom_index in problem.region.fixed_context_atom_indices:
        force_field.AddFixedPoint(rdkit_atom_index_by_structure_atom_index[atom_index])

    return force_field


def add_position_constraints(
    force_field,
    *,
    atom_input: AtomInput,
    force_field_family: ContinuousRelaxationForceField,
    movable_atom_indices: tuple[AtomIndex, ...],
    rdkit_atom_index_by_structure_atom_index: dict[AtomIndex, int],
) -> None:
    """Apply conservative soft position constraints to movable atoms."""

    max_displacement = position_constraint_max_displacement(atom_input)
    if max_displacement is None:
        return

    for atom_index in movable_atom_indices:
        add_rdkit_position_constraint(
            force_field,
            force_field_family=force_field_family,
            atom_index=rdkit_atom_index_by_structure_atom_index[atom_index],
            max_displacement=max_displacement,
            force_constant=POSITION_CONSTRAINT_FORCE_CONSTANT,
        )


def add_bond_and_angle_constraints(
    force_field,
    *,
    region: ContinuousRelaxationRegion,
    restraint_library: RestraintLibrary,
    force_field_family: ContinuousRelaxationForceField,
    rdkit_atom_index_by_structure_atom_index: dict[AtomIndex, int],
) -> None:
    """Apply residue-local bond and angle soft constraints for movable atoms."""

    constitution = region.snapshot.structure.constitution
    movable_atom_names_by_residue_index: dict[ResidueIndex, set[str]] = {}
    for atom_index in region.movable_atom_indices:
        residue_index = constitution.residue_index_for_atom_index(atom_index)
        movable_atom_names_by_residue_index.setdefault(residue_index, set()).add(
            constitution.atom_site_at(atom_index).name
        )

    for (
        residue_index,
        movable_atom_names,
    ) in movable_atom_names_by_residue_index.items():
        residue_site = constitution.residue_site_at(residue_index)
        if not movable_atom_names:
            continue

        restraint_template = restraint_library.get(residue_site.component_id)
        if restraint_template is None:
            continue

        add_residue_bond_constraints(
            force_field,
            force_field_family=force_field_family,
            constitution=constitution,
            residue_index=residue_index,
            movable_atom_names=movable_atom_names,
            rdkit_atom_index_by_structure_atom_index=(
                rdkit_atom_index_by_structure_atom_index
            ),
            restraint_template=restraint_template,
            force_constant=bond_constraint_force_constant(region),
        )
        add_residue_angle_constraints(
            force_field,
            force_field_family=force_field_family,
            constitution=constitution,
            residue_index=residue_index,
            movable_atom_names=movable_atom_names,
            rdkit_atom_index_by_structure_atom_index=(
                rdkit_atom_index_by_structure_atom_index
            ),
            restraint_template=restraint_template,
        )


def add_inter_residue_peptide_constraints(
    force_field,
    *,
    region: ContinuousRelaxationRegion,
    force_field_family: ContinuousRelaxationForceField,
    rdkit_atom_index_by_structure_atom_index: dict[AtomIndex, int],
) -> None:
    """Apply peptide C-N distance constraints across included residue boundaries."""

    movable_atom_index_set = set(region.movable_atom_indices)
    for bond in region.inter_residue_bonds:
        if (
            bond.atom_index_1 not in rdkit_atom_index_by_structure_atom_index
            or bond.atom_index_2 not in rdkit_atom_index_by_structure_atom_index
        ):
            continue
        if (
            bond.atom_index_1 not in movable_atom_index_set
            and bond.atom_index_2 not in movable_atom_index_set
        ):
            continue
        if not _is_peptide_cn_bond(region, bond):
            continue

        add_rdkit_distance_constraint(
            force_field,
            force_field_family=force_field_family,
            atom_index_1=rdkit_atom_index_by_structure_atom_index[bond.atom_index_1],
            atom_index_2=rdkit_atom_index_by_structure_atom_index[bond.atom_index_2],
            minimum_distance=(
                PEPTIDE_CN_TARGET_DISTANCE_ANGSTROM
                - PEPTIDE_CN_DISTANCE_TOLERANCE_ANGSTROM
            ),
            maximum_distance=(
                PEPTIDE_CN_TARGET_DISTANCE_ANGSTROM
                + PEPTIDE_CN_DISTANCE_TOLERANCE_ANGSTROM
            ),
            force_constant=BOND_CONSTRAINT_FORCE_CONSTANT,
        )


def _is_peptide_cn_bond(
    region: ContinuousRelaxationRegion,
    bond: PlannedBond,
) -> bool:
    """Return whether one inter-residue planned bond is a peptide C-N bond."""

    constitution = region.snapshot.structure.constitution
    residue_index_1 = constitution.residue_index_for_atom_index(bond.atom_index_1)
    residue_index_2 = constitution.residue_index_for_atom_index(bond.atom_index_2)
    if residue_index_1 == residue_index_2:
        return False

    atom_names = {
        constitution.atom_site_at(bond.atom_index_1).name,
        constitution.atom_site_at(bond.atom_index_2).name,
    }
    return atom_names == {"C", "N"}


def add_residue_bond_constraints(
    force_field,
    *,
    force_field_family: ContinuousRelaxationForceField,
    constitution: StructureConstitution,
    residue_index: ResidueIndex,
    movable_atom_names: set[str],
    rdkit_atom_index_by_structure_atom_index: dict[AtomIndex, int],
    restraint_template: ResidueRestraintTemplate,
    force_constant: float = BOND_CONSTRAINT_FORCE_CONSTANT,
) -> None:
    """Apply residue-local bond soft constraints for one movable residue."""

    residue_site = constitution.residue_site_at(residue_index)
    for target in restraint_template.bond_targets:
        if (
            target.atom_name_1 not in movable_atom_names
            and target.atom_name_2 not in movable_atom_names
        ):
            continue

        if not residue_site.has_atom_site(target.atom_name_1):
            continue
        if not residue_site.has_atom_site(target.atom_name_2):
            continue

        atom_index_1 = constitution.atom_index_in_residue(
            residue_index,
            target.atom_name_1,
        )
        atom_index_2 = constitution.atom_index_in_residue(
            residue_index,
            target.atom_name_2,
        )

        tolerance = bond_constraint_tolerance_angstrom(target)
        add_rdkit_distance_constraint(
            force_field,
            force_field_family=force_field_family,
            atom_index_1=rdkit_atom_index_by_structure_atom_index[atom_index_1],
            atom_index_2=rdkit_atom_index_by_structure_atom_index[atom_index_2],
            minimum_distance=target.target_distance_angstrom - tolerance,
            maximum_distance=target.target_distance_angstrom + tolerance,
            force_constant=force_constant,
        )


def bond_constraint_force_constant(region: ContinuousRelaxationRegion) -> float:
    """Return the residue-local bond restraint force constant for one region."""

    if region.atom_input.realizes_residue_backbones():
        return BACKBONE_WINDOW_BOND_CONSTRAINT_FORCE_CONSTANT

    return BOND_CONSTRAINT_FORCE_CONSTANT


def add_residue_angle_constraints(
    force_field,
    *,
    force_field_family: ContinuousRelaxationForceField,
    constitution: StructureConstitution,
    residue_index: ResidueIndex,
    movable_atom_names: set[str],
    rdkit_atom_index_by_structure_atom_index: dict[AtomIndex, int],
    restraint_template: ResidueRestraintTemplate,
) -> None:
    """Apply residue-local angle soft constraints for one movable residue."""

    residue_site = constitution.residue_site_at(residue_index)
    for target in restraint_template.angle_targets:
        if (
            target.atom_name_1 not in movable_atom_names
            and target.center_atom_name not in movable_atom_names
            and target.atom_name_2 not in movable_atom_names
        ):
            continue

        if not residue_site.has_atom_site(target.atom_name_1):
            continue
        if not residue_site.has_atom_site(target.center_atom_name):
            continue
        if not residue_site.has_atom_site(target.atom_name_2):
            continue

        atom_index_1 = constitution.atom_index_in_residue(
            residue_index,
            target.atom_name_1,
        )
        center_atom_index = constitution.atom_index_in_residue(
            residue_index,
            target.center_atom_name,
        )
        atom_index_2 = constitution.atom_index_in_residue(
            residue_index,
            target.atom_name_2,
        )

        tolerance = angle_constraint_tolerance_degrees(target)
        add_rdkit_angle_constraint(
            force_field,
            force_field_family=force_field_family,
            atom_index_1=rdkit_atom_index_by_structure_atom_index[atom_index_1],
            center_atom_index=rdkit_atom_index_by_structure_atom_index[
                center_atom_index
            ],
            atom_index_2=rdkit_atom_index_by_structure_atom_index[atom_index_2],
            minimum_angle_degrees=target.target_angle_degrees - tolerance,
            maximum_angle_degrees=target.target_angle_degrees + tolerance,
            force_constant=ANGLE_CONSTRAINT_FORCE_CONSTANT,
        )


def position_constraint_max_displacement(
    atom_input: AtomInput,
) -> float | None:
    """Return the per-domain soft displacement cap for RDKit refinement."""

    if atom_input.realizes_residue_backbones():
        return RESIDUEWISE_POSITION_CONSTRAINT_MAX_DISPLACEMENT_ANGSTROM

    if atom_input.realizes_residue_sidechains():
        return None

    if atom_input.is_atomwise():
        return ATOMWISE_POSITION_CONSTRAINT_MAX_DISPLACEMENT_ANGSTROM

    return RESIDUEWISE_POSITION_CONSTRAINT_MAX_DISPLACEMENT_ANGSTROM


def construct_rdkit_force_field(
    molecule,
    *,
    force_field_family: ContinuousRelaxationForceField,
):
    """Return one RDKit force field for the requested family."""

    if force_field_family is ContinuousRelaxationForceField.UFF:
        return build_uff_force_field(molecule)
    if force_field_family is ContinuousRelaxationForceField.MMFF:
        return build_mmff_force_field(molecule)

    raise RefinementError(
        f"refinement force field {force_field_family.value!r} is not implemented"
    )


def build_uff_force_field(molecule):
    """Return one RDKit UFF force field for a continuous-relaxation molecule."""

    assert rdForceFieldHelpers is not None

    try:
        force_field = rdForceFieldHelpers.UFFGetMoleculeForceField(
            molecule,
            confId=0,
            ignoreInterfragInteractions=False,
        )
    except Exception as error:  # pragma: no cover - defensive backend boundary
        raise RefinementError(
            "RDKit UFF could not parameterize the selected continuous-relaxation region"
        ) from error

    if force_field is None:
        raise RefinementError(
            "RDKit UFF could not build a force field for the selected local region"
        )

    return force_field


def build_mmff_force_field(molecule):
    """Return one RDKit MMFF force field for a continuous-relaxation molecule."""

    assert rdForceFieldHelpers is not None

    try:
        has_all_parameters = rdForceFieldHelpers.MMFFHasAllMoleculeParams(molecule)
    except Exception as error:  # pragma: no cover - defensive backend boundary
        raise RefinementError(
            "RDKit MMFF could not parameterize the selected "
            "continuous-relaxation region"
        ) from error

    if not has_all_parameters:
        raise RefinementError(
            "RDKit MMFF could not parameterize the selected "
            "continuous-relaxation region"
        )

    try:
        mmff_properties = rdForceFieldHelpers.MMFFGetMoleculeProperties(
            molecule,
            mmffVariant=MMFF_VARIANT,
        )
    except Exception as error:  # pragma: no cover - defensive backend boundary
        raise RefinementError(
            "RDKit MMFF could not parameterize the selected "
            "continuous-relaxation region"
        ) from error

    if mmff_properties is None:
        raise RefinementError(
            "RDKit MMFF could not parameterize the selected "
            "continuous-relaxation region"
        )

    try:
        force_field = rdForceFieldHelpers.MMFFGetMoleculeForceField(
            molecule,
            mmff_properties,
            confId=0,
            ignoreInterfragInteractions=False,
        )
    except Exception as error:  # pragma: no cover - defensive backend boundary
        raise RefinementError(
            "RDKit MMFF could not parameterize the selected "
            "continuous-relaxation region"
        ) from error

    if force_field is None:
        raise RefinementError(
            "RDKit MMFF could not build a force field for the selected local region"
        )

    return force_field


def add_rdkit_position_constraint(
    force_field,
    *,
    force_field_family: ContinuousRelaxationForceField,
    atom_index: int,
    max_displacement: float,
    force_constant: float,
) -> None:
    """Apply one position constraint through the selected RDKit force field."""

    if force_field_family is ContinuousRelaxationForceField.UFF:
        force_field.UFFAddPositionConstraint(
            atom_index,
            max_displacement,
            force_constant,
        )
        return

    force_field.MMFFAddPositionConstraint(
        atom_index,
        max_displacement,
        force_constant,
    )


def add_rdkit_distance_constraint(
    force_field,
    *,
    force_field_family: ContinuousRelaxationForceField,
    atom_index_1: int,
    atom_index_2: int,
    minimum_distance: float,
    maximum_distance: float,
    force_constant: float,
) -> None:
    """Apply one distance constraint through the selected RDKit force field."""

    if force_field_family is ContinuousRelaxationForceField.UFF:
        force_field.UFFAddDistanceConstraint(
            atom_index_1,
            atom_index_2,
            False,
            minimum_distance,
            maximum_distance,
            force_constant,
        )
        return

    force_field.MMFFAddDistanceConstraint(
        atom_index_1,
        atom_index_2,
        False,
        minimum_distance,
        maximum_distance,
        force_constant,
    )


def add_rdkit_angle_constraint(
    force_field,
    *,
    force_field_family: ContinuousRelaxationForceField,
    atom_index_1: int,
    center_atom_index: int,
    atom_index_2: int,
    minimum_angle_degrees: float,
    maximum_angle_degrees: float,
    force_constant: float,
) -> None:
    """Apply one angle constraint through the selected RDKit force field."""

    if force_field_family is ContinuousRelaxationForceField.UFF:
        force_field.UFFAddAngleConstraint(
            atom_index_1,
            center_atom_index,
            atom_index_2,
            False,
            minimum_angle_degrees,
            maximum_angle_degrees,
            force_constant,
        )
        return

    force_field.MMFFAddAngleConstraint(
        atom_index_1,
        center_atom_index,
        atom_index_2,
        False,
        minimum_angle_degrees,
        maximum_angle_degrees,
        force_constant,
    )


def bond_constraint_tolerance_angstrom(target: BondRestraintTarget) -> float:
    """Return one conservative soft-window width for a bond target."""

    if target.esd_angstrom is None:
        return BOND_CONSTRAINT_MIN_TOLERANCE_ANGSTROM

    return max(
        BOND_CONSTRAINT_MIN_TOLERANCE_ANGSTROM,
        target.esd_angstrom * BOND_CONSTRAINT_ESD_MULTIPLIER,
    )


def angle_constraint_tolerance_degrees(target: AngleRestraintTarget) -> float:
    """Return one conservative soft-window width for an angle target."""

    if target.esd_degrees is None:
        return ANGLE_CONSTRAINT_MIN_TOLERANCE_DEGREES

    return max(
        ANGLE_CONSTRAINT_MIN_TOLERANCE_DEGREES,
        target.esd_degrees * ANGLE_CONSTRAINT_ESD_MULTIPLIER,
    )


def rdkit_bond_type(bond: PlannedBond):
    """Return the RDKit bond type for one planned bond."""

    assert Chem is not None

    if bond.aromatic:
        return Chem.BondType.AROMATIC

    if bond.order == 1:
        return Chem.BondType.SINGLE

    if bond.order == 2:
        return Chem.BondType.DOUBLE

    if bond.order == 3:
        return Chem.BondType.TRIPLE

    raise RefinementError(f"unsupported planned bond order {bond.order}")


def conformer_position(conformer, atom_index: int) -> Vec3:
    """Return one RDKit conformer position as a canonical ``Vec3``."""

    coordinates = conformer.GetAtomPosition(atom_index)
    return Vec3(
        x=float(coordinates.x),
        y=float(coordinates.y),
        z=float(coordinates.z),
    )

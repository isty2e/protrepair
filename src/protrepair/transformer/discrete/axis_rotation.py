"""Axis-rotation candidate construction for discrete pre-refinement."""

from collections.abc import Iterable
from math import pi

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.diagnostics.clashes import (
    ClashDetectionBasis,
    StericClash,
    detect_clashes_from_context,
    prepare_clash_detection_context,
)
from protrepair.diagnostics.geometry import detect_heavy_geometry
from protrepair.scope import AtomSetScope
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.labels import ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.atom_input import AtomInput
from protrepair.transformer.discrete.models import (
    AxisRotationCorrectionDefinition,
    DiscreteCorrectionCandidate,
    DiscreteCorrectionScore,
)
from protrepair.transformer.discrete.orientation import (
    replace_residue,
    rotate_residue_atoms_about_axis,
)

AXIS_ROTATION_RADIANS = pi

AXIS_ROTATION_CORRECTION_DEFINITIONS: dict[str, AxisRotationCorrectionDefinition] = {
    "ASN": AxisRotationCorrectionDefinition(
        component_id="ASN",
        axis_atom_names=("CB", "CG"),
        rotating_atom_names=("OD1", "ND2"),
        hydrogen_anchor_atom_names=("ND2",),
    ),
    "GLN": AxisRotationCorrectionDefinition(
        component_id="GLN",
        axis_atom_names=("CG", "CD"),
        rotating_atom_names=("OE1", "NE2"),
        hydrogen_anchor_atom_names=("NE2",),
    ),
    "HIS": AxisRotationCorrectionDefinition(
        component_id="HIS",
        axis_atom_names=("CB", "CG"),
        rotating_atom_names=("ND1", "CD2", "CE1", "NE2"),
        hydrogen_anchor_atom_names=("ND1", "CD2", "CE1", "NE2"),
    ),
}


def select_best_discrete_correction_candidate(
    *,
    snapshot: ProteinStructureSnapshot,
    atom_input: AtomInput,
    structure: ProteinStructure,
    residue_id: ResidueId,
    component_library: ComponentLibrary,
    clash_basis: ClashDetectionBasis | None = None,
) -> DiscreteCorrectionCandidate | None:
    """Return one improved residue-local discrete correction candidate if any."""

    residue_site = structure.constitution.residue_or_ligand(residue_id)
    if residue_site is None:
        return None

    residue_index = structure.constitution.residue_index(residue_id)
    residue_geometry = structure.residue_geometry(residue_index)

    component_id = component_library.normalize_component_id(residue_site.component_id)
    definition = AXIS_ROTATION_CORRECTION_DEFINITIONS.get(component_id)
    if definition is None or not selection_allows_axis_rotation_correction(
        atom_input=atom_input,
        residue_site=residue_site,
        residue_geometry=residue_geometry,
        residue_id=residue_id,
        definition=definition,
    ):
        return None

    if not definition.supports_residue(residue_site):
        return None

    original_candidate = score_discrete_correction_candidate(
        structure=structure,
        residue_id=residue_id,
        component_library=component_library,
        moved_atom_indices=(),
        clash_basis=clash_basis,
    )
    flipped_candidate = build_rotated_axis_correction_candidate(
        structure=structure,
        residue_site=residue_site,
        residue_geometry=residue_geometry,
        formal_charge_by_atom_name=structure.residue_formal_charge_by_atom_name(
            residue_index
        ),
        definition=definition,
        component_library=component_library,
        clash_basis=clash_basis,
    )
    if flipped_candidate.score < original_candidate.score:
        return flipped_candidate

    return None


def snapshot_has_applicable_axis_rotation_correction(
    snapshot: ProteinStructureSnapshot,
    atom_input: AtomInput,
    component_library: ComponentLibrary,
) -> bool:
    """Return whether any referenced residue admits axis-rotation correction."""

    for residue_id in atom_input.referenced_residue_ids():
        residue_site = snapshot.structure.constitution.residue_or_ligand(residue_id)
        if residue_site is None:
            continue

        residue_geometry = snapshot.structure.residue_geometry(
            snapshot.structure.constitution.residue_index(residue_id)
        )

        component_id = component_library.normalize_component_id(
            residue_site.component_id
        )
        definition = AXIS_ROTATION_CORRECTION_DEFINITIONS.get(component_id)
        if definition is None:
            continue

        if not selection_allows_axis_rotation_correction(
            atom_input=atom_input,
            residue_site=residue_site,
            residue_geometry=residue_geometry,
            residue_id=residue_id,
            definition=definition,
        ):
            continue

        if definition.supports_residue(residue_site):
            return True

    return False


def selection_allows_axis_rotation_correction(
    *,
    atom_input: AtomInput,
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    residue_id: ResidueId,
    definition: AxisRotationCorrectionDefinition,
) -> bool:
    """Return whether the current refinement selection authorizes one axis rotation."""

    if atom_input.is_residuewise():
        return residue_id in atom_input.referenced_residue_ids()

    selected_scope = atom_input.as_scope()
    if not isinstance(selected_scope, AtomSetScope):
        raise TypeError("atomwise axis-rotation selection requires an AtomSetScope")

    selected_atom_names = {
        atom_ref.atom_name
        for atom_ref in selected_scope.atom_refs
        if atom_ref.residue_id == residue_id
    }
    required_atom_names = definition.rotated_atom_names(
        residue_site=residue_site,
        residue_geometry=residue_geometry,
    )
    return required_atom_names.issubset(selected_atom_names)


def build_rotated_axis_correction_candidate(
    *,
    structure: ProteinStructure,
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    formal_charge_by_atom_name: tuple[tuple[str, int | None], ...],
    definition: AxisRotationCorrectionDefinition,
    component_library: ComponentLibrary,
    clash_basis: ClashDetectionBasis | None = None,
) -> DiscreteCorrectionCandidate:
    """Return one 180-degree axis-rotated residue candidate."""

    (
        updated_residue_site,
        updated_residue_geometry,
        updated_formal_charge_by_atom_name,
        moved_atom_names,
    ) = rotate_residue_atoms_about_axis(
        residue_site=residue_site,
        residue_geometry=residue_geometry,
        formal_charge_by_atom_name=formal_charge_by_atom_name,
        axis_atom_names=definition.axis_atom_names,
        rotated_atom_names=set(
            definition.rotated_atom_names(
                residue_site=residue_site,
                residue_geometry=residue_geometry,
            )
        ),
        theta_radians=AXIS_ROTATION_RADIANS,
    )
    updated_structure = replace_residue(
        structure=structure,
        residue_site=updated_residue_site,
        residue_geometry=updated_residue_geometry,
        formal_charge_by_atom_name=updated_formal_charge_by_atom_name,
    )
    return score_discrete_correction_candidate(
        structure=updated_structure,
        residue_id=residue_site.residue_id,
        component_library=component_library,
        moved_atom_indices=tuple(
            structure.constitution.atom_index_in_residue(
                structure.constitution.residue_index(residue_site.residue_id),
                atom_name,
            )
            for atom_name in moved_atom_names
        ),
        clash_basis=clash_basis,
    )


def score_discrete_correction_candidate(
    *,
    structure: ProteinStructure,
    residue_id: ResidueId,
    component_library: ComponentLibrary,
    moved_atom_indices: tuple[AtomIndex, ...],
    clash_basis: ClashDetectionBasis | None = None,
) -> DiscreteCorrectionCandidate:
    """Return one scored residue-local discrete correction candidate."""

    clash_context = prepare_clash_detection_context(
        structure,
        component_library=component_library,
        basis=clash_basis,
    )
    focus_clashes = detect_clashes_from_context(
        clash_context,
        focus_residue_ids=frozenset((residue_id,)),
    ).clashes
    focus_heavy_clashes = _heavy_heavy_clashes(
        structure,
        focus_clashes,
    )
    geometry_report = detect_heavy_geometry(
        structure,
        component_library=component_library,
        residue_ids=frozenset((residue_id,)),
    )
    return DiscreteCorrectionCandidate(
        structure=structure,
        moved_atom_indices=moved_atom_indices,
        score=DiscreteCorrectionScore(
            focus_heavy_clash_count=len(focus_heavy_clashes),
            focus_clash_count=len(focus_clashes),
            focus_geometry_outlier_count=(
                len(geometry_report.bond_length_outliers)
                + len(geometry_report.bond_angle_outliers)
            ),
            focus_heavy_fractional_clash_overlap_sum=(
                focus_fractional_clash_overlap_sum(focus_heavy_clashes)
            ),
            focus_fractional_clash_overlap_sum=(
                focus_fractional_clash_overlap_sum(focus_clashes)
            ),
            focus_heavy_clash_overlap_sum_angstrom=(
                focus_clash_overlap_sum_angstrom(focus_heavy_clashes)
            ),
            focus_clash_overlap_sum_angstrom=focus_clash_overlap_sum_angstrom(
                focus_clashes
            ),
        ),
    )


def focus_clash_overlap_sum_angstrom(
    clashes: Iterable[StericClash],
) -> float:
    """Return total focus-clash overlap for one residue-local candidate."""

    return sum(clash.overlap_angstrom for clash in clashes)


def focus_fractional_clash_overlap_sum(
    clashes: Iterable[StericClash],
) -> float:
    """Return radius-normalized clash severity for residue-local candidate ordering."""

    return sum(
        clash.overlap_angstrom / (clash.distance_angstrom + clash.overlap_angstrom)
        for clash in clashes
    )


def _heavy_heavy_clashes(
    structure: ProteinStructure,
    clashes: Iterable[StericClash],
) -> tuple[StericClash, ...]:
    """Return only heavy-atom clashes for heavy-atom orientation scoring."""

    return tuple(
        clash
        for clash in clashes
        if _clash_endpoint_element(
            structure,
            residue_id=clash.left_residue_id,
            atom_name=clash.left_atom_name,
        )
        != "H"
        and _clash_endpoint_element(
            structure,
            residue_id=clash.right_residue_id,
            atom_name=clash.right_atom_name,
        )
        != "H"
    )


def _clash_endpoint_element(
    structure: ProteinStructure,
    *,
    residue_id: ResidueId,
    atom_name: str,
) -> str:
    """Return the element symbol for one clash endpoint."""

    return structure.constitution.atom_site_at(
        structure.constitution.atom_index_in_residue(
            structure.constitution.residue_index(residue_id),
            atom_name,
        )
    ).element

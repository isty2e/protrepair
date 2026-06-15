"""Branched side-chain pre-refinement seed generation."""

from itertools import product
from math import pi

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.scope import AtomSetScope
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.labels import ResidueId
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.atom_input import AtomInput
from protrepair.transformer.base import (
    BranchingContextOperation,
    TransformationCandidate,
)
from protrepair.transformer.context import ProteinTransformationContext
from protrepair.transformer.discrete.models import (
    BranchedSidechainCorrectionDefinition,
    BranchedSidechainRotationStep,
    BranchedSidechainSeedProvenance,
)
from protrepair.transformer.discrete.orientation import (
    replace_residue,
    rotate_residue_atoms_about_axis,
)

BRANCHED_SIDECHAIN_CORRECTION_DEFINITIONS: dict[
    str, BranchedSidechainCorrectionDefinition
] = {
    "LEU": BranchedSidechainCorrectionDefinition(
        component_id="LEU",
        rotation_steps=(
            BranchedSidechainRotationStep(
                axis_atom_names=("CA", "CB"),
                rotating_atom_names=("CG", "CD1", "CD2"),
                hydrogen_anchor_atom_names=("CG", "CD1", "CD2"),
                angle_degrees_options=(-120, 120, 180),
            ),
            BranchedSidechainRotationStep(
                axis_atom_names=("CB", "CG"),
                rotating_atom_names=("CD1", "CD2"),
                hydrogen_anchor_atom_names=("CD1", "CD2"),
                angle_degrees_options=(-120, 120, 180),
            ),
        ),
    ),
}


class BranchedSidechainSeedTransformer(
    BranchingContextOperation[
        ProteinStructureSnapshot, BranchedSidechainSeedProvenance
    ]
):
    """Branching transformer for branched-sidechain seed snapshots."""

    def __init__(self, component_library: ComponentLibrary) -> None:
        self._component_library = component_library

    def is_applicable(
        self,
        context: ProteinTransformationContext,
    ) -> bool:
        """Return whether one snapshot-domain pair admits seed expansion."""

        resolved_target = resolve_branched_sidechain_seed_target(
            context.source_snapshot,
            context.atom_input,
            component_library=self._component_library,
        )
        if resolved_target is None:
            return False

        residue_site, residue_geometry, residue_id, definition = resolved_target
        return selection_allows_branched_sidechain_correction(
            atom_input=context.atom_input,
            residue_site=residue_site,
            residue_geometry=residue_geometry,
            residue_id=residue_id,
            definition=definition,
        ) and definition.supports_residue(residue_site)

    def transform(
        self,
        context: ProteinTransformationContext,
    ) -> tuple[
        TransformationCandidate[
            ProteinStructureSnapshot,
            BranchedSidechainSeedProvenance,
        ],
        ...,
    ]:
        """Return branched-sidechain seed candidates for one snapshot-domain pair."""

        resolved_target = resolve_branched_sidechain_seed_target(
            context.source_snapshot,
            context.atom_input,
            component_library=self._component_library,
        )
        if resolved_target is None:
            return ()

        residue_site, residue_geometry, residue_id, definition = resolved_target
        if not selection_allows_branched_sidechain_correction(
            atom_input=context.atom_input,
            residue_site=residue_site,
            residue_geometry=residue_geometry,
            residue_id=residue_id,
            definition=definition,
        ):
            return ()

        if not definition.supports_residue(residue_site):
            return ()

        seed_candidates: list[
            TransformationCandidate[
                ProteinStructureSnapshot,
                BranchedSidechainSeedProvenance,
            ]
        ] = []
        for angle_degrees_by_step in product(
            *(
                (0, *rotation_step.angle_degrees_options)
                for rotation_step in definition.rotation_steps
            )
        ):
            if all(angle_degrees == 0 for angle_degrees in angle_degrees_by_step):
                continue

            seed_candidates.append(
                TransformationCandidate(
                    payload=build_branched_sidechain_seed_candidate(
                        snapshot=context.source_snapshot,
                        residue_site=residue_site,
                        residue_geometry=residue_geometry,
                        formal_charge_by_atom_name=(
                            context.source_snapshot.structure.topology.residue_formal_charge_by_atom_name(
                                constitution=context.source_snapshot.structure.constitution,
                                residue_index=context.source_snapshot.structure.constitution.residue_index(
                                    residue_id
                                ),
                            )
                        ),
                        definition=definition,
                        angle_degrees_by_step=angle_degrees_by_step,
                    ),
                    provenance=BranchedSidechainSeedProvenance(
                        residue_id=residue_id,
                        component_id=definition.component_id,
                        angle_degrees_by_step=angle_degrees_by_step,
                    ),
                )
            )

        return tuple(seed_candidates)


def resolve_branched_sidechain_seed_target(
    snapshot: ProteinStructureSnapshot,
    atom_input: AtomInput,
    *,
    component_library: ComponentLibrary,
) -> tuple[
    ResidueSite,
    ResidueGeometry,
    ResidueId,
    BranchedSidechainCorrectionDefinition,
] | None:
    """Return the single residue and correction definition for one seedable domain."""

    referenced_residue_ids = atom_input.referenced_residue_ids()
    if len(referenced_residue_ids) != 1:
        return None

    residue_id = referenced_residue_ids[0]
    residue_site = snapshot.structure.constitution.residue_or_ligand(residue_id)
    if residue_site is None:
        return None

    residue_geometry = snapshot.structure.residue_geometry(
        snapshot.structure.constitution.residue_index(residue_id)
    )

    component_id = component_library.normalize_component_id(residue_site.component_id)
    definition = BRANCHED_SIDECHAIN_CORRECTION_DEFINITIONS.get(component_id)
    if definition is None:
        return None

    return residue_site, residue_geometry, residue_id, definition


def selection_allows_branched_sidechain_correction(
    *,
    atom_input: AtomInput,
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    residue_id: ResidueId,
    definition: BranchedSidechainCorrectionDefinition,
) -> bool:
    """Return whether the current selection authorizes branched-sidechain seeding."""

    if atom_input.is_residuewise():
        return residue_id in atom_input.referenced_residue_ids()

    selected_scope = atom_input.as_scope()
    if not isinstance(selected_scope, AtomSetScope):
        raise TypeError(
            "atomwise branched-sidechain selection requires an AtomSetScope"
        )

    selected_atom_names = {
        atom_ref.atom_name
        for atom_ref in selected_scope.atom_refs
        if atom_ref.residue_id == residue_id
    }
    required_atom_names = definition.required_rotated_atom_names(
        residue_site=residue_site,
        residue_geometry=residue_geometry,
    )
    return required_atom_names.issubset(selected_atom_names)


def build_branched_sidechain_seed_candidate(
    *,
    snapshot: ProteinStructureSnapshot,
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    formal_charge_by_atom_name: tuple[tuple[str, int | None], ...],
    definition: BranchedSidechainCorrectionDefinition,
    angle_degrees_by_step: tuple[int, ...],
) -> ProteinStructureSnapshot:
    """Return one stepwise branched-sidechain seed candidate."""

    working_residue_site = residue_site
    working_residue_geometry = residue_geometry
    working_formal_charge_by_atom_name = formal_charge_by_atom_name
    for rotation_step, angle_degrees in zip(
        definition.rotation_steps, angle_degrees_by_step, strict=True
    ):
        if angle_degrees == 0:
            continue

        (
            updated_residue_site,
            updated_residue_geometry,
            updated_formal_charge_by_atom_name,
            rotated_atom_names,
        ) = rotate_residue_atoms_about_axis(
            residue_site=working_residue_site,
            residue_geometry=working_residue_geometry,
            formal_charge_by_atom_name=working_formal_charge_by_atom_name,
            axis_atom_names=rotation_step.axis_atom_names,
            rotated_atom_names=set(
                rotation_step.rotated_atom_names(
                    residue_site=working_residue_site,
                    residue_geometry=working_residue_geometry,
                )
            ),
            theta_radians=angle_degrees * (pi / 180.0),
        )
        del rotated_atom_names
        working_residue_site = updated_residue_site
        working_residue_geometry = updated_residue_geometry
        working_formal_charge_by_atom_name = updated_formal_charge_by_atom_name

    updated_structure = replace_residue(
        structure=snapshot.structure,
        residue_site=working_residue_site,
        residue_geometry=working_residue_geometry,
        formal_charge_by_atom_name=working_formal_charge_by_atom_name,
    )
    return snapshot.with_structure(updated_structure)

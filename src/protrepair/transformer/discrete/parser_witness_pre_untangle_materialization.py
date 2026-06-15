"""Side-chain rotation materialization for parser-witness pre-untangle."""

from collections.abc import Callable
from math import pi

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.diagnostics.parser_readability import RDKitProximityBondCluster
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.transformer.discrete.orientation import (
    attached_hydrogen_atom_names,
    rotate_residue_atoms_about_axis,
)
from protrepair.transformer.discrete.parser_witness_pre_untangle_scoring import (
    parser_witness_pre_untangle_score_from_context,
)
from protrepair.transformer.discrete.parser_witness_pre_untangle_types import (
    ParserWitnessPreUntangleCandidate,
    ParserWitnessScoringContext,
    RotatedSidechainPayload,
    SidechainRootRotationPlan,
)

BACKBONE_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})
SIDECHAIN_ROOT_AXIS_ATOM_NAMES = ("CA", "CB")


def deduplicated_sidechain_root_rotation_plans(
    structure: ProteinStructure,
    cluster: RDKitProximityBondCluster,
    *,
    component_library: ComponentLibrary,
) -> tuple[SidechainRootRotationPlan, ...]:
    """Return at most one CA-CB sidechain rotation plan per residue."""

    plans_by_residue_id: dict[ResidueId, SidechainRootRotationPlan] = {}
    for atom_ref in ordered_cluster_endpoint_atom_refs(cluster):
        if atom_ref.residue_id in plans_by_residue_id:
            continue

        plan = sidechain_root_rotation_plan(
            structure,
            atom_ref,
            component_library=component_library,
        )
        if plan is None:
            continue

        plans_by_residue_id[atom_ref.residue_id] = plan

    return tuple(plans_by_residue_id.values())


def ordered_cluster_endpoint_atom_refs(
    cluster: RDKitProximityBondCluster,
) -> tuple[AtomRef, ...]:
    """Return deduplicated witness endpoints in deterministic order."""

    return AtomRef.deduplicated(
        atom_ref
        for witness in cluster.bonds
        for atom_ref in (witness.atom_ref_1, witness.atom_ref_2)
    )


def sidechain_root_rotation_plan(
    structure: ProteinStructure,
    atom_ref: AtomRef,
    *,
    component_library: ComponentLibrary,
) -> SidechainRootRotationPlan | None:
    """Return a safe CA-CB side-chain rotation plan for one witness endpoint."""

    residue_site = structure.constitution.residue_or_ligand(atom_ref.residue_id)
    if residue_site is None or residue_site.is_hetero:
        return None

    if atom_ref.atom_name in BACKBONE_ATOM_NAMES:
        return None

    if not all(
        residue_site.has_atom_site(atom_name)
        for atom_name in SIDECHAIN_ROOT_AXIS_ATOM_NAMES
    ):
        return None

    residue_index = structure.constitution.residue_index(atom_ref.residue_id)
    residue_geometry = structure.residue_geometry(residue_index)
    template = component_library.get(residue_site.component_id)
    if template is None:
        return None

    rotating_atom_names = sidechain_root_rotating_atom_names(
        residue_site,
        residue_geometry,
        atom_ref=atom_ref,
        component_bonded_atom_names=template.definition.bonded_atom_names,
    )
    if not rotating_atom_names:
        return None

    return SidechainRootRotationPlan(
        residue_site=residue_site,
        residue_geometry=residue_geometry,
        residue_index=residue_index,
        axis_atom_names=SIDECHAIN_ROOT_AXIS_ATOM_NAMES,
        rotating_atom_names=rotating_atom_names,
        formal_charge_by_atom_name=structure.residue_formal_charge_by_atom_name(
            residue_index
        ),
    )


def sidechain_root_rotating_atom_names(
    residue_site: ResidueSite,
    residue_geometry: ResidueGeometry,
    *,
    atom_ref: AtomRef,
    component_bonded_atom_names: Callable[[str], frozenset[str]],
) -> frozenset[str]:
    """Return atoms that can rotate around CA-CB without moving backbone atoms."""

    present_atom_names = set(residue_site.atom_site_names())
    if not {"CA", "CB"} <= present_atom_names:
        return frozenset()

    reachable_from_cb = reachable_component_atom_names_after_cut(
        start_atom_name="CB",
        blocked_atom_name="CA",
        present_atom_names=present_atom_names,
        component_bonded_atom_names=component_bonded_atom_names,
    )
    rotating_heavy_atom_names = {
        atom_name
        for atom_name in reachable_from_cb
        if atom_name not in {"CA", "CB"}
        and residue_site.atom_site(atom_name).element != "H"
    }
    if atom_ref.atom_name not in rotating_heavy_atom_names:
        return frozenset()

    if rotating_heavy_atom_names & BACKBONE_ATOM_NAMES:
        return frozenset()

    attached_hydrogens = set(
        attached_hydrogen_atom_names(
            residue_site=residue_site,
            residue_geometry=residue_geometry,
            anchor_atom_names=tuple(sorted(rotating_heavy_atom_names)),
        )
    )
    rotating_atom_names = rotating_heavy_atom_names | attached_hydrogens
    if any(
        not residue_geometry.has_atom(atom_name) for atom_name in rotating_atom_names
    ):
        return frozenset()

    return frozenset(rotating_atom_names)


def reachable_component_atom_names_after_cut(
    *,
    start_atom_name: str,
    blocked_atom_name: str,
    present_atom_names: set[str],
    component_bonded_atom_names: Callable[[str], frozenset[str]],
) -> set[str]:
    """Return present component atoms reachable after cutting one graph edge."""

    reachable: set[str] = set()
    stack = [start_atom_name]
    while stack:
        atom_name = stack.pop()
        if atom_name in reachable or atom_name not in present_atom_names:
            continue

        reachable.add(atom_name)
        for neighbor_atom_name in component_bonded_atom_names(atom_name):
            if atom_name == start_atom_name and neighbor_atom_name == blocked_atom_name:
                continue
            if neighbor_atom_name in reachable:
                continue

            stack.append(neighbor_atom_name)

    return reachable


def build_rotated_sidechain_payload(
    scoring_context: ParserWitnessScoringContext,
    plan: SidechainRootRotationPlan,
    *,
    angle_degrees: int,
) -> RotatedSidechainPayload:
    """Return one parser-witness payload after a bounded side-chain rotation."""

    (
        _updated_residue_site,
        updated_residue_geometry,
        _updated_formal_charge_by_atom_name,
        moved_atom_names,
    ) = rotate_residue_atoms_about_axis(
        residue_site=plan.residue_site,
        residue_geometry=plan.residue_geometry,
        formal_charge_by_atom_name=plan.formal_charge_by_atom_name,
        axis_atom_names=plan.axis_atom_names,
        rotated_atom_names=set(plan.rotating_atom_names),
        theta_radians=angle_degrees * (pi / 180.0),
    )
    return RotatedSidechainPayload(
        plan=plan,
        angle_degrees=angle_degrees,
        residue_geometry=updated_residue_geometry,
        moved_atom_names=moved_atom_names,
        score=parser_witness_pre_untangle_score_from_context(
            scoring_context,
            residue_id=plan.residue_site.residue_id,
            residue_geometry=updated_residue_geometry,
        ),
    )


def materialize_rotated_sidechain_candidate(
    structure: ProteinStructure,
    payload: RotatedSidechainPayload,
) -> ParserWitnessPreUntangleCandidate:
    """Return one fully materialized parser-witness pre-untangle candidate."""

    updated_structure = structure.with_updated_residue_geometries(
        (
            (
                payload.plan.residue_site.residue_id,
                payload.residue_geometry,
            ),
        )
    )
    moved_atom_indices = tuple(
        structure.constitution.atom_index_in_residue(
            payload.plan.residue_index,
            atom_name,
        )
        for atom_name in payload.moved_atom_names
        if (
            structure.geometry.position(
                structure.constitution.atom_index_in_residue(
                    payload.plan.residue_index,
                    atom_name,
                )
            )
            != updated_structure.geometry.position(
                structure.constitution.atom_index_in_residue(
                    payload.plan.residue_index,
                    atom_name,
                )
            )
        )
    )
    return ParserWitnessPreUntangleCandidate(
        structure=updated_structure,
        moved_atom_indices=moved_atom_indices,
        score=payload.score,
    )

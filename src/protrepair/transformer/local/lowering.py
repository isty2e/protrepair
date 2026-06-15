"""Lower canonical local scope specs into atom-local realized inputs."""

from collections.abc import Mapping

from protrepair.chemistry import ComponentLibrary, build_default_component_library
from protrepair.scope import AtomSetScope, ResidueSetScope
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.transformer.atom_input import (
    AtomInput,
    AtomInputBasis,
    AtomInputRealization,
    structure_ordered_atom_refs,
)
from protrepair.transformer.local.models import (
    LocalScopeLowering,
    LocalScopeSpec,
)

BACKBONE_HEAVY_ATOM_NAMES = frozenset({"N", "CA", "C", "O", "OXT"})
HYDROGEN_ATTACHMENT_DISTANCE_MAX_ANGSTROM = 1.35


def atom_input_from_local_scope_spec(
    snapshot: ProteinStructureSnapshot,
    scope_spec: LocalScopeSpec,
    *,
    component_library: ComponentLibrary | None = None,
) -> AtomInput:
    """Lower one canonical local scope request into one atom domain."""

    ordered_atom_refs = structure_ordered_atom_refs(snapshot)
    ordered_atom_indices = tuple(
        snapshot.structure.constitution.atom_index(atom_ref)
        for atom_ref in ordered_atom_refs
    )
    if scope_spec.is_atomwise():
        atom_scope = scope_spec.scope
        if not isinstance(atom_scope, AtomSetScope):
            raise TypeError(
                "atom-local scope lowering requires an AtomSetScope"
            )
        selected_atom_ref_set = _selected_atom_ref_set(
            snapshot,
            atom_scope.atom_refs,
        )
        if scope_spec.lowering is LocalScopeLowering.ATTACHED_PRESENT_HYDROGENS:
            selected_atom_ref_set |= _attached_hydrogen_atom_refs(
                snapshot,
                scope_spec,
                component_library=component_library,
            )

        return AtomInput(
            atom_indices=_ordered_selected_atom_indices(
                ordered_atom_refs,
                ordered_atom_indices,
                selected_atom_ref_set,
            ),
            basis=AtomInputBasis.ATOMWISE,
            realization=AtomInputRealization.EXACT_ATOMS,
            selected_scope=AtomSetScope(
                atom_refs=_ordered_selected_atom_refs(
                    ordered_atom_refs,
                    selected_atom_ref_set,
                )
            ),
        )

    residue_scope = scope_spec.scope
    if not isinstance(residue_scope, ResidueSetScope):
        raise TypeError(
            "residue-local scope lowering requires a ResidueSetScope"
        )
    selected_residue_id_set = set(residue_scope.residue_ids)
    for residue_id in selected_residue_id_set:
        if snapshot.structure.constitution.residue_or_ligand(residue_id) is None:
            raise ValueError(
                "local scope references unknown residue "
                f"{residue_id.display_token()}"
            )

    if scope_spec.lowering is LocalScopeLowering.RESIDUE_SIDECHAIN_ATOMS:
        active_component_library = (
            build_default_component_library()
            if component_library is None
            else component_library
        )
        selected_atom_ref_set = _selected_residue_sidechain_atom_ref_set(
            snapshot,
            residue_ids=residue_scope.residue_ids,
            component_library=active_component_library,
        )
        return AtomInput(
            atom_indices=_ordered_selected_atom_indices(
                ordered_atom_refs,
                ordered_atom_indices,
                selected_atom_ref_set,
            ),
            basis=AtomInputBasis.RESIDUEWISE,
            realization=AtomInputRealization.RESIDUE_SIDECHAIN_ATOMS,
            selected_scope=ResidueSetScope(residue_ids=residue_scope.residue_ids),
        )

    return AtomInput(
        atom_indices=tuple(
            atom_index
            for atom_ref, atom_index in zip(
                ordered_atom_refs,
                ordered_atom_indices,
                strict=True,
            )
            if atom_ref.residue_id in selected_residue_id_set
        ),
        basis=AtomInputBasis.RESIDUEWISE,
        realization=AtomInputRealization.RESIDUE_ATOMS,
        selected_scope=ResidueSetScope(residue_ids=residue_scope.residue_ids),
    )


def _selected_atom_ref_set(
    snapshot: ProteinStructureSnapshot,
    atom_refs: tuple[AtomRef, ...],
) -> set[AtomRef]:
    """Return one validated atom-ref set from exact local scope atoms."""

    selected_atom_ref_set = set(atom_refs)
    for atom_ref in selected_atom_ref_set:
        if snapshot.structure.constitution.resolve_atom_site(atom_ref) is None:
            raise ValueError(
                "local scope references unknown atom "
                f"{atom_ref.display_token()}"
            )

    return selected_atom_ref_set


def _ordered_selected_atom_refs(
    ordered_atom_refs: tuple[AtomRef, ...],
    selected_atom_ref_set: set[AtomRef],
) -> tuple[AtomRef, ...]:
    """Return selected atom refs in canonical structure order."""

    return tuple(
        atom_ref
        for atom_ref in ordered_atom_refs
        if atom_ref in selected_atom_ref_set
    )


def _ordered_selected_atom_indices(
    ordered_atom_refs: tuple[AtomRef, ...],
    ordered_atom_indices: tuple[AtomIndex, ...],
    selected_atom_ref_set: set[AtomRef],
) -> tuple[AtomIndex, ...]:
    """Return selected atom indices in canonical structure order."""

    return tuple(
        atom_index
        for atom_ref, atom_index in zip(
            ordered_atom_refs,
            ordered_atom_indices,
            strict=True,
        )
        if atom_ref in selected_atom_ref_set
    )


def _attached_hydrogen_atom_refs(
    snapshot: ProteinStructureSnapshot,
    scope_spec: LocalScopeSpec,
    *,
    component_library: ComponentLibrary | None,
) -> set[AtomRef]:
    """Return present template-attached hydrogens for one focused atom scope."""

    active_component_library = (
        build_default_component_library()
        if component_library is None
        else component_library
    )
    attached_hydrogen_atom_refs: set[AtomRef] = set()
    atom_scope = scope_spec.scope
    if not isinstance(atom_scope, AtomSetScope):
        raise TypeError(
            "attached-hydrogen local scope requests require an AtomSetScope"
        )
    for residue_id in scope_spec.referenced_residue_ids():
        residue = snapshot.structure.constitution.residue_or_ligand(residue_id)
        if residue is None:
            raise ValueError(
                "local scope references unknown residue "
                f"{residue_id.display_token()}"
            )
        template = active_component_library.get(residue.component_id)
        if template is None:
            raise ValueError(
                "attached-hydrogen local scope requests require a component "
                "template "
                f"for {residue.component_id!r}"
            )

        residue_focus_atom_names = {
            atom_ref.atom_name
            for atom_ref in atom_scope.atom_refs
            if atom_ref.residue_id == residue_id
        }
        for hydrogen_atom_name, anchor_atom_name in (
            template.template_hydrogen_anchor_by_name(
                residue.atom_site_names()
            ).items()
        ):
            if anchor_atom_name not in residue_focus_atom_names:
                continue

            attached_hydrogen_atom_refs.add(
                AtomRef(residue_id=residue_id, atom_name=hydrogen_atom_name)
            )

    return attached_hydrogen_atom_refs


def _selected_residue_sidechain_atom_ref_set(
    snapshot: ProteinStructureSnapshot,
    *,
    residue_ids: tuple[ResidueId, ...],
    component_library: ComponentLibrary,
) -> set[AtomRef]:
    """Return sidechain-local atom refs for residuewise local correction."""

    selected_atom_ref_set: set[AtomRef] = set()
    for residue_id in residue_ids:
        residue = snapshot.structure.constitution.residue_or_ligand(residue_id)
        if residue is None:
            raise ValueError(
                "local scope references unknown residue "
                f"{residue_id.display_token()}"
            )
        template = component_library.get(residue.component_id)
        selected_atom_names = _selected_residue_sidechain_atom_names(
            snapshot,
            residue,
            template_hydrogen_anchor_by_name=(
                {}
                if template is None
                else template.template_hydrogen_anchor_by_name(
                    residue.atom_site_names()
                )
            ),
        )
        selected_atom_ref_set.update(
            AtomRef(residue_id=residue_id, atom_name=atom_name)
            for atom_name in selected_atom_names
        )

    return selected_atom_ref_set


def _selected_residue_sidechain_atom_names(
    snapshot: ProteinStructureSnapshot,
    residue: ResidueSite,
    *,
    template_hydrogen_anchor_by_name: Mapping[str, str],
) -> tuple[str, ...]:
    """Return sidechain-local movable atom names for one residue selection."""

    explicit_anchor_by_name = dict(template_hydrogen_anchor_by_name)
    present_heavy_atom_names = {
        atom_site.name
        for atom_site in residue.atom_sites
        if atom_site.element != "H"
    }
    sidechain_heavy_atom_names = {
        atom_name
        for atom_name in present_heavy_atom_names
        if atom_name not in BACKBONE_HEAVY_ATOM_NAMES
    }
    if not sidechain_heavy_atom_names:
        return residue.atom_site_names()

    selected_atom_names = set(sidechain_heavy_atom_names)
    for atom_site in residue.atom_sites:
        if atom_site.element != "H":
            continue

        anchor_atom_name = explicit_anchor_by_name.get(atom_site.name)
        if anchor_atom_name is None:
            anchor_atom_name = _nearest_hydrogen_anchor_atom_name(
                snapshot,
                residue=residue,
                hydrogen_atom_name=atom_site.name,
                heavy_atom_names=tuple(present_heavy_atom_names),
            )
        if anchor_atom_name in sidechain_heavy_atom_names:
            selected_atom_names.add(atom_site.name)

    return tuple(
        atom_name
        for atom_name in residue.atom_site_names()
        if atom_name in selected_atom_names
    )


def _nearest_hydrogen_anchor_atom_name(
    snapshot: ProteinStructureSnapshot,
    *,
    residue: ResidueSite,
    hydrogen_atom_name: str,
    heavy_atom_names: tuple[str, ...],
) -> str | None:
    """Return the nearest heavy-atom anchor for one present hydrogen."""

    residue_geometry = snapshot.structure.residue_geometry(
        snapshot.structure.constitution.residue_index(residue.residue_id)
    )
    hydrogen_atom_geometry = residue_geometry.atom_geometry(hydrogen_atom_name)
    nearest_atom_name: str | None = None
    nearest_distance = float("inf")
    for heavy_atom_name in heavy_atom_names:
        pair_distance = hydrogen_atom_geometry.distance_to(
            residue_geometry.atom_geometry(heavy_atom_name)
        )
        if pair_distance > HYDROGEN_ATTACHMENT_DISTANCE_MAX_ANGSTROM:
            continue
        if pair_distance < nearest_distance:
            nearest_atom_name = heavy_atom_name
            nearest_distance = pair_distance

    return nearest_atom_name

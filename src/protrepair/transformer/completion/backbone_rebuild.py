"""Rebuild partial selected backbones before downstream geometry consumers run."""

from collections import defaultdict
from dataclasses import dataclass, replace
from itertools import product
from math import isfinite

import numpy as np

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.retained_non_polymer.hydrogen_correspondence import (
    source_hydrogen_bond_length_is_compatible,
)
from protrepair.chemistry.standard.components import STANDARD_COMPONENT_ATOMS
from protrepair.diagnostics.events import RepairEvent, ValidationIssue
from protrepair.diagnostics.kinds import (
    IssueSeverity,
    RepairEventKind,
    ValidationIssueKind,
)
from protrepair.geometry import GeometryPlacementError, TetrahedralCenter, Vec3
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.backbone_availability import (
    BackboneAvailability,
    BackbonePlacement,
)
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import (
    WithheldBackboneHydrogen,
    ZeroOccupancyOmission,
)
from protrepair.structure.source_connection import SourceAtomIdentity
from protrepair.transformer.completion.backbone_environment import BackboneEnvironment
from protrepair.transformer.completion.backbone_fit import BackboneFit
from protrepair.transformer.result import TransformationResult

_BACKBONE = frozenset(("N", "CA", "C", "O"))
_FILE_PRECISION = 0.002


@dataclass(slots=True)
class _Candidate:
    site: ResidueSite
    names: frozenset[str]
    changes: dict[AtomRef, Vec3]
    anchors: dict[AtomRef, Vec3]
    withheld: dict[str, AtomRef]


def _propose(
    site: ResidueSite,
    names: frozenset[str],
    env: BackboneEnvironment,
    template: ResidueTemplate,
) -> _Candidate:
    s = env.structure
    c = s.constitution
    rid = site.residue_id
    if site.component_id not in STANDARD_COMPONENT_ATOMS or any(
        AtomRef(rid, k) in env.protected for k in names
    ):
        raise GeometryPlacementError("unsupported or externally bound backbone site")
    anchors: dict[str, Vec3] = {}
    source_points: dict[AtomRef, Vec3] = {}
    before, after = env.left.get(rid), env.right.get(rid)
    if after is not None and after.component_id not in STANDARD_COMPONENT_ATOMS:
        raise GeometryPlacementError("unverified modified amide anchor")
    if (
        "CA" in names
        and before is not None
        and before.component_id not in STANDARD_COMPONENT_ATOMS
    ):
        raise GeometryPlacementError("unverified modified carbonyl anchor")
    for prefix, neighbor, keys in (
        ("", site, ("N", "CA", "C", "O", "CB", "CD")),
        ("-", before, ("CA", "C", "O")),
        ("+", after, ("N", "CA", "CD")),
    ):
        if neighbor is None:
            continue
        if prefix == "-" and names != {"CA", "C"}:
            continue
        for k in keys:
            if k == "CD" and neighbor.component_id != "PRO":
                continue
            ref = AtomRef(neighbor.residue_id, k)
            idx = c.resolve_atom_index(ref)
            if idx is None or c.atom_site_at(idx).element != k[0]:
                continue
            g = s.geometry.atom_geometry(idx)
            if (
                g.occupancy > 0
                and isfinite(g.occupancy)
                and all(isfinite(x) for x in g.position)
            ):
                anchors[prefix + k] = g.position
                source_points[ref] = g.position
    if any(site.atom_site(k).element != k[0] for k in names):
        raise GeometryPlacementError("backbone atom identity mismatch")
    if names == {"C"} and (
        "N" not in anchors or (site.component_id != "GLY" and "CB" not in anchors)
    ):
        raise GeometryPlacementError("missing anchors for alpha-carbon geometry")
    out = BackboneFit(
        anchors, site.component_id, after.component_id if after else ""
    ).place(names)
    changes = {AtomRef(rid, k): point for k, point in out.items()}
    candidate = _Candidate(site, names, changes, source_points, {})
    _attached_hydrogens(candidate, env, template, {**anchors, **out})
    for ref, position in changes.items():
        env.check_contacts(ref, position)
    return candidate


def _attached_hydrogens(
    candidate: _Candidate,
    env: BackboneEnvironment,
    template: ResidueTemplate,
    points: dict[str, Vec3],
) -> None:
    site = candidate.site
    s = env.structure
    aliases = template.polymer_hydrogen_parent_aliases()
    alpha_slots = [
        PolymerMicrostateSite.hydrogen_slot_name(a)
        for a in site.atom_sites
        if a.is_hydrogen()
        and aliases.get(PolymerMicrostateSite.hydrogen_slot_name(a)) == "CA"
    ]
    if "CA" in candidate.names and (
        len(alpha_slots) != len(set(alpha_slots))
        or len(alpha_slots) > (2 if site.component_id == "GLY" else 1)
    ):
        raise GeometryPlacementError(
            "multiple hydrogens claim the same alpha-carbon vertex"
        )
    gly_vertices: dict[str, int] = {}
    if site.component_id == "GLY" and "CA" in candidate.names:
        labels = set(alpha_slots) & {"HA1", "HA2", "HA3", "1HA", "2HA"}
        if labels & {"1HA", "2HA"}:
            gly_vertices = {"1HA": 0, "2HA": 1}
        elif "HA1" in labels:
            gly_vertices = {"HA1": 0, "HA2": 1}
        elif "HA3" in labels:
            gly_vertices = {"HA2": 0, "HA3": 1}
        if gly_vertices and not labels <= gly_vertices.keys():
            raise GeometryPlacementError("ambiguous glycine alpha-hydrogen names")
    for atom in site.atom_sites:
        if not atom.is_hydrogen():
            continue
        ref = AtomRef(site.residue_id, atom.name)
        slot = PolymerMicrostateSite.hydrogen_slot_name(atom)
        parents = env.neighbors.get(ref, set())
        alias = aliases.get(slot)
        if len(parents) > 1 or (
            parents and alias and AtomRef(site.residue_id, alias) not in parents
        ):
            raise GeometryPlacementError("ambiguous hydrogen attachment")
        parent = next(iter(parents)).atom_name if parents else alias
        if parent is None:
            raise GeometryPlacementError("unassigned hydrogen attachment")
        env.connect(ref, AtomRef(site.residue_id, parent))
        if parent not in candidate.names:
            continue
        idx = s.constitution.atom_index(ref)
        g = s.geometry.atom_geometry(idx)
        if parent == "C":
            raise GeometryPlacementError("hydrogen attached to the carbonyl carbon")
        if g.occupancy > 0:
            if not source_hydrogen_bond_length_is_compatible(
                g.position.distance_to(points[parent]), 1.09
            ):
                raise GeometryPlacementError("incompatible retained " + atom.name)
            # A retained H on the CB side of CA can have the right bond length
            # while occupying the wrong tetrahedral vertex.
            if parent == "CA":
                ray = g.position.to_array() - points[parent].to_array()
                if any(
                    float(np.dot(ray, points[k].to_array() - points[parent].to_array()))
                    >= 0
                    for k in ("N", "C", "CB")
                    if k in points
                ):
                    raise GeometryPlacementError(
                        "incompatible retained hydrogen direction"
                    )
            candidate.anchors[ref] = g.position
        elif g.occupancy != 0 or ref in env.protected:
            raise GeometryPlacementError("unusable hydrogen with declared attachment")
        elif parent == "CA" and {"N", "C", "CB"} <= points.keys() and slot == "HA":
            ca = points["CA"].to_array()
            directions = np.array([points[k].to_array() - ca for k in ("N", "C", "CB")])
            ray = -np.sum(
                directions / np.linalg.norm(directions, axis=1)[:, None], axis=0
            )
            norm = np.linalg.norm(ray)
            if norm < 1e-6:
                raise GeometryPlacementError("undetermined alpha hydrogen")
            candidate.changes[ref] = Vec3.coerce(ca + 1.09 * ray / norm)
        elif parent == "CA" and slot in gly_vertices:
            pair = TetrahedralCenter(points["C"], points["N"], points["CA"]).pair()
            candidate.changes[ref] = pair[gly_vertices[slot]]
        else:
            obs = s.provenance.ingress.observation
            if s.topology.formal_charge(idx) is not None:
                raise GeometryPlacementError(
                    "cannot withhold hydrogen with declared charge: " + atom.name
                )
            if obs and obs.bonds_for_atom(ref):
                raise GeometryPlacementError(
                    "cannot withhold explicitly bonded " + atom.name
                )
            candidate.withheld[atom.name] = AtomRef(site.residue_id, parent)


def rebuild_selected_backbones(
    structure: ProteinStructure, library: ComponentLibrary
) -> tuple[TransformationResult, BackboneAvailability | None]:
    """Stage every proposal on the unchanged carrier; then publish disjoint writes."""
    c = structure.constitution
    selections: dict[ResidueId, frozenset[str]] = {}
    for site in c.iter_residues():
        g = structure.residue_geometry(c.residue_index(site.residue_id))
        names = frozenset(
            k for k in _BACKBONE if g.has_atom(k) and g.atom_geometry(k).occupancy == 0
        )
        if names and any(
            not a.is_hydrogen()
            and isfinite(g.atom_geometry(a.name).occupancy)
            and g.atom_geometry(a.name).occupancy > 0
            for a in site.atom_sites
        ):
            selections[site.residue_id] = names
    if not selections:
        return TransformationResult(structure=structure, repairs=(), issues=()), None

    env = BackboneEnvironment(structure)
    failures: dict[ResidueId, str] = {}
    candidates: dict[ResidueId, _Candidate] = {}
    for left, right in env.right.items():
        if selections.get(left, frozenset()) & {"CA", "C", "O"} and selections.get(
            right.residue_id, frozenset()
        ) & {"N", "CA"}:
            failures[left] = failures[right.residue_id] = (
                "unsupported coupled cross-junction mask"
            )
    for rid, names in selections.items():
        site = c.residue_site_at(c.residue_index(rid))
        template = library.get(site.component_id)
        if rid in failures:
            continue
        if names == {"O"}:
            failures[rid] = (
                "isolated carbonyl oxygen placement has not met the positional error bound"
            )
            continue
        if template is None:
            failures[rid] = "no component template"
            continue
        try:
            candidates[rid] = _propose(site, names, env, template)
        except (GeometryPlacementError, np.linalg.LinAlgError) as error:
            failures[rid] = str(error)

    proposed_cells: dict[tuple[int, ...], list[tuple[AtomRef, Vec3]]] = defaultdict(
        list
    )
    for candidate in candidates.values():
        for ref, point in candidate.changes.items():
            local, cell = env.local_atoms(ref), env.cell(point)
            for offset in product((-1, 0, 1), repeat=3):
                key = tuple(a + b for a, b in zip(cell, offset, strict=True))
                for other, q in proposed_cells.get(key, ()):
                    if (
                        other.residue_id != ref.residue_id
                        and other not in local
                        and env.clash(ref, point, other, q)
                    ):
                        failures[ref.residue_id] = failures[other.residue_id] = (
                            "colliding backbone placements"
                        )
            proposed_cells[cell].append((ref, point))

    records = {
        r.source_identity: r for r in structure.provenance.original_backbone_coordinates
    }
    hydrogen_history = {
        r.original.source_identity: r
        for r in structure.provenance.withheld_backbone_hydrogens
    }
    events: list[RepairEvent] = []
    issues: list[ValidationIssue] = []
    placements: dict[ResidueId, BackbonePlacement] = {}
    facets = []
    for rid, names in selections.items():
        idx = c.residue_index(rid)
        site = c.residue_site_at(idx)
        if rid in failures:
            placements[rid] = BackbonePlacement(
                site.component_id, names, {}, failures[rid]
            )
            issues.append(
                ValidationIssue.for_residue(
                    kind=ValidationIssueKind.BACKBONE_REBUILD_REFUSED,
                    severity=IssueSeverity.WARNING,
                    residue_id=rid,
                    component_id=site.component_id,
                    atom_names=tuple(sorted(names)),
                    message=failures[rid],
                )
            )
            continue
        cand = candidates[rid]
        changes = cand.changes
        placements[rid] = BackbonePlacement(
            site.component_id, names, {**cand.anchors, **changes}, None
        )
        old = structure.residue_geometry(idx)
        moved = {
            ref.atom_name: old.atom_geometry(ref.atom_name).with_position(p)
            for ref, p in changes.items()
            if p.distance_to(old.position(ref.atom_name)) > _FILE_PRECISION
        }
        if not moved and not cand.withheld:
            continue
        for k in (*moved, *cand.withheld):
            g = old.atom_geometry(k)
            identity = SourceAtomIdentity(AtomRef(rid, k), site.component_id, g.altloc)
            records.setdefault(
                identity,
                ZeroOccupancyOmission(identity, site.atom_site(k).element_identity, g),
            )
            if k in cand.withheld:
                parent = cand.withheld[k]
                bond = structure.topology.bond_between(
                    c.atom_index(identity.atom_ref), c.atom_index(parent)
                )
                hydrogen_history.setdefault(
                    identity,
                    WithheldBackboneHydrogen(
                        records[identity],
                        parent,
                        None if bond is None else bond.provenance,
                    ),
                )
        after = old.with_atom_geometries(moved.items())
        facets.append(
            (
                site.without_atom_sites(cand.withheld),
                after.without_atoms(cand.withheld),
                tuple(
                    (k, v)
                    for k, v in structure.residue_formal_charge_by_atom_name(idx)
                    if k not in cand.withheld
                ),
            )
        )
        if moved:
            events.append(
                RepairEvent.for_residue(
                    kind=RepairEventKind.BACKBONE_ATOMS_REBUILT,
                    residue_id=rid,
                    component_id=site.component_id,
                    atom_names=tuple(sorted(moved)),
                )
            )
        if cand.withheld:
            events.append(
                RepairEvent.for_residue(
                    kind=RepairEventKind.HYDROGENS_REMOVED,
                    residue_id=rid,
                    component_id=site.component_id,
                    atom_names=tuple(sorted(cand.withheld)),
                    details="Undetermined zero-occupancy dependent hydrogen",
                )
            )
    result = structure.with_updated_residue_facets_batch(facets)
    if events:
        result = result.with_provenance(
            replace(
                result.provenance,
                original_backbone_coordinates=tuple(records.values()),
                withheld_backbone_hydrogens=tuple(hydrogen_history.values()),
            )
        )
    return TransformationResult(
        structure=result, repairs=tuple(events), issues=tuple(issues)
    ), BackboneAvailability(result, placements)

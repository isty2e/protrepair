"""Atomic assembly of an internal cap and the linked peptide flank."""

from dataclasses import dataclass

import gemmi

from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.peptide_caps import observe_linked_peptide_cap
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import AtomSite, ResidueRole, ResidueSite
from protrepair.structure.geometry import AtomGeometry, ResidueGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.topology import (
    BondProvenance,
    StructureTopology,
    TopologyBond,
)
from protrepair.structure.topology import BondRelationshipType as Rel
from protrepair.transformer.completion.shared.peptide_attachment import (
    amide_hydrogens,
    check_unbonded_carbonyl_h,
    neighbors,
    require_preserved_amide_hydrogens,
)
from protrepair.transformer.completion.terminal.cap_geometry import (
    PeptideCapFrame,
    PeptideCapPose,
)
from protrepair.transformer.completion.terminal.cap_sequence import linking_component


def _flank(
    snapshot: ProteinStructure | StructureObservation,
    port: AtomRef,
    *,
    cap: ResidueId | None = None,
) -> tuple[str, ...]:
    c, t = snapshot.constitution, snapshot.topology
    rid, side = port.residue_id, port.atom_name
    residue = c.residue_or_ligand(rid)
    if (
        residue is None
        or c.role_at(c.residue_index(rid)) is not ResidueRole.POLYMER
        or not linking_component(residue.component_id, side)
    ):
        raise ValueError("unsupported_flank")
    if side == "C" and any(
        residue.has_atom_site(a) for a in ("OXT", "OT1", "OT2", "O1", "O2")
    ):
        raise ValueError("existing_terminal_oxygen")
    if side == "C":
        check_unbonded_carbonyl_h(snapshot, residue)

    template = build_default_component_library().get(residue.component_id)
    assert template is not None
    heavy = set(template.expected_heavy_atom_names())
    if any(not a.is_hydrogen() and a.name not in heavy for a in residue.atom_sites):
        raise ValueError("unqualified_flank_heavy_atom")
    ring = "CD" in template.definition.bonded_atom_names("N")
    hs = amide_hydrogens(snapshot, residue, template) if side == "N" else ()
    if len(hs) > (0 if ring else 1):
        raise ValueError("surplus_terminal_h")
    wanted = (
        {"N": {"CA": 1, **({"CD": 1} if ring else {}), **dict.fromkeys(hs, 1)}}
        if side == "N"
        else {"C": {"CA": 1, "O": 2}, "O": {"C": 2}}
    )
    for name, allowed in wanted.items():
        ref = AtomRef(rid, name)
        idx = c.resolve_atom_index(ref)
        if idx is None:
            continue
        if c.atom_site_at(idx).element != name or t.formal_charge(idx) not in (None, 0):
            raise ValueError("incompatible_flank_element_or_charge")
        for other, bond in neighbors(snapshot, ref):
            if (
                cap is not None
                and name == side
                and other == AtomRef(cap, "C" if side == "N" else "N")
            ):
                continue
            if other.residue_id != rid:
                if (
                    name == "O"
                    and bond.relationship_type is Rel.METAL_COORDINATION
                    and gemmi.Element(
                        c.atom_site_at(c.atom_index(other)).element
                    ).is_metal
                ):
                    continue
                raise ValueError("occupied_or_coordinated_flank")
            if (
                other.atom_name not in allowed
                or bond.relationship_type not in (Rel.COVALENT, Rel.UNKNOWN)
                or bond.order not in (None, allowed[other.atom_name])
                or bond.aromatic
            ):
                raise ValueError("incompatible_flank_bond")
    return hs


def frame_for(
    parent: ProteinStructure, port: AtomRef, *, cap: ResidueId | None = None
) -> PeptideCapFrame:
    """Resolve a fixed flank frame against the original and current chemistry."""
    c = parent.constitution
    site = c.residue_or_ligand(port.residue_id)
    obs = parent.provenance.ingress.observation
    if site is None or obs is None:
        raise ValueError("absent_or_unobserved_flank")
    original = obs.constitution.residue_or_ligand(port.residue_id)
    if original is None or original.component_id != site.component_id:
        raise ValueError("changed_flank_identity")
    original_h = _flank(obs, port, cap=cap)
    hs = _flank(parent, port, cap=cap)
    require_preserved_amide_hydrogens(parent, obs, port, original_h, hs)
    template = build_default_component_library().get(site.component_id)
    assert template is not None
    ring = port.atom_name == "N" and "CD" in template.definition.bonded_atom_names("N")
    n_sub = "CD" if ring else next(iter(hs), None)
    required = {"CA", "C", "O"} if port.atom_name == "C" else {"N", "CA", "C"}
    if n_sub is not None:
        required.add(n_sub)
    points = {}
    for name in required:
        ref = AtomRef(port.residue_id, name)
        idx = c.resolve_atom_index(ref)
        if idx is None:
            raise ValueError("missing_heavy_anchor")
        if (name in ("CA", "C", "CD") and c.atom_site_at(idx).element != "C") or (
            name in ("N", "O") and c.atom_site_at(idx).element != name
        ):
            raise ValueError("wrong_anchor_element")
        if parent.geometry.atom_geometry(idx).occupancy <= 0:
            raise ValueError("unoccupied_anchor")
        points[name] = parent.geometry.position(idx)
    return PeptideCapFrame(port.atom_name, points, n_sub)


def occupied_cap(
    parent: ProteinStructure, port: AtomRef
) -> tuple[str, ResidueId] | None:
    """Verify both ends of an existing cap or identify a partial product."""
    for other, _bond in neighbors(parent, port):
        if other.residue_id == port.residue_id:
            continue
        partner = parent.constitution.residue_or_ligand(other.residue_id)
        if (
            partner is None
            or partner.component_id not in ("ACE", "NME")
            or parent.constitution.role_at(
                parent.constitution.residue_index(other.residue_id)
            )
            is not ResidueRole.LIGAND
        ):
            continue
        cap = observe_linked_peptide_cap(parent, partner)
        if cap is None or cap.partner != port or not cap.hydrogens_are_complete():
            return "partial_or_incompatible_cap", other.residue_id
        frame = frame_for(parent, port, cap=other.residue_id)
        if port.atom_name == "N" and frame.n_substituent is None:
            return "partial_or_incompatible_flank", other.residue_id
        coords = parent.residue_geometry(
            parent.constitution.residue_index(other.residue_id)
        )
        assert cap.methyl_atom is not None
        methyl = cap.methyl_atom.atom_name
        q = {"CH3" if port.atom_name == "N" else "C": coords.position(methyl)}
        anchor = "C" if port.atom_name == "N" else "N"
        q[anchor] = coords.position(anchor)
        if anchor == "C":
            q["O"] = coords.position("O")
        methyl_h = sorted(h for h, a in cap.existing_hydrogen_parents if a == methyl)
        if len(methyl_h) != 3:
            return "partial_or_incompatible_cap", other.residue_id
        for i, h in enumerate(methyl_h, 1):
            q[f"H{i}"] = coords.position(h)
        if anchor == "N":
            q["HN2"] = coords.position(
                next(h for h, a in cap.existing_hydrogen_parents if a == "N")
            )
        frame.verify(PeptideCapPose(q, None, 0, 0), trans=False)
        required = (
            ((port.atom_name, "CA", 1), ("C", "O", 2))
            if port.atom_name == "C"
            else (("N", "CA", 1), ("N", frame.n_substituent, 1))
        )
        for a, b, order in required:
            if b is None or not any(
                ref == AtomRef(port.residue_id, b)
                and edge.relationship_type is Rel.COVALENT
                and edge.order == order
                and not edge.aromatic
                for ref, edge in neighbors(parent, AtomRef(port.residue_id, a))
            ):
                return "partial_or_incompatible_flank", other.residue_id
        return "already_capped", other.residue_id
    return None


@dataclass(frozen=True, slots=True)
class StagedPeptideCap:
    """One complete cap and its flank bonded in the canonical address space."""

    structure: ProteinStructure
    port: AtomRef
    cap_id: ResidueId

    @classmethod
    def build(
        cls,
        parent: ProteinStructure,
        port: AtomRef,
        cap_id: ResidueId,
        frame: PeptideCapFrame,
        pose: PeptideCapPose,
    ) -> "StagedPeptideCap":
        """Assemble a complete neutral junction without changing existing coordinates."""
        rid = port.residue_id
        side = port.atom_name
        c = parent.constitution
        ri = c.residue_index(rid)
        residue = c.residue_site_at(ri)
        geom = parent.residue_geometry(ri)
        charges = dict(parent.residue_formal_charge_by_atom_name(ri))
        anchor = "C" if side == "N" else "N"
        methyl = "CH3" if side == "N" else "C"
        elements = {
            a: ("H" if a.startswith("H") else "C" if a == "CH3" else a)
            for a in pose.cap
        }
        edges = [
            (AtomRef(cap_id, methyl), AtomRef(cap_id, anchor), 1),
            (port, AtomRef(cap_id, anchor), 1),
        ]
        for i in range(1, 4):
            edges.append((AtomRef(cap_id, methyl), AtomRef(cap_id, f"H{i}"), 1))
        if side == "C":
            edges.extend(
                (
                    (AtomRef(cap_id, "N"), AtomRef(cap_id, "HN2"), 1),
                    (port, AtomRef(rid, "O"), 2),
                )
            )
            charges["O"] = 0
        else:
            edges.append((AtomRef(cap_id, "C"), AtomRef(cap_id, "O"), 2))
            if pose.flank_h is not None:
                if residue.has_atom_site("H"):
                    raise ValueError("flank_h_name_collision")
                residue = residue.with_atom_site(AtomSite("H", "H"))
                geom = geom.with_atom_geometry("H", AtomGeometry(pose.flank_h))
            substituent = frame.n_substituent or "H"
            edges.append((port, AtomRef(rid, substituent), 1))
            if substituent != "CD":
                charges[substituent] = 0
        edges.append((port, AtomRef(rid, "CA"), 1))
        charges[side] = 0
        staged = parent.with_updated_residue_facets(
            residue, residue_geometry=geom, formal_charge_by_atom_name=charges.items()
        )
        capsite = ResidueSite(
            "ACE" if side == "N" else "NME",
            cap_id,
            tuple(AtomSite(a, e) for a, e in elements.items()),
            is_hetero=True,
        )
        ligands = staged.constitution.ligands
        indices = tuple(
            staged.constitution.residue_index(s.residue_id) for s in ligands
        )
        capgeom = ResidueGeometry({a: AtomGeometry(v) for a, v in pose.cap.items()})
        staged = staged.with_ligand_facets(
            ligand_sites=(*ligands, capsite),
            ligand_geometries=(*(staged.residue_geometry(i) for i in indices), capgeom),
            ligand_formal_charge_payloads=(
                *(staged.residue_formal_charge_by_atom_name(i) for i in indices),
                tuple(dict.fromkeys(elements, 0).items()),
            ),
        )
        graph = {b.endpoint_pair(): b for b in staged.topology.bonds}
        for a, b, order in edges:
            ca = staged.constitution
            edge = TopologyBond(
                ca.atom_index(a),
                ca.atom_index(b),
                order=order,
                provenance=BondProvenance.REPAIR_INFERRED,
            )
            key = edge.endpoint_pair()
            graph[key] = (
                graph[key].with_resolved_chemistry(edge) if key in graph else edge
            )
        top = StructureTopology(
            constitution=staged.constitution,
            atom_topologies=staged.topology.atom_topologies,
            bonds=tuple(graph.values()),
        )
        result = ProteinStructure.from_payload(
            constitution=staged.constitution,
            geometry=staged.geometry,
            topology=top,
            polymer_blueprint=staged.polymer_blueprint,
            provenance=staged.provenance,
        )
        bound = observe_linked_peptide_cap(result, capsite)
        if bound is None or bound.partner != port or not bound.hydrogens_are_complete():
            raise ValueError("unverified_cap_chemistry")
        return cls(result, port, cap_id)

    def at(self, pose: PeptideCapPose) -> ProteinStructure:
        """Realize another pose of the same staged atom and bond graph."""
        out = self.structure
        geoms = [
            (
                self.cap_id,
                ResidueGeometry({a: AtomGeometry(x) for a, x in pose.cap.items()}),
            )
        ]
        if pose.flank_h is not None:
            rid = self.port.residue_id
            geoms.append(
                (
                    rid,
                    out.residue_geometry(
                        out.constitution.residue_index(rid)
                    ).with_atom_geometry("H", AtomGeometry(pose.flank_h)),
                )
            )
        return out.with_updated_residue_geometries(geoms)

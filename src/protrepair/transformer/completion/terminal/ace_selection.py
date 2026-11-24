"""Choose a backbone orientation for a newly generated, unconstrained ACE."""

from collections.abc import Iterable, Iterator
from dataclasses import dataclass
from enum import Enum
from math import isfinite
from zlib import error as ZlibError

from protrepair.analysis.backbone_contours import BackboneContour
from protrepair.chemistry.component.defaults import build_default_component_library
from protrepair.chemistry.standard.components import STANDARD_COMPONENT_ATOMS
from protrepair.geometry import Vec3
from protrepair.geometry.dihedral import measured_dihedral_degrees
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueRole
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.topology import BondRelationshipType, TopologyBond


class AceBackboneResult(str, Enum):
    """Assessment of one pose after its chemical, steric and water checks."""

    FAVORED = "favored"
    ALLOWED = "allowed"
    OUTLIER = "outlier"
    NO_FOLLOWING_PEPTIDE = "no_following_peptide"
    INVALID_CONNECTIVITY = "invalid_connectivity"
    INVALID_TORSION = "invalid_torsion"
    REFERENCE_UNAVAILABLE = "reference_unavailable"


@dataclass(frozen=True, slots=True)
class GeneratedAceSelection:
    """Outcome for one generated ACE, including rejected poses in trial order.

    If no pose qualifies, `structure` is the unchanged parent and `accepted`
    is None. Rejections cover only poses that passed the generator's hard checks.
    """

    structure: ProteinStructure
    accepted: AceBackboneResult | None
    rejected: tuple[AceBackboneResult, ...]

    @classmethod
    def from_verified_poses(
        cls,
        parent: ProteinStructure,
        poses: Iterable[ProteinStructure],
        *,
        flank_id: ResidueId,
        cap_id: ResidueId,
    ) -> "GeneratedAceSelection":
        """Select in order from one port's bounded, hard-admitted candidates.

        The generator owns the total trial budget, including poses it rejects
        before yielding. It supplies complete staged ACE products only when it
        also placed the missing amide H. A fixed H/D/T or Pro constrains the
        plane and is not a reason to try this search.

        Parameters
        ----------
        parent : ProteinStructure
            Structure before placing this cap, including previously placed caps.
        poses : Iterable[ProteinStructure]
            Verified products from that parent, in the generator's trial order.
        flank_id : ResidueId
            Polymer residue receiving the cap.
        cap_id : ResidueId
            Address assigned to the new ACE.

        Returns
        -------
        GeneratedAceSelection
            The first qualifying pose and the assessments preceding it, or the
            original parent with the reasons that no pose was selected.
        """
        rejected: list[AceBackboneResult] = []
        saw_outlier = False
        for pose in poses:
            result = _assess(parent, pose, flank_id, cap_id)
            if result in (AceBackboneResult.FAVORED, AceBackboneResult.ALLOWED) or (
                result is AceBackboneResult.NO_FOLLOWING_PEPTIDE and not saw_outlier
            ):
                return cls(pose, result, tuple(rejected))
            if result is AceBackboneResult.OUTLIER:
                saw_outlier = True
            rejected.append(result)
        return cls(parent, None, tuple(rejected))


def _external_edges(
    structure: ProteinStructure | StructureObservation, port: AtomRef
) -> Iterator[tuple[AtomRef, TopologyBond]]:
    c = structure.constitution
    idx = c.resolve_atom_index(port)
    if idx is None:
        return
    for edge in structure.topology.incident_bonds_by_atom().get(idx, ()):
        if edge.relationship_type in (
            BondRelationshipType.HYDROGEN_BOND,
            BondRelationshipType.METAL_COORDINATION,
        ):
            continue
        a, b = edge.endpoint_pair()
        other = c.atom_ref_at(b if a == idx else a)
        if other.residue_id != port.residue_id:
            yield other, edge


def _is_single_peptide_edge(edge: TopologyBond) -> bool:
    return (
        edge.relationship_type is BondRelationshipType.COVALENT
        and edge.order == 1
        and not edge.aromatic
    )


def _torsion(a: Vec3, b: Vec3, c: Vec3, d: Vec3) -> float | None:
    try:
        angle = measured_dihedral_degrees(a, b, c, d)
    except (ArithmeticError, ValueError):
        return None
    return angle if angle is not None and isfinite(angle) else None


def _unchanged_opposite(
    parent: ProteinStructure, carbon: AtomRef, right: AtomRef | None
) -> bool:
    if parent.constitution.resolve_atom_index(carbon) is None:
        return False
    edges = tuple(_external_edges(parent, carbon))
    if edges:
        if (
            len(edges) != 1
            or edges[0][0] != right
            or not _is_single_peptide_edge(edges[0][1])
        ):
            return False
    elif right is not None:
        return False

    observation = parent.provenance.ingress.observation
    if observation is None:
        return True
    if (
        carbon in observation.unresolved_polymer_endpoints
        or carbon in observation.unresolved_conect_ports
    ):
        return False
    return all(
        other == right
        and not edge.aromatic
        and edge.order in (None, 1)
        and edge.relationship_type
        in (BondRelationshipType.UNKNOWN, BondRelationshipType.COVALENT)
        for other, edge in _external_edges(observation, carbon)
    )


def _assess(
    parent: ProteinStructure, s: ProteinStructure, flank: ResidueId, cap: ResidueId
) -> AceBackboneResult:
    c = s.constitution
    flank_site = c.residue_or_ligand(flank)
    cap_site = c.residue_or_ligand(cap)
    if (
        flank_site is None
        or cap_site is None
        or cap_site.component_id != "ACE"
        or c.role_at(c.residue_index(flank)) is not ResidueRole.POLYMER
        or c.role_at(c.residue_index(cap)) is not ResidueRole.LIGAND
    ):
        return AceBackboneResult.INVALID_CONNECTIVITY

    library = build_default_component_library()
    template = library.get(flank_site.component_id)
    if (
        template is None
        or template.backbone_family_component_id not in STANDARD_COMPONENT_ATOMS
        or template.backbone_family_component_id == "PRO"
    ):
        return AceBackboneResult.INVALID_CONNECTIVITY

    left = AtomRef(cap, "C")
    n, ca, carbon = (AtomRef(flank, name) for name in ("N", "CA", "C"))
    indices = []
    for ref, element in ((left, "C"), (n, "N"), (ca, "C"), (carbon, "C")):
        idx = c.resolve_atom_index(ref)
        if idx is None or c.atom_site_at(idx).element != element:
            return AceBackboneResult.INVALID_CONNECTIVITY
        indices.append(idx)

    left_edges = tuple(_external_edges(s, n))
    if (
        len(left_edges) != 1
        or left_edges[0][0] != left
        or not _is_single_peptide_edge(left_edges[0][1])
    ):
        return AceBackboneResult.INVALID_CONNECTIVITY
    prev_c, n_pos, ca_pos, c_pos = (s.geometry.position(i) for i in indices)
    phi = _torsion(prev_c, n_pos, ca_pos, c_pos)
    if phi is None:
        return AceBackboneResult.INVALID_TORSION

    opposite = tuple(_external_edges(s, carbon))
    if len(opposite) > 1 or (opposite and not _is_single_peptide_edge(opposite[0][1])):
        return AceBackboneResult.INVALID_CONNECTIVITY
    right = opposite[0][0] if opposite else None
    if not _unchanged_opposite(parent, carbon, right):
        return AceBackboneResult.INVALID_CONNECTIVITY
    if right is None:
        return AceBackboneResult.NO_FOLLOWING_PEPTIDE

    right_idx = c.atom_index(right)
    following = c.residue_site_at(c.residue_index(right.residue_id))
    role = c.role_at(c.residue_index(right.residue_id))
    if right.atom_name != "N" or c.atom_site_at(right_idx).element != "N":
        return AceBackboneResult.INVALID_CONNECTIVITY
    if role is ResidueRole.POLYMER:
        next_tmpl = library.get(following.component_id)
        if (
            next_tmpl is None
            or next_tmpl.backbone_family_component_id not in STANDARD_COMPONENT_ATOMS
        ):
            return AceBackboneResult.INVALID_CONNECTIVITY
        following_family = next_tmpl.backbone_family_component_id
    elif role is ResidueRole.LIGAND and following.component_id == "NME":
        following_family = "NME"
    else:
        return AceBackboneResult.INVALID_CONNECTIVITY

    psi = _torsion(n_pos, ca_pos, c_pos, s.geometry.position(right_idx))
    if psi is None:
        return AceBackboneResult.INVALID_TORSION
    contour = BackboneContour.for_nonproline(
        template.backbone_family_component_id, following_family
    )
    try:
        score = contour.density(phi, psi)
    except (ArithmeticError, OSError, EOFError, ValueError, ZlibError):
        return AceBackboneResult.REFERENCE_UNAVAILABLE
    if not isfinite(score) or not 0.0 <= score <= 1.0:
        return AceBackboneResult.REFERENCE_UNAVAILABLE
    if score >= 0.02:
        return AceBackboneResult.FAVORED
    if score >= contour.allowed_floor:
        return AceBackboneResult.ALLOWED
    return AceBackboneResult.OUTLIER

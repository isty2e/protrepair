"""Bonded/topological pair exclusion rules for clash diagnostics."""

from collections.abc import Mapping
from math import acos, degrees, sqrt

from typing_extensions import Protocol

from protrepair.chemistry import ResidueTemplate
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import ResidueGeometry

DISULFIDE_BOND_DISTANCE_CUTOFF_ANGSTROM = 3.0
HYDROGEN_BOND_MIN_DISTANCE_ANGSTROM = 1.6
HYDROGEN_BOND_MAX_DISTANCE_ANGSTROM = 2.4
# Baker-Hubbard-style minimum for diagnostic suppression, not full H-bond analysis.
HYDROGEN_BOND_MIN_DONOR_HYDROGEN_ACCEPTOR_ANGLE_DEGREES = 120.0
HYDROGEN_BOND_DEGENERATE_NORM_EPSILON = 1e-12
DONOR_ELEMENTS = frozenset({"N", "O", "S"})
ACCEPTOR_ELEMENTS = frozenset({"N", "O", "S"})


class ClashTopologyResidueContext(Protocol):
    """Residue context surface required by clash topology exclusions."""

    @property
    def residue_site(self) -> ResidueSite:
        """Return residue site."""

        ...

    @property
    def residue_geometry(self) -> ResidueGeometry:
        """Return residue geometry."""

        ...

    @property
    def template(self) -> ResidueTemplate | None:
        """Return residue template."""

        ...

    @property
    def hydrogen_anchor_by_name(self) -> Mapping[str, str]:
        """Return inferred hydrogen anchors."""

        ...

    @property
    def chain_index(self) -> int | None:
        """Return chain index."""

        ...

    @property
    def residue_index(self) -> int | None:
        """Return residue index."""

        ...

    def is_adjacent_polymer_residue(
        self,
        other: "ClashTopologyResidueContext",
    ) -> bool:
        """Return whether two residue contexts are adjacent polymer residues."""

        ...


class ClashTopologyPositionedGeometry(Protocol):
    """Atom geometry surface required by coordinate-angle rules."""

    @property
    def position(self) -> "ClashTopologyVector":
        """Return atom position."""

        ...


class ClashTopologyAtomGeometry(ClashTopologyPositionedGeometry, Protocol):
    """Atom geometry surface required by direct topology rules."""

    def distance_to(self, other: "ClashTopologyAtomGeometry") -> float:
        """Return distance to another atom geometry."""

        ...


class ClashTopologyVector(Protocol):
    """Coordinate vector surface required by angle-based topology rules."""

    @property
    def x(self) -> float:
        """Return x coordinate."""

        ...

    @property
    def y(self) -> float:
        """Return y coordinate."""

        ...

    @property
    def z(self) -> float:
        """Return z coordinate."""

        ...


class ClashTopologyAtomSite(Protocol):
    """Atom-site surface required by bonded/topological exclusion rules."""

    @property
    def residue_id(self) -> object:
        """Return residue id."""

        ...

    @property
    def component_id(self) -> str:
        """Return component id."""

        ...

    @property
    def atom_name(self) -> str:
        """Return atom name."""

        ...

    @property
    def element(self) -> str:
        """Return element."""

        ...

    @property
    def geometry(self) -> ClashTopologyAtomGeometry:
        """Return geometry."""

        ...

    @property
    def context(self) -> ClashTopologyResidueContext:
        """Return residue context."""

        ...

    def is_hydrogen(self) -> bool:
        """Return whether this site is a hydrogen."""

        ...


class ClashTopologyPolicy(Protocol):
    """Policy surface required by bonded/topological exclusion rules."""

    @property
    def ignore_same_residue_bond_hops(self) -> int:
        """Return same-residue ignore hop cutoff."""

        ...

    @property
    def ignore_adjacent_polymer_bond_hops(self) -> int:
        """Return adjacent-polymer ignore hop cutoff."""

        ...


def should_ignore_pair(
    left_site: ClashTopologyAtomSite,
    right_site: ClashTopologyAtomSite,
    *,
    policy: ClashTopologyPolicy,
) -> bool:
    """Return whether one atom pair should be ignored as bonded or near-bonded."""

    if (
        left_site.residue_id == right_site.residue_id
        and not left_site.is_hydrogen()
        and not right_site.is_hydrogen()
    ):
        return True

    if left_site.residue_id == right_site.residue_id:
        bond_hops = same_residue_bond_hops(left_site, right_site)
        return (
            bond_hops is not None and bond_hops <= policy.ignore_same_residue_bond_hops
        )

    if direct_disulfide_bond(left_site, right_site):
        return True

    if left_site.context.is_adjacent_polymer_residue(right_site.context):
        bond_hops = adjacent_polymer_bond_hops(left_site, right_site)
        return (
            bond_hops is not None
            and bond_hops <= policy.ignore_adjacent_polymer_bond_hops
        )

    return False


def same_residue_bond_hops(
    left_site: ClashTopologyAtomSite,
    right_site: ClashTopologyAtomSite,
) -> int | None:
    """Return same-residue bond hops for two atom sites if the template knows them."""

    template = left_site.context.template
    if template is None:
        return None

    return atom_to_atom_bond_hops_within_residue(
        left_site.atom_name,
        right_site.atom_name,
        context=left_site.context,
        template=template,
    )


def adjacent_polymer_bond_hops(
    left_site: ClashTopologyAtomSite,
    right_site: ClashTopologyAtomSite,
) -> int | None:
    """Return bond hops for atom sites across one peptide bond if available."""

    if left_site.context.chain_index != right_site.context.chain_index:
        return None

    if (
        left_site.context.residue_index is None
        or right_site.context.residue_index is None
    ):
        return None

    if left_site.context.residue_index + 1 == right_site.context.residue_index:
        left_to_c = atom_to_named_backbone_hops(left_site, target_atom_name="C")
        right_to_n = atom_to_named_backbone_hops(right_site, target_atom_name="N")
    elif right_site.context.residue_index + 1 == left_site.context.residue_index:
        left_to_c = atom_to_named_backbone_hops(right_site, target_atom_name="C")
        right_to_n = atom_to_named_backbone_hops(left_site, target_atom_name="N")
    else:
        return None

    if left_to_c is None or right_to_n is None:
        return None

    return left_to_c + 1 + right_to_n


def atom_to_named_backbone_hops(
    site: ClashTopologyAtomSite,
    *,
    target_atom_name: str,
) -> int | None:
    """Return bond hops from one atom site to a named backbone heavy atom."""

    template = site.context.template
    if template is None:
        return None

    return atom_to_atom_bond_hops_within_residue(
        site.atom_name,
        target_atom_name,
        context=site.context,
        template=template,
    )


def atom_to_atom_bond_hops_within_residue(
    atom_name_1: str,
    atom_name_2: str,
    *,
    context: ClashTopologyResidueContext,
    template: ResidueTemplate,
) -> int | None:
    """Return intra-residue bond hops using heavy topology plus inferred H anchors."""

    normalized_atom_name_1 = atom_name_1.strip().upper()
    normalized_atom_name_2 = atom_name_2.strip().upper()
    if normalized_atom_name_1 == normalized_atom_name_2:
        return 0

    atom_name_1_is_hydrogen = atom_is_hydrogen(
        context.residue_site,
        normalized_atom_name_1,
    )
    atom_name_2_is_hydrogen = atom_is_hydrogen(
        context.residue_site,
        normalized_atom_name_2,
    )

    if not atom_name_1_is_hydrogen and not atom_name_2_is_hydrogen:
        return template.bond_hop_distance(
            normalized_atom_name_1,
            normalized_atom_name_2,
        )

    if atom_name_1_is_hydrogen and atom_name_2_is_hydrogen:
        anchor_atom_name_1 = context.hydrogen_anchor_by_name.get(normalized_atom_name_1)
        anchor_atom_name_2 = context.hydrogen_anchor_by_name.get(normalized_atom_name_2)
        if anchor_atom_name_1 is None or anchor_atom_name_2 is None:
            return None
        if anchor_atom_name_1 == anchor_atom_name_2:
            return 2
        heavy_hops = template.bond_hop_distance(anchor_atom_name_1, anchor_atom_name_2)
        if heavy_hops is None:
            return None
        return heavy_hops + 2

    if atom_name_1_is_hydrogen:
        anchor_atom_name = context.hydrogen_anchor_by_name.get(normalized_atom_name_1)
        if anchor_atom_name is None:
            return None
        if anchor_atom_name == normalized_atom_name_2:
            return 1
        heavy_hops = template.bond_hop_distance(
            anchor_atom_name,
            normalized_atom_name_2,
        )
        if heavy_hops is None:
            return None
        return heavy_hops + 1

    anchor_atom_name = context.hydrogen_anchor_by_name.get(normalized_atom_name_2)
    if anchor_atom_name is None:
        return None
    if anchor_atom_name == normalized_atom_name_1:
        return 1
    heavy_hops = template.bond_hop_distance(normalized_atom_name_1, anchor_atom_name)
    if heavy_hops is None:
        return None
    return heavy_hops + 1


def atom_is_hydrogen(residue_site: ResidueSite, atom_name: str) -> bool:
    """Return whether a named atom within a residue is a hydrogen."""

    return residue_site.atom_site(atom_name).element == "H"


def direct_disulfide_bond(
    left_site: ClashTopologyAtomSite,
    right_site: ClashTopologyAtomSite,
) -> bool:
    """Return whether one atom pair looks like a bonded disulfide sulfur pair."""

    return (
        left_site.component_id == "CYS"
        and right_site.component_id == "CYS"
        and left_site.atom_name == "SG"
        and right_site.atom_name == "SG"
        and left_site.geometry.distance_to(right_site.geometry)
        <= DISULFIDE_BOND_DISTANCE_CUTOFF_ANGSTROM
    )


def probable_hydrogen_bond(
    left_site: ClashTopologyAtomSite,
    right_site: ClashTopologyAtomSite,
    pair_distance: float,
) -> bool:
    """Return whether one H-involving pair is plausibly a hydrogen bond."""

    if left_site.residue_id == right_site.residue_id:
        return False

    if left_site.is_hydrogen() == right_site.is_hydrogen():
        return False

    if (
        pair_distance < HYDROGEN_BOND_MIN_DISTANCE_ANGSTROM
        or pair_distance > HYDROGEN_BOND_MAX_DISTANCE_ANGSTROM
    ):
        return False

    if left_site.is_hydrogen():
        hydrogen_site = left_site
        acceptor_site = right_site
    else:
        hydrogen_site = right_site
        acceptor_site = left_site

    anchor_atom_name = hydrogen_site.context.hydrogen_anchor_by_name.get(
        hydrogen_site.atom_name
    )
    if anchor_atom_name is None:
        return False

    donor_atom = hydrogen_site.context.residue_site.atom_site(anchor_atom_name)
    if (
        donor_atom.element not in DONOR_ELEMENTS
        or acceptor_site.element not in ACCEPTOR_ELEMENTS
    ):
        return False

    angle_degrees = _donor_hydrogen_acceptor_angle_degrees(
        hydrogen_site.context.residue_geometry.atom_geometry(anchor_atom_name),
        hydrogen_site.geometry,
        acceptor_site.geometry,
    )
    return (
        angle_degrees is not None
        and angle_degrees
        > HYDROGEN_BOND_MIN_DONOR_HYDROGEN_ACCEPTOR_ANGLE_DEGREES
    )


def _donor_hydrogen_acceptor_angle_degrees(
    donor_geometry: ClashTopologyPositionedGeometry,
    hydrogen_geometry: ClashTopologyPositionedGeometry,
    acceptor_geometry: ClashTopologyPositionedGeometry,
) -> float | None:
    """Return D-H-A angle in degrees, or None when geometry is undefined."""

    donor_vector = _vector_from_hydrogen_to_atom(donor_geometry, hydrogen_geometry)
    acceptor_vector = _vector_from_hydrogen_to_atom(
        acceptor_geometry,
        hydrogen_geometry,
    )
    donor_norm = _vector_norm(donor_vector)
    acceptor_norm = _vector_norm(acceptor_vector)
    if (
        donor_norm <= HYDROGEN_BOND_DEGENERATE_NORM_EPSILON
        or acceptor_norm <= HYDROGEN_BOND_DEGENERATE_NORM_EPSILON
    ):
        return None

    cosine = _dot_product(donor_vector, acceptor_vector) / (donor_norm * acceptor_norm)
    clamped_cosine = max(-1.0, min(1.0, cosine))
    return degrees(acos(clamped_cosine))


def _vector_from_hydrogen_to_atom(
    atom_geometry: ClashTopologyPositionedGeometry,
    hydrogen_geometry: ClashTopologyPositionedGeometry,
) -> tuple[float, float, float]:
    """Return vector from hydrogen position to another atom position."""

    return (
        atom_geometry.position.x - hydrogen_geometry.position.x,
        atom_geometry.position.y - hydrogen_geometry.position.y,
        atom_geometry.position.z - hydrogen_geometry.position.z,
    )


def _vector_norm(vector: tuple[float, float, float]) -> float:
    """Return Euclidean vector norm."""

    return sqrt(_dot_product(vector, vector))


def _dot_product(
    left_vector: tuple[float, float, float],
    right_vector: tuple[float, float, float],
) -> float:
    """Return 3D vector dot product."""

    return (
        left_vector[0] * right_vector[0]
        + left_vector[1] * right_vector[1]
        + left_vector[2] * right_vector[2]
    )

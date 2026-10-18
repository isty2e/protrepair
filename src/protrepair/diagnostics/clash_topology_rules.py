"""Bonded/topological pair exclusion rules for clash diagnostics."""

from collections.abc import Mapping
from math import acos, degrees, isfinite, sqrt

from typing_extensions import Protocol

from protrepair.chemistry import ResidueTemplate
from protrepair.chemistry.polar_contacts import PolarContactChemistry
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import ResidueGeometry
from protrepair.structure.labels import AtomRef, ResidueId

HYDROGEN_BOND_MIN_DISTANCE_ANGSTROM = 1.6
HYDROGEN_BOND_MAX_DISTANCE_ANGSTROM = 2.4
HYDROGEN_DONOR_MAX_DISTANCE_ANGSTROM = 1.45
# Baker-Hubbard-style diagnostic cutoff: DOI 10.1016/0079-6107(84)90007-5.
HYDROGEN_BOND_MIN_DONOR_HYDROGEN_ACCEPTOR_ANGLE_DEGREES = 120.0
# Clash classification treats undefined tuple-vector angles as non-exempt;
# it intentionally does not inherit geometry-placement failure semantics.
HYDROGEN_BOND_DEGENERATE_NORM_EPSILON = 1e-12
DONOR_ELEMENTS = frozenset({"N", "O", "S"})
ACCEPTOR_ELEMENTS = frozenset({"N", "O", "S"})


class ClashTopologyResidueContext(Protocol):
    """Residue context surface required by clash topology exclusions."""

    @property
    def polar_chemistry(self) -> PolarContactChemistry:
        """Return the shared coordinate-independent polar-contact observation."""

        ...

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
        """Return chemically supported hydrogen anchors."""

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
    def residue_id(self) -> ResidueId:
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
    def geometry(self) -> ClashTopologyPositionedGeometry:
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
    """Return residue-local hops from canonical and supported projected bonds."""

    return left_site.context.polar_chemistry.bond_hop_distance(
        AtomRef(left_site.residue_id, left_site.atom_name),
        AtomRef(right_site.residue_id, right_site.atom_name),
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

    if not site.context.residue_site.has_atom_site(target_atom_name):
        return None
    return site.context.polar_chemistry.bond_hop_distance(
        AtomRef(site.residue_id, site.atom_name),
        AtomRef(site.residue_id, target_atom_name),
    )


def probable_hydrogen_bond(
    left_site: ClashTopologyAtomSite,
    right_site: ClashTopologyAtomSite,
    pair_distance: float,
) -> bool:
    """Return whether one H-involving pair is plausibly a hydrogen bond."""

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

    if (
        acceptor_site.atom_name
        not in acceptor_site.context.polar_chemistry.acceptor_names(
            acceptor_site.residue_id
        )
    ):
        return False

    anchor_atom_name = hydrogen_site.context.hydrogen_anchor_by_name.get(
        hydrogen_site.atom_name
    )
    if anchor_atom_name is None:
        return False

    if not hydrogen_site.context.polar_chemistry.supports_donor(
        AtomRef(hydrogen_site.residue_id, anchor_atom_name)
    ):
        return False

    donor_atom = hydrogen_site.context.residue_site.atom_site(anchor_atom_name)
    if (
        donor_atom.element not in DONOR_ELEMENTS
        or acceptor_site.element not in ACCEPTOR_ELEMENTS
    ):
        return False

    donor_position = hydrogen_site.context.residue_geometry.atom_geometry(
        anchor_atom_name
    ).position
    hydrogen_position = hydrogen_site.geometry.position
    acceptor_position = acceptor_site.geometry.position
    return probable_hydrogen_bond_geometry(
        donor_element=donor_atom.element,
        acceptor_element=acceptor_site.element,
        hydrogen_acceptor_distance_angstrom=pair_distance,
        donor_x=donor_position.x,
        donor_y=donor_position.y,
        donor_z=donor_position.z,
        hydrogen_x=hydrogen_position.x,
        hydrogen_y=hydrogen_position.y,
        hydrogen_z=hydrogen_position.z,
        acceptor_x=acceptor_position.x,
        acceptor_y=acceptor_position.y,
        acceptor_z=acceptor_position.z,
    )


def probable_hydrogen_bond_geometry(
    *,
    donor_element: str,
    acceptor_element: str,
    hydrogen_acceptor_distance_angstrom: float,
    donor_x: float,
    donor_y: float,
    donor_z: float,
    hydrogen_x: float,
    hydrogen_y: float,
    hydrogen_z: float,
    acceptor_x: float,
    acceptor_y: float,
    acceptor_z: float,
) -> bool:
    """Return whether one D-H-A geometry may suppress a steric clash."""

    if (
        donor_element not in DONOR_ELEMENTS
        or acceptor_element not in ACCEPTOR_ELEMENTS
        or not isfinite(hydrogen_acceptor_distance_angstrom)
        or not (
            HYDROGEN_BOND_MIN_DISTANCE_ANGSTROM
            <= hydrogen_acceptor_distance_angstrom
            <= HYDROGEN_BOND_MAX_DISTANCE_ANGSTROM
        )
    ):
        return False

    donor_vector = (
        donor_x - hydrogen_x,
        donor_y - hydrogen_y,
        donor_z - hydrogen_z,
    )
    acceptor_vector = (
        acceptor_x - hydrogen_x,
        acceptor_y - hydrogen_y,
        acceptor_z - hydrogen_z,
    )
    donor_norm = _vector_norm(donor_vector)
    acceptor_norm = _vector_norm(acceptor_vector)
    if (
        donor_norm <= HYDROGEN_BOND_DEGENERATE_NORM_EPSILON
        or donor_norm > HYDROGEN_DONOR_MAX_DISTANCE_ANGSTROM
        or acceptor_norm <= HYDROGEN_BOND_DEGENERATE_NORM_EPSILON
    ):
        return False

    cosine = _dot_product(donor_vector, acceptor_vector) / (donor_norm * acceptor_norm)
    if not isfinite(cosine):
        return False

    clamped_cosine = max(-1.0, min(1.0, cosine))
    return (
        degrees(acos(clamped_cosine))
        > HYDROGEN_BOND_MIN_DONOR_HYDROGEN_ACCEPTOR_ANGLE_DEGREES
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

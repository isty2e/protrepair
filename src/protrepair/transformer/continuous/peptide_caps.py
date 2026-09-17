"""Temporary peptide support graph construction and source-only projection."""

from dataclasses import dataclass, replace

import numpy as np
import numpy.typing as npt

from protrepair.diagnostics.events import ValidationIssue
from protrepair.diagnostics.kinds import IssueSeverity, ValidationIssueKind
from protrepair.errors import RefinementError
from protrepair.geometry import Vec3
from protrepair.geometry.internal_coordinates import InternalCoordinateFrame
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import (
    AtomSite,
    ResidueSite,
    StructureConstitution,
)
from protrepair.structure.geometry import AtomGeometry, StructureGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.slots import AtomIndex, ResidueIndex
from protrepair.structure.snapshot import ProteinStructureSnapshot
from protrepair.structure.topology import (
    AtomTopology,
    BondProvenance,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.artifacts import RegionTransformationResult
from protrepair.transformer.artifacts.patch import MovedAtomDelta, StructureDelta
from protrepair.transformer.continuous.bonds import PlannedBond
from protrepair.transformer.continuous.domain import ContinuousRelaxationProblem
from protrepair.transformer.continuous.peptide_boundary import (
    PeptideBoundary,
    PeptideBoundaryPlan,
)
from protrepair.transformer.continuous.shared import apply_position_updates


@dataclass(frozen=True, slots=True)
class CappedRelaxationProblem:
    """Own a temporary graph and checked projection to its original source.

    Parameters
    ----------
    plan : PeptideBoundaryPlan
        Validated source-scoped cuts and original calculation.
    source_problem : ContinuousRelaxationProblem
        Original execution settings and source graph.
    problem : ContinuousRelaxationProblem
        Completed native calculation; source slots retain their original indices.
    """

    plan: PeptideBoundaryPlan
    source_problem: ContinuousRelaxationProblem
    problem: ContinuousRelaxationProblem

    def project_result(
        self, result: RegionTransformationResult
    ) -> RegionTransformationResult:
        """Discard support atoms and return only allowed original coordinate edits.

        Parameters
        ----------
        result : RegionTransformationResult
            Backend output in this model's address space.

        Returns
        -------
        RegionTransformationResult
            Source constitution, topology and provenance, with a source-slot delta.

        Raises
        ------
        RefinementError
            The backend changed chemistry, fixed context or model identity.
        """
        if not self.plan.boundaries:
            return result
        model = self.problem.region.snapshot.structure
        refined = result.refined_structure
        if (
            refined.constitution != model.constitution
            or refined.topology != model.topology
            or refined.provenance != model.provenance
        ):
            raise RefinementError("cap calculation changed its graph or provenance")
        movable = frozenset(self.source_problem.region.movable_atom_indices)
        for i in range(len(model.constitution.atom_slots)):
            index = AtomIndex(i)
            before = model.geometry.atom_geometry(index)
            after = refined.geometry.atom_geometry(index)
            if after.with_position(before.position) != before:
                raise RefinementError("cap calculation changed atom metadata")
            if index not in movable and after != before:
                raise RefinementError("cap calculation moved a fixed context atom")
        source = self.source_problem.region.snapshot.structure
        moved = {
            index: refined.geometry.atom_geometry(index).position
            for index in self.source_problem.region.movable_atom_indices
            if refined.geometry.atom_geometry(index)
            != source.geometry.atom_geometry(index)
        }
        projected = apply_position_updates(source, moved_positions=moved)
        support_issues = tuple(
            ValidationIssue.for_residue(
                kind=ValidationIssueKind.COMPUTATIONAL_PEPTIDE_CAP,
                severity=IssueSeverity.INFO,
                residue_id=boundary.endpoint.residue_id,
                atom_names=(boundary.endpoint.atom_name,),
                message=(
                    "Local refinement used a fixed calculation-only peptide cap "
                    + (
                        "from omitted source geometry"
                        if boundary.source_partner is not None
                        else "at an unresolved internal peptide boundary"
                    )
                    + "; source chemistry and exported atoms are unchanged."
                ),
            )
            for boundary in self.plan.boundaries
        )
        return RegionTransformationResult(
            refined_structure=projected,
            delta=StructureDelta(
                before_constitution=source.constitution,
                after_constitution=source.constitution,
                moved_atoms=tuple(MovedAtomDelta(index, index) for index in moved),
            ),
            issues=(*result.issues, *support_issues),
            backend_name=result.backend_name,
            backend_version=result.backend_version,
        )


@dataclass(frozen=True, slots=True)
class PeptideCapTransformer:
    """Construct fixed source-backed or compact synthetic peptide support.

    Notes
    -----
    Source cuts use ACE/NME geometry from the omitted partner. Missing partners
    use formyl/primary-amide groups, not invented biological free termini.
    Idealized geometry is an initialization, not reconstructed missing residues.
    Caps stay fixed; there is no cap-only optimization or convergence requirement.
    """

    def transform(
        self, problem: ContinuousRelaxationProblem, plan: PeptideBoundaryPlan
    ) -> CappedRelaxationProblem:
        """Append temporary topology without editing any source atom or bond.

        Parameters
        ----------
        problem : ContinuousRelaxationProblem
            Source problem with the same region and bonds as the plan.
        plan : PeptideBoundaryPlan
            Validated cuts in a source-indexed continuous problem.

        Returns
        -------
        CappedRelaxationProblem
            Completed graph and its source-only projection owner.

        Raises
        ------
        RefinementError
            Required source anchors are missing or geometrically degenerate.
        """
        if problem.region is not plan.region or problem.bonds != plan.bonds:
            raise RefinementError("peptide cap plan belongs to a different problem")
        if not plan.boundaries:
            return CappedRelaxationProblem(plan, problem, problem)
        region = problem.region
        source = region.snapshot.structure
        constitution = source.constitution
        chain_id = "CAP"
        while any(
            site.residue_id.chain_id == chain_id for site in constitution.residue_slots
        ):
            chain_id += "_"
        sites = []
        geometries = []
        cap_edges: list[tuple[AtomRef, AtomRef, int]] = []
        for number, boundary in enumerate(plan.boundaries, 1):
            residue_id = ResidueId(chain_id, number)
            try:
                atoms, edges, anchor = _cap_atoms(problem, boundary)
            except ValueError as error:
                raise RefinementError(
                    f"cannot place peptide cap at {boundary.endpoint}: {error}"
                ) from error
            component = (
                ("ACE" if anchor == "C" else "NME")
                if boundary.source_partner is not None
                else ("FMT" if anchor == "C" else "AMD")
            )
            sites.append(
                ResidueSite(
                    component,
                    residue_id,
                    tuple(AtomSite(name, element) for name, element, _ in atoms),
                    is_hetero=True,
                )
            )
            geometries.extend(AtomGeometry(position) for _, _, position in atoms)
            cap_edges.extend(
                (AtomRef(residue_id, first), AtomRef(residue_id, second), order)
                for first, second, order in edges
            )
            cap_edges.append((boundary.endpoint, AtomRef(residue_id, anchor), 1))
            if boundary.supplemental_hydrogen:
                cap_edges.append((boundary.endpoint, AtomRef(residue_id, "HN"), 1))
        modeled_constitution = StructureConstitution(
            chains=constitution.chains, ligands=(*constitution.ligands, *sites)
        )
        bonds = list(source.topology.bonds)
        # Validated planning can supply an existing H's attachment in this model;
        # that does not repair the original topology as a side effect of FF binding.
        pairs = {bond.endpoint_pair() for bond in bonds}
        for bond in problem.bonds:
            pair = (bond.atom_index_1, bond.atom_index_2)
            if pair not in pairs:
                bonds.append(
                    TopologyBond(
                        *pair,
                        order=bond.order,
                        aromatic=bond.aromatic,
                        provenance=BondProvenance.REPAIR_INFERRED,
                    )
                )
        new_bonds = tuple(
            TopologyBond(
                modeled_constitution.atom_index(left),
                modeled_constitution.atom_index(right),
                order=order,
                provenance=BondProvenance.REPAIR_INFERRED,
            )
            for left, right, order in cap_edges
        )
        modeled = ProteinStructure.from_payload(
            constitution=modeled_constitution,
            geometry=StructureGeometry(
                constitution=modeled_constitution,
                atom_geometries=tuple(
                    source.geometry.atom_geometry(AtomIndex(i))
                    for i in range(len(constitution.atom_slots))
                )
                + tuple(geometries),
            ),
            topology=StructureTopology(
                constitution=modeled_constitution,
                atom_topologies=tuple(
                    source.topology.atom_topology(AtomIndex(i))
                    for i in range(len(constitution.atom_slots))
                )
                + tuple(AtomTopology(0) for _ in geometries),
                bonds=(*bonds, *new_bonds),
            ),
            provenance=source.provenance,
        )
        cap_indices = tuple(
            AtomIndex(i)
            for i in range(
                len(constitution.atom_slots), len(modeled_constitution.atom_slots)
            )
        )
        new_planned = tuple(
            PlannedBond(b.atom_index_1, b.atom_index_2, b.order) for b in new_bonds
        )
        modeled_region = replace(
            region,
            snapshot=ProteinStructureSnapshot.from_structure(modeled),
            fixed_context_atom_indices=(
                *region.fixed_context_atom_indices,
                *cap_indices,
            ),
            included_residue_indices=(
                *region.included_residue_indices,
                *(
                    ResidueIndex(i)
                    for i in range(
                        len(constitution.residue_slots),
                        len(modeled_constitution.residue_slots),
                    )
                ),
            ),
            inter_residue_bonds=(
                *region.inter_residue_bonds,
                *(
                    b
                    for b in new_planned
                    if modeled_constitution.atom_ref_at(b.atom_index_1).residue_id
                    != modeled_constitution.atom_ref_at(b.atom_index_2).residue_id
                ),
            ),
        )
        return CappedRelaxationProblem(
            plan,
            problem,
            replace(
                problem,
                region=modeled_region,
                bonds=(*problem.bonds, *new_planned),
            ),
        )


def _unit(vector: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
    norm = float(np.linalg.norm(vector))
    if not np.isfinite(norm) or norm < 1e-8:
        raise RefinementError("degenerate peptide cap attachment frame")
    return vector / norm


def _third_direction(
    center: npt.NDArray[np.float64],
    first: npt.NDArray[np.float64],
    second: npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    return -_unit(_unit(first - center) + _unit(second - center))


def _place(
    a: npt.NDArray[np.float64],
    b: npt.NDArray[np.float64],
    c: npt.NDArray[np.float64],
    length: float,
    angle: float,
    torsion: float,
) -> npt.NDArray[np.float64]:
    return (
        InternalCoordinateFrame(a, b, c)
        .place(bond_length=length, bond_angle_degrees=angle, dihedral_degrees=torsion)
        .to_array()
    )


def _cap_atoms(
    problem: ContinuousRelaxationProblem, boundary: PeptideBoundary
) -> tuple[tuple[tuple[str, str, Vec3], ...], tuple[tuple[str, str, int], ...], str]:
    source = problem.region.snapshot.structure
    constitution = source.constitution

    def point(residue_id: ResidueId, name: str) -> npt.NDArray[np.float64]:
        index = constitution.resolve_atom_index(AtomRef(residue_id, name))
        if index is None:
            raise RefinementError(
                f"missing peptide cap anchor {residue_id.display_token()}.{name}"
            )
        return source.geometry.atom_geometry(index).position.to_array()

    def nitrogen_hydrogen(residue_id: ResidueId) -> npt.NDArray[np.float64]:
        index = constitution.atom_index(AtomRef(residue_id, "N"))
        hydrogens = set()
        for bond in (*source.topology.bonds, *problem.bonds):
            if index not in (bond.atom_index_1, bond.atom_index_2):
                continue
            other = (
                bond.atom_index_2 if bond.atom_index_1 == index else bond.atom_index_1
            )
            if constitution.atom_site_at(other).is_hydrogen():
                hydrogens.add(other)
        if len(hydrogens) != 1:
            raise RefinementError("peptide cap needs one unambiguous N-H anchor")
        return source.geometry.atom_geometry(next(iter(hydrogens))).position.to_array()

    residue_id = boundary.endpoint.residue_id
    endpoint = point(residue_id, boundary.endpoint.atom_name)
    ca = point(residue_id, "CA")
    partner = boundary.source_partner
    if boundary.endpoint.atom_name == "N":
        if partner is not None:
            attachment, oxygen, methyl = (
                point(partner.residue_id, name) for name in ("C", "O", "CA")
            )
        else:
            site = constitution.residue_or_ligand(residue_id)
            assert site is not None
            if boundary.supplemental_hydrogen:
                attachment = _place(
                    point(residue_id, "C"), ca, endpoint, 1.329, 121.7, 180.0
                )
            else:
                other = (
                    point(residue_id, "CD")
                    if site.component_id == "PRO"
                    else nitrogen_hydrogen(residue_id)
                )
                attachment = endpoint + 1.329 * _third_direction(endpoint, ca, other)
            methyl = _place(ca, endpoint, attachment, 1.525, 116.2, 180.0)
            oxygen = _place(ca, endpoint, attachment, 1.231, 123.0, 0.0)
        atoms = [("C", "C", attachment), ("O", "O", oxygen), ("CH3", "C", methyl)]
        edges = [("C", "O", 2), ("C", "CH3", 1)]
        anchor = "C"
    else:
        if partner is not None:
            partner_site = constitution.residue_or_ligand(partner.residue_id)
            if partner_site is None or partner_site.component_id == "PRO":
                raise RefinementError(
                    "source proline requires intact fixed context, not NME"
                )
            attachment, methyl = (
                point(partner.residue_id, name) for name in ("N", "CA")
            )
            if boundary.source_partner_hydrogen is None:
                raise RefinementError("source-backed NME requires a validated H anchor")
            hydrogen = point(
                partner.residue_id, boundary.source_partner_hydrogen.atom_name
            )
        else:
            attachment = endpoint + 1.329 * _third_direction(
                endpoint, ca, point(residue_id, "O")
            )
            methyl = _place(ca, endpoint, attachment, 1.458, 121.7, 180.0)
            hydrogen = attachment + 1.01 * _third_direction(
                attachment, endpoint, methyl
            )
        atoms = [("N", "N", attachment), ("H", "H", hydrogen), ("CH3", "C", methyl)]
        edges = [("N", "H", 1), ("N", "CH3", 1)]
        anchor = "N"
    if partner is None:
        hydrogen = attachment + (1.09 if anchor == "C" else 1.01) * _unit(
            methyl - attachment
        )
        atoms[-1] = ("HM", "H", hydrogen)
        edges[-1] = (anchor, "HM", 1)
    else:
        for i, torsion in enumerate((60.0, 180.0, 300.0), 1):
            atoms.append(
                (
                    f"H{i}",
                    "H",
                    _place(endpoint, attachment, methyl, 1.09, 109.4712206, torsion),
                )
            )
            edges.append(("CH3", f"H{i}", 1))
    if boundary.supplemental_hydrogen:
        atoms.append(
            ("HN", "H", endpoint + 1.01 * _third_direction(endpoint, ca, attachment))
        )
    return (
        tuple(
            (name, element, Vec3.from_iterable(position))
            for name, element, position in atoms
        ),
        tuple(edges),
        anchor,
    )

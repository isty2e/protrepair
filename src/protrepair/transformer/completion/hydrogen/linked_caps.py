"""Publish admitted cap topology and place only missing cap hydrogens."""

from dataclasses import replace

import numpy as np

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - exercised by backend-availability tests
    Chem = None

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.component.semantics import IdealGeometryHydrogenSemantics
from protrepair.chemistry.peptide_caps import LinkedPeptideCap
from protrepair.diagnostics.events import RepairEvent
from protrepair.diagnostics.kinds import RepairEventKind
from protrepair.diagnostics.polymer_hydrogen import linked_cap_issue
from protrepair.errors import RdkitUnavailableError
from protrepair.geometry import GeometryPlacementError, Vec3
from protrepair.geometry.placement_vector import normalized_vector_or_none
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import AtomSite
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.topology import StructureTopology, TopologyBond
from protrepair.transformer.completion.hydrogen.component_patch import (
    generate_component_hydrogen_patch,
)
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload
from protrepair.transformer.completion.shared.patch import OrderedAtomPatch
from protrepair.transformer.result import TransformationResult


def publish_linked_cap_topology(
    structure: ProteinStructure,
    cap: LinkedPeptideCap,
) -> TransformationResult:
    """Resolve admitted heavy bonds before any cap H can be placed."""
    if cap.refusal is not None:
        return TransformationResult(
            structure, (), (linked_cap_issue(cap, cap.refusal),)
        )
    if cap.heavy_bond_updates:
        bonds = cap.heavy_bond_updates
        detail = "resolved heavy bonds required for linked-cap hydrogenation"
    elif not cap.missing_hydrogen_parents:
        bonds = cap.hydrogen_bond_updates
        detail = "attached the cap's existing hydrogens"
    else:
        return TransformationResult(structure, (), ())
    return _publish_bonds(structure, cap, bonds, detail=detail)


def place_linked_cap_hydrogens(
    structure: ProteinStructure,
    cap: LinkedPeptideCap,
    component_library: ComponentLibrary,
) -> TransformationResult:
    """Add missing cap H and their bonds as one source-preserving patch."""
    if not cap.link_is_realized():
        reason = cap.refusal or "the admitted heavy cap bonds have not been realized"
        return TransformationResult(structure, (), (linked_cap_issue(cap, reason),))
    if not cap.missing_hydrogen_parents:
        return _publish_bonds(
            structure,
            cap,
            cap.hydrogen_bond_updates,
            detail="attached the cap's existing hydrogens",
        )

    payload = _cap_payload(structure, cap.residue_id)
    try:
        if cap.component_id == "NME":
            positions = _linked_nme_positions(structure, cap)
        else:
            positions = _ace_positions(payload, cap, component_library)
    except (
        GeometryPlacementError,
        RdkitUnavailableError,
        RuntimeError,
        ValueError,
    ) as error:
        return TransformationResult(structure, (), (linked_cap_issue(cap, str(error)),))

    for (name, _), position in zip(
        cap.missing_hydrogen_parents, positions, strict=True
    ):
        payload = payload.with_atom_payload(
            AtomSite(name, "H"),
            atom_geometry=AtomGeometry(position),
            formal_charge=None,
        )
    updated = structure.with_updated_residue_facets_batch(
        (
            (
                payload.residue_site,
                payload.residue_geometry,
                payload.formal_charge_by_atom_name,
            ),
        )
    )
    before = structure.constitution
    after = updated.constitution
    bonds = [
        replace(
            bond,
            atom_index_1=after.atom_index(before.atom_ref_at(bond.atom_index_1)),
            atom_index_2=after.atom_index(before.atom_ref_at(bond.atom_index_2)),
        )
        for bond in cap.hydrogen_bond_updates
    ]
    bonds.extend(
        TopologyBond(
            after.atom_index(AtomRef(cap.residue_id, parent)),
            after.atom_index(AtomRef(cap.residue_id, name)),
        )
        for name, parent in cap.missing_hydrogen_parents
    )
    repair = RepairEvent.for_residue(
        kind=RepairEventKind.HYDROGENS_ADDED,
        residue_id=cap.residue_id,
        component_id=cap.component_id,
        atom_names=tuple(name for name, _ in cap.missing_hydrogen_parents),
    )
    return TransformationResult(_with_bonds(updated, tuple(bonds)), (repair,), ())


def _publish_bonds(
    structure: ProteinStructure,
    cap: LinkedPeptideCap,
    bonds: tuple[TopologyBond, ...],
    *,
    detail: str,
) -> TransformationResult:
    if not bonds:
        return TransformationResult(structure, (), ())
    repair = RepairEvent.for_residue(
        kind=RepairEventKind.HYDROGEN_TOPOLOGY_RESOLVED,
        residue_id=cap.residue_id,
        component_id=cap.component_id,
        atom_names=tuple(
            dict.fromkeys(
                ref.atom_name
                for bond in bonds
                for index in bond.endpoint_pair()
                if (ref := structure.constitution.atom_ref_at(index)).residue_id
                == cap.residue_id
            )
        ),
        details=detail,
    )
    return TransformationResult(_with_bonds(structure, bonds), (repair,), ())


def _with_bonds(
    structure: ProteinStructure, updates: tuple[TopologyBond, ...]
) -> ProteinStructure:
    bonds = {bond.endpoint_pair(): bond for bond in structure.topology.bonds}
    bonds.update((bond.endpoint_pair(), bond) for bond in updates)
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=tuple(bonds.values()),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def _cap_payload(
    structure: ProteinStructure, residue_id: ResidueId
) -> CompletionResiduePayload:
    index = structure.constitution.residue_index(residue_id)
    return CompletionResiduePayload(
        structure.constitution.residue_site_at(index),
        structure.residue_geometry(index),
        structure.residue_formal_charge_by_atom_name(index),
    )


def _ace_positions(
    payload: CompletionResiduePayload,
    cap: LinkedPeptideCap,
    component_library: ComponentLibrary,
) -> tuple[Vec3, ...]:
    template = component_library.require("ACE")
    semantics = template.hydrogen_semantics
    if not isinstance(semantics, IdealGeometryHydrogenSemantics):
        raise GeometryPlacementError("the acetyl cap has no stock placement reference")
    heavy = payload.without_atom_sites(
        {name for name, _ in cap.existing_hydrogen_parents}
    )
    patch = generate_component_hydrogen_patch(
        residue=heavy,
        patch=OrderedAtomPatch.from_residue_payload(
            heavy.residue_site, residue_geometry=heavy.residue_geometry
        ),
        semantics=semantics,
        selected_hydrogen_names=frozenset(
            name for name, _ in cap.missing_hydrogen_parents
        ),
    )
    if patch is None:
        raise GeometryPlacementError("the acetyl cap has no usable heavy frame")
    placed = heavy.apply_patch(patch)
    return tuple(placed.position(name) for name, _ in cap.missing_hydrogen_parents)


def _linked_nme_positions(
    structure: ProteinStructure, cap: LinkedPeptideCap
) -> tuple[Vec3, ...]:
    chem = Chem
    if chem is None:
        raise RdkitUnavailableError(
            "linked NME hydrogen placement requires operational RDKit"
        )
    partner = cap.partner
    assert partner is not None
    # The calculation contains just the admitted amide and its real heavy frame.
    refs = tuple(
        AtomRef(partner.residue_id, name) for name in ("CA", "C", "O")
    ) + tuple(
        AtomRef(cap.residue_id, name)
        for name in ("N", "C", *(name for name, _ in cap.existing_hydrogen_parents))
    )
    constitution = structure.constitution
    source_indices = tuple(constitution.atom_index(ref) for ref in refs)
    positions = tuple(
        structure.geometry.position(index).to_array() for index in source_indices
    )
    alpha, carbonyl, oxygen, nitrogen, methyl = positions[:5]
    if (
        not all(np.all(np.isfinite(point)) for point in positions)
        or normalized_vector_or_none(
            np.asarray(
                np.cross(carbonyl - nitrogen, methyl - nitrogen), dtype=np.float64
            )
        )
        is None
        or normalized_vector_or_none(
            np.asarray(np.cross(alpha - carbonyl, oxygen - carbonyl), dtype=np.float64)
        )
        is None
    ):
        raise GeometryPlacementError(
            "the linked NME amide has a degenerate placement frame"
        )

    molecule = chem.RWMol()
    conformer = chem.Conformer(len(refs))
    owners: dict[str, int] = {"N": 3, "C": 4}
    for offset, (index, position) in enumerate(
        zip(source_indices, positions, strict=True)
    ):
        element = constitution.atom_site_at(index).element
        atom = chem.Atom("H" if element in ("D", "T") else element)
        if element in ("D", "T"):
            atom.SetIsotope(2 if element == "D" else 3)
        molecule.AddAtom(atom)
        conformer.SetAtomPosition(offset, tuple(map(float, position)))
    for first, second in ((0, 1), (1, 3), (3, 4)):
        molecule.AddBond(first, second, chem.BondType.SINGLE)
    molecule.AddBond(1, 2, chem.BondType.DOUBLE)
    for index, (_, parent) in enumerate(cap.existing_hydrogen_parents, start=5):
        molecule.AddBond(owners[parent], index, chem.BondType.SINGLE)
    molecule.AddConformer(conformer, assignId=True)
    chem.SanitizeMol(molecule)
    protonated = chem.AddHs(molecule, addCoords=True, onlyOnAtoms=list(owners.values()))

    placed: dict[int, list[Vec3]] = {index: [] for index in owners.values()}
    result_coordinates = protonated.GetConformer()
    for index in range(len(refs), protonated.GetNumAtoms()):
        (parent,) = protonated.GetAtomWithIdx(index).GetNeighbors()
        point = result_coordinates.GetAtomPosition(index)
        if not np.all(np.isfinite((point.x, point.y, point.z))):
            raise GeometryPlacementError(
                "the linked NME placement returned a nonfinite coordinate"
            )
        placed[parent.GetIdx()].append(Vec3(point.x, point.y, point.z))
    if len(cap.missing_hydrogen_parents) != sum(map(len, placed.values())) or any(
        len(placed[index])
        != sum(parent == name for _, parent in cap.missing_hydrogen_parents)
        for name, index in owners.items()
    ):
        raise GeometryPlacementError(
            "the linked NME placement disagrees with its admitted H inventory"
        )
    return tuple(
        placed[owners[parent]].pop(0) for _, parent in cap.missing_hydrogen_parents
    )

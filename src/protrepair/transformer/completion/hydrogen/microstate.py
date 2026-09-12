"""Place H from a selected site graph, without letting the backend select chemistry."""

from collections import Counter, defaultdict
from math import isfinite
from typing import TYPE_CHECKING

try:
    from rdkit import Chem
except ImportError:  # pragma: no cover - tested through the unavailable boundary
    Chem = None

from protrepair.chemistry.microstate.catalog import PeptideLinkage, PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.graph import MicrostateGraph
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.resolution import (
    AppliedMicrostateOverride,
    MicrostateConstraints,
    MicrostateSelectionBasis,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.geometry import GeometryPlacementError, Vec3
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.transformer.polymer_microstate import (
    MicrostateHydrogenPlacement,
    PolymerMicrostatePatch,
)

if TYPE_CHECKING:
    from rdkit.Chem.rdchem import Mol


def place_polymer_microstate_hydrogens(
    context: PolymerMicrostateContext,
    residue_id: ResidueId,
    site: PolymerMicrostateSite,
    *,
    override: MicrostateConstraints | None = None,
    preferences: tuple[MicrostateConstraints, ...] = (),
    rebuild_existing: bool = False,
) -> PolymerMicrostatePatch:
    """Construct a complete atomic patch with graph-constrained H coordinates.

    Parameters
    ----------
    context : PolymerMicrostateContext
        Immutable snapshot, original observations and applied chemical choices.
    residue_id : ResidueId
        Polymer residue to prepare.
    site : PolymerMicrostateSite
        Supported site with established linkage or a marked terminal assumption.
    override : MicrostateConstraints or None
        Explicit chemical request; otherwise a compatible applied choice survives.
    preferences : tuple[MicrostateConstraints, ...]
        Source-compatible preparation preferences, not coordinate instructions.
    rebuild_existing : bool
        Regenerate selected H coordinates, including original H whose heavy anchors
        have moved. False preserves current or original H coordinates and uses
        them as anchors when adding missing H. Neither mode changes heavy atoms.

    Returns
    -------
    PolymerMicrostatePatch
        Snapshot-bound, complete H/charge/bond update. The caller applies it with
        other disjoint sites; no partly changed structure escapes this operation.

    Raises
    ------
    TypeError
        Coordinate rebuilding is not a boolean.
    ValueError
        Chemistry, identities or current boundaries are unresolved/unsupported.
    GeometryPlacementError
        Native placement fails or produces invalid H vectors.
    RdkitUnavailableError
        Required RDKit installation is unavailable for coordinate generation.

    Notes
    -----
    The native molecule is a coordinate stencil, not an FF molecule or a new
    chemical authority. Only selected site H coordinates are copied back. Other
    standard heavy atoms provide anchors with template valence, and an actual
    peptide partner supplies the carbonyl needed for planar amide N placement.
    No minimization, pKa prediction or clash-free pose is promised here.
    """
    if type(rebuild_existing) is not bool:
        raise TypeError("microstate coordinate rebuilding must be a boolean")
    resolution = context.resolve(
        residue_id, site, override=override, preferences=preferences
    )
    graph = resolution.graph
    if graph is None:
        raise ValueError(
            f"cannot place {resolution.status.value} polymer chemistry: "
            f"{'; '.join(resolution.details)}"
        )
    identities = context.hydrogen_atom_sites(residue_id, site, resolution)
    preserved = {
        atom.name: geometry
        for atom, _parent in identities
        if not rebuild_existing
        for geometry in (_existing_geometry(context, AtomRef(residue_id, atom.name)),)
        if geometry is not None
    }
    placements = tuple(
        MicrostateHydrogenPlacement(atom, parent, preserved[atom.name])
        for atom, parent in identities
        if atom.name in preserved
    )
    missing = [
        (atom, parent) for atom, parent in identities if atom.name not in preserved
    ]
    if missing:
        if Chem is None:
            raise RdkitUnavailableError(
                "polymer microstate placement requires an operational "
                "RDKit installation"
            )
        try:
            molecule, indices = _coordinate_stencil(
                context, residue_id, site, graph, placements
            )
            hydrogenated = Chem.AddHs(
                molecule,
                explicitOnly=True,
                addCoords=True,
                onlyOnAtoms=[indices[atom.name] for atom in graph.atoms],
            )
        except (ValueError, RuntimeError) as error:
            raise GeometryPlacementError(
                f"microstate H placement failed: {error}"
            ) from error
        coordinates = _new_hydrogen_coordinates(molecule, hydrogenated, indices, graph)
        generated = []
        for atom, parent in missing:
            position = coordinates[parent].pop(0)
            parent_geometry = _existing_geometry(context, AtomRef(residue_id, parent))
            assert parent_geometry is not None
            generated.append(
                MicrostateHydrogenPlacement(
                    atom, parent, parent_geometry.with_position(position)
                )
            )
        placements += tuple(generated)

    applied = (
        AppliedMicrostateOverride(residue_id, site.template.component_id, graph)
        if resolution.basis is MicrostateSelectionBasis.OVERRIDE
        else None
    )
    return PolymerMicrostatePatch(
        context.source,
        residue_id,
        site,
        placements,
        reposition_hydrogens=frozenset(atom.name for atom, _ in identities)
        if rebuild_existing
        else frozenset(),
        override=override,
        applied_override=applied,
        preferences=preferences,
    )


def _existing_geometry(
    context: PolymerMicrostateContext, atom_ref: AtomRef
) -> AtomGeometry | None:
    source = context.source
    index = source.constitution.resolve_atom_index(atom_ref)
    if index is not None:
        return source.geometry.atom_geometry(index)
    observation = source.provenance.ingress.observation
    if observation is not None:
        index = observation.constitution.resolve_atom_index(atom_ref)
        if index is not None:
            return observation.geometry.atom_geometry(index)
    return None


def _coordinate_stencil(
    context: PolymerMicrostateContext,
    residue_id: ResidueId,
    site: PolymerMicrostateSite,
    graph: MicrostateGraph,
    preserved: tuple[MicrostateHydrogenPlacement, ...],
) -> tuple["Mol", dict[str, int]]:
    assert Chem is not None
    source = context.source
    residue = source.constitution.residue_site_at(
        source.constitution.residue_index(residue_id)
    )
    selected = {atom.name: atom for atom in graph.atoms}
    counts = Counter(entry.parent_name for entry in preserved)
    molecule = Chem.RWMol()
    indices: dict[str, int] = {}
    positions: list[Vec3] = []
    for atom in sorted(residue.atom_sites, key=lambda entry: entry.name):
        if atom.is_hydrogen():
            continue
        native = Chem.Atom(atom.element)
        if atom.name in selected:
            target = selected[atom.name]
            native.SetFormalCharge(target.charge)
            native.SetNoImplicit(True)
            native.SetNumExplicitHs(target.hydrogens - counts[atom.name])
        indices[atom.name] = molecule.AddAtom(native)
        geometry = _existing_geometry(context, AtomRef(residue_id, atom.name))
        assert geometry is not None
        positions.append(geometry.position)

    bonds = {
        tuple(sorted((bond.atom_name_1, bond.atom_name_2))): bond.order
        for bond in site.template.definition.bonds
        if bond.atom_name_1 in indices and bond.atom_name_2 in indices
    }
    bonds.update(
        {
            tuple(sorted((bond.atom_name_1, bond.atom_name_2))): bond.order
            for bond in graph.bonds
        }
    )
    for (first, second), order in bonds.items():
        molecule.AddBond(
            indices[first],
            indices[second],
            Chem.BondType.SINGLE if order == 1 else Chem.BondType.DOUBLE,
        )

    if (
        site.kind is not PolymerChemicalSite.SIDECHAIN
        and site.linkage is PeptideLinkage.LINKED
    ):
        parent = "N" if site.kind is PolymerChemicalSite.BACKBONE_N else "C"
        external = [
            ref
            for ref in context.covalent_neighbors(AtomRef(residue_id, parent))
            if ref.residue_id != residue_id
        ]
        if len(external) != 1:
            raise ValueError(
                "linked microstate placement requires one actual peptide partner"
            )
        partner = external[0]
        partner_index = molecule.AddAtom(Chem.Atom(partner.atom_name))
        geometry = _existing_geometry(context, partner)
        assert geometry is not None
        positions.append(geometry.position)
        molecule.AddBond(indices[parent], partner_index, Chem.BondType.SINGLE)
        if parent == "N":
            oxygen = AtomRef(partner.residue_id, "O")
            oxygen_index = source.constitution.resolve_atom_index(oxygen)
            if (
                oxygen_index is None
                or source.constitution.atom_site_at(oxygen_index).element != "O"
                or oxygen not in context.covalent_neighbors(partner, order=2)
            ):
                raise ValueError(
                    "amide H placement requires the actual partner C=O anchor"
                )
            native_oxygen = molecule.AddAtom(Chem.Atom("O"))
            positions.append(source.geometry.position(oxygen_index))
            molecule.AddBond(partner_index, native_oxygen, Chem.BondType.DOUBLE)

    for entry in preserved:
        native = Chem.Atom("H")
        if entry.atom.element in {"D", "T"}:
            native.SetIsotope(2 if entry.atom.element == "D" else 3)
        index = molecule.AddAtom(native)
        positions.append(entry.geometry.position)
        molecule.AddBond(indices[entry.parent_name], index, Chem.BondType.SINGLE)
    conformer = Chem.Conformer(molecule.GetNumAtoms())
    conformer.Set3D(True)
    for index, position in enumerate(positions):
        conformer.SetAtomPosition(index, tuple(position))
    molecule.AddConformer(conformer, assignId=True)
    result = molecule.GetMol()
    Chem.SanitizeMol(result)
    for atom in graph.atoms:
        native = result.GetAtomWithIdx(indices[atom.name])
        if (
            native.GetTotalNumHs(includeNeighbors=True) != atom.hydrogens
            or native.GetFormalCharge() != atom.charge
            or native.GetNumRadicalElectrons()
        ):
            raise ValueError(
                "native placement stencil changed selected microstate chemistry"
            )
    return result, indices


def _new_hydrogen_coordinates(
    original: "Mol",
    hydrogenated: "Mol",
    indices: dict[str, int],
    graph: MicrostateGraph,
) -> dict[str, list[Vec3]]:
    conformer = hydrogenated.GetConformer()
    original_conformer = original.GetConformer()
    if any(
        tuple(conformer.GetAtomPosition(index))
        != tuple(original_conformer.GetAtomPosition(index))
        for index in range(original.GetNumAtoms())
    ):
        raise GeometryPlacementError(
            "native H placement moved a fixed coordinate anchor"
        )
    coordinates: dict[str, list[Vec3]] = defaultdict(list)
    parents = {indices[atom.name]: atom for atom in graph.atoms}
    for index, atom in parents.items():
        native = hydrogenated.GetAtomWithIdx(index)
        neighbors = [
            neighbor
            for neighbor in native.GetNeighbors()
            if neighbor.GetAtomicNum() == 1
        ]
        if (
            len(neighbors) != atom.hydrogens
            or native.GetFormalCharge() != atom.charge
            or native.GetNumRadicalElectrons()
        ):
            raise GeometryPlacementError(
                "native H placement changed selected microstate chemistry"
            )
        parent_position = Vec3.from_iterable(conformer.GetAtomPosition(index))
        for neighbor in neighbors:
            if neighbor.GetIdx() < original.GetNumAtoms():
                continue
            position = Vec3.from_iterable(conformer.GetAtomPosition(neighbor.GetIdx()))
            distance = position.distance_to(parent_position)
            if (
                not all(isfinite(value) for value in position)
                or not isfinite(distance)
                or distance <= 0
            ):
                raise GeometryPlacementError(
                    "native H placement produced an invalid bond vector"
                )
            coordinates[atom.name].append(position)
    return coordinates

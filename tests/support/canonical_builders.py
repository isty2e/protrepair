"""Canonical test-support builders over constitution and geometry facets."""

from protrepair.geometry import Vec3
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import (
    AtomSite,
    ChainSite,
    ResidueSite,
    StructureConstitution,
)
from protrepair.structure.geometry import (
    AtomGeometry,
    ResidueGeometry,
    StructureGeometry,
)
from protrepair.structure.labels import ResidueId
from protrepair.structure.polymer_blueprint import PolymerBlueprint
from protrepair.structure.provenance import (
    FileFormat,
    StructureIngress,
    StructureProvenance,
)
from protrepair.structure.topology import AtomTopology, StructureTopology
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload

CanonicalAtomPayload = tuple[AtomSite, AtomGeometry, int | None]
CanonicalResiduePayload = tuple[
    ResidueSite,
    ResidueGeometry,
    tuple[tuple[str, int | None], ...],
]
CanonicalChainPayload = tuple[
    ChainSite,
    tuple[tuple[ResidueId, ResidueGeometry], ...],
    tuple[tuple[ResidueId, tuple[tuple[str, int | None], ...]], ...],
]


def atom_payload(
    name: str,
    element: str,
    position: Vec3,
    *,
    occupancy: float = 1.0,
    b_factor: float | None = None,
    formal_charge: int | None = None,
    altloc: str | None = None,
) -> CanonicalAtomPayload:
    """Return one canonical atom payload tuple for tests."""

    return (
        AtomSite(name=name, element=element),
        AtomGeometry(
            position=Vec3.coerce(position),
            occupancy=occupancy,
            b_factor=b_factor,
            altloc=altloc,
        ),
        formal_charge,
    )


def residue_payload(
    *,
    component_id: str,
    residue_id: ResidueId,
    atoms: tuple[CanonicalAtomPayload, ...],
    is_hetero: bool = False,
) -> CanonicalResiduePayload:
    """Return one canonical residue payload tuple for tests."""

    residue_site = ResidueSite(
        component_id=component_id,
        residue_id=residue_id,
        atom_sites=tuple(atom_site for atom_site, _, _ in atoms),
        is_hetero=is_hetero,
    )
    residue_geometry = ResidueGeometry(
        atoms_by_name={
            atom_site.name: atom_geometry
            for atom_site, atom_geometry, _ in atoms
        },
    )
    formal_charge_by_atom_name = tuple(
        (atom_site.name, formal_charge)
        for atom_site, _, formal_charge in atoms
        if formal_charge is not None
    )
    return (
        residue_site,
        residue_geometry,
        formal_charge_by_atom_name,
    )


def completion_payload(
    *,
    component_id: str,
    residue_id: ResidueId,
    atoms: tuple[CanonicalAtomPayload, ...],
    is_hetero: bool = False,
) -> CompletionResiduePayload:
    """Return one completion payload built from canonical facet inputs."""

    residue_site, residue_geometry, formal_charge_by_atom_name = residue_payload(
        component_id=component_id,
        residue_id=residue_id,
        atoms=atoms,
        is_hetero=is_hetero,
    )
    return CompletionResiduePayload(
        residue_site=residue_site,
        residue_geometry=residue_geometry,
        formal_charge_by_atom_name=formal_charge_by_atom_name,
    )


def chain_payload(
    chain_id: str,
    residues: tuple[CanonicalResiduePayload, ...],
) -> CanonicalChainPayload:
    """Return one canonical chain payload tuple for tests."""

    return (
        ChainSite(
            chain_id=chain_id,
            residues=tuple(residue_site for residue_site, _, _ in residues),
        ),
        tuple(
            (residue_site.residue_id, residue_geometry)
            for residue_site, residue_geometry, _ in residues
        ),
        tuple(
            (residue_site.residue_id, formal_charge_by_atom_name)
            for residue_site, _, formal_charge_by_atom_name in residues
            if formal_charge_by_atom_name
        ),
    )


def build_structure(
    *,
    chains: tuple[CanonicalChainPayload, ...],
    ligands: tuple[CanonicalResiduePayload, ...] = (),
    source_format: FileFormat,
    source_name: str | None = None,
    polymer_blueprint: PolymerBlueprint | None = None,
) -> ProteinStructure:
    """Return one canonical structure from facet payload tuples."""

    constitution = StructureConstitution(
        chains=tuple(chain_site for chain_site, _, _ in chains),
        ligands=tuple(ligand_site for ligand_site, _, _ in ligands),
    )
    residue_geometry_by_id = {
        residue_id: residue_geometry
        for _chain_site, residue_geometry_by_id, _residue_formal_charge_by_id in chains
        for residue_id, residue_geometry in residue_geometry_by_id
    }
    residue_geometry_by_id.update(
        {
            ligand_site.residue_id: ligand_geometry
            for ligand_site, ligand_geometry, _ in ligands
        }
    )
    formal_charge_by_residue_id = {
        residue_id: dict(formal_charge_by_atom_name)
        for _chain_site, _residue_geometry_by_id, residue_formal_charge_by_id in chains
        for residue_id, formal_charge_by_atom_name in residue_formal_charge_by_id
    }
    formal_charge_by_residue_id.update(
        {
            ligand_site.residue_id: dict(ligand_formal_charge_by_atom_name)
            for (
                ligand_site,
                _ligand_geometry,
                ligand_formal_charge_by_atom_name,
            ) in ligands
        }
    )
    return ProteinStructure.from_payload(
        constitution=constitution,
        geometry=StructureGeometry(
            constitution=constitution,
            atom_geometries=tuple(
                residue_geometry_by_id[residue_site.residue_id].atom_geometry(
                    atom_site.name
                )
                for residue_site in constitution.residue_slots
                for atom_site in residue_site.atom_sites
            ),
        ),
        topology=StructureTopology(
            constitution=constitution,
            atom_topologies=tuple(
                (
                    None
                    if formal_charge is None
                    else AtomTopology(formal_charge=formal_charge)
                )
                for residue_site in constitution.residue_slots
                for atom_site in residue_site.atom_sites
                for formal_charge in (
                    formal_charge_by_residue_id.get(
                        residue_site.residue_id,
                        {},
                    ).get(atom_site.name),
                )
            ),
        ),
        polymer_blueprint=polymer_blueprint,
        provenance=StructureProvenance(
            ingress=StructureIngress(
                source_format=source_format,
                source_name=source_name,
            )
        ),
    )

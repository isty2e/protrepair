"""Synthetic and extracted fixture builders for correction-state cases."""

from pathlib import Path

from protrepair.chemistry import ComponentLibrary
from protrepair.geometry import Vec3
from protrepair.io import FileFormat, read_structure
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
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import StructureIngress, StructureProvenance
from protrepair.structure.topology import (
    AtomTopology,
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)
from protrepair.workflow.contracts import LigandPolicy
from tests.support.request_builders import ingress_options

REFINEMENT_FIXTURE_ROOT = Path("tests/fixtures/pdb/refinement")


SyntheticResiduePayload = tuple[
    ResidueSite,
    ResidueGeometry,
    tuple[tuple[str, int | None], ...],
]
SyntheticChainPayload = tuple[
    ChainSite,
    tuple[tuple[ResidueId, ResidueGeometry], ...],
    tuple[tuple[ResidueId, tuple[tuple[str, int | None], ...]], ...],
]


def build_structure(
    source_name: str,
    chains: tuple[SyntheticChainPayload, ...],
    *,
    ligands: tuple[SyntheticResiduePayload, ...] = (),
) -> ProteinStructure:
    """Return one canonical synthetic structure for correction-state fixtures."""

    ligand_sites = tuple(ligand_site for ligand_site, _, _ in ligands)
    constitution = StructureConstitution(
        chains=tuple(chain_site for chain_site, _, _ in chains),
        ligands=ligand_sites,
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
        residue_id: dict(residue_formal_charge_by_atom_name)
        for (
            _chain_site,
            _residue_geometry_by_id,
            residue_formal_charge_by_id,
        ) in chains
        for (
            residue_id,
            residue_formal_charge_by_atom_name,
        ) in residue_formal_charge_by_id
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
        provenance=StructureProvenance(
            ingress=StructureIngress(
                source_format=FileFormat.PDB,
                source_name=source_name,
            )
        ),
    )


def residue_bond_specs(
    residue_id: ResidueId,
    atom_name_pairs: tuple[tuple[str, str], ...],
    *,
    provenance: BondProvenance = BondProvenance.TEMPLATE_RESOLVED,
) -> tuple[tuple[ResidueId, str, str, BondProvenance], ...]:
    """Return residue-local topology bond specs for correction-state fixtures."""

    return tuple(
        (residue_id, atom_name_1, atom_name_2, provenance)
        for atom_name_1, atom_name_2 in atom_name_pairs
    )


def with_topology_bonds(
    structure: ProteinStructure,
    *bond_specs: tuple[ResidueId, str, str, BondProvenance],
) -> ProteinStructure:
    """Return a fixture copy with canonical residue-local topology bonds."""

    topology_bonds = tuple(
        _topology_bond_from_spec(structure, bond_spec) for bond_spec in bond_specs
    )
    return ProteinStructure.from_payload(
        constitution=structure.constitution,
        geometry=structure.geometry,
        topology=StructureTopology(
            constitution=structure.constitution,
            atom_topologies=structure.topology.atom_topologies,
            bonds=(*structure.topology.bonds, *topology_bonds),
        ),
        polymer_blueprint=structure.polymer_blueprint,
        provenance=structure.provenance,
    )


def _topology_bond_from_spec(
    structure: ProteinStructure,
    bond_spec: tuple[ResidueId, str, str, BondProvenance],
) -> TopologyBond:
    """Return one topology bond from a residue-local fixture spec."""

    residue_id, atom_name_1, atom_name_2, provenance = bond_spec
    return TopologyBond(
        atom_index_1=structure.constitution.atom_index(
            AtomRef(residue_id, atom_name_1)
        ),
        atom_index_2=structure.constitution.atom_index(
            AtomRef(residue_id, atom_name_2)
        ),
        relationship_type=BondRelationshipType.COVALENT,
        provenance=provenance,
    )


def build_chain(
    chain_id: str,
    residues: tuple[SyntheticResiduePayload, ...],
) -> SyntheticChainPayload:
    """Return one canonical synthetic chain."""

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
            (
                residue_site.residue_id,
                formal_charge_by_atom_name,
            )
            for residue_site, _, formal_charge_by_atom_name in residues
            if formal_charge_by_atom_name
        ),
    )


def build_residue(
    component_id: str,
    chain_id: str,
    seq_num: int,
    atom_names: tuple[str, ...],
    *,
    is_hetero: bool = False,
) -> SyntheticResiduePayload:
    """Return one canonical synthetic residue with deterministic coordinates."""

    atom_sites: list[AtomSite] = []
    atom_geometries: dict[str, AtomGeometry] = {}
    for atom_index, atom_name in enumerate(atom_names, start=1):
        atom_site = AtomSite(
            name=atom_name,
            element=infer_element(atom_name),
        )
        atom_sites.append(atom_site)
        atom_geometries[atom_site.name] = AtomGeometry(
            position=deterministic_position(atom_index)
        )

    residue_id = ResidueId(chain_id=chain_id, seq_num=seq_num)
    return (
        ResidueSite(
            component_id=component_id,
            residue_id=residue_id,
            atom_sites=tuple(atom_sites),
            is_hetero=is_hetero,
        ),
        ResidueGeometry(
            atoms_by_name=atom_geometries,
        ),
        (),
    )


def deterministic_position(atom_index: int) -> Vec3:
    """Return one deterministic synthetic atom coordinate."""

    preset_positions = (
        Vec3(0.000, 0.000, 0.000),
        Vec3(1.458, 0.000, 0.000),
        Vec3(2.028, 1.417, 0.000),
        Vec3(3.235, 1.593, 0.248),
        Vec3(1.145, -0.842, 1.074),
        Vec3(2.318, -1.152, 1.556),
        Vec3(3.476, -0.240, 1.883),
        Vec3(0.271, 0.913, -0.842),
    )
    return preset_positions[(atom_index - 1) % len(preset_positions)]


def infer_element(atom_name: str) -> str:
    """Infer one simple element token for synthetic atom names."""

    normalized_atom_name = atom_name.strip().upper()
    if normalized_atom_name.startswith("CL"):
        return "CL"
    if normalized_atom_name.startswith("BR"):
        return "BR"
    return normalized_atom_name[0]


def refinement_fixture_path(filename: str) -> Path:
    """Return one canonical extracted refinement-fixture path."""

    return REFINEMENT_FIXTURE_ROOT / filename


def load_refinement_fixture(
    fixture_filename: str,
) -> ProteinStructure:
    """Load one extracted refinement fixture through canonical I/O policy."""

    return read_structure(
        refinement_fixture_path(fixture_filename),
        policy=ingress_options(
            ligand_policy=LigandPolicy.KEEP
        ).structure_normalization_policy(),
    )


def hydrogenated_refinement_fixture(
    case_id: str,
    *,
    component_library: ComponentLibrary,
) -> ProteinStructure:
    """Return one extracted refinement fixture after canonical hydrogenation."""

    fixture_filename_by_case_id = {
        "1bkr-thr101": "1bkr_thr101_local.pdb",
        "3g8l-asn182": "3g8l_asn182_local.pdb",
        "1xgo-leu253": "1xgo_leu253_local.pdb",
        "1ywr-asn155": "1ywr_asn155_local.pdb",
    }
    structure = load_refinement_fixture(fixture_filename_by_case_id[case_id])
    polymer_hydrogenated_structure = add_hydrogens(
        structure,
        component_library=component_library,
        local_refinement=None,
    ).structure
    return add_retained_non_polymer_hydrogens(
        polymer_hydrogenated_structure,
        component_library=component_library,
    ).structure

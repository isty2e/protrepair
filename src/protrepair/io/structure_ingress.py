"""Canonical raw-structure ingress normalization transformations."""

from dataclasses import dataclass

from protrepair.chemistry.component.defaults import build_default_component_library
from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.errors import StructureNormalizationError
from protrepair.geometry import Vec3
from protrepair.io.gemmi_normalization import (
    gemmi,
    normalize_altloc,
    normalize_chain_id,
    normalize_formal_charge,
    normalize_insertion_code,
)
from protrepair.io.ingress_policy import (
    MutationPolicy,
    OccupancyPolicy,
    StructureNormalizationPolicy,
)
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
from protrepair.structure.provenance import (
    FileFormat,
    StructureIngress,
    StructureProvenance,
)
from protrepair.structure.slots import ResidueIndex
from protrepair.structure.topology import (
    AtomTopology,
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    StructureTopology,
    TopologyBond,
)


@dataclass(frozen=True, slots=True)
class _NormalizedResiduePayload:
    """One normalized residue payload over constitution and geometry facets."""

    constitution: ResidueSite
    geometry: ResidueGeometry
    formal_charge_by_atom_name: tuple[tuple[str, int | None], ...]

    @property
    def residue_id(self) -> ResidueId:
        """Return the canonical residue identifier."""

        return self.constitution.residue_id

    @property
    def component_id(self) -> str:
        """Return the canonical residue component identifier."""

        return self.constitution.component_id

    def occupancy_score(self) -> float:
        """Aggregate atom occupancies for residue-variant comparison."""

        return sum(
            atom_geometry.occupancy
            for atom_geometry in self.geometry.atoms_by_name.values()
        )


@dataclass(frozen=True, slots=True)
class _SourceExplicitInterResidueConnection:
    """Ingress-local source connection before lowering to TopologyBond."""

    atom_ref_1: AtomRef
    atom_ref_2: AtomRef
    relationship_type: BondRelationshipType
    source_id: str | None = None
    reported_distance_angstrom: float | None = None

    def __post_init__(self) -> None:
        atom_ref_1 = self.atom_ref_1
        atom_ref_2 = self.atom_ref_2
        if atom_ref_1 == atom_ref_2:
            raise ValueError("source connections require two distinct atoms")
        if atom_ref_2 < atom_ref_1:
            atom_ref_1, atom_ref_2 = atom_ref_2, atom_ref_1

        if not isinstance(self.relationship_type, BondRelationshipType):
            raise TypeError(
                "source connection relationship_type must be a BondRelationshipType"
            )

        source_id = None if self.source_id is None else self.source_id.strip() or None
        reported_distance = self.reported_distance_angstrom
        if reported_distance is not None:
            if isinstance(reported_distance, bool) or reported_distance <= 0.0:
                raise ValueError(
                    "source connection reported distance must be positive or None"
                )
            reported_distance = float(reported_distance)

        object.__setattr__(self, "atom_ref_1", atom_ref_1)
        object.__setattr__(self, "atom_ref_2", atom_ref_2)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "reported_distance_angstrom", reported_distance)


def normalize_raw_structure(
    raw_structure,
    *,
    file_format: FileFormat,
    policy: StructureNormalizationPolicy,
    source_name: str | None = None,
    pdb_conect_atom_ref_pairs: tuple[tuple[AtomRef, AtomRef], ...] = (),
) -> ProteinStructure:
    """Normalize one raw gemmi structure into the canonical structure model."""

    chains: list[tuple[str, list[_NormalizedResiduePayload]]] = []
    ligands: list[_NormalizedResiduePayload] = []

    if len(raw_structure) == 0:
        empty_constitution = StructureConstitution(chains=(), ligands=())
        return ProteinStructure.from_payload(
            constitution=empty_constitution,
            geometry=StructureGeometry(
                constitution=empty_constitution,
                atom_geometries=(),
            ),
            topology=StructureTopology.empty(constitution=empty_constitution),
            provenance=StructureProvenance(
                ingress=StructureIngress(
                    source_format=file_format,
                    source_name=source_name,
                )
            ),
        )

    model = raw_structure[0]
    for raw_chain in model:
        chain_id = normalize_chain_id(raw_chain.name)
        if not policy.selects_chain(chain_id):
            continue

        polymer_residues = _normalize_polymer_residues(raw_chain, chain_id, policy)
        chain_ligands = _normalize_ligands(raw_chain, chain_id, policy)
        if polymer_residues:
            chains.append((chain_id, polymer_residues))
        ligands.extend(chain_ligands)

    constitution = StructureConstitution(
        chains=tuple(
            ChainSite(
                chain_id=chain_id,
                residues=tuple(
                    residue_payload.constitution for residue_payload in residues
                ),
            )
            for chain_id, residues in chains
        ),
        ligands=tuple(residue_payload.constitution for residue_payload in ligands),
    )
    normalized_residue_payloads = (
        *(
            residue_payload
            for _, residue_payloads in chains
            for residue_payload in residue_payloads
        ),
        *ligands,
    )
    geometry = _structure_geometry_from_payloads(
        constitution=constitution,
        residue_payloads=normalized_residue_payloads,
    )
    source_connections = _source_inter_residue_connections_from_raw_structure(
        raw_structure,
        constitution=constitution,
    )
    connection_record_type = (
        SourceBondRecordType.PDB_LINK
        if file_format is FileFormat.PDB
        else SourceBondRecordType.MMCIF_STRUCT_CONN
    )
    connection_bonds = _topology_bonds_from_source_connections(
        source_connections,
        constitution=constitution,
        record_type=connection_record_type,
    )
    conect_bonds = _topology_bonds_from_conect_pairs(
        pdb_conect_atom_ref_pairs,
        constitution=constitution,
    )
    connection_endpoint_pairs = frozenset(
        bond.endpoint_pair() for bond in connection_bonds
    )
    # LINK/struct_conn carries a resolved relationship type and optional reported
    # distance, so it takes precedence over duplicate PDB CONECT endpoint pairs.
    source_topology_bonds = connection_bonds + tuple(
        bond
        for bond in conect_bonds
        if bond.endpoint_pair() not in connection_endpoint_pairs
    )
    source_endpoint_pairs = frozenset(
        bond.endpoint_pair() for bond in source_topology_bonds
    )
    template_bonds = tuple(
        bond
        for bond in _template_resolved_topology_bonds(
            constitution,
            component_library=build_default_component_library(),
        )
        if bond.endpoint_pair() not in source_endpoint_pairs
    )
    template_endpoint_pairs = frozenset(bond.endpoint_pair() for bond in template_bonds)
    sequence_bonds = tuple(
        bond
        for bond in _sequence_inferred_topology_bonds(constitution)
        if bond.endpoint_pair() not in source_endpoint_pairs
        and bond.endpoint_pair() not in template_endpoint_pairs
    )
    topology_bonds = source_topology_bonds + template_bonds + sequence_bonds
    return ProteinStructure.from_payload(
        constitution=constitution,
        geometry=geometry,
        topology=StructureTopology(
            constitution=constitution,
            atom_topologies=_atom_topologies_from_payloads(
                constitution=constitution,
                residue_payloads=normalized_residue_payloads,
            ),
            bonds=topology_bonds,
        ),
        provenance=StructureProvenance(
            ingress=StructureIngress(
                source_format=file_format,
                source_name=source_name,
            )
        ),
    )


def _structure_geometry_from_payloads(
    *,
    constitution: StructureConstitution,
    residue_payloads: tuple[_NormalizedResiduePayload, ...],
) -> StructureGeometry:
    """Build structure geometry from normalized residue facet payloads."""

    payload_by_residue_id = {
        residue_payload.residue_id: residue_payload
        for residue_payload in residue_payloads
    }
    return StructureGeometry(
        constitution=constitution,
        atom_geometries=tuple(
            payload_by_residue_id[residue_site.residue_id].geometry.atom_geometry(
                atom_site.name
            )
            for residue_site in constitution.residue_slots
            for atom_site in residue_site.atom_sites
        ),
    )


def _atom_topologies_from_payloads(
    *,
    constitution: StructureConstitution,
    residue_payloads: tuple[_NormalizedResiduePayload, ...],
) -> tuple[AtomTopology | None, ...]:
    """Build slot-aligned atom topology payloads from normalized residues."""

    formal_charge_by_residue_id = {
        residue_payload.residue_id: dict(residue_payload.formal_charge_by_atom_name)
        for residue_payload in residue_payloads
    }
    return tuple(
        (None if formal_charge is None else AtomTopology(formal_charge=formal_charge))
        for residue_site in constitution.residue_slots
        for atom_site in residue_site.atom_sites
        for formal_charge in (
            formal_charge_by_residue_id.get(residue_site.residue_id, {}).get(
                atom_site.name
            ),
        )
    )


def apply_structure_normalization_policy(
    structure: ProteinStructure,
    *,
    policy: StructureNormalizationPolicy,
) -> ProteinStructure:
    """Apply one canonical normalization policy to an already canonical structure."""

    if policy.selected_chain_ids is None:
        selected_chain_ids = structure.constitution.chain_ids()
        selected_constitution = structure.constitution
    else:
        selected_chain_ids = tuple(
            chain_id.strip() for chain_id in policy.selected_chain_ids
        )
        selected_constitution = structure.constitution.with_chains(
            structure.constitution.chain(chain_id) for chain_id in selected_chain_ids
        )

    if policy.drops_ligands():
        selected_ligands = ()
    else:
        selected_chain_id_set = frozenset(selected_chain_ids)
        selected_ligands = tuple(
            ligand
            for ligand in structure.constitution.ligands
            if ligand.residue_id.chain_id in selected_chain_id_set
        )

    if (
        selected_constitution == structure.constitution
        and selected_ligands == structure.constitution.ligands
    ):
        return structure

    selected_structure_constitution = selected_constitution.with_ligands(
        selected_ligands
    )
    selected_residue_ids = {
        residue_site.residue_id
        for residue_site in selected_structure_constitution.iter_residues(
            include_ligands=True
        )
    }
    selected_geometry = StructureGeometry(
        constitution=selected_structure_constitution,
        atom_geometries=tuple(
            structure.residue_geometry(
                structure.constitution.residue_index(residue_site.residue_id)
            ).atom_geometry(atom_site.name)
            for residue_site in selected_structure_constitution.residue_slots
            for atom_site in residue_site.atom_sites
        ),
    )
    selected_formal_charges = {
        residue_site.residue_id: dict(
            structure.residue_formal_charge_by_atom_name(
                structure.constitution.residue_index(residue_site.residue_id)
            )
        )
        for residue_site in selected_structure_constitution.residue_slots
        if residue_site.residue_id in selected_residue_ids
    }
    selected_polymer_blueprint = (
        structure.polymer_blueprint
        if structure.polymer_blueprint is None or policy.selected_chain_ids is None
        else structure.polymer_blueprint.select_chains(selected_chain_ids)
    )

    return ProteinStructure.from_payload(
        constitution=selected_structure_constitution,
        geometry=selected_geometry,
        topology=StructureTopology(
            constitution=selected_structure_constitution,
            atom_topologies=tuple(
                (
                    None
                    if formal_charge is None
                    else AtomTopology(formal_charge=formal_charge)
                )
                for residue_site in selected_structure_constitution.residue_slots
                for atom_site in residue_site.atom_sites
                for formal_charge in (
                    selected_formal_charges.get(residue_site.residue_id, {}).get(
                        atom_site.name
                    ),
                )
            ),
            bonds=structure.topology.bonds_for_constitution(
                source_constitution=structure.constitution,
                target_constitution=selected_structure_constitution,
            ),
        ),
        polymer_blueprint=selected_polymer_blueprint,
        provenance=structure.provenance,
    )


def _source_inter_residue_connections_from_raw_structure(
    raw_structure,
    *,
    constitution: StructureConstitution,
) -> tuple[_SourceExplicitInterResidueConnection, ...]:
    """Extract source-declared inter-residue connections from gemmi."""

    connections: list[_SourceExplicitInterResidueConnection] = []
    for connection in raw_structure.connections:
        atom_ref_1 = _atom_ref_from_connection_partner(connection.partner1)
        atom_ref_2 = _atom_ref_from_connection_partner(connection.partner2)
        if atom_ref_1 is None or atom_ref_2 is None:
            continue
        if atom_ref_1.residue_id == atom_ref_2.residue_id:
            continue
        if (
            constitution.resolve_atom_site(atom_ref_1) is None
            or constitution.resolve_atom_site(atom_ref_2) is None
        ):
            continue

        reported_distance = _normalize_reported_connection_distance(
            connection.reported_distance
        )
        connections.append(
            _SourceExplicitInterResidueConnection(
                atom_ref_1=atom_ref_1,
                atom_ref_2=atom_ref_2,
                relationship_type=_relationship_type_from_connection(connection.type),
                source_id=connection.link_id or connection.name,
                reported_distance_angstrom=reported_distance,
            )
        )

    return tuple(dict.fromkeys(connections))


def _atom_ref_from_connection_partner(partner) -> AtomRef | None:
    """Return a canonical atom reference for one gemmi connection partner."""

    chain_id = normalize_chain_id(partner.chain_name)
    atom_name = partner.atom_name.strip()
    if not atom_name:
        return None

    seqid = partner.res_id.seqid
    return AtomRef(
        residue_id=ResidueId(
            chain_id=chain_id,
            seq_num=int(seqid.num),
            insertion_code=normalize_insertion_code(seqid.icode),
        ),
        atom_name=atom_name,
    )


def _relationship_type_from_connection(
    connection_type,
) -> BondRelationshipType:
    """Map gemmi connection types into topology relationship types."""

    if gemmi is None:
        return BondRelationshipType.UNKNOWN

    if connection_type is gemmi.ConnectionType.Covale:
        return BondRelationshipType.COVALENT
    if connection_type is gemmi.ConnectionType.Disulf:
        return BondRelationshipType.DISULFIDE
    if connection_type is gemmi.ConnectionType.Hydrog:
        return BondRelationshipType.HYDROGEN_BOND
    if connection_type is gemmi.ConnectionType.MetalC:
        return BondRelationshipType.METAL_COORDINATION

    return BondRelationshipType.UNKNOWN


def _normalize_reported_connection_distance(
    reported_distance: float,
) -> float | None:
    """Return a positive source-reported connection distance when available."""

    if reported_distance <= 0.0:
        return None

    return float(reported_distance)


def _normalize_polymer_residues(
    raw_chain,
    chain_id: str,
    policy: StructureNormalizationPolicy,
) -> list[_NormalizedResiduePayload]:
    """Normalize polymer residues in one raw chain."""

    grouped_residues: dict[ResidueId, list[_NormalizedResiduePayload]] = {}
    residue_order: list[ResidueId] = []

    for raw_residue in raw_chain:
        if _is_water_residue(raw_residue) or _is_ligand_residue(raw_residue):
            continue

        residue = _normalize_residue(
            raw_residue,
            chain_id,
            policy.occupancy_policy,
        )
        residue_id = residue.residue_id
        if residue_id not in grouped_residues:
            grouped_residues[residue_id] = []
            residue_order.append(residue_id)

        grouped_residues[residue_id].append(residue)

    return [
        _select_residue_variant(grouped_residues[residue_id], policy.mutation_policy)
        for residue_id in residue_order
    ]


def _normalize_ligands(
    raw_chain,
    chain_id: str,
    policy: StructureNormalizationPolicy,
) -> list[_NormalizedResiduePayload]:
    """Normalize ligand residues in one raw chain under one normalization policy."""

    if policy.drops_ligands():
        return []

    ligands: list[_NormalizedResiduePayload] = []
    for raw_residue in raw_chain:
        if not _is_ligand_residue(raw_residue):
            continue

        if policy.rejects_ligands():
            residue_id = ResidueId(
                chain_id=chain_id,
                seq_num=int(raw_residue.seqid.num),
                insertion_code=normalize_insertion_code(raw_residue.seqid.icode),
            )
            raise StructureNormalizationError(
                "structure normalization rejected unexpected ligand "
                f"{raw_residue.name} at {residue_id.display_token()}"
            )

        ligands.append(
            _normalize_residue(raw_residue, chain_id, policy.occupancy_policy)
        )

    return ligands


def _normalize_residue(
    raw_residue,
    chain_id: str,
    occupancy_policy: OccupancyPolicy,
) -> _NormalizedResiduePayload:
    """Normalize one raw gemmi residue into the canonical residue entity."""

    residue_id = ResidueId(
        chain_id=chain_id,
        seq_num=int(raw_residue.seqid.num),
        insertion_code=normalize_insertion_code(raw_residue.seqid.icode),
    )
    selected_atom_payloads = _select_atom_variants(raw_residue, occupancy_policy)
    return _NormalizedResiduePayload(
        constitution=ResidueSite(
            component_id=raw_residue.name,
            residue_id=residue_id,
            atom_sites=tuple(atom_site for atom_site, _, _ in selected_atom_payloads),
            is_hetero=_is_ligand_residue(raw_residue),
        ),
        geometry=ResidueGeometry(
            atoms_by_name={
                atom_site.name: atom_geometry
                for atom_site, atom_geometry, _ in selected_atom_payloads
            },
        ),
        formal_charge_by_atom_name=tuple(
            (atom_site.name, formal_charge)
            for atom_site, _, formal_charge in selected_atom_payloads
            if formal_charge is not None
        ),
    )


def _select_atom_variants(
    raw_residue,
    occupancy_policy: OccupancyPolicy,
) -> list[tuple[AtomSite, AtomGeometry, int | None]]:
    """Resolve duplicate atom sites by atom name using one occupancy policy."""

    selected_atoms = {}
    atom_order: list[str] = []

    for raw_atom in raw_residue:
        atom_name = raw_atom.name.strip().upper()
        if atom_name not in selected_atoms:
            selected_atoms[atom_name] = raw_atom
            atom_order.append(atom_name)
            continue

        current_atom = selected_atoms[atom_name]
        if _should_replace_atom(
            float(current_atom.occ),
            float(raw_atom.occ),
            occupancy_policy,
        ):
            selected_atoms[atom_name] = raw_atom

    return [
        _atom_payload_from_raw_site(selected_atoms[atom_name])
        for atom_name in atom_order
    ]


def _should_replace_atom(
    current_occupancy: float,
    candidate_occupancy: float,
    occupancy_policy: OccupancyPolicy,
) -> bool:
    """Decide whether a candidate atom site should replace the current choice."""

    if occupancy_policy is OccupancyPolicy.LOWEST:
        return candidate_occupancy < current_occupancy

    return candidate_occupancy > current_occupancy


def _atom_payload_from_raw_site(
    raw_atom,
) -> tuple[AtomSite, AtomGeometry, int | None]:
    """Project one selected gemmi atom site into constitution and geometry payload."""

    altloc = normalize_altloc(raw_atom.altloc)
    return (
        AtomSite(
            name=raw_atom.name,
            element=raw_atom.element.name,
        ),
        AtomGeometry(
            position=Vec3(
                x=float(raw_atom.pos.x),
                y=float(raw_atom.pos.y),
                z=float(raw_atom.pos.z),
            ),
            occupancy=float(raw_atom.occ),
            b_factor=float(raw_atom.b_iso),
            altloc=altloc,
        ),
        normalize_formal_charge(int(raw_atom.charge)),
    )


def _select_residue_variant(
    residues: list[_NormalizedResiduePayload],
    mutation_policy: MutationPolicy,
) -> _NormalizedResiduePayload:
    """Resolve duplicate residue ids produced by microheterogeneity."""

    if len(residues) == 1:
        return residues[0]

    best_residue = residues[0]
    for residue in residues[1:]:
        if _should_replace_residue(best_residue, residue, mutation_policy):
            best_residue = residue

    return best_residue


def _should_replace_residue(
    current_residue: _NormalizedResiduePayload,
    candidate_residue: _NormalizedResiduePayload,
    mutation_policy: MutationPolicy,
) -> bool:
    """Decide whether a residue variant should replace the current choice."""

    current_score = current_residue.occupancy_score()
    candidate_score = candidate_residue.occupancy_score()
    if mutation_policy is MutationPolicy.LOWEST_OCCUPANCY:
        return candidate_score < current_score

    return candidate_score > current_score


def _is_ligand_residue(raw_residue) -> bool:
    """Return whether one raw residue should be classified as a ligand."""

    return bool(
        not _is_water_residue(raw_residue)
        and (
            raw_residue.het_flag == "H"
            or raw_residue.entity_type is gemmi.EntityType.NonPolymer
        )
    )


def _is_water_residue(raw_residue) -> bool:
    """Return whether one raw residue represents water."""

    return bool(
        raw_residue.is_water() or raw_residue.entity_type is gemmi.EntityType.Water
    )


def _topology_bonds_from_source_connections(
    connections: tuple[_SourceExplicitInterResidueConnection, ...],
    *,
    constitution: StructureConstitution,
    record_type: SourceBondRecordType,
) -> tuple[TopologyBond, ...]:
    """Project source-explicit inter-residue connections into topology bonds."""

    bonds: list[TopologyBond] = []
    for link in connections:
        if (
            constitution.resolve_atom_site(link.atom_ref_1) is None
            or constitution.resolve_atom_site(link.atom_ref_2) is None
        ):
            continue

        bonds.append(
            TopologyBond(
                atom_index_1=constitution.atom_index(link.atom_ref_1),
                atom_index_2=constitution.atom_index(link.atom_ref_2),
                order=1,
                relationship_type=link.relationship_type,
                provenance=BondProvenance.SOURCE_EXPLICIT,
                source_metadata=SourceBondMetadata(
                    record_type=record_type,
                    source_id=link.source_id,
                    reported_distance_angstrom=link.reported_distance_angstrom,
                ),
            )
        )
    return tuple(bonds)


def _topology_bonds_from_conect_pairs(
    pairs: tuple[tuple[AtomRef, AtomRef], ...],
    *,
    constitution: StructureConstitution,
) -> tuple[TopologyBond, ...]:
    """Project PDB CONECT atom-ref pairs into canonical topology bonds."""

    bonds: list[TopologyBond] = []
    for atom_ref_1, atom_ref_2 in pairs:
        if (
            constitution.resolve_atom_site(atom_ref_1) is None
            or constitution.resolve_atom_site(atom_ref_2) is None
        ):
            continue

        bonds.append(
            TopologyBond(
                atom_index_1=constitution.atom_index(atom_ref_1),
                atom_index_2=constitution.atom_index(atom_ref_2),
                order=1,
                relationship_type=BondRelationshipType.UNKNOWN,
                provenance=BondProvenance.SOURCE_EXPLICIT,
                source_metadata=SourceBondMetadata(
                    record_type=SourceBondRecordType.PDB_CONECT,
                    source_id="CONECT",
                ),
            )
        )
    return tuple(bonds)


def _template_resolved_topology_bonds(
    constitution: StructureConstitution,
    *,
    component_library: ComponentLibrary,
) -> tuple[TopologyBond, ...]:
    """Return template-resolved topology bonds for materialized residue atoms."""

    bonds: list[TopologyBond] = []
    for residue_index, residue_site in enumerate(constitution.residue_slots):
        template = component_library.get(residue_site.component_id)
        if template is None:
            continue

        present_atom_names = frozenset(
            atom_site.name for atom_site in residue_site.atom_sites
        )
        for bond_definition in template.definition.bonds:
            topology_bond = _topology_bond_from_residue_template_bond(
                constitution,
                residue_index=ResidueIndex(residue_index),
                bond_definition=bond_definition,
                present_atom_names=present_atom_names,
            )
            if topology_bond is not None:
                bonds.append(topology_bond)

    return tuple(bonds)


def _topology_bond_from_residue_template_bond(
    constitution: StructureConstitution,
    *,
    residue_index: ResidueIndex,
    bond_definition: BondDefinition,
    present_atom_names: frozenset[str],
) -> TopologyBond | None:
    """Project one template bond when both endpoint atoms are present."""

    if (
        bond_definition.atom_name_1 not in present_atom_names
        or bond_definition.atom_name_2 not in present_atom_names
    ):
        return None

    return TopologyBond(
        atom_index_1=constitution.atom_index_in_residue(
            residue_index,
            bond_definition.atom_name_1,
        ),
        atom_index_2=constitution.atom_index_in_residue(
            residue_index,
            bond_definition.atom_name_2,
        ),
        order=bond_definition.order,
        aromatic=bond_definition.aromatic,
        relationship_type=BondRelationshipType.COVALENT,
        provenance=BondProvenance.TEMPLATE_RESOLVED,
    )


def _sequence_inferred_topology_bonds(
    constitution: StructureConstitution,
) -> tuple[TopologyBond, ...]:
    """Return sequence-inferred polymer backbone connectivity bonds."""

    bonds: list[TopologyBond] = []
    for chain_site in constitution.chains:
        for left_residue, right_residue in zip(
            chain_site.residues,
            chain_site.residues[1:],
            strict=False,
        ):
            if not _residue_sites_are_peptide_bonded(left_residue, right_residue):
                continue

            bonds.append(
                TopologyBond(
                    atom_index_1=constitution.atom_index_in_residue(
                        constitution.residue_index(left_residue.residue_id),
                        "C",
                    ),
                    atom_index_2=constitution.atom_index_in_residue(
                        constitution.residue_index(right_residue.residue_id),
                        "N",
                    ),
                    relationship_type=BondRelationshipType.COVALENT,
                    provenance=BondProvenance.SEQUENCE_INFERRED,
                )
            )

    return tuple(bonds)


def _residue_sites_are_peptide_bonded(
    left_residue: ResidueSite,
    right_residue: ResidueSite,
) -> bool:
    """Return whether adjacent polymer residues imply a peptide C-N bond."""

    return (
        not left_residue.is_hetero
        and not right_residue.is_hetero
        and _residue_ids_are_sequential_peptide_neighbors(
            left_residue.residue_id,
            right_residue.residue_id,
        )
        and left_residue.has_atom_site("C")
        and right_residue.has_atom_site("N")
    )


def _residue_ids_are_sequential_peptide_neighbors(
    left_residue_id: ResidueId,
    right_residue_id: ResidueId,
) -> bool:
    """Return whether residue identifiers imply adjacent peptide neighbors."""

    if left_residue_id.chain_id != right_residue_id.chain_id:
        return False
    if (
        left_residue_id.insertion_code is not None
        or right_residue_id.insertion_code is not None
    ):
        return right_residue_id.seq_num in {
            left_residue_id.seq_num,
            left_residue_id.seq_num + 1,
        }

    return right_residue_id.seq_num == left_residue_id.seq_num + 1

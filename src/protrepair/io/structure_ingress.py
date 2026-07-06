"""Canonical raw-structure ingress normalization transformations."""

from dataclasses import dataclass
from math import isfinite

from protrepair.chemistry.component.defaults import build_default_component_library
from protrepair.chemistry.component.topology import template_resolved_topology_bonds
from protrepair.errors import StructureNormalizationError
from protrepair.geometry import Vec3
from protrepair.io.gemmi_normalization import (
    gemmi,
    normalize_formal_charge,
)
from protrepair.io.ingress_policy import (
    MutationPolicy,
    OccupancyPolicy,
    StructureNormalizationPolicy,
)
from protrepair.io.source_identity import (
    SourceAtomIdentity,
    normalize_altloc,
    normalize_chain_id,
    normalize_insertion_code,
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
        """Return mean atom occupancy for residue-variant comparison."""

        occupancies = tuple(
            atom_geometry.occupancy
            for atom_geometry in self.geometry.atoms_by_name.values()
        )
        if not occupancies:
            return 0.0

        return sum(occupancies) / len(occupancies)


@dataclass(frozen=True, slots=True)
class _SourceExplicitInterResidueConnection:
    """Ingress-local source connection before lowering to TopologyBond."""

    endpoint_1: SourceAtomIdentity
    endpoint_2: SourceAtomIdentity
    relationship_type: BondRelationshipType
    source_id: str | None = None
    reported_distance_angstrom: float | None = None

    def __post_init__(self) -> None:
        endpoint_1 = self.endpoint_1
        endpoint_2 = self.endpoint_2
        if endpoint_1.atom_ref == endpoint_2.atom_ref:
            raise ValueError("source connections require two distinct atoms")
        if endpoint_2.sort_key() < endpoint_1.sort_key():
            endpoint_1, endpoint_2 = endpoint_2, endpoint_1

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

        object.__setattr__(self, "endpoint_1", endpoint_1)
        object.__setattr__(self, "endpoint_2", endpoint_2)
        object.__setattr__(self, "source_id", source_id)
        object.__setattr__(self, "reported_distance_angstrom", reported_distance)


@dataclass(frozen=True, slots=True)
class _RawAtomPayload:
    """Validated raw atom projection before residue-level variant selection."""

    atom_site: AtomSite
    geometry: AtomGeometry
    formal_charge: int | None

    @property
    def atom_name_key(self) -> str:
        """Return the canonical atom-name key used for duplicate resolution."""

        return self.atom_site.name.strip().upper()


def normalize_raw_structure(
    raw_structure,
    *,
    file_format: FileFormat,
    policy: StructureNormalizationPolicy,
    source_name: str | None = None,
    pdb_conect_atom_identity_pairs: tuple[
        tuple[SourceAtomIdentity, SourceAtomIdentity], ...
    ] = (),
) -> ProteinStructure:
    """Normalize the first model from one raw gemmi structure."""

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
        geometry=geometry,
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
        pdb_conect_atom_identity_pairs,
        constitution=constitution,
        geometry=geometry,
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
        for bond in template_resolved_topology_bonds(
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

    selected_chain_id_set = frozenset(selected_chain_ids)
    selected_ligand_candidates = tuple(
        ligand
        for ligand in structure.constitution.ligands
        if ligand.residue_id.chain_id in selected_chain_id_set
    )
    if policy.rejects_ligands() and selected_ligand_candidates:
        rejected_ligand = selected_ligand_candidates[0]
        raise StructureNormalizationError(
            "structure normalization rejected unexpected ligand "
            f"{rejected_ligand.component_id} at "
            f"{rejected_ligand.residue_id.display_token()}"
        )

    if policy.drops_ligands() or policy.rejects_ligands():
        selected_ligands = ()
    else:
        selected_ligands = selected_ligand_candidates

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
    geometry: StructureGeometry,
) -> tuple[_SourceExplicitInterResidueConnection, ...]:
    """Extract source-declared inter-residue connections from gemmi."""

    connections: list[_SourceExplicitInterResidueConnection] = []
    for connection in raw_structure.connections:
        endpoint_1 = _source_atom_identity_from_connection_partner(connection.partner1)
        endpoint_2 = _source_atom_identity_from_connection_partner(connection.partner2)
        if endpoint_1 is None or endpoint_2 is None:
            continue
        if endpoint_1.atom_ref.residue_id == endpoint_2.atom_ref.residue_id:
            continue
        if not _source_endpoint_survived(
            endpoint_1, constitution, geometry=geometry
        ) or not _source_endpoint_survived(endpoint_2, constitution, geometry=geometry):
            continue

        reported_distance = _normalize_reported_connection_distance(
            connection.reported_distance
        )
        connections.append(
            _SourceExplicitInterResidueConnection(
                endpoint_1=endpoint_1,
                endpoint_2=endpoint_2,
                relationship_type=_relationship_type_from_connection(connection.type),
                source_id=connection.link_id or connection.name,
                reported_distance_angstrom=reported_distance,
            )
        )

    return tuple(dict.fromkeys(connections))


def _source_atom_identity_from_connection_partner(
    partner,
) -> SourceAtomIdentity | None:
    """Return source endpoint identity for one gemmi connection partner."""

    chain_id = normalize_chain_id(partner.chain_name)
    atom_name = partner.atom_name.strip()
    if not atom_name:
        return None
    component_id = partner.res_id.name.strip()
    if not component_id:
        return None

    seqid = partner.res_id.seqid
    return SourceAtomIdentity(
        atom_ref=AtomRef(
            residue_id=ResidueId(
                chain_id=chain_id,
                seq_num=int(seqid.num),
                insertion_code=normalize_insertion_code(seqid.icode),
            ),
            atom_name=atom_name,
        ),
        component_id=component_id,
        altloc=normalize_altloc(partner.altloc),
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

    try:
        normalized_distance = float(reported_distance)
    except (TypeError, ValueError):
        return None

    if not isfinite(normalized_distance) or normalized_distance <= 0.0:
        return None

    return normalized_distance


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

    grouped_ligands: dict[ResidueId, list[_NormalizedResiduePayload]] = {}
    ligand_order: list[ResidueId] = []
    for raw_residue in raw_chain:
        if not _is_ligand_residue(raw_residue):
            continue

        residue_id = ResidueId(
            chain_id=chain_id,
            seq_num=int(raw_residue.seqid.num),
            insertion_code=normalize_insertion_code(raw_residue.seqid.icode),
        )
        if policy.rejects_ligands():
            raise StructureNormalizationError(
                "structure normalization rejected unexpected ligand "
                f"{raw_residue.name} at {residue_id.display_token()}"
            )

        ligand = _normalize_residue(raw_residue, chain_id, policy.occupancy_policy)
        if residue_id not in grouped_ligands:
            grouped_ligands[residue_id] = []
            ligand_order.append(residue_id)

        grouped_ligands[residue_id].append(ligand)

    return [
        _select_residue_variant(grouped_ligands[residue_id], policy.mutation_policy)
        for residue_id in ligand_order
    ]


def _residue_altloc_score_by_label(
    raw_atom_payloads: tuple[_RawAtomPayload, ...],
) -> tuple[tuple[str, float], ...]:
    """Return nonblank altloc labels and mean residue-level occupancy scores."""

    score_by_altloc: dict[str, float] = {}
    count_by_altloc: dict[str, int] = {}
    altloc_order: list[str] = []
    for atom_payload in raw_atom_payloads:
        altloc = atom_payload.geometry.altloc
        if altloc is None:
            continue
        if altloc not in score_by_altloc:
            score_by_altloc[altloc] = 0.0
            count_by_altloc[altloc] = 0
            altloc_order.append(altloc)

        score_by_altloc[altloc] += atom_payload.geometry.occupancy
        count_by_altloc[altloc] += 1

    return tuple(
        (altloc, score_by_altloc[altloc] / count_by_altloc[altloc])
        for altloc in altloc_order
    )


def _select_residue_altloc(
    raw_atom_payloads: tuple[_RawAtomPayload, ...],
    occupancy_policy: OccupancyPolicy,
) -> str | None:
    """Select one residue-level altloc cohort or None for non-altloc residues."""

    altloc_scores = _residue_altloc_score_by_label(raw_atom_payloads)
    if not altloc_scores:
        return None

    selected_altloc, selected_score = altloc_scores[0]
    for candidate_altloc, candidate_score in altloc_scores[1:]:
        if candidate_score == selected_score:
            should_replace = candidate_altloc < selected_altloc
        else:
            should_replace = _should_replace_occupancy_score(
                selected_score,
                candidate_score,
                occupancy_policy,
            )
        if should_replace:
            selected_altloc = candidate_altloc
            selected_score = candidate_score

    return selected_altloc


def _raw_atoms_for_selected_altloc(
    raw_atom_payloads: tuple[_RawAtomPayload, ...],
    selected_altloc: str | None,
):
    """Yield atom payloads belonging to the selected residue-level altloc cohort."""

    for atom_payload in raw_atom_payloads:
        atom_altloc = atom_payload.geometry.altloc
        if (
            selected_altloc is None
            or atom_altloc is None
            or atom_altloc == selected_altloc
        ):
            yield atom_payload


def _select_atom_name_variants(
    raw_atom_payloads,
    *,
    selected_altloc: str | None,
    occupancy_policy: OccupancyPolicy,
) -> list[_RawAtomPayload]:
    """Resolve duplicate atom names within one selected residue altloc cohort."""

    selected_atoms = {}
    atom_order: list[str] = []

    for atom_payload in raw_atom_payloads:
        atom_name = atom_payload.atom_name_key
        if atom_name not in selected_atoms:
            selected_atoms[atom_name] = atom_payload
            atom_order.append(atom_name)
            continue

        current_atom = selected_atoms[atom_name]
        current_priority = _selected_altloc_priority(current_atom, selected_altloc)
        candidate_priority = _selected_altloc_priority(atom_payload, selected_altloc)
        if candidate_priority < current_priority:
            continue
        if candidate_priority > current_priority or _should_replace_occupancy_score(
            current_score=current_atom.geometry.occupancy,
            candidate_score=atom_payload.geometry.occupancy,
            occupancy_policy=occupancy_policy,
        ):
            selected_atoms[atom_name] = atom_payload

    return [selected_atoms[atom_name] for atom_name in atom_order]


def _selected_altloc_priority(
    atom_payload: _RawAtomPayload,
    selected_altloc: str | None,
) -> int:
    """Return whether one raw atom is explicit member of the selected cohort."""

    if selected_altloc is None:
        return 0

    return int(atom_payload.geometry.altloc == selected_altloc)


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
    selected_atom_payloads = _select_atom_variants(
        raw_residue,
        residue_id=residue_id,
        occupancy_policy=occupancy_policy,
    )
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
    *,
    residue_id: ResidueId,
    occupancy_policy: OccupancyPolicy,
) -> list[tuple[AtomSite, AtomGeometry, int | None]]:
    """Resolve atom sites by residue altloc cohort, then by atom name."""

    raw_atom_payloads = tuple(
        _atom_payload_from_raw_site(raw_atom, residue_id=residue_id)
        for raw_atom in raw_residue
    )
    selected_altloc = _select_residue_altloc(raw_atom_payloads, occupancy_policy)
    selected_atoms = _select_atom_name_variants(
        _raw_atoms_for_selected_altloc(raw_atom_payloads, selected_altloc),
        selected_altloc=selected_altloc,
        occupancy_policy=occupancy_policy,
    )
    return [
        (
            atom_payload.atom_site,
            atom_payload.geometry,
            atom_payload.formal_charge,
        )
        for atom_payload in selected_atoms
    ]


def _should_replace_occupancy_score(
    current_score: float,
    candidate_score: float,
    occupancy_policy: OccupancyPolicy,
) -> bool:
    """Decide whether a candidate occupancy score should replace current choice."""

    if occupancy_policy is OccupancyPolicy.LOWEST:
        return candidate_score < current_score

    return candidate_score > current_score


def _atom_payload_from_raw_site(
    raw_atom,
    *,
    residue_id: ResidueId,
) -> _RawAtomPayload:
    """Validate and project one raw gemmi atom before variant selection."""

    altloc = normalize_altloc(raw_atom.altloc)
    x = _validated_raw_atom_float(
        raw_atom.pos.x,
        "x coordinate",
        raw_atom.name,
        residue_id,
    )
    y = _validated_raw_atom_float(
        raw_atom.pos.y,
        "y coordinate",
        raw_atom.name,
        residue_id,
    )
    z = _validated_raw_atom_float(
        raw_atom.pos.z,
        "z coordinate",
        raw_atom.name,
        residue_id,
    )
    occupancy = _validated_raw_atom_occupancy(raw_atom, residue_id)
    b_factor = _validated_raw_atom_b_factor(raw_atom, residue_id)
    return _RawAtomPayload(
        AtomSite(
            name=raw_atom.name,
            element=raw_atom.element.name,
        ),
        AtomGeometry(
            position=Vec3(
                x=x,
                y=y,
                z=z,
            ),
            occupancy=occupancy,
            b_factor=b_factor,
            altloc=altloc,
        ),
        formal_charge=normalize_formal_charge(int(raw_atom.charge)),
    )


def _validated_raw_atom_float(
    value: float,
    field_name: str,
    atom_name: str,
    residue_id: ResidueId,
) -> float:
    """Return one finite raw atom float or raise a normalization error."""

    normalized_value = float(value)
    if not isfinite(normalized_value):
        raise StructureNormalizationError(
            "structure normalization rejected non-finite "
            f"{field_name} for atom {atom_name.strip()!r} "
            f"at {residue_id.display_token()}"
        )

    return normalized_value


def _validated_raw_atom_occupancy(raw_atom, residue_id: ResidueId) -> float:
    """Return a finite PDB/mmCIF occupancy in the accepted [0, 1] range."""

    occupancy = _validated_raw_atom_float(
        raw_atom.occ,
        "occupancy",
        raw_atom.name,
        residue_id,
    )
    if occupancy < 0.0 or occupancy > 1.0:
        raise StructureNormalizationError(
            "structure normalization rejected occupancy outside [0.0, 1.0] "
            f"for atom {raw_atom.name.strip()!r} "
            f"at {residue_id.display_token()}: {occupancy}"
        )

    return occupancy


def _validated_raw_atom_b_factor(raw_atom, residue_id: ResidueId) -> float:
    """Return a finite non-negative source B factor with no upper bound."""

    b_factor = _validated_raw_atom_float(
        raw_atom.b_iso,
        "B factor",
        raw_atom.name,
        residue_id,
    )
    if b_factor < 0.0:
        raise StructureNormalizationError(
            "structure normalization rejected negative B factor "
            f"for atom {raw_atom.name.strip()!r} "
            f"at {residue_id.display_token()}: {b_factor}"
        )

    return b_factor


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
        bonds.append(
            TopologyBond(
                atom_index_1=constitution.atom_index(link.endpoint_1.atom_ref),
                atom_index_2=constitution.atom_index(link.endpoint_2.atom_ref),
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
    pairs: tuple[tuple[SourceAtomIdentity, SourceAtomIdentity], ...],
    *,
    constitution: StructureConstitution,
    geometry: StructureGeometry,
) -> tuple[TopologyBond, ...]:
    """Project selected PDB CONECT endpoint pairs into canonical topology bonds."""

    bonds: list[TopologyBond] = []
    for endpoint_1, endpoint_2 in pairs:
        if not _source_endpoint_survived(
            endpoint_1, constitution, geometry=geometry
        ) or not _source_endpoint_survived(endpoint_2, constitution, geometry=geometry):
            continue

        atom_ref_1 = endpoint_1.atom_ref
        atom_ref_2 = endpoint_2.atom_ref
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


def _source_endpoint_survived(
    endpoint: SourceAtomIdentity,
    constitution: StructureConstitution,
    *,
    geometry: StructureGeometry,
) -> bool:
    """Return whether one source endpoint survived canonical normalization."""

    residue_site = constitution.residue_or_ligand(endpoint.atom_ref.residue_id)
    if residue_site is None or residue_site.component_id != endpoint.component_id:
        return False

    atom_index = constitution.resolve_atom_index(endpoint.atom_ref)
    if atom_index is None:
        return False

    return geometry.atom_geometry(atom_index).altloc == endpoint.altloc


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

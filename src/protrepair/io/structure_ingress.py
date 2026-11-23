"""Canonical raw-structure ingress normalization transformations."""

import itertools
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, replace
from math import isfinite

from protrepair.chemistry.component.defaults import build_default_component_library
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.component.topology import template_resolved_topology_bonds
from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.nonstandard.registry import (
    build_bundled_nonstandard_registry,
)
from protrepair.chemistry.standard.components import (
    build_standard_component_library,
)
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
    normalize_altloc,
    normalize_chain_id,
    normalize_insertion_code,
)
from protrepair.io.source_peptide import (
    SourcePolymerLabels,
    source_compatible_peptide_bonds,
    source_polymer_paths,
    source_sequence_inferred_polymer_topology_bonds,
)
from protrepair.io.source_peptide_caps import PdbCapSourceContext
from protrepair.io.zero_occupancy import (
    project_zero_occupancy_omissions,
    zero_occupancy_residue_omissions,
    zero_occupancy_sidechain_omissions,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import (
    AtomSite,
    ChainSite,
    ResidueRole,
    ResidueSite,
    StructureConstitution,
)
from protrepair.structure.geometry import (
    AtomGeometry,
    ResidueGeometry,
    StructureGeometry,
)
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.peptide import (
    PEPTIDE_CAP_COMPONENT_IDS,
    peptide_ports_allow_link,
)
from protrepair.structure.provenance import (
    FileFormat,
    StructureIngress,
    StructureProvenance,
)
from protrepair.structure.sequence_boundary import DepositedPeptideBoundary
from protrepair.structure.slots import AtomIndex
from protrepair.structure.source_connection import SourceAtomIdentity, SourceConnection
from protrepair.structure.topology import (
    AtomTopology,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)


@dataclass(frozen=True, slots=True)
class SourceAtomAnnotation:
    """Attributes of one coordinate record before native scalar conversion.

    Parameters
    ----------
    formal_charge : int or None, default=None
        Declared charge, distinguishing zero from an absent annotation.
    explicitly_zero_occupancy : bool or None, default=None
        True for a token declaring zero; None means it was not inspected.
    declared_element : str or None, default=None
        Source spelling used to detect ambiguous isotope or element custody
        when binding zero-occupancy evidence.
    explicitly_positive_occupancy : bool, default=False
        The source token is positive, when the whole-residue policy needs
        flank evidence. Missing tokens do not provide this evidence.
    """

    formal_charge: int | None = None
    explicitly_zero_occupancy: bool | None = None
    declared_element: str | None = None
    explicitly_positive_occupancy: bool = False

    def __post_init__(self) -> None:
        if self.explicitly_zero_occupancy and self.explicitly_positive_occupancy:
            raise ValueError(
                "source occupancy cannot be both explicitly zero and positive"
            )


@dataclass(frozen=True, slots=True)
class _NormalizedResiduePayload:
    """One normalized residue payload over constitution and geometry facets."""

    constitution: ResidueSite
    geometry: ResidueGeometry
    formal_charge_by_atom_name: tuple[tuple[str, int | None], ...]
    source_atom_name_projection: tuple[tuple[str, str], ...] = ()
    explicitly_zero_atom_names: frozenset[str] = frozenset()
    explicitly_positive_atom_names: frozenset[str] = frozenset()

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

    def has_positive_backbone(self) -> bool:
        """Return whether all scaffold positions have positive source evidence."""

        return {"N", "CA", "C", "O"} <= {
            atom.name
            for atom in self.constitution.atom_sites
            if atom.name in self.explicitly_positive_atom_names
            and atom.element == atom.name[0]
        }

    def with_atom_name_projection(
        self, projection: tuple[tuple[str, str], ...]
    ) -> "_NormalizedResiduePayload":
        """Project selected identities across every name-keyed payload facet."""

        if not projection:
            return self

        names = dict(projection)
        return replace(
            self,
            constitution=replace(
                self.constitution,
                atom_sites=tuple(
                    replace(atom, name=names.get(atom.name, atom.name))
                    for atom in self.constitution.atom_sites
                ),
            ),
            geometry=ResidueGeometry(
                atoms_by_name={
                    names.get(name, name): geometry
                    for name, geometry in self.geometry.atoms_by_name.items()
                }
            ),
            formal_charge_by_atom_name=tuple(
                (names.get(name, name), charge)
                for name, charge in self.formal_charge_by_atom_name
            ),
            source_atom_name_projection=projection,
            explicitly_zero_atom_names=frozenset(
                names.get(name, name) for name in self.explicitly_zero_atom_names
            ),
            explicitly_positive_atom_names=frozenset(
                names.get(name, name) for name in self.explicitly_positive_atom_names
            ),
        )


@dataclass(frozen=True, slots=True)
class _RawAtomPayload:
    """Validated raw atom projection before residue-level variant selection."""

    atom_site: AtomSite
    geometry: AtomGeometry
    formal_charge: int | None
    explicitly_zero_occupancy: bool
    explicitly_positive_occupancy: bool

    @property
    def atom_name_key(self) -> str:
        """Return the canonical atom-name key used for duplicate resolution."""

        return self.atom_site.name.strip().upper()


def _terminal_oxygen_name_projection(
    residue: ResidueSite,
    *,
    template: ResidueTemplate | None,
    formal_charge_by_atom_name: Mapping[str, int | None],
) -> tuple[tuple[str, str], ...]:
    """Choose terminal labels from declared charge or the naming convention."""

    aliases = frozenset({"OT1", "OT2", "OC1", "OC2"})
    present = frozenset(residue.atom_site_names())
    observed_aliases = present & aliases
    if not observed_aliases or residue.is_hetero or template is None:
        return ()
    expected = frozenset(template.expected_atom_names())
    if expected & aliases:
        return ()
    if not (
        build_standard_component_library().has(template.backbone_family_component_id)
        and {"N", "CA", "C", "O"} <= expected
    ):
        return ()

    if observed_aliases == {"OT1", "OT2"}:
        projection = (("OT1", "O"), ("OT2", "OXT"))
    elif observed_aliases == {"OC1", "OC2"}:
        projection = (("OC1", "OXT"), ("OC2", "O"))
    else:
        projection = ()

    error_prefix = (
        "unsupported terminal oxygen name reconciliation for "
        f"{residue.component_id} at {residue.residue_id.display_token()}: "
    )
    if not projection or present & {"O", "OXT"}:
        raise StructureNormalizationError(
            error_prefix + "requires one complete OT1/OT2 or OC1/OC2 pair without O/OXT"
        )
    if any(residue.atom_site(name).element != "O" for name in observed_aliases):
        raise StructureNormalizationError(error_prefix + "aliases must be oxygen atoms")

    site = PolymerMicrostateSite(template, PolymerChemicalSite.BACKBONE_C)
    if {"O", "OXT"} & set(site.named_hydrogen_parents(residue).values()):
        raise StructureNormalizationError(
            error_prefix + "source H/D/T names depend on the newly created O/OXT names"
        )

    for charged, _ in projection:
        if formal_charge_by_atom_name.get(charged) == -1 and all(
            formal_charge_by_atom_name.get(other) in (None, 0)
            for other, _ in projection
            if other != charged
        ):
            return tuple(
                (name, "OXT" if name == charged else "O") for name, _ in projection
            )
    return projection


def _project_source_connection_names(
    connections: tuple[SourceConnection, ...],
    residue_payloads: tuple[_NormalizedResiduePayload, ...],
) -> tuple[SourceConnection, ...]:
    """Project selected source endpoints while retaining component and altloc qualifiers."""

    projections = {
        (payload.residue_id, payload.component_id): dict(
            payload.source_atom_name_projection
        )
        for payload in residue_payloads
        if payload.source_atom_name_projection
    }
    if not projections:
        return connections

    projected = []
    for connection in connections:
        endpoints = []
        for endpoint in connection.endpoint_pair():
            names = projections.get(
                (endpoint.atom_ref.residue_id, endpoint.component_id), {}
            )
            name = names.get(endpoint.atom_ref.atom_name, endpoint.atom_ref.atom_name)
            endpoints.append(
                replace(endpoint, atom_ref=AtomRef(endpoint.atom_ref.residue_id, name))
            )
        projected.append(
            replace(connection, endpoint_1=endpoints[0], endpoint_2=endpoints[1])
        )
    return tuple(projected)


def normalize_raw_structure(
    raw_structure: gemmi.Structure,
    *,
    file_format: FileFormat,
    policy: StructureNormalizationPolicy,
    source_name: str | None = None,
    source_connections: tuple[SourceConnection, ...],
    source_element_by_atom_identity: Mapping[SourceAtomIdentity, str] | None = None,
    source_annotations_by_atom_identity: Mapping[
        SourceAtomIdentity, Sequence[SourceAtomAnnotation]
    ]
    | None = None,
    source_peptide_break_after: frozenset[ResidueId] = frozenset(),
    source_labels_by_residue: Mapping[ResidueId, frozenset[SourcePolymerLabels]]
    | None = None,
    pdb_cap_source: PdbCapSourceContext | None = None,
    deposited_peptide_boundaries: tuple[DepositedPeptideBoundary, ...] = (),
    unresolved_conect_ports: frozenset[AtomRef] = frozenset(),
) -> ProteinStructure:
    """Normalize one first model from format-boundary-projected source facts.

    Raw Gemmi connection records are intentionally interpreted by the format
    adapter before this format-agnostic normalization seam.

    Parameters
    ----------
    raw_structure : gemmi.Structure
        Parsed models; only the first is normalized.
    file_format : FileFormat
        Source format recorded in provenance.
    policy : StructureNormalizationPolicy
        Selection and validation policy for the input.
    source_name : str or None, default=None
        Optional input name.
    source_connections : tuple[SourceConnection, ...]
        Boundary-decoded declarations, before selected-atom filtering.
    source_element_by_atom_identity : Mapping[SourceAtomIdentity, str] or None
        Source isotope symbols that the native parser may collapse.
    source_annotations_by_atom_identity : Mapping or None
        Boundary-decoded annotations in source occurrence order. Zero-occupancy
        omission requires token evidence; a numeric Gemmi zero is insufficient.
    source_peptide_break_after : frozenset[ResidueId]
        Residues followed by a source TER record, even if the native parser
        did not retain that boundary.
    source_labels_by_residue : Mapping or None, default=None
        All declared atom-row polymer labels for each first-model source slot.
    pdb_cap_source : PdbCapSourceContext or None, default=None
        Raw PDB adjacency and serial ambiguity facts for retained cap selection.
    deposited_peptide_boundaries : tuple[DepositedPeptideBoundary, ...], default=()
        Positive source sequence facts computed before retention or omission.
    unresolved_conect_ports : frozenset[AtomRef], default=frozenset()
        Possible endpoints on undecodable CONECT rows in sequence-bearing PDB.

    Returns
    -------
    ProteinStructure
        Selected canonical structure with its original input observation.

    Raises
    ------
    StructureNormalizationError
        Input scalars, identities, roles, or annotations cannot be normalized.
    ModelInvariantError
        Selected facets or explicit connections violate canonical invariants.
    """

    chains: list[tuple[str, list[_NormalizedResiduePayload]]] = []
    ligands: list[_NormalizedResiduePayload] = []
    waters: list[_NormalizedResiduePayload] = []

    omit_sidechains = policy.omit_zero_occupancy_sidechain_heavy_atoms
    omit_residues = policy.omit_zero_occupancy_polymer_residues
    prove_zero_occupancy = omit_sidechains or omit_residues
    source_annotations = _source_annotation_iterators(
        source_annotations_by_atom_identity, require_occupancy=prove_zero_occupancy
    )
    if len(raw_structure) == 0:
        empty_constitution = StructureConstitution(chains=(), ligands=())
        empty_geometry = StructureGeometry(
            constitution=empty_constitution, atom_geometries=()
        )
        empty_topology = StructureTopology.empty(constitution=empty_constitution)
        return ProteinStructure.from_payload(
            constitution=empty_constitution,
            geometry=empty_geometry,
            topology=empty_topology,
            provenance=StructureProvenance(
                ingress=StructureIngress(
                    source_format=file_format,
                    source_name=source_name,
                    observation=StructureObservation(
                        constitution=empty_constitution,
                        geometry=empty_geometry,
                        topology=empty_topology,
                    ),
                    zero_occupancy_omissions=() if omit_sidechains else None,
                    zero_occupancy_residue_omissions=() if omit_residues else None,
                )
            ),
        )

    normalized_source_elements = (
        {}
        if source_element_by_atom_identity is None
        else source_element_by_atom_identity
    )
    entity_type_by_id = _entity_type_by_id(raw_structure)
    source_peptide_link_pairs = _source_peptide_link_pairs(
        raw_structure,
        source_connections,
    )
    model = raw_structure[0]
    coordination_endpoints_by_residue: dict[ResidueId, list[SourceAtomIdentity]] = {}
    for connection in source_connections:
        if (
            connection.source_metadata.reported_relationship_type
            is BondRelationshipType.METAL_COORDINATION
        ):
            for endpoint in connection.endpoint_pair():
                coordination_endpoints_by_residue.setdefault(
                    endpoint.atom_ref.residue_id, []
                ).append(endpoint)
    excluded_source_endpoints: set[SourceAtomIdentity] = set()
    reordered_polymer_chains: set[str] = set()
    for raw_chain in model:
        chain_id = normalize_chain_id(raw_chain.name)
        if not policy.selects_chain(chain_id):
            for raw_residue in raw_chain:
                excluded_source_endpoints.update(
                    _referenced_source_endpoints(
                        raw_residue, chain_id, coordination_endpoints_by_residue
                    )
                )
            continue

        residue_role_by_id = _canonical_residue_roles_for_chain(
            raw_chain,
            chain_id,
            entity_type_by_id=entity_type_by_id,
            source_peptide_link_pairs=source_peptide_link_pairs,
        )
        polymer_residues, non_polymer, excluded_endpoints, source_order_preserved = (
            _normalize_chain_residues(
                raw_chain,
                chain_id,
                policy,
                residue_role_by_id=residue_role_by_id,
                source_element_by_atom_identity=normalized_source_elements,
                source_annotations=source_annotations,
                coordination_endpoints_by_residue=coordination_endpoints_by_residue,
            )
        )
        excluded_source_endpoints.update(excluded_endpoints)
        if not source_order_preserved:
            reordered_polymer_chains.add(chain_id)
        if polymer_residues:
            chains.append((chain_id, polymer_residues))
        for payload in non_polymer:
            if residue_role_by_id[payload.residue_id] is ResidueRole.WATER:
                waters.append(payload)
            else:
                ligands.append(payload)

    component_library = build_default_component_library()
    chains = [
        (
            chain_id,
            [
                payload.with_atom_name_projection(
                    _terminal_oxygen_name_projection(
                        payload.constitution,
                        template=component_library.get(payload.component_id),
                        formal_charge_by_atom_name=dict(
                            payload.formal_charge_by_atom_name
                        ),
                    )
                )
                for payload in residues
            ],
        )
        for chain_id, residues in chains
    ]
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
        waters=tuple(payload.constitution for payload in waters),
    )
    normalized_residue_payloads = (
        *(
            residue_payload
            for _, residue_payloads in chains
            for residue_payload in residue_payloads
        ),
        *ligands,
        *waters,
    )
    geometry = _structure_geometry_from_payloads(
        constitution=constitution,
        residue_payloads=normalized_residue_payloads,
    )
    template_bonds = template_resolved_topology_bonds(
        constitution,
        component_library=component_library,
    )
    expected_bonds_by_pair = {bond.endpoint_pair(): bond for bond in template_bonds}
    atom_topologies = _atom_topologies_from_payloads(
        constitution=constitution,
        residue_payloads=normalized_residue_payloads,
    )
    successions, unverified_ports = source_polymer_paths(
        raw_structure,
        file_format=file_format,
        constitution=constitution,
        peptide_break_after=source_peptide_break_after,
        verify_source_paths=omit_residues,
        source_labels_by_residue=source_labels_by_residue,
    )
    peptide_candidates = source_sequence_inferred_polymer_topology_bonds(
        successions, constitution=constitution, geometry=geometry
    )
    for bond in peptide_candidates:
        expected_bonds_by_pair.setdefault(bond.endpoint_pair(), bond)
    projected_source_connections = _project_source_connection_names(
        source_connections, normalized_residue_payloads
    )
    source_topology_bonds = _topology_bonds_from_source_connections(
        projected_source_connections,
        constitution=constitution,
        geometry=geometry,
        expected_bonds_by_pair=expected_bonds_by_pair,
    )
    source_endpoint_pairs = frozenset(
        bond.endpoint_pair() for bond in source_topology_bonds
    )
    for bond in peptide_candidates:
        expected_bonds_by_pair.pop(bond.endpoint_pair(), None)
    for bond in source_compatible_peptide_bonds(
        peptide_candidates,
        source_bonds=source_topology_bonds,
        template_bonds=template_bonds,
        constitution=constitution,
        atom_topologies=atom_topologies,
    ):
        expected_bonds_by_pair[bond.endpoint_pair()] = bond
    remaining_expected_bonds = tuple(
        bond
        for bond in expected_bonds_by_pair.values()
        if bond.endpoint_pair() not in source_endpoint_pairs
    )
    topology_bonds = source_topology_bonds + remaining_expected_bonds
    topology = StructureTopology(
        constitution=constitution,
        atom_topologies=atom_topologies,
        bonds=topology_bonds,
    )
    observation = StructureObservation.from_source_facets(
        constitution=constitution,
        geometry=geometry,
        topology=topology,
        source_connections=projected_source_connections,
        unverified_polymer_ports=unverified_ports,
        deposited_peptide_boundaries=tuple(
            fact
            for fact in deposited_peptide_boundaries
            if fact.port.residue_id.chain_id not in reordered_polymer_chains
        ),
        unresolved_conect_ports=unresolved_conect_ports,
        excluded_coordination_connections=_excluded_coordination_connections(
            source_connections,
            excluded_source_endpoints,
            constitution=constitution,
            geometry=geometry,
            include_dropped_metals=policy.non_polymer_retention is not None,
        ),
    )
    selected = ProteinStructure.from_payload(
        constitution=constitution,
        geometry=geometry,
        topology=topology,
        provenance=StructureProvenance(
            ingress=StructureIngress(
                source_format=file_format,
                source_name=source_name,
                observation=observation,
                pdb_adjacent_cap_links=(
                    ()
                    if pdb_cap_source is None
                    else pdb_cap_source.links_for(
                        observation, projected_source_connections
                    )
                ),
            )
        ),
    )
    if not prove_zero_occupancy:
        return selected

    atom_omissions = None
    if omit_sidechains:
        atom_omissions = zero_occupancy_sidechain_omissions(
            selected,
            explicitly_zero_refs=frozenset(
                AtomRef(payload.residue_id, atom_name)
                for payload in normalized_residue_payloads
                for atom_name in payload.explicitly_zero_atom_names
            ),
            source_connections=projected_source_connections,
        )
    residue_omissions = None
    if omit_residues:
        assert source_annotations_by_atom_identity is not None
        all_zero_ids, charged_atoms = _all_zero_source_residues(
            source_annotations_by_atom_identity
        )
        residue_omissions = zero_occupancy_residue_omissions(
            selected,
            all_zero_residue_ids=frozenset(
                payload.residue_id
                for payload in normalized_residue_payloads
                if (
                    payload.residue_id in all_zero_ids
                    and payload.explicitly_zero_atom_names
                    and len(payload.explicitly_zero_atom_names)
                    == len(payload.constitution.atom_sites)
                )
            ),
            positive_backbone_residue_ids=frozenset(
                payload.residue_id
                for payload in normalized_residue_payloads
                if payload.has_positive_backbone()
            ),
            verified_neighbors=frozenset(
                (pair.preceding_residue_id, pair.following_residue_id)
                for pair in successions
                if pair.verified_source_path
            ),
            charged_source_atoms_by_residue=charged_atoms,
            source_connections=projected_source_connections,
        )
    return project_zero_occupancy_omissions(
        selected,
        atom_omissions=atom_omissions,
        residue_omissions=residue_omissions,
    )


def _all_zero_source_residues(
    annotations: Mapping[SourceAtomIdentity, Sequence[SourceAtomAnnotation]],
) -> tuple[frozenset[ResidueId], dict[ResidueId, AtomRef]]:
    """Account for every source row, even if its conformer was not selected."""

    components: dict[ResidueId, str] = {}
    ineligible: set[ResidueId] = set()
    charged: dict[ResidueId, AtomRef] = {}
    for identity, occurrences in annotations.items():
        residue_id = identity.atom_ref.residue_id
        component = components.setdefault(residue_id, identity.component_id)
        if component != identity.component_id or not all(
            item.explicitly_zero_occupancy is True for item in occurrences
        ):
            ineligible.add(residue_id)
        if any(item.formal_charge is not None for item in occurrences):
            charged.setdefault(residue_id, identity.atom_ref)
    return frozenset(components.keys() - ineligible), charged


def _source_annotation_iterators(
    annotations: Mapping[SourceAtomIdentity, Sequence[SourceAtomAnnotation]] | None,
    *,
    require_occupancy: bool,
) -> dict[SourceAtomIdentity, Iterator[SourceAtomAnnotation]] | None:
    """Bind annotations without assuming duplicate source rows survive in order."""

    if annotations is None:
        if require_occupancy:
            raise StructureNormalizationError(
                "zero-occupancy omission requires reading the source coordinate text"
            )
        return None

    if require_occupancy:
        for identity, occurrences in annotations.items():
            if not occurrences or occurrences[0].explicitly_zero_occupancy is None:
                raise StructureNormalizationError(
                    f"missing source occupancy evidence at {identity.atom_ref.display_token()}"
                )
            # Native identity and occupancy cannot separate blank and numeric zero,
            # or preserve the association through interleaved duplicate rows.
            if any(occurrence != occurrences[0] for occurrence in occurrences):
                raise StructureNormalizationError(
                    f"ambiguous source atom annotations at {identity.atom_ref.display_token()}"
                )

    return {
        identity: iter(occurrences) for identity, occurrences in annotations.items()
    }


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

    if (
        policy.omit_zero_occupancy_sidechain_heavy_atoms
        and structure.provenance.ingress.zero_occupancy_omissions is None
    ):
        raise StructureNormalizationError(
            "zero-occupancy omission requires re-reading the original coordinate input"
        )
    if (
        policy.omit_zero_occupancy_polymer_residues
        and structure.provenance.ingress.zero_occupancy_residue_omissions is None
    ):
        raise StructureNormalizationError(
            "zero-occupancy residue omission requires re-reading the original coordinate input"
        )

    if policy.selected_chain_ids is None:
        selected_chain_ids = structure.constitution.chain_ids()
        selected_constitution = structure.constitution
    else:
        selected_chain_ids = tuple(
            chain_id.strip() for chain_id in policy.selected_chain_ids
        )
        non_polymer_chains = {
            water.residue_id.chain_id for water in structure.constitution.waters
        }
        if policy.non_polymer_retention is not None:
            non_polymer_chains.update(
                site.residue_id.chain_id for site in structure.constitution.ligands
            )
        selected_constitution = structure.constitution.with_chains(
            structure.constitution.chain(chain_id)
            for chain_id in selected_chain_ids
            if structure.constitution.has_chain(chain_id)
            or chain_id not in non_polymer_chains
        )

    selected_ligand_candidates = tuple(
        ligand
        for ligand in structure.constitution.ligands
        if policy.selects_chain(ligand.residue_id.chain_id)
    )
    if policy.rejects_ligands() and selected_ligand_candidates:
        rejected_ligand = selected_ligand_candidates[0]
        raise StructureNormalizationError(
            "structure normalization rejected unexpected ligand "
            f"{rejected_ligand.component_id} at "
            f"{rejected_ligand.residue_id.display_token()}"
        )

    selected_waters: tuple[ResidueSite, ...] = ()
    if policy.non_polymer_retention is not None:
        selected_ligands = tuple(
            site
            for site in selected_ligand_candidates
            if policy.non_polymer_retention.keeps(ResidueRole.LIGAND, site)
        )
        selected_waters = tuple(
            site
            for site in structure.constitution.waters
            if policy.selects_chain(site.residue_id.chain_id)
            and policy.non_polymer_retention.keeps(ResidueRole.WATER, site)
        )
    elif policy.drops_ligands() or policy.rejects_ligands():
        selected_ligands = ()
    else:
        selected_ligands = selected_ligand_candidates

    component_library = build_default_component_library()
    for chain in selected_constitution.chains:
        for residue in chain.residues:
            if _terminal_oxygen_name_projection(
                residue,
                template=component_library.get(residue.component_id),
                formal_charge_by_atom_name=dict(
                    structure.residue_formal_charge_by_atom_name(
                        structure.constitution.residue_index(residue.residue_id)
                    )
                ),
            ):
                raise StructureNormalizationError(
                    "terminal oxygen aliases in a canonical structure require "
                    "re-reading the original coordinate input"
                )

    if (
        selected_constitution == structure.constitution
        and selected_ligands == structure.constitution.ligands
        and selected_waters == structure.constitution.waters
    ):
        return structure

    selected_structure_constitution = replace(
        selected_constitution, ligands=selected_ligands, waters=selected_waters
    )
    selected_residue_ids = {
        residue_site.residue_id
        for residue_site in selected_structure_constitution.residue_slots
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


def _raw_residue_seq_num(raw_residue: gemmi.Residue) -> int:
    """Return a source residue sequence number or reject malformed raw input."""

    seq_num = raw_residue.seqid.num
    if seq_num is None:
        raise StructureNormalizationError(
            "structure normalization rejected residue without a sequence number: "
            f"{raw_residue.name.strip()!r}"
        )

    return int(seq_num)


def _raw_residue_id(raw_residue: gemmi.Residue, chain_id: str) -> ResidueId:
    """Return the canonical residue id for one raw residue."""

    return ResidueId(
        chain_id=chain_id,
        seq_num=_raw_residue_seq_num(raw_residue),
        insertion_code=normalize_insertion_code(raw_residue.seqid.icode),
    )


def _entity_type_by_id(
    raw_structure: gemmi.Structure,
) -> dict[str, gemmi.EntityType]:
    """Return source-declared entity types keyed by explicit entity id."""

    entity_type_by_id: dict[str, gemmi.EntityType] = {}
    for entity in raw_structure.entities:
        entity_id = entity.name.strip()
        if not entity_id:
            continue

        existing_entity_type = entity_type_by_id.get(entity_id)
        if (
            existing_entity_type is not None
            and existing_entity_type is not entity.entity_type
        ):
            raise StructureNormalizationError(
                "structure normalization rejected contradictory entity types for "
                f"entity {entity_id!r}"
            )

        entity_type_by_id[entity_id] = entity.entity_type

    return entity_type_by_id


def _canonical_role_for_entity_type(
    entity_type: gemmi.EntityType,
) -> ResidueRole | None:
    """Project one explicit source entity type into a canonical residue role."""

    if entity_type is gemmi.EntityType.Polymer:
        return ResidueRole.POLYMER
    if entity_type in {
        gemmi.EntityType.NonPolymer,
        gemmi.EntityType.Branched,
    }:
        return ResidueRole.LIGAND
    if entity_type is gemmi.EntityType.Water:
        return ResidueRole.WATER
    return None


def _explicit_entity_role(
    raw_residue: gemmi.Residue,
    *,
    entity_type_by_id: Mapping[str, gemmi.EntityType],
) -> ResidueRole | None:
    """Return a role only when the residue references a declared source entity."""

    entity_id = raw_residue.entity_id.strip()
    if not entity_id:
        return None

    entity_type = entity_type_by_id.get(entity_id)
    if entity_type is None:
        return None

    return _canonical_role_for_entity_type(entity_type)


def _is_supported_polymer_component(component_id: str) -> bool:
    """Return whether curated chemistry identifies a peptide-linking component."""

    return bool(
        build_standard_component_library().has(component_id)
        or build_bundled_nonstandard_registry().get(component_id) is not None
    )


def _source_peptide_link_pairs(
    raw_structure: gemmi.Structure,
    source_connections: tuple[SourceConnection, ...],
) -> frozenset[tuple[ResidueId, ResidueId]]:
    """Return source-explicit inter-residue C-N peptide linkage pairs."""

    if len(raw_structure) == 0 or not source_connections:
        return frozenset()

    components_by_id: dict[ResidueId, set[str]] = {}
    for raw_chain in raw_structure[0]:
        chain_id = normalize_chain_id(raw_chain.name)
        for raw in raw_chain:
            components_by_id.setdefault(_raw_residue_id(raw, chain_id), set()).add(
                raw.name.strip().upper()
            )
    pairs: set[tuple[ResidueId, ResidueId]] = set()
    for connection in source_connections:
        if not connection.is_peptide_link_candidate():
            continue

        endpoint_1 = connection.endpoint_1
        endpoint_2 = connection.endpoint_2
        residue_id_1 = endpoint_1.atom_ref.residue_id
        residue_id_2 = endpoint_2.atom_ref.residue_id
        if any(
            endpoint.component_id not in components
            or (
                len(components) > 1
                and PEPTIDE_CAP_COMPONENT_IDS.intersection(components)
            )
            for endpoint in (endpoint_1, endpoint_2)
            for components in (
                components_by_id.get(endpoint.atom_ref.residue_id, set()),
            )
        ):
            continue

        carbonyl, amide = (
            (endpoint_1, endpoint_2)
            if endpoint_1.atom_ref.atom_name == "C"
            else (endpoint_2, endpoint_1)
        )
        if not peptide_ports_allow_link(carbonyl.component_id, amide.component_id):
            continue

        left_residue_id, right_residue_id = sorted((residue_id_1, residue_id_2))
        pairs.add((left_residue_id, right_residue_id))

    return frozenset(pairs)


def _candidate_has_polymer_context(
    candidate_id: ResidueId,
    polymer_ids: set[ResidueId],
    *,
    sequence_neighbor_pairs: frozenset[tuple[ResidueId, ResidueId]],
    source_peptide_link_pairs: frozenset[tuple[ResidueId, ResidueId]],
) -> bool:
    """Return whether one peptide-compatible HET slot joins known polymer context."""

    return any(
        tuple(sorted((candidate_id, polymer_id))) in sequence_neighbor_pairs
        or tuple(sorted((candidate_id, polymer_id))) in source_peptide_link_pairs
        for polymer_id in polymer_ids
    )


def _canonical_residue_roles_for_chain(
    raw_chain: gemmi.Chain,
    chain_id: str,
    *,
    entity_type_by_id: Mapping[str, gemmi.EntityType],
    source_peptide_link_pairs: frozenset[tuple[ResidueId, ResidueId]],
) -> dict[ResidueId, ResidueRole]:
    """Resolve one canonical role per residue slot before payload normalization."""

    raw_residues_by_id: dict[ResidueId, list[gemmi.Residue]] = {}
    for raw_residue in raw_chain:
        residue_id = _raw_residue_id(raw_residue, chain_id)
        raw_residues_by_id.setdefault(residue_id, []).append(raw_residue)

    ordered_residue_ids = tuple(sorted(raw_residues_by_id))
    components_by_id = {
        residue_id: {raw.name.strip().upper() for raw in variants}
        for residue_id, variants in raw_residues_by_id.items()
    }
    mixed_cap_slots = {
        residue_id
        for residue_id, components in components_by_id.items()
        if len(components) > 1 and PEPTIDE_CAP_COMPONENT_IDS.intersection(components)
    }
    sequence_neighbor_pairs = frozenset(
        (left_residue_id, right_residue_id)
        for left_residue_id, right_residue_id in itertools.pairwise(ordered_residue_ids)
        if _residue_ids_are_sequential_peptide_neighbors(
            left_residue_id,
            right_residue_id,
        )
        and left_residue_id not in mixed_cap_slots
        and right_residue_id not in mixed_cap_slots
        and all(
            peptide_ports_allow_link(left_component, right_component)
            for left_component in components_by_id[left_residue_id]
            for right_component in components_by_id[right_residue_id]
        )
    )

    role_by_id: dict[ResidueId, ResidueRole] = {}
    polymer_candidate_ids: set[ResidueId] = set()
    for residue_id, raw_residues in raw_residues_by_id.items():
        water_flags = tuple(_is_water_residue(residue) for residue in raw_residues)
        if any(water_flags):
            if not all(water_flags):
                raise StructureNormalizationError(
                    "structure normalization rejected mixed water and non-water "
                    f"variants at {residue_id.display_token()}"
                )
            role_by_id[residue_id] = ResidueRole.WATER
            continue

        explicit_roles = {
            role
            for raw_residue in raw_residues
            if (
                role := _explicit_entity_role(
                    raw_residue,
                    entity_type_by_id=entity_type_by_id,
                )
            )
            is not None
        }
        if len(explicit_roles) > 1:
            raise StructureNormalizationError(
                "structure normalization rejected contradictory entity roles at "
                f"{residue_id.display_token()}"
            )
        if explicit_roles:
            role_by_id[residue_id] = explicit_roles.pop()
            continue

        has_polymer_record = any(
            raw_residue.het_flag != "H" for raw_residue in raw_residues
        )
        if has_polymer_record:
            unsupported_het_components = {
                raw_residue.name.strip().upper()
                for raw_residue in raw_residues
                if raw_residue.het_flag == "H"
                and not _is_supported_polymer_component(raw_residue.name)
            }
            if unsupported_het_components:
                raise StructureNormalizationError(
                    "structure normalization rejected polymer/non-polymer variant "
                    f"ambiguity at {residue_id.display_token()}: "
                    f"{', '.join(sorted(unsupported_het_components))}"
                )
            role_by_id[residue_id] = ResidueRole.POLYMER
            continue

        supported_component_flags = tuple(
            _is_supported_polymer_component(raw_residue.name)
            for raw_residue in raw_residues
        )
        if all(supported_component_flags):
            polymer_candidate_ids.add(residue_id)
            continue
        if any(supported_component_flags):
            raise StructureNormalizationError(
                "structure normalization rejected peptide/non-polymer variant "
                f"ambiguity at {residue_id.display_token()}"
            )

        role_by_id[residue_id] = ResidueRole.LIGAND

    polymer_ids = {
        residue_id
        for residue_id, role in role_by_id.items()
        if role is ResidueRole.POLYMER
    }
    for residue_id_1, residue_id_2 in source_peptide_link_pairs:
        if (
            residue_id_1 in polymer_candidate_ids
            and residue_id_2 in polymer_candidate_ids
        ):
            polymer_ids.update((residue_id_1, residue_id_2))

    unresolved_candidate_ids = polymer_candidate_ids - polymer_ids
    while unresolved_candidate_ids:
        promoted_ids = {
            candidate_id
            for candidate_id in unresolved_candidate_ids
            if _candidate_has_polymer_context(
                candidate_id,
                polymer_ids,
                sequence_neighbor_pairs=sequence_neighbor_pairs,
                source_peptide_link_pairs=source_peptide_link_pairs,
            )
        }
        if not promoted_ids:
            break

        polymer_ids.update(promoted_ids)
        unresolved_candidate_ids.difference_update(promoted_ids)

    role_by_id.update((residue_id, ResidueRole.POLYMER) for residue_id in polymer_ids)
    role_by_id.update(
        (residue_id, ResidueRole.LIGAND) for residue_id in unresolved_candidate_ids
    )
    return role_by_id


def _normalize_chain_residues(
    raw_chain: gemmi.Chain,
    chain_id: str,
    policy: StructureNormalizationPolicy,
    *,
    residue_role_by_id: Mapping[ResidueId, ResidueRole],
    source_element_by_atom_identity: Mapping[SourceAtomIdentity, str],
    source_annotations: Mapping[SourceAtomIdentity, Iterator[SourceAtomAnnotation]]
    | None,
    coordination_endpoints_by_residue: Mapping[ResidueId, list[SourceAtomIdentity]],
) -> tuple[
    list[_NormalizedResiduePayload],
    list[_NormalizedResiduePayload],
    set[SourceAtomIdentity],
    bool,
]:
    """Select residues and report omitted endpoints and polymer reordering."""

    grouped_residues: dict[ResidueId, list[_NormalizedResiduePayload]] = {}
    residue_order: list[ResidueId] = []
    grouped_ligands: dict[ResidueId, list[_NormalizedResiduePayload]] = {}
    ligand_order: list[ResidueId] = []
    contains_het_encoded_polymer = False
    excluded_endpoints: set[SourceAtomIdentity] = set()

    for raw_residue in raw_chain:
        residue_id = _raw_residue_id(raw_residue, chain_id)
        role = residue_role_by_id[residue_id]
        if role is ResidueRole.WATER and policy.non_polymer_retention is None:
            excluded_endpoints.update(
                _referenced_source_endpoints(
                    raw_residue, chain_id, coordination_endpoints_by_residue
                )
            )
            continue

        is_hetero = role is not ResidueRole.POLYMER
        if is_hetero:
            if policy.non_polymer_retention is None and policy.drops_ligands():
                excluded_endpoints.update(
                    _referenced_source_endpoints(
                        raw_residue, chain_id, coordination_endpoints_by_residue
                    )
                )
                continue
            if policy.non_polymer_retention is None and policy.rejects_ligands():
                raise StructureNormalizationError(
                    "structure normalization rejected unexpected ligand "
                    f"{raw_residue.name} at {residue_id.display_token()}"
                )
            grouped_payloads = grouped_ligands
            payload_order = ligand_order
        else:
            contains_het_encoded_polymer = bool(
                contains_het_encoded_polymer or raw_residue.het_flag == "H"
            )
            grouped_payloads = grouped_residues
            payload_order = residue_order

        residue = _normalize_residue(
            raw_residue,
            chain_id,
            policy.occupancy_policy,
            is_hetero=is_hetero,
            source_element_by_atom_identity=source_element_by_atom_identity,
            source_annotations=source_annotations,
        )
        if residue_id not in grouped_payloads:
            grouped_payloads[residue_id] = []
            payload_order.append(residue_id)

        grouped_payloads[residue_id].append(residue)

    selected_residue_order = (
        sorted(residue_order) if contains_het_encoded_polymer else residue_order
    )
    non_polymer = []
    for residue_id in ligand_order:
        payload = _select_residue_variant(
            grouped_ligands[residue_id], policy.mutation_policy
        )
        if policy.non_polymer_retention is None or policy.non_polymer_retention.keeps(
            residue_role_by_id[residue_id], payload.constitution
        ):
            non_polymer.append(payload)
        else:
            for endpoint in coordination_endpoints_by_residue.get(residue_id, ()):
                atom = payload.geometry.atoms_by_name.get(endpoint.atom_ref.atom_name)
                if (
                    endpoint.component_id == payload.component_id
                    and atom is not None
                    and atom.altloc == endpoint.altloc
                ):
                    excluded_endpoints.add(endpoint)
    return (
        [
            _select_residue_variant(
                grouped_residues[residue_id],
                policy.mutation_policy,
            )
            for residue_id in selected_residue_order
        ],
        non_polymer,
        excluded_endpoints,
        selected_residue_order == residue_order,
    )


def _referenced_source_endpoints(
    raw_residue: gemmi.Residue,
    chain_id: str,
    endpoints_by_residue: Mapping[ResidueId, list[SourceAtomIdentity]],
) -> tuple[SourceAtomIdentity, ...]:
    """Read only referenced identities, not ignored atom numeric payloads."""
    if not endpoints_by_residue or raw_residue.seqid.num is None:
        return ()
    residue_id = _raw_residue_id(raw_residue, chain_id)
    candidates = endpoints_by_residue.get(residue_id, ())
    component_id = raw_residue.name.strip().upper()
    return tuple(
        endpoint
        for endpoint in candidates
        if endpoint.component_id == component_id
        and any(
            atom.name.strip().upper() == endpoint.atom_ref.atom_name
            and normalize_altloc(atom.altloc) == endpoint.altloc
            for atom in raw_residue
        )
    )


def _excluded_coordination_connections(
    connections: tuple[SourceConnection, ...],
    excluded_endpoints: set[SourceAtomIdentity],
    *,
    constitution: StructureConstitution,
    geometry: StructureGeometry,
    include_dropped_metals: bool,
) -> tuple[SourceConnection, ...]:
    """Capture typed declarations crossing the actual source selection boundary."""
    excluded_connections: list[SourceConnection] = []
    for connection in connections:
        if (
            connection.source_metadata.reported_relationship_type
            is not BondRelationshipType.METAL_COORDINATION
        ):
            continue
        for retained, excluded in (
            connection.endpoint_pair(),
            tuple(reversed(connection.endpoint_pair())),
        ):
            if excluded not in excluded_endpoints:
                continue
            if excluded.resolve_index(constitution, geometry) is not None:
                continue
            index = retained.resolve_index(constitution, geometry)
            if index is not None and (
                include_dropped_metals
                or gemmi.Element(constitution.atom_site_at(index).element).is_metal
            ):
                excluded_connections.append(connection)
                break
    return tuple(dict.fromkeys(excluded_connections))


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
) -> Iterable[_RawAtomPayload]:
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
    raw_atom_payloads: Iterable[_RawAtomPayload],
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
    raw_residue: gemmi.Residue,
    chain_id: str,
    occupancy_policy: OccupancyPolicy,
    *,
    is_hetero: bool,
    source_element_by_atom_identity: Mapping[SourceAtomIdentity, str],
    source_annotations: Mapping[SourceAtomIdentity, Iterator[SourceAtomAnnotation]]
    | None,
) -> _NormalizedResiduePayload:
    """Normalize one raw gemmi residue into the canonical residue entity."""

    residue_id = _raw_residue_id(raw_residue, chain_id)
    selected_atom_payloads = _select_atom_variants(
        raw_residue,
        residue_id=residue_id,
        occupancy_policy=occupancy_policy,
        source_element_by_atom_identity=source_element_by_atom_identity,
        source_annotations=source_annotations,
    )
    return _NormalizedResiduePayload(
        constitution=ResidueSite(
            component_id=raw_residue.name,
            residue_id=residue_id,
            atom_sites=tuple(payload.atom_site for payload in selected_atom_payloads),
            is_hetero=is_hetero,
        ),
        geometry=ResidueGeometry(
            atoms_by_name={
                payload.atom_site.name: payload.geometry
                for payload in selected_atom_payloads
            },
        ),
        formal_charge_by_atom_name=tuple(
            (payload.atom_site.name, payload.formal_charge)
            for payload in selected_atom_payloads
            if payload.formal_charge is not None
        ),
        explicitly_zero_atom_names=frozenset(
            payload.atom_site.name
            for payload in selected_atom_payloads
            if payload.explicitly_zero_occupancy
        ),
        explicitly_positive_atom_names=frozenset(
            payload.atom_site.name
            for payload in selected_atom_payloads
            if payload.explicitly_positive_occupancy
        ),
    )


def _select_atom_variants(
    raw_residue: gemmi.Residue,
    *,
    residue_id: ResidueId,
    occupancy_policy: OccupancyPolicy,
    source_element_by_atom_identity: Mapping[SourceAtomIdentity, str],
    source_annotations: Mapping[SourceAtomIdentity, Iterator[SourceAtomAnnotation]]
    | None,
) -> list[_RawAtomPayload]:
    """Resolve atom sites by residue altloc cohort, then by atom name."""

    raw_atom_payloads = tuple(
        _atom_payload_from_raw_site(
            raw_atom,
            residue_id=residue_id,
            component_id=raw_residue.name,
            source_element_by_atom_identity=source_element_by_atom_identity,
            source_annotations=source_annotations,
        )
        for raw_atom in raw_residue
    )
    selected_altloc = _select_residue_altloc(raw_atom_payloads, occupancy_policy)
    return _select_atom_name_variants(
        _raw_atoms_for_selected_altloc(raw_atom_payloads, selected_altloc),
        selected_altloc=selected_altloc,
        occupancy_policy=occupancy_policy,
    )


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
    raw_atom: gemmi.Atom,
    *,
    residue_id: ResidueId,
    component_id: str,
    source_element_by_atom_identity: Mapping[SourceAtomIdentity, str],
    source_annotations: Mapping[SourceAtomIdentity, Iterator[SourceAtomAnnotation]]
    | None,
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
    source_identity = SourceAtomIdentity(
        atom_ref=AtomRef(residue_id=residue_id, atom_name=raw_atom.name),
        component_id=component_id,
        altloc=altloc,
    )
    explicitly_zero_occupancy = False
    explicitly_positive_occupancy = False
    if source_annotations is None:
        formal_charge = normalize_formal_charge(int(raw_atom.charge))
    else:
        try:
            annotation = next(source_annotations[source_identity])
        except (KeyError, StopIteration) as error:
            raise StructureNormalizationError(
                "source charge annotations do not match "
                f"{source_identity.atom_ref.display_token()}"
            ) from error
        formal_charge = annotation.formal_charge
        if (0 if formal_charge is None else formal_charge) != int(raw_atom.charge):
            raise StructureNormalizationError(
                "source charge annotation disagrees with parsed atom "
                f"{source_identity.atom_ref.display_token()}"
            )
        explicitly_zero_occupancy = annotation.explicitly_zero_occupancy is True
        explicitly_positive_occupancy = (
            annotation.explicitly_positive_occupancy and occupancy > 0
        )
        if explicitly_zero_occupancy and occupancy != 0.0:
            raise StructureNormalizationError(
                "source zero occupancy disagrees with parsed atom "
                f"{source_identity.atom_ref.display_token()}"
            )

    return _RawAtomPayload(
        AtomSite(
            name=raw_atom.name,
            element=source_element_by_atom_identity.get(
                source_identity,
                raw_atom.element.name,
            ),
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
        formal_charge=formal_charge,
        explicitly_zero_occupancy=explicitly_zero_occupancy,
        explicitly_positive_occupancy=explicitly_positive_occupancy,
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


def _validated_raw_atom_occupancy(
    raw_atom: gemmi.Atom,
    residue_id: ResidueId,
) -> float:
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


def _validated_raw_atom_b_factor(
    raw_atom: gemmi.Atom,
    residue_id: ResidueId,
) -> float:
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


def _is_water_residue(raw_residue: gemmi.Residue) -> bool:
    """Return whether one raw residue represents water."""

    return bool(
        raw_residue.is_water() or raw_residue.entity_type is gemmi.EntityType.Water
    )


def _topology_bonds_from_source_connections(
    connections: tuple[SourceConnection, ...],
    *,
    constitution: StructureConstitution,
    geometry: StructureGeometry,
    expected_bonds_by_pair: dict[tuple[AtomIndex, AtomIndex], TopologyBond],
) -> tuple[TopologyBond, ...]:
    """Lower surviving source connections into canonical topology bonds."""

    connections_by_pair: dict[tuple[AtomIndex, AtomIndex], SourceConnection] = {}
    for connection in connections:
        endpoint_1 = connection.endpoint_1
        endpoint_2 = connection.endpoint_2
        require_altloc_match = connection.requires_exact_altloc_match()
        if (
            endpoint_1.resolve_index(
                constitution, geometry, require_altloc_match=require_altloc_match
            )
            is None
            or endpoint_2.resolve_index(
                constitution, geometry, require_altloc_match=require_altloc_match
            )
            is None
        ):
            continue

        index_1 = constitution.atom_index(endpoint_1.atom_ref)
        index_2 = constitution.atom_index(endpoint_2.atom_ref)
        pair = (
            (index_1, index_2) if index_1.value < index_2.value else (index_2, index_1)
        )
        existing = connections_by_pair.get(pair)
        connections_by_pair[pair] = (
            connection if existing is None else existing.merge(connection)
        )

    return tuple(
        connection.to_topology_bond(
            *pair, expected_bond=expected_bonds_by_pair.get(pair)
        )
        for pair, connection in connections_by_pair.items()
    )


def _residue_ids_are_sequential_peptide_neighbors(
    left_residue_id: ResidueId,
    right_residue_id: ResidueId,
) -> bool:
    """Return whether residue identifiers imply adjacent peptide neighbors."""

    return left_residue_id.immediately_precedes(right_residue_id)

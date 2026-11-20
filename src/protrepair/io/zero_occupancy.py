"""Project admitted zero-occupancy coordinates out of selected source facets."""

from collections.abc import Mapping
from dataclasses import replace

from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.context import AminoAcidMicrostateContext
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.errors import StructureNormalizationError
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.geometry import StructureGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.provenance import (
    ZeroOccupancyOmission,
    ZeroOccupancyResidueOmission,
)
from protrepair.structure.source_connection import SourceAtomIdentity, SourceConnection
from protrepair.structure.topology import (
    BondRelationshipType,
    StructureTopology,
    is_covalent_like_relationship,
)

_REQUIRED_BACKBONE = frozenset({"N", "CA", "C", "O"})
_BACKBONE_AND_TERMINAL = _REQUIRED_BACKBONE | {"OXT"}
_PROTECTED_EXTERNAL_RELATIONSHIPS = frozenset(
    {
        BondRelationshipType.COVALENT,
        BondRelationshipType.DISULFIDE,
        BondRelationshipType.METAL_COORDINATION,
        BondRelationshipType.UNKNOWN,
    }
)


def zero_occupancy_sidechain_omissions(
    selected: ProteinStructure,
    *,
    explicitly_zero_refs: frozenset[AtomRef],
    source_connections: tuple[SourceConnection, ...],
) -> tuple[ZeroOccupancyOmission, ...]:
    """Check which selected side-chain coordinates can be withheld.

    Parameters
    ----------
    selected : ProteinStructure
        Provisional first-model structure after conformer selection.
    explicitly_zero_refs : frozenset of AtomRef
        Selected sites with exact zero-token evidence from the format adapter.
    source_connections : tuple of SourceConnection
        Decoded declarations in normalized names, before endpoint filtering.

    Returns
    -------
    tuple of ZeroOccupancyOmission
        Eligible coordinates with no changes to the selected structure.

    Raises
    ------
    StructureNormalizationError
        A proposed omission would compromise source chemistry or an observed
        fragment.
    """
    zero_names_by_residue: dict[ResidueId, set[str]] = {}
    for atom_ref in explicitly_zero_refs:
        zero_names_by_residue.setdefault(atom_ref.residue_id, set()).add(
            atom_ref.atom_name
        )

    templates = build_standard_component_library()
    omissions: list[ZeroOccupancyOmission] = []
    protected_sidechain_refs: set[AtomRef] = set()
    attachment_context = None
    for residue in selected.constitution.iter_residues():
        residue_id = residue.residue_id
        zero_names = zero_names_by_residue.get(residue_id)
        if not zero_names or zero_names & _REQUIRED_BACKBONE:
            continue
        template = templates.get(residue.component_id)
        if template is None or template.component_id != residue.component_id:
            continue

        expected = frozenset(template.expected_polymer_heavy_atom_names())
        sidechain = expected - _BACKBONE_AND_TERMINAL
        if not (zero_names & sidechain):
            continue
        if {
            neighbor
            for name in sidechain
            for neighbor in template.definition.bonded_atom_names(name)
            if neighbor in _BACKBONE_AND_TERMINAL
        } != {"CA"}:
            continue

        # The stock amino-acid names encode the one-letter element first.
        matching = {
            atom.name for atom in residue.atom_sites if atom.element == atom.name[0]
        }
        if not _REQUIRED_BACKBONE <= matching:
            continue
        withheld = matching & zero_names & sidechain
        if not withheld:
            continue

        present_heavy = {
            atom.name for atom in residue.atom_sites if not atom.is_hydrogen()
        }
        present_scaffold = expected & present_heavy
        retained = present_scaffold - withheld
        orphaned = retained & (
            _reachable_from_ca(template, present_scaffold)
            - _reachable_from_ca(template, retained)
        )
        if orphaned:
            raise StructureNormalizationError(
                f"zero-occupancy omission at {residue_id.display_token()} would "
                f"disconnect retained atoms: {', '.join(sorted(orphaned))}"
            )

        sidechain_names = present_heavy - _BACKBONE_AND_TERMINAL
        if any(atom.is_hydrogen() for atom in residue.atom_sites):
            if attachment_context is None:
                attachment_context = AminoAcidMicrostateContext(selected)
            parents = _hydrogen_parents(attachment_context, residue, template)
            for hydrogen, parent in parents.items():
                if parent not in present_heavy:
                    raise StructureNormalizationError(
                        f"cannot omit zero-occupancy atoms at {residue_id.display_token()}: "
                        f"hydrogen {hydrogen} has no selected heavy parent"
                    )
                if parent in sidechain_names:
                    sidechain_names.add(hydrogen)
                if parent not in withheld:
                    continue
                if hydrogen not in zero_names:
                    raise StructureNormalizationError(
                        "zero-occupancy omission would remove the parent of retained "
                        f"H/D/T {AtomRef(residue_id, hydrogen).display_token()}"
                    )
                withheld.add(hydrogen)

        residue_index = selected.constitution.residue_index(residue_id)
        for name, charge in selected.residue_formal_charge_by_atom_name(residue_index):
            if charge is not None and (
                name in withheld or (charge != 0 and name in sidechain_names)
            ):
                raise StructureNormalizationError(
                    "zero-occupancy omission cannot preserve the formal charge at "
                    f"{AtomRef(residue_id, name).display_token()}"
                )

        protected_sidechain_refs.update(
            AtomRef(residue_id, name) for name in sidechain_names
        )
        omissions.extend(_omission_atoms(selected, residue, withheld))

    withheld_refs = frozenset(
        omission.source_identity.atom_ref for omission in omissions
    )
    if withheld_refs:
        _check_source_connections(
            selected, source_connections, withheld_refs, protected_sidechain_refs
        )

    return tuple(omissions)


def zero_occupancy_residue_omissions(
    selected: ProteinStructure,
    *,
    all_zero_residue_ids: frozenset[ResidueId],
    positive_backbone_residue_ids: frozenset[ResidueId],
    verified_neighbors: frozenset[tuple[ResidueId, ResidueId]],
    charged_source_atoms_by_residue: Mapping[ResidueId, AtomRef],
    source_connections: tuple[SourceConnection, ...],
) -> tuple[ZeroOccupancyResidueOmission, ...]:
    """Check which original all-zero slots can be treated as internal gaps.

    Parameters
    ----------
    selected : ProteinStructure
        Provisional structure before either occupancy rule.
    all_zero_residue_ids : frozenset of ResidueId
        Single-component slots proven zero in every raw occurrence.
    positive_backbone_residue_ids : frozenset of ResidueId
        Selected residues with positively declared N, CA, C and O.
    verified_neighbors : frozenset of tuple of ResidueId
        Adjacent source slots without a conflicting sequence or declared break.
    charged_source_atoms_by_residue : Mapping of ResidueId to AtomRef
        One source charge witness per slot, including explicit neutral values.
    source_connections : tuple of SourceConnection
        Decoded declarations before endpoint filtering.

    Returns
    -------
    tuple of ZeroOccupancyResidueOmission
        Eligible internal singleton gaps in source order.

    Raises
    ------
    StructureNormalizationError
        An eligible gap would discard a source connection or formal charge.
    """
    if not all_zero_residue_ids:
        return ()

    incident_connections: dict[ResidueId, SourceConnection] = {}
    for connection in source_connections:
        for endpoint in connection.endpoint_pair():
            residue_id = endpoint.atom_ref.residue_id
            if residue_id in all_zero_residue_ids:
                incident_connections.setdefault(residue_id, connection)

    templates = build_standard_component_library()
    omissions: list[ZeroOccupancyResidueOmission] = []
    for chain in selected.constitution.chains:
        residues = chain.residues
        for preceding, residue, following in zip(
            residues, residues[1:], residues[2:], strict=False
        ):
            before, target, after = (
                site.residue_id for site in (preceding, residue, following)
            )
            if (
                target not in all_zero_residue_ids
                or before not in positive_backbone_residue_ids
                or after not in positive_backbone_residue_ids
                or (before, target) not in verified_neighbors
                or (target, after) not in verified_neighbors
                or any(
                    rid.insertion_code is not None for rid in (before, target, after)
                )
                or not before.immediately_precedes(target)
                or not target.immediately_precedes(after)
                or preceding.has_atom_site("OXT")
                or residue.has_atom_site("OXT")
            ):
                continue
            template = templates.get(residue.component_id)
            if template is None or template.component_id != residue.component_id:
                continue
            bond = selected.topology.bond_between(
                selected.constitution.atom_index(AtomRef(before, "C")),
                selected.constitution.atom_index(AtomRef(after, "N")),
            )
            if bond is not None and (
                is_covalent_like_relationship(bond)
                or bond.relationship_type is BondRelationshipType.UNKNOWN
            ):
                continue

            charge = charged_source_atoms_by_residue.get(target)
            if charge is not None:
                raise StructureNormalizationError(
                    "zero-occupancy residue omission cannot preserve the formal charge at "
                    f"{charge.display_token()}"
                )
            connection = incident_connections.get(target)
            if connection is not None:
                endpoint = next(
                    endpoint
                    for endpoint in connection.endpoint_pair()
                    if endpoint.atom_ref.residue_id == target
                )
                raise StructureNormalizationError(
                    "zero-occupancy residue omission cannot preserve "
                    f"{connection.source_metadata.record_type.value} connection at "
                    f"{endpoint.atom_ref.display_token()}"
                )

            omissions.append(
                ZeroOccupancyResidueOmission(
                    before,
                    after,
                    _omission_atoms(selected, residue, set(residue.atom_site_names())),
                )
            )
    return tuple(omissions)


def _omission_atoms(
    selected: ProteinStructure, residue: ResidueSite, names: set[str]
) -> tuple[ZeroOccupancyOmission, ...]:
    omissions = []
    for atom in residue.atom_sites:
        if atom.name in names:
            ref = AtomRef(residue.residue_id, atom.name)
            geometry = selected.geometry.atom_geometry(
                selected.constitution.atom_index(ref)
            )
            omissions.append(
                ZeroOccupancyOmission(
                    SourceAtomIdentity(ref, residue.component_id, geometry.altloc),
                    atom.element_identity,
                    geometry,
                )
            )
    return tuple(omissions)


def project_zero_occupancy_omissions(
    selected: ProteinStructure,
    *,
    atom_omissions: tuple[ZeroOccupancyOmission, ...] | None,
    residue_omissions: tuple[ZeroOccupancyResidueOmission, ...] | None,
) -> ProteinStructure:
    """Publish both preflighted omissions and their filtered source observation.

    Parameters
    ----------
    selected : ProteinStructure
        Original selected snapshot used by both preflights.
    atom_omissions, residue_omissions : tuple or None
        Independent history for each applied policy; None means not applied.

    Returns
    -------
    ProteinStructure
        Retained facets with only the admitted omissions withheld.
    """
    ingress = selected.provenance.ingress
    observation = ingress.observation
    constitution = selected.constitution
    geometry = selected.geometry
    topology = selected.topology
    if atom_omissions or residue_omissions:
        omitted_residues = {record.residue_id for record in residue_omissions or ()}
        names_by_residue: dict[ResidueId, set[str]] = {}
        for omission in atom_omissions or ():
            ref = omission.source_identity.atom_ref
            names_by_residue.setdefault(ref.residue_id, set()).add(ref.atom_name)
        constitution = constitution.with_chains(
            chain.with_residues(
                residue.without_atom_sites(names_by_residue[residue.residue_id])
                if residue.residue_id in names_by_residue
                else residue
                for residue in chain.residues
                if residue.residue_id not in omitted_residues
            )
            for chain in constitution.chains
        )
        retained_indices = tuple(
            selected.constitution.atom_index(AtomRef(residue.residue_id, atom.name))
            for residue in constitution.residue_slots
            for atom in residue.atom_sites
        )
        geometry = StructureGeometry(
            constitution=constitution,
            atom_geometries=tuple(
                selected.geometry.atom_geometry(i) for i in retained_indices
            ),
        )
        topology = StructureTopology(
            constitution=constitution,
            atom_topologies=tuple(
                selected.topology.atom_topologies[i.value] for i in retained_indices
            ),
            bonds=selected.topology.bonds_for_constitution(
                source_constitution=selected.constitution,
                target_constitution=constitution,
            ),
        )
        assert observation is not None
        observation = StructureObservation.from_source_facets(
            constitution=constitution,
            geometry=geometry,
            topology=topology,
            excluded_coordination_connections=observation.excluded_coordination_connections,
        )
    return ProteinStructure.from_payload(
        constitution=constitution,
        geometry=geometry,
        topology=topology,
        polymer_blueprint=selected.polymer_blueprint,
        provenance=replace(
            selected.provenance,
            ingress=replace(
                ingress,
                observation=observation,
                zero_occupancy_omissions=atom_omissions,
                zero_occupancy_residue_omissions=residue_omissions,
                pdb_adjacent_cap_links=(
                    ()
                    if observation is None
                    else tuple(
                        link
                        for link in ingress.pdb_adjacent_cap_links
                        if link.bond_in(
                            observation.constitution,
                            observation.geometry,
                            observation.topology,
                        )
                        is not None
                    )
                ),
            ),
        ),
    )


def _reachable_from_ca(
    template: ResidueTemplate, atom_names: frozenset[str]
) -> set[str]:
    """Follow template bonds through the present heavy scaffold."""

    reachable = {"CA"}
    pending = ["CA"]
    while pending:
        neighbors = (
            template.definition.bonded_atom_names(pending.pop()) & atom_names
        ) - reachable
        reachable.update(neighbors)
        pending.extend(neighbors)
    return reachable


def _hydrogen_parents(
    context: AminoAcidMicrostateContext, residue: ResidueSite, template: ResidueTemplate
) -> dict[str, str]:
    try:
        return context.hydrogen_parents(
            residue.residue_id,
            PolymerMicrostateSite(template, PolymerChemicalSite.SIDECHAIN),
        )
    except ValueError as error:
        raise StructureNormalizationError(
            f"cannot omit zero-occupancy atoms at {residue.residue_id.display_token()}: {error}"
        ) from error


def _check_source_connections(
    selected: ProteinStructure,
    source_connections: tuple[SourceConnection, ...],
    withheld: frozenset[AtomRef],
    protected_sidechain: set[AtomRef],
) -> None:
    """Protect selected endpoints even if the other endpoint was excluded."""

    for connection in source_connections:
        first, second = connection.endpoint_pair()
        metadata = connection.source_metadata
        external_link = (
            first.atom_ref.residue_id != second.atom_ref.residue_id
            and metadata.reported_relationship_type in _PROTECTED_EXTERNAL_RELATIONSHIPS
        )
        for endpoint in (first, second):
            ref = endpoint.atom_ref
            if ref not in withheld and not (
                external_link and ref in protected_sidechain
            ):
                continue
            if (
                endpoint.resolve_index(
                    selected.constitution,
                    selected.geometry,
                    require_altloc_match=connection.requires_exact_altloc_match(),
                )
                is not None
            ):
                raise StructureNormalizationError(
                    f"zero-occupancy omission cannot preserve {metadata.record_type.value} "
                    f"connection at {ref.display_token()}"
                )

"""Project admitted zero-occupancy coordinates out of selected source facets."""

from collections.abc import Mapping
from dataclasses import replace
from itertools import pairwise

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
    ZeroOccupancyRunOmission,
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
    singletons: bool,
    runs: bool,
    run_local_links: tuple[SourceConnection, ...],
    all_zero_residue_ids: frozenset[ResidueId],
    positive_backbone_residue_ids: frozenset[ResidueId],
    verified_neighbors: frozenset[tuple[ResidueId, ResidueId]],
    charged_source_atoms_by_residue: Mapping[ResidueId, AtomRef],
    source_connections: tuple[SourceConnection, ...],
    uncertain_atoms: frozenset[AtomRef],
    raw_terminal_ids: frozenset[ResidueId],
) -> tuple[
    tuple[ZeroOccupancyResidueOmission, ...], tuple[ZeroOccupancyRunOmission, ...]
]:
    """Preflight each entire interval between successive usable flanks."""
    if not all_zero_residue_ids:
        return (), ()

    incident: dict[ResidueId, SourceConnection] = {}
    partners: dict[AtomRef, set[AtomRef]] = {}
    for connection in source_connections:
        first, second = connection.endpoint_pair()
        for end, other in ((first, second), (second, first)):
            if end.atom_ref.residue_id in all_zero_residue_ids:
                incident.setdefault(end.atom_ref.residue_id, connection)
            if (
                runs
                and end.atom_ref.atom_name in {"N", "C"}
                and end.atom_ref.residue_id != other.atom_ref.residue_id
                and connection.source_metadata.reported_relationship_type
                is not BondRelationshipType.HYDROGEN_BOND
            ):
                partners.setdefault(end.atom_ref, set()).add(other.atom_ref)

    uncertain_slots = {ref.residue_id for ref in uncertain_atoms}
    local_declared = {
        link.endpoint_1.atom_ref.residue_id: link for link in run_local_links
    }
    templates = build_standard_component_library()
    isolated: list[ZeroOccupancyResidueOmission] = []
    windows: list[ZeroOccupancyRunOmission] = []
    for chain in selected.constitution.chains:
        residues = chain.residues
        flanks = (
            i
            for i, residue in enumerate(residues)
            if residue.residue_id in positive_backbone_residue_ids
        )
        for left, right in pairwise(flanks):
            size = right - left - 1
            if not size or not (singletons if size == 1 else runs):
                continue
            stretch = residues[left : right + 1]
            preceding, following = stretch[0], stretch[-1]
            members = stretch[1:-1]
            path = tuple(site.residue_id for site in stretch)
            before, after = path[0], path[-1]
            if (
                any(r.insertion_code is not None for r in path)
                or any(r not in all_zero_residue_ids for r in path[1:-1])
                or any(
                    (a, b) not in verified_neighbors or not a.immediately_precedes(b)
                    for a, b in pairwise(path)
                )
                or any(site.has_atom_site("OXT") for site in stretch[:-1])
                or any(
                    (t := templates.get(site.component_id)) is None
                    or t.component_id != site.component_id
                    for site in members
                )
            ):
                continue
            ports = AtomRef(before, "C"), AtomRef(after, "N")
            bond = selected.topology.bond_between(
                *(selected.constitution.atom_index(p) for p in ports)
            )
            if bond is not None and (
                is_covalent_like_relationship(bond)
                or bond.relationship_type is BondRelationshipType.UNKNOWN
            ):
                continue
            if size > 1:
                targets = set(path[1:-1])
                if (
                    targets & uncertain_slots
                    or any(p in uncertain_atoms for p in ports)
                    or any(r in raw_terminal_ids for r in path[:-1])
                    or any(
                        other.residue_id not in targets
                        for p in ports
                        for other in partners.get(p, ())
                    )
                ):
                    continue

            for target in path[1:-1]:
                charge = charged_source_atoms_by_residue.get(target)
                if charge is not None:
                    raise StructureNormalizationError(
                        "zero-occupancy residue omission cannot preserve the formal charge at "
                        f"{charge.display_token()}"
                    )
                connection = incident.get(target)
                if connection is None and size > 1:
                    connection = local_declared.get(target)
                if connection is not None:
                    endpoint = next(
                        end
                        for end in connection.endpoint_pair()
                        if end.atom_ref.residue_id == target
                    )
                    raise StructureNormalizationError(
                        "zero-occupancy residue omission cannot preserve "
                        f"{connection.source_metadata.record_type.value} connection at "
                        f"{endpoint.atom_ref.display_token()}"
                    )
            archive = tuple(
                _omission_atoms(selected, site, set(site.atom_site_names()))
                for site in members
            )
            if size == 1:
                isolated.append(ZeroOccupancyResidueOmission(before, after, archive[0]))
            else:
                windows.append(
                    ZeroOccupancyRunOmission(
                        before,
                        after,
                        preceding.component_id,
                        following.component_id,
                        archive,
                    )
                )
    return tuple(isolated), tuple(windows)


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
    run_omissions: tuple[ZeroOccupancyRunOmission, ...] | None = None,
) -> ProteinStructure:
    """Publish both preflighted omissions and their filtered source observation.

    Parameters
    ----------
    selected : ProteinStructure
        Original selected snapshot used by both preflights.
    atom_omissions, residue_omissions, run_omissions : tuple or None
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
    if atom_omissions or residue_omissions or run_omissions:
        omitted_residues = {record.residue_id for record in residue_omissions or ()}
        omitted_residues.update(
            r for run in run_omissions or () for r in run.residue_ids
        )
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
            source_connections=observation.unresolved_non_polymer_links,
            unresolved_polymer_endpoints=observation.unresolved_polymer_endpoints,
            unverified_polymer_ports=observation.unverified_polymer_ports,
            deposited_peptide_boundaries=observation.deposited_peptide_boundaries,
            unresolved_conect_ports=observation.unresolved_conect_ports,
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
                zero_occupancy_run_omissions=run_omissions,
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

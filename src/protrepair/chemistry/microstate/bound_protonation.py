"""Run-scoped custody of caller-selected polymer occurrences and input evidence."""

from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from types import MappingProxyType
from typing import TYPE_CHECKING

from protrepair.chemistry.microstate.catalog import PeptideLinkage, PolymerChemicalSite
from protrepair.chemistry.microstate.resolution import (
    MicrostateConstraints,
    MicrostateResolution,
    MicrostateResolutionStatus,
    SiteProtonationReason,
)
from protrepair.chemistry.microstate.selection import SiteProtonation
from protrepair.chemistry.standard.components import STANDARD_COMPONENT_ATOMS
from protrepair.relation.sequence_alignment import (
    ObservedChainSequence,
    sequence_requires_observed_end,
)
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.bound_peptide_ports import active_peptide_ports
from protrepair.structure.constitution import ResidueRole
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.topology import BondRelationshipType

if TYPE_CHECKING:
    from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite


@dataclass(frozen=True, slots=True, eq=False)
class BoundSiteProtonations:
    """Bind requests once, before any stage can add residues or manufacture evidence."""

    source: ProteinStructure
    requests: Mapping[tuple[ResidueId, PolymerChemicalSite], SiteProtonation]
    observation: StructureObservation

    @classmethod
    def bind(
        cls, source: ProteinStructure, requests: tuple[SiteProtonation, ...]
    ) -> "BoundSiteProtonations | None":
        if not requests:
            return None
        observation = source.provenance.ingress.observation
        if observation is None:
            observation = StructureObservation.from_source_facets(
                constitution=source.constitution,
                geometry=source.geometry,
                topology=source.topology,
            )
        return cls(
            source,
            MappingProxyType({(r.residue_id, r.site): r for r in requests}),
            observation,
        )

    def identity_failure(
        self, current: ProteinStructure, request: SiteProtonation
    ) -> SiteProtonationReason | None:
        rid = request.residue_id
        constitution = self.source.constitution
        original = constitution.residue_or_ligand(rid)
        if original is None:
            return SiteProtonationReason.ABSENT_AT_INPUT
        if (
            constitution.role_at(constitution.residue_index(rid))
            is not ResidueRole.POLYMER
        ):
            return SiteProtonationReason.NON_POLYMER
        if original.component_id not in STANDARD_COMPONENT_ATOMS:
            return SiteProtonationReason.UNSUPPORTED_COMPONENT
        current_site = current.constitution.residue_or_ligand(rid)
        if (
            current_site is None
            or current_site.component_id != original.component_id
            or current.constitution.role_at(current.constitution.residue_index(rid))
            is not ResidueRole.POLYMER
        ):
            return SiteProtonationReason.SUPERSEDED_IDENTITY
        return None

    def free_end_failure(
        self, current: ProteinStructure, selection: SiteProtonation
    ) -> str | None:
        rid = selection.residue_id
        n_end = selection.site is PolymerChemicalSite.BACKBONE_N
        for constitution in (self.source.constitution, self.observation.constitution):
            if any(
                chain.chain_id == rid.chain_id
                and chain.residues
                and chain.residues[0 if n_end else -1].residue_id != rid
                for chain in constitution.chains
            ):
                return "site is not the corresponding input chain end"
        refs = {AtomRef(rid, name) for name in (("N",) if n_end else ("C", "O", "OXT"))}
        if refs.intersection(self.observation.unverified_polymer_ports):
            return "deposited sequence does not establish this terminus"
        if refs.intersection(self.observation.unresolved_polymer_endpoints):
            return "source terminal relationship to an excluded partner"
        if any(
            b.relationship_type is not BondRelationshipType.HYDROGEN_BOND
            and self.observation.constitution.atom_ref_at(b.atom_index_1).residue_id
            != self.observation.constitution.atom_ref_at(b.atom_index_2).residue_id
            and refs.intersection(
                self.observation.constitution.atom_ref_at(i) for i in b.endpoint_pair()
            )
            for b in self.observation.bonds_for_residue(rid)
        ):
            return "source terminal port is occupied"
        if (
            current.polymer_blueprint is not None
            and rid.chain_id in current.polymer_blueprint.chain_ids()
        ):
            sequence = current.polymer_blueprint.chain(rid.chain_id).sequence()
            tokens = ObservedChainSequence.from_chain(
                current.constitution.chain(rid.chain_id)
            ).tokens()

            if "X" in tokens or not sequence_requires_observed_end(
                tokens, sequence, n_terminal=n_end
            ):
                return "terminal position is not established on the supplied sequence"
        return None

    def limitation(
        self,
        current: ProteinStructure,
        selection: SiteProtonation,
        site: "PolymerMicrostateSite",
    ) -> MicrostateResolution | None:
        if site.kind in (
            PolymerChemicalSite.BACKBONE_N,
            PolymerChemicalSite.BACKBONE_C,
        ):
            ports = active_peptide_ports()
            port = AtomRef(
                selection.residue_id,
                "N" if site.kind is PolymerChemicalSite.BACKBONE_N else "C",
            )
            if (
                ports is not None
                and ports.reserves(current, port)
                and (
                    site.linkage is PeptideLinkage.UNKNOWN
                    or (
                        site.linkage is PeptideLinkage.LINKED
                        and _requires_free_end(site, selection)
                    )
                )
            ):
                return MicrostateResolution(
                    MicrostateResolutionStatus.CONFLICT,
                    details=(
                        "selected peptide port is reserved against free-terminal chemistry",
                    ),
                    reason=SiteProtonationReason.PHYSICAL_CONSTRAINT,
                )
        catalog = site.candidates()
        if catalog is None:
            return None
        names = {atom.name for atom in catalog.candidates[0].atoms}
        rid = selection.residue_id

        def insufficient(
            reason: SiteProtonationReason, detail: str
        ) -> MicrostateResolution:
            return MicrostateResolution(
                MicrostateResolutionStatus.INSUFFICIENT,
                details=(detail,),
                reason=reason,
            )

        if self.source.provenance.ingress.observation is None and any(
            choice.residue_id == rid
            and any(atom.name in names for atom in choice.graph.atoms)
            for choice in self.source.provenance.microstate_choices
        ):
            return insufficient(
                SiteProtonationReason.INSUFFICIENT_PROVENANCE,
                "no pre-edit observation distinguishes supplied from generated chemistry",
            )
        refs = {AtomRef(rid, name) for name in names}
        observation = self.observation
        if refs.intersection(observation.unresolved_polymer_endpoints):
            return insufficient(
                SiteProtonationReason.EXCLUDED_PARTNER,
                "source relationship to an excluded partner",
            )
        if site.linkage is PeptideLinkage.FREE:
            failure = self.free_end_failure(current, selection)
            if failure is not None:
                return insufficient(SiteProtonationReason.UNVERIFIED_BOUNDARY, failure)
        if site.kind is PolymerChemicalSite.SIDECHAIN:
            for bond in current.topology.inter_residue_bonds_for_residue(
                constitution=current.constitution,
                residue_index=current.constitution.residue_index(rid),
            ):
                if bond.relationship_type in (
                    BondRelationshipType.HYDROGEN_BOND,
                    BondRelationshipType.METAL_COORDINATION,
                ):
                    continue
                if refs.intersection(
                    current.constitution.atom_ref_at(i) for i in bond.endpoint_pair()
                ):
                    return MicrostateResolution(
                        MicrostateResolutionStatus.UNSUPPORTED,
                        reason=SiteProtonationReason.PHYSICAL_CONSTRAINT,
                        details=(
                            "side chain has a covalent or untyped external attachment",
                        ),
                    )
        for bond in observation.bonds_for_residue(rid):
            if bond.relationship_type is BondRelationshipType.HYDROGEN_BOND:
                continue
            endpoints = tuple(
                observation.constitution.atom_ref_at(i) for i in bond.endpoint_pair()
            )
            if endpoints[0].residue_id == endpoints[
                1
            ].residue_id or not refs.intersection(endpoints):
                continue
            indices = tuple(
                current.constitution.resolve_atom_index(ref) for ref in endpoints
            )
            a, b = indices
            if a is None or b is None or current.topology.bond_between(a, b) is None:
                return insufficient(
                    SiteProtonationReason.EXCLUDED_PARTNER,
                    "source relationship to an excluded or changed partner",
                )
        return None


def _requires_free_end(
    site: "PolymerMicrostateSite", selection: SiteProtonation
) -> bool:
    def admits(linkage: PeptideLinkage) -> bool:
        catalog = replace(site, linkage=linkage).candidates()
        if catalog is None:
            return False
        request = catalog.resolve_protonation(
            MicrostateConstraints(),
            site_charge=selection.site_charge,
            parent_hydrogens=selection.parent_hydrogens,
        )
        return bool(request.candidates)

    return not admits(PeptideLinkage.LINKED) and admits(PeptideLinkage.FREE)


_ACTIVE: ContextVar[BoundSiteProtonations | None] = ContextVar(
    "polymer_site_protonations", default=None
)


def active_site_protonations() -> BoundSiteProtonations | None:
    return _ACTIVE.get()


def requested_site(
    current: ProteinStructure, rid: ResidueId, kind: PolymerChemicalSite
) -> SiteProtonation | None:
    binding = _ACTIVE.get()
    if binding is not None:
        request = binding.requests.get((rid, kind))
        if request is not None and binding.identity_failure(current, request) is None:
            return request
    return None


@contextmanager
def site_protonation_run(
    binding: BoundSiteProtonations | None,
) -> Iterator[None]:
    """Use this run's binding, restoring any enclosing run on exit."""
    token = _ACTIVE.set(binding)
    try:
        yield
    finally:
        _ACTIVE.reset(token)

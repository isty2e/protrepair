"""Read-only H coverage for polymer references without admitted preparation."""

from collections import Counter
from dataclasses import dataclass

from protrepair.chemistry.component.defaults import build_default_component_library
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.microstate.catalog import PeptideLinkage, PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.graph import MicrostateGraph
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.resolution import (
    MicrostateResolution,
    MicrostateResolutionStatus,
)
from protrepair.state.reference_hydrogen import (
    ReferenceHydrogenFailure,
    ReferenceHydrogenScope,
    assess_reference_hydrogens,
)
from protrepair.state.structure_axes import HydrogenCoverageState
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import AtomRef


@dataclass(frozen=True, slots=True)
class PolymerReferenceHydrogenAssessment:
    """Report coverage without granting H generation or topology authority.

    Parameters
    ----------
    coverage_state : HydrogenCoverageState or None
        COMPLETE only after reference-body and contextual terminal verification.
    failure : ReferenceHydrogenFailure or None
        Current cause of indeterminate coverage.
    """

    coverage_state: HydrogenCoverageState | None
    failure: ReferenceHydrogenFailure | None


def assess_polymer_reference_hydrogens(
    context: PolymerMicrostateContext,
    residue: ResidueSite,
    template: ResidueTemplate,
) -> PolymerReferenceHydrogenAssessment:
    """Verify existing H against a reference and actual peptide boundaries.

    Parameters
    ----------
    context : PolymerMicrostateContext
        Current polymer snapshot and immutable source observations.
    residue : ResidueSite
        Polymer occurrence, not a retained component.
    template : ResidueTemplate
        Active reference-bearing template without admitted H generation.

    Returns
    -------
    PolymerReferenceHydrogenAssessment
        Read-only COMPLETE or an indeterminate requirement with a typed cause.
        No default state is selected and no missing atom is reconstructed.

    Raises
    ------
    ValueError
        The template is not a reference without admitted H preparation.
    """
    reference = template.reference_component
    if reference is None or not template.has_unprepared_hydrogen_reference():
        raise ValueError("polymer reference assessment needs an unprepared reference")
    stock = build_default_component_library().get(template.component_id)
    unavailable = ReferenceHydrogenFailure(
        "no_admitted_preparation",
        "no admitted polymer H preparation; existing H can be verified against "
        "the reference and supported peptide boundaries, but missing body H "
        "cannot be generated",
    )
    if not any(atom.is_hydrogen() for atom in residue.atom_sites):
        return PolymerReferenceHydrogenAssessment(None, unavailable)

    sites: list[tuple[PolymerMicrostateSite, tuple[MicrostateGraph, ...]]] = []
    excluded: set[str] = set()
    omitted: set[str] = set()
    external: set[frozenset[AtomRef]] = set()
    controlled: set[frozenset[str]] = set()
    for kind in (PolymerChemicalSite.BACKBONE_N, PolymerChemicalSite.BACKBONE_C):
        site = context.backbone_site(
            residue.residue_id, template, kind, assume_free_chain_ends=True
        )
        if site.linkage is PeptideLinkage.UNKNOWN:
            return PolymerReferenceHydrogenAssessment(
                None,
                ReferenceHydrogenFailure(
                    "context_unresolved",
                    f"reference H validation requires a known {kind.value} boundary",
                ),
            )
        catalog = site.candidates() if template == stock else None
        if catalog is None:
            continue
        # Boundary admission is independent of protonation preference. A catalog
        # graph here is only a structural stencil, never a preparation choice.
        try:
            context.hydrogen_attachments(
                residue.residue_id,
                site,
                MicrostateResolution(
                    MicrostateResolutionStatus.RESOLVED,
                    candidates=(catalog.candidates[0],),
                ),
            )
        except ValueError:
            continue
        sites.append((site, catalog.candidates))
        excluded.update(atom.name for atom in catalog.candidates[0].atoms)
        controlled.update(
            frozenset((bond.atom_name_1, bond.atom_name_2))
            for candidate in catalog.candidates
            for bond in candidate.bonds
        )
        parent = "N" if kind is PolymerChemicalSite.BACKBONE_N else "C"
        for bond in context.external_heavy_bonds(AtomRef(residue.residue_id, parent)):
            external.add(
                frozenset(
                    context.source.constitution.atom_ref_at(index)
                    for index in bond.endpoint_pair()
                )
            )
        if (
            kind is PolymerChemicalSite.BACKBONE_C
            and site.linkage is PeptideLinkage.LINKED
        ):
            oxt_neighbors = {
                bond.atom_name_2 if bond.atom_name_1 == "OXT" else bond.atom_name_1
                for bond in reference.heavy_bonds()
                if "OXT" in (bond.atom_name_1, bond.atom_name_2)
            }
            if not residue.has_atom_site("OXT") and oxt_neighbors == {"C"}:
                omitted.add("OXT")
                excluded.add("OXT")

    aliases = dict(template.polymer_hydrogen_parent_aliases())
    for atom in residue.atom_sites:
        if atom.is_hydrogen() and atom.name.startswith(("D", "T")):
            normalized = "H" + atom.name[1:]
            parent = reference.hydrogen_anchor_atom_name(normalized) or aliases.get(
                normalized
            )
            if parent is not None:
                aliases[atom.name] = parent
    assessment = assess_reference_hydrogens(
        context.source,
        residue,
        reference,
        scope=ReferenceHydrogenScope(
            excluded_parent_names=frozenset(excluded),
            omitted_heavy_names=frozenset(omitted),
            external_pairs=frozenset(external),
            controlled_bond_pairs=frozenset(controlled),
            hydrogen_aliases=tuple(aliases.items()),
        ),
    )
    if isinstance(assessment, ReferenceHydrogenFailure):
        return PolymerReferenceHydrogenAssessment(None, assessment)
    if assessment.coverage_state() is not HydrogenCoverageState.COMPLETE:
        return PolymerReferenceHydrogenAssessment(None, unavailable)
    counts = Counter(parent for _, parent, _ in assessment.hydrogen_parents)
    for site, candidates in sites:
        if not any(
            _terminal_matches(context, residue, candidate, counts)
            for candidate in candidates
        ):
            return PolymerReferenceHydrogenAssessment(
                None,
                ReferenceHydrogenFailure(
                    "reference_state_not_covered",
                    f"existing H, charges or orders at {site.kind.value} do not "
                    "match an admitted state under the current peptide boundary",
                ),
            )
    return PolymerReferenceHydrogenAssessment(HydrogenCoverageState.COMPLETE, None)


def _terminal_matches(
    context: PolymerMicrostateContext,
    residue: ResidueSite,
    candidate: MicrostateGraph,
    counts: Counter[str],
) -> bool:
    if any(counts[atom.name] != atom.hydrogens for atom in candidate.atoms):
        return False
    snapshots = [(context.source.constitution, context.source.topology)]
    observation = context.source.provenance.ingress.observation
    if observation is not None:
        snapshots.append((observation.constitution, observation.topology))
    site_names = {atom.name for atom in candidate.atoms}
    for constitution, topology in snapshots:
        for atom in candidate.atoms:
            index = constitution.resolve_atom_index(
                AtomRef(residue.residue_id, atom.name)
            )
            if index is not None and topology.formal_charge(index) not in (
                None,
                atom.charge,
            ):
                return False
        incident = topology.incident_bonds_by_atom()
        bonds = dict.fromkeys(
            bond
            for atom in candidate.atoms
            for index in (
                constitution.resolve_atom_index(AtomRef(residue.residue_id, atom.name)),
            )
            if index is not None
            for bond in incident.get(index, ())
        )
        for bond in bonds:
            first, second = (
                constitution.atom_ref_at(index) for index in bond.endpoint_pair()
            )
            if (
                first.residue_id == second.residue_id == residue.residue_id
                and first.atom_name in site_names
                and second.atom_name in site_names
                and (
                    bond.aromatic
                    or (
                        bond.order is not None
                        and bond.order
                        != candidate.bond_order(first.atom_name, second.atom_name)
                    )
                )
            ):
                return False
    return True

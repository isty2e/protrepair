"""Assess a bound request against one live site, independently of execution."""

from dataclasses import dataclass

from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.microstate.bound_protonation import (
    BoundSiteProtonations,
    active_site_protonations,
)
from protrepair.chemistry.microstate.catalog import PolymerChemicalSite
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.graph import MicrostateGraph
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.preparation import AminoAcidSitePreparation
from protrepair.chemistry.microstate.resolution import (
    MicrostateResolution,
    MicrostateResolutionStatus,
    MicrostateSelectionBasis,
    SiteProtonationReason,
)
from protrepair.chemistry.microstate.selection import SiteProtonation


@dataclass(frozen=True, slots=True)
class RequestedSiteAssessment:
    """A source-compatible target and its actually realized drawing, if any."""

    resolution: MicrostateResolution
    target: AminoAcidSitePreparation | None = None
    realized: MicrostateGraph | None = None

    @classmethod
    def observe(
        cls,
        binding: BoundSiteProtonations,
        context: PolymerMicrostateContext,
        library: ComponentLibrary,
        request: SiteProtonation,
    ) -> "RequestedSiteAssessment":
        if binding is not active_site_protonations():
            raise ValueError("site assessment requires its original workflow binding")
        rid = request.residue_id
        structure = context.source
        failure = binding.identity_failure(structure, request)
        if failure is not None:
            return cls(
                MicrostateResolution(
                    MicrostateResolutionStatus.UNSUPPORTED,
                    reason=failure,
                )
            )
        residue = structure.constitution.residue_site_at(
            structure.constitution.residue_index(rid)
        )
        template = library.get(residue.component_id)
        if template is None:
            return cls(
                MicrostateResolution(
                    MicrostateResolutionStatus.UNSUPPORTED,
                    details=("no component definition",),
                    reason=SiteProtonationReason.UNSUPPORTED_COMPONENT,
                )
            )
        site = (
            PolymerMicrostateSite(template, request.site)
            if request.site is PolymerChemicalSite.SIDECHAIN
            else context.backbone_site(
                rid, template, request.site, assume_free_chain_ends=True
            )
        )
        target = AminoAcidSitePreparation(
            context, rid, site, preferences=(), retain_applied_choice=False
        )
        return cls(target.resolution, target, target.realized_requested_graph())

    def requires_commit(self) -> bool:
        if self.target is None or self.resolution.graph is None:
            return False
        if self.realized is None:
            return True
        target = self.target
        names = {atom.name for atom in self.realized.atoms}
        saved = tuple(
            choice
            for choice in target.context.source.provenance.microstate_choices
            if choice.residue_id == target.residue_id
            and any(a.name in names for a in choice.graph.atoms)
        )
        return not (
            len(saved) == 1
            and saved[0].basis is MicrostateSelectionBasis.COMMITTED_REQUEST
            and saved[0].matches_site(
                target.residue_id, target.site.template.component_id, self.realized
            )
            and saved[0].graph.protonation_key() == self.realized.protonation_key()
        )

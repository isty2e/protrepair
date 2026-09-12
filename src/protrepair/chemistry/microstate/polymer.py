"""Pure interpretation of original observations against standard site chemistry."""

from collections import Counter
from dataclasses import dataclass, replace

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.microstate.catalog import (
    PeptideLinkage,
    PolymerChemicalSite,
    standard_microstate_candidates,
)
from protrepair.chemistry.microstate.graph import MicrostateGraph
from protrepair.chemistry.microstate.resolution import (
    AppliedMicrostateOverride,
    HydrogenAttachmentBasis,
    MicrostateConstraints,
    MicrostateResolution,
    MicrostateResolutionStatus,
    ObservedHydrogenAttachment,
)
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.structure.constitution import ResidueSite
from protrepair.structure.labels import AtomRef
from protrepair.structure.observation import StructureObservation
from protrepair.structure.topology import BondRelationshipType


@dataclass(frozen=True, slots=True)
class PolymerMicrostateSite:
    """Interpret original evidence for one standard chemical site.

    Parameters
    ----------
    template : ResidueTemplate
        Active template. Custom chemistry/semantics are refused, not replaced by
        a standard graph just because the component has a familiar name.
    kind : PolymerChemicalSite
        Independent side-chain or terminal decision unit.
    linkage : PeptideLinkage
        Caller-established backbone linkage. Must be UNKNOWN for a side chain.
    free_terminal_assumption : bool
        Mark FREE supplied as a preparation assumption rather than established
        chemical context. Only allowed for a free backbone site; it neither
        becomes source evidence nor overrides contrary source chemistry.

    Notes
    -----
    This resolver neither mutates a structure nor selects a physiological pH.
    Callers must bind current linkage or mark a preparation assumption, and
    rebind after chemical edits. Original observations cannot describe newly
    introduced crosslinks.

    Raises
    ------
    TypeError
        Site, linkage or assumption has a noncanonical type.
    ValueError
        A backbone linkage is supplied for a side-chain site, or a free-terminal
        assumption is attached to a site that is not FREE.
    """

    template: ResidueTemplate
    kind: PolymerChemicalSite
    linkage: PeptideLinkage = PeptideLinkage.UNKNOWN
    free_terminal_assumption: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PolymerChemicalSite) or not isinstance(
            self.linkage, PeptideLinkage
        ):
            raise TypeError("site and linkage must be canonical enum members")
        if type(self.free_terminal_assumption) is not bool:
            raise TypeError("free-terminal assumption must be a boolean")
        if (
            self.kind is PolymerChemicalSite.SIDECHAIN
            and self.linkage is not PeptideLinkage.UNKNOWN
        ):
            raise ValueError("side-chain sites do not have a backbone linkage")
        if self.free_terminal_assumption and self.linkage is not PeptideLinkage.FREE:
            raise ValueError(
                "a free-terminal assumption requires FREE backbone linkage"
            )

    def resolve(
        self,
        residue: ResidueSite,
        observation: StructureObservation | None,
        *,
        override: MicrostateConstraints | None = None,
        applied_override: AppliedMicrostateOverride | None = None,
        preferences: tuple[MicrostateConstraints, ...] = (),
    ) -> MicrostateResolution:
        """Resolve chemistry without treating generated atoms as source evidence.

        Parameters
        ----------
        residue : ResidueSite
            Current scaffold with canonical heavy-atom names.
        observation : StructureObservation or None
            Original selected input. None is unavailable, not neutral chemistry.
        override : MicrostateConstraints or None
            Explicit site-scoped authority to supersede interpretable source facts.
        applied_override : AppliedMicrostateOverride or None
            Previously applied explicit choice. Reused only for the same component,
            site atoms and boundary valence. A new override takes precedence.
            Current boundary endpoints still require validation by the caller.
        preferences : tuple[MicrostateConstraints, ...]
            Ordered preferences among source-compatible candidates only.

        Returns
        -------
        MicrostateResolution
            Coupled graph, ambiguity, conflict, or explicit inapplicability.
            Missing heavy atoms and uninterpretable H attachments are insufficient
            even with an override; an override cannot invent an atom mapping.

        Raises
        ------
        ValueError
            An otherwise applicable resolution receives an empty override.
        """
        library = build_standard_component_library()
        stock = library.get(self.template.component_id)
        if (
            stock is None
            or stock != self.template
            or library.normalize_component_id(residue.component_id)
            != stock.component_id
        ):
            return MicrostateResolution(
                MicrostateResolutionStatus.UNSUPPORTED,
                details=("custom or mismatched component chemistry",),
            )
        if (
            self.kind is not PolymerChemicalSite.SIDECHAIN
            and self.linkage is PeptideLinkage.UNKNOWN
        ):
            return MicrostateResolution(
                MicrostateResolutionStatus.INSUFFICIENT,
                details=("backbone linkage is unknown",),
            )
        site = standard_microstate_candidates(
            stock.component_id, self.kind, self.linkage
        )
        if site is None:
            return MicrostateResolution(
                MicrostateResolutionStatus.UNSUPPORTED,
                details=("no supported standard site catalog",),
            )

        reference = site.candidates[0]
        if override is None and applied_override is not None:
            if not applied_override.matches_site(
                residue.residue_id, residue.component_id, reference
            ):
                return MicrostateResolution(
                    MicrostateResolutionStatus.UNSUPPORTED,
                    details=("applied override no longer matches the chemical site",),
                )
            override = applied_override.constraints()
        required = set(stock.expected_heavy_atom_names()) | {
            atom.name for atom in reference.atoms
        }
        missing = required - set(residue.atom_site_names())
        if missing:
            return MicrostateResolution(
                MicrostateResolutionStatus.INSUFFICIENT,
                details=(f"missing heavy atoms: {', '.join(sorted(missing))}",),
            )
        if not self._scaffold_matches_template(residue):
            return MicrostateResolution(
                MicrostateResolutionStatus.UNSUPPORTED,
                details=("current heavy scaffold differs from standard chemistry",),
            )
        if (
            self.kind is PolymerChemicalSite.BACKBONE_C
            and self.linkage is PeptideLinkage.LINKED
            and residue.has_atom_site("OXT")
        ):
            return MicrostateResolution(
                MicrostateResolutionStatus.UNSUPPORTED,
                details=("linked backbone C also has a terminal oxygen",),
            )
        if any(
            residue.atom_site(atom.name).element != atom.element
            for atom in reference.atoms
        ):
            return MicrostateResolution(
                MicrostateResolutionStatus.UNSUPPORTED,
                details=("site atom elements disagree with standard chemistry",),
            )

        evidence = self._source_evidence(residue, observation, reference)
        if isinstance(evidence, MicrostateResolution):
            return evidence
        constraints, attachments, details = evidence
        if self.free_terminal_assumption:
            details = (*details, "free-terminal linkage is a preparation assumption")
        return replace(
            site.resolve(constraints, override=override, preferences=preferences),
            details=details,
            observed_hydrogens=attachments,
        )

    def _source_evidence(
        self,
        residue: ResidueSite,
        observation: StructureObservation | None,
        graph: MicrostateGraph,
    ) -> (
        tuple[
            MicrostateConstraints,
            tuple[ObservedHydrogenAttachment, ...],
            tuple[str, ...],
        ]
        | MicrostateResolution
    ):
        if observation is None:
            return MicrostateConstraints(), (), ("original observation unavailable",)
        source = observation.constitution.residue_or_ligand(residue.residue_id)
        if source is None:
            return (
                MicrostateConstraints(),
                (),
                ("residue absent from original observation",),
            )
        # Aliases are not all chemically equivalent: HSD/HSE/HSP carry conventions
        # not represented by the current generic HIS template. Do not erase them.
        if source.component_id != self.template.component_id:
            return MicrostateResolution(
                MicrostateResolutionStatus.UNSUPPORTED,
                details=(
                    "source component alias or identity needs explicit reconciliation",
                ),
            )
        if not self._scaffold_matches_template(source):
            return MicrostateResolution(
                MicrostateResolutionStatus.UNSUPPORTED,
                details=("original heavy scaffold differs from standard chemistry",),
            )
        if any(
            source.has_atom_site(atom.name)
            and source.atom_site(atom.name).element != atom.element
            for atom in graph.atoms
        ):
            return MicrostateResolution(
                MicrostateResolutionStatus.UNSUPPORTED,
                details=("source atom elements disagree with the site",),
            )

        names = {atom.name for atom in graph.atoms}
        refs = {AtomRef(residue.residue_id, name) for name in names}
        charges = tuple(
            (name, charge)
            for name in sorted(names)
            if (charge := observation.formal_charge(AtomRef(residue.residue_id, name)))
            is not None
        )
        orders: list[BondDefinition] = []
        hydrogen_neighbors: dict[AtomRef, list[AtomRef]] = {}
        untyped_hydrogen_neighbors: dict[AtomRef, list[AtomRef]] = {}
        attachments: list[ObservedHydrogenAttachment] = []
        external_peptide_partners: set[AtomRef] = set()
        for bond in observation.bonds_for_residue(residue.residue_id):
            first = observation.constitution.atom_ref_at(bond.atom_index_1)
            second = observation.constitution.atom_ref_at(bond.atom_index_2)
            for atom_ref, other_ref in ((first, second), (second, first)):
                if (
                    atom_ref.residue_id == residue.residue_id
                    and observation.constitution.atom_site_at(
                        bond.atom_index_1 if atom_ref == first else bond.atom_index_2
                    ).is_hydrogen()
                ):
                    if bond.relationship_type is BondRelationshipType.COVALENT:
                        hydrogen_neighbors.setdefault(atom_ref, []).append(other_ref)
                    elif bond.relationship_type is BondRelationshipType.UNKNOWN:
                        untyped_hydrogen_neighbors.setdefault(atom_ref, []).append(
                            other_ref
                        )
            if not ({first, second} & refs):
                continue
            first_atom = observation.constitution.atom_site_at(bond.atom_index_1)
            second_atom = observation.constitution.atom_site_at(bond.atom_index_2)
            if first_atom.is_hydrogen() or second_atom.is_hydrogen():
                hydrogen_ref = first if first_atom.is_hydrogen() else second
                if (
                    hydrogen_ref.residue_id != residue.residue_id
                    and bond.relationship_type
                    in (
                        BondRelationshipType.COVALENT,
                        BondRelationshipType.UNKNOWN,
                    )
                ):
                    return MicrostateResolution(
                        MicrostateResolutionStatus.UNSUPPORTED,
                        details=(
                            "external source H attachment requires reconciliation",
                        ),
                    )
                if (
                    bond.relationship_type is BondRelationshipType.COVALENT
                    and bond.order not in (None, 1)
                ):
                    return MicrostateResolution(
                        MicrostateResolutionStatus.CONFLICT,
                        details=("non-single source hydrogen bond order",),
                    )
                continue
            if bond.relationship_type is BondRelationshipType.HYDROGEN_BOND:
                continue
            if bond.relationship_type not in (
                BondRelationshipType.COVALENT,
                BondRelationshipType.UNKNOWN,
            ):
                return MicrostateResolution(
                    MicrostateResolutionStatus.UNSUPPORTED,
                    details=("site participates in unsupported source relationship",),
                )
            if first in refs and second in refs:
                if graph.bond_order(first.atom_name, second.atom_name) is None:
                    return MicrostateResolution(
                        MicrostateResolutionStatus.UNSUPPORTED,
                        details=("source changes the standard site skeleton",),
                    )
                if bond.order is not None:
                    orders.append(
                        BondDefinition(
                            first.atom_name, second.atom_name, order=bond.order
                        )
                    )
                continue
            inside, outside = (first, second) if first in refs else (second, first)
            if outside.residue_id == residue.residue_id:
                expected = next(
                    (
                        b
                        for b in self.template.definition.bonds
                        if {b.atom_name_1, b.atom_name_2}
                        == {inside.atom_name, outside.atom_name}
                    ),
                    None,
                )
                allowed = expected is not None and bond.order in (None, expected.order)
                if (
                    self.kind is PolymerChemicalSite.BACKBONE_C
                    and outside.atom_name == "OXT"
                ):
                    allowed = False
            else:
                expected_names = (
                    ("N", "C")
                    if self.kind is PolymerChemicalSite.BACKBONE_N
                    else ("C", "N")
                )
                allowed = (
                    self.kind is not PolymerChemicalSite.SIDECHAIN
                    and self.linkage is PeptideLinkage.LINKED
                    and (inside.atom_name, outside.atom_name) == expected_names
                    and bond.order in (None, 1)
                )
                if allowed:
                    external_peptide_partners.add(outside)
            if not allowed or len(external_peptide_partners) > 1:
                return MicrostateResolution(
                    MicrostateResolutionStatus.UNSUPPORTED,
                    details=("source external bonds disagree with site boundary",),
                )

        hydrogens: Counter[str] = Counter()
        anchors = self.named_hydrogen_parents(source)
        for hydrogen in observation.hydrogen_atoms(residue.residue_id):
            parents = hydrogen_neighbors.get(hydrogen, [])
            if len(parents) > 1:
                return MicrostateResolution(
                    MicrostateResolutionStatus.CONFLICT,
                    details=("source H has multiple covalent parents",),
                )
            if parents:
                parent = parents[0]
                parent_atom = observation.constitution.resolve_atom_site(parent)
                if (
                    parent.residue_id != residue.residue_id
                    or parent_atom is None
                    or parent_atom.is_hydrogen()
                ):
                    return MicrostateResolution(
                        MicrostateResolutionStatus.UNSUPPORTED,
                        details=("source H parent is not a local heavy atom",),
                    )
                basis = HydrogenAttachmentBasis.SOURCE_COVALENT
            else:
                name = anchors.get(hydrogen.atom_name)
                if name is None:
                    return MicrostateResolution(
                        MicrostateResolutionStatus.INSUFFICIENT,
                        details=(
                            f"unresolved source H attachment: {hydrogen.atom_name}",
                        ),
                    )
                parent = AtomRef(residue.residue_id, name)
                basis = HydrogenAttachmentBasis.STANDARD_NAME
            if any(
                neighbor != parent
                for neighbor in untyped_hydrogen_neighbors.get(hydrogen, ())
            ):
                return MicrostateResolution(
                    MicrostateResolutionStatus.INSUFFICIENT,
                    details=(
                        "untyped H connectivity disagrees with interpreted attachment",
                    ),
                )
            if parent not in refs:
                continue
            if observation.formal_charge(hydrogen) not in (None, 0):
                return MicrostateResolution(
                    MicrostateResolutionStatus.UNSUPPORTED,
                    details=("charged source H is outside the closed-shell catalog",),
                )
            hydrogens[parent.atom_name] += 1
            attachments.append(ObservedHydrogenAttachment(hydrogen, parent, basis))
        return (
            MicrostateConstraints(
                charges=charges,
                minimum_hydrogens=tuple(hydrogens.items()),
                bonds=tuple(orders),
            ),
            tuple(attachments),
            (),
        )

    def _scaffold_matches_template(self, residue: ResidueSite) -> bool:
        expected = set(self.template.expected_atom_names())
        return all(
            atom.element == atom.name[0]
            if atom.name in expected
            else atom.is_hydrogen()
            for atom in residue.atom_sites
        )

    def named_hydrogen_parents(self, residue: ResidueSite) -> dict[str, str]:
        """Interpret standard H names without asserting actual connectivity.

        Parameters
        ----------
        residue : ResidueSite
            Original or current residue carrying H/D/T identities.

        Returns
        -------
        dict[str, str]
            Recognized H names and heavy parents, including terminal and isotope
            aliases. Unrecognized names are omitted, not guessed geometrically.
            Actual bonds must be checked separately by the consumer.
        """
        aliases = {
            "H": "N",
            "H1": "N",
            "H2": "N",
            "H3": "N",
            "1H": "N",
            "2H": "N",
            "3H": "N",
            "HXT": "OXT",
            "HO": "O",
        }
        if self.template.component_id == "HIS":
            aliases.update({"HD1": "ND1", "HE2": "NE2"})
        elif self.template.component_id == "ARG":
            aliases.update({"HH11": "NH1", "HH12": "NH1", "HH21": "NH2", "HH22": "NH2"})
        elif self.template.component_id == "ASP":
            aliases.update({"HD1": "OD1", "HD2": "OD2"})
        elif self.template.component_id == "GLU":
            aliases.update({"HE1": "OE1", "HE2": "OE2"})
        names = {}
        for atom in residue.atom_sites:
            if not atom.is_hydrogen():
                continue
            canonical = atom.name
            isotope_position = 1 if canonical[0].isdigit() else 0
            if (
                atom.element in {"D", "T"}
                and canonical[isotope_position : isotope_position + 1] == atom.element
            ):
                canonical = (
                    canonical[:isotope_position]
                    + "H"
                    + canonical[isotope_position + 1 :]
                )
            names[atom.name] = canonical
        anchors = self.template.template_hydrogen_anchor_by_name(names.values())
        aliases.update(anchors)
        return {
            name: aliases[canonical]
            for name, canonical in names.items()
            if canonical in aliases
        }

    def hydrogen_name_candidates(self, parent_name: str) -> tuple[str, ...]:
        """Return preferred new H names, without selecting a protonation state.

        Parameters
        ----------
        parent_name : str
            Canonical site heavy-atom name.

        Returns
        -------
        tuple[str, ...]
            Template names followed by supported microstate-specific names.
            Existing and original identities take precedence in the context.
        """
        if parent_name == "N":
            return (
                ("H",) if self.linkage is PeptideLinkage.LINKED else ("H1", "H2", "H3")
            )
        template_names = self.template.expected_hydrogen_atom_names()
        anchors = self.template.template_hydrogen_anchor_by_name(template_names)
        names = [name for name in template_names if anchors.get(name) == parent_name]
        extra = {
            "HIS": {"ND1": "HD1", "NE2": "HE2"},
            "ASP": {"OD1": "HD1", "OD2": "HD2"},
            "GLU": {"OE1": "HE1", "OE2": "HE2"},
        }.get(self.template.component_id, {})
        extra.update({"O": "HO", "OXT": "HXT"})
        if parent_name in extra and extra[parent_name] not in names:
            names.append(extra[parent_name])
        return tuple(names)

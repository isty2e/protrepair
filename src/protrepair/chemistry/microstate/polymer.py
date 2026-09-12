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
        Caller-established backbone linkage. Chain position or an absent bond
        record alone cannot establish FREE. Must be UNKNOWN for a side chain.

    Notes
    -----
    This resolver neither mutates a structure nor selects a physiological pH.
    Callers must establish current linkage and rebind after chemical edits;
    original observations cannot describe newly introduced crosslinks.

    Raises
    ------
    TypeError
        Site or linkage is not a canonical enum member.
    ValueError
        A backbone linkage is supplied for a side-chain site.
    """

    template: ResidueTemplate
    kind: PolymerChemicalSite
    linkage: PeptideLinkage = PeptideLinkage.UNKNOWN

    def __post_init__(self) -> None:
        if not isinstance(self.kind, PolymerChemicalSite) or not isinstance(
            self.linkage, PeptideLinkage
        ):
            raise TypeError("site and linkage must be canonical enum members")
        if (
            self.kind is PolymerChemicalSite.SIDECHAIN
            and self.linkage is not PeptideLinkage.UNKNOWN
        ):
            raise ValueError("side-chain sites do not have a backbone linkage")

    def resolve(
        self,
        residue: ResidueSite,
        observation: StructureObservation | None,
        *,
        override: MicrostateConstraints | None = None,
        default: MicrostateConstraints | None = None,
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
        default : MicrostateConstraints or None
            Preference among source-compatible candidates only.

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
        required = set(stock.expected_heavy_atom_names()) | {
            atom.name for atom in reference.atoms
        }
        missing = required - set(residue.atom_site_names())
        if missing:
            return MicrostateResolution(
                MicrostateResolutionStatus.INSUFFICIENT,
                details=(f"missing heavy atoms: {', '.join(sorted(missing))}",),
            )
        if any(
            not atom.is_hydrogen()
            and (
                atom.name not in stock.expected_atom_names()
                or atom.element != atom.name[0]
            )
            for atom in residue.atom_sites
        ):
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
        return replace(
            site.resolve(constraints, override=override, default=default),
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
        if any(
            not atom.is_hydrogen()
            and (
                atom.name not in self.template.expected_atom_names()
                or atom.element != atom.name[0]
            )
            for atom in source.atom_sites
        ):
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
        anchors = self._hydrogen_names(source)
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

    def _hydrogen_names(self, source: ResidueSite) -> dict[str, str]:
        aliases = {
            "H": "N",
            "H1": "N",
            "H2": "N",
            "H3": "N",
            "1H": "N",
            "2H": "N",
            "3H": "N",
            "HXT": "OXT",
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
        for atom in source.atom_sites:
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

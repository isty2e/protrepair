"""Coupled site chemistry, evidence authority and native closed-shell graphs."""

from dataclasses import replace

import pytest
from rdkit import Chem

from protrepair.chemistry.component.graph import BondDefinition
from protrepair.chemistry.microstate.catalog import (
    PeptideLinkage,
    PolymerChemicalSite,
    standard_microstate_candidates,
)
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.graph import MicrostateAtom, MicrostateGraph
from protrepair.chemistry.microstate.polymer import PolymerMicrostateSite
from protrepair.chemistry.microstate.preparation import PolymerMicrostatePreparation
from protrepair.chemistry.microstate.resolution import (
    HydrogenAttachmentBasis,
    MicrostateConstraints,
    MicrostateResolution,
    MicrostateResolutionStatus,
    MicrostateSelectionBasis,
    MicrostateSite,
    ObservedHydrogenAttachment,
)
from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.io import read_structure_string
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import AtomSite
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.provenance import FileFormat
from protrepair.structure.slots import ResidueIndex
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    StructureTopology,
    TopologyBond,
)


def _site(component: str) -> MicrostateSite:
    result = standard_microstate_candidates(component, PolymerChemicalSite.SIDECHAIN)
    assert result is not None
    return result


def _source(
    component: str,
    *,
    hydrogens: tuple[tuple[str, str], ...] = (),
    charges: tuple[tuple[str, int], ...] = (),
    terminal: bool = False,
    extra_pdb: str = "",
) -> ProteinStructure:
    template = build_standard_component_library().require(component)
    atoms = [(name, name[0]) for name in template.expected_heavy_atom_names()]
    if terminal:
        atoms.append(("OXT", "O"))
    atoms.extend(hydrogens)
    charge_by_name = dict(charges)
    lines = []
    for index, (name, element) in enumerate(atoms, 1):
        charge = charge_by_name.get(name)
        token = "" if charge is None else f"{abs(charge)}{'-' if charge < 0 else '+'}"
        lines.append(
            f"ATOM  {index:5d} {name:^4} {component:>3} A   1    "
            f"{index * 1.1:8.3f}{0:8.3f}{0:8.3f}{1:6.2f}{20:6.2f}"
            f"          {element:>2}{token:>2}\n"
        )
    return read_structure_string("".join(lines) + extra_pdb + "END\n", FileFormat.PDB)


def _resolve(
    structure: ProteinStructure,
    *,
    kind: PolymerChemicalSite = PolymerChemicalSite.SIDECHAIN,
    linkage: PeptideLinkage = PeptideLinkage.UNKNOWN,
    override: MicrostateConstraints | None = None,
) -> MicrostateResolution:
    residue = structure.constitution.residue_site_at(ResidueIndex(0))
    return PolymerMicrostateSite(
        build_standard_component_library().require(residue.component_id), kind, linkage
    ).resolve(residue, structure.provenance.ingress.observation, override=override)


@pytest.mark.parametrize(
    "kind", [PolymerChemicalSite.BACKBONE_N, PolymerChemicalSite.BACKBONE_C]
)
def test_backbone_resolution_requires_its_boundary_not_unrelated_sidechain(kind):
    source = _source("HIS")
    residue = source.constitution.residue_site_at(ResidueIndex(0))
    backbone = replace(
        residue,
        atom_sites=tuple(
            a for a in residue.atom_sites if a.name in {"N", "CA", "C", "O"}
        ),
    )
    site = PolymerMicrostateSite(
        build_standard_component_library().require("HIS"), kind, PeptideLinkage.LINKED
    )
    assert (
        site.resolve(backbone, source.provenance.ingress.observation).graph is not None
    )
    incomplete = replace(
        backbone, atom_sites=tuple(a for a in backbone.atom_sites if a.name != "CA")
    )
    assert (
        site.resolve(incomplete, source.provenance.ingress.observation).status
        is MicrostateResolutionStatus.INSUFFICIENT
    )
    sidechain = PolymerMicrostateSite(site.template, PolymerChemicalSite.SIDECHAIN)
    assert (
        sidechain.resolve(backbone, source.provenance.ingress.observation).status
        is MicrostateResolutionStatus.INSUFFICIENT
    )


def _with_source_bonds(
    structure: ProteinStructure,
    *specs: tuple[str, str, BondRelationshipType, int | None],
    second_residue_id: ResidueId | None = None,
) -> ProteinStructure:
    observation = structure.provenance.ingress.observation
    assert observation is not None
    bonds = []
    for first, second, relationship, order in specs:
        first_index = observation.constitution.resolve_atom_index(
            AtomRef(ResidueId("A", 1), first)
        )
        second_index = observation.constitution.resolve_atom_index(
            AtomRef(second_residue_id or ResidueId("A", 1), second)
        )
        assert first_index is not None and second_index is not None
        bonds.append(
            TopologyBond(
                first_index,
                second_index,
                order=order,
                relationship_type=relationship,
                provenance=BondProvenance.SOURCE_EXPLICIT,
                source_metadata=SourceBondMetadata(
                    record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
                    reported_order=order,
                    reported_relationship_type=relationship,
                ),
            )
        )
    topology = StructureTopology(
        constitution=observation.constitution,
        atom_topologies=observation.topology.atom_topologies,
        bonds=tuple(bonds),
    )
    observed = StructureObservation(
        observation.constitution, observation.geometry, topology
    )
    return structure.with_provenance(
        replace(
            structure.provenance,
            ingress=replace(structure.provenance.ingress, observation=observed),
        )
    )


@pytest.mark.parametrize("element", ("H", "D", "T"))
def test_fixed_h_identity_retains_source_alias_and_isotope(element: str) -> None:
    source = _source("SER", hydrogens=(("HB3", element),))
    preparation = PolymerMicrostatePreparation(
        PolymerMicrostateContext(source), build_standard_component_library()
    )
    selected = preparation.fixed_hydrogen_atom_sites(ResidueId("A", 1))
    cb = [(atom.name, atom.element) for atom, parent in selected if parent == "CB"]
    assert len(cb) == 2
    assert ("HB3", element) in cb


def test_fixed_h_identity_uses_explicit_source_parent_for_nonstandard_name() -> None:
    source = _with_source_bonds(
        _source("SER", hydrogens=(("DX", "D"),)),
        ("CB", "DX", BondRelationshipType.COVALENT, 1),
    )
    preparation = PolymerMicrostatePreparation(
        PolymerMicrostateContext(source), build_standard_component_library()
    )
    selected = preparation.fixed_hydrogen_atom_sites(ResidueId("A", 1))
    assert (AtomSite("DX", "D"), "CB") in selected
    assert len([atom for atom, parent in selected if parent == "CB"]) == 2


@pytest.mark.parametrize("failure", ("overfull", "unresolved", "charged", "multiple"))
def test_fixed_h_identity_refuses_incompatible_source(failure: str) -> None:
    source = _source(
        "SER",
        hydrogens=(
            (("HB1", "H"), ("HB2", "H"), ("HB3", "H"))
            if failure == "overfull"
            else (("HX" if failure == "unresolved" else "HB2", "H"),)
        ),
        charges=(("HB2", 1),) if failure == "charged" else (),
    )
    if failure == "multiple":
        source = _with_source_bonds(
            source,
            ("CB", "HB2", BondRelationshipType.COVALENT, 1),
            ("CA", "HB2", BondRelationshipType.COVALENT, 1),
        )
    preparation = PolymerMicrostatePreparation(
        PolymerMicrostateContext(source), build_standard_component_library()
    )
    with pytest.raises(ValueError):
        preparation.fixed_hydrogen_atom_sites(ResidueId("A", 1))


def _native_graph(graph: MicrostateGraph) -> Chem.Mol:
    molecule = Chem.RWMol()
    indices = {}
    for atom in graph.atoms:
        native = Chem.Atom(atom.element)
        native.SetFormalCharge(atom.charge)
        native.SetNumExplicitHs(atom.hydrogens)
        native.SetNoImplicit(True)
        indices[atom.name] = molecule.AddAtom(native)
    for bond in graph.bonds:
        molecule.AddBond(
            indices[bond.atom_name_1],
            indices[bond.atom_name_2],
            Chem.BondType.DOUBLE if bond.order == 2 else Chem.BondType.SINGLE,
        )
    # Cap explicit boundary valence, rather than calling an open fragment a radical.
    for atom in graph.atoms:
        for _ in range(atom.boundary_order):
            cap = Chem.Atom("C")
            cap.SetNumExplicitHs(3)
            cap.SetNoImplicit(True)
            molecule.AddBond(
                indices[atom.name], molecule.AddAtom(cap), Chem.BondType.SINGLE
            )
    result = molecule.GetMol()
    Chem.SanitizeMol(result)
    return result


@pytest.mark.parametrize(
    "component, representations, states",
    (("ARG", 6, 4), ("HIS", 4, 3), ("ASP", 4, 3), ("GLU", 4, 3), ("LYS", 2, 2)),
)
def test_catalog_has_coupled_states_without_implicit_default(
    component: str, representations: int, states: int
) -> None:
    site = _site(component)
    assert len(site.candidates) == representations
    assert len({graph.protonation_key() for graph in site.candidates}) == states
    result = site.resolve(MicrostateConstraints())
    assert result.status is MicrostateResolutionStatus.AMBIGUOUS
    assert result.graph is None
    for graph in site.candidates:
        native = _native_graph(graph)
        assert all(atom.GetNumRadicalElectrons() == 0 for atom in native.GetAtoms())
        assert Chem.GetFormalCharge(native) == sum(atom.charge for atom in graph.atoms)
        for index, atom in enumerate(graph.atoms):
            actual = native.GetAtomWithIdx(index)
            assert actual.GetTotalNumHs() == atom.hydrogens
            assert actual.GetFormalCharge() == atom.charge
            if component in {"ARG", "HIS", "ASP", "GLU"}:
                assert actual.GetHybridization() == Chem.HybridizationType.SP2
            if component == "HIS":
                assert actual.GetIsAromatic()


@pytest.mark.parametrize(
    "component", tuple(build_standard_component_library().templates)
)
@pytest.mark.parametrize("linkage", (PeptideLinkage.LINKED, PeptideLinkage.FREE))
@pytest.mark.parametrize(
    "kind", (PolymerChemicalSite.BACKBONE_N, PolymerChemicalSite.BACKBONE_C)
)
def test_terminal_and_linked_graphs_cover_standard_residues(
    component: str, linkage: PeptideLinkage, kind: PolymerChemicalSite
) -> None:
    site = standard_microstate_candidates(component, kind, linkage)
    assert site is not None
    for graph in site.candidates:
        native = _native_graph(graph)
        assert all(atom.GetNumRadicalElectrons() == 0 for atom in native.GetAtoms())
        if kind is PolymerChemicalSite.BACKBONE_N:
            atom = graph.atom("N")
            assert (
                atom.hydrogens
                == 2
                - int(component == "PRO")
                - int(linkage is PeptideLinkage.LINKED)
                + atom.charge
            )
        elif linkage is PeptideLinkage.LINKED:
            assert graph.atom("C").hydrogens == 0
            assert graph.atom("O").hydrogens == 0
            assert graph.bond_order("C", "O") == 2
    assert (site.resolve(MicrostateConstraints()).graph is not None) == (
        linkage is PeptideLinkage.LINKED
    )


@pytest.mark.parametrize("nitrogen", ("NE", "NH1", "NH2"))
def test_arg_cation_localization_preserves_exact_source_constraint(
    nitrogen: str,
) -> None:
    result = _site("ARG").resolve(MicrostateConstraints(charges=((nitrogen, 1),)))
    graph = result.graph
    assert graph is not None
    assert graph.atom(nitrogen).charge == 1
    assert graph.bond_order("CZ", nitrogen) == 2
    assert {
        atom.name: atom.hydrogens for atom in graph.atoms if atom.element == "N"
    } == {"NE": 1, "NH1": 2, "NH2": 2}


def test_partial_h_does_not_distinguish_neutral_his_from_cation() -> None:
    partial = MicrostateConstraints(minimum_hydrogens=(("ND1", 1),))
    site = _site("HIS")
    assert site.resolve(partial).status is MicrostateResolutionStatus.AMBIGUOUS
    result = site.resolve(replace(partial, charges=(("ND1", 0), ("NE2", 0))))
    assert result.graph is not None
    assert result.graph.atom("ND1").hydrogens == 1
    assert result.graph.atom("NE2").hydrogens == 0
    both = site.resolve(
        MicrostateConstraints(minimum_hydrogens=(("ND1", 1), ("NE2", 1)))
    )
    assert both.graph is not None and both.graph.protonation_key()[0] == 1
    assert len(both.candidates) == 2


@pytest.mark.parametrize("oxygen", ("OD1", "OD2"))
def test_acid_h_and_anion_localization_are_not_interchangeable(oxygen: str) -> None:
    site = _site("ASP")
    acid = site.resolve(MicrostateConstraints(minimum_hydrogens=((oxygen, 1),)))
    assert acid.graph is not None
    assert acid.graph.protonation_key()[0] == 0
    assert acid.graph.bond_order("CG", oxygen) == 1
    anion = site.resolve(MicrostateConstraints(charges=((oxygen, -1),)))
    assert anion.graph is not None and anion.graph.atom(oxygen).hydrogens == 0
    conflict = site.resolve(
        MicrostateConstraints(charges=((oxygen, -1),), minimum_hydrogens=((oxygen, 1),))
    )
    assert conflict.status is MicrostateResolutionStatus.CONFLICT


def test_default_never_replaces_source_but_override_records_replacement() -> None:
    site = _site("LYS")
    source = MicrostateConstraints(charges=(("NZ", 0),))
    cation = MicrostateConstraints(hydrogens=(("NZ", 3),))
    result = site.resolve(source, preferences=(cation,))
    assert result.graph is not None and result.graph.atom("NZ").charge == 0
    result = site.resolve(source, override=cation)
    assert result.graph is not None and result.graph.atom("NZ").charge == 1
    assert result.basis is MicrostateSelectionBasis.OVERRIDE
    assert result.superseded_source == source
    result = site.resolve(MicrostateConstraints(), preferences=(cation,))
    assert result.basis is MicrostateSelectionBasis.DEFAULT
    assert result.graph is not None and result.graph.atom("NZ").hydrogens == 3
    conflict = MicrostateConstraints(charges=(("NZ", -1),))
    assert (
        site.resolve(conflict, preferences=(cation,)).status
        is MicrostateResolutionStatus.CONFLICT
    )
    with pytest.raises(ValueError, match="override"):
        site.resolve(source, override=MicrostateConstraints())


def test_override_keeps_source_compatible_resonance_before_canonical_choice() -> None:
    site = _site("ARG")
    source = MicrostateConstraints(charges=(("NH1", 1),))
    result = site.resolve(
        source,
        override=MicrostateConstraints(hydrogens=(("NH1", 2), ("NH2", 2), ("NE", 1))),
    )
    assert result.graph is not None and result.graph.atom("NH1").charge == 1
    assert result.superseded_source.is_empty()


@pytest.mark.parametrize(
    "element, name", (("H", "HD1"), ("D", "DD1"), ("T", "TD1"), ("D", "HD1"))
)
def test_original_hydrogen_isotopes_remain_evidence_after_stripping(
    element: str, name: str
) -> None:
    source = _source(
        "HIS", hydrogens=((name, element),), charges=(("ND1", 0), ("NE2", 0))
    )
    before = _resolve(source)
    after = _resolve(source.without_hydrogens())
    assert before == after
    assert after.graph is not None and after.graph.atom("ND1").hydrogens == 1
    assert len(after.observed_hydrogens) == 1
    attachment = after.observed_hydrogens[0]
    assert attachment.hydrogen.atom_name == name
    assert attachment.parent.atom_name == "ND1"
    assert attachment.basis is HydrogenAttachmentBasis.STANDARD_NAME


def test_current_charge_edits_are_not_new_source_authority() -> None:
    source = _source("LYS", charges=(("NZ", 0),))
    edited = source.with_updated_residue_facets(
        source.constitution.residue_site_at(ResidueIndex(0)),
        residue_geometry=source.residue_geometry(ResidueIndex(0)),
        formal_charge_by_atom_name=(("NZ", 1),),
    )
    assert _resolve(edited) == _resolve(source)
    assert (
        _resolve(edited).graph
        == _site("LYS").resolve(MicrostateConstraints(charges=(("NZ", 0),))).graph
    )


def test_covalent_h_attachment_not_arbitrary_h_name_is_authoritative() -> None:
    source = _source("HIS", hydrogens=(("QH", "H"),), charges=(("ND1", 0), ("NE2", 0)))
    assert _resolve(source).status is MicrostateResolutionStatus.INSUFFICIENT
    typed = _with_source_bonds(source, ("QH", "ND1", BondRelationshipType.COVALENT, 1))
    result = _resolve(typed)
    assert result.graph is not None and result.graph.atom("ND1").hydrogens == 1
    assert result.observed_hydrogens[0].basis is HydrogenAttachmentBasis.SOURCE_COVALENT
    for relationship in (
        BondRelationshipType.UNKNOWN,
        BondRelationshipType.HYDROGEN_BOND,
        BondRelationshipType.METAL_COORDINATION,
    ):
        assert (
            _resolve(
                _with_source_bonds(source, ("QH", "ND1", relationship, None))
            ).status
            is MicrostateResolutionStatus.INSUFFICIENT
        )


def test_unknown_connectivity_cannot_override_h_naming() -> None:
    source = _source("HIS", hydrogens=(("HD1", "H"),))
    result = _resolve(
        _with_source_bonds(source, ("HD1", "NE2", BondRelationshipType.UNKNOWN, None))
    )
    assert result.status is MicrostateResolutionStatus.INSUFFICIENT


@pytest.mark.parametrize(
    "order, expected",
    (
        (1, MicrostateResolutionStatus.RESOLVED),
        (2, MicrostateResolutionStatus.CONFLICT),
        (None, MicrostateResolutionStatus.RESOLVED),
    ),
)
def test_source_order_filters_before_resonance_selection(
    order: int | None, expected: MicrostateResolutionStatus
) -> None:
    source = _source("ASP", charges=(("OD1", -1),))
    result = _resolve(
        _with_source_bonds(source, ("CG", "OD1", BondRelationshipType.COVALENT, order))
    )
    assert result.status is expected
    if result.graph is not None:
        assert result.graph.bond_order("CG", "OD1") == 1


@pytest.mark.parametrize(
    "problem", ("missing", "custom", "alias", "unknown_linkage", "external", "skeleton")
)
def test_unsupported_or_incomplete_chemistry_never_falls_back(problem: str) -> None:
    source = _source("LYS")
    residue = source.constitution.residue_site_at(ResidueIndex(0))
    template = build_standard_component_library().require("LYS")
    kind = PolymerChemicalSite.SIDECHAIN
    expected = MicrostateResolutionStatus.UNSUPPORTED
    if problem == "missing":
        residue = replace(
            residue,
            atom_sites=tuple(atom for atom in residue.atom_sites if atom.name != "CE"),
        )
        expected = MicrostateResolutionStatus.INSUFFICIENT
    elif problem == "custom":
        template = replace(
            template, definition=replace(template.definition, formal_charges={"NZ": 1})
        )
    elif problem == "alias":
        source = _source("HSD")
        residue = source.constitution.residue_site_at(ResidueIndex(0))
        template = build_standard_component_library().require("HIS")
    elif problem == "unknown_linkage":
        kind = PolymerChemicalSite.BACKBONE_N
        expected = MicrostateResolutionStatus.INSUFFICIENT
    elif problem == "external":
        source = _with_source_bonds(
            source, ("NZ", "CA", BondRelationshipType.COVALENT, 1)
        )
    else:
        source = _with_source_bonds(
            source, ("NZ", "CE", BondRelationshipType.METAL_COORDINATION, None)
        )
    result = PolymerMicrostateSite(template, kind).resolve(
        residue, source.provenance.ingress.observation
    )
    assert result.status is expected and result.graph is None


@pytest.mark.parametrize("component", ("ALA", "PRO"))
def test_explicit_free_n_context_does_not_automatically_choose_cation(
    component: str,
) -> None:
    source = _source(component)
    free = _resolve(
        source, kind=PolymerChemicalSite.BACKBONE_N, linkage=PeptideLinkage.FREE
    )
    assert free.status is MicrostateResolutionStatus.AMBIGUOUS
    linked = _resolve(
        source, kind=PolymerChemicalSite.BACKBONE_N, linkage=PeptideLinkage.LINKED
    )
    assert linked.graph is not None and linked.graph.atom("N").hydrogens == int(
        component != "PRO"
    )
    positive = _resolve(
        _source(component, charges=(("N", 1),)),
        kind=PolymerChemicalSite.BACKBONE_N,
        linkage=PeptideLinkage.FREE,
    )
    assert positive.graph is not None and positive.graph.atom("N").hydrogens == 3 - int(
        component == "PRO"
    )


def test_terminal_c_needs_oxygen_but_not_an_implicit_protonation_default() -> None:
    assert (
        _resolve(
            _source("ALA"),
            kind=PolymerChemicalSite.BACKBONE_C,
            linkage=PeptideLinkage.FREE,
        ).status
        is MicrostateResolutionStatus.INSUFFICIENT
    )
    assert (
        _resolve(
            _source("ALA", terminal=True),
            kind=PolymerChemicalSite.BACKBONE_C,
            linkage=PeptideLinkage.FREE,
        ).status
        is MicrostateResolutionStatus.AMBIGUOUS
    )
    source = _source("ALA", terminal=True, hydrogens=(("HXT", "H"),))
    result = _resolve(
        source, kind=PolymerChemicalSite.BACKBONE_C, linkage=PeptideLinkage.FREE
    )
    assert result.graph is not None and result.graph.atom("OXT").hydrogens == 1
    assert (
        _resolve(
            source, kind=PolymerChemicalSite.BACKBONE_C, linkage=PeptideLinkage.LINKED
        ).status
        is MicrostateResolutionStatus.UNSUPPORTED
    )


def test_canonical_models_reject_incoherent_valence_and_status() -> None:
    with pytest.raises(ValueError, match="valence"):
        MicrostateGraph((MicrostateAtom("N", "N", 0, 4, 0),), ())
    with pytest.raises(TypeError):
        MicrostateAtom("N", "N", True, 3, 0)
    with pytest.raises(ValueError, match="unique"):
        MicrostateGraph((MicrostateAtom("N", "N", 0, 3, 0),) * 2, ())
    with pytest.raises(ValueError, match="skeleton"):
        MicrostateSite((_site("LYS").candidates[0], _site("ARG").candidates[0]))
    with pytest.raises(ValueError, match="status"):
        MicrostateResolution(
            MicrostateResolutionStatus.RESOLVED, _site("LYS").candidates
        )
    with pytest.raises(ValueError, match="unique"):
        MicrostateConstraints(charges=(("NZ", 0), ("NZ", 1)))
    with pytest.raises(ValueError):
        MicrostateConstraints(minimum_hydrogens=(("NZ", -1),))
    with pytest.raises(TypeError):
        MicrostateConstraints(charges=(("NZ", True),))


def test_unknown_constraint_subject_does_not_disappear() -> None:
    site = _site("LYS")
    assert (
        site.resolve(MicrostateConstraints(charges=(("OTHER", 0),))).status
        is MicrostateResolutionStatus.CONFLICT
    )
    assert (
        site.resolve(
            MicrostateConstraints(bonds=(BondDefinition("NZ", "OTHER"),))
        ).status
        is MicrostateResolutionStatus.CONFLICT
    )


def test_unavailable_observation_is_reported_without_assuming_neutrality() -> None:
    source = _source("LYS")
    residue = source.constitution.residue_site_at(ResidueIndex(0))
    site = PolymerMicrostateSite(
        build_standard_component_library().require("LYS"), PolymerChemicalSite.SIDECHAIN
    )
    result = site.resolve(residue, None)
    assert result.status is MicrostateResolutionStatus.AMBIGUOUS
    assert result.details == ("original observation unavailable",)


def test_generated_h_inventory_is_not_original_evidence() -> None:
    original = _source("LYS")
    generated = _source("LYS", hydrogens=(("HZ1", "H"), ("HZ2", "H"), ("HZ3", "H")))
    generated = generated.with_provenance(original.provenance)
    assert _resolve(generated).status is MicrostateResolutionStatus.AMBIGUOUS
    assert _resolve(generated).observed_hydrogens == ()


def test_hydrogen_overflow_and_multiple_parents_are_conflicts() -> None:
    source = _source(
        "LYS",
        hydrogens=(("HZ1", "H"), ("HZ2", "H"), ("HZ3", "H")),
        charges=(("NZ", 0),),
    )
    assert _resolve(source).status is MicrostateResolutionStatus.CONFLICT
    multiple = _with_source_bonds(
        source,
        ("HZ1", "NZ", BondRelationshipType.COVALENT, 1),
        ("HZ1", "CE", BondRelationshipType.COVALENT, 1),
    )
    assert _resolve(multiple).status is MicrostateResolutionStatus.CONFLICT
    double = _with_source_bonds(source, ("HZ1", "NZ", BondRelationshipType.COVALENT, 2))
    assert _resolve(double).status is MicrostateResolutionStatus.CONFLICT


def test_override_retains_original_h_mapping_and_exact_superseded_constraints() -> None:
    source = _source("LYS", hydrogens=(("HZ1", "H"), ("HZ2", "D"), ("HZ3", "T")))
    result = _resolve(source, override=MicrostateConstraints(charges=(("NZ", 0),)))
    assert result.graph is not None and result.graph.atom("NZ").hydrogens == 2
    assert result.superseded_source.minimum_hydrogens == (("NZ", 3),)
    assert len(result.observed_hydrogens) == 3


def test_complete_neutral_his_request_does_not_append_an_extra_proton() -> None:
    source = _source("HIS", hydrogens=(("HE2", "H"),))
    result = _resolve(
        source, override=MicrostateConstraints(hydrogens=(("ND1", 1), ("NE2", 0)))
    )
    graph = result.graph
    assert graph is not None and graph.protonation_key()[0] == 0
    assert graph.bond_order("CE1", "NE2") == 2
    assert graph.bond_order("ND1", "CE1") == 1
    assert result.superseded_source.minimum_hydrogens == (("NE2", 1),)


def test_site_independence_and_original_scaffold_stereochemistry() -> None:
    source = _source("LYS", charges=(("N", 1), ("NZ", 0)))
    before = (source.constitution, source.topology, source.geometry, source.provenance)
    sidechain = _resolve(source)
    terminal = _resolve(
        source, kind=PolymerChemicalSite.BACKBONE_N, linkage=PeptideLinkage.FREE
    )
    assert sidechain.graph is not None and sidechain.graph.atom("NZ").charge == 0
    assert terminal.graph is not None and terminal.graph.atom("N").charge == 1
    assert (
        source.constitution,
        source.topology,
        source.geometry,
        source.provenance,
    ) == before
    assert all(atom.name != "CA" for atom in sidechain.graph.atoms)


@pytest.mark.parametrize("component", ("HIS", "ARG", "LYS"))
def test_no_catalog_for_unknown_backbone_linkage_or_nonstandard_component(
    component: str,
) -> None:
    assert (
        standard_microstate_candidates(component, PolymerChemicalSite.BACKBONE_N)
        is None
    )
    assert (
        standard_microstate_candidates(
            "HEM", PolymerChemicalSite.BACKBONE_N, PeptideLinkage.FREE
        )
        is None
    )


def test_graph_requires_connected_local_site() -> None:
    with pytest.raises(ValueError, match="connected"):
        MicrostateGraph(
            (MicrostateAtom("N1", "N", 0, 3, 0), MicrostateAtom("N2", "N", 0, 3, 0)), ()
        )


@pytest.mark.parametrize("reason", ("aromatic", "duplicate", "absent", "self_loop"))
def test_graph_rejects_invalid_bond_declarations(reason: str) -> None:
    graph = _site("ASP").candidates[0]
    if reason == "aromatic":
        bonds = (replace(graph.bonds[0], aromatic=True), *graph.bonds[1:])
    elif reason == "duplicate":
        bonds = (*graph.bonds, graph.bonds[0])
    elif reason == "absent":
        bonds = (*graph.bonds, BondDefinition("CG", "NO_ATOM"))
    else:
        bonds = (*graph.bonds, BondDefinition("CG", "CG"))
    with pytest.raises(ValueError, match="unique integral"):
        MicrostateGraph(graph.atoms, bonds)


def test_bond_constraints_are_canonical_under_endpoint_and_input_order() -> None:
    first = MicrostateConstraints(
        bonds=(BondDefinition("OD1", "CG", order=2), BondDefinition("OD2", "CG"))
    )
    second = MicrostateConstraints(
        bonds=(BondDefinition("CG", "OD2"), BondDefinition("CG", "OD1", order=2))
    )
    assert first == second and hash(first) == hash(second)
    assert _site("ASP").resolve(first) == _site("ASP").resolve(second)


@pytest.mark.parametrize(
    "relationship, expected",
    (
        (BondRelationshipType.COVALENT, MicrostateResolutionStatus.UNSUPPORTED),
        (BondRelationshipType.UNKNOWN, MicrostateResolutionStatus.UNSUPPORTED),
        (BondRelationshipType.HYDROGEN_BOND, MicrostateResolutionStatus.RESOLVED),
    ),
)
def test_h_from_another_residue_is_not_silently_ignored(
    relationship: BondRelationshipType,
    expected: MicrostateResolutionStatus,
) -> None:
    source = _source(
        "LYS",
        charges=(("NZ", 0),),
        extra_pdb=(
            "ATOM     99  H   ALA A   2       0.000   0.000   0.000"
            "  1.00 20.00           H\n"
        ),
    )
    source = _with_source_bonds(
        source, ("NZ", "H", relationship, None), second_residue_id=ResidueId("A", 2)
    )
    assert _resolve(source).status is expected


def test_linkage_does_not_create_redundant_sidechain_states() -> None:
    with pytest.raises(ValueError, match="side-chain"):
        standard_microstate_candidates(
            "LYS", PolymerChemicalSite.SIDECHAIN, PeptideLinkage.FREE
        )
    with pytest.raises(ValueError, match="side-chain"):
        PolymerMicrostateSite(
            build_standard_component_library().require("LYS"),
            PolymerChemicalSite.SIDECHAIN,
            PeptideLinkage.FREE,
        )


def test_resolved_graph_rejects_attachment_from_another_site() -> None:
    graph = _site("LYS").candidates[0]
    attachment = ObservedHydrogenAttachment(
        AtomRef(ResidueId("A", 1), "H"),
        AtomRef(ResidueId("A", 1), "N"),
        HydrogenAttachmentBasis.STANDARD_NAME,
    )
    with pytest.raises(ValueError, match="this site"):
        MicrostateResolution(
            MicrostateResolutionStatus.RESOLVED,
            (graph,),
            observed_hydrogens=(attachment,),
        )


@pytest.mark.parametrize("element", ("H", "D", "T"))
def test_named_heavy_boundary_atom_cannot_be_a_hydrogen_isotope(element: str) -> None:
    source = _source("LYS", charges=(("NZ", 0),))
    residue = source.constitution.residue_site_at(ResidueIndex(0))
    malformed = replace(
        residue,
        atom_sites=tuple(
            AtomSite(atom.name, element) if atom.name == "CE" else atom
            for atom in residue.atom_sites
        ),
    )
    site = PolymerMicrostateSite(
        build_standard_component_library().require("LYS"), PolymerChemicalSite.SIDECHAIN
    )
    assert (
        site.resolve(malformed, source.provenance.ingress.observation).status
        is MicrostateResolutionStatus.UNSUPPORTED
    )

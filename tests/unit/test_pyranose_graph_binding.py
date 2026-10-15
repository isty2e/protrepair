"""Independent named pyranoses through source-qualified glycosidic preparation."""

from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    residue_payload,
)
from tests.support.refinement_contract import build_continuous_relaxation_problem

from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.component.graph import (
    BondDefinition,
    ChemicalComponentDefinition,
)
from protrepair.chemistry.component.idealized import (
    IdealizedComponent,
    IdealizedComponentAtom,
)
from protrepair.chemistry.component.library import ComponentLibrary
from protrepair.chemistry.component.pyranose import PyranoseScaffold
from protrepair.chemistry.component.semantics import IdealGeometryHydrogenSemantics
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.retained_non_polymer.glycosidic import (
    GlycosidicTemplateBinding,
)
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.topology import TopologyAvailabilityState
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import AtomSite
from protrepair.structure.geometry import AtomGeometry
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
    resolve_retained_non_polymer_topology,
)
from protrepair.transformer.continuous.rdkit import build_rdkit_molecule
from protrepair.transformer.local import LocalScopeSpec

GLUCOSE = "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
DEOXY = "C[C@@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
CHARGED = "O=C([O-])[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O"
METHYL = "CO[C@H]1O[C@@H](CO)[C@H](O)[C@@H](O)[C@@H]1O"
DONOR = ResidueId("G", 1)
ACCEPTOR = ResidueId("G", 2)


def _template(
    smiles: str, component_id: str, *, reverse: bool = False
) -> tuple[ResidueTemplate, str, str, tuple[str, ...]]:
    precursor = Chem.MolFromSmiles(smiles)
    assert precursor is not None
    anomer_query = Chem.MolFromSmarts("[C;R;H1](-[O;!R])-[O;R]")
    assert anomer_query is not None
    anomer, leaving, _ = precursor.GetSubstructMatches(anomer_query)[0]
    molecule = Chem.AddHs(precursor)
    assert rdDistGeom.EmbedMolecule(molecule, randomSeed=79) == 0
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    names = tuple(
        f"Q{90 - a.GetIdx()}" if a.GetAtomicNum() != 1 else f"Y{a.GetIdx()}"
        for a in molecule.GetAtoms()
    )
    conformer = molecule.GetConformer()
    atoms = tuple(
        IdealizedComponentAtom(
            names[a.GetIdx()],
            a.GetSymbol().upper(),
            a.GetFormalCharge(),
            a.GetProp("_CIPCode") if a.HasProp("_CIPCode") else None,
            tuple(conformer.GetAtomPosition(a.GetIdx())),
        )
        for a in molecule.GetAtoms()
    )
    bonds = tuple(
        BondDefinition(
            names[b.GetBeginAtomIdx()],
            names[b.GetEndAtomIdx()],
            int(b.GetBondTypeAsDouble()),
        )
        for b in molecule.GetBonds()
    )
    hydroxyls = tuple(
        names[a.GetIdx()]
        for a in molecule.GetAtoms()
        if a.GetAtomicNum() == 8
        and not a.IsInRing()
        and a.GetIdx() != leaving
        and len(a.GetNeighbors()) == 2
        and any(n.GetAtomicNum() == 1 for n in a.GetNeighbors())
        and any(
            n.GetAtomicNum() == 6 and n.GetHybridization() == Chem.HybridizationType.SP3
            for n in a.GetNeighbors()
        )
    )
    component = IdealizedComponent(
        component_id,
        None,
        atoms[::-1] if reverse else atoms,
        bonds[::-1] if reverse else bonds,
    )
    template = ResidueTemplate(
        definition=ChemicalComponentDefinition(
            component_id,
            component.heavy_atom_names(),
            component.heavy_bonds(),
            {a.atom_name: a.formal_charge for a in component.heavy_atoms()},
        ),
        hydrogen_semantics=IdealGeometryHydrogenSemantics(component),
    )
    return template, names[anomer], names[leaving], hydroxyls


def _source(
    donor_smiles: str = GLUCOSE, acceptor_smiles: str = DEOXY, *, reverse: bool = False
) -> tuple[ProteinStructure, ComponentLibrary, str, str]:
    donor, carbon, leaving, _ = _template(donor_smiles, "DGR", reverse=reverse)
    acceptor, _, _, hydroxyls = _template(acceptor_smiles, "AGR", reverse=reverse)
    left = donor.hydrogen_semantics
    right = acceptor.hydrogen_semantics
    assert isinstance(left, IdealGeometryHydrogenSemantics) and isinstance(
        right, IdealGeometryHydrogenSemantics
    )
    oxygen = hydroxyls[-1]
    leave_atom = left.component.atom(leaving)
    accept_atom = right.component.atom(oxygen)
    assert leave_atom is not None and accept_atom is not None
    shift = np.asarray(leave_atom.ideal_position) - np.asarray(
        accept_atom.ideal_position
    )
    payloads = []
    for template, residue_id, translation, excluded in (
        (donor, DONOR, np.zeros(3), {leaving}),
        (acceptor, ACCEPTOR, shift, set()),
    ):
        semantics = template.hydrogen_semantics
        assert isinstance(semantics, IdealGeometryHydrogenSemantics)
        payloads.append(
            residue_payload(
                component_id=template.component_id,
                residue_id=residue_id,
                is_hetero=True,
                atoms=tuple(
                    atom_payload(
                        a.atom_name,
                        a.element,
                        Vec3.from_iterable(np.asarray(a.ideal_position) + translation),
                        occupancy=0.72,
                        b_factor=17.3,
                        formal_charge=a.formal_charge,
                    )
                    for a in semantics.component.heavy_atoms()
                    if a.atom_name not in excluded
                ),
            )
        )
    source = build_structure(
        chains=(), ligands=tuple(payloads), source_format=FileFormat.MMCIF
    )
    link = TopologyBond(
        source.constitution.atom_index(AtomRef(DONOR, carbon)),
        source.constitution.atom_index(AtomRef(ACCEPTOR, oxygen)),
        order=None,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
            source_id="independent-glycoside",
        ),
    )
    source = _with_bonds(source, (link,))
    source = read_structure_string(
        write_structure_string(source, FileFormat.MMCIF), FileFormat.MMCIF
    )
    library = (
        build_default_component_library().with_template(donor).with_template(acceptor)
    )
    return source, library, carbon, oxygen


def _with_bonds(
    source: ProteinStructure, bonds: tuple[TopologyBond, ...]
) -> ProteinStructure:
    return ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=bonds,
        ),
        provenance=source.provenance,
        polymer_blueprint=source.polymer_blueprint,
    )


def _assert_preserved(source: ProteinStructure, result: ProteinStructure) -> None:
    for index, _ in source.geometry.iter_positions():
        new = result.constitution.atom_index(source.constitution.atom_ref_at(index))
        assert result.constitution.atom_site_at(
            new
        ) == source.constitution.atom_site_at(index)
        assert result.geometry.atom_geometry(new) == source.geometry.atom_geometry(
            index
        )
        assert result.topology.formal_charge(new) == source.topology.formal_charge(
            index
        )
    for bond in source.topology.bonds:
        assert bond.provenance is BondProvenance.SOURCE_EXPLICIT
        refs = tuple(source.constitution.atom_ref_at(i) for i in bond.endpoint_pair())
        result_bond = result.topology.bond_between(
            *(result.constitution.atom_index(ref) for ref in refs)
        )
        assert (
            result_bond is not None
            and result_bond.source_metadata == bond.source_metadata
        )
        assert result_bond.provenance is bond.provenance


@pytest.mark.parametrize("donor_smiles", (GLUCOSE, DEOXY, CHARGED))
@pytest.mark.parametrize("acceptor_smiles", (DEOXY, METHYL))
@pytest.mark.parametrize("reverse", (False, True))
def test_independent_pyranoses_publish_coupled_inventory_and_native_graph(
    donor_smiles, acceptor_smiles, reverse
):
    source, library, carbon, oxygen = _source(
        donor_smiles, acceptor_smiles, reverse=reverse
    )
    result = add_retained_non_polymer_hydrogens(
        source, library, allow_retained_non_polymer_rdkit_fallback=False
    )
    assert not result.has_errors()
    assert all(
        issue.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
        for issue in result.issues
    )
    _assert_preserved(source, result.structure)
    for residue_id, lost_h in ((DONOR, 1), (ACCEPTOR, 1)):
        residue = result.structure.constitution.residue_or_ligand(residue_id)
        assert residue is not None
        template = library.require(residue.component_id)
        original = source.constitution.residue_or_ligand(residue_id)
        assert original is not None
        binding = GlycosidicTemplateBinding(source, original, template, library)
        assert (
            sum(a.is_hydrogen() for a in residue.atom_sites)
            == len(template.expected_hydrogen_atom_names()) - lost_h
        )
        assert set(binding.template.expected_heavy_atom_names()) == {
            a.name for a in residue.atom_sites if not a.is_hydrogen()
        }
    facts = (
        StructureChemistryObservation(result.structure, component_library=library)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts
    )
    assert all(
        f.heavy_atom_topology_availability_state is TopologyAvailabilityState.PRESENT
        and f.hydrogen_topology_availability_state is TopologyAvailabilityState.PRESENT
        for f in facts
    )
    problem = build_continuous_relaxation_problem(
        result.structure,
        LocalScopeSpec.from_atoms(
            tuple(
                result.structure.constitution.atom_ref_at(i)
                for i, _ in result.structure.geometry.iter_positions()
            )
        ),
        component_library=library,
    )
    molecule, mapping = build_rdkit_molecule(problem)
    Chem.SanitizeMol(molecule)
    assert all(a.GetNumRadicalElectrons() == 0 for a in molecule.GetAtoms())
    assert rdForceFieldHelpers.UFFHasAllMoleculeParams(molecule)
    assert rdForceFieldHelpers.MMFFHasAllMoleculeParams(molecule)
    native_link = molecule.GetBondBetweenAtoms(
        mapping[result.structure.constitution.atom_index(AtomRef(DONOR, carbon))],
        mapping[result.structure.constitution.atom_index(AtomRef(ACCEPTOR, oxygen))],
    )
    assert native_link is not None and native_link.GetBondType() == Chem.BondType.SINGLE
    repeated = add_retained_non_polymer_hydrogens(result.structure, library)
    assert (
        repeated.issues == result.issues
        and not repeated.repairs
        and repeated.structure == result.structure
    )
    for file_format in (FileFormat.PDB, FileFormat.MMCIF):
        recovered = read_structure_string(
            write_structure_string(result.structure, file_format), file_format
        )
        resolved = resolve_retained_non_polymer_topology(recovered, library)
        assert not resolved.has_errors()
        assert all(
            issue.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
            for issue in resolved.issues
        )
        chemistry = (
            StructureChemistryObservation(resolved.structure, component_library=library)
            .whole_structure_facts()[1]
            .retained_non_polymer_facts
        )
        assert all(
            f.hydrogen_topology_availability_state is TopologyAvailabilityState.PRESENT
            for f in chemistry
        )


def test_methylated_anomer_is_acceptor_only_not_a_donor():
    template, carbon, _, hydroxyls = _template(METHYL, "MET")
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    scaffold = PyranoseScaffold(semantics.component)
    with pytest.raises(ValueError, match="donor"):
        scaffold.require_donor(carbon)
    for name in hydroxyls:
        scaffold.require_acceptor(name)
    source, library, _, _ = _source(METHYL)
    result = add_retained_non_polymer_hydrogens(
        source, library, allow_retained_non_polymer_rdkit_fallback=False
    )
    assert result.issues and result.structure is source


def _binding(
    source: ProteinStructure, library: ComponentLibrary, residue_id: ResidueId = DONOR
) -> GlycosidicTemplateBinding:
    site = source.constitution.residue_or_ligand(residue_id)
    assert site is not None
    return GlycosidicTemplateBinding(
        source, site, library.require(site.component_id), library
    )


def _with_atom(
    source: ProteinStructure,
    residue_id: ResidueId,
    name: str,
    element: str,
    position: Vec3,
    charge: int = 0,
) -> ProteinStructure:
    site = source.constitution.residue_or_ligand(residue_id)
    assert site is not None
    geometry = source.residue_geometry(source.constitution.residue_index(residue_id))
    charges = tuple(
        (
            atom.name,
            source.topology.formal_charge(
                source.constitution.atom_index(AtomRef(residue_id, atom.name))
            ),
        )
        for atom in site.atom_sites
    )
    return source.with_updated_residue_facets(
        site.with_atom_site(AtomSite(name, element)),
        residue_geometry=geometry.with_atom_geometry(
            name, AtomGeometry(position, occupancy=0.63, b_factor=21.7)
        ),
        formal_charge_by_atom_name=(*charges, (name, charge)),
    )


def _redeposit(source: ProteinStructure) -> ProteinStructure:
    return read_structure_string(
        write_structure_string(source, FileFormat.MMCIF), FileFormat.MMCIF
    )


@pytest.mark.parametrize(
    "change", ("unknown", "double", "aromatic", "non_anomer", "ether", "two_donors")
)
def test_non_glycosidic_external_declarations_never_publish_a_partial_pair(change):
    source, library, _, _ = _source()
    link = source.topology.bonds[0]
    if change == "unknown":
        link = replace(
            link, relationship_type=BondRelationshipType.UNKNOWN, source_metadata=None
        )
    elif change == "double":
        link = replace(link, order=2, source_metadata=None)
    elif change == "aromatic":
        link = replace(link, aromatic=True, source_metadata=None)
    else:
        donor = library.require("DGR").hydrogen_semantics
        acceptor = library.require("AGR").hydrogen_semantics
        assert isinstance(donor, IdealGeometryHydrogenSemantics) and isinstance(
            acceptor, IdealGeometryHydrogenSemantics
        )
        if change == "non_anomer":
            carbon = next(
                a.atom_name
                for a in donor.component.heavy_atoms()
                if a.element == "C"
                and a.atom_name != PyranoseScaffold(donor.component).anomeric_carbon
            )
            link = replace(
                link,
                atom_index_1=source.constitution.atom_index(AtomRef(DONOR, carbon)),
            )
        else:
            roles = PyranoseScaffold(acceptor.component)
            endpoint = (
                roles.anomeric_carbon
                if change == "two_donors"
                else next(
                    a.atom_name
                    for a in acceptor.component.heavy_atoms()
                    if a.element == "O" and a.atom_name not in roles.hydroxyl_hydrogens
                )
            )
            link = replace(
                link,
                atom_index_2=source.constitution.atom_index(
                    AtomRef(ACCEPTOR, endpoint)
                ),
            )
    source = _with_bonds(source, (link,))
    result = add_retained_non_polymer_hydrogens(
        source, library, allow_retained_non_polymer_rdkit_fallback=False
    )
    assert result.issues and result.structure is source and not result.repairs


@pytest.mark.parametrize(
    "mutation",
    (
        "unknown_stereo",
        "wrong_stereo",
        "missing_ring_h",
        "duplicate_bond",
        "definition",
    ),
)
def test_selected_reference_contradictions_cannot_be_hidden_by_library_dispatch(
    mutation,
):
    source, library, carbon, _ = _source()
    template = library.require("DGR")
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    component = semantics.component
    if mutation == "definition":
        template = replace(
            template,
            definition=replace(
                template.definition, bonds=template.definition.bonds[:-1]
            ),
        )
    elif mutation == "duplicate_bond":
        component = replace(component, bonds=(*component.bonds, component.bonds[0]))
    else:
        if mutation == "missing_ring_h":
            hydrogen = next(
                a.atom_name
                for a in component.atoms
                if a.is_hydrogen()
                and carbon in component.bonded_atom_names(a.atom_name)
            )
            component = replace(
                component,
                atoms=tuple(a for a in component.atoms if a.atom_name != hydrogen),
                bonds=tuple(
                    b
                    for b in component.bonds
                    if hydrogen not in (b.atom_name_1, b.atom_name_2)
                ),
            )
        else:
            atom = component.atom(carbon)
            assert atom is not None and atom.stereo_config is not None
            stereo = (
                None
                if mutation == "unknown_stereo"
                else {"R": "S", "S": "R"}[atom.stereo_config]
            )
            component = replace(
                component,
                atoms=tuple(
                    replace(a, stereo_config=stereo) if a.atom_name == carbon else a
                    for a in component.atoms
                ),
            )
    if mutation != "definition":
        template = replace(
            template, hydrogen_semantics=IdealGeometryHydrogenSemantics(component)
        )
    site = source.constitution.residue_or_ligand(DONOR)
    assert site is not None
    try:
        binding = GlycosidicTemplateBinding(source, site, template, library)
    except ValueError:
        pass
    else:
        assert not binding.external_bonds
    result = add_retained_non_polymer_hydrogens(
        source,
        library.with_template(template),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert result.structure is source and result.issues and not result.repairs


def test_mirrored_source_ring_is_not_repaired_by_overwriting_observed_stereo():
    source, library, _, _ = _source()
    site = source.constitution.residue_or_ligand(DONOR)
    assert site is not None
    geometry = source.residue_geometry(source.constitution.residue_index(DONOR))
    for atom in site.atom_sites:
        stored = geometry.atom_geometry(atom.name)
        x, y, z = stored.position.to_array()
        geometry = geometry.with_atom_geometry(
            atom.name, replace(stored, position=Vec3(-x, y, z))
        )
    source = source.with_updated_residue_facets(site, residue_geometry=geometry)
    with pytest.raises(ValueError, match="stereo"):
        _binding(source, library)
    result = add_retained_non_polymer_hydrogens(
        source, library, allow_retained_non_polymer_rdkit_fallback=False
    )
    assert result.structure is source and not result.repairs


@pytest.mark.parametrize("isotope", ("H", "D", "T"))
@pytest.mark.parametrize("occupied", (False, True))
def test_original_isotopes_with_opaque_names_preserve_parent_identity(
    isotope, occupied
):
    source, library, carbon, oxygen = _source()
    residue_id = ACCEPTOR if occupied else DONOR
    template = library.require("AGR" if occupied else "DGR")
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    parent = oxygen if occupied else carbon
    name = next(
        a.atom_name
        for a in semantics.component.atoms
        if a.is_hydrogen()
        and parent in semantics.component.bonded_atom_names(a.atom_name)
    )
    ideal = semantics.component.atom(name)
    assert ideal is not None and ideal.ideal_position is not None
    position = Vec3.from_iterable(ideal.ideal_position)
    if occupied:
        position = Vec3.from_iterable(
            source.geometry.position(
                source.constitution.atom_index(AtomRef(residue_id, parent))
            ).to_array()
            + (0, 0, 1)
        )
    source = _with_atom(source, residue_id, name, isotope, position)
    source = _redeposit(source)
    result = add_retained_non_polymer_hydrogens(
        source, library, allow_retained_non_polymer_rdkit_fallback=False
    )
    if occupied:
        assert result.structure is source and result.issues and not result.repairs
    else:
        assert not result.issues
        _assert_preserved(source, result.structure)
        index = result.structure.constitution.atom_index(AtomRef(DONOR, name))
        assert result.structure.constitution.atom_site_at(index).element == isotope
        assert (
            result.structure.topology.bond_between(
                index, result.structure.constitution.atom_index(AtomRef(DONOR, carbon))
            )
            is not None
        )


@pytest.mark.parametrize("mode", ("generated", "charged", "original", "topology_only"))
def test_occupied_hydroxyl_cleanup_requires_generated_neutral_hydrogen(mode):
    source, library, _, oxygen = _source()
    template = library.require("AGR")
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    roles = PyranoseScaffold(semantics.component)
    name = roles.hydroxyl_hydrogens[oxygen]
    position = source.geometry.position(
        source.constitution.atom_index(AtomRef(ACCEPTOR, oxygen))
    )
    source = _with_atom(
        source,
        ACCEPTOR,
        name,
        "H",
        Vec3.from_iterable(position.to_array() + (0, 0, 1)),
        charge=int(mode == "charged"),
    )
    if mode == "original":
        source = _redeposit(source)
    operation = (
        resolve_retained_non_polymer_topology
        if mode == "topology_only"
        else add_retained_non_polymer_hydrogens
    )
    result = operation(source, library)
    if mode == "generated":
        assert not result.issues
        assert (
            result.structure.constitution.resolve_atom_site(AtomRef(ACCEPTOR, name))
            is None
        )
    else:
        assert result.structure is source and result.issues and not result.repairs


@pytest.mark.parametrize("hidden_fact", ("charge", "leaving_atom"))
def test_later_mutation_cannot_hide_original_precursor_contradictions(hidden_fact):
    source, library, carbon, _ = _source()
    template = library.require("DGR")
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    roles = PyranoseScaffold(semantics.component)
    atom_name = carbon if hidden_fact == "charge" else roles.exocyclic_oxygen
    atom = semantics.component.atom(atom_name)
    assert atom is not None and atom.ideal_position is not None
    source = _redeposit(
        _with_atom(
            source,
            DONOR,
            atom_name,
            atom.element,
            Vec3.from_iterable(atom.ideal_position),
            charge=int(hidden_fact == "charge"),
        )
    )
    if hidden_fact == "leaving_atom":
        source = source.without_atom_refs((AtomRef(DONOR, atom_name),))
    else:
        source = _with_atom(
            source,
            DONOR,
            atom_name,
            atom.element,
            Vec3.from_iterable(atom.ideal_position),
        )
    with pytest.raises(ValueError, match="original"):
        _binding(source, library)
    result = add_retained_non_polymer_hydrogens(
        source, library, allow_retained_non_polymer_rdkit_fallback=False
    )
    assert result.structure is source and result.issues and not result.repairs


def _branched_source() -> tuple[ProteinStructure, ComponentLibrary, ResidueId]:
    source, library, carbon, oxygen = _source()
    third = ResidueId("G", 3)
    donor = library.require("DGR").hydrogen_semantics
    acceptor = library.require("AGR").hydrogen_semantics
    assert isinstance(donor, IdealGeometryHydrogenSemantics) and isinstance(
        acceptor, IdealGeometryHydrogenSemantics
    )
    roles = PyranoseScaffold(donor.component)
    acceptor_roles = PyranoseScaffold(acceptor.component)
    second_oxygen = next(
        name
        for name in acceptor_roles.hydroxyl_hydrogens
        if name != oxygen and name != acceptor_roles.exocyclic_oxygen
    )
    leaving = donor.component.atom(roles.exocyclic_oxygen)
    assert leaving is not None
    shift = source.geometry.position(
        source.constitution.atom_index(AtomRef(ACCEPTOR, second_oxygen))
    ).to_array() - np.asarray(leaving.ideal_position)
    payloads = []
    for site in source.constitution.ligands:
        payloads.append(
            residue_payload(
                component_id=site.component_id,
                residue_id=site.residue_id,
                is_hetero=True,
                atoms=tuple(
                    atom_payload(
                        a.name,
                        a.element,
                        source.geometry.position(
                            source.constitution.atom_index(
                                AtomRef(site.residue_id, a.name)
                            )
                        ),
                        formal_charge=source.topology.formal_charge(
                            source.constitution.atom_index(
                                AtomRef(site.residue_id, a.name)
                            )
                        ),
                    )
                    for a in site.atom_sites
                ),
            )
        )
    payloads.append(
        residue_payload(
            component_id="DGR",
            residue_id=third,
            is_hetero=True,
            atoms=tuple(
                atom_payload(
                    a.atom_name,
                    a.element,
                    Vec3.from_iterable(np.asarray(a.ideal_position) + shift),
                    formal_charge=a.formal_charge,
                )
                for a in donor.component.heavy_atoms()
                if a.atom_name != roles.exocyclic_oxygen
            ),
        )
    )
    result = build_structure(
        chains=(), ligands=tuple(payloads), source_format=FileFormat.MMCIF
    )
    links = tuple(
        TopologyBond(
            result.constitution.atom_index(AtomRef(residue_id, carbon)),
            result.constitution.atom_index(AtomRef(ACCEPTOR, acceptor_oxygen)),
            order=None,
            provenance=BondProvenance.SOURCE_EXPLICIT,
            source_metadata=SourceBondMetadata(
                record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
                source_id=f"branch-{residue_id.seq_num}",
            ),
        )
        for residue_id, acceptor_oxygen in ((DONOR, oxygen), (third, second_oxygen))
    )
    return _redeposit(_with_bonds(result, links)), library, third


@pytest.mark.parametrize("missing_partner", (False, True))
def test_multiple_occupied_hydroxyls_are_admitted_as_one_connected_product(
    missing_partner,
):
    source, library, third = _branched_source()
    if missing_partner:
        site = source.constitution.residue_or_ligand(third)
        assert site is not None
        removable = next(a.name for a in site.atom_sites if a.element == "O")
        source = source.without_atom_refs((AtomRef(third, removable),))
    result = add_retained_non_polymer_hydrogens(
        source, library, allow_retained_non_polymer_rdkit_fallback=False
    )
    if missing_partner:
        assert result.structure is source and result.issues and not result.repairs
        return
    assert not result.issues
    _assert_preserved(source, result.structure)
    site = result.structure.constitution.residue_or_ligand(ACCEPTOR)
    assert site is not None
    assert (
        sum(a.is_hydrogen() for a in site.atom_sites)
        == len(library.require("AGR").expected_hydrogen_atom_names()) - 2
    )
    problem = build_continuous_relaxation_problem(
        result.structure,
        LocalScopeSpec.from_atoms(
            tuple(
                result.structure.constitution.atom_ref_at(i)
                for i, _ in result.structure.geometry.iter_positions()
            )
        ),
        component_library=library,
    )
    molecule, _ = build_rdkit_molecule(problem)
    Chem.SanitizeMol(molecule)
    assert all(a.GetNumRadicalElectrons() == 0 for a in molecule.GetAtoms())
    assert rdForceFieldHelpers.MMFFHasAllMoleculeParams(molecule)


def test_active_acceptor_template_does_not_require_a_duplicate_library_registration():
    source, library, _, _ = _source()
    site = source.constitution.residue_or_ligand(ACCEPTOR)
    assert site is not None
    partner_only = ComponentLibrary({"DGR": library.require("DGR")})
    binding = GlycosidicTemplateBinding(
        source, site, library.require("AGR"), partner_only
    )
    assert len(binding.external_bonds) == 1


@pytest.mark.parametrize(
    "smiles",
    (
        "OC[C@H]1O[C@@H](O)[C@H](O)[C@@H]1O",
        "O[C@H]1O[C@@H](O)[C@H](O)[C@@H](O)[C@@H]1O",
    ),
)
def test_furanose_and_ambiguous_anomeric_graphs_do_not_acquire_pyranose_roles(smiles):
    template, _, _, _ = _template(smiles, "OUT")
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    with pytest.raises(ValueError):
        PyranoseScaffold(semantics.component)


def test_two_pyranose_scaffolds_in_one_component_are_not_selected_by_atom_order():
    first, _, _, _ = _template(GLUCOSE, "DBL")
    semantics = first.hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    component = semantics.component
    combined = replace(
        component,
        atoms=(
            *component.atoms,
            *(replace(a, atom_name=f"{a.atom_name}B") for a in component.atoms),
        ),
        bonds=(
            *component.bonds,
            *(
                replace(
                    b, atom_name_1=f"{b.atom_name_1}B", atom_name_2=f"{b.atom_name_2}B"
                )
                for b in component.bonds
            ),
        ),
    )
    with pytest.raises(ValueError, match="unambiguous"):
        PyranoseScaffold(combined)


def test_four_heavy_neighbor_stereocenter_retains_its_mapped_orientation():
    branched = "OC[C@H]1O[C@@H](O)[C@H](O)[C@@](O)(C)[C@@H]1O"
    source, library, _, _ = _source(branched)
    result = add_retained_non_polymer_hydrogens(
        source, library, allow_retained_non_polymer_rdkit_fallback=False
    )
    assert not result.issues
    _assert_preserved(source, result.structure)
    semantics = library.require("DGR").hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    hydrogen_names = {a.atom_name for a in semantics.component.hydrogen_atoms()}
    centers = tuple(
        a
        for a in semantics.component.atoms
        if a.stereo_config is not None
        and semantics.component.bonded_atom_names(a.atom_name).isdisjoint(
            hydrogen_names
        )
    )
    assert len(centers) == 1
    _binding(result.structure, library).require_hydrogen_stereochemistry(
        result.structure.residue_geometry(
            result.structure.constitution.residue_index(DONOR)
        ),
        library.require("DGR").template_hydrogen_anchor_by_name(
            library.require("DGR").expected_hydrogen_atom_names()
        ),
    )


def test_close_unlinked_scaffolds_do_not_acquire_substitution_inventory():
    source, library, _, _ = _source()
    source = _with_bonds(source, ())
    binding = _binding(source, library)
    assert not binding.external_bonds and binding.template is library.require("DGR")


def test_two_partners_at_one_oxygen_are_not_merged_into_one_glycosidic_role():
    source, library, third = _branched_source()
    first, second = source.topology.bonds
    first_acceptor = next(
        source.constitution.atom_ref_at(i)
        for i in first.endpoint_pair()
        if source.constitution.atom_ref_at(i).residue_id == ACCEPTOR
    )
    second = replace(
        second,
        atom_index_1=source.constitution.atom_index(first_acceptor),
        atom_index_2=source.constitution.atom_index(
            AtomRef(
                third, _binding(source, library, third).external_bonds[0][0].atom_name
            )
        ),
    )
    source = _with_bonds(source, (first, second))
    result = add_retained_non_polymer_hydrogens(
        source, library, allow_retained_non_polymer_rdkit_fallback=False
    )
    assert result.structure is source and result.issues and not result.repairs


def test_remote_acid_oxygen_is_not_a_hydroxyl_acceptor():
    template, _, _, _ = _template(CHARGED, "ACD")
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    roles = PyranoseScaffold(semantics.component)
    acid_oxygen = next(
        a.atom_name for a in semantics.component.atoms if a.formal_charge == -1
    )
    with pytest.raises(ValueError, match="hydroxyl"):
        roles.require_acceptor(acid_oxygen)

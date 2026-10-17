"""Mapped independent acrylamides through coupled preparation and projection."""

from dataclasses import replace

import numpy as np
import pytest
from rdkit import Chem
from rdkit.Chem import rdDistGeom, rdForceFieldHelpers
from tests.support.cysteine_michael_adduct import (
    CYSTEINE,
    LIGAND,
    residue_payload,
)
from tests.support.cysteine_michael_adduct import (
    assert_original_preserved as _assert_original_preserved,
)
from tests.support.cysteine_michael_adduct import (
    qs7_source as _source,
)
from tests.support.cysteine_michael_adduct import (
    with_bonds as _with_bonds,
)
from tests.support.cysteine_michael_adduct import (
    with_thiol_hydrogen as _with_hydrogen,
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
from protrepair.chemistry.component.semantics import IdealGeometryHydrogenSemantics
from protrepair.chemistry.component.template import ResidueTemplate
from protrepair.chemistry.inference import idealized_stereochemistry
from protrepair.chemistry.microstate.context import PolymerMicrostateContext
from protrepair.chemistry.microstate.preparation import PolymerMicrostatePreparation
from protrepair.chemistry.restraint import structure as restraint_module
from protrepair.chemistry.restraint.library import RestraintLibrary
from protrepair.chemistry.restraint.structure import StructureRestraints
from protrepair.chemistry.restraint.template import (
    BondRestraintTarget,
    ResidueRestraintTemplate,
)
from protrepair.chemistry.retained_non_polymer.michael_adduct import (
    CysteineMichaelAdductBinding,
)
from protrepair.errors import RdkitUnavailableError
from protrepair.geometry import Vec3
from protrepair.io import (
    read_structure_string,
    to_rdkit_molecule,
    write_structure_string,
)
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.topology import TopologyAvailabilityState
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.constitution import AtomSite, ResidueSite
from protrepair.structure.geometry import AtomGeometry, ResidueGeometry
from protrepair.structure.labels import AtomRef
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    SourceBondMetadata,
    SourceBondRecordType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.hydrogen.core import materialize_hydrogens_core
from protrepair.transformer.completion.retained_non_polymer_hydrogen import repair
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)
from protrepair.transformer.continuous.rdkit import build_rdkit_molecule
from protrepair.transformer.local import LocalScopeSpec


def _example(
    smiles: str = "CCNC(=O)C=C", *, reverse: bool = False
) -> tuple[ProteinStructure, ComponentLibrary, str, str]:
    precursor = Chem.MolFromSmiles(smiles)
    assert precursor is not None
    query = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
    assert query is not None
    beta, alpha, *_ = precursor.GetSubstructMatches(query)[0]
    molecule = Chem.AddHs(precursor)
    assert rdDistGeom.EmbedMolecule(molecule, randomSeed=41) == 0
    Chem.AssignStereochemistry(molecule, cleanIt=True, force=True)
    names = tuple(
        f"X{50 - atom.GetIdx()}" if atom.GetAtomicNum() != 1 else f"Y{atom.GetIdx()}"
        for atom in molecule.GetAtoms()
    )
    conformer = molecule.GetConformer()
    atoms = tuple(
        IdealizedComponentAtom(
            names[atom.GetIdx()],
            atom.GetSymbol().upper(),
            atom.GetFormalCharge(),
            atom.GetProp("_CIPCode") if atom.HasProp("_CIPCode") else None,
            tuple(conformer.GetAtomPosition(atom.GetIdx())),
        )
        for atom in molecule.GetAtoms()
    )
    bonds = tuple(
        BondDefinition(
            names[b.GetBeginAtomIdx()],
            names[b.GetEndAtomIdx()],
            int(b.GetBondTypeAsDouble()),
        )
        for b in molecule.GetBonds()
    )
    component = IdealizedComponent(
        "XMA",
        None,
        atoms[::-1] if reverse else atoms,
        bonds[::-1] if reverse else bonds,
    )
    template = ResidueTemplate(
        definition=ChemicalComponentDefinition(
            "XMA",
            component.heavy_atom_names(),
            component.heavy_bonds(),
            {atom.atom_name: atom.formal_charge for atom in component.heavy_atoms()},
        ),
        hydrogen_semantics=IdealGeometryHydrogenSemantics(component),
    )
    source = _source(linked=False)
    cysteine = residue_payload(source, CYSTEINE)
    carbon = cysteine.position("C")
    cysteine = cysteine.with_atom_payload(
        AtomSite("OXT", "O"),
        atom_geometry=AtomGeometry(Vec3(carbon.x, carbon.y + 1.25, carbon.z)),
    )
    source = source.with_updated_residue_facets(
        cysteine.residue_site,
        residue_geometry=cysteine.residue_geometry,
        formal_charge_by_atom_name=cysteine.formal_charge_by_atom_name,
    )
    sulfur = source.geometry.position(
        source.constitution.atom_index(AtomRef(CYSTEINE, "SG"))
    )
    shift = (
        sulfur.to_array()
        + np.array([2.0, 0, 0])
        - np.asarray(atoms[beta].ideal_position)
    )
    residue = ResidueSite(
        "XMA",
        LIGAND,
        tuple(AtomSite(a.atom_name, a.element) for a in component.heavy_atoms()),
        True,
    )
    geometry = ResidueGeometry(
        {
            a.atom_name: AtomGeometry(
                Vec3(*(np.asarray(a.ideal_position) + shift)),
                occupancy=0.73,
                b_factor=21.4,
            )
            for a in component.heavy_atoms()
        }
    )
    source = source.with_updated_residue_facets(
        residue,
        residue_geometry=geometry,
        formal_charge_by_atom_name=tuple(
            (a.atom_name, a.formal_charge) for a in component.heavy_atoms()
        ),
    )
    link = TopologyBond(
        source.constitution.atom_index(AtomRef(LIGAND, names[beta])),
        source.constitution.atom_index(AtomRef(CYSTEINE, "SG")),
        order=None,
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
            source_id="independent-adduct",
        ),
    )
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(*source.topology.bonds, link),
        ),
        polymer_blueprint=source.polymer_blueprint,
        provenance=source.provenance,
    )
    # Re-ingress makes this independently mapped input, not an edited QS7 witness.
    source = read_structure_string(
        write_structure_string(source, FileFormat.MMCIF), FileFormat.MMCIF
    )
    return (
        source,
        build_default_component_library().with_template(template),
        names[alpha],
        names[beta],
    )


def _binding(
    source: ProteinStructure, library: ComponentLibrary
) -> CysteineMichaelAdductBinding:
    residue = source.constitution.residue_or_ligand(LIGAND)
    assert residue is not None
    return CysteineMichaelAdductBinding(
        source, residue, library.require("XMA"), library
    )


@pytest.mark.parametrize(
    "smiles",
    (
        "CCNC(=O)C=C",
        "C[N+](C)(C)CCNC(=O)C=C",
        "C[C@H](F)NC(=O)C=C",
        "C[C@@H](F)NC(=O)C=C",
        "C[C@](F)(Cl)NC(=O)C=C",
        "CN(C)C(=O)C=C",
        "C=CC(=O)NCCNC(=O)C=C",
    ),
)
@pytest.mark.parametrize("reverse", (False, True))
def test_independent_precursors_publish_coupled_product_and_repeat(smiles, reverse):
    source, library, alpha, beta = _example(smiles, reverse=reverse)
    result = add_retained_non_polymer_hydrogens(
        source, library, allow_retained_non_polymer_rdkit_fallback=False
    )
    assert not result.issues
    _assert_original_preserved(source, result.structure)
    binding = _binding(result.structure, library)
    assert binding.cysteine_is_realized()
    assert binding.external_bonds() == (
        (AtomRef(LIGAND, beta), AtomRef(CYSTEINE, "SG")),
    )
    assert len(binding.reaction_hydrogen_names()) == 4
    product = result.structure.constitution.residue_or_ligand(LIGAND)
    assert product is not None
    assert (
        sum(a.is_hydrogen() for a in product.atom_sites)
        == len(library.require("XMA").expected_hydrogen_atom_names()) + 1
    )
    prep = PolymerMicrostatePreparation(
        PolymerMicrostateContext(result.structure), library
    )
    assert {parent for _, parent in prep.fixed_hydrogen_atom_sites(CYSTEINE)} == {
        "CA",
        "CB",
    }
    observation = StructureChemistryObservation(
        result.structure, component_library=library
    )
    facts = observation.whole_structure_facts()[1].retained_non_polymer_facts
    fact = next(f for f in facts if f.residue_id == LIGAND)
    assert (
        fact.heavy_atom_topology_availability_state is TopologyAvailabilityState.PRESENT
    )
    assert (
        fact.hydrogen_topology_availability_state is TopologyAvailabilityState.PRESENT
    )
    repeated = add_retained_non_polymer_hydrogens(result.structure, library)
    assert not repeated.issues and not repeated.repairs
    assert repeated.structure == result.structure
    for file_format in (FileFormat.PDB, FileFormat.MMCIF):
        recovered = read_structure_string(
            write_structure_string(result.structure, file_format), file_format
        )
        assert _binding(recovered, library).cysteine_is_realized() is (
            file_format is FileFormat.MMCIF
        )
        resolved = repair.resolve_retained_non_polymer_topology(recovered, library)
        assert not resolved.issues
        assert _binding(resolved.structure, library).cysteine_is_realized()
    bond = result.structure.topology.bond_between(
        *(
            result.structure.constitution.atom_index(AtomRef(LIGAND, name))
            for name in (alpha, beta)
        )
    )
    assert bond is not None and bond.order == 1
    for precursor_bond in library.require("XMA").definition.bonds:
        expected = (
            1
            if {precursor_bond.atom_name_1, precursor_bond.atom_name_2} == {alpha, beta}
            else precursor_bond.order
        )
        actual = result.structure.topology.bond_between(
            result.structure.constitution.atom_index(
                AtomRef(LIGAND, precursor_bond.atom_name_1)
            ),
            result.structure.constitution.atom_index(
                AtomRef(LIGAND, precursor_bond.atom_name_2)
            ),
        )
        assert actual is not None and actual.order == expected


@pytest.mark.parametrize(
    "mutation",
    ("saturated", "ester", "charged", "extra_bond", "missing_h", "disconnected"),
)
def test_graph_controls_do_not_admit_a_product(mutation):
    source, library, alpha, beta = _example()
    template = library.require("XMA")
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    component = semantics.component
    if mutation == "saturated":
        component = replace(
            component,
            bonds=tuple(
                replace(b, order=1)
                if {b.atom_name_1, b.atom_name_2} == {alpha, beta}
                else b
                for b in component.bonds
            ),
        )
    elif mutation == "charged":
        component = replace(
            component,
            atoms=tuple(
                replace(a, formal_charge=1) if a.atom_name == beta else a
                for a in component.atoms
            ),
        )
    elif mutation == "ester":
        component = replace(
            component,
            atoms=tuple(
                replace(a, element="O") if a.element == "N" else a
                for a in component.atoms
            ),
        )
    elif mutation == "extra_bond":
        oxygen = next(a.atom_name for a in component.atoms if a.element == "O")
        component = replace(
            component, bonds=(*component.bonds, BondDefinition(beta, oxygen))
        )
    elif mutation == "missing_h":
        hydrogen = next(
            a.atom_name
            for a in component.hydrogen_atoms()
            if component.hydrogen_anchor_atom_name(a.atom_name) == beta
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
        component = replace(
            component, atoms=(*component.atoms, IdealizedComponentAtom("ZZ", "HE", 0))
        )
    template = replace(
        template,
        definition=replace(
            template.definition,
            atom_names=component.heavy_atom_names(),
            bonds=component.heavy_bonds(),
            formal_charges={
                a.atom_name: a.formal_charge for a in component.heavy_atoms()
            },
        ),
        hydrogen_semantics=IdealGeometryHydrogenSemantics(component),
    )
    with pytest.raises(ValueError):
        _binding(source, library.with_template(template))


def test_custom_restraints_remain_distinct_from_precursor_projection():
    source, library, alpha, beta = _example()
    product = add_retained_non_polymer_hydrogens(source, library).structure
    dictionary = ResidueRestraintTemplate(
        "XMA",
        bond_targets=(BondRestraintTarget(alpha, beta, 1.34, 0.02),),
        angle_targets=(),
    )
    assert _binding(source, library).applicable_restraints(dictionary) == dictionary
    assert not _binding(product, library).applicable_restraints(dictionary).bond_targets
    assert (
        StructureRestraints(
            product, library, RestraintLibrary({"XMA": dictionary})
        ).for_residue(LIGAND)
        == dictionary
    )


def test_reference_cache_does_not_hide_backend_unavailability(monkeypatch):
    source, library, *_ = _example()
    _binding(source, library)
    monkeypatch.setattr(idealized_stereochemistry, "Chem", None)
    with pytest.raises(RdkitUnavailableError):
        _binding(source, library)


@pytest.mark.parametrize(
    "mutation", ("unknown", "metal", "alpha", "multiple", "absent", "double")
)
def test_deposited_attachment_is_required_and_must_identify_the_beta_carbon(mutation):
    source, library, alpha, beta = _example()
    link = next(
        b
        for b in source.topology.bonds
        if len(
            {source.constitution.atom_ref_at(i).residue_id for i in b.endpoint_pair()}
        )
        == 2
    )
    bonds = tuple(b for b in source.topology.bonds if b is not link)
    if mutation == "unknown":
        bonds += (replace(link, relationship_type=BondRelationshipType.UNKNOWN),)
    elif mutation == "metal":
        bonds += (
            replace(link, relationship_type=BondRelationshipType.METAL_COORDINATION),
        )
    elif mutation == "double":
        bonds += (replace(link, order=2),)
    elif mutation == "alpha":
        bonds += (
            replace(
                link,
                atom_index_2=source.constitution.atom_index(AtomRef(LIGAND, alpha)),
            ),
        )
    elif mutation == "multiple":
        bonds += (
            link,
            TopologyBond(
                source.constitution.atom_index(AtomRef(LIGAND, alpha)),
                source.constitution.atom_index(AtomRef(CYSTEINE, "CB")),
                provenance=BondProvenance.SOURCE_EXPLICIT,
            ),
        )
    changed = _with_bonds(source, bonds)
    with pytest.raises(ValueError):
        _binding(changed, library)
    result = add_retained_non_polymer_hydrogens(
        changed, library, allow_retained_non_polymer_rdkit_fallback=False
    )
    if mutation not in {"metal", "absent"}:
        assert result.issues and result.structure is changed
    else:
        assert not result.issues
        alkene = result.structure.topology.bond_between(
            *(
                result.structure.constitution.atom_index(AtomRef(LIGAND, n))
                for n in (alpha, beta)
            )
        )
        assert alkene is not None and alkene.order == 2


@pytest.mark.parametrize("element", ("H", "D", "T"))
@pytest.mark.parametrize("source_name", ("IS1", "H1"))
def test_independent_source_hydrogen_names_and_isotopes_survive(element, source_name):
    source, library, alpha, beta = _example()
    binding = _binding(source, library)
    positions = binding.reaction_hydrogen_positions(
        source.residue_geometry(source.constitution.residue_index(LIGAND))
    )
    hydrogen = positions[-1][1]
    payload = residue_payload(source, LIGAND).with_atom_payload(
        AtomSite(source_name, element),
        atom_geometry=AtomGeometry(hydrogen, occupancy=0.47, b_factor=18.2),
        formal_charge=0,
    )
    source = source.with_updated_residue_facets(
        payload.residue_site,
        residue_geometry=payload.residue_geometry,
        formal_charge_by_atom_name=payload.formal_charge_by_atom_name,
    )
    bond = TopologyBond(
        atom_index_1=source.constitution.atom_index(AtomRef(LIGAND, beta)),
        atom_index_2=source.constitution.atom_index(AtomRef(LIGAND, source_name)),
        provenance=BondProvenance.SOURCE_EXPLICIT,
        source_metadata=SourceBondMetadata(
            record_type=SourceBondRecordType.MMCIF_STRUCT_CONN,
            source_id="isotope-parent",
        ),
    )
    source = _with_bonds(source, (*source.topology.bonds, bond))
    source = read_structure_string(
        write_structure_string(source, FileFormat.MMCIF), FileFormat.MMCIF
    )
    result = add_retained_non_polymer_hydrogens(
        source, library, allow_retained_non_polymer_rdkit_fallback=False
    )
    assert not result.issues
    _assert_original_preserved(source, result.structure)
    assert _binding(result.structure, library).cysteine_is_realized()
    atom = result.structure.constitution.resolve_atom_site(AtomRef(LIGAND, source_name))
    assert atom is not None and atom.element == element
    parents = result.structure.topology.residue_hydrogen_anchor_by_name(
        constitution=result.structure.constitution,
        residue_index=result.structure.constitution.residue_index(LIGAND),
    )
    assert sum(parent == alpha for parent in parents.values()) == 2
    assert sum(parent == beta for parent in parents.values()) == 2
    assert parents[source_name] == beta
    observation = StructureChemistryObservation(
        result.structure, component_library=library
    )
    fact = next(
        f
        for f in observation.whole_structure_facts()[1].retained_non_polymer_facts
        if f.residue_id == LIGAND
    )
    assert (
        fact.hydrogen_topology_availability_state is TopologyAvailabilityState.PRESENT
    )
    prepared = materialize_hydrogens_core(result.structure, library)
    assert not prepared.issues
    structure = prepared.structure
    problem = build_continuous_relaxation_problem(
        structure,
        LocalScopeSpec.from_atoms(
            tuple(
                structure.constitution.atom_ref_at(i)
                for i, _ in structure.geometry.iter_positions()
            )
        ),
        component_library=library,
    )
    molecule, mapping = build_rdkit_molecule(problem)
    hydrogen = molecule.GetAtomWithIdx(
        mapping[structure.constitution.atom_index(AtomRef(LIGAND, source_name))]
    )
    assert hydrogen.GetIsotope() == {"H": 0, "D": 2, "T": 3}[element]
    assert tuple(a.GetIdx() for a in hydrogen.GetNeighbors()) == (
        mapping[structure.constitution.atom_index(AtomRef(LIGAND, beta))],
    )


@pytest.mark.parametrize("element", ("H", "D", "T"))
@pytest.mark.parametrize("original", (False, True))
def test_thiol_isotope_removal_requires_generated_custody(element, original):
    source, library, *_ = _example()
    source = _with_hydrogen(source, original=original, element=element)
    result = add_retained_non_polymer_hydrogens(
        source, library, allow_retained_non_polymer_rdkit_fallback=False
    )
    if original:
        assert result.issues and result.structure is source
    else:
        assert not result.issues
        assert not result.structure.constitution.residue_site_at(
            result.structure.constitution.residue_index(CYSTEINE)
        ).has_atom_site("HG")
        assert _binding(result.structure, library).cysteine_is_realized()


@pytest.mark.parametrize(
    "mutation",
    (
        "reflected",
        "wrong_label",
        "missing_reference",
        "nonfinite_reference",
        "explicit_precursor_order",
        "original_charge",
    ),
)
def test_context_and_reference_contradictions_are_not_overridden(mutation):
    source, library, alpha, beta = _example("C[C@H](F)NC(=O)C=C")
    template = library.require("XMA")
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    component = semantics.component
    center = next(a for a in component.atoms if a.stereo_config)
    if mutation == "reflected":
        payload = residue_payload(source, LIGAND)
        geometry = ResidueGeometry(
            {
                n: replace(g, position=Vec3(-g.position.x, g.position.y, g.position.z))
                for n, g in payload.residue_geometry.atoms_by_name.items()
            }
        )
        source = source.with_updated_residue_facets(
            payload.residue_site,
            residue_geometry=geometry,
            formal_charge_by_atom_name=payload.formal_charge_by_atom_name,
        )
    elif mutation == "explicit_precursor_order":
        bond = TopologyBond(
            *(
                source.constitution.atom_index(AtomRef(LIGAND, n))
                for n in (alpha, beta)
            ),
            order=2,
            provenance=BondProvenance.SOURCE_EXPLICIT,
        )
        source = _with_bonds(source, (*source.topology.bonds, bond))
    elif mutation == "original_charge":
        payload = residue_payload(source, LIGAND)
        source = source.with_updated_residue_facets(
            payload.residue_site,
            residue_geometry=payload.residue_geometry,
            formal_charge_by_atom_name=((alpha, 1),),
        )
        source = read_structure_string(
            write_structure_string(source, FileFormat.MMCIF), FileFormat.MMCIF
        )
        payload = residue_payload(source, LIGAND)
        source = source.with_updated_residue_facets(
            payload.residue_site,
            residue_geometry=payload.residue_geometry,
            formal_charge_by_atom_name=((alpha, 0),),
        )
    else:
        changed = (
            replace(center, stereo_config="S" if center.stereo_config == "R" else "R")
            if mutation == "wrong_label"
            else replace(
                center,
                ideal_position=None
                if mutation == "missing_reference"
                else (float("nan"), 0, 0),
            )
        )
        component = replace(
            component,
            atoms=tuple(
                changed if a.atom_name == center.atom_name else a
                for a in component.atoms
            ),
        )
        library = library.with_template(
            replace(
                template, hydrogen_semantics=IdealGeometryHydrogenSemantics(component)
            )
        )
    result = add_retained_non_polymer_hydrogens(
        source, library, allow_retained_non_polymer_rdkit_fallback=False
    )
    assert result.issues and result.structure is source


def test_bundled_precursor_dictionary_projection_is_not_component_named(monkeypatch):
    source, library, alpha, beta = _example()
    dictionary = ResidueRestraintTemplate(
        "XMA",
        bond_targets=(BondRestraintTarget(alpha, beta, 1.34, 0.02),),
        angle_targets=(),
    )
    restraints = RestraintLibrary({"XMA": dictionary})
    monkeypatch.setattr(
        restraint_module,
        "build_bundled_retained_non_polymer_restraint_library",
        lambda: restraints,
    )
    assert (
        StructureRestraints(source, library, restraints).for_residue(LIGAND)
        == dictionary
    )
    result = add_retained_non_polymer_hydrogens(source, library)
    assert not result.issues
    projected = StructureRestraints(result.structure, library, restraints).for_residue(
        LIGAND
    )
    assert projected is not None and not projected.bond_targets


def test_reference_graph_rejects_disconnected_salt_and_radical_inventory():
    _, library, *_ = _example()
    semantics = library.require("XMA").hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    component = semantics.component
    with pytest.raises(ValueError, match="closed-shell"):
        idealized_stereochemistry.require_compatible_component_reference(
            replace(
                component,
                atoms=(*component.atoms, IdealizedComponentAtom("ZZ", "HE", 0)),
            )
        )
    h = component.hydrogen_atoms()[0].atom_name
    with pytest.raises(ValueError, match="closed-shell"):
        idealized_stereochemistry.require_compatible_component_reference(
            replace(
                component,
                atoms=tuple(a for a in component.atoms if a.atom_name != h),
                bonds=tuple(
                    b
                    for b in component.bonds
                    if h not in (b.atom_name_1, b.atom_name_2)
                ),
            )
        )


@pytest.mark.parametrize(
    "smiles", ("CCNC(=O)C=C", "C[N+](C)(C)CCNC(=O)C=C", "C[C@H](F)NC(=O)C=C")
)
def test_independent_product_uses_the_common_native_graph_projection(smiles):
    source, library, alpha, beta = _example(smiles)
    retained = add_retained_non_polymer_hydrogens(source, library)
    assert not retained.issues
    prepared = materialize_hydrogens_core(retained.structure, library)
    assert not prepared.issues
    structure = prepared.structure
    native_export = to_rdkit_molecule(structure)
    assert native_export.GetNumAtoms() == len(structure.constitution.atom_slots)
    assert all(atom.GetNumRadicalElectrons() == 0 for atom in native_export.GetAtoms())
    problem = build_continuous_relaxation_problem(
        structure,
        LocalScopeSpec.from_atoms(
            tuple(
                structure.constitution.atom_ref_at(i)
                for i, _ in structure.geometry.iter_positions()
            )
        ),
        component_library=library,
    )
    molecule, mapping = build_rdkit_molecule(problem)
    Chem.SanitizeMol(molecule)
    assert len(mapping) == sum(1 for _ in structure.geometry.iter_positions())
    assert all(atom.GetNumRadicalElectrons() == 0 for atom in molecule.GetAtoms())
    assert rdForceFieldHelpers.UFFHasAllMoleculeParams(molecule)
    assert rdForceFieldHelpers.MMFFHasAllMoleculeParams(molecule)
    for first, second in (
        (AtomRef(LIGAND, alpha), AtomRef(LIGAND, beta)),
        (AtomRef(LIGAND, beta), AtomRef(CYSTEINE, "SG")),
    ):
        native = molecule.GetBondBetweenAtoms(
            *(
                mapping[structure.constitution.atom_index(ref)]
                for ref in (first, second)
            )
        )
        assert native is not None and native.GetBondType() == Chem.BondType.SINGLE


@pytest.mark.parametrize("reverse_endpoints", (False, True))
@pytest.mark.parametrize("reverse_storage", (False, True))
def test_equivalent_precursor_bond_storage_preserves_product(
    reverse_endpoints, reverse_storage
):
    source, library, *_ = _example()
    expected = add_retained_non_polymer_hydrogens(source, library)
    assert not expected.issues
    template = library.require("XMA")
    bonds = tuple(
        replace(bond, atom_name_1=bond.atom_name_2, atom_name_2=bond.atom_name_1)
        if reverse_endpoints
        else bond
        for bond in template.definition.bonds
    )
    library = library.with_template(
        replace(
            template,
            definition=replace(
                template.definition, bonds=bonds[::-1] if reverse_storage else bonds
            ),
        )
    )
    actual = add_retained_non_polymer_hydrogens(source, library)
    assert not actual.issues
    assert actual.structure == expected.structure


def test_product_pose_change_invalidates_polymer_selection():
    source, library, *_ = _example("C[C@H](F)NC(=O)C=C")
    realized = add_retained_non_polymer_hydrogens(source, library)
    assert not realized.issues
    structure = realized.structure
    preparation = PolymerMicrostatePreparation(
        PolymerMicrostateContext(structure), library
    )
    context = preparation.fixed_hydrogen_context(CYSTEINE)
    assert context is not None and not context.unresolved_parent_names
    payload = residue_payload(structure, LIGAND)
    reflected = ResidueGeometry(
        {
            name: replace(g, position=Vec3(-g.position.x, g.position.y, g.position.z))
            for name, g in payload.residue_geometry.atoms_by_name.items()
        }
    )
    changed = structure.with_updated_residue_geometries(((LIGAND, reflected),))
    assert not preparation.matches_chemistry(changed)
    rebound = PolymerMicrostatePreparation(PolymerMicrostateContext(changed), library)
    context = rebound.fixed_hydrogen_context(CYSTEINE)
    assert context is not None and context.unresolved_parent_names == {"SG"}
    rejected = add_retained_non_polymer_hydrogens(changed, library)
    assert rejected.issues and rejected.structure is changed

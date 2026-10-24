"""Dictionary H inventories are independent of polymer-only anchor exclusion."""

from dataclasses import replace
from pathlib import Path

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_bonded_structure,
    chain_payload,
    residue_payload,
)
from tests.unit.test_retained_component_scaffolds import component_structure

from protrepair.api import process_structure
from protrepair.chemistry import (
    ComponentLibrary,
    IdealGeometryHydrogenSemantics,
    build_default_component_library,
)
from protrepair.chemistry.nonstandard.registry import build_bundled_nonstandard_registry
from protrepair.chemistry.retained_non_polymer.registry import (
    build_bundled_retained_non_polymer_registry,
)
from protrepair.diagnostics.clashes import ClashPolicy, detect_clashes
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, to_rdkit_molecule
from protrepair.scope import WholeStructureScope
from protrepair.state import HydrogenCoverageState
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.topology import TopologyAvailabilityState
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import AtomRef, ResidueId
from protrepair.structure.observation import StructureObservation
from protrepair.structure.provenance import FileFormat
from protrepair.structure.topology import (
    BondProvenance,
    BondRelationshipType,
    StructureTopology,
    TopologyBond,
)
from protrepair.transformer.completion.hydrogen import add_hydrogens
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
)
from protrepair.workflow.contracts import (
    LigandPolicy,
    StructureIngressOptions,
    requested_process_goal,
)


@pytest.mark.parametrize("component_id,count", (("SAH", 20), ("SAM", 22)))
def test_retained_full_h_inventory_is_realized_and_exports_closed_shell(
    component_id: str, count: int
) -> None:
    source = component_structure(component_id)
    result = add_retained_non_polymer_hydrogens(
        source, allow_retained_non_polymer_rdkit_fallback=False
    )
    assert not result.issues
    residue = result.structure.constitution.ligands[0]
    assert sum(atom.is_hydrogen() for atom in residue.atom_sites) == count
    parents = result.structure.topology.residue_hydrogen_anchor_by_name(
        constitution=result.structure.constitution,
        residue_index=result.structure.constitution.residue_index(residue.residue_id),
    )
    assert {name for name, parent in parents.items() if parent == "N"} == {"HN1", "HN2"}
    assert (parents.get("HXT") == "OXT") is (component_id == "SAH")
    molecule = to_rdkit_molecule(result.structure)
    assert not any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
    fact = (
        StructureChemistryObservation(result.structure)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert (
        fact.hydrogen_topology_availability_state is TopologyAvailabilityState.PRESENT
    )
    for index, _ in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(index)
        assert result.structure.geometry.atom_geometry(
            result.structure.constitution.atom_index(ref)
        ) == source.geometry.atom_geometry(index)


@pytest.mark.parametrize("component_id,omitted", (("SAH", "OXT"), ("SAM", "CE")))
def test_missing_remote_heavy_atom_does_not_remove_available_amine_h(
    component_id: str, omitted: str
) -> None:
    source = component_structure(component_id, omitted=(omitted,))
    result = add_retained_non_polymer_hydrogens(source)
    residue = result.structure.constitution.ligands[0]
    assert {"HN1", "HN2"} <= set(residue.atom_site_names())
    if omitted == "OXT":
        assert not residue.has_atom_site("HXT")
    assert not residue.has_atom_site(omitted)


@pytest.mark.parametrize("element", ("H", "D", "T"))
def test_source_amine_hydrogen_isotopes_and_scalars_are_preserved(element: str) -> None:
    record = build_bundled_retained_non_polymer_registry().get("SAH")
    assert record is not None
    atoms = []
    for atom in record.atoms:
        if atom.is_hydrogen() and atom.atom_name != "HN1":
            continue
        assert atom.ideal_position is not None
        atoms.append(
            atom_payload(
                atom.atom_name,
                element if atom.atom_name == "HN1" else atom.element,
                Vec3.coerce(atom.ideal_position),
                occupancy=0.37,
                b_factor=11.5,
            )
        )
    source = build_bonded_structure(
        chains=(),
        source_format=FileFormat.MMCIF,
        ligands=(
            residue_payload(
                component_id="SAH", residue_id=ResidueId("L", 1), atoms=tuple(atoms)
            ),
        ),
    )
    observation = StructureObservation.from_source_facets(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=source.topology,
    )
    source = source.with_provenance(
        replace(
            source.provenance,
            ingress=replace(source.provenance.ingress, observation=observation),
        )
    )
    result = add_retained_non_polymer_hydrogens(source)
    ref = AtomRef(ResidueId("L", 1), "HN1")
    before, after = (
        source.constitution.atom_index(ref),
        result.structure.constitution.atom_index(ref),
    )
    assert result.structure.constitution.atom_site_at(after).element == element
    assert result.structure.geometry.atom_geometry(
        after
    ) == source.geometry.atom_geometry(before)
    assert result.structure.constitution.ligands[0].has_atom_site("HN2")
    assert result.structure.provenance.ingress.observation is observation


@pytest.mark.parametrize(
    "relationship", (BondRelationshipType.COVALENT, BondRelationshipType.UNKNOWN)
)
def test_unexplained_retained_n_link_does_not_fall_back_to_isolated_amine(
    relationship: BondRelationshipType,
) -> None:
    partner = residue_payload(
        component_id="XXX",
        residue_id=ResidueId("L", 2),
        atoms=(atom_payload("C1", "C", Vec3(30, 0, 0)),),
    )
    record = build_bundled_retained_non_polymer_registry().get("SAH")
    assert record is not None
    source = build_bonded_structure(
        chains=(),
        source_format=FileFormat.MMCIF,
        ligands=(
            residue_payload(
                component_id="SAH",
                residue_id=ResidueId("L", 1),
                atoms=tuple(
                    atom_payload(a.atom_name, a.element, Vec3.coerce(a.ideal_position))
                    for a in record.heavy_atoms()
                    if a.ideal_position is not None
                ),
            ),
            partner,
        ),
    )
    source = ProteinStructure.from_payload(
        constitution=source.constitution,
        geometry=source.geometry,
        topology=StructureTopology(
            constitution=source.constitution,
            atom_topologies=source.topology.atom_topologies,
            bonds=(
                *source.topology.bonds,
                TopologyBond(
                    source.constitution.atom_index(AtomRef(ResidueId("L", 1), "N")),
                    source.constitution.atom_index(AtomRef(ResidueId("L", 2), "C1")),
                    provenance=BondProvenance.SOURCE_EXPLICIT,
                    relationship_type=relationship,
                ),
            ),
        ),
        provenance=source.provenance,
    )
    result = add_retained_non_polymer_hydrogens(
        source, target_residue_ids=frozenset((ResidueId("L", 1),))
    )
    assert result.structure is source
    assert result.issues and not result.repairs
    assert not result.structure.constitution.ligands[0].has_atom_site("HN1")


@pytest.mark.parametrize("component_id", ("SAH", "SAM"))
def test_conflicting_source_amine_charge_is_not_overwritten(component_id: str) -> None:
    source = component_structure(component_id, charges=(("N", 1),))
    result = add_retained_non_polymer_hydrogens(source)
    assert result.structure is source
    assert result.issues and not result.repairs
    assert (
        source.topology.formal_charge(
            source.constitution.atom_index(AtomRef(ResidueId("L", 1), "N"))
        )
        == 1
    )


@pytest.mark.parametrize("retained", (False, True))
def test_same_modified_component_keeps_role_specific_terminal_chemistry(
    retained: bool,
) -> None:
    record = build_bundled_nonstandard_registry().get("HYP")
    assert record is not None
    payload = residue_payload(
        component_id="HYP",
        residue_id=ResidueId("A", 1),
        atoms=tuple(
            atom_payload(a.atom_name, a.element, Vec3.coerce(a.ideal_position))
            for a in record.heavy_atoms()
            if a.ideal_position is not None
        ),
    )
    source = build_bonded_structure(
        chains=() if retained else (chain_payload("A", (payload,)),),
        ligands=(payload,) if retained else (),
        source_format=FileFormat.MMCIF,
    )
    execute = add_retained_non_polymer_hydrogens if retained else add_hydrogens
    result = execute(source)
    parents = result.structure.topology.residue_hydrogen_anchor_by_name(
        constitution=result.structure.constitution,
        residue_index=result.structure.constitution.residue_index(ResidueId("A", 1)),
    )
    assert sum(parent == "N" for parent in parents.values()) == (1 if retained else 2)
    assert sum(parent == "OXT" for parent in parents.values()) == int(retained)
    assert not result.issues


@pytest.mark.parametrize("component_id", ("SAH", "SAM"))
def test_public_h_workflow_prepares_retained_component_for_native_export(
    component_id: str,
) -> None:
    result = process_structure(
        component_structure(component_id),
        ingress=StructureIngressOptions(ligand_policy=LigandPolicy.KEEP),
        requested_goals=(
            requested_process_goal(
                scope=WholeStructureScope(), value=HydrogenCoverageState.COMPLETE
            ),
        ),
    )
    molecule = to_rdkit_molecule(result.structure)
    assert not any(atom.GetNumRadicalElectrons() for atom in molecule.GetAtoms())
    assert not result.relaxation_executions


def test_custom_retained_template_keeps_its_n_and_oxt_h_inventory() -> None:
    template = build_default_component_library().require("SAH")
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    component = replace(semantics.component, component_id="ZZQ")
    custom = replace(
        template,
        definition=replace(template.definition, component_id="ZZQ"),
        hydrogen_semantics=IdealGeometryHydrogenSemantics(component),
    )
    payload = residue_payload(
        component_id="ZZQ",
        residue_id=ResidueId("L", 1),
        atoms=tuple(
            atom_payload(atom.atom_name, atom.element, Vec3.coerce(atom.ideal_position))
            for atom in component.heavy_atoms()
            if atom.ideal_position is not None
        ),
    )
    source = build_bonded_structure(
        chains=(), ligands=(payload,), source_format=FileFormat.MMCIF
    )
    result = add_retained_non_polymer_hydrogens(
        source,
        component_library=ComponentLibrary(templates={"ZZQ": custom}),
        allow_retained_non_polymer_rdkit_fallback=False,
    )
    assert not result.issues
    assert {"HN1", "HN2", "HXT"} <= set(
        result.structure.constitution.ligands[0].atom_site_names()
    )
    assert to_rdkit_molecule(result.structure).GetNumAtoms() == len(component.atoms)


def test_linkless_het_peptide_contact_is_visible_in_retained_clash_scope() -> None:
    records = [
        "HETATM" + line[6:]
        for line in Path("tests/fixtures/corpus/pdb1j2v.ent").read_text().splitlines()
        if line.startswith(("ATOM  ", "HETATM"))
        and line[21] == "A"
        and int(line[22:26]) in (1, 2)
    ]
    source = read_structure_string("\n".join((*records, "END")), FileFormat.PDB)
    assert not source.constitution.chains
    assert len(source.constitution.ligands) == 2
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.structure.constitution.chains
    pair = frozenset((AtomRef(ResidueId("A", 1), "C"), AtomRef(ResidueId("A", 2), "N")))
    clashes = detect_clashes(
        result.structure,
        component_library=build_default_component_library(),
        policy=ClashPolicy(include_ligands=True),
    )
    assert any(
        frozenset(
            (
                AtomRef(c.left_residue_id, c.left_atom_name),
                AtomRef(c.right_residue_id, c.right_atom_name),
            )
        )
        == pair
        for c in clashes.clashes
    )

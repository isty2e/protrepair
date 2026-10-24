"""Constitution membership owns canonical residue-role projections."""

import pickle

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_bonded_structure,
    chain_payload,
    residue_payload,
)

from protrepair.chemistry.nonstandard.registry import build_bundled_nonstandard_registry
from protrepair.geometry import Vec3
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.structure.constitution import (
    AtomSite,
    ChainSite,
    ResidueSite,
    StructureConstitution,
)
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.hydrogen import add_hydrogens


@pytest.mark.parametrize("polymer_flag", (False, True))
@pytest.mark.parametrize("ligand_flag", (False, True))
def test_constitution_normalizes_roles_without_mutating_supplied_members(
    polymer_flag: bool, ligand_flag: bool
) -> None:
    polymer = ResidueSite("MSE", ResidueId("A", 1), (AtomSite("N", "N"),), polymer_flag)
    ligand = ResidueSite("SAH", ResidueId("A", 2), (AtomSite("N", "N"),), ligand_flag)
    chain = ChainSite("A", (polymer,))

    constitution = StructureConstitution(chains=(chain,), ligands=(ligand,))
    canonical_polymer = constitution.chain("A").residues[0]
    canonical_ligand = constitution.ligands[0]

    assert not canonical_polymer.is_hetero
    assert canonical_ligand.is_hetero
    assert polymer.is_hetero is polymer_flag
    assert ligand.is_hetero is ligand_flag
    assert chain.residues[0] is polymer
    assert canonical_polymer.atom_sites == polymer.atom_sites
    assert canonical_ligand.atom_sites == ligand.atom_sites
    assert tuple(ref.residue_id for ref in constitution.residue_slots) == (
        polymer.residue_id,
        ligand.residue_id,
    )
    if not polymer_flag:
        assert constitution.chains[0] is chain
        assert canonical_polymer is polymer
    if ligand_flag:
        assert canonical_ligand is ligand
    assert pickle.loads(pickle.dumps(constitution)) == constitution


@pytest.mark.parametrize("flag", (False, True))
def test_normalization_preserves_explicitly_retained_water_inventory(
    flag: bool,
) -> None:
    water = ResidueSite("HOH", ResidueId("W", 1), (AtomSite("O", "O"),), flag)
    constitution = StructureConstitution(chains=(), ligands=(water,))

    assert constitution.ligands[0].component_id == "HOH"
    assert constitution.ligands[0].is_hetero
    assert constitution.atom_slots == water.atom_sites
    assert water.is_hetero is flag


@pytest.mark.parametrize("retained", (False, True))
@pytest.mark.parametrize("flag", (False, True))
def test_whole_and_projected_observations_agree_on_occurrence_role(
    retained: bool, flag: bool
) -> None:
    record = build_bundled_nonstandard_registry().get("HYP")
    assert record is not None
    payload = residue_payload(
        component_id="HYP",
        residue_id=ResidueId("A", 1),
        is_hetero=flag,
        atoms=tuple(
            atom_payload(atom.atom_name, atom.element, Vec3.coerce(atom.ideal_position))
            for atom in record.heavy_atoms()
            if atom.ideal_position is not None
        ),
    )
    source = build_bonded_structure(
        chains=() if retained else (chain_payload("A", (payload,)),),
        ligands=(payload,) if retained else (),
        source_format=FileFormat.MMCIF,
    )
    observation = StructureChemistryObservation(
        source, allow_retained_non_polymer_rdkit_fallback=False
    )
    whole = observation.whole_structure_facts()
    projected = observation.projected_facts((ResidueId("A", 1),))

    assert (
        len(whole[0].residue_facts)
        == len(projected[0].residue_facts)
        == int(not retained)
    )
    assert (
        len(whole[1].retained_non_polymer_facts)
        == len(projected[1].retained_non_polymer_facts)
        == int(retained)
    )
    if not retained:
        result = add_hydrogens(source)
        residue = result.structure.constitution.chain("A").residues[0]
        assert not residue.is_hetero
        assert {"H1", "H2"} <= set(residue.atom_site_names())

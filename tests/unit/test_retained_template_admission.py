"""Charge declarations constrain compatible publication, not partial placement."""

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    residue_payload,
)

from protrepair.chemistry import build_default_component_library
from protrepair.chemistry.retained_non_polymer.registry import (
    build_bundled_retained_non_polymer_registry,
)
from protrepair.geometry import Vec3
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.state.topology import TopologyAvailabilityState
from protrepair.structure.aggregate import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
    resolve_retained_non_polymer_topology,
)


def _mannose(
    *, charges: tuple[tuple[str, int], ...] = (), omitted: tuple[str, ...] = ()
) -> ProteinStructure:
    record = build_bundled_retained_non_polymer_registry().get("MAN")
    assert record is not None
    charge_by_name = dict(charges)
    atoms = []
    for atom in record.heavy_atoms():
        if atom.atom_name in omitted:
            continue
        assert atom.ideal_position is not None
        atoms.append(
            atom_payload(
                atom.atom_name,
                atom.element,
                Vec3.coerce(atom.ideal_position),
                formal_charge=charge_by_name.get(atom.atom_name),
            )
        )

    return build_structure(
        chains=(),
        source_format=FileFormat.PDB,
        ligands=(
            residue_payload(
                component_id="MAN",
                residue_id=ResidueId("L", 1),
                is_hetero=True,
                atoms=tuple(atoms),
            ),
        ),
    )


@pytest.mark.parametrize("charge", (-1, 1))
@pytest.mark.parametrize("topology_only", (False, True))
@pytest.mark.parametrize("missing_heavy", (False, True))
def test_preexisting_neutral_bundled_component_preserves_charge_conflict(
    charge: int, topology_only: bool, missing_heavy: bool
) -> None:
    source = _mannose(
        charges=(("O1", charge),), omitted=("O3",) if missing_heavy else ()
    )
    execute = (
        resolve_retained_non_polymer_topology
        if topology_only
        else add_retained_non_polymer_hydrogens
    )
    result = execute(source, build_default_component_library())
    assert result.issues and not result.repairs
    assert result.structure == source
    template = build_default_component_library().require("MAN")
    assert dict(template.heavy_atom_charge_requirements())["O1"] == 0
    fact = (
        StructureChemistryObservation(source)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert (
        fact.heavy_atom_topology_availability_state
        is not TopologyAvailabilityState.PRESENT
    )


@pytest.mark.parametrize("explicit_zero", (False, True))
def test_known_neutral_requirement_does_not_invent_source_charge(
    explicit_zero: bool,
) -> None:
    record = build_bundled_retained_non_polymer_registry().get("MAN")
    assert record is not None
    source = _mannose(
        charges=tuple((atom.atom_name, 0) for atom in record.heavy_atoms())
        if explicit_zero
        else (),
    )
    result = add_retained_non_polymer_hydrogens(source)
    assert not result.issues
    for old, _ in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(old)
        new = result.structure.constitution.atom_index(ref)
        assert result.structure.topology.formal_charge(
            new
        ) == source.topology.formal_charge(old)
        assert result.structure.geometry.atom_geometry(
            new
        ) == source.geometry.atom_geometry(old)
    fact = (
        StructureChemistryObservation(result.structure)
        .whole_structure_facts()[1]
        .retained_non_polymer_facts[0]
    )
    assert (
        fact.heavy_atom_topology_availability_state is TopologyAvailabilityState.PRESENT
    )
    assert (
        fact.hydrogen_topology_availability_state is TopologyAvailabilityState.PRESENT
    )

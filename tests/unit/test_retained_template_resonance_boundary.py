"""Source preservation at the unsupported fixed-template resonance boundary."""

from typing import Literal

import pytest
from tests.support.canonical_builders import (
    atom_payload,
    build_structure,
    residue_payload,
)

from protrepair.chemistry import (
    IdealGeometryHydrogenSemantics,
    build_default_component_library,
)
from protrepair.diagnostics.kinds import ValidationIssueKind
from protrepair.geometry import Vec3
from protrepair.io import read_structure_string, write_structure_string
from protrepair.state import TopologyAvailabilityState
from protrepair.state.chemistry_observation import StructureChemistryObservation
from protrepair.structure import ProteinStructure
from protrepair.structure.labels import ResidueId
from protrepair.structure.provenance import FileFormat
from protrepair.transformer.completion.retained_non_polymer_hydrogen.repair import (
    add_retained_non_polymer_hydrogens,
    resolve_retained_non_polymer_topology,
)
from protrepair.transformer.continuous.readiness import (
    ContinuousRelaxationReadinessPolicy,
)


def _fmn_source(
    form: Literal["connectivity", "matching", "alternate"], *, with_hydrogens: bool
) -> ProteinStructure:
    template = build_default_component_library().require("FMN")
    semantics = template.hydrogen_semantics
    assert isinstance(semantics, IdealGeometryHydrogenSemantics)
    component = semantics.component
    atoms = tuple(
        atom for atom in component.atoms if with_hydrogens or not atom.is_hydrogen()
    )
    assert all(atom.ideal_position is not None for atom in atoms)
    residue_id = ResidueId("L", 1)
    source = build_structure(
        chains=(),
        ligands=(
            residue_payload(
                component_id="FMN",
                residue_id=residue_id,
                is_hetero=True,
                atoms=tuple(
                    atom_payload(
                        atom.atom_name,
                        atom.element,
                        Vec3(*atom.ideal_position),
                        formal_charge=atom.formal_charge,
                    )
                    for atom in atoms
                    if atom.ideal_position is not None
                ),
            ),
        ),
        source_format=FileFormat.PDB,
    )
    block = (
        "\n".join(
            line
            for line in write_structure_string(source, FileFormat.PDB).splitlines()
            if not line.startswith(("END", "CONECT"))
        )
        + "\n"
    )
    serial_by_name = {atom.atom_name: i for i, atom in enumerate(atoms, 1)}
    ring_names = frozenset(("C6", "C7", "C8", "C9", "C9A", "C5A"))
    alternate_doubles = {
        frozenset(pair) for pair in (("C6", "C7"), ("C8", "C9"), ("C9A", "C5A"))
    }
    for bond in component.bonds:
        if (
            bond.atom_name_1 not in serial_by_name
            or bond.atom_name_2 not in serial_by_name
        ):
            continue
        order = 1 if form == "connectivity" else bond.order
        pair = frozenset((bond.atom_name_1, bond.atom_name_2))
        if form == "alternate" and pair.issubset(ring_names):
            order = 2 if pair in alternate_doubles else 1
        block += (
            f"CONECT{serial_by_name[bond.atom_name_1]:5d}"
            + f"{serial_by_name[bond.atom_name_2]:5d}" * order
            + "\n"
        )
    return read_structure_string(block + "END\n", FileFormat.PDB)


@pytest.mark.parametrize("form", ("connectivity", "matching", "alternate"))
@pytest.mark.parametrize("with_hydrogens", (False, True))
def test_fixed_fmn_template_representation_boundary_preserves_rejected_source(
    form: Literal["connectivity", "matching", "alternate"], with_hydrogens: bool
) -> None:
    source = _fmn_source(form, with_hydrogens=with_hydrogens)
    library = build_default_component_library()
    result = add_retained_non_polymer_hydrogens(source, library)
    if form != "alternate":
        assert (
            sum(atom.is_hydrogen() for atom in result.structure.constitution.atom_slots)
            == 21
        )
        assert not any(
            issue.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
            for issue in result.issues
        )
        return
    assert result.structure is source
    assert result.repairs == ()
    assert any(
        issue.kind is ValidationIssueKind.CHEMISTRY_CONTRADICTION
        for issue in result.issues
    )
    if with_hydrogens:
        fact = (
            StructureChemistryObservation(source, component_library=library)
            .whole_structure_facts()[1]
            .retained_non_polymer_facts[0]
        )
        assert (
            fact.heavy_atom_topology_availability_state
            is TopologyAvailabilityState.ABSENT
        )
        assert fact.requires_topology_resolution()
        policy = ContinuousRelaxationReadinessPolicy()
        assert policy.retained_non_polymer_readiness_blocker((fact,))
        graph_only = resolve_retained_non_polymer_topology(source, library)
        assert graph_only.structure is source
        assert graph_only.repairs == ()

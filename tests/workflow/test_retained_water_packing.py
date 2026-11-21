"""Real packer output around a stationary retained water."""

from pathlib import Path

import pytest

from protrepair.diagnostics import ValidationIssueKind
from protrepair.io import (
    FileFormat,
    NonPolymerRetentionPolicy,
    RetentionAction,
    read_structure_string,
)
from protrepair.io.ingress_policy import StructureNormalizationPolicy
from protrepair.transformer.packing.faspr.paths import faspr_executable_path
from protrepair.transformer.packing.runtime import execute_sidechain_packing
from protrepair.transformer.packing.spec import PackingSpec

pytestmark = pytest.mark.workflow


def test_native_faspr_keeps_a_remote_water_and_refuses_a_blocked_output():
    try:
        faspr_executable_path()
    except FileNotFoundError:
        pytest.skip("packaged FASPR executable is unavailable")

    path = Path(__file__).parents[1] / "fixtures/pdb/span-reconstruction/1ubq.pdb"
    fragment = tuple(
        r
        for r in path.read_text().splitlines()
        if r.startswith("ATOM  ") and 3 <= int(r[22:26]) <= 7
    )
    source = read_structure_string("\n".join((*fragment, "END", "")), FileFormat.PDB)
    spec = PackingSpec("faspr")
    dry = execute_sidechain_packing(source, spec)
    displacements = []
    for r in source.constitution.residue_slots:
        index = source.constitution.residue_index(r.residue_id)
        initial = source.residue_geometry(index)
        packed = dry.packed_structure.residue_geometry(
            dry.packed_structure.constitution.residue_index(r.residue_id)
        )
        for atom in r.atom_sites:
            if atom.element == "C" and packed.has_atom(atom.name):
                point = packed.position(atom.name)
                displacements.append(
                    (point.distance_to(initial.position(atom.name)), point)
                )
    shift, obstacle = max(displacements, key=lambda item: item[0])
    assert shift > 1.0
    policy = StructureNormalizationPolicy(
        non_polymer_retention=NonPolymerRetentionPolicy(
            RetentionAction.KEEP, RetentionAction.DROP
        )
    )

    for xyz, blocked in ((obstacle, True), ((90.0, 90.0, 90.0), False)):
        x, y, z = xyz
        water = f"HETATM  999  O   HOH W   1    {x:8.3f}{y:8.3f}{z:8.3f}  1.00 11.00           O"
        wet = read_structure_string(
            "\n".join((*fragment, "TER", water, "END", "")),
            FileFormat.PDB,
            policy=policy,
        )
        result = execute_sidechain_packing(wet, spec)
        assert result.packed_structure.constitution.waters == wet.constitution.waters
        assert bool(result.changed_residue_ids) != blocked
        if blocked:
            assert result.packed_structure is wet
            assert any(
                i.kind is ValidationIssueKind.RETAINED_WATER_REPAIR_REFUSED
                for i in result.issues
            )
        else:
            assert (
                result.packed_structure.without_waters().geometry
                == dry.packed_structure.geometry
            )

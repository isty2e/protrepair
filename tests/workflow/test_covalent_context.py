"""Fixed covalent partners must not block otherwise supported H completion."""

from dataclasses import replace
from pathlib import Path

from tests.support.request_builders import whole_structure_requested_goals

from protrepair.api import process_structure
from protrepair.io import read_structure
from protrepair.state import HydrogenCoverageState


def test_disulfide_context_allows_2lyz_default_hydrogen_completion():
    path = Path(__file__).parents[1] / "fixtures/corpus/pdb2lyz.ent"
    source = read_structure(path)
    result = process_structure(
        path,
        requested_goals=whole_structure_requested_goals(HydrogenCoverageState.COMPLETE),
    )
    assert result.requested_goal_report is not None
    assert result.requested_goal_report.unmet_count() == 0
    assert result.requested_goal_report.blocked_count() == 0
    assert result.requested_goal_report.unsupported_count() == 0
    assert len(source.constitution.atom_slots) == 1001
    assert len(result.structure.constitution.atom_slots) == 1960
    for index, _ in source.geometry.iter_positions():
        ref = source.constitution.atom_ref_at(index)
        current = result.structure.constitution.atom_index(ref)
        old = source.geometry.atom_geometry(index)
        new = result.structure.geometry.atom_geometry(current)
        assert replace(old, position=new.position) == new
        assert result.structure.constitution.atom_site_at(
            current
        ) == source.constitution.atom_site_at(index)
    observation = result.structure.provenance.ingress.observation
    assert observation is not None
    assert observation.geometry == source.geometry

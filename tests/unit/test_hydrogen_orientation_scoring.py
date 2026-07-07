"""Focused tests for placement-time rotatable-hydrogen scoring."""

import numpy as np
import pytest
from tests.support.canonical_builders import (
    CanonicalAtomPayload,
    atom_payload,
    completion_payload,
)

from protrepair.chemistry.standard.components import build_standard_component_library
from protrepair.geometry import GeometryPlacementError, Vec3
from protrepair.structure.labels import ResidueId
from protrepair.transformer.completion.hydrogen.geometry import scale_bond
from protrepair.transformer.completion.hydrogen.rotatable import (
    RotatableHydrogenEnvironment,
    RotatableHydrogenSearch,
    build_rotatable_hydrogen_environments,
)
from protrepair.transformer.completion.hydrogen.scoring import recalculate_coordinate
from protrepair.transformer.completion.shared.domain import CompletionResiduePayload


def test_optimize_rotatable_hydrogen_penalizes_same_residue_two_hop_clashes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Two-hop same-residue heavy contacts should influence rotatable-H scoring."""

    candidate_hydrogens = (Vec3(0.0, 0.0, 0.0), Vec3(3.0, 0.0, 0.0))
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_positions",
        lambda self: candidate_hydrogens,
    )
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "potential_energy",
        lambda self, hydrogen, environment: 0.0,
    )

    environments = build_rotatable_hydrogen_environments(
        residues=(
            build_residue(
                "SER",
                "A",
                7,
                (
                    atom_payload("N", "N", Vec3(-1.0, 0.0, 0.0)),
                    atom_payload("CA", "C", Vec3(0.7, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(8.0, 0.0, 0.0)),
                    atom_payload("O", "O", Vec3(9.0, 0.0, 0.0)),
                    atom_payload("CB", "C", Vec3(0.0, 2.0, 0.0)),
                    atom_payload("OG", "O", Vec3(0.0, 1.0, 0.0)),
                ),
            ),
        ),
        residue_numbers=["7"],
        templates=(build_standard_component_library().require("SER"),),
    )

    result = search_template().optimized_coordinate(
        residue_number="7",
        environments=environments,
    )

    assert result == Vec3(3.0, 0.0, 0.0)


def test_optimize_rotatable_hydrogen_keeps_probable_hydrogen_bond_orientation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Probable donor-H...acceptor contacts should not be penalized as clashes."""

    candidate_hydrogens = (Vec3(1.9, 0.0, 0.0), Vec3(0.0, 2.0, 0.0))
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_positions",
        lambda self: candidate_hydrogens,
    )
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "potential_energy",
        lambda self, hydrogen, environment: 0.0,
    )

    result = search_template().optimized_coordinate(
        residue_number="11",
        environments=(
            RotatableHydrogenEnvironment(
                residue_number="11",
                atom_x=(0.0, 0.0),
                atom_y=(0.0, 2.7),
                atom_z=(0.0, 0.0),
                elements=("O", "C"),
                charges=(0.0, 0.0),
                sigmas_nm=(0.0, 0.0),
                epsilons_kj_mol=(0.0, 0.0),
            ),
        ),
    )

    assert result == Vec3(1.9, 0.0, 0.0)


def test_optimize_rotatable_hydrogen_selects_least_bad_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When every orientation clashes, the least-overlapping candidate should win."""

    candidate_hydrogens = (
        Vec3(0.8, 0.0, 0.0),
        Vec3(1.2, 0.0, 0.0),
        Vec3(1.8, 0.0, 0.0),
    )
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_positions",
        lambda self: candidate_hydrogens,
    )
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "potential_energy",
        lambda self, hydrogen, environment: 0.0,
    )

    result = search_template().optimized_coordinate(
        residue_number="19",
        environments=(
            RotatableHydrogenEnvironment(
                residue_number="19",
                atom_x=(0.0,),
                atom_y=(0.0,),
                atom_z=(0.0,),
                elements=("C",),
                charges=(0.0,),
                sigmas_nm=(0.0,),
                epsilons_kj_mol=(0.0,),
            ),
        ),
    )

    assert result == Vec3(1.8, 0.0, 0.0)


def test_optimize_rotatable_hydrogen_preserves_candidate_identity_across_environments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The best score from a later environment should still select its candidate."""

    candidate_hydrogens = (Vec3(1.0, 0.0, 0.0), Vec3(2.0, 0.0, 0.0))
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_positions",
        lambda self: candidate_hydrogens,
    )
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_score",
        lambda self, hydrogen, environment: (
            1.0
            if environment.atom_y == (2.0,) and Vec3.coerce(hydrogen).x == 2.0
            else 5.0
        ),
    )

    result = search_template().optimized_coordinate(
        residue_number="19",
        environments=(
            RotatableHydrogenEnvironment(
                residue_number="19",
                atom_x=(0.0,),
                atom_y=(1.0,),
                atom_z=(0.0,),
                elements=("C",),
                charges=(0.0,),
                sigmas_nm=(0.0,),
                epsilons_kj_mol=(0.0,),
            ),
            RotatableHydrogenEnvironment(
                residue_number="19",
                atom_x=(0.0,),
                atom_y=(2.0,),
                atom_z=(0.0,),
                elements=("C",),
                charges=(0.0,),
                sigmas_nm=(0.0,),
                epsilons_kj_mol=(0.0,),
            ),
        ),
    )

    assert result == Vec3(2.0, 0.0, 0.0)


def test_optimize_rotatable_hydrogen_keeps_first_candidate_on_score_tie(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tie handling should remain deterministic in environment scan order."""

    candidate_hydrogens = (Vec3(1.0, 0.0, 0.0), Vec3(2.0, 0.0, 0.0))
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_positions",
        lambda self: candidate_hydrogens,
    )
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_score",
        lambda self, hydrogen, environment: (
            1.0
            if (
                environment.atom_y == (1.0,)
                and Vec3.coerce(hydrogen).x == 2.0
            )
            or (
                environment.atom_y == (2.0,)
                and Vec3.coerce(hydrogen).x == 1.0
            )
            else 5.0
        ),
    )

    result = search_template().optimized_coordinate(
        residue_number="19",
        environments=(
            RotatableHydrogenEnvironment(
                residue_number="19",
                atom_x=(0.0,),
                atom_y=(1.0,),
                atom_z=(0.0,),
                elements=("C",),
                charges=(0.0,),
                sigmas_nm=(0.0,),
                epsilons_kj_mol=(0.0,),
            ),
            RotatableHydrogenEnvironment(
                residue_number="19",
                atom_x=(0.0,),
                atom_y=(2.0,),
                atom_z=(0.0,),
                elements=("C",),
                charges=(0.0,),
                sigmas_nm=(0.0,),
                epsilons_kj_mol=(0.0,),
            ),
        ),
    )

    assert result == Vec3(2.0, 0.0, 0.0)


def test_optimize_rotatable_hydrogen_does_not_score_nonmatching_environments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Residue-number filtering should happen before any candidate scoring."""

    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_positions",
        lambda self: (Vec3(1.0, 0.0, 0.0),),
    )

    def score_matching_environment(
        self,
        hydrogen,
        environment: RotatableHydrogenEnvironment,
    ) -> float:
        if environment.residue_number != "target":
            pytest.fail("nonmatching environments must not be scored")
        return 0.0

    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_score",
        score_matching_environment,
    )

    result = search_template().optimized_coordinate(
        residue_number="target",
        environments=(
            minimal_environment("other", marker=1.0),
            minimal_environment("target", marker=2.0),
        ),
    )

    assert result == Vec3(1.0, 0.0, 0.0)


def test_optimize_rotatable_hydrogen_handles_single_candidate_across_environments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A one-candidate scan should still evaluate every matching environment."""

    candidate = Vec3(4.0, 0.0, 0.0)
    scored_environment_markers: list[float] = []
    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_positions",
        lambda self: (candidate,),
    )

    def score_environment(
        self,
        hydrogen,
        environment: RotatableHydrogenEnvironment,
    ) -> float:
        scored_environment_markers.append(environment.atom_y[0])
        return environment.atom_y[0]

    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_score",
        score_environment,
    )

    result = search_template().optimized_coordinate(
        residue_number="target",
        environments=(
            minimal_environment("target", marker=2.0),
            minimal_environment("target", marker=1.0),
        ),
    )

    assert result == candidate
    assert scored_environment_markers == [2.0, 1.0]


def test_optimize_rotatable_hydrogen_does_not_score_empty_candidate_scan(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An empty candidate scan should fall back without calling the scorer."""

    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_positions",
        lambda self: (),
    )

    def score_empty_candidate_scan(self, hydrogen, environment) -> float:
        pytest.fail("empty candidate scans must not call candidate_score")

    monkeypatch.setattr(
        RotatableHydrogenSearch,
        "candidate_score",
        score_empty_candidate_scan,
    )

    result = search_template().optimized_coordinate(
        residue_number="target",
        environments=(minimal_environment("target", marker=1.0),),
    )

    assert result == Vec3(9.0, 9.0, 9.0)


def test_rotatable_hydrogen_search_skips_undefined_candidate_frames() -> None:
    """Degenerate anchor frames should fall back instead of leaking math errors."""

    search = RotatableHydrogenSearch(
        outer_anchor=[0.0, 0.0, 0.0],
        inner_anchor=[1.0, 0.0, 0.0],
        donor=[1.0, 0.0, 0.0],
        hydrogen=[9.0, 9.0, 9.0],
        build_bond_length=1.0,
        reproject_bond_length=1.0,
        dihedral=0.0,
        partial_charge=0.0,
        sigma=0.0,
        epsilon=0.0,
        donor_element="O",
    )

    assert search.candidate_positions() == ()
    assert search.optimized_coordinate(
        residue_number="7",
        environments=(
            RotatableHydrogenEnvironment(
                residue_number="7",
                atom_x=(0.0,),
                atom_y=(0.0,),
                atom_z=(0.0,),
                elements=("C",),
                charges=(0.0,),
                sigmas_nm=(0.0,),
                epsilons_kj_mol=(0.0,),
            ),
        ),
    ) == Vec3(9.0, 9.0, 9.0)


def test_rotatable_hydrogen_reprojection_rejects_zero_length_bonds() -> None:
    """Rotatable-H reprojection should reject undefined vectors explicitly."""

    with pytest.raises(GeometryPlacementError, match="non-zero vectors"):
        recalculate_coordinate(
            atom_b=[0.0, 0.0, 0.0],
            atom_c=[0.0, 0.0, 0.0],
            atom_d=[1.0, 0.0, 0.0],
            bond_length=1.0,
        )


def test_hydrogen_scale_bond_rejects_degenerate_candidate() -> None:
    """Hydrogen-specific bond scaling should not divide by a zero norm."""

    origin = np.array((1.0, 1.0, 1.0), dtype=np.float64)

    with pytest.raises(GeometryPlacementError, match="degenerate bond vector"):
        scale_bond(origin, origin.copy(), 1.0)


def test_build_rotatable_hydrogen_environments_pack_local_steric_sites() -> None:
    """Rotatable residues should carry donor-centered same-residue steric sites."""

    library = build_standard_component_library()
    environments = build_rotatable_hydrogen_environments(
        residues=(
            build_residue(
                "SER",
                "A",
                1,
                (
                    atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                    atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                    atom_payload("CB", "C", Vec3(1.0, 1.0, 0.0)),
                    atom_payload("OG", "O", Vec3(1.0, 2.0, 0.0)),
                ),
            ),
            build_residue(
                "GLY",
                "A",
                2,
                (
                    atom_payload("N", "N", Vec3(4.0, 0.0, 0.0)),
                    atom_payload("CA", "C", Vec3(5.0, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(6.0, 0.0, 0.0)),
                    atom_payload("O", "O", Vec3(7.0, 0.0, 0.0)),
                ),
            ),
        ),
        residue_numbers=["1", "2"],
        templates=(
            library.require("SER"),
            library.require("GLY"),
        ),
    )

    serine_sites = {
        local_site.atom_name: local_site for local_site in environments[0].local_sites
    }

    assert serine_sites["OG"].bond_hops_from_donor == 0
    assert serine_sites["CB"].bond_hops_from_donor == 1
    assert serine_sites["CA"].bond_hops_from_donor == 2
    assert serine_sites["N"].bond_hops_from_donor == 3
    assert environments[1].local_sites == ()


def test_build_rotatable_hydrogen_environments_filters_far_chain_sites() -> None:
    """Rotatable-H environments should not pack heavy sites outside score range."""

    library = build_standard_component_library()
    environments = build_rotatable_hydrogen_environments(
        residues=(
            build_residue(
                "SER",
                "A",
                1,
                (
                    atom_payload("N", "N", Vec3(0.0, 0.0, 0.0)),
                    atom_payload("CA", "C", Vec3(1.0, 0.0, 0.0)),
                    atom_payload("C", "C", Vec3(2.0, 0.0, 0.0)),
                    atom_payload("O", "O", Vec3(3.0, 0.0, 0.0)),
                    atom_payload("CB", "C", Vec3(0.0, 1.0, 0.0)),
                    atom_payload("OG", "O", Vec3(0.0, 2.0, 0.0)),
                ),
            ),
            build_residue(
                "GLY",
                "A",
                2,
                (atom_payload("N", "N", Vec3(0.0, 5.0, 0.0)),),
            ),
            build_residue(
                "GLY",
                "A",
                3,
                (atom_payload("N", "N", Vec3(0.0, 20.0, 0.0)),),
            ),
        ),
        residue_numbers=["1", "2", "3"],
        templates=(
            library.require("SER"),
            library.require("GLY"),
            library.require("GLY"),
        ),
    )

    assert environments[0].atom_y == (5.0,)
    assert environments[1].atom_y == ()
    assert environments[2].atom_y == ()


def search_template() -> RotatableHydrogenSearch:
    """Return a minimal rotatable-hydrogen search input for direct scoring tests."""

    return RotatableHydrogenSearch(
        outer_anchor=[0.0, 0.0, 0.0],
        inner_anchor=[0.0, 0.0, 0.0],
        donor=[0.0, 0.0, 0.0],
        hydrogen=[9.0, 9.0, 9.0],
        build_bond_length=1.0,
        reproject_bond_length=1.0,
        dihedral=0.0,
        partial_charge=0.0,
        sigma=0.0,
        epsilon=0.0,
        donor_element="O",
    )


def minimal_environment(
    residue_number: str,
    *,
    marker: float,
) -> RotatableHydrogenEnvironment:
    """Return one marker-bearing environment for identity-focused tests."""

    return RotatableHydrogenEnvironment(
        residue_number=residue_number,
        atom_x=(0.0,),
        atom_y=(marker,),
        atom_z=(0.0,),
        elements=("C",),
        charges=(0.0,),
        sigmas_nm=(0.0,),
        epsilons_kj_mol=(0.0,),
    )


def build_residue(
    component_id: str,
    chain_id: str,
    seq_num: int,
    atoms: tuple[CanonicalAtomPayload, ...],
) -> CompletionResiduePayload:
    """Return one canonical completion payload for hydrogen-scoring tests."""

    return completion_payload(
        component_id=component_id,
        residue_id=ResidueId(chain_id=chain_id, seq_num=seq_num),
        atoms=atoms,
    )

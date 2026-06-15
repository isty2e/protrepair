"""Standard heavy-atom completion semantics expressed as placement programs."""

from protrepair.chemistry.component.semantics import HeavyAtomSemantics
from protrepair.chemistry.internal_coordinates import (
    InternalCoordinatePlacement,
    InternalCoordinateProgram,
    PlaceWhenAbsent,
    PlaceWhenAnyMissing,
)

STANDARD_HEAVY_ATOM_SEMANTICS: dict[str, HeavyAtomSemantics] = {
    "ALA": HeavyAtomSemantics(
        program=InternalCoordinateProgram.with_backbone_and_beta_carbon(
            cb_dihedral_degrees=122.69
        ),
        atom_order=("N", "CA", "C", "O", "CB"),
    ),
    "ARG": HeavyAtomSemantics(
        program=InternalCoordinateProgram.with_backbone_and_beta_carbon(
            cb_dihedral_degrees=122.76,
            operations=(
                PlaceWhenAnyMissing(
                    trigger_atom_names=("CG", "CD", "NE", "CZ"),
                    placements=(
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CG",
                            anchor_atom_names=("N", "CA", "CB"),
                            bond_length=1.52,
                            bond_angle=113.83,
                            dihedral_degrees=-65.2,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CD",
                            anchor_atom_names=("CA", "CB", "CG"),
                            bond_length=1.52,
                            bond_angle=111.79,
                            dihedral_degrees=-179.2,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="NE",
                            anchor_atom_names=("CB", "CG", "CD"),
                            bond_length=1.46,
                            bond_angle=111.68,
                            dihedral_degrees=-179.3,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CZ",
                            anchor_atom_names=("CG", "CD", "NE"),
                            bond_length=1.33,
                            bond_angle=124.79,
                            dihedral_degrees=-178.7,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="NH1",
                            anchor_atom_names=("CD", "NE", "CZ"),
                            bond_length=1.33,
                            bond_angle=120.64,
                            dihedral_degrees=0.0,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="NH2",
                            anchor_atom_names=("CD", "NE", "CZ"),
                            bond_length=1.33,
                            bond_angle=119.63,
                            dihedral_degrees=180.0,
                        ),
                    ),
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.absolute(
                        target_atom_name="NH1",
                        anchor_atom_names=("CD", "NE", "CZ"),
                        bond_length=1.33,
                        bond_angle=120.64,
                        dihedral_degrees=0.0,
                    )
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.absolute(
                        target_atom_name="NH2",
                        anchor_atom_names=("CD", "NE", "CZ"),
                        bond_length=1.33,
                        bond_angle=119.63,
                        dihedral_degrees=180.0,
                    )
                ),
            ),
        ),
        atom_order=("N", "CA", "C", "O", "CB", "CG", "CD", "NE", "CZ", "NH1", "NH2"),
    ),
    "ASN": HeavyAtomSemantics(
        program=InternalCoordinateProgram.with_backbone_and_beta_carbon(
            cb_dihedral_degrees=123.23,
            operations=(
                PlaceWhenAnyMissing(
                    trigger_atom_names=("CG", "OD1"),
                    placements=(
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CG",
                            anchor_atom_names=("N", "CA", "CB"),
                            bond_length=1.52,
                            bond_angle=112.62,
                            dihedral_degrees=-65.5,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="OD1",
                            anchor_atom_names=("CA", "CB", "CG"),
                            bond_length=1.23,
                            bond_angle=120.85,
                            dihedral_degrees=-58.3,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="ND2",
                            anchor_atom_names=("CA", "CB", "CG"),
                            bond_length=1.33,
                            bond_angle=116.48,
                            dihedral_degrees=121.7,
                        ),
                    ),
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.relative_to_torsion(
                        target_atom_name="ND2",
                        anchor_atom_names=("CA", "CB", "CG"),
                        bond_length=1.33,
                        bond_angle=116.48,
                        reference_atom_names=("CA", "CB", "CG", "OD1"),
                        offset_degrees=180.0,
                    )
                ),
            ),
        ),
        atom_order=("N", "CA", "C", "O", "CB", "CG", "OD1", "ND2"),
    ),
    "ASP": HeavyAtomSemantics(
        program=InternalCoordinateProgram.with_backbone_and_beta_carbon(
            cb_dihedral_degrees=122.82,
            operations=(
                PlaceWhenAnyMissing(
                    trigger_atom_names=("CG", "OD1"),
                    placements=(
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CG",
                            anchor_atom_names=("N", "CA", "CB"),
                            bond_length=1.52,
                            bond_angle=113.06,
                            dihedral_degrees=-66.4,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="OD1",
                            anchor_atom_names=("CA", "CB", "CG"),
                            bond_length=1.25,
                            bond_angle=119.22,
                            dihedral_degrees=-46.7,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="OD2",
                            anchor_atom_names=("CA", "CB", "CG"),
                            bond_length=1.25,
                            bond_angle=118.21,
                            dihedral_degrees=133.3,
                        ),
                    ),
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.relative_to_torsion(
                        target_atom_name="OD2",
                        anchor_atom_names=("CA", "CB", "CG"),
                        bond_length=1.25,
                        bond_angle=118.21,
                        reference_atom_names=("CA", "CB", "CG", "OD1"),
                        offset_degrees=180.0,
                    )
                ),
            ),
        ),
        atom_order=("N", "CA", "C", "O", "CB", "CG", "OD1", "OD2"),
    ),
    "CYS": HeavyAtomSemantics(
        program=InternalCoordinateProgram.with_backbone_and_beta_carbon(
            cb_dihedral_degrees=122.50,
            operations=(
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.absolute(
                        target_atom_name="SG",
                        anchor_atom_names=("N", "CA", "CB"),
                        bond_length=1.81,
                        bond_angle=113.82,
                        dihedral_degrees=-62.2,
                    )
                ),
            ),
        ),
        atom_order=("N", "CA", "C", "O", "CB", "SG"),
    ),
    "GLN": HeavyAtomSemantics(
        program=InternalCoordinateProgram.with_backbone_and_beta_carbon(
            cb_dihedral_degrees=122.81,
            operations=(
                PlaceWhenAnyMissing(
                    trigger_atom_names=("CG", "CD", "OE1"),
                    placements=(
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CG",
                            anchor_atom_names=("N", "CA", "CB"),
                            bond_length=1.52,
                            bond_angle=113.75,
                            dihedral_degrees=-60.2,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CD",
                            anchor_atom_names=("CA", "CB", "CG"),
                            bond_length=1.52,
                            bond_angle=112.78,
                            dihedral_degrees=-69.6,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="OE1",
                            anchor_atom_names=("CB", "CG", "CD"),
                            bond_length=1.24,
                            bond_angle=120.86,
                            dihedral_degrees=-50.5,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="NE2",
                            anchor_atom_names=("CB", "CG", "CD"),
                            bond_length=1.33,
                            bond_angle=116.50,
                            dihedral_degrees=129.5,
                        ),
                    ),
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.relative_to_torsion(
                        target_atom_name="NE2",
                        anchor_atom_names=("CB", "CG", "CD"),
                        bond_length=1.33,
                        bond_angle=116.50,
                        reference_atom_names=("CB", "CG", "CD", "OE1"),
                        offset_degrees=180.0,
                    )
                ),
            ),
        ),
        atom_order=("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "NE2"),
    ),
    "GLU": HeavyAtomSemantics(
        program=InternalCoordinateProgram.with_backbone_and_beta_carbon(
            cb_dihedral_degrees=122.87,
            operations=(
                PlaceWhenAnyMissing(
                    trigger_atom_names=("CG", "CD", "OE1"),
                    placements=(
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CG",
                            anchor_atom_names=("N", "CA", "CB"),
                            bond_length=1.52,
                            bond_angle=113.82,
                            dihedral_degrees=-63.8,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CD",
                            anchor_atom_names=("CA", "CB", "CG"),
                            bond_length=1.52,
                            bond_angle=119.02,
                            dihedral_degrees=-179.8,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="OE1",
                            anchor_atom_names=("CB", "CG", "CD"),
                            bond_length=1.25,
                            bond_angle=119.02,
                            dihedral_degrees=-6.2,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="OE2",
                            anchor_atom_names=("CB", "CG", "CD"),
                            bond_length=1.25,
                            bond_angle=118.08,
                            dihedral_degrees=173.8,
                        ),
                    ),
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.relative_to_torsion(
                        target_atom_name="OE2",
                        anchor_atom_names=("CB", "CG", "CD"),
                        bond_length=1.25,
                        bond_angle=118.08,
                        reference_atom_names=("CB", "CG", "CD", "OE1"),
                        offset_degrees=180.0,
                    )
                ),
            ),
        ),
        atom_order=("N", "CA", "C", "O", "CB", "CG", "CD", "OE1", "OE2"),
    ),
    "GLY": HeavyAtomSemantics(
        program=InternalCoordinateProgram.backbone_only(),
        atom_order=("N", "CA", "C", "O"),
    ),
    "HIS": HeavyAtomSemantics(
        program=InternalCoordinateProgram.with_backbone_and_beta_carbon(
            cb_dihedral_degrees=122.67,
            operations=(
                PlaceWhenAnyMissing(
                    trigger_atom_names=("CG", "ND1"),
                    placements=(
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CG",
                            anchor_atom_names=("N", "CA", "CB"),
                            bond_length=1.49,
                            bond_angle=113.74,
                            dihedral_degrees=-63.2,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="ND1",
                            anchor_atom_names=("CA", "CB", "CG"),
                            bond_length=1.38,
                            bond_angle=122.85,
                            dihedral_degrees=-75.7,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CD2",
                            anchor_atom_names=("CA", "CB", "CG"),
                            bond_length=1.35,
                            bond_angle=130.61,
                            dihedral_degrees=104.3,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CE1",
                            anchor_atom_names=("CB", "CG", "ND1"),
                            bond_length=1.32,
                            bond_angle=108.5,
                            dihedral_degrees=180.0,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="NE2",
                            anchor_atom_names=("CB", "CG", "CD2"),
                            bond_length=1.35,
                            bond_angle=108.5,
                            dihedral_degrees=180.0,
                        ),
                    ),
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.relative_to_torsion(
                        target_atom_name="CD2",
                        anchor_atom_names=("CA", "CB", "CG"),
                        bond_length=1.35,
                        bond_angle=130.61,
                        reference_atom_names=("CA", "CB", "CG", "ND1"),
                        offset_degrees=180.0,
                    )
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.absolute(
                        target_atom_name="CE1",
                        anchor_atom_names=("CB", "CG", "ND1"),
                        bond_length=1.32,
                        bond_angle=108.5,
                        dihedral_degrees=180.0,
                    )
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.absolute(
                        target_atom_name="NE2",
                        anchor_atom_names=("CB", "CG", "CD2"),
                        bond_length=1.35,
                        bond_angle=108.5,
                        dihedral_degrees=180.0,
                    )
                ),
            ),
        ),
        atom_order=("N", "CA", "C", "O", "CB", "CG", "ND1", "CD2", "CE1", "NE2"),
    ),
    "ILE": HeavyAtomSemantics(
        program=InternalCoordinateProgram.with_backbone_and_beta_carbon(
            cb_dihedral_degrees=123.23,
            operations=(
                PlaceWhenAnyMissing(
                    trigger_atom_names=("CG1", "CD1"),
                    placements=(
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CG1",
                            anchor_atom_names=("N", "CA", "CB"),
                            bond_length=1.527,
                            bond_angle=110.7,
                            dihedral_degrees=59.7,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CG2",
                            anchor_atom_names=("N", "CA", "CB"),
                            bond_length=1.527,
                            bond_angle=110.4,
                            dihedral_degrees=-60.3,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CD1",
                            anchor_atom_names=("CA", "CB", "CG1"),
                            bond_length=1.52,
                            bond_angle=113.97,
                            dihedral_degrees=169.8,
                        ),
                    ),
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.relative_to_torsion(
                        target_atom_name="CG2",
                        anchor_atom_names=("N", "CA", "CB"),
                        bond_length=1.527,
                        bond_angle=110.4,
                        reference_atom_names=("N", "CA", "CB", "CG1"),
                        offset_degrees=-120.0,
                    )
                ),
            ),
        ),
        atom_order=("N", "CA", "C", "O", "CB", "CG1", "CG2", "CD1"),
    ),
    "LEU": HeavyAtomSemantics(
        program=InternalCoordinateProgram.with_backbone_and_beta_carbon(
            cb_dihedral_degrees=122.49,
            operations=(
                PlaceWhenAnyMissing(
                    trigger_atom_names=("CG", "CD1"),
                    placements=(
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CG",
                            anchor_atom_names=("N", "CA", "CB"),
                            bond_length=1.53,
                            bond_angle=116.10,
                            dihedral_degrees=-60.1,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CD1",
                            anchor_atom_names=("CA", "CB", "CG"),
                            bond_length=1.524,
                            bond_angle=112.50,
                            dihedral_degrees=174.9,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CD2",
                            anchor_atom_names=("CA", "CB", "CG"),
                            bond_length=1.525,
                            bond_angle=112.50,
                            dihedral_degrees=294.9,
                        ),
                    ),
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.relative_to_torsion(
                        target_atom_name="CD2",
                        anchor_atom_names=("CA", "CB", "CG"),
                        bond_length=1.525,
                        bond_angle=112.50,
                        reference_atom_names=("CA", "CB", "CG", "CD1"),
                        offset_degrees=120.0,
                    )
                ),
            ),
        ),
        atom_order=("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2"),
    ),
    "LYS": HeavyAtomSemantics(
        program=InternalCoordinateProgram.with_backbone_and_beta_carbon(
            cb_dihedral_degrees=122.76,
            operations=(
                PlaceWhenAnyMissing(
                    trigger_atom_names=("CG", "CD", "CE", "NZ"),
                    placements=(
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CG",
                            anchor_atom_names=("N", "CA", "CB"),
                            bond_length=1.52,
                            bond_angle=113.83,
                            dihedral_degrees=-64.5,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CD",
                            anchor_atom_names=("CA", "CB", "CG"),
                            bond_length=1.52,
                            bond_angle=111.79,
                            dihedral_degrees=-178.1,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CE",
                            anchor_atom_names=("CB", "CG", "CD"),
                            bond_length=1.46,
                            bond_angle=111.68,
                            dihedral_degrees=-179.6,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="NZ",
                            anchor_atom_names=("CG", "CD", "CE"),
                            bond_length=1.33,
                            bond_angle=124.79,
                            dihedral_degrees=179.6,
                        ),
                    ),
                ),
            ),
        ),
        atom_order=("N", "CA", "C", "O", "CB", "CG", "CD", "CE", "NZ"),
    ),
    "MET": HeavyAtomSemantics(
        program=InternalCoordinateProgram.with_backbone_and_beta_carbon(
            cb_dihedral_degrees=122.67,
            operations=(
                PlaceWhenAnyMissing(
                    trigger_atom_names=("CG", "SD", "CE"),
                    placements=(
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CG",
                            anchor_atom_names=("N", "CA", "CB"),
                            bond_length=1.52,
                            bond_angle=113.68,
                            dihedral_degrees=-64.4,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="SD",
                            anchor_atom_names=("CA", "CB", "CG"),
                            bond_length=1.81,
                            bond_angle=112.69,
                            dihedral_degrees=-179.6,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CE",
                            anchor_atom_names=("CB", "CG", "SD"),
                            bond_length=1.79,
                            bond_angle=100.61,
                            dihedral_degrees=70.1,
                        ),
                    ),
                ),
            ),
        ),
        atom_order=("N", "CA", "C", "O", "CB", "CG", "SD", "CE"),
    ),
    "PHE": HeavyAtomSemantics(
        program=InternalCoordinateProgram.with_backbone_and_beta_carbon(
            cb_dihedral_degrees=122.61,
            operations=(
                PlaceWhenAnyMissing(
                    trigger_atom_names=("CG", "CD1"),
                    placements=(
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CG",
                            anchor_atom_names=("N", "CA", "CB"),
                            bond_length=1.50,
                            bond_angle=113.85,
                            dihedral_degrees=-64.7,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CD1",
                            anchor_atom_names=("CA", "CB", "CG"),
                            bond_length=1.39,
                            bond_angle=120.0,
                            dihedral_degrees=93.3,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CD2",
                            anchor_atom_names=("CA", "CB", "CG"),
                            bond_length=1.39,
                            bond_angle=120.0,
                            dihedral_degrees=-86.7,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CE1",
                            anchor_atom_names=("CB", "CG", "CD1"),
                            bond_length=1.39,
                            bond_angle=120.0,
                            dihedral_degrees=180.0,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CE2",
                            anchor_atom_names=("CB", "CG", "CD2"),
                            bond_length=1.39,
                            bond_angle=120.0,
                            dihedral_degrees=180.0,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CZ",
                            anchor_atom_names=("CG", "CD1", "CE1"),
                            bond_length=1.39,
                            bond_angle=120.0,
                            dihedral_degrees=0.0,
                        ),
                    ),
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.relative_to_torsion(
                        target_atom_name="CD2",
                        anchor_atom_names=("CA", "CB", "CG"),
                        bond_length=1.39,
                        bond_angle=120.0,
                        reference_atom_names=("CA", "CB", "CG", "CD1"),
                        offset_degrees=-180.0,
                    )
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.absolute(
                        target_atom_name="CE1",
                        anchor_atom_names=("CB", "CG", "CD1"),
                        bond_length=1.39,
                        bond_angle=120.0,
                        dihedral_degrees=180.0,
                    )
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.absolute(
                        target_atom_name="CE2",
                        anchor_atom_names=("CB", "CG", "CD2"),
                        bond_length=1.39,
                        bond_angle=120.0,
                        dihedral_degrees=180.0,
                    )
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.absolute(
                        target_atom_name="CZ",
                        anchor_atom_names=("CG", "CD1", "CE1"),
                        bond_length=1.39,
                        bond_angle=120.0,
                        dihedral_degrees=0.0,
                    )
                ),
            ),
        ),
        atom_order=("N", "CA", "C", "O", "CB", "CG", "CD1", "CD2", "CE1", "CE2", "CZ"),
    ),
    "PRO": HeavyAtomSemantics(
        program=InternalCoordinateProgram.with_backbone_and_beta_carbon(
            cb_dihedral_degrees=115.30,
            operations=(
                PlaceWhenAnyMissing(
                    trigger_atom_names=("CG", "CD"),
                    placements=(
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CG",
                            anchor_atom_names=("N", "CA", "CB"),
                            bond_length=1.49,
                            bond_angle=104.21,
                            dihedral_degrees=29.6,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CD",
                            anchor_atom_names=("CA", "CB", "CG"),
                            bond_length=1.50,
                            bond_angle=105.03,
                            dihedral_degrees=-34.8,
                        ),
                    ),
                ),
            ),
        ),
        atom_order=("N", "CA", "C", "O", "CB", "CG", "CD"),
    ),
    "SER": HeavyAtomSemantics(
        program=InternalCoordinateProgram.with_backbone_and_beta_carbon(
            cb_dihedral_degrees=122.66,
            operations=(
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.absolute(
                        target_atom_name="OG",
                        anchor_atom_names=("N", "CA", "CB"),
                        bond_length=1.417,
                        bond_angle=110.773,
                        dihedral_degrees=-63.3,
                    )
                ),
            ),
        ),
        atom_order=("N", "CA", "C", "O", "CB", "OG"),
    ),
    "THR": HeavyAtomSemantics(
        program=InternalCoordinateProgram.with_backbone_and_beta_carbon(
            cb_dihedral_degrees=123.10,
            operations=(
                PlaceWhenAnyMissing(
                    trigger_atom_names=("OG1",),
                    placements=(
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="OG1",
                            anchor_atom_names=("N", "CA", "CB"),
                            bond_length=1.43,
                            bond_angle=109.18,
                            dihedral_degrees=60.0,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CG2",
                            anchor_atom_names=("N", "CA", "CB"),
                            bond_length=1.53,
                            bond_angle=111.13,
                            dihedral_degrees=-60.0,
                        ),
                    ),
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.relative_to_torsion(
                        target_atom_name="CG2",
                        anchor_atom_names=("N", "CA", "CB"),
                        bond_length=1.53,
                        bond_angle=111.13,
                        reference_atom_names=("N", "CA", "CB", "OG1"),
                        offset_degrees=-120.0,
                    )
                ),
            ),
        ),
        atom_order=("N", "CA", "C", "O", "CB", "OG1", "CG2"),
    ),
    "TRP": HeavyAtomSemantics(
        program=InternalCoordinateProgram.with_backbone_and_beta_carbon(
            cb_dihedral_degrees=122.61,
            operations=(
                PlaceWhenAnyMissing(
                    trigger_atom_names=("CG", "CD1"),
                    placements=(
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CG",
                            anchor_atom_names=("N", "CA", "CB"),
                            bond_length=1.50,
                            bond_angle=114.10,
                            dihedral_degrees=-66.4,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CD1",
                            anchor_atom_names=("CA", "CB", "CG"),
                            bond_length=1.37,
                            bond_angle=127.07,
                            dihedral_degrees=96.3,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CD2",
                            anchor_atom_names=("CA", "CB", "CG"),
                            bond_length=1.43,
                            bond_angle=126.66,
                            dihedral_degrees=-83.7,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="NE1",
                            anchor_atom_names=("CB", "CG", "CD1"),
                            bond_length=1.38,
                            bond_angle=108.5,
                            dihedral_degrees=180.0,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CE2",
                            anchor_atom_names=("CB", "CG", "CD2"),
                            bond_length=1.40,
                            bond_angle=108.5,
                            dihedral_degrees=180.0,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CE3",
                            anchor_atom_names=("CB", "CG", "CD2"),
                            bond_length=1.40,
                            bond_angle=133.83,
                            dihedral_degrees=0.0,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CZ2",
                            anchor_atom_names=("CG", "CD2", "CE2"),
                            bond_length=1.40,
                            bond_angle=120.0,
                            dihedral_degrees=180.0,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CZ3",
                            anchor_atom_names=("CG", "CD2", "CE3"),
                            bond_length=1.40,
                            bond_angle=120.0,
                            dihedral_degrees=180.0,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CH2",
                            anchor_atom_names=("CD2", "CE2", "CZ2"),
                            bond_length=1.40,
                            bond_angle=120.0,
                            dihedral_degrees=0.0,
                        ),
                    ),
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.relative_to_torsion(
                        target_atom_name="CD2",
                        anchor_atom_names=("CA", "CB", "CG"),
                        bond_length=1.43,
                        bond_angle=126.66,
                        reference_atom_names=("CA", "CB", "CG", "CD1"),
                        offset_degrees=-180.0,
                    )
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.absolute(
                        target_atom_name="NE1",
                        anchor_atom_names=("CB", "CG", "CD1"),
                        bond_length=1.38,
                        bond_angle=108.5,
                        dihedral_degrees=180.0,
                    )
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.absolute(
                        target_atom_name="CE2",
                        anchor_atom_names=("CB", "CG", "CD2"),
                        bond_length=1.40,
                        bond_angle=108.5,
                        dihedral_degrees=180.0,
                    )
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.absolute(
                        target_atom_name="CE3",
                        anchor_atom_names=("CB", "CG", "CD2"),
                        bond_length=1.40,
                        bond_angle=133.83,
                        dihedral_degrees=0.0,
                    )
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.absolute(
                        target_atom_name="CZ2",
                        anchor_atom_names=("CG", "CD2", "CE2"),
                        bond_length=1.40,
                        bond_angle=120.0,
                        dihedral_degrees=180.0,
                    )
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.absolute(
                        target_atom_name="CZ3",
                        anchor_atom_names=("CG", "CD2", "CE3"),
                        bond_length=1.40,
                        bond_angle=120.0,
                        dihedral_degrees=180.0,
                    )
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.absolute(
                        target_atom_name="CH2",
                        anchor_atom_names=("CD2", "CE2", "CZ2"),
                        bond_length=1.40,
                        bond_angle=120.0,
                        dihedral_degrees=0.0,
                    )
                ),
            ),
        ),
        atom_order=(
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "NE1",
            "CE2",
            "CE3",
            "CZ2",
            "CZ3",
            "CH2",
        ),
    ),
    "TYR": HeavyAtomSemantics(
        program=InternalCoordinateProgram.with_backbone_and_beta_carbon(
            cb_dihedral_degrees=122.60,
            operations=(
                PlaceWhenAnyMissing(
                    trigger_atom_names=("CG", "CD1"),
                    placements=(
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CG",
                            anchor_atom_names=("N", "CA", "CB"),
                            bond_length=1.51,
                            bond_angle=113.8,
                            dihedral_degrees=-64.3,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CD1",
                            anchor_atom_names=("CA", "CB", "CG"),
                            bond_length=1.39,
                            bond_angle=120.98,
                            dihedral_degrees=93.1,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CD2",
                            anchor_atom_names=("CA", "CB", "CG"),
                            bond_length=1.39,
                            bond_angle=120.82,
                            dihedral_degrees=273.1,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CE1",
                            anchor_atom_names=("CB", "CG", "CD1"),
                            bond_length=1.39,
                            bond_angle=120.0,
                            dihedral_degrees=180.0,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CE2",
                            anchor_atom_names=("CB", "CG", "CD2"),
                            bond_length=1.39,
                            bond_angle=120.0,
                            dihedral_degrees=180.0,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CZ",
                            anchor_atom_names=("CG", "CD1", "CE1"),
                            bond_length=1.39,
                            bond_angle=120.0,
                            dihedral_degrees=0.0,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="OH",
                            anchor_atom_names=("CD1", "CE1", "CZ"),
                            bond_length=1.39,
                            bond_angle=119.78,
                            dihedral_degrees=180.0,
                        ),
                    ),
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.relative_to_torsion(
                        target_atom_name="CD2",
                        anchor_atom_names=("CA", "CB", "CG"),
                        bond_length=1.39,
                        bond_angle=120.82,
                        reference_atom_names=("CA", "CB", "CG", "CD1"),
                        offset_degrees=180.0,
                    )
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.absolute(
                        target_atom_name="CE1",
                        anchor_atom_names=("CB", "CG", "CD1"),
                        bond_length=1.39,
                        bond_angle=120.0,
                        dihedral_degrees=180.0,
                    )
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.absolute(
                        target_atom_name="CE2",
                        anchor_atom_names=("CB", "CG", "CD2"),
                        bond_length=1.39,
                        bond_angle=120.0,
                        dihedral_degrees=180.0,
                    )
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.absolute(
                        target_atom_name="CZ",
                        anchor_atom_names=("CG", "CD1", "CE1"),
                        bond_length=1.39,
                        bond_angle=120.0,
                        dihedral_degrees=0.0,
                    )
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.absolute(
                        target_atom_name="OH",
                        anchor_atom_names=("CD1", "CE1", "CZ"),
                        bond_length=1.39,
                        bond_angle=119.78,
                        dihedral_degrees=180.0,
                    )
                ),
            ),
        ),
        atom_order=(
            "N",
            "CA",
            "C",
            "O",
            "CB",
            "CG",
            "CD1",
            "CD2",
            "CE1",
            "CE2",
            "CZ",
            "OH",
        ),
    ),
    "VAL": HeavyAtomSemantics(
        program=InternalCoordinateProgram.with_backbone_and_beta_carbon(
            cb_dihedral_degrees=123.23,
            operations=(
                PlaceWhenAnyMissing(
                    trigger_atom_names=("CG1",),
                    placements=(
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CG1",
                            anchor_atom_names=("N", "CA", "CB"),
                            bond_length=1.527,
                            bond_angle=110.7,
                            dihedral_degrees=177.2,
                        ),
                        InternalCoordinatePlacement.absolute(
                            target_atom_name="CG2",
                            anchor_atom_names=("N", "CA", "CB"),
                            bond_length=1.527,
                            bond_angle=110.4,
                            dihedral_degrees=297.2,
                        ),
                    ),
                ),
                PlaceWhenAbsent(
                    placement=InternalCoordinatePlacement.relative_to_torsion(
                        target_atom_name="CG2",
                        anchor_atom_names=("N", "CA", "CB"),
                        bond_length=1.527,
                        bond_angle=110.4,
                        reference_atom_names=("N", "CA", "CB", "CG1"),
                        offset_degrees=120.0,
                    )
                ),
            ),
        ),
        atom_order=("N", "CA", "C", "O", "CB", "CG1", "CG2"),
    ),
}

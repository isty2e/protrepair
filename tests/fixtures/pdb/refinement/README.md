# Local Refinement Fixtures

This directory contains extracted local neighborhoods used for local refinement
benchmarking and regression.

These files are derived from wwPDB archive entries tracked by
`tests/support/refinement_cases.py` and
`tests/support/scenario_fixture_matrix.py`.

First-wave extracted cases:

- `1bkr_thr101_local.pdb`
- `1bkr_his42_local.pdb`
- `1jd0_gln92_local.pdb`
- `1lxa_his125_local.pdb`
- `1uaq_gln55_local.pdb`
- `1xgo_leu253_local.pdb`
- `1xgs_leu253_reference_local.pdb`
- `1ywr_his148_local.pdb`
- `1ywr_asn155_local.pdb`
- `3g8l_asn182_local.pdb`
- `2q6f_cys143_pje_local.pdb`

Nonstandard exploratory cases:

- `4pep_sep68_local.pdb`
- `1uhg_sep68_local.pdb`
- `8pyr_sep164_local.pdb`
- `8pyr_tpo170_local.pdb`
- `4jfx_ptr8_local.pdb`

Literature/problematic exploratory cases:

- `3j6b_terminal_helix_misthread_local.pdb`
- `3j9e_loop_backbone_error_local.pdb`
- `2h6o_glycan_problematic_local.pdb`
- `5xm5_metal_sites_local.pdb`

These are small neighborhood fixtures, not full deposited structures.

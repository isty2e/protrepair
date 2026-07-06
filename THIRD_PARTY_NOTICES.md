# Third-Party Notices

This repository contains ProtRepair project code under the terms of the root
[LICENSE](LICENSE). It also includes third-party materials that remain
subject to their own notices and provenance requirements.

## PRAS upstream

- Upstream project: `osita-sunday-nnyigide/Pras_Server`
- License: MIT

The redesign work in `src/protrepair/` was derived from and scientifically validated
against the upstream PRAS implementation during migration. The legacy source
trees have been removed from the repository working tree; provenance now lives
in git history and the upstream project.

## FASPR

- Upstream project: `tommyhuangthu/FASPR`
- Local vendored path: `vendor/faspr/`
- License: MIT
- Source notice: [vendor/faspr/LICENSE](vendor/faspr/LICENSE)
- Provenance record: [vendor/faspr/PROVENANCE.md](vendor/faspr/PROVENANCE.md)
- Upstream README snapshot: [vendor/faspr/README.upstream.md](vendor/faspr/README.upstream.md)

This repository vendors the FASPR source code and rotamer library asset for the
optional packaged side-chain packing backend. The upstream LICENSE is MIT; the
upstream README includes academic-user wording, which this repository preserves
as provenance and notice text.

## Structure-data fixtures

Some test fixtures and migration inputs are derived from Protein Data Bank
structure files. Those coordinate-data files are not covered by the repository
MIT code license.

- Representative local paths include:
  - `tests/fixtures/pdb/`
  - `tests/fixtures/corpus/`
- Upstream data source: wwPDB / RCSB PDB archive
- Data policy: CC0 1.0
- Reference: <https://www.rcsb.org/pages/usage-policy>

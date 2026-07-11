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

## RDKit

- Runtime dependency: `rdkit`
- PyPI package: `rdkit`
- Upstream project: `rdkit/rdkit`
- RDKit source license: BSD 3-Clause
- PyPI package license: BSD 3-Clause
- Bundled snapshot license text: [vendor/rdkit/LICENSE](vendor/rdkit/LICENSE)

ProtRepair uses RDKit as a required runtime dependency for retained-ligand
chemistry and local refinement. It also includes a static element-radius
snapshot in `src/protrepair/chemistry/radii.py` generated from
`rdkit==2026.3.2` (`rdBase.rdkitVersion=2026.03.2`) via
`rdkit.Chem.GetPeriodicTable().GetRvdw` and `GetRcovalent` for atomic numbers
1 through 118. The snapshot is committed so diagnostics do not import RDKit at
radius-lookup time.

The pinned RDKit source attributes covalent radii to Cordero et al.
(DOI `10.1039/B801115J`) and states that `1.9` is used when a covalent radius is
unknown. It attributes van der Waals radii to BODR v10.1 and explicitly uses
`2.0` for listed superheavy elements missing BODR values. ProtRepair preserves
those upstream source-default distinctions in its radius resolution metadata.
The complete RDKit BSD 3-Clause copyright notice, conditions, and disclaimer
are reproduced in `vendor/rdkit/LICENSE` and included in release artifacts.

## FASPR

- Upstream project: `tommyhuangthu/FASPR`
- Local vendored path: `vendor/faspr/`
- Source code license: MIT
- Source notice: [vendor/faspr/LICENSE](vendor/faspr/LICENSE)
- Provenance record: [vendor/faspr/PROVENANCE.md](vendor/faspr/PROVENANCE.md)
- Upstream README snapshot: [vendor/faspr/README.upstream.md](vendor/faspr/README.upstream.md)
- Bundled rotamer-library notice:
  [vendor/faspr/DUNBRACK_ROTAMER_LIBRARY_NOTICE.md](vendor/faspr/DUNBRACK_ROTAMER_LIBRARY_NOTICE.md)

This repository vendors the FASPR source code and rotamer library asset for the
optional packaged side-chain packing backend. The upstream LICENSE is MIT; the
upstream README includes academic-user wording, which this repository preserves
as provenance and notice text.

The bundled `dun2010bbdep.bin` rotamer-library asset is data from the Dunbrack
Lab Smooth Backbone-Dependent Rotamer Library 2010. The Dunbrack source page
states that the library is distributed under Creative Commons Attribution 4.0
International (CC BY 4.0) and requires source acknowledgement.

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

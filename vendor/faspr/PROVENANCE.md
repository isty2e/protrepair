# FASPR Provenance

ProtRepair vendors FASPR as the native side-chain packing executable used by
the optional packaged FASPR backend.

## Upstream

- Project: `tommyhuangthu/FASPR`
- Repository: <https://github.com/tommyhuangthu/FASPR>
- Snapshot commit: `0d55732fd6307f373018c6bddd842291c355c5f7`
- Snapshot verification date: 2026-07-06
- Local vendored path: `vendor/faspr/`

The vendored `src/` tree, `LICENSE`, `README.upstream.md`, and
`dun2010bbdep.bin` match the upstream snapshot above. The upstream `README.md`
is stored locally as `README.upstream.md` to keep its origin explicit.

Upstream files intentionally not vendored:

- `build.sh`
- `example/`

## License And Notice

FASPR is vendored under the MIT license supplied in
[`vendor/faspr/LICENSE`](LICENSE). The upstream source headers also carry the
MIT permission notice.

The upstream README contains academic-user wording. ProtRepair preserves that
wording in [`vendor/faspr/README.upstream.md`](README.upstream.md) as provenance
and notice text. The vendored code is treated under the supplied MIT license.

The bundled `dun2010bbdep.bin` rotamer-library asset is data from the Dunbrack
Lab Smooth Backbone-Dependent Rotamer Library 2010. The source page states that
the 2010 backbone-dependent rotamer library is distributed under Creative
Commons Attribution 4.0 International (CC BY 4.0), is free for academia and
industry, and requires source acknowledgement. ProtRepair records that
attribution in
[`vendor/faspr/DUNBRACK_ROTAMER_LIBRARY_NOTICE.md`](DUNBRACK_ROTAMER_LIBRARY_NOTICE.md).

## Local Build

ProtRepair builds the vendored C++ sources with the repository root
`CMakeLists.txt` and installs the resulting executable alongside
`dun2010bbdep.bin` in the package FASPR asset directory.

Current local build configuration:

- Release build type from `tool.scikit-build.cmake.build-type = "Release"`
- `PROTREPAIR_FASPR_ENABLE_FAST_MATH=OFF` by default
- If `PROTREPAIR_FASPR_ENABLE_FAST_MATH=ON`, non-MSVC C++ builds add
  `-ffast-math`

Fast-math may improve native FASPR speed or match upstream build suggestions,
but it can weaken floating-point reproducibility. Packaged builds therefore use
the safer default unless explicitly overridden at CMake configuration time.

No local source patches are applied to the vendored FASPR source files.

## Checksums

SHA-256 checksums recorded on 2026-07-06:

- `vendor/faspr/LICENSE`:
  `de1d50743d73db2567b0688e7d5e156f17055234c3d6b15fea13453ccb2a9300`
- `vendor/faspr/README.upstream.md`:
  `20ebfbe9e6227abb29a0d4bd09ff0e2fbb70c275294758b8399a02caad27e1be`
- `vendor/faspr/dun2010bbdep.bin`:
  `ed3f7be5f33b5fa947ac5e83cb024c6a6af6440bb50a1c8073aacabe6d792d0e`
- `vendor/faspr/DUNBRACK_ROTAMER_LIBRARY_NOTICE.md`: tracked notice file for
  the CC BY 4.0 rotamer-library asset
- `vendor/faspr/src/` sorted file manifest:
  `cc2ca78f30069e0e9308a0b68674da00864fa1ca09386128b96b36148a3c09d7`
- `vendor/faspr/` shipped upstream core manifest
  (`LICENSE`, `README.upstream.md`, `dun2010bbdep.bin`, `src/`):
  `1931fca1e60f3cfcbc2e651e38cee20fae0d770242a431068e65de18140f5322`

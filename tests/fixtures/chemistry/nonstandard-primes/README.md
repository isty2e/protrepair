# Prime-Containing CCD Components

These four unmodified Chemical Component Dictionary (CCD) files are the source
references for `tests/unit/test_nonstandard_prime_assets.py`. They cover quoted
atom identifiers, primed/unprimed name collisions, hydrogen parents, and geometry
restraints. Tests read these local copies without network access.

| File | CCD modified date | Source |
| --- | --- | --- |
| `HRG.cif` | 2024-09-27 | [RCSB HRG](https://files.rcsb.org/ligands/download/HRG.cif) |
| `60F.cif` | 2024-09-27 | [RCSB 60F](https://files.rcsb.org/ligands/download/60F.cif) |
| `IEL.cif` | 2011-06-04 | [RCSB IEL](https://files.rcsb.org/ligands/download/IEL.cif) |
| `YPZ.cif` | 2024-09-27 | [RCSB YPZ](https://files.rcsb.org/ligands/download/YPZ.cif) |

SHA-256 hashes of the frozen files:

```text
ee7edc8018b1c1867306494c404ddc110a862c49db16ef8e49e10726170b25ce  HRG.cif
bbaedf1ccc4fa80f1176187da5d9a041e28923ece36611e359e8b99c6902ea48  60F.cif
c7f1615674e197005d013379619bc3b913e58f19cba9bb00c7368754757a2828  IEL.cif
5aa70038d63cf7b30c77ea9c809348eb4e17af317761c207313ae96e51c17c19  YPZ.cif
```

The bundled entries preserve the decoded atom names, including primes, and use
the component reader's existing restraint derivations. Updating these records
does not add component-specific repair support. The files above pin the source
data independently of later CCD revisions.

These files are archive data under [CC0 1.0](https://www.rcsb.org/pages/usage-policy),
not code under the repository's MIT license.

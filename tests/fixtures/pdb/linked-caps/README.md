These two PDBs are OpenMM 8.4.0 exports of the ACE-Gly-Ala-NME example in
[issue #86](https://github.com/isty2e/protrepair/issues/86). OpenMM read
the input with `PDBFile` and wrote it with `PDBFile.writeFile(keepIds=True)`.
For the populated fixture, `Modeller.addHydrogens` used `amber14-all.xml`
on the Reference platform before writing. Only trailing blanks were
removed from the PDB records. The fixtures do not require OpenMM at test time.

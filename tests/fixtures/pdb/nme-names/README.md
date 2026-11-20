This PDB is an AmberTools 26.0 LEaP export of N-terminal glycine with a
C-terminal NME cap using the legacy ff94 names. Only trailing blanks were
removed from the PDB records. AmberTools and its force-field files are not
included or required to run the tests.

```text
source oldff/leaprc.ff94
capped = sequence { NGLY NME }
savePdb capped amber-ff94-nme.pdb
quit
```

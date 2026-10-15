# Changelog

## Unreleased

### Changed

- require selected chemistry to explain external covalent-like and untyped
  attachments before retained H/topology preparation. Unsupported connected
  components, including glycans without a contextual model, remain unchanged
  with diagnostics rather than receiving an isolated H inventory
- include sulfate chemistry in the retained-component bundle, using the common
  template preparation path
- use local polar contacts by default to choose between source-compatible
  neutral HIS tautomers; epsilon remains the fallback when there is no
  preference. This changes some prepared H positions, not the original
  evidence, and does not estimate pKa
- retain applied default microstate choices separately from explicit overrides;
  defaults remain source-compatible preferences, not replacement authority
- retain fitted spans with residue-local reference outliers and report remaining
  geometry warnings on the selected result, after any follow-up repair
- refuse fixed-template aromatic inputs whose reported orders use a different
  Kekule form, leaving the ligand unchanged. This conservative representation
  limit can reject chemically equivalent inputs; see the retained-ligand policy

### Fixed

- validate both ends of retained covalent preparation: refuse caps on already
  occupied backbone nitrogens and disulfides with charged or multiply attached
  sulfur partners, without changing the source structure
- reuse a topology-local inter-residue bond index instead of scanning the whole
  graph for each retained component's connection checks
- reserve acyl-cap diagnostics for carbonyl C--polymer backbone N candidates;
  other unsupported aldehyde attachments receive the common relationship diagnostic
- bind deposited neutral carbonyl C--polymer N caps through shared contextual
  chemistry, publishing the single bond before polymer N preparation while
  preserving source atoms, H/D/T, charges and link metadata
- preserve tetrahedral stereochemistry during RDKit retained-ligand H placement
  by correcting only newly generated H, without moving original atoms; reject
  confirmed pose conflicts with mapped SMILES in preparation and readiness
- choose a corrected H's side from the declared stereo and fixed-neighbor plane,
  rather than a near-planar center's unassigned signed displacement
- recheck native configuration after retained H/D/T correspondence; this does not
  independently validate each source H's geometry. Keep their coordinates
  fixed during subsequent H clash cleanup, including with chemistry overrides
- retain input ligand H/D/T custody through the final `add_hydrogens` cleanup,
  including after retained prerequisites or rejected local refinement. Explicit
  FF motion is unchanged and its resulting positions are not restored
- report unresolved rotatable-H clashes even when every target is fixed, and
  retain relevant residue-pair warnings in scoped cleanup calls
- check existing external covalent-like and untyped bonds across all provenances
  before preparing an isolated zero-H component; preserve typed noncovalent context
- preserve full retained-component inventories, including SAH/SAM OXT, separately
  from polymer scaffold projections; restore their H placement, topology readiness
  and SAM OXT charge requirements
- retain supported partial H preparation for charged cofactors and assign new
  template charges only at locally complete heavy-atom neighborhoods; incomplete
  components remain unready for refinement
- apply template name/declared-element checks regardless of charge declarations,
  and report scaffold/charge failures without mislabeling them as source connectivity
- publish retained-ligand formal charges with admitted bond chemistry, including
  explicit neutral SMILES assignments; preserve known source charges and reject
  conflicting or incompatible scaffolds before applying residue changes
- distinguish unspecified source charges from declared zero in template
  diagnostics and report charge-only repairs in mixed ligand batches
- distinguish declared H attachments from incompatible model-derived parent
  hints during retained-ligand repair, and complete feasible one-to-one H
  correspondences before refusing a residue. Existing distance and candidate
  limits, source atom payloads and declared-connection guards are unchanged
- report graph-only retained-ligand repairs even when another ligand receives
  hydrogens in the same call, without treating shifted atom slots as bond changes
- match retained H/D/T identities within compatible parent groups before applying
  correspondence limits. Unmatched source hydrogens or an exceeded limit leave
  the residue unchanged with a diagnostic rather than silently replacing them
- resolve retained-ligand bond chemistry while preserving source connectivity
  metadata and explicitly reported attributes. Complete hydrogen inventories can
  receive topology-only repair without atom replacement or hydrogen clash
  cleanup; conflicting source connections leave the ligand unchanged
- restore primed atom coordinates and restraints in the bundled FAD, FMN, SAH and
  SAM templates. FAD hydrogen completion now covers the full template inventory
- preserve retained-ligand source hydrogens when an incomplete template cannot
  reconcile them. Report unsupported placement without committing partial atoms
  or moving the failed residue during hydrogen clash cleanup
- bind shared chemistry observations to the current structure, component library,
  ligand evidence and fallback policy, so readiness and local refinement cannot
  reuse hydrogen expectations from incompatible inputs
- reject supplied residue scaffolds that differ from their context instead of
  silently reporting facts for another atom inventory or component
- use aliphatic LYS dictionary geometry for reconstructed CD-CE and CE-NZ
  bonds and the CD-CE-NZ angle, without changing the fallback torsions
- preserve connected materialized atoms during partial standard-residue completion;
  missing branches follow available anchors, with ring closure where dictionary
  targets are available. Partial HIS rings have no state-independent closure target;
  impossible anchored closures leave atoms absent with a placement warning
- preserve supported donor sidechain charges and bond orders during span grafting,
  adopting donor choices as preferences rather than transferring override authority
- bind HIS ring restraints to the supported realized microstate in geometry
  diagnostics and FF construction, rather than applying one protonation table
- resolve preserved polymer H/D/T attachments without replacing their identity,
  geometry or source evidence, including through a topology-only workflow action
- reconsider stale default microstate choices after site or linkage changes;
  incompatible explicit overrides still require reconciliation
- screen individual unrestrained bond angles with the broad geometry window,
  including unresolved HIS rings, without assigning a protonation state
- preserve retained non-polymers on chains with no polymer residues
- decode quoted CCD atom identifiers without stripping meaningful primes, and
  correct the corresponding HRG, 60F, IEL and YPZ bundled component records
- reject component dictionaries with explicit unknown charges, coordination or
  metal-pi flags, or incomplete bond columns instead of losing that chemistry
- preserve resolved disulfide relationships through PDB SSBOND and CONECT output,
  independently of bond provenance; mmCIF retains the canonical relationship
- recover peptide connectivity across nonmonotonic author residue numbering
  from source sequence evidence and corroborating geometry. Explicit gaps,
  boundary chemistry, and source connections still constrain inference
- reject donor-span requests whose declared anchors disagree with post-insertion
  chain neighbors, or whose source carbonyl already has an additional covalent
  attachment or terminal OXT. Internal and one-anchor spans fail without changing
  the source rather than adding unvalidated peptide bonds
- reject donor spans with missing or unresolved peptide connections, and reject
  insertion across an existing source anchor-to-anchor C-N bond without deleting
  that bond. Both paths report an atomic span failure
- apply polymer H counts, charges, and bond orders together in direct and workflow
  preparation, including H-complete structures with inconsistent chemistry.
  Histidine ratio requests now produce a complete cationic ring graph
- preserve original H names and isotopes, and keep ligand edits made by earlier
  workflow actions. Rebuilding H after heavy movement preserves identity without
  restoring stale coordinates
- add the C-OXT bond when completing a free C terminus, and require realized
  chemistry or explicit calculation-only peptide support throughout the FF region
- close ordinary local peptide cuts with fixed source-backed or compact synthetic
  caps. They preserve boundary valence without declaring biological free termini,
  changing source chemistry, or exporting support atoms
- preserve canonical H parents during FF bond planning even when distorted
  coordinates place another heavy atom closer. Backbone microstate resolution
  checks its own scaffold rather than requiring an unrelated side chain
- score parser defects even with partial H coverage and recheck previously
  satisfied goals against the returned structure
- retain original charge, hydrogen/isotope, coordinate, and explicit-connection
  observations across repair and canonical reprocessing, separately from the
  current topology. Template-filled connection types remain distinguishable
  from source declarations; these observations do not select a protonation state
- preserve explicit zero formal charges rather than conflating them with
  unspecified charges at PDB/mmCIF ingress and egress, including alternate-location
  and duplicate-atom selection
- stop copying the first atom's formal charge onto newly placed atoms during
  heavy-atom and hydrogen completion. Existing atom charges and charges
  explicitly assigned to new atoms are preserved
- retain explicitly reported source bond orders separately from template-filled
  topology orders, including through atom repair and hydrogen placement. This
  preserves evidence for later chemistry decisions without changing the graph
  used by writers or force-field binding
- assign Kekule bond orders to standard PHE, TYR, and TRP rings so RDKit
  recognizes their aromatic chemistry instead of saturated radical centers.
  Source charges and hydrogens are unchanged; explicit source orders still
  override the defaults
- preserve bond-order evidence through PDB CONECT and mmCIF `struct_conn`
  roundtrips, including intra-residue and repaired bonds. Connectivity-only
  records no longer erase known component orders; explicit source orders
  survive template and force-field projection
- give standard backbone and ASN/GLN side-chain carbonyls their double bonds
  in the component library, so force-field binding no longer treats them as
  saturated radical centers. This changes some local refinement coordinates;
  protonation-dependent bond orders and charges are not resolved by this fix
- constrain each donor-span anchor separately when joint least-squares fitting
  leaves one outside its tolerance. Existing successful fits and the geometry
  checks after fitting are unchanged; no optimizer dependency is added
- retry unsuccessful donor-span closure with joint donor pose/torsion fitting
  against both fixed source anchors and their actual peptide junctions, without
  deforming donor bonds or angles. Each anchor must meet the existing tolerance;
  repair details report the maximum anchor RMSD and total fitting iterations
- keep explicit refinement requests usable alongside span reconstruction and
  atom completion; planning no longer tries to bind missing atoms before earlier
  transformations can create them. Unresolved scopes produce refinement diagnostics
  instead of aborting the workflow
- preserve cross-residue covalent geometry during donor-backed span closure;
  Pro ring bonds and fixed source carbonyl planes no longer act as independent
  torsions. Amide-N substituents are checked before insertion. CCD chooses its
  initial sweep from measured endpoint descent and retains deterministic
  retries without moving source atoms

### Breaking changes

- allow `TopologyBond.order` to be `None` for unresolved source connectivity,
  instead of reporting an unsupported single-bond assumption. Contradictory
  explicit orders are rejected; unresolved covalent orders cannot be bound
  to RDKit as single bonds
- normalize `ExternalSpanReconstructionSpec` around a canonical
  `AbsentResidueSpanScope`; remove the source-specific
  `blueprint_coverage_gap` field, the behavior-free `supporting_role` field,
  and the duplicate transformer scope

### Added

- bundle standalone SAN and CMP chemistry and the complete isolated ACE
  dictionary, without adding component-ID-specific preparation branches
- export prepared canonical structures directly to a sanitized RDKit molecule,
  keeping explicit H/D/T, assigned charges, typed coordination and full-precision
  coordinates without PDB parsing. Refuse unresolved or reinterpreted chemistry;
  native representation does not imply force-field support
- retain typed source coordination declarations to policy-excluded partners and
  report non-blocking omission warnings beside retained metals, without restoring
  excluded atoms or inferring metal chemistry
- add a pure standard-polymer microstate resolver that keeps H counts,
  formal charges, and bond orders coupled. It preserves source evidence and
  distinguishes ambiguity, conflicts, and unsupported chemistry
- retain explicitly applied polymer microstate choices separately from original
  observations, and add a snapshot-bound check of current H/charge/bond realization
- add marked free-terminal assumptions to the internal PRAS preparation policy,
  without replacing current peptide connections or contrary original chemistry
- generate graph-constrained H coordinates for internal polymer microstate
  application, with separate chemistry-only preservation and explicit H
  rebuilding
- reconstruct explicitly mapped missing polymer spans from external donor
  structures with anchor-frame placement, bounded donor-seeded CCD closure,
  chemistry and stereochemistry gates, atomic topology updates, and structured
  failure reporting
- export observed-chain sequence alignment helpers from `protrepair.sources`
  for building AlphaFold-backed reconstruction requests
- export `PackingMode`, `PackingScope`, and `PackingSpec` from
  `protrepair.workflow.contracts` so callers can configure side-chain packing
  without importing an internal transformer module
- reject OXT goals unless they target the C-terminal side of a
  `ResidueBoundaryScope`

### Documentation

- organize the documentation around tutorial, how-to, explanation, and
  reference needs; add an executable getting-started path, common repair
  recipes, workflow concepts, and a supported public API reference

## 0.2.0 - 2026-07-27

### Breaking changes

- require RDKit at runtime through the `rdkit` package and remove the former
  `refinement` extra and no-RDKit execution mode
- replace the partial public Bondi/Cordero radius tables with a bundled RDKit PeriodicTable radius snapshot
  of `GetRvdw` and `GetRcovalent` values for elements 1-118
- remove `BONDI_VAN_DER_WAALS_RADII_ANGSTROM`,
  `CORDERO_COVALENT_RADII_ANGSTROM`, `DEFAULT_COVALENT_RADIUS_ANGSTROM`, and
  `DEFAULT_VAN_DER_WAALS_RADIUS_ANGSTROM`; radius lookup now raises
  `UnknownElementRadiusError` instead of silently using a default for unsupported
  elements
- make low-level clash bases and contexts factory-only topology-bound artifacts;
  remove their low-level construction and query functions from the diagnostics
  facade; remove the responsibility-free intermediate frame type and binding
  surface; and remove the redundant `constitution_address_space_key` projection

### Changed

- move external CCD and monomer-CIF file parsing to the I/O boundary while
  keeping normalized nonstandard-component registry projections in chemistry
- separate source-format connection parsing from canonical structure
  normalization through one internal source-connection contract, while
  preserving LINK/struct_conn precedence over fallback PDB CONECT records
- classify HET-encoded peptide-linking modifications by declared entity,
  supported polymer chemistry, and chain/linkage context at ingress so canonical
  polymer diagnostics and topology include them without absorbing free ligands
- reuse workflow branch evaluations across sibling selection, regression
  retention, budget pruning, and terminal reporting while evicting discarded
  branch results from the runtime cache
- reuse topology/radius preparation and one focus-aware spatial index across
  parser-witness clash and near-covalent candidate assessments while keeping
  their metric policies and radius systems independent
- restrict local-refinement and targeted-correction stereochemistry diagnostics
  to directly selected residue slots instead of scanning the whole polymer
  structure
- reuse one serialization-compatible no-CONECT PDB projector across
  local-refinement and dependent-hydrogen parser assessments, patching only
  changed coordinates while falling back on any unpatched fact drift
- batch residue geometry and stereochemistry facet updates and reuse one
  rotation matrix per residue-local transform instead of rebuilding immutable
  structures or matrices for each atom
- consolidate retained-ligand RDKit hydrogen patch projection while rejecting
  invalid generated names and non-finite coordinates before materialization
- centralize NumPy geometry-placement vector degeneracy rules while preserving
  domain-specific error and diagnostic semantics, and reject non-finite or
  overflowing placement frames before they emit invalid coordinates
- report recoverable heavy-atom, terminal OXT, and polymer-hydrogen geometry
  placement failures as atom-scoped validation issues while preserving any
  successfully placed partial repair
- harden ingress normalization by selecting coherent residue-level altloc
  cohorts by mean occupancy with lexical tie-breaks, validating non-finite atom
  coordinates/occupancy/B-factors at the boundary, preserving first-model PDB
  CONECT records safely, and dropping source connectivity that refers to
  discarded altloc or source component variants
- tighten source-bond metadata validation so public topology metadata requires
  finite positive numeric reported distances rather than float-like strings
- expose unsupported topology as an explicit readiness state for polymer
  components whose expected topology cannot be resolved from supported chemistry
  evidence, including separate absent-vs-unsupported workflow deficit fields
- wrap public structure parsing and unsupported-format failures in project-owned
  errors so callers can catch `ProtrepairError` subclasses consistently
- model thiol H/D/T atoms on canonical CYS disulfides as typed chemistry
  contradictions, remove them through planner-selected topology-preserving
  normalization, and report chemistry work under the broader
  `CHEMISTRY_NORMALIZATION` workflow phase

### Documentation

- refocus the README on installation and first use while moving exact ingress,
  retained-ligand, histidine, analysis, and FASPR contracts to domain-owned
  reference documents

## 0.1.0

Initial public release candidate of the rewritten `protrepair` package.

- expose a small release-facing API centered on `process_structure()`,
  `StructureIngressOptions`, `requested_process_goal()`, and typed workflow
  result contracts
- support canonical PDB/mmCIF ingress and egress through `gemmi`, including
  topology-backed connection emission
- model structure constitution, geometry, topology, provenance, semantic
  scopes, and observed state axes as typed canonical domain objects
- implement workflow planning over explicit heavy-atom completion, hydrogen
  completion, retained non-polymer hydrogenation, stereochemistry correction,
  side-chain packing, and local refinement actions
- support standard residues, bundled polymer-like nonstandard components, and
  bundled retained non-polymer/cofactor component templates
- support RDKit-backed local refinement and parser-readability diagnostics when
  the optional `refinement` dependency is installed
- package the vendored FASPR source and rotamer library for the optional
  side-chain packing backend
- include representative workflow regressions, fixture-backed refinement
  benchmarks, public API smoke tests, release-facing README examples, typed
  package metadata, and third-party licensing notices

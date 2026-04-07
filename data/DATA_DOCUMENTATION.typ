// Data Documentation — Physical Property Modeling Using NMR Spectra
// Compiled: April 7, 2026

#set page(margin: (x: 1in, y: 1in))
#set text(font: "New Computer Modern", size: 11pt)
#set heading(numbering: "1.1")
#set par(justify: true)

#align(center)[
  #text(size: 18pt, weight: "bold")[Data Documentation]
  #v(4pt)
  #text(size: 14pt)[Physical Property Modeling Using NMR Spectra\ and Functional Group Representation]
  #v(8pt)
  #text(size: 11pt, style: "italic")[J.C. Vaught]
  #v(4pt)
  #text(size: 10pt)[Compiled April 7, 2026]
]

#v(12pt)

= Overview

This document describes the datasets collected for training machine learning models that predict thermophysical properties (viscosity, surface tension, density, and others) of organic compounds and fuel mixtures from NMR spectral features, functional group representations, and molecular descriptors. The data was aggregated from public databases, government reports, journal supplementary information, and open-source Python libraries.

The original repository by Burnett predicted viscosity and surface tension for ~65 hydrocarbon compounds using simulated 1H NMR spectra and 7 functional group counts. This expanded dataset provides over 1.2 million experimental data points across thousands of compounds, paired with multiple feature representations.

= Directory Structure

The `data/` directory is organized into five top-level categories.

#table(
  columns: (1fr, 2.5fr),
  stroke: 0.5pt,
  inset: 6pt,
  fill: (col, row) => if row == 0 { rgb("#73000A").lighten(85%) },
  [*Directory*], [*Contents*],
  [`data/nmr/`], [NMR spectral data (experimental and database dumps)],
  [`data/thermo/`], [Thermophysical property data (experimental, correlated, estimated)],
  [`data/features/`], [Precomputed molecular fingerprints and descriptors],
  [`data/fuels/`], [Aviation fuel-specific data and extracted paper tables],
  [`data/papers/`], [Source PDFs and supplementary information files],
)

= NMR Spectral Data (`data/nmr/`)

== NMRexp (`data/nmr/nmrexp/`)

#table(
  columns: (1fr, 2.5fr),
  stroke: 0.5pt,
  inset: 6pt,
  [*File*], [`NMRexp_1H_13C.parquet` (631 MB)],
  [*Source*], [Zenodo: `10.5281/zenodo.17296666`],
  [*Reference*], [Scientific Data, 2025. "NMRexp: A database of 3.3 million experimental NMR spectra"],
  [*License*], [CC-BY-4.0],
  [*Contents*], [3,372,987 experimental NMR spectra extracted from published literature],
  [*Nuclei*], [1,667,135 #super[1]H; 1,455,670 #super[13]C; 197,186 #super[19]F; 33,230 #super[31]P; others],
  [*Columns*], [`SMILES`, `NMR_type`, `NMR_frequency`, `NMR_solvent`, `NMR_processed` (peak list with shift, multiplicity, J-coupling)],
  [*Format*], [Apache Parquet, loadable with `pandas.read_parquet()`],
)

Each row represents one compound--spectrum pair. The `NMR_processed` column contains a list of tuples `(chemical_shift_ppm, multiplicity, J_coupling_Hz)`. Spectra are peak lists, not continuous intensity arrays. To use with a CNN, Lorentzian broadening must be applied to reconstruct continuous spectra.

== NMRShiftDB2 (`data/nmr/nmrshiftdb/`)

Source: SourceForge (`sourceforge.net/projects/nmrshiftdb2/files/data/`). NMRShiftDB2 is an open-access, peer-reviewed NMR database maintained as part of the NFDI4Chem initiative. License: open content.

#table(
  columns: (2fr, 1fr, 2fr),
  stroke: 0.5pt,
  inset: 6pt,
  fill: (col, row) => if row == 0 { rgb("#73000A").lighten(85%) },
  [*File*], [*Size*], [*Contents*],
  [`nmrshiftdb2.nmredata.sd`], [271 MB], [64,723 compounds in NMReDATA format with full NMR assignments],
  [`nmrshiftdb2withsignals.sd`], [151 MB], [53,072 compounds with explicit NMR signal data],
  [`nmrshiftdb2_3d.sd`], [75 MB], [40,699 compounds with 3D molecular coordinates],
  [`nmrshiftdb2withsignals_3d.sd`], [90 MB], [40,699 compounds with 3D coordinates and NMR signals],
  [`nmrshiftdb.sql.gz`], [1.1 GB], [Complete PostgreSQL database dump (all tables, metadata, assignments)],
)

The NMReDATA files are SDF format with NMR data fields (chemical shifts, multiplicities, J-couplings, atom assignments). Parseable with RDKit's `ForwardSDMolSupplier`. The 3D files include optimized molecular geometries suitable for graph neural network features.

== PubChem Identifiers (`data/nmr/pubchem/`)

#table(
  columns: (1fr, 2.5fr),
  stroke: 0.5pt,
  inset: 6pt,
  [*File*], [`CID-SMILES.gz` (1.4 GB compressed)],
  [*Source*], [NCBI FTP: `ftp.ncbi.nlm.nih.gov/pubchem/Compound/Extras/`],
  [*Contents*], [PubChem Compound ID to SMILES mapping for all PubChem compounds],
  [*Purpose*], [Identifier resolution: link CAS numbers or names to canonical SMILES for featurization],
)

= Thermophysical Property Data (`data/thermo/`)

== NIST ThermoML Archive (`data/thermo/thermoml/`)

#table(
  columns: (1fr, 2.5fr),
  stroke: 0.5pt,
  inset: 6pt,
  [*File*], [`ThermoML.v2020-09-30.tgz` (181 MB, expands to ~3.9 GB)],
  [*Source*], [NIST: `data.nist.gov/od/id/mds2-2422`],
  [*Contents*], [23,846 JSON files from 5 journals (J. Chem. Eng. Data, J. Chem. Thermodyn., Fluid Phase Equilibria, Thermochimica Acta, Int. J. Thermophysics), 2003--2019],
  [*License*], [Public domain (US Government work)],
)

=== Parsed ThermoML (`data/thermo/thermoml_parsed/`)

The raw archive was parsed to extract viscosity, surface tension, and density data for both pure compounds and mixtures.

#table(
  columns: (2fr, 1fr, 1fr),
  stroke: 0.5pt,
  inset: 6pt,
  fill: (col, row) => if row == 0 { rgb("#73000A").lighten(85%) },
  [*File*], [*Data Points*], [*Type*],
  [`pure_viscosity.csv`], [44,440], [Pure compounds],
  [`pure_surface_tension.csv`], [9,175], [Pure compounds],
  [`pure_density.csv`], [186,394], [Pure compounds],
  [`mixture_viscosity.csv`], [223,807], [Binary/ternary mixtures],
  [`mixture_surface_tension.csv`], [38,294], [Binary/ternary mixtures],
  [`mixture_density.csv`], [659,550], [Binary/ternary mixtures],
)

Total: 1,161,660 experimental data points. Compounds are identified by InChI and InChIKey (CAS numbers are largely absent from this archive). Mixture files include component identifiers and composition (mole fraction, mass fraction, or molality as reported in the source).

== Chemicals Library Correlations (`data/thermo/chemicals_tsv/`)

Source: `chemicals` Python package (Caleb Bell, MIT license, `github.com/CalebBell/chemicals`). These are temperature-dependent correlation coefficients from Perry's Chemical Engineers' Handbook (8th ed.), VDI Heat Atlas, and other standard references.

#table(
  columns: (2.5fr, 1fr, 1.5fr),
  stroke: 0.5pt,
  inset: 6pt,
  fill: (col, row) => if row == 0 { rgb("#73000A").lighten(85%) },
  [*File*], [*Compounds*], [*Correlation Type*],
  [`Table 2-313 Viscosity...Liquids.tsv`], [337], [DIPPR Eq. 101 (Perry's)],
  [`VDI PPDS Dynamic viscosity...liquids.tsv`], [271], [VDI-PPDS polynomials],
  [`Viswanath Natarajan Dynamic 3 term.tsv`], [432], [3-term Viswanath-Natarajan],
  [`Dutt Prasad 3 term.tsv`], [100], [3-term Dutt-Prasad],
  [`Jasper-Lange.tsv`], [522], [Linear surface tension],
  [`VDI PPDS surface tensions.tsv`], [272], [VDI-PPDS polynomials],
  [`MuleroCachadinaParameters.tsv`], [115], [Mulero-Cachadina],
)

Total unique compounds: ~1,042 with viscosity data, ~690 with surface tension data. All indexed by CAS number.

== NIST WebBook Reference Fluids (`data/thermo/nist_webbook/`)

Source: NIST Chemistry WebBook SRD 69 (`webbook.nist.gov/chemistry/fluid/`). High-accuracy equation-of-state data for 74 reference fluids along the liquid saturation curve.

Each fluid has a CSV file with columns: `Temperature_K`, `Viscosity_uPas`, `SurfaceTension_Nm`, `Density_kgm3`. Approximately 600 data points per fluid (1 K increments from triple point to critical point). A master index is at `nist_webbook_all.csv`. 73 fluids have viscosity data; all 74 have surface tension.

== NIST ILThermo (`data/thermo/ilthermo/`)

Source: NIST Ionic Liquids Thermodynamics Database (`ilthermo.boulder.nist.gov`). Scraped via REST API with 0.3 s delay between requests.

#table(
  columns: (1fr, 1fr, 1fr),
  stroke: 0.5pt,
  inset: 6pt,
  fill: (col, row) => if row == 0 { rgb("#73000A").lighten(85%) },
  [*Property*], [*Compounds*], [*Data Points*],
  [Viscosity], [1,345], [26,189],
  [Surface tension], [506], [6,875],
)

Total: 1,456 unique ionic liquid compounds. Each compound has its own CSV file in `viscosity/` or `surface_tension/` with columns: `compound_name`, `formula`, `temperature_K`, `pressure_kPa`, and the property value in mPa$dot$s or mN/m. A `master_index.csv` lists all compounds with data availability flags.

== Density Data (`data/thermo/density/`)

Compiled from two sources.

- `thermo_liquid_density.csv`: 441 compounds, 8,718 data points generated from the `thermo` Python library using Perry's DIPPR Eq. 105 and VDI-PPDS correlations. Columns: CAS, name, SMILES, T_K, density_kgm3.
- `nist_webbook_density.csv`: 74 reference fluids, 44,353 data points compiled from the NIST WebBook CSVs.
- 8 additional TSV files with raw correlation coefficients (COSTALD, Perry 105, VDI-PPDS, CRC).

== Group Contribution Estimates (`data/thermo/group_contribution/`)

Synthetic property labels generated using the Joback group contribution method (via `thermo.group_contribution.joback`) combined with corresponding-states methods (Letsou-Stiel for viscosity, Brock-Bird for surface tension).

- `gc_viscosity.csv` and `gc_surface_tension.csv`: 4,988 compounds sampled from the NMRexp SMILES pool. Evaluated at 5 temperatures (250, 300, 350, 400, 450 K). These are approximate values (10--30% error typical for hydrocarbons) suitable for pretraining or data augmentation, not as ground truth.
- `chemicals_viscosity_expanded.csv` and `chemicals_surface_tension_expanded.csv`: 786 viscosity and 689 surface tension compounds from the `chemicals` library, evaluated at 10 temperatures using the best available correlation for each compound.

== Additional Properties (`data/thermo/additional_properties/`)

Multi-task learning targets generated from the `thermo`/`chemicals` Python libraries.

#table(
  columns: (2fr, 1fr, 1fr),
  stroke: 0.5pt,
  inset: 6pt,
  fill: (col, row) => if row == 0 { rgb("#73000A").lighten(85%) },
  [*File*], [*Compounds*], [*Data Points*],
  [`thermal_conductivity.csv`], [9,861], [101,333],
  [`heat_capacity.csv`], [9,861], [101,361],
  [`flash_boiling_points.csv`], [12,178], [12,178 (single-point)],
)

Raw TSV correlation files from the `chemicals` package are archived in `all_tsv/`.

== CoolProp and DDB Free Data (`data/thermo/ddb_free/`)

- `coolprop_properties_vs_T.csv`: 124 fluids from CoolProp (v7.2.0, MIT license) with 7 properties at 20 temperatures each: liquid density, vapor pressure, viscosity, surface tension, thermal conductivity, heat capacity, enthalpy of vaporization.
- `coolprop_fluid_info.csv`: Critical properties, molecular weight, acentric factor, CAS numbers for all 124 fluids.
- `SETUP_TDE_Free_10.4.5.zip` (104 MB): NIST ThermoData Engine free public version installer (Windows application containing 2.2 million property values). Source: `data.nist.gov/od/ds/mds2-3179/`.
- `nist_webbook_*.csv` (6 files): Antoine parameters, phase change data, enthalpies, and thermochemical properties for 94 common C5--C20 hydrocarbons scraped from the NIST Chemistry WebBook.
- `ddb_unifac_*.csv` (3 files): UNIFAC group interaction parameters from the Dortmund Data Bank published parameters page.

= Molecular Features (`data/features/`)

Precomputed molecular descriptors for 1,225 compounds (all unique CAS numbers from the `chemicals` viscosity and surface tension TSV files with valid SMILES resolved via `chemicals.identifiers.search_chemical`).

#table(
  columns: (2fr, 1fr, 1.5fr),
  stroke: 0.5pt,
  inset: 6pt,
  fill: (col, row) => if row == 0 { rgb("#73000A").lighten(85%) },
  [*File*], [*Dimensions*], [*Description*],
  [`compound_smiles_index.csv`], [1225 #sym.times 4], [CAS, Name, SMILES, Formula],
  [`morgan_r2_2048.csv`], [1225 #sym.times 2050], [Morgan/ECFP4 fingerprint (radius 2, 2048 bits)],
  [`morgan_r3_2048.csv`], [1225 #sym.times 2050], [Morgan/ECFP6 fingerprint (radius 3, 2048 bits)],
  [`maccs_166.csv`], [1225 #sym.times 168], [MACCS structural keys (166 bits)],
  [`rdkit_descriptors.csv`], [1225 #sym.times 219], [217 RDKit molecular descriptors (MolLogP, TPSA, MolMR, etc.)],
)

Generated with RDKit 2026.3.1.

= Aviation Fuel Data (`data/fuels/`)

== AFRL Edwards 2020 (`data/fuels/afrl_extracted/`)

Source: AFRL-RQ-WP-TR-2020-0017, "Jet Fuel Properties" by James T. Edwards, January 2020. Downloaded from DTIC (`apps.dtic.mil/sti/pdfs/AD1093317.pdf`). Approved for public release, unlimited distribution.

Contains 13 extracted CSV files covering viscosity, surface tension, density, derived cetane number, distillation, speed of sound, bulk modulus, GCxGC composition, molecular weights, critical properties, and lubricity for Category A fuels (JP-5 POSF 10289, JP-8 POSF 10264, Jet A POSF 10325) and 11+ alternative/synthetic fuels.

Limitation: most temperature-dependent property data is presented as figures in the report, not numerical tables. Extracted data includes tabulated values and linear fit equations from the text.

== Parker et al. 2025 (`data/fuels/parker2025_extracted/`)

Source: Supporting Information from Parker, Kelly, Watson-Murphy, Ghaani, and Dooley, "Predictive Modelling of Liquid Density and Surface Tension for Sustainable Aviation Fuels Using Nuclear Magnetic Resonance Atom Types," Energy and Fuels, 2025, 39(7), 3690--3702. DOI: `10.1021/acs.energyfuels.4c05601`.

#table(
  columns: (2fr, 1fr, 2fr),
  stroke: 0.5pt,
  inset: 6pt,
  fill: (col, row) => if row == 0 { rgb("#73000A").lighten(85%) },
  [*File*], [*Rows*], [*Contents*],
  [`nmr_spectra.csv`], [309,990], [Raw 1H NMR spectra (shift, intensity) for 6 aviation fuels],
  [`surface_tension_atom_types.csv`], [57], [12 HSQC-NMR atom types for surface tension model compounds],
  [`density_atom_types.csv`], [55], [12 HSQC-NMR atom types for density model compounds],
  [`training_stats.csv`], [28], [Descriptor statistics for both models],
)

The 12 atom types are: pCH3 (paraffinic methyl), pCH2, pCH, pC, cyCH2 (cycloparaffinic), cyCH, alCH3 (alkyl-aromatic), alCH2, alCH, ArCH (aromatic), ArC, rjC (ring junction). Compounds are identified by common name only (no CAS or SMILES).

== Abdul Jameel et al. 2018 (`data/fuels/jameel2018_extracted/`)

Source: Abdul Jameel et al., "Predicting Octane Number Using Nuclear Magnetic Resonance Spectroscopy and Artificial Neural Networks," Energy and Fuels, 2018. DOI: `10.1021/acs.energyfuels.8b00556`.

- `pure_compounds.csv`: 128 pure hydrocarbons (n-paraffins, isoparaffins, olefins, naphthenes, aromatics) with RON and MON values.
- `blends.csv`: 123 hydrocarbon-ethanol blends (PRF, TPRF, and multicomponent) with volume fractions and RON/MON.
- `face_gasoline_blends.csv`: 30 FACE gasoline-ethanol blends (6 gasolines #sym.times 5 ethanol levels).

The paper defines 6 functional groups from 1H NMR (paraffinic CH3, CH2, CH; olefinic; naphthenic; aromatic) but does not tabulate the NMR functional group fractions for each compound. For pure compounds, these can be computed from molecular structure using RDKit SMARTS matching.

= Source Papers (`data/papers/`)

#table(
  columns: (2fr, 3fr),
  stroke: 0.5pt,
  inset: 6pt,
  fill: (col, row) => if row == 0 { rgb("#73000A").lighten(85%) },
  [*File*], [*Description*],
  [`Parker2025_SI_002_atom_types.xlsx`], [SI Excel with NMR spectra + atom types (6.3 MB)],
  [`Parker2025_SI_001_NMR_spectra.pdf`], [SI PDF with NMR figures for 6 fuels],
  [`AbdulJameel2018_octane_NMR_ANN.pdf`], [Full paper: 281-fuel NMR + octane number dataset],
  [`AbdulJameel2016_SI.pdf`], [SI: NMR spectra figures for fuel ignition quality],
  [`jameel2021_supp.zip`], [SI: gasoline NMR methodology (3 samples only)],
)

= Python Environment

All data processing was performed with the following packages installed in a Python 3.14 virtual environment at `.venv/`:

#table(
  columns: (1fr, 1fr, 2fr),
  stroke: 0.5pt,
  inset: 6pt,
  fill: (col, row) => if row == 0 { rgb("#73000A").lighten(85%) },
  [*Package*], [*Version*], [*Purpose*],
  [`thermo`], [0.6.0], [T-dependent property evaluation from correlations],
  [`chemicals`], [1.5.1], [Raw correlation data, identifier resolution, equation implementations],
  [`rdkit`], [2026.3.1], [Molecular fingerprints, descriptors, SMARTS matching],
  [`CoolProp`], [7.2.0], [High-accuracy EOS properties for 124 fluids],
  [`pandas`], [3.0.2], [Data manipulation],
  [`pyarrow`], [23.0.1], [Parquet file I/O],
  [`scipy`], [1.17.1], [Signal processing, optimization],
)

= Data Scale Summary

#table(
  columns: (2fr, 1fr, 1fr),
  stroke: 0.5pt,
  inset: 6pt,
  fill: (col, row) => if row == 0 { rgb("#73000A").lighten(85%) },
  [*Category*], [*Compounds*], [*Data Points*],
  [NMR spectra (NMRexp)], [1,482,385 SMILES], [3,372,987],
  [NMR spectra (NMRShiftDB2)], [64,723], [database],
  [ThermoML (pure + mixture)], [thousands], [1,161,660],
  [ILThermo], [1,456], [33,064],
  [NIST WebBook], [74], [~44,000],
  [CoolProp], [124], [2,480],
  [Group contribution estimates], [4,988], [~50,000],
  [Thermal conductivity], [9,861], [101,333],
  [Heat capacity], [9,861], [101,361],
  [Molecular fingerprints], [1,225], [--],
  [Aviation fuel NMR + properties], [~300 fuels/blends], [~310,000],
)

Original repository: 65 compounds. This collection: over 1.2 million experimental property data points across thousands of compounds, with multiple feature representations and multi-task learning targets.

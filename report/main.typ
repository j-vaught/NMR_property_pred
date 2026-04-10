// =============================================================================
// NMR-Based Thermophysical Property Prediction: A Comprehensive Report
// =============================================================================

#set document(
  title: "Predicting Thermophysical Properties of Organic Compounds and Aviation Fuel Mixtures from Nuclear Magnetic Resonance Spectra",
  author: "J.C. Vaught",
  date: datetime.today(),
)

// --- Page layout ---
#set page(
  paper: "us-letter",
  margin: 0.5in,
  numbering: "1",
  number-align: center,
)

// --- Typography ---
#set text(
  font: "New Computer Modern",
  size: 11pt,
  lang: "en",
)

#set par(
  justify: true,
  leading: 0.55em,
  first-line-indent: 1.5em,
)

// --- Headings ---
#set heading(numbering: "1.1.1.")

#show heading.where(level: 1): it => {
  v(1.2em)
  text(size: 14pt, weight: "bold", it)
  v(0.6em)
}

#show heading.where(level: 2): it => {
  v(0.8em)
  text(size: 12pt, weight: "bold", it)
  v(0.4em)
}

#show heading.where(level: 3): it => {
  v(0.6em)
  text(size: 11pt, weight: "bold", it)
  v(0.3em)
}

// --- Figures ---
#show figure.caption: it => {
  text(size: 10pt, it)
}

// --- Tables ---
#set table(
  stroke: 0.5pt,
  inset: 5pt,
)

// --- Equations ---
#set math.equation(numbering: "(1)")

// =============================================================================
// Title Page
// =============================================================================

#align(center)[
  #v(0.8in)
  #text(size: 18pt, weight: "bold")[
    Predicting Thermophysical Properties of Organic Compounds and Aviation Fuel Mixtures from Nuclear Magnetic Resonance Spectra
  ]
  #v(0.5in)
  #text(size: 14pt)[J.C. Vaught]
  #v(0.2in)
  #text(size: 12pt)[#datetime.today().display("[month repr:long] [day], [year]")]
  #v(0.4in)
]

// --- Abstract ---
#heading(numbering: none)[Abstract]

Rapid prediction of thermophysical properties from molecular spectra would accelerate the qualification of sustainable aviation fuels, a process that currently requires years of physical testing per candidate pathway. This report develops a four-phase machine learning pipeline that predicts temperature-dependent viscosity and surface tension directly from proton nuclear magnetic resonance (NMR) spectra. Phase 1 establishes baselines using Morgan fingerprints and RDKit descriptors, achieving cross-validated $R^2 = 0.64$ for viscosity and $R^2 = 0.56$ for surface tension on 856 compounds. Phase 2 replaces fingerprints with 120 domain-informed features extracted from experimental 1H NMR peak lists, demonstrating that combined NMR and fingerprint representations reach hydrocarbon $R^2 = 0.69$ for viscosity and $R^2 = 0.81$ for surface tension, though NMR alone lacks molecular weight information critical for viscosity. Phase 3 extends prediction to real fuel mixtures, validating against four AFRL reference fuels (JP-5, Jet-A, HRJ-Camelina, Shell SPK, Sasol IPK, Gevo ATJ) at a baseline of 53.8% mean absolute percentage error (MAPE). Phase 4 introduces transfer learning: a multilayer perceptron pretrained on 296,000 NMR-to-descriptor mappings from the NMRexp database is fine-tuned on 943 compounds at the NMR--viscosity intersection, reducing fuel viscosity MAPE to 29.3%, a 46% relative improvement. The pipeline draws on fourteen NMR spectral databases totaling 1.8 million compounds, six thermophysical property sources exceeding one million data points, and six fuel-specific datasets. Systematic experiments with Arrhenius-structured networks, embedding extraction, cold-temperature augmentation, sample weighting, and synthetic NMR generation all failed to improve beyond the fine-tuned transfer learning result, identifying data quality at the NMR--viscosity intersection, not model architecture, as the binding constraint. All code, data pipelines, and trained model weights are publicly available.

#pagebreak()

// =============================================================================
// Introduction
// =============================================================================

#include "01-introduction.typ"

// =============================================================================
// Related Work
// =============================================================================

= Related Work <related-work>

== Classical Quantitative Structure--Property Relationships <classical-qspr>
#include "02a-related-classical-qspr.typ"

== Machine Learning for Molecular Property Prediction <ml-molecular>
#include "02b-related-ml-molecular.typ"

== Deep Learning and Spectral Methods <dl-molecular>
#include "02c-related-dl-molecular.typ"

== NMR and Machine Learning <nmr-ml>
#include "02d-related-nmr-ml.typ"

== Fuel Property Prediction from Spectroscopy <fuel-nmr>
#include "02e-related-fuel-nmr.typ"

// =============================================================================
// Data & Datasets
// =============================================================================

= Data and Datasets <data>

== NMR Spectral Databases <nmr-databases>
#include "03a-data-nmr-databases.typ"

== Thermophysical Property and Fuel-Specific Databases <thermo-databases>
#include "03b-data-thermo-fuel.typ"

== Fuel-Specific Data <fuel-specific>
#include "03c-data-fuel-specific.typ"

== Feature Engineering and Data Quality <features-quality>
#include "03d-data-features-quality.typ"

// =============================================================================
// Model Designs
// =============================================================================

= Model Designs <models>

== Phase 1: Molecular Fingerprint Baselines <model-phase1>
#include "04a-model-phase1.typ"

== Phase 2: NMR Spectrum Models <model-phase2>
#include "04b-model-phase2.typ"

== Phases 3--4: Fuel Mixture Prediction and Transfer Learning <model-phase3-4>
#include "04c-model-phase3-4.typ"

// =============================================================================
// Methodology
// =============================================================================

= Methodology <methodology>
#include "05-methodology.typ"

// =============================================================================
// Discussion
// =============================================================================

= Discussion <discussion>
#include "06-discussion.typ"

// =============================================================================
// Conclusion
// =============================================================================

= Conclusion <conclusion>
#include "07-conclusion.typ"

// =============================================================================
// Bibliography
// =============================================================================

#bibliography("references.bib", style: "ieee")

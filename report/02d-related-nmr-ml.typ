// 02d-related-nmr-ml — NMR and Machine Learning

// =============================================================================
// 2.4  NMR and Machine Learning
// =============================================================================
// The heading is provided by main.typ as:  == NMR and Machine Learning <nmr-ml>

=== First-Principles NMR Prediction <dft-nmr>

The computational prediction of NMR chemical shifts from molecular structure has
a long history in quantum chemistry and provides the foundation against which all
subsequent machine-learning approximations are benchmarked. Wolinski, Hinton, and
Pulay introduced the gauge-including atomic orbital (GIAO) method in 1990,
enabling density functional theory (DFT) level calculation of NMR shielding
tensors for isolated molecules by resolving the gauge-origin problem that arises
in finite basis set calculations @Wolinski1990GIAO. For periodic solids, Pickard
and Mauri extended first-principles NMR prediction to crystalline systems through
the gauge-including projector augmented wave (GIPAW) method, which operates under
periodic boundary conditions and has become the standard approach for solid-state
NMR calculations @Pickard2001GIPAW. Together, GIAO and GIPAW form the two
pillars of ab initio NMR prediction. However, both approaches carry substantial
computational cost. GIAO calculations scale steeply with basis set size and
molecular complexity, often requiring hours per molecule even on modern hardware.
GIPAW calculations on molecular crystals with hundreds of atoms in the unit cell
can demand days of wall time on computing clusters. This prohibitive expense
motivated the search for faster alternatives that could retain chemical accuracy
while enabling high-throughput screening of thousands or millions of compounds.


=== Machine Learning for Chemical Shift Prediction <ml-shifts>

The first generation of machine-learning models for NMR chemical shift prediction
sought to reproduce DFT-quality results at a fraction of the computational cost.
Jonas and Kuhn developed a deep neural network with quantified uncertainty
estimates that achieved root-mean-square errors of 1.43 ppm for #super[13]C and
0.28 ppm for #super[1]H, both within or near DFT accuracy, while running orders
of magnitude faster than quantum chemical methods @Jonas2019NMRPrediction. Their
model's built-in uncertainty quantification represented an important advance, as
it allowed users to assess the reliability of individual predictions rather than
relying solely on aggregate error statistics.

Paruzzo et al. introduced ShiftML, a Gaussian process regression model trained on
GIPAW-computed shifts for approximately 2,000 molecular solids selected by
farthest-point sampling from a database of over 60,000 crystal structures
@Paruzzo2018ShiftML. By encoding local chemical environments with smooth overlap
of atomic positions (SOAP) descriptors, ShiftML achieved $R^2$ values of 0.97 for
#super[1]H, 0.99 for #super[13]C, 0.99 for #super[15]N, and 0.99 for #super[17]O
chemical shifts, all within DFT accuracy, while accelerating predictions by up to
six orders of magnitude for the largest structures. The success of ShiftML
demonstrated that local atomic environment descriptors, originally developed for
interatomic potentials, could be repurposed for spectroscopic property prediction
with remarkable fidelity.

Graph-based architectures subsequently emerged as a dominant paradigm for
solution-state NMR prediction. Kwon et al. applied message passing neural
networks (MPNNs) operating on molecular graphs with implicit hydrogen
representation, demonstrating high accuracy for both #super[1]H and #super[13]C
shifts @Kwon2020MessagePassing. Yang, Chakraborty, and White extended this line
of work with a graph neural network framework that captured hydrogen-bonding-
induced downfield shifts and protein secondary structure effects, establishing
GNNs as a versatile tool for NMR prediction across small molecules and
biomolecules alike @Yang2021GNN. These graph-based models offer a conceptually
natural representation for molecules: atoms become nodes, bonds become edges, and
message passing aggregates local chemical environment information in a manner
analogous to the physical interactions that determine chemical shifts.


=== Structure Elucidation and Spectral Interpretation <nmr-structure>

While the models described above solve the _forward_ problem of predicting NMR
observables from known structures, a parallel and arguably more challenging line
of research tackles the _inverse_ problem: deducing molecular structure from NMR
spectra. Computer-assisted structure elucidation (CASE) systems have been under
development for over five decades, but their integration with modern machine
learning has accelerated markedly in recent years. Elyashberg reviewed the state
of the field in 2021, noting that contemporary CASE systems exploit one-
dimensional and two-dimensional NMR data in combination with DFT-computed
reference shifts and are capable of elucidating complex natural products, though
they still rely heavily on expert-curated rules and spectral databases
@Elyashberg2021CASE.

More recently, deep learning has begun to replace or augment these rule-based
pipelines. Wei et al. developed pseudo-Siamese convolutional neural networks
(pSCNN) that achieved 99.8% accuracy for compound identification from
one-dimensional NMR spectra of mixtures, demonstrating that raw spectral vectors
could serve as effective inputs for convolutional architectures
@Wei2022pSCNN. Li et al. introduced 2DNMRGym, an annotated experimental dataset
for atom-level molecular representation learning from two-dimensional NMR spectra
via surrogate supervision, providing a benchmark for training models that bridge
spectral data and molecular structure @Li2025_2DNMRGym. These efforts reflect a
broader trend toward end-to-end learning from spectral data, moving away from the
traditional pipeline of peak picking, assignment, and rule-based interpretation
toward architectures that learn directly from the spectral signal.


=== NMR Metabolomics and Automated Profiling <nmr-metabolomics>

In metabolomics, automated compound identification and quantification from complex
NMR mixture spectra has long been a bottleneck limiting throughput.
Ravanbakhsh et al. introduced BAYESIL, the first fully automated system for
quantitative #super[1]H NMR spectral profiling, which framed metabolite
identification as probabilistic inference within a graphical model and achieved
approximately 90% correct identification with approximately 10% quantification
error in under five minutes @Ravanbakhsh2015BAYESIL. The Human Metabolome
Database (HMDB), which has grown to contain over 220,000 metabolite entries with
associated NMR reference spectra, provides the foundational spectral library
against which such automated systems operate @Wishart2022HMDB.

Subsequent deep learning approaches improved upon the BAYESIL foundation.
The pseudo-Siamese CNN architecture of Wei et al. demonstrated that
mixture spectra could be deconvolved and individual compounds identified with
near-perfect accuracy @Wei2022pSCNN, while one-dimensional CNN architectures
applied to low-field NMR transverse relaxation signals achieved robust
classification of edible oils and other complex mixtures. These metabolomics
applications established a critical precedent for the present work: they proved
that minimally processed NMR spectral vectors, treated as fixed-length numerical
feature vectors, could serve as effective inputs for supervised learning models.
The spectral featurization strategies developed in the metabolomics context---
including bucket integration, peak alignment, and normalization---translate
directly to the problem of encoding NMR spectra as molecular descriptors for
property prediction.


=== The Gap: NMR Spectra as Features for Property Prediction <nmr-property-gap>

Despite the maturity of machine learning methods for NMR shift prediction
(structure to spectrum) and structure elucidation (spectrum to structure),
remarkably few studies have explored using NMR spectral representations to predict
bulk physical or thermophysical properties such as viscosity, density, surface
tension, or heat capacity. The overwhelming majority of molecular property
prediction models rely on structural fingerprints (ECFP, MACCS), computed
descriptors (RDKit, Dragon), SMILES strings, or molecular graphs as input
representations. NMR spectra, whether experimental or predicted, have been almost
entirely absent from this literature.

Leniak, Pietrus, and Kurczab published one of the first explicit investigations
of this direction in 2024, designing NMR-derived chemical representations for
predicting logD using support vector regression and gradient boosting methods
@Leniak2024NMRtoAI. Their approach discretized predicted #super[1]H NMR spectra
into bucket-integrated feature vectors and achieved RMSE of 0.66, competitive
with models trained on conventional molecular fingerprints. In a follow-up study,
the same group fused predicted #super[1]H and #super[13]C spectra into compact
descriptor vectors and trained one-dimensional CNNs that matched the performance
of traditional ECFP4 fingerprints at one-fifth the dimensionality
@Leniak2025FusedNMR. Their work demonstrated that NMR-derived representations
carry sufficient structural information to support quantitative structure-property
relationship modeling, though they focused exclusively on pharmaceutical
properties (logD, solubility) rather than thermophysical quantities.

In parallel, Zipoli, Alberts, and Laino released a large-scale computational
IR--NMR multimodal spectral dataset covering 177,000 patent-extracted organic
molecules @Zipoli2025IRNMR, providing the kind of large, standardized spectral
dataset that data-driven property prediction models require. The availability of
such datasets, combined with the growing repositories of experimental NMR data
from HMDB @Wishart2022HMDB and NMRShiftDB @Kuhn2024TwentyYears, creates the
infrastructure necessary for training models that map spectral features to
physical properties at scale.

The gap that remains is clear. Abundant work exists on predicting NMR spectra from
structure, and substantial progress has been made on deducing structure from NMR
spectra. However, the direct path from NMR spectral features to thermophysical
property values---bypassing explicit structural identification---remains largely
unexplored. This gap is significant because NMR spectra encode rich information
about molecular topology, functional group environments, and conformational
preferences that conventional molecular fingerprints may fail to capture. A
#super[1]H NMR spectrum, for instance, simultaneously encodes the presence and
relative abundance of aromatic, olefinic, aliphatic, and heteroatom-adjacent
proton environments in a single measurement. These are precisely the structural
features that govern intermolecular interactions and, consequently, bulk transport
and interfacial properties. The present work addresses this gap by treating NMR
spectral vectors---both experimental and predicted---as molecular descriptors for
viscosity, density, and surface tension regression, bridging the mature
NMR-machine-learning literature with the thermophysical property prediction
domain discussed in the preceding sections.

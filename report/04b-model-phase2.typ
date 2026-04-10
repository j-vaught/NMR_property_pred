// 04b-model-phase2 — Phase 2: NMR Spectrum Models
// The heading is provided by main.typ as:  == Phase 2: NMR Spectrum Models <model-phase2>

Phase 1 established strong baselines using molecular fingerprints derived from
SMILES strings, achieving hydrocarbon $R^2$ values above 0.9 for viscosity with
XGBoost on Morgan fingerprints. However, the Phase 1 pipeline requires the
molecular structure to be known at inference time, which is not always the case
for novel or proprietary fuel blends. Phase 2 pursues a fundamentally different
objective: predicting thermophysical properties directly from #super[1]H NMR
spectral data, without requiring SMILES strings or molecular fingerprints at
inference. If successful, such a model would accept a raw NMR measurement as
input and return a property estimate, eliminating the need for compound
identification entirely.


=== Peak List Processing and Spectrum Generation <peak-processing>

The NMRexp database stores #super[1]H NMR data as structured peak lists rather
than raw free induction decays. Each entry is a list of tuples encoding
multiplicity, J-coupling constants in Hz, an integral string (e.g., "2H"),
and the upper and lower bounds of the chemical shift range in ppm. The first
step in the Phase 2 pipeline parses these structured strings into a list of peak
dictionaries, each containing the center chemical shift (the mean of the upper
and lower bounds), the numerical integral, the multiplicity label, and a list of
J-coupling values.

Converting discrete peak lists into fixed-length continuous spectra is necessary
because convolutional and residual architectures require uniform-length input
vectors. Each parsed peak is placed on a 1200-point grid spanning 0--12 ppm and
broadened using a Lorentzian lineshape,

$ L(x) = I dot (gamma slash 2)^2 / ((x - x_0)^2 + (gamma slash 2)^2) $ <lorentzian>

where $x_0$ is the chemical shift center, $I$ is the integral (number of
hydrogens), and $gamma = 0.01$ ppm is the linewidth parameter. For peaks with
resolved J-coupling information, first-order splitting patterns are generated
using Pascal's triangle coefficients. A doublet with coupling constant $J$ places
two lines separated by $J slash nu_0$ ppm (where $nu_0 = 400$ MHz is the assumed
spectrometer frequency), with relative intensities 1:1. Triplets, quartets, and
higher-order multiplets follow the standard first-order splitting rules. For
complex multiplets (dd, dt, dq, and similar), successive splitting is applied:
each existing line is split into a doublet by each successive coupling constant.
Unresolved multiplets labeled "m" with a finite shift range are represented by
Gaussian envelopes whose width matches the reported range.

After all peaks have been placed and broadened, the resulting spectrum is
max-normalized to the interval $[0, 1]$, yielding a fixed-length vector of 1200
intensity values suitable for input to neural network architectures.


=== Feature Extraction from Spectra and Peak Lists <feature-extraction>

While deep learning architectures can operate directly on the 1200-point
spectrum, the dataset size of approximately 500 compounds with matched NMR and
property data is far too small to train convolutional or transformer models
reliably. This motivates the extraction of a compact, domain-informed feature
vector that reduces dimensionality while preserving chemically meaningful
information.

The spectrum feature extractor computes 95 features from the normalized 1200-point
spectrum. These include 60 fine-grained spectral bin features (30 bins of 0.4 ppm
width, each contributing a mean and maximum intensity), 27 region-based features
(integral, fractional integral, and peak count for each of nine chemically
significant regions: alkyl high-field 0--1 ppm, alkyl mid 1--2 ppm, alkyl
$alpha$-heteroatom 2--3 ppm, oxygenated 3--4.5 ppm, olefinic low 4.5--5.5 ppm,
olefinic high 5.5--6.5 ppm, aromatic low 6.5--7.5 ppm, aromatic high
7.5--8.5 ppm, and aldehyde/downfield 8.5--12 ppm), and 8 global spectral
statistics (centroid, spread, skewness, kurtosis, total peak count, normalized
Shannon entropy, nonzero fraction, and maximum peak height).

However, 95 spectrum features alone proved insufficient for property prediction,
particularly for viscosity. The critical issue is that max-normalization of the
spectrum to $[0, 1]$ destroys absolute hydrogen count information. A molecule
with 6 hydrogens and a molecule with 60 hydrogens can produce nearly identical
normalized spectra if their proton environments are proportionally similar. Yet
these molecules will differ dramatically in molecular weight and, consequently,
in viscosity.

To address this information loss, 25 additional features are extracted directly
from the parsed peak list before normalization. These peaklist features include
three size proxies (total hydrogen count, number of distinct peaks, and maximum
single-peak integral), nine absolute hydrogen counts per chemical shift region,
seven multiplicity distribution fractions (singlet, doublet, triplet, quartet,
multiplet, complex, and other), three J-coupling statistics (mean, maximum, and
fraction of peaks with coupling information), and three derived ratios (alkyl
hydrogen fraction, aromatic hydrogen fraction, and a CH#sub[2]/CH#sub[3] proxy
computed as the ratio of 1--2 ppm integral to the sum of 0--1 and 1--2 ppm
integrals). The CH#sub[2]/CH#sub[3] proxy is particularly informative for
viscosity prediction because it correlates with chain length: longer alkyl chains
have higher CH#sub[2] content relative to terminal CH#sub[3] groups, and longer
chains produce higher viscosities through increased intermolecular van der Waals
interactions.

The total hydrogen count, the most important of the 25 peaklist features, serves
as a strong proxy for molecular weight. This is the single most critical piece of
information that raw NMR spectra encode imperfectly: while the relative ratios of
proton environments are captured faithfully by the spectrum shape, the absolute
scale of the molecule is only accessible through the integral values recorded in
the peak list.

Combining the 95 spectrum features with the 25 peaklist features yields a
120-dimensional feature vector per compound.


=== Model Architectures <phase2-architectures>

Four model families were evaluated on the Phase 2 task. Three are deep learning
architectures operating on the raw 1200-point spectrum, and one is a gradient
boosted tree ensemble operating on the 120-dimensional extracted feature vector.

The 1D CNN architecture (NMR1DCNN) processes the spectrum through a sequence of
convolutional layers with batch normalization, ReLU activations, and max-pooling
operations. In its small configuration, designed for the available dataset size
of fewer than 1000 compounds, the architecture applies three convolutional layers
with 16, 32, and 64 filters (kernel sizes 7, 5, and 3 respectively), each
followed by max-pooling that progressively reduces the spatial dimension from
1200 to 300 to 75 to 25 points. Global average pooling and dropout ($p = 0.5$)
produce a 64-dimensional embedding. The large configuration adds a fourth
convolutional layer and doubles channel widths throughout (32, 64, 128, 256
filters), yielding a 256-dimensional embedding with approximately 145,000
trainable parameters.

The 1D ResNet architecture (NMR1DResNet) introduces residual connections to
enable deeper feature extraction. Each residual block consists of two
convolutional layers (kernel size 3) with batch normalization and a skip
connection that bypasses both layers, with a $1 times 1$ projection applied when
the channel count or spatial resolution changes. The small variant stacks four
residual blocks across three stages (channels 12, 24, 48, 64) with strided
convolutions for downsampling, producing a 64-dimensional embedding with
approximately 18,000 parameters. The large variant uses five blocks across three
stages (18, 36, 72, 144 channels), yielding a 144-dimensional embedding with
approximately 190,000 parameters. Both configurations apply global average
pooling and dropout before the prediction head.

The Transformer architecture (NMRTransformer) takes a fundamentally different
approach by operating on the peak list directly rather than on the broadened
spectrum. Each peak is encoded as a 12-dimensional feature vector containing the
shift center, shift range, an 8-dimensional one-hot multiplicity encoding, and
two normalized J-coupling values. A two-layer MLP embeds each peak into $d$-dimensional
space, and learned positional encodings (maximum 50 peaks) are added
to preserve ordering information. The embedded sequence is processed by a
Transformer encoder with multi-head self-attention. The small variant uses
$d_"model" = 32$, 4 heads, 2 layers, and a feed-forward dimension of 64. The
large variant uses $d_"model" = 64$, 4 heads, 4 layers, and a feed-forward
dimension of 128 with approximately 130,000 parameters. Mean pooling over
non-padded positions produces the final embedding.

All three neural architectures support two prediction head configurations. The
direct head concatenates the embedding with three temperature features
(temperature in K, its reciprocal, and its squared reciprocal) and produces a
single property value through a two-layer MLP. The Arrhenius head predicts two
parameters $A$ and $B$ from the embedding alone, from which the
temperature-dependent property is computed analytically. Both heads use Kaiming
initialization and dropout regularization.

The fourth model family, XGBoost, operates on the extracted 120-dimensional
feature vector (95 spectrum + 25 peaklist features) rather than on the raw
spectrum. This choice is motivated by the classical machine learning principle
that hand-engineered features outperform learned representations when training
data is severely limited @Chen2016XGBoost. With only approximately 500 compounds,
XGBoost's built-in L1/L2 regularization, column subsampling, and early stopping
provide far more effective overfitting control than the implicit regularization
of neural network training.


=== Results and Critical Findings <phase2-results>

The Phase 2 comparison experiments evaluated each input representation (NMR
spectrum features alone, NMR full features including peaklist features, Phase 1
fingerprints matched to the same compound subset, and a combined NMR + fingerprint
representation) with both Ridge regression and XGBoost, using 5-fold
cross-validation with compound-level splitting to prevent data leakage. All
results are reported on the hydrocarbon subset, which provides a more meaningful
evaluation than the full dataset because it excludes extreme outliers from highly
polar or polymeric compounds.

#figure(
  table(
    columns: (1.6fr, 1fr, 1fr, 1fr, 1fr),
    align: (left, center, center, center, center),
    stroke: 0.5pt,
    table.header(
      [*Input Representation*], [*Model*], [*Visc. HC $R^2$*], [*ST HC $R^2$*], [*Note*],
    ),
    [NMR spectrum (95 feat.)], [Ridge], [$-0.63$], [$0.16$], [No size info],
    [NMR spectrum (95 feat.)], [XGBoost], [$-0.59$], [$0.15$], [No size info],
    [NMR full (120 feat.)], [Ridge], [$-10.4$], [$0.16$], [Unstable],
    [NMR full (120 feat.)], [XGBoost], [$-0.39$], [$0.40$], [Improved],
    [Phase 1 FP (matched)], [Ridge], [$0.49$], [$0.68$], [Baseline],
    [Phase 1 FP (matched)], [XGBoost], [$0.69$], [$0.81$], [Baseline],
    [Combined (NMR + FP)], [Ridge], [$0.40$], [$0.64$], [No gain],
    [Combined (NMR + FP)], [XGBoost], [$0.69$], [$0.81$], [Matches FP],
  ),
  caption: [Phase 2 hydrocarbon $R^2$ comparison across input representations and models.
  Viscosity values are averaged across 5 folds. Negative $R^2$ indicates predictions
  worse than the training set mean. ST denotes surface tension.],
) <phase2-results-table>

The results in @phase2-results-table reveal several important findings. NMR
spectrum features alone produce negative $R^2$ values for viscosity, meaning the
model predictions are worse than simply predicting the mean. This is not a
modeling failure but an information-theoretic limitation: the max-normalized
spectrum encodes the _shape_ of the proton distribution but not the _scale_ of
the molecule. Two alkanes with identical branching patterns but different chain
lengths (e.g., decane and eicosane) produce very similar normalized spectra yet
differ in viscosity by nearly an order of magnitude. For surface tension, NMR
spectrum features alone achieve modest positive $R^2$ values (0.15--0.16),
consistent with the fact that surface tension depends more on functional group
polarity (which is partially encoded in spectral shape) than on molecular size.

Adding the 25 peaklist features (NMR full, 120 dimensions) improves surface
tension prediction with XGBoost to $R^2 = 0.40$, confirming that the absolute
hydrogen count and multiplicity distribution carry information complementary to
the normalized spectrum. For viscosity, however, Ridge regression becomes
unstable (mean HC $R^2 = -10.4$ due to a single catastrophic fold), while
XGBoost shows modest improvement to $R^2 = -0.39$, still far below useful
prediction accuracy. The NMR peaklist features provide a proxy for molecular
weight through the total hydrogen count, but this proxy is noisy: molecules with
the same hydrogen count can differ substantially in molecular weight depending on
the number of carbon, oxygen, and nitrogen atoms, none of which are directly
observable in a #super[1]H NMR spectrum.

The Phase 1 fingerprints, evaluated on the same matched compound subset,
achieve HC $R^2 = 0.69$ for viscosity and $R^2 = 0.81$ for surface tension
with XGBoost. These represent the ceiling against which NMR-based models are
compared. Importantly, combining NMR features with fingerprints (the "combined"
representation) does not improve upon fingerprints alone, indicating that the NMR
features carry little information beyond what is already encoded in the
Morgan/RDKit fingerprint representation when the molecular structure is known.

The three deep learning architectures (1D CNN, ResNet, and Transformer) were also
evaluated on the raw spectrum and peak list inputs but consistently
underperformed XGBoost on extracted features. This outcome was expected given the
dataset size: with approximately 500 training compounds and 1200-dimensional
input spectra, the neural architectures have far too many parameters relative to
training examples, even in their small configurations. The XGBoost models on
120-dimensional extracted features benefit from aggressive regularization and
require far fewer effective parameters to achieve good generalization.


=== The Molecular Weight Bottleneck <mw-bottleneck>

The Phase 2 results expose a fundamental information bottleneck in NMR-based
property prediction. Viscosity scales approximately with $M^alpha$ where
$alpha approx 1$--$2$ for small molecules and can be much larger near the
entanglement threshold @Bilodeau2024Advancing. A #super[1]H NMR spectrum encodes
the proton environment distribution with high fidelity but does not directly
encode molecular weight. The total hydrogen count is a weak proxy for $M$ because
the hydrogen-to-heavy-atom ratio varies substantially across chemical classes.
Aromatic hydrocarbons, for instance, have far fewer hydrogens per unit molecular
weight than saturated alkanes.

This bottleneck has two implications for the broader research program. First, NMR
alone is unlikely to achieve competitive viscosity prediction without supplemental
information about molecular size. Second, the information that NMR spectra _do_
encode---functional group environments, branching patterns, aromaticity,
heteroatom proximity---is precisely the kind of structural information that could
complement a separate estimate of molecular weight to produce accurate property
predictions. This insight motivates Phase 4's approach of combining NMR-derived
descriptors with molecular weight estimation, and it explains why the combined
NMR + fingerprint representation in Phase 2 matches but does not exceed
fingerprint-only performance: the fingerprints already encode molecular weight
implicitly through the presence of size-dependent substructure bits.

The Phase 2 results for surface tension are more encouraging. Surface tension is
governed primarily by the nature of intermolecular interactions at the
liquid-vapor interface, which depends on functional group polarity and
hydrogen bonding capacity more than on molecular size. The NMR full feature set
achieves $R^2 = 0.40$ for surface tension even without fingerprint augmentation,
suggesting that NMR spectra carry meaningful information about the intermolecular
interaction potential. This finding motivates continued development of NMR-based
models for properties that depend on chemical environment rather than molecular
scale.

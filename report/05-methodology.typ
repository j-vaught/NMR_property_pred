// 05-methodology.typ — Evaluation methodology, metrics, and experimental setup

== Evaluation Metrics <eval-metrics>

All models in this work are assessed with three complementary metrics that together characterize accuracy, scale sensitivity, and practical relevance. Let $y_i$ denote the $i$-th observed value, $hat(y)_i$ the corresponding prediction, and $overline(y)$ the mean of the observed values over $n$ test samples.

The coefficient of determination quantifies the fraction of variance in the observed data that is explained by the model.

$ R^2 = 1 - (sum_(i=1)^n (y_i - hat(y)_i)^2) / (sum_(i=1)^n (y_i - overline(y))^2) $ <r2-def>

An $R^2$ of 1.0 indicates perfect prediction, while a value of 0.0 indicates that the model performs no better than predicting the mean. Negative values are possible when the model is systematically worse than the mean baseline. The implementation guards against a degenerate denominator: if the total sum of squares is zero (all observations identical), $R^2$ is returned as zero rather than undefined.

Root mean squared error measures prediction accuracy in the same units as the target variable.

$ "RMSE" = sqrt(1/n sum_(i=1)^n (y_i - hat(y)_i)^2) $ <rmse-def>

Because RMSE is computed after back-transformation to physical units (Pa$dot$s for viscosity, N/m for surface tension), it provides an absolute sense of the error magnitude that can be compared directly against measurement uncertainty and specification tolerances.

Mean absolute percentage error expresses accuracy as a dimensionless fraction of the true value, making it interpretable across properties that differ by orders of magnitude.

$ "MAPE" = 100/n sum_(i=1)^n lr(|frac(y_i - hat(y)_i, y_i)|) $ <mape-def>

To avoid division by zero, observations with $|y_i| < 10^(-10)$ are excluded from the MAPE calculation. MAPE is reported as a percentage and is the primary metric used when comparing models across properties, because a 5% error in viscosity and a 5% error in surface tension carry similar practical implications despite the values differing by several orders of magnitude.

== Cross-Validation Strategy <cv-strategy>

A naive random split of individual data points would allow training and test sets to share measurements from the same compound at different temperatures. Because the temperature dependence of viscosity and surface tension is smooth and well-behaved for any single compound, such leakage would inflate apparent accuracy: the model could interpolate along a known curve rather than generalize to unseen molecules. To eliminate this source of optimistic bias, all cross-validation in this work is performed at the compound level.

Specifically, the set of unique canonical SMILES strings is partitioned into $k = 5$ folds using scikit-learn's `KFold` with a fixed random seed of 42. All temperature--property pairs belonging to a given compound are assigned to the same fold. When fold $j$ serves as the test set, folds $1, dots, j-1, j+1, dots, 5$ provide the training data. This procedure ensures that the model has never seen any measurement from any test compound during training, providing an honest estimate of generalization to new molecules @Chen2016.

Within each training fold, a further random 90/10 split of individual data points (not compounds) is used to create a validation set for early stopping. This inner split need not be compound-level because its purpose is to detect overfitting of the current training run rather than to estimate generalization error. For XGBoost models, the validation set monitors the evaluation metric and halts boosting when performance has not improved for a specified number of rounds. For neural network models, validation loss triggers early stopping with a patience of 30--50 epochs, depending on the phase.

== Hydrocarbon-Focused Evaluation <hc-eval>

Although the training set spans a broad chemical space including alcohols, ethers, esters, and other heteroatom-containing compounds, the downstream application of this work is aviation fuel property prediction. Jet fuels are almost exclusively hydrocarbon mixtures: petroleum-derived Jet A-1 consists of $n$-alkanes, isoalkanes, cycloalkanes, and aromatics, while approved SAF blending components are similarly constrained to hydrocarbons or near-hydrocarbons @Edwards2017 @ASTMD7566. A model that achieves excellent aggregate metrics may still perform poorly on the hydrocarbon subset if it relies on heteroatom-specific features to drive its predictions.

For this reason, every results table reports two sets of metrics: one computed over all test compounds and one restricted to hydrocarbon-only (HC) compounds, defined as molecules whose atoms are exclusively carbon and hydrogen. The HC filter is applied by parsing the canonical SMILES with RDKit and checking that the set of atomic symbols is a subset of ${"C", "H"}$. Hydrocarbon-only $R^2$ and MAPE are the primary figures of merit when evaluating suitability for fuel applications, because they reflect the chemical domain in which the model will ultimately be deployed.

== Log-Viscosity as Prediction Target <log-viscosity>

Dynamic viscosity of organic liquids spans roughly six orders of magnitude across the temperature and compositional ranges encountered in this study, from approximately $10^(-5)$ Pa$dot$s for light hydrocarbons near their boiling points to $10^1$ Pa$dot$s for heavy compounds near their freezing points. Training a regression model on raw viscosity in Pa$dot$s would cause the loss function to be dominated by the handful of high-viscosity compounds, effectively ignoring errors on low-viscosity materials that are most relevant to aviation fuels.

The Arrhenius relationship $ln eta = A + B slash T$ provides both physical and statistical motivation for using the natural logarithm of viscosity as the prediction target @ASTMD341. In log space, viscosity is approximately linear in inverse temperature, the dynamic range compresses from six orders of magnitude to roughly 14 units, and the implicit assumption of multiplicative (rather than additive) errors aligns with the constant relative uncertainty typical of viscometric measurements. All viscosity models in this work therefore predict $ln(eta)$ and back-transform to Pa$dot$s only for final metric computation and physical interpretation. Surface tension, which varies over a much narrower range (roughly 0.015--0.075 N/m), is predicted directly without transformation.

== Temperature Feature Engineering <temp-features>

Because the target properties are temperature-dependent, every model receives temperature as an explicit input alongside the molecular representation. Rather than supplying raw temperature in Kelvin, which would place a single feature on a scale of 200--500 while fingerprint bits are binary, a three-dimensional temperature feature vector is constructed.

$ bold(t)(T) = [T slash 1000, quad 1 slash T, quad (T slash 1000)^2] $ <temp-feature-vec>

The $1 slash T$ term directly encodes the Arrhenius-like inverse-temperature dependence of viscosity. The scaled temperature $T slash 1000$ provides a normalized linear term, and its square $(T slash 1000)^2$ captures any residual curvature beyond the Arrhenius approximation. This feature set allows even a linear model to represent the dominant $1 slash T$ dependence while giving nonlinear models (neural networks, gradient-boosted trees) additional flexibility to capture departures from ideal Arrhenius behavior. The scaling by 1000 ensures that all three features are of order unity, avoiding numerical issues during gradient-based optimization.

== Data Filtering and Quality Control <data-filtering>

Raw thermophysical data aggregated from ThermoML @Frenkel2006ThermoML and the chemicals library @Bell2025chemicals undergo several filtering steps before model training. Temperature is restricted to the range 200--500 K, which covers the service envelope of aviation fuels with a margin for extrapolation studies. Pressure is limited to atmospheric or near-atmospheric conditions ($lt.eq$ 110 kPa) where available, because the models in this work do not include pressure as an input variable. Viscosity values are retained only if they fall within $10^(-5)$ to 30 Pa$dot$s; values outside this range are flagged as likely unit-conversion errors or non-Newtonian materials and are discarded. Surface tension values are similarly bounded between 0 and 1 N/m.

Molecular features are computed using RDKit @Landrum2024RDKit. Morgan fingerprints (radius 2, 2048 bits) and a suite of RDKit molecular descriptors are generated for each compound from its canonical SMILES. Compounds that yield `NaN` or infinite descriptor values after sanitization are removed. Fingerprint bits and descriptors with zero variance across the training set are dropped, because they carry no discriminative information. When ThermoML and group-contribution sources provide overlapping measurements for the same compound at the same temperature (rounded to 0.1 K), the ThermoML value is preferred on the grounds that it originates from peer-reviewed experimental data.

SMILES strings are canonicalized through RDKit's `MolFromSmiles` and `MolToSmiles` round-trip to ensure consistent molecular representation. InChI identifiers from ThermoML are first converted to RDKit molecule objects via `MolFromInchi` and then output as canonical SMILES via `MolToSmiles`. This two-step canonicalization guarantees that the same compound is represented identically regardless of its original identifier format, preventing duplicate entries and ensuring correct feature--label joins.

== Training and Early Stopping <training-details>

Neural network models are trained with the AdamW optimizer at an initial learning rate of $3 times 10^(-4)$ and weight decay of $10^(-4)$. A cosine annealing schedule with 10-epoch linear warm-up reduces the learning rate to a floor of $10^(-6)$ over the course of training. Gradient norms are clipped to 1.0 to prevent instability. The loss function is the Huber loss with $delta = 1.0$, which combines the sensitivity of mean squared error near zero with the robustness of mean absolute error for large residuals, providing resilience against occasional outlier measurements in the aggregated dataset. Training proceeds for up to 500 epochs with early stopping triggered after 50 epochs of no improvement on the validation set.

For each cross-validation fold, an ensemble of five independently initialized models is trained with different random seeds. At inference time, the ensemble prediction is the arithmetic mean of the five individual predictions. Ensembling reduces variance without increasing bias and provides a simple mechanism for uncertainty quantification through the spread of individual predictions.

XGBoost models @Chen2016 use an analogous early-stopping protocol. The validation set monitors the evaluation metric (RMSE in log space for viscosity, RMSE in original units for surface tension), and boosting terminates when the metric has not improved for a specified number of rounds. Hyperparameters including maximum tree depth, learning rate, subsample ratio, and regularization terms are selected via grid search within each cross-validation fold.

Feature selection is performed via Lasso regression ($ell_1$-penalized linear model) applied to the compound-level mean of each target property. Features whose Lasso coefficients exceed $10^(-6)$ in absolute value for any target are retained; all others are discarded. This step reduces the input dimensionality from several thousand (2048 fingerprint bits plus several hundred RDKit descriptors) to a few hundred informative features, improving training speed and reducing overfitting risk. If Lasso selects fewer than 10 features (indicating an overly aggressive penalty), the full feature set is retained as a fallback.

== Hardware and Software <hardware-software>

All experiments are conducted on a shared compute server (comech-2422) equipped with an NVIDIA RTX 6000 Ada Generation GPU (48 GB VRAM). PyTorch models are trained on the GPU with mixed-precision arithmetic where supported. XGBoost training uses CPU-based histogram methods, which complete within minutes for the dataset sizes in this study.

The software stack consists of Python 3.10 with the following principal libraries: RDKit @Landrum2024RDKit for molecular parsing, fingerprint generation, and descriptor computation; XGBoost @Chen2016 for gradient-boosted tree models; PyTorch for neural network construction and training; and scikit-learn for preprocessing (standard scaling), cross-validation splitting, and Lasso feature selection. All random seeds are fixed at 42 (or deterministic offsets thereof) to ensure reproducibility of data splits, model initialization, and training dynamics.

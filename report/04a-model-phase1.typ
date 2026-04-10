// 04a-model-phase1.typ — Phase 1: Molecular Fingerprint Baselines

The objective of Phase 1 is to establish baseline predictive accuracy for viscosity and surface tension using only molecular structure, without any NMR spectral information. These baselines quantify how well standard cheminformatics representations capture thermophysical behavior and provide a reference against which subsequent NMR-augmented models (Phases 2--4) are compared. All data sources, feature generation procedures, and quality filters described in @data apply to this phase.

=== Input Feature Representation <phase1-features>

Each compound is represented by the concatenation of three complementary structural descriptors. First, 2048-bit Morgan circular fingerprints @Rogers2010 computed at radius 2 encode the presence or absence of substructural neighborhoods up to four bonds from each atom, providing a hashed topological description that captures functional-group context. Second, 166-bit MACCS structural keys @Carhart1985 record the presence of predefined pharmacophore-like fragments, including specific ring systems, heteroatom patterns, and chain lengths. Third, a set of RDKit molecular descriptors @Landrum2024RDKit computed by the RDKit toolkit supply continuous physicochemical quantities such as molecular weight, Wildman--Crippen $log P$, topological polar surface area, Balaban's $J$ index, and various partial-charge and surface-area partitions. After removing zero-variance columns, the combined feature vector has approximately 1,696 dimensions per compound.

Temperature enters the model through three engineered features: the absolute temperature $T$ in kelvin, its reciprocal $1 slash T$ (motivated by the Arrhenius form of @eq-arrhenius), and the reduced temperature $T slash 1000$. These three scalar features are concatenated with the molecular embedding at the prediction head, enabling the model to learn temperature-dependent property surfaces for each compound.

=== Traditional Machine Learning Models <phase1-traditional>

Five model families were compared in a compound-level five-fold cross-validation protocol using the Arrhenius parameterization. In this formulation, each model predicts the two Arrhenius coefficients $A$ and $B$ of @eq-arrhenius from the molecular feature vector, and reconstructed viscosity curves are evaluated against held-out experimental data. Hyperparameters were tuned by randomized search with an inner three-fold cross-validation loop on the training partition of each outer fold. The five model families are XGBoost @Chen2016, Random Forest @Breiman2001, LightGBM @Ke2017, Ridge regression, and Lasso regression.

@tab-arrhenius-visc and @tab-arrhenius-st report the Arrhenius coefficient $R^2$ and the reconstructed-property $R^2$ for viscosity and surface tension, respectively. For viscosity, XGBoost achieves the highest reconstructed $R^2$ of 0.274, with Random Forest close behind at 0.308. The linear models (Ridge and Lasso) produce more stable but lower reconstructed performance ($R^2 approx 0.11 dash 0.13$). All models display high variance across folds, reflecting the difficulty of predicting Arrhenius intercepts and slopes from fingerprints alone. For surface tension, XGBoost again leads with a reconstructed $R^2$ of 0.761, substantially higher than the viscosity case, suggesting that surface tension is more directly encoded in molecular topology.

#figure(
  table(
    columns: (1.2fr, 0.8fr, 0.8fr, 0.8fr, 0.8fr),
    align: (left, center, center, center, center),
    table.header(
      [*Model*], [$R^2 (A)$], [$R^2 (B)$], [$R^2 ("recon")$], [*Time (s)*],
    ),
    [XGBoost],     [0.305], [0.629], [0.274], [593],
    [LightGBM],    [0.269], [0.641], [0.211], [350],
    [Random Forest],[0.327], [0.622], [0.308], [191],
    [Ridge],       [0.332], [0.536], [0.132], [0.5],
    [Lasso],       [0.409], [0.592], [0.106], [24],
  ),
  caption: [Arrhenius-parameterized viscosity prediction: five-fold compound-level CV. $R^2 (A)$ and $R^2 (B)$ are the coefficient-of-determination for the Arrhenius intercept and slope, respectively. $R^2 ("recon")$ evaluates reconstructed $ln eta$ against experimental values.],
) <tab-arrhenius-visc>

#figure(
  table(
    columns: (1.2fr, 0.8fr, 0.8fr, 0.8fr, 0.8fr),
    align: (left, center, center, center, center),
    table.header(
      [*Model*], [$R^2 (A)$], [$R^2 (B)$], [$R^2 ("recon")$], [*Time (s)*],
    ),
    [XGBoost],     [0.564], [0.456], [0.761], [1119],
    [LightGBM],    [0.560], [0.468], [0.748], [892],
    [Random Forest],[0.548], [0.442], [0.728], [92],
    [Ridge],       [0.445], [0.428], [0.622], [0.9],
    [Lasso],       [0.478], [0.438], [0.621], [82],
  ),
  caption: [Arrhenius-parameterized surface tension prediction: five-fold compound-level CV, same protocol as @tab-arrhenius-visc.],
) <tab-arrhenius-st>

=== Neural Network Architecture <phase1-nn>

To move beyond the two-coefficient Arrhenius parameterization, a neural network was trained to predict $ln eta$ (viscosity) or surface tension directly, with temperature supplied as an explicit input alongside the molecular features. This direct formulation avoids forcing the model into a two-parameter functional form and allows the network to learn nonlinear temperature dependencies that deviate from ideal Arrhenius behavior.

The architecture follows a shared-encoder, property-specific-head design. The shared encoder is a stack of three fully connected layers with dimensions $d_"in" arrow.r 512 arrow.r 256 arrow.r 128$, where $d_"in"$ is the number of selected molecular features. Each layer applies a linear transformation, batch normalization, ReLU activation, and dropout ($p = 0.3$) in sequence, with Kaiming normal initialization for all weight matrices. The encoder output is a 128-dimensional molecular embedding. The three temperature features described in @phase1-features are concatenated with this embedding to form a 131-dimensional vector, which is passed to a property-specific prediction head. Each head consists of a single hidden layer of 32 units (batch normalization, ReLU, dropout $p = 0.2$) followed by a linear output producing a scalar prediction.

Because Lasso feature selection is applied per fold to the full 1,696-dimensional feature vector, the effective input dimension varies between 22 and 38 features across folds, with a median of approximately 28. The Lasso step (regularization strength $alpha = 0.1$) fits a linear model to the per-compound mean property value and retains only those features whose coefficients exceed $10^(-6)$ in absolute value. This aggressive dimensionality reduction is intentional: it prevents the network from memorizing bit-level fingerprint noise and forces reliance on the most informative descriptors.

=== Training Protocol <phase1-training>

Training proceeds under a five-fold compound-level cross-validation protocol. Compound-level splitting ensures that all temperature-dependent measurements for a given molecule appear in exactly one partition, preventing data leakage between training and test sets. Within each training fold, 10% of the data is held out as a validation set for early stopping.

The loss function is the Huber loss with $delta = 1.0$, which interpolates between mean squared error for small residuals and mean absolute error for large residuals. This choice mitigates the influence of outliers in the property databases without discarding them entirely. The optimizer is AdamW with a learning rate of $3 times 10^(-4)$ and weight decay of $10^(-4)$. A cosine-annealing learning rate schedule with a 10-epoch linear warmup ramps the learning rate from near zero to its peak, then decays it smoothly to $10^(-6)$ over the remaining epochs. Gradient norms are clipped at 1.0 to stabilize training. Training runs for up to 500 epochs with early stopping triggered after 50 epochs of no improvement on the validation loss. Input features and targets are standardized to zero mean and unit variance using statistics computed on the training fold.

An ensemble of five independently initialized networks is trained per fold (25 total across the five-fold CV). Predictions at test time are the arithmetic mean of the five ensemble members, which reduces variance and improves calibration.

=== Results <phase1-results>

@tab-nn-results reports the neural network's five-fold CV performance in both the log/linear prediction space and in real physical units. For viscosity, the network achieves a mean $R^2$ of 0.638 in log-space across all compounds. When restricted to the hydrocarbon subset, performance improves to $R^2 = 0.703$, consistent with the expectation that structurally simpler hydrocarbons are easier to model than oxygenated or heteroatom-containing species. For surface tension, the network achieves a mean $R^2$ of 0.558 across all compounds, rising to 0.623 for hydrocarbons. Fold-to-fold variance is substantial: surface tension $R^2$ ranges from 0.107 to 0.868 across the five folds, reflecting the sensitivity of the model to the particular compounds in each test partition.

#figure(
  table(
    columns: (1.4fr, 0.7fr, 0.7fr, 0.7fr, 0.7fr),
    align: (left, center, center, center, center),
    table.header(
      [*Property*], [$R^2$ (all)], [$R^2$ (HC)], [RMSE], [Compounds],
    ),
    [Viscosity ($ln eta$)],   [0.638], [0.703], [1.54], [856],
    [Surface tension (N/m)],  [0.558], [0.623], [0.009], [714],
  ),
  caption: [Neural network direct-prediction results: five-fold compound-level CV with five-member ensembles. Viscosity $R^2$ is evaluated in log-space; surface tension in physical units. RMSE is the mean across folds in the respective prediction space.],
) <tab-nn-results>

Comparing the neural network to the traditional models reveals an important asymmetry. For viscosity, the Arrhenius-parameterized XGBoost model (reconstructed $R^2 = 0.274$) operates on a smaller compound set (203 compounds with sufficient temperature coverage for Arrhenius fitting), while the neural network's direct formulation exploits the full dataset of 856 compounds. The neural network's log-space $R^2$ of 0.638 is therefore not directly comparable to the Arrhenius reconstruction $R^2$, though both indicate that fingerprints alone capture only moderate viscosity variation. For surface tension, the XGBoost Arrhenius model achieves $R^2 = 0.761$ on its 654-compound set, while the neural network reaches $R^2 = 0.558$ on the same compounds. The gap likely reflects the XGBoost model's ability to exploit nonlinear feature interactions without the aggressive dimensionality reduction imposed by the neural network's Lasso preprocessing.

=== Feature Importance Analysis <phase1-feature-importance>

XGBoost's built-in feature-importance scores (sum of split gains across all trees) reveal which input dimensions drive the predictions. @tab-top-features lists the ten most important features for each property, ranked by mean importance across folds. For both viscosity and surface tension, the top features are overwhelmingly RDKit molecular descriptors rather than individual fingerprint bits. The QED drug-likeness score ranks first for both properties, followed by BCUT2D descriptors that encode eigenvalues of the modified burden matrix weighted by atomic $log P$, partial charge, and molar refractivity. Topological indices such as Balaban's $J$ and Kappa shape indices also appear prominently, as do partial-charge statistics and EState surface-area partitions.

#figure(
  table(
    columns: (0.3fr, 1.5fr, 0.6fr, 1.5fr, 0.6fr),
    align: (center, left, center, left, center),
    table.header(
      [*Rank*], [*Viscosity Feature*], [*Score*], [*Surface Tension Feature*], [*Score*],
    ),
    [1],  [qed],              [269], [qed],              [491],
    [2],  [BCUT2D_LOGPLOW],   [260], [BCUT2D_LOGPHI],    [387],
    [3],  [BCUT2D_CHGHI],     [233], [MinAbsEStateIndex], [372],
    [4],  [VSA_EState5],      [217], [MolLogP],           [337],
    [5],  [BCUT2D_LOGPHI],    [208], [BCUT2D_MWLOW],      [324],
    [6],  [MinEStateIndex],   [203], [BCUT2D_LOGPLOW],    [316],
    [7],  [BCUT2D_CHGLO],     [188], [BCUT2D_CHGLO],      [311],
    [8],  [BalabanJ],         [184], [BalabanJ],           [304],
    [9],  [BCUT2D_MRHI],      [175], [Kappa3],             [278],
    [10], [FpDensityMorgan1], [174], [VSA_EState8],        [278],
  ),
  caption: [Top 10 XGBoost features by split-gain importance for viscosity and surface tension. All top features are RDKit molecular descriptors; no individual Morgan or MACCS fingerprint bits appear in the top 20.],
) <tab-top-features>

Quantitatively, the top 20 features for viscosity are all RDKit descriptors, with the highest-ranked fingerprint bit appearing beyond rank 50. The cumulative importance of all RDKit descriptor features exceeds that of all 2,214 fingerprint bits by a factor of 50--60$times$. This finding carries two implications. First, continuous physicochemical descriptors encode more predictive information for thermophysical properties than binary substructure indicators, which is consistent with the physical intuition that properties like viscosity depend on molecular size, shape, and polarity rather than on the presence of any single substructure. Second, the Morgan and MACCS fingerprints, despite comprising over 90% of the input dimensions, contribute marginally to the models' predictive power. This motivates the introduction of NMR spectral features in Phase 2, which provide a fundamentally different, experimentally grounded representation of molecular structure.

=== Summary and Limitations <phase1-summary>

Phase 1 establishes that molecular fingerprints and computed descriptors capture moderate predictive power for thermophysical properties: $R^2 approx 0.64$ for viscosity in log-space and $R^2 approx 0.56 dash 0.76$ for surface tension, depending on the model and evaluation protocol. The hydrocarbon subset is consistently easier to predict ($R^2$ improves by 0.05--0.15), reflecting both the structural regularity of hydrocarbons and their narrower property distribution. Three principal limitations motivate the subsequent phases. First, fingerprints are computed from molecular structure, which requires knowing the compound's identity. For a fuel mixture of unknown composition, this prerequisite is not met. Second, the RDKit descriptors that dominate the predictions are calculated, not measured, quantities; their accuracy degrades for unusual molecular scaffolds outside the training distribution. Third, the high fold-to-fold variance indicates that the models do not generalize robustly across chemical families, suggesting that richer molecular representations are needed to improve prediction stability.

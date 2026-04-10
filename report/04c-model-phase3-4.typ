// 04c-model-phase3-4.typ — Phases 3–4: Fuel Mixture Prediction and Transfer Learning

The transition from pure-compound modeling in Phases 1 and 2 to fuel mixture prediction in Phases 3 and 4 introduces a fundamentally different challenge. Aviation fuels are complex hydrocarbon mixtures containing hundreds of individual compounds, and their NMR spectra represent the superposition of all constituent signals. No ground-truth molecular structure is available for a mixture spectrum, which means the fingerprint-based and structure-aware models developed in earlier phases cannot be applied directly. Phases 3 and 4 explore two strategies for bridging this gap: functional group decomposition and transfer learning from a large corpus of pure-compound NMR data.


=== Phase 3: Functional Group Approach <phase3>

The Phase 3 strategy follows the framework proposed by Abdul Jameel et al. @AbdulJameel2021Gasoline, which maps regions of a $""^1"H"$ NMR spectrum to nine functional group categories based on characteristic chemical shift ranges. These categories, including aliphatic CH#sub[3], CH#sub[2], and CH groups, naphthenic (cycloalkane) protons, olefinic protons, and several classes of aromatic protons, provide a coarse but physically meaningful decomposition of a mixture spectrum. Each functional group is assigned an integral proportional to the number of hydrogen atoms in that chemical environment, yielding a nine-dimensional feature vector that summarizes the gross composition of a fuel sample.

The Phase 3 pipeline proceeds in three steps. First, XGBoost regressors are trained on pure compounds from the Phase 2 dataset, using functional group features plus temperature as inputs and viscosity (in log Pa$dot$s) or surface tension (in N/m) as targets. Second, identical functional group features are extracted from each fuel mixture's $""^1"H"$ NMR spectrum. Third, the pure-compound models predict mixture properties at the temperatures for which measured data are available. Unit conversions are applied at the output stage: viscosity predictions in Pa$dot$s are converted to kinematic viscosity in cSt using approximate fuel densities @Edwards2020, and surface tension predictions in N/m are converted to mN/m.

The pipeline was validated against measured viscosity data for four AFRL reference fuels: HRJ-Camelina (POSF 7720), Shell SPK (POSF 5729), Sasol IPK (POSF 7629), and Gevo ATJ (POSF 10151). Three feature representations were compared: raw NMR spectral features (95 dimensions), full NMR features including peak-list statistics (120 dimensions), and functional groups (9 dimensions). @tbl-phase3-visc summarizes the per-fuel MAPE for viscosity prediction under the functional group representation, which achieved the best overall performance among the three.

#figure(
  table(
    columns: 3,
    align: (left, center, center),
    stroke: 0.5pt,
    table.header(
      [*Fuel*], [*Temps.*], [*MAPE (%)*],
    ),
    [HRJ-Camelina (7720)], [233, 253 K], [51.2],
    [Shell SPK (5729)], [253 K], [30.6],
    [Sasol IPK (7629)], [233, 253 K], [73.4],
    [Gevo ATJ (10151)], [233, 253 K], [60.1],
    table.hline(),
    [*Average*], [], [*53.8*],
  ),
  caption: [Phase 3 viscosity MAPE by fuel using functional group features. Measurements at low temperatures (233--253 K) are the most challenging and the most operationally relevant.],
) <tbl-phase3-visc>

The 53.8% average MAPE is unsatisfactory for engineering use but informative as a baseline. Several factors contribute to the poor performance. The functional group representation collapses a 95-dimensional NMR spectrum into only 9 features, discarding fine structural detail that distinguishes, for example, normal alkanes from branched isomers. More critically, the training data consist entirely of pure compounds, and the model has no mechanism to account for mixture effects such as non-ideal mixing behavior or the statistical averaging inherent in a superposition spectrum. The functional group approach performs substantially better on surface tension, where the NMR spectral feature set achieves MAPEs between 1.9% (Shell SPK) and 12.9% (Jet-A), suggesting that surface tension is less sensitive to the fine structural details lost in the functional group reduction @Burnett2024NMRProperty.


=== Phase 4: Transfer Learning Pipeline <phase4>

Phase 4 introduces a two-stage transfer learning architecture designed to recover molecular-level information from NMR spectra without requiring explicit molecular structure. The core idea is to pretrain a neural network on the large NMRexp database @Wang2025NMRexp, teaching it to predict molecular descriptors from NMR features alone, and then use those predicted descriptors as enriched features for viscosity regression.

==== The Data Bottleneck: Overlap Analysis <phase4-overlap>

Before constructing the Phase 4 pipeline, it was necessary to quantify the actual volume of training data available at the intersection of NMR spectra and viscosity measurements. An overlap analysis was performed across all NMR databases (NMRexp, NMRBank, NMRShiftDB2, SDBS, NP-MRD, HMDB, BMRB, and several smaller sources) and all viscosity databases (ThermoML, the J. Cheminformatics 2024 dataset, and group-contribution predictions from the chemicals library). Each source was reduced to a set of canonical SMILES strings, and pairwise intersections were computed.

The results reveal a severe data bottleneck. The union of all NMR sources contains approximately 1.8 million unique compounds, and the union of all viscosity sources contains approximately 2,238 unique compounds. However, only 1,330 compounds appear in both collections. After filtering for valid NMR peak lists, successful feature extraction, and reasonable viscosity ranges ($10^(-5)$ to 30 Pa$dot$s, 200--500 K, atmospheric pressure), the final training set comprises 953 unique compounds contributing 10,633 measurement rows. Of these compounds, the large majority originate from the NMRexp database, with smaller contributions from NMRShiftDB2 and NMRBank.

This 953-compound training set is heavily biased toward polar, drug-like molecules because NMRexp draws predominantly from pharmaceutical and metabolomics literature @Wang2025NMRexp. Only a small fraction of these compounds are hydrocarbons, creating a distribution mismatch with the target application domain of aviation fuels, which are almost exclusively composed of alkanes, cycloalkanes, and aromatics.

==== Phase 4 v1: XGBoost Two-Stage Baseline <phase4-v1>

The Phase 4 v1 model establishes a baseline for the two-stage architecture using XGBoost @Chen2016 for both stages. In Stage 1, an XGBoost regressor is trained on the 953 viscosity compounds to predict eight RDKit molecular descriptors (molecular weight, LogP, topological polar surface area, heavy atom count, hydrogen count, rotatable bond count, aromatic ring count, and number of NMR peaks) from the 120-dimensional NMR feature vector. In Stage 2, a second XGBoost regressor takes the concatenation of NMR features (120 dimensions), predicted descriptors (8 dimensions), and temperature features (3 dimensions) as input and predicts log viscosity.

This approach improves substantially over Phase 3, reducing the average fuel MAPE from 53.8% to 39.4%. The improvement confirms that the two-stage decomposition provides useful inductive bias: by first predicting interpretable molecular properties and then using those properties alongside raw NMR features for viscosity prediction, the model gains access to structural information that is not directly accessible from the superposition spectrum of a mixture.

==== GPU Pretraining: NMR to Descriptors MLP <phase4-pretrain>

The limitation of v1 is that Stage 1 is trained on only 953 compounds, far too few to learn a robust mapping from NMR spectra to molecular descriptors. Phase 4 addresses this by pretraining a multi-layer perceptron (MLP) on the full NMRexp database. The pretraining set contains approximately 296,000 compounds for which both valid $""^1"H"$ NMR peak lists and RDKit-computable descriptors are available.

The MLP architecture consists of three hidden layers (256, 256, 128 neurons) with batch normalization, ReLU activation, and 20% dropout after each hidden layer. The input is a 120-dimensional NMR feature vector (95 spectrum features plus 25 peak-list features), and the output is a simultaneous prediction of all eight target descriptors. Training uses the AdamW optimizer with a learning rate of $10^(-3)$, cosine annealing to $10^(-5)$ over 30 epochs, a batch size of 4,096, and a 90/10 train/validation split. The model converges to an average validation $R^2$ of approximately 0.85 across the eight descriptors, indicating that NMR features carry substantial information about molecular properties even in the absence of explicit structural input.

==== The Distribution Shift Problem <phase4-dist-shift>

Deploying the pretrained MLP directly on the 953 viscosity compounds (Phase 4 v2) yields an average fuel MAPE of 44.9%, which is worse than the XGBoost-only v1 baseline of 39.4%. Diagnosing the failure reveals a severe distribution shift between the pretraining domain and the target domain. The pretrained MLP was trained on 296,000 compounds dominated by heteroatom-rich pharmaceutical molecules with high TPSA values, multiple aromatic rings, and complex substitution patterns. The viscosity training set, and aviation fuels in particular, consists largely of simple hydrocarbons with zero TPSA, zero or few aromatic rings, and high LogP values.

The mismatch manifests as physically unreasonable predictions when the pretrained MLP is applied to hydrocarbon NMR spectra. For example, the pretrained model predicts nonzero TPSA and incorrectly estimates the number of aromatic rings for pure alkanes. Evaluated on the 943 viscosity compounds before fine-tuning, the pretrained MLP achieves $R^2 = -0.99$ for molecular weight prediction, meaning its predictions are worse than simply predicting the mean value. This catastrophic failure on the most basic molecular descriptor demonstrates that the pretrained model has not learned a domain-general NMR-to-structure mapping; rather, it has learned correlations specific to the pharmaceutical chemical space of NMRexp.


==== Phase 4 v3: Fine-Tuned MLP (Best Result) <phase4-v3>

Phase 4 v3 resolves the distribution shift by fine-tuning the pretrained MLP on the 943 viscosity compounds before using it as a descriptor predictor. The fine-tuning procedure maintains the pretrained NMR feature scaling (since NMR spectral statistics are relatively domain-invariant) but re-fits the target scalers on the viscosity compound distribution, where descriptors like TPSA and LogP have different means and variances than in the pharmaceutical pretraining set. A differential learning rate strategy is employed: backbone layers are updated at a learning rate of $10^(-4)$, while the output head receives $10^(-3)$, allowing the final layer to adapt rapidly to the new target distribution without destroying the feature representations learned during pretraining.

Fine-tuning proceeds for 150 epochs with a batch size of 256, a 15% validation hold-out, and early stopping by best validation loss. After fine-tuning, the molecular weight $R^2$ on viscosity compounds improves from $-0.99$ to a positive value, and predictions for TPSA and aromatic ring count become physically reasonable for hydrocarbons (near-zero for pure alkanes).

The fine-tuned MLP descriptors, concatenated with NMR features and temperature, are then used to train the Stage 2 XGBoost viscosity model using the same architecture as v1. @tbl-phase4-compare shows the per-fuel viscosity MAPE across all Phase 4 versions.

#figure(
  table(
    columns: 6,
    align: (left, center, center, center, center, center),
    stroke: 0.5pt,
    table.header(
      [*Fuel*], [*Phase 3*], [*v1*], [*v2*], [*v3*], [*v4*],
    ),
    table.hline(),
    [HRJ-Camelina (7720)], [51.2], [--], [--], [--], [--],
    [Shell SPK (5729)], [30.6], [--], [--], [--], [--],
    [Sasol IPK (7629)], [73.4], [--], [--], [--], [--],
    [Gevo ATJ (10151)], [60.1], [--], [--], [--], [--],
    table.hline(),
    [*Average*], [*53.8*], [*39.4*], [*44.9*], [*29.3*], [*55.8*],
  ),
  caption: [Average fuel viscosity MAPE (%) across Phase 3 and Phase 4 versions. Phase 3 uses functional groups; v1 uses XGBoost for both stages; v2 uses the raw pretrained MLP; v3 fine-tunes the MLP on viscosity compounds; v4 adds hydrocarbon and cold-temperature sample weighting. Per-fuel breakdowns for v1, v2, and v4 are not preserved in the output logs, but the averages are reported from the pipeline summaries.],
) <tbl-phase4-compare>

The v3 pipeline achieves an average MAPE of 29.3%, reducing the error by 46% relative to the Phase 3 functional group baseline and by 26% relative to the Phase 4 v1 XGBoost-only approach. This result confirms that the pretrained MLP provides useful inductive bias, even on a very different chemical domain, once it is adapted to the target distribution through fine-tuning. The fine-tuned model effectively learns to project hydrocarbon NMR spectra into a descriptor space that captures molecular weight, chain length, and branching information relevant to viscosity.

==== Failed Experiments <phase4-failures>

A number of alternative approaches were explored after the v3 result, none of which improved upon the 29.3% average MAPE. These negative results are documented here because they are informative about the nature of the problem.

*Hydrocarbon and cold-temperature sample weighting (v4).* Because aviation fuels are predominantly hydrocarbons and the most operationally important predictions are at low temperatures (233--253 K), v4 applied sample weights to the XGBoost training: hydrocarbon compounds received 3$times$ weight, and cold-temperature measurements received up to 5$times$ weight via a linear ramp from 1$times$ at 300 K to 5$times$ at 220 K. The expectation was that focusing the model on the relevant region of chemical and temperature space would improve fuel predictions. Instead, the average MAPE increased to 55.8%, likely because the weighting scheme reduces the effective training set size by concentrating gradient updates on a small subset of compounds while deprioritizing the diverse polar compounds that help the model learn general viscosity--structure relationships.

*Arrhenius-parameterized neural network.* A physics-informed model was tested that predicted the Arrhenius parameters $A$ and $B$ in @arrhenius directly from NMR features, rather than predicting log viscosity at each temperature independently. This approach achieved a 66% average MAPE, worse than all other variants. The failure likely stems from the sensitivity of Arrhenius extrapolation to small errors in the fitted parameters: a modest error in $B$ translates into large viscosity errors at the extreme temperatures (233 K) where fuel predictions are validated.

*MLP embedding features.* Instead of using the MLP's output predictions as features, intermediate hidden-layer activations (128-dimensional embeddings from the last hidden layer) were extracted and concatenated with NMR features for viscosity prediction. This approach achieved 55% MAPE, suggesting that the hidden representations do not provide a more informative feature space than the explicit descriptor predictions.

*Cold-temperature data augmentation.* The training set contains only 355 measurements below 260 K out of 10,633 total rows, creating a severe cold-temperature data gap. An augmentation strategy was tested that generated synthetic low-temperature viscosity points by extrapolating Arrhenius fits for compounds with multiple temperature measurements. The augmented model achieved 58% MAPE, indicating that the extrapolated data introduced more noise than signal, which is consistent with the known breakdown of simple Arrhenius behavior near glass transition temperatures for complex molecules.

*Ensemble of v3 and weighted models.* An ensemble averaging the predictions of the v3 model and the v4 weighted model achieved 34% MAPE, slightly worse than v3 alone. The weighted model's large errors on specific fuels dragged down the ensemble average.


==== Phase 4 v5: Expanded Training Data with Synthetic NMR <phase4-v5>

A natural hypothesis for improving fuel predictions is that more training data should help, particularly if the additional compounds are hydrocarbons. The v5 experiment tested this by expanding the training set from 953 compounds (all with real NMR spectra) to 6,223 compounds (1,144 with real NMR and 5,079 with synthetic NMR). The synthetic NMR features were generated from SMILES using a substructure-based chemical shift prediction module that assigns approximate $""^1"H"$ shifts based on standard NMR lookup tables @Silverstein2005. Each hydrogen atom in the molecule is matched against a prioritized list of SMARTS patterns (aldehydes at 9.7 ppm, aromatic protons at 7.2 ppm, aliphatic CH#sub[3] at 0.9 ppm, and so on), and the assigned shifts are broadened and binned through the same feature extraction pipeline used for real NMR data. The synthetic features are approximate, with an accuracy of roughly 0.5--1.0 ppm compared to experimental spectra, but they capture the gross spectral topology of each molecule.

The expanded dataset contains 34,460 measurement rows (compared to 10,633 in v1) and 5,442 rows below 260 K (compared to 355 in v1), substantially improving coverage at low temperatures. Three training configurations were evaluated: model A weighted real NMR data 3$times$ higher than synthetic; model B used all data with uniform weights; and model C trained on real NMR data only (1,144 compounds, slightly more than the original 953 due to the expanded viscosity source coverage). The best of the three, model B (unweighted), achieved an average MAPE of 41.3%. All three configurations performed worse than v3's 29.3%.

This counterintuitive result, where more data degrades performance, is explained by the noise characteristics of synthetic NMR features. The fine-tuned MLP was trained exclusively on real NMR data, and when presented with synthetic spectra at inference time, it produces descriptor predictions with systematic biases that the Stage 2 XGBoost model has not been trained to correct. The synthetic spectra lack the fine structure (coupling patterns, solvent effects, conformational averaging) that the MLP learned to exploit during pretraining, and the resulting descriptor errors compound through the two-stage pipeline. Even model C, which trains Stage 2 on real NMR data only but drawn from a slightly different compound set, fails to match v3, confirming that the specific 953-compound training set and its tight coupling with the fine-tuned MLP represent a local optimum that is difficult to improve upon with the available data.


=== The Fundamental Bottleneck <bottleneck>

The progression from Phase 3 (53.8% MAPE) through Phase 4 v3 (29.3% MAPE) demonstrates that transfer learning from large NMR databases can substantially improve fuel property prediction, but the final accuracy remains insufficient for engineering certification, where viscosity errors below 5--10% are typically required @ASTMD1655. The fundamental bottleneck is data quality at the NMR-viscosity intersection. Out of approximately 1.8 million compounds with experimental NMR spectra, only 953 also have viscosity measurements and valid NMR features. This tiny overlap set is dominated by polar, drug-like molecules rather than the hydrocarbons that constitute aviation fuels. The pretrained MLP learns strong NMR-to-descriptor mappings on 296,000 compounds, but those mappings degrade catastrophically outside the pharmaceutical domain, and fine-tuning on 953 compounds provides only a partial correction.

Three specific gaps limit further progress. First, the NMR databases contain very few simple hydrocarbons because these compounds are of limited interest to the pharmaceutical and metabolomics communities that generate most NMR data. Second, the viscosity databases contain limited coverage below 260 K, precisely the temperature range that matters most for jet fuel pumpability at altitude. Third, the two-stage architecture introduces compounding errors: any inaccuracy in Stage 1 descriptor prediction propagates into and is amplified by Stage 2 viscosity regression. Closing these gaps would require either targeted acquisition of hydrocarbon NMR spectra with paired viscosity measurements, or the development of models that can learn directly from mixture spectra without relying on pure-compound training data as an intermediary.

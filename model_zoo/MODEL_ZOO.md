<div align="center">

# Shape2Fate Model Zoo

</div>

All pretrained model checkpoints used in the Shape2Fate pipeline are listed below. Each checkpoint is stored in this directory and can be loaded directly with PyTorch.

## Model Family Details

- **Developed by:** Shape2Fate authors (Harmanec et al.)
- **Repository:** https://github.com/harmanea/shape2fate
- **Project website:** https://shape2fate.utia.cas.cz/
- **Paper:** Harmanec, A. et al. (2026). *Shape2Fate: a morphology-aware deep learning framework for tracking endocytic and exocytic carriers at nanoscale.* bioRxiv. https://doi.org/10.64898/2026.03.29.715120
- **Primary modality:** live-cell TIRF-SIM microscopy
- **Framework:** PyTorch 1.13.1
- **License:** BSD 3-Clause License

## Intended Uses

### Direct use

These checkpoints are intended for research use within the Shape2Fate workflow on membrane-trafficking carriers imaged by TIRF-SIM or closely matched data:

- detection and morphology inference for **clathrin-coated pits (CCPs)**,
- detection of **exocytic carriers** and **candidate fusion events**,
- secondary filtering of candidate exocytic fusion events using a temporal classifier.

### Downstream use

Reasonable downstream uses include:

- reproducing the example workflows distributed with Shape2Fate,
- retraining or adapting detectors using the provided synthetic-data generator,
- adapting the fusion classifier to a new reporter, cell type, or microscope setting using newly annotated real trajectories.

### Out-of-scope use

These checkpoints are **not** validated for:

- clinical or diagnostic decision-making,
- conventional TIRF, confocal, widefield, or electron microscopy without adaptation,
- biological structures outside the morphology classes modelled in the simulator,
- direct quantification of molecular mechanisms below the spatiotemporal resolution of the input data,
- standalone use without quality control when the data differ strongly from the microscopes, reporters, temporal sampling, or carrier morphologies studied in the paper.

##  Bias, Risks, and Limitations

- The detector checkpoints are trained primarily on **synthetic data** generated from an explicit forward model of TIRF-SIM image formation. They therefore inherit the coverage limits of the simulator.
- Carrier morphologies not represented in the simulator (for example caveolar rosettes, tubular networks, or large static flat lattices) may be missed or misclassified.
- Very brief productive exocytic events shorter than a frame, or objects separated by less than the effective resolution limit, may not be resolved reliably.
- The exocytosis fusion classifier is trained on **real annotated data** from the systems studied in the manuscript; transfer to substantially different reporters or assays may require retraining.
- Human-annotated microscopy trajectories are not an absolute ground truth in dense scenes; evaluation should be interpreted in that context.

### Recommendations

- Start with the example scripts or Colab notebooks before running on new data.
- Use the checkpoint that matches the biological regime: standard CME, adipocyte CME, exocytosis detection, or exocytosis fusion filtering.
- Visually inspect reconstructions, detections, and trajectories on representative movies before drawing biological conclusions.
- If your structures fall outside the morphology range studied in the manuscript, extend the synthetic generator and retrain rather than relying on zero-shot transfer.

## How to Get Started

Install the package as described in the repository `README.md`, then load the checkpoint with PyTorch. Minimal examples are included in each checkpoint section below.

---

## CCP Detector — `ccp-detector-sandy-wildflower-269.pt`

### Model Details

|                              |                                                                                   |
|------------------------------|-----------------------------------------------------------------------------------|
| **Model type**               | 2D U-Net detector                                                                 |
| **Task**                     | Clathrin-coated pit (CCP) detection in TIRF-SIM movies                            |
| **Architecture**             | UNet (`depth=3`, `start_filters=16`, `up_mode='nearest'`)                         |
| **Parameters**               | 450,193                                                                           |
| **Input**                    | Single-channel TIRF-SIM grayscale image (1 &times; H &times; W)                   |
| **Output**                   | Single-channel normalized interior-distance map (1 &times; H &times; W)           |
| **Interpretation of output** | Local maxima define CCP centres; peak values encode the Shape Index (SI)          |

### Direct Use

Use this checkpoint for CME analysis in the standard CCP regime described in the manuscript, especially for RPE-1-like pit sizes and imaging settings similar to the validation and testing datasets.

### Out-of-Scope Use

Do not use this checkpoint as the default choice for adipocyte CCPs, exocytic carriers, or non-TIRF-SIM data.

### Training Data

This checkpoint was trained on **synthetic CCP images** generated by `shape2fate.synthetic_data.SyntheticCCPDataset`.

The synthetic data were designed to emulate:

- compact and annular CCP morphologies,
- TIRF-SIM image formation,
- noise, motion, and reconstruction artefacts,
- varied SNR and clustering patterns.

No manual framewise detector annotations from experimental images were required for this checkpoint.

### Training Procedure

- **Training regime:** supervised learning on synthetic reconstructed TIRF-SIM patches
- **Loss:** binary cross-entropy on the normalized interior-distance-map target
- **Optimizer:** AdamW
- **Learning rate:** `1.6e-4`
- **Weight decay:** `0.05`
- **Batch size:** `16`
- **Training length:** `300` cycles, `100` batches per cycle
- **Preprocessing:** per-frame normalization to zero mean and unit standard deviation

### Evaluation

#### Testing data, factors, and metrics

This checkpoint is evaluated in the paper **as part of the full CME pipeline**, not as an isolated detector.

Pipeline-level evaluation used:

- a publicly available **CME tracking validation** dataset,
- an independently acquired **CME tracking testing** dataset from a different commercial microscope platform,
- tracking metrics including MOTA, 1-MOTP, HOTA, DetA, AssA, and μTIOU.

#### Results

Standalone detector-only metrics are **not separately reported** in the manuscript.

As part of the full CME pipeline, Shape2Fate:

- achieved performance within the inter-annotator range on the validation benchmark,
- generalized to an independent CME testing dataset acquired on a different platform,
- reached human-level performance on difficult cases in blinded paired comparison.

### Known Limitations

- This checkpoint assumes the CCP size regime used for the standard CME detector.
- Performance may degrade if CCP radii are systematically larger than those seen in the standard synthetic generator.
- It is not a segmentation model for arbitrary membrane structures.

### Example Usage

```python
import torch
import shape2fate.models as models

model = models.UNet(depth=3, start_filters=16, up_mode='nearest')
checkpoint = torch.load('model_zoo/ccp-detector-sandy-wildflower-269.pt', map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()
```

---

## Adipocyte CCP Detector — `ccp-detector-sandy-feather-310.pt`

### Model Details

|                              |                                                                                            |
|------------------------------|--------------------------------------------------------------------------------------------|
| **Model type**               | 2D U-Net detector                                                                          |
| **Task**                     | Clathrin-coated pit (CCP) detection in adipocyte TIRF-SIM movies                           |
| **Architecture**             | UNet (`depth=3`, `start_filters=16`, `up_mode='nearest'`)                                  |
| **Parameters**               | 450,193                                                                                    |
| **Input**                    | Single-channel TIRF-SIM grayscale image (1 &times; H &times; W)                            |
| **Output**                   | Single-channel normalized interior-distance map (1 &times; H &times; W)                    |
| **Interpretation of output** | Local maxima define CCP centres; peak values encode the adipocyte-adapted Shape Index (SI) |

### Direct Use

Use this checkpoint for adipocyte CME analysis, where CCPs are larger than in the standard RPE-1-like regime.

### Out-of-Scope Use

Do not assume this checkpoint is the best default for non-adipocyte CME or for exocytic carriers.

### Training Data

This checkpoint was trained on **synthetic CCP images adapted for adipocytes** using `shape2fate.synthetic_data.SyntheticAdipocytesCCPDataset`.

Compared with the standard CCP detector, the synthetic generator was adjusted to cover a larger CCP size range consistent with adipocyte measurements.

### Training Procedure

- **Training regime:** supervised learning on synthetic reconstructed TIRF-SIM patches
- **Loss:** binary cross-entropy on the normalized interior-distance-map target
- **Optimizer:** AdamW
- **Learning rate:** `1.6e-4`
- **Weight decay:** `0.05`
- **Batch size:** `16`
- **Training length:** `300` cycles, `100` batches per cycle
- **Preprocessing:** per-frame normalization to zero mean and unit standard deviation

### Known Limitations

- This checkpoint is specialized to the adipocyte CCP size regime studied in the paper.
- Transfer to other cell types with different CCP sizes or morphologies should be checked empirically.

### Example Usage

```python
import torch
import shape2fate.models as models

model = models.UNet(depth=3, start_filters=16, up_mode='nearest')
checkpoint = torch.load('model_zoo/ccp-detector-sandy-feather-310.pt', map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()
```

---

## Exocytosis Detector — `exo-detector-rural-wind-13.pt`

### Model Details

|                              |                                                                                                        |
|------------------------------|--------------------------------------------------------------------------------------------------------|
| **Model type**               | 2D U-Net detector                                                                                      |
| **Task**                     | Detection of exocytic carriers in exocytosis TIRF-SIM movies                                           |
| **Architecture**             | UNet (`out_channels=2`, `depth=3`, `start_filters=16`, `up_mode='nearest'`)                            |
| **Parameters**               | 450,210                                                                                                |
| **Input**                    | Single-channel TIRF-SIM grayscale image (1 &times; H &times; W)                                        |
| **Output**                   | Two-channel normalized interior-distance map (2 &times; H &times; W)                                   |
| **Interpretation of output** | One channel represents intact vesicles/carriers; the second channel represents candidate fusion events |

### Direct Use

Use this checkpoint to detect pleomorphic exocytic carriers and candidate fusion events in TIRF-SIM exocytosis workflows that resemble the RUSH or adipocyte systems studied in the manuscript.

### Out-of-Scope Use

This checkpoint is not intended to be a final standalone productivity classifier. In the manuscript, candidate fusion calls are further filtered by the auxiliary fusion classifier.

### Training Data

This checkpoint was trained on **synthetic exocytosis images** generated by `shape2fate.synthetic_data.SyntheticExocytosisDataset`.

The synthetic generator includes:

- approximately spherical vesicles,
- elongated tubular carriers,
- simulated fusion flashes represented as isotropic signal bursts,
- simulated filopodia negatives,
- imaging artefacts, motion, background structure, and noise.

### Training Procedure

- **Training regime:** supervised learning on synthetic reconstructed TIRF-SIM patches
- **Loss:** binary cross-entropy on two output channels
- **Optimizer:** AdamW
- **Learning rate:** `1.6e-4`
- **Weight decay:** `0.05`
- **Batch size:** `16`
- **Training length:** `300` cycles, `100` batches per cycle
- **Preprocessing:** per-frame normalization to zero mean and unit standard deviation

### Evaluation

#### Testing data, factors, and metrics

This checkpoint is evaluated in the paper **as part of the full exocytosis tracking pipeline** on an expert-annotated RUSH test set.

Pipeline-level metrics reported for exocytosis tracking are:

- **MOTA:** `0.53`
- **1-MOTP:** `0.69`
- **HOTA:** `0.72`
- **DetA:** `0.67`
- **AssA:** `0.77`
- **μTIOU:** `0.45`

#### Results

The paper reports that Shape2Fate tracked the arrival and fusion of most annotated exocytic carriers on the RUSH test set (`745` carrier trajectories; approximately `25,000` detections).

These are **integrated pipeline metrics**, not isolated detector metrics.

### Known Limitations

- The exocytosis detector is intentionally tuned to favor sensitivity, so candidate fusion detections may include artifact-driven false positives before secondary filtering.
- Transfer to exocytic systems with substantially different carrier morphology or reporter dynamics may require retraining.

### Example Usage

```python
import torch
import shape2fate.models as models

model = models.UNet(out_channels=2, depth=3, start_filters=16, up_mode='nearest')
checkpoint = torch.load('model_zoo/exo-detector-rural-wind-13.pt', map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()
```

---

## Exocytosis Fusion Classifier — `exo-fusion-detector-giddy-yogurt-32.pt`

### Model Details

|                  |                                                                                                                                                                        |
|------------------|------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| **Model type**   | CNN + Transformer sequence classifier                                                                                                                                  |
| **Task**         | Classification of exocytic fusion productivity from tracked carrier trajectories                                                                                       |
| **Architecture** | TransformerModel (`cnn_filters=[8, 24]`, `pe_dropout=0.1`, `pe_max_len=45`, `n_attention_heads=4`, `dim_feedforward=48`, `num_transformer_layers=3`, `fc_dropout=0.5`) |
| **Parameters**   | 22,377                                                                                                                                                                 |
| **Input**        | Sequence of raw unreconstructed TIRF-SIM image crops around a tracked carrier (batch &times; seq &times; H &times; W)                                                  |
| **Output**       | Scalar fusion score per trajectory                                                                                                                                     |

### Direct Use

Use this checkpoint as a **secondary filter** after exocytosis detection and linking. In the Shape2Fate workflow it suppresses artifact-driven false positives among candidate fusion events.

### Out-of-Scope Use

This model is not intended to serve as a general-purpose exocytosis detector on arbitrary movies without upstream candidate generation.

### Training Data

Unlike the detector checkpoints, this model was trained on **real annotated exocytic trajectories**, because the manuscript deliberately avoids imposing pathway-specific temporal priors in the synthetic generator.

Training data reported in the manuscript:

- **Training set:** `4,192` trajectories from `34` cells
  - `18` adipocyte cells
  - `16` RUSH cells
- **Validation set:** `1,571` trajectories from `12` independent cells
  - `5` adipocyte cells
  - `7` RUSH cells

Reported class counts:

- **Adipocytes (train):** `2,532` negative / `712` positive
- **Adipocytes (validation):** `719` negative / `183` positive
- **RUSH (train):** `422` negative / `526` positive
- **RUSH (validation):** `346` negative / `323` positive

### Training Procedure

- **Training regime:** supervised binary sequence classification on real candidate fusion trajectories
- **Preprocessing:** extraction of centered `21 x 21` raw TIRF crops from the last five TIRF-SIM time points for each candidate event
- **Optimizer:** AdamW
- **Learning rate:** `1e-4`
- **Weight decay:** `0.05`
- **Batch size:** `8`
- **Training length:** `300` epochs
- **Loss:** binary cross-entropy

### Evaluation

#### Testing data, factors, and metrics

The manuscript compares this model against the pre-trained **IVEA** model under matched TIRF-resolution input conditions.

Reported validation metrics for Shape2Fate:

- **Accuracy:** `0.93`
- **Precision:** `0.89`
- **Recall:** `0.90`
- **F1:** `0.90`

Reported comparison values for IVEA (as listed in the manuscript supplementary table):

- **IVEA raw:** accuracy `0.68`, precision `0.15`, recall `0.59`, F1 `0.23`
- **IVEA avg 3:** accuracy `0.74`, precision `0.26`, recall `0.87`, F1 `0.40`
- **IVEA slid. win.:** accuracy `0.67`, precision `0.03`, recall `0.89`, F1 `0.06`

#### Results

The manuscript reports that, under matched TIRF-resolution conditions, the Shape2Fate fusion classifier substantially outperformed the pre-trained IVEA model for productive exocytosis classification.

### Known Limitations

- The model is trained on the two exocytic systems used in the paper and may require retraining for different reporters, temporal sampling schemes, or microscope regimes.
- Because it evaluates only the final candidate window, upstream candidate quality still affects downstream performance.
- This checkpoint was validated on independent cells, but the manuscript does not report a separate final held-out test set beyond the reported validation comparison.

### Example Usage

**Usage:**
```python
import torch
import shape2fate.models as models

model = models.TransformerModel(
    cnn_filters=[8, 24],
    pe_dropout=0.1,
    pe_max_len=45,
    n_attention_heads=4,
    dim_feedforward=48,
    num_transformer_layers=3,
    fc_dropout=0.5
)
checkpoint = torch.load('model_zoo/exo-fusion-detector-giddy-yogurt-32.pt', map_location='cpu')
model.load_state_dict(checkpoint)
model.eval()
```

---

## Summary

| Checkpoint                               | Architecture     | Task                                          | Params  |
|------------------------------------------|------------------|-----------------------------------------------|---------|
| `ccp-detector-sandy-wildflower-269.pt`   | UNet             | Endocytosis CCP detection                     | 450,193 |
| `ccp-detector-sandy-feather-310.pt`      | UNet             | Adipocyte CCP detection                       | 450,193 |
| `exo-detector-rural-wind-13.pt`          | UNet (2-ch out)  | Exocytosis detection                          | 450,210 |
| `exo-fusion-detector-giddy-yogurt-32.pt` | TransformerModel | Exocytosis fusion productivity classification | 22,377  |

## Model Card Contact

For questions, feature requests, or bug reports, please use the GitHub repository issue tracker or the contact details provided on the project website and in the manuscript.
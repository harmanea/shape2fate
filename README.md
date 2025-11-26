<div align="center">
  <img src="assets/imgs/head.png" alt="shape2fate header" width="100%" />

<h1>Shape2Fate</h1>
<p><strong>A morphology-aware deep learning framework for tracking endocytic and exocytic carriers in super-resolution.</strong></p>




<p>
<a href="https://shape2fate.utia.cas.cz/" target="_blank"><img src="https://img.shields.io/badge/Project%20Page-Shape2Fate-1a5fd1?style=plastic&logo=webflow&logoColor=white"/></a>
<a href="https://tinyurl.com/shape2fate" target="_blank"><img src="https://img.shields.io/badge/Dataset-Zenodo-1682d4?style=plastic&logo=zenodo&logoColor=white"/></a>
<img src="https://img.shields.io/badge/Citation-Coming%20Soon-b02525?style=plastic&logo=readthedocs&logoColor=white"/>
<img src="https://img.shields.io/badge/Python-3.9%2B-3776ab?style=plastic&logo=python&logoColor=white"/>
</p>
</div>

## Table of contents
- [Overview](#overview)
  - [Abstract](#abstract)
  - [Associated paper](#associated-paper)
  - [Key capabilities](#key-capabilities)
- [Quick start](#quick-start)
  - [Try in Colab](#try-in-colab)
  - [Install locally](#install-locally)
- [Repository tour](#repository-tour)
- [Datasets](#datasets)
- [How to cite](#how-to-cite)
- [Contact](#contact)


## Overview

### Abstract
Plasma membrane homeostasis depends on balanced exocytosis and endocytosis, yet their spatiotemporal coordination has been difficult to resolve. We present Shape2Fate, a fully automated, shape-aware deep-learning pipeline that resolves individual exocytic and endocytic carriers in live-cell TIRF-SIM movies. Unlike conventional intensity-based approaches, Shape2Fate exploits morphology: it detects trafficking carriers, tracks them across lifetime shape changes, and classifies cargo-delivery outcomes from their shape evolution. Trained entirely on realistic synthetic data, Shape2Fate reconstructs complete trajectories in crowded scenes and matches expert-level tracking accuracy on experimental data. Using Shape2Fate, we uncover distinct coupling architectures across local and global scales and in spatial and temporal hierarchies: exocytic fusion can initiate clathrin‑coated pits (CCPs), whereas in other contexts exocytic carriers target pre‑existing CCP hotspots for immediate cargo capture. As an open‑source, broadly applicable framework, Shape2Fate yields quantitative, event‑level maps of membrane trafficking, revealing organizing principles and enabling mechanistic dissection across diverse cellular contexts.

🔬 TIRF-SIM → 🧩 Reconstruction → 🎯 Detection → 🔗 Linking → 🧭 Analysis → 📊 Metrics

### Associated paper
Manuscript in preparation. DOI and journal link will be added here when available.

### Key capabilities
- Structured illumination microscopy (SIM) reconstruction with automatic parameter estimation and optional GPU acceleration.
- Shape-aware detection, tracking, and outcome classification for endocytic and exocytic carriers in TIRF-SIM time series.
- Ready-to-run examples that mirror the validation experiments, producing reconstructed movies, detections, trajectories, and summary metrics.

## Quick start
### Try in Colab
<div align="center">
  <img src="assets/imgs/colab_prtsc.png" alt="shape2fate Colab preview" width="85%" />
</div>
<br />

Jump straight into the workflows in your browser, each notebook is preloaded with example data and the necessary dependencies:
- <a href="https://colab.research.google.com/drive/1e4JXQfR0QsalDeyKlTbp1krUh5z8drFo?usp=sharing"><img src="https://img.shields.io/badge/Colab-Endocytosis-f9b234?style=plastic&logo=googlecolab&logoColor=white"/></a> Reconstruct, detect, and track endocytic events in TIRF-SIM data.
- <a href="https://colab.research.google.com/drive/1BOhpfVb5DBjqGEuZb3WZhJKuMizBQeJA?usp=sharing"><img src="https://img.shields.io/badge/Colab-Exocytosis-f9b234?style=plastic&logo=googlecolab&logoColor=white"/></a> Reconstruct, detect, and track exocytic events in TIRF-SIM data.
- <a href="https://colab.research.google.com/drive/1DYNfea3Jl1lQza07vOoDMsKSFvylYnIc?usp=sharing"><img src="https://img.shields.io/badge/Colab-Detector-f9b234?style=plastic&logo=googlecolab&logoColor=white"/></a> Train a custom detector for clathrin-coated pits.

### Install locally

1. Requirements

   Before installing shape2fate, make sure you have:

   * Python 3.9–3.12
   * Git installed on your system
   * (Optional) A CUDA-capable GPU for GPU acceleration

2. Clone the repository

   ```bash
   git clone https://github.com/harmanea/shape2fate.git
   cd shape2fate
   ```

3. (Optional) Create and activate a virtual environment

   ```bash
   python -m venv .venv
   source .venv/bin/activate   # On Windows: .venv\Scripts\activate
   ```

4. Install the package

   ```bash
   pip install .
   ```

   To include optional I/O dependencies for reading various microscopy image formats, use:

   ```bash
   pip install .[io]
   ```

   **NOTE:** Optional dependencies are required to run the example and benchmarking scripts.

5. Run a quick test

   ```bash
   python -c "import shape2fate; print(shape2fate.__version__)"
   ```

   If the installation was successful, this command will print a Shape2Fate version number such as: `0.1.0`

6. (Optional) Run a SIM reconstruction example

   ```bash
   python examples/reconstruction_example.py
   ```

   This will:

   * download an example **raw TIRF-SIM endocytosis dataset**,
   * estimate the SIM illumination parameters (frequency, angle, phase, amplitude),
   * run the full SIM reconstruction pipeline (CPU or GPU, depending on availability), and
   * save the outputs into `./data`:

     * `reconstruction.tiff` – the reconstructed TIRF-SIM time series

7. (Optional) Reproduce the tracking results from the paper

   ```bash
   python examples/tracking_example.py
   ```

   This will:

   * download the **validation dataset used in the manuscript**,
   * run the full detection → linking → evaluation pipeline on the example TIRF-SIM movie, and
   * save the outputs into `./data`:

     * `detections.csv` – per-frame CCP detections
     * `trajectories.csv` – linked trajectories after untangling and filtering
     * `metrics.txt` – summary tracking metrics (MOTA, HOTA, μTIOU, …)

    The script will also print a summary of the tracking metrics to verify the reproduction of the performance reported in the paper.

## Repository tour
- `shape2fate/` — core package implementing SIM reconstruction, detection, tracking, metrics, and synthetic-data utilities.
  - `otf.py` — optical transfer function builders and utilities for reconstruction.
  - `parameters.py` — acquisition, reconstruction and linking parameter containers with defaults.
  - `parameter_estimation.py` — automatic estimation of SIM shifts, phases, amplitudes, and frequencies from raw data.
  - `reconstruction.py` — CPU/GPU reconstruction pipeline with preprocessing, padding, filtering, and OTF mapping.
  - `detection.py` — shape-aware exocytosis/endocytosis detection routines and post-processing.
  - `models.py` — deep-learning architectures used for detector training and inference.
  - `linking.py` — trajectory assembly, untangling, filtering, and class-based linking helpers.
  - `metrics.py` — trajectory-level evaluation metrics (MOTA, HOTA, μTIOU, per-class summaries).
  - `synthetic_data.py` — generators for realistic synthetic TIF-SIM movies and labels used for training.
  - `utils.py` — shared utilities for data loading, normalization, padding, and batching.
  - `sim.py` — SIM illumination geometry helpers (diffraction limits, carrier shifts, separation matrices).
- `examples/` — runnable scripts that download sample datasets and reproduce the reconstruction and tracking pipelines from the paper.
  - `reconstruction_example.py` — full SIM reconstruction demo with automatic parameter estimation.
  - `tracking_example.py` — detection, linking, and metrics reporting demo.
- `assets/` — static images used in the README and other documentation.
- `pyproject.toml` — package metadata and dependencies.

## Datasets
Curated training, validation, and demo datasets are now publicly available on Zenodo:
<a href="https://tinyurl.com/shape2fate" target="_blank"><img src="https://img.shields.io/badge/Dataset-Zenodo-1682d4?style=plastic&logo=zenodo&logoColor=white"/></a>

## How to cite
If you use shape2fate in your research, please cite:
- Citation details will be posted here soon.

## Contact
Questions or feedback? Reach out at **shape2fate@utia.cas.cz**.

<div align="center">
  <sub>
    <img src="assets/imgs/logos.png" alt="Shape2Fate partners" width="80%" />
  </sub>
</div>

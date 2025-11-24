# shape2fate

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/harmanea/shape2fate.git
cd shape2fate
```

### 2. (Optional) Create and activate a virtual environment

```bash
python3 -m venv .venv
source .venv/bin/activate   # On Windows: .venv\Scripts\activate
```

### 3. Install the package

```bash
pip install .
```

To include optional I/O dependencies for reading various microscopy image formats, use:

```bash
pip install .[io]
```

**NOTE:** Optional dependencies are required to run the example and benchmarking scripts.

### 4. Run a quick test

```bash
python -c "import shape2fate; print(shape2fate.__version__)"
```

### 5. (Optional) Run a SIM reconstruction example

```bash
python examples/reconstruction_example.py
```

This should

* download an example **raw TIRF-SIM endocytosis dataset**,
* estimate the SIM illumination parameters (frequency, angle, phase, amplitude),
* run the full SIM reconstruction pipeline (CPU or GPU, depending on availability), and
* save the outputs into `./data`:

  * `reconstruction.tiff` – the reconstructed TIRF-SIM time series

### 6. (Optional) Reproduce the tracking results from the paper

```bash
python examples/tracking_example.py
```

This should

- download the **validation dataset used in the manuscript**,
- run the full detection → linking → evaluation pipeline on the example TIRF-SIM movie, and
- save the outputs into `./data`:

  - `detections.csv` – per-frame CCP detections  
  - `trajectories.csv` – linked trajectories after untangling and filtering  
  - `metrics.txt` – summary tracking metrics (MOTA, HOTA, μTIOU, …)

The script will also print a summary of the tracking metrics to verify the reproduction of the performance reported in the paper.  

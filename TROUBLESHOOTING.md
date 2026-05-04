# Troubleshooting

This guide helps diagnose common issues when applying Shape2Fate to your own data. It is organized by pipeline stage so that problems can be checked in order:

1. Reconstruction
2. Detection
3. Linking
4. Analysis

For best results, inspect the output visually at each stage before moving to the next one. Most downstream problems are caused by issues in an earlier step.

---

## 1. Reconstruction

Reconstruction problems usually come from one of three sources:

* incorrect acquisition or reconstruction parameters,
* failed illumination-parameter estimation,
* input data with too little usable signal for stable estimation.

Start by checking the illumination-parameter estimates before tuning reconstruction settings.

---

### 1.1 Parameter estimation

Shape2Fate estimates the illumination pattern parameters needed for SIM reconstruction. If these estimates are wrong, the reconstructed images will usually contain strong artifacts or poor resolution improvement.

#### How to recognize failed parameter estimation

Parameter estimation may have failed if:

* the estimated frequency or orientation is inconsistent across illumination orientations,
* the estimated illumination frequency is far from the expected cutoff frequency,
* the three orientations are not approximately evenly spaced around the unit circle,
* reconstructed images show strong directional artifacts, striping, or poor contrast.

For a typical three-orientation SIM acquisition, the orientations should be approximately 60° or 120° apart.

Use the built-in utility to inspect parameter estimates:

```python
s2f.parameter_estimation.check_parameter_estimates
```

#### Common causes and fixes

##### A. Microscope alignment or acquisition issues

Possible signs:

* illumination orientations are inconsistent,
* raw frames have uneven intensity,
* the sample is not in true TIRF mode,
* there is visible out-of-focus or membrane-distant signal.

Recommended checks:

* run any automated microscope alignment routines,
* ask facility or technical staff to inspect the SIM/TIRF alignment,
* confirm that the sample is imaged in true TIRF mode,
* check that all nine low-resolution raw frames have roughly similar intensity,
* verify that the sample preparation and imaging conditions are suitable for TIRF-SIM.

##### B. Incorrect acquisition parameters

SIM reconstruction is sensitive to microscope-specific acquisition parameters. Check that the values passed to Shape2Fate match the acquisition metadata.

The most important parameters are:

* numerical aperture (`NA`),
* emission wavelength, **not** *excitation wavelength*,
* pixel size at the sample plane.

In code, these are defined with:

```python
s2f.parameters.AcquisitionParameters
```

##### C. Low-content or low-SNR data

Some datasets contain too little reliable spatial content for stable automatic parameter estimation. This is especially common in sparse or low-signal exocytosis data.

Recommended options:

1. Increase `minimum_peak_distance`.

   ```python
   s2f.parameters.ReconstructionParameters.minimum_peak_distance
   ```

   A value around `0.9` is a reasonable starting point. This restricts the search region closer to the cutoff frequency and can stabilize the initial estimate.

   Use this only when the expected illumination pattern frequency is close to the cutoff frequency.

2. Provide approximate illumination shifts.

   Estimate parameters from a content-rich sample, such as beads or another high-SNR dataset, and reuse them as a starting point for lower-content data.

   In code, pass the approximate shifts to:

   ```python
   s2f.parameter_estimation.estimate_parameters(..., approximate_shifts=...)
   ```

   The same function returns `shifts` as the first value in its returned tuple, so it can first be run on a reliable sample and then reused as an approximate estimate for more difficult data.

---

### 1.2 Illumination amplitude estimation

Illumination amplitude estimation can be unstable, especially for low-content datasets. It may underestimate the amplitude, sometimes unevenly across orientations. Using underestimated or inconsistent amplitude values can produce poor reconstructions.

In many cases, it is better to use a fixed amplitude value, such as `0.5` or `1.0`.

The amplitude can be set using the `amplitudes` parameter in either:

```python
s2f.reconstruction.cpu_reconstruct
```

or:

```python
s2f.reconstruction.GPUReconstruction
```

---

### 1.3 SIM reconstruction quality

If parameter estimation looks reasonable but the reconstruction is still poor, check the reconstruction parameters.

#### Common causes of poor reconstruction

##### A. Acquisition or parameter-estimation problems

Before changing reconstruction parameters, confirm that:

* illumination parameters are plausible,
* amplitudes are not severely underestimated,
* amplitudes are not strongly inconsistent between orientations,
* `s2f.parameters.AcquisitionParameters` matches the acquisition metadata.

##### B. Non-standard noise or contrast levels

Shape2Fate was optimized for typical live-cell TIRF-SIM noise and contrast levels. The default Wiener parameter is `0.05`.

If the data is much noisier or much cleaner than expected, adjust:

```python
s2f.parameters.ReconstructionParameters.wiener_parameter
```

General guidance:

* increase the Wiener parameter for noisier data,
* decrease it for high-contrast data where less denoising is needed.

This may require empirical tuning and visual inspection.

---

### 1.4 Reconstruction parameter reference

Most users should only need to adjust a small number of reconstruction parameters. Change the others only if there is a clear reason and the effect has been verified visually.

Parameters are defined in:

```python
s2f.parameters.ReconstructionParameters
```

#### `background_intensity`

Microscopes often add a constant camera offset to the image. This offset should be subtracted before reconstruction.

The default value is `100`.

Some microscopes may use a different offset, such as `50`. Check the acquisition system and adjust accordingly.

#### `wiener_parameter`

Controls regularization in the Wiener reconstruction.

* Higher values: stronger denoising, useful for noisy data.
* Lower values: weaker denoising, useful for high-contrast data.

#### `minimum_peak_distance`

Used during the initial rough illumination-parameter estimate. It masks low spatial frequencies and restricts where peaks are searched.

Increase this value when low-content data produces unstable frequency estimates, especially if the expected illumination frequency is close to the cutoff frequency.

#### `apodization_cutoff`

Controls the maximum spatial frequency passed by the apodization filter. It is related to the expected resolution improvement, typically around 2×.

Change this only in special cases.

#### `apodization_bend`

Controls the shape of the apodization filter. This usually has a minor effect and should normally be left unchanged.

#### `border_fade`

Fades the image edges in the spatial domain to reduce Fourier artifacts.

Adjust this if:

* the image size is very different from the typical 512–1024 px range,
* edge information is essential and should not be faded,
* strong edge artifacts are visible.

#### `rl_deconvolution_iterations`

Controls the number of Richardson–Lucy deconvolution iterations used to improve parameter estimation. See [DOI:10.1038/srep37149](https://doi.org/10.1038/srep37149) for reference. The default should work for most datasets.

Change this only in special cases.

---

## 2. Detection

Detection problems usually appear as missing objects, false positives, incorrect shape-index values, or detections that do not correspond to visible structures.

Before changing detection thresholds or downstream analysis, confirm that the correct model and preprocessing are being used.

---

### 2.1 Incorrect model selection

Make sure the detection model matches the biological structure and dataset.

Check in particular:

* endocytosis vs. exocytosis model,
* standard CCP model vs. adipocyte CCP model,
* expected CCP size range.

Use the adipocyte model for larger CCPs, especially when CCP diameters are greater than approximately 175 nm.

For model-specific intended use, limitations, and out-of-scope cases, see the [Model Cards](model_zoo/MODEL_ZOO.md).

---

### 2.2 Data outside the intended domain

The detector may perform poorly if the input data differs substantially from the data distribution used for training and validation.

Examples include:

* different membrane-trafficking structures with unsupported shapes,
* very different pixel sizes (outside of 0.06–0.08 µm),
* poor focus,
* severe phototoxicity,
* low contrast,
* strong reconstruction artifacts,
* excessive clustering or large clathrin assemblies,
* imaging conditions outside the expected TIRF-SIM regime.

If applying Shape2Fate to new structures, cell types, or imaging conditions, validate the detections carefully before relying on downstream measurements.

---

### 2.3 Incorrect preprocessing

Detection models expect input images normalized to zero mean and unit variance.

The standard detection functions handle this automatically:

```python
s2f.detection.generate_exo_detections
s2f.detection.generate_ccp_detections
```

If using custom code, normalize images with:

```python
s2f.detection.normalize_img
```

---

### 2.4 Incorrect output processing

If you are processing model outputs manually, make sure the same output-processing logic is used as in the standard detection functions:

```python
s2f.detection.generate_exo_detections
s2f.detection.generate_ccp_detections
```

Small changes in thresholding, local maxima detection, connected-component extraction, or class assignment can substantially affect downstream tracking and productivity classification.

---

### 2.5 Clusters and large clathrin assemblies

Dense clusters and large clathrin assemblies are difficult to analyze reliably. If the field of view contains many of them, they may affect detections, trajectories, and downstream statistics.

Recommended checks:

* inspect detections visually,
* verify that clusters are not dominating the analysis,
* check whether downstream measurements change when clustered regions are excluded,
* confirm that cell treatment and imaging conditions minimize photodamage and abnormal clathrin accumulation.

Large clusters are often associated with sample-preparation problems, phototoxicity, or unsuitable imaging conditions.

---

### 2.6 Quantifying unexplained signal

Shape2Fate includes a utility for estimating how much signal is not explained by the detected objects. This can help assess whether the input data is in the expected domain and whether the model is detecting the relevant structures.

TODO: Add link to the unexplained-signal utility or example once available.

---

## 3. Linking

Linking connects detections across frames into trajectories. Most linking issues are caused by one of the following:

* missed or spurious detections from the previous stage,
* unsuitable linking parameters,
* objects moving too far between frames,
* dense clusters or ambiguous object identities,
* applying the default parameters to a new biological regime without validation.

Always inspect both detections and linked tracks before adjusting linking parameters.

---

### 3.1 Distance function

The linking distance function usually combines:

* Euclidean distance between detections,
* optionally, shape-index difference.

The default approach is sufficient for most supported use cases. See the [Tracking Example](examples/tracking_example.py) for implementation details.

There is usually no need to modify the distance function unless adapting Shape2Fate to a substantially different dataset or biological process.

---

### 3.2 Linking parameter reference

Linking parameters are defined in:

```python
s2f.parameters.LinkingParameters
```

#### `birth_death_cost`

Penalty for starting or terminating a trajectory during the matching step.

Suggested defaults:

* endocytosis: `5`,
* exocytosis: `10`.

Changing this parameter can strongly affect trajectory fragmentation and false linking. Adjust only after visual inspection and parameter testing.

#### `edge_removal_cost`

Also referred to as `gamma` in the associated paper. This controls the cost of cutting an edge during the untangling step.

It determines when the optimizer should prefer:

* splitting or merging nodes (where each created or removed node has a cost of `1`), or
* cutting an edge when there are multiple predecessors or successors.

Suggested default is `10`.

Adjust only with extensive validation.

#### `feature_cost_multiplier`

Controls how strongly shape-index differences influence the linking cost.

Suggested default is `1`.

This is mainly relevant for endocytic trajectories where shape evolution is informative for linking.

#### `maximum_distance`

Prunes unlikely links by setting the maximum distance between detections that can be considered for matching.

Suggested defaults:

* endocytosis: `7.5`,
* exocytosis: `15`.

Increase this only if:

* the frame rate is low,
* objects move substantially between frames,
* visual inspection shows that true trajectories are being broken because the object moved beyond the default distance.

Increasing this value can also increase false links, especially in dense regions.

#### `maximum_skipped_frames`

Maximum number of frames that can be skipped when linking detections. This helps recover from occasional missed detections.

Suggested default is `1`.

Increasing this is usually not recommended because it can create incorrect links, especially in dense data.

#### `minimum_length`

Minimum trajectory length retained after linking.

Suggested default is `6`.

Lower values may be needed for very fast or short events, such as some adipocyte exocytosis events. However, lowering this threshold will also retain more short spurious trajectories, often caused by noise.

If short exocytic trajectories are important, use additional filtering, such as the auxiliary exocytosis productivity classifier.

---

### 3.3 Adapting linking to new data

If Shape2Fate is applied to new cell types, imaging conditions, frame rates, or membrane-trafficking processes, linking parameters may need to be re-tuned.

Recommended validation steps:

1. Inspect detections before linking.
2. Inspect linked trajectories visually.
3. Compare several parameter settings on representative movies.
4. Check that trajectory counts, lifetimes, and productivity rates are biologically plausible.
5. Avoid changing several parameters at once unless the effect of each change is understood.

---

## 4. Analysis

Analysis problems are often caused by incorrect assumptions about the data, productivity criteria, or filtering rules. Before interpreting biological results, verify that reconstruction, detection, and linking outputs are reasonable.

---

### 4.1 Productivity classification

Use a productivity criterion that matches the detection model and biological process.

#### CME productivity

For RPE-1 cells or similar CCPs, the default shape-index threshold is `0.7`.

For adipocyte CCPs, which are larger and use a different model, the threshold is `0.45`.

Do not transfer thresholds between models without validation.

#### Exocytosis productivity

For exocytosis, do not rely only on shape-index predictions from the detector. Use the auxiliary productivity classifier, especially for adipocyte exocytosis and other short or artifact-prone fusion events.

---

### 4.2 Check earlier pipeline stages first

Before troubleshooting analysis code, confirm that:

* reconstructions look reasonable,
* detections match visible structures,
* trajectories preserve object identity,
* the number of trajectories is plausible,
* the number of productive trajectories is plausible,
* parameter-estimation and detection checks pass.

Downstream analysis should not be trusted if earlier stages are visibly poor.

---

### 4.3 Filtering and masking

Filtering should match the biological question and the measurement being performed.

Recommended practices:

* use only in-focus, well-adhered regions of the field of view,
* mask background and irrelevant cells or regions,
* exclude trajectories too close to the field-of-view edge when required,
* exclude trajectories not fully contained within the movie when measuring lifetimes or complete event dynamics,
* filter very short trajectories unless short events are explicitly part of the analysis,
* use additional filtering when short trajectories are retained.

If productivity is important, ensure that trajectories are long enough to support reliable classification. For CCP productivity analysis, trajectories should last at least 30 seconds or be fully contained within the movie.

---

## 5. General troubleshooting workflow

When results look unexpected, use the following order:

1. Check raw data quality.
2. Check acquisition parameters.
3. Check illumination-parameter estimates.
4. Inspect reconstructed images.
5. Inspect detections.
6. Inspect linked trajectories.
7. Check productivity classification.
8. Check masks and filtering rules.
9. Only then interpret downstream analysis results.

In general:

* verify results visually at each stage,
* keep track of any non-default parameters,
* change one parameter at a time when troubleshooting,
* validate parameter changes on representative datasets,
* compare trajectory counts and productivity rates against expectations,
* consult the associated paper, especially the Methods section, for implementation details.

For questions or unresolved issues, open a GitHub issue or contact:

```text
shape2fate@utia.cas.cz
```

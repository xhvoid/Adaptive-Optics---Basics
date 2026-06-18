# Shack-Hartmann Wavefront Sensing and Adaptive Optics Simulation Portfolio

[![tests](https://github.com/xhvoid/Shack-Hartmann-AO-Simulation/actions/workflows/ci.yml/badge.svg)](https://github.com/xhvoid/Shack-Hartmann-AO-Simulation/actions/workflows/ci.yml)

A compact Python portfolio project for adaptive optics (AO), wavefront sensing (WFS), wavefront reconstruction, closed-loop correction, and science-facing PSF diagnostics.

The project started from geometric Shack-Hartmann wavefront sensing and Zernike modal reconstruction, then expanded into detector-level centroiding, response-matrix calibration, TSVD regularization, simplified closed-loop AO, exploratory PWFS modelling, a high-order actuator-space AO performance notebook for a 10 m-class telescope-like setup, and lightweight noise / latency / gain-stability trade studies.

This is not intended to be a calibrated observatory AO package. The goal is to make the modelling chain explicit and inspectable:

```text
pupil-plane phase error
→ wavefront-sensor measurement
→ response / interaction matrix
→ regularized reconstruction
→ deformable-mirror or modal correction
→ residual wavefront
→ PSF, Strehl, FWHM, and encircled-energy diagnostics
```

## Project scope

The project demonstrates how wavefront errors can be measured, reconstructed, regularized, and corrected in a compact simulation environment.

The main components are:

* geometric Shack-Hartmann slope sensing;
* Zernike and Zernike-like modal reconstruction;
* detector-level Shack-Hartmann lenslet spot simulation and centroiding;
* response-matrix and interaction-matrix calibration;
* singular-value diagnostics and TSVD regularization;
* simplified deformable-mirror closed-loop AO;
* detector-level closed-loop SH-WFS demonstrations;
* exploratory Fourier-optics PWFS forward modelling;
* high-order actuator-space SH-AO with NIR PSF performance diagnostics;
* photon-noise, read-noise, loop-delay, and gain-stability scans.

The most science-facing notebook, `09_ao_psf_instrument_performance_high_order_ao.ipynb`, upgrades the earlier low-order/modal examples into a high-order actuator-space AO loop using a 48 × 48 Shack-Hartmann sampling grid, a 49 × 49 nominal actuator grid, actuator poke-matrix calibration, TSVD command reconstruction, and J/H/K PSF metrics for a 10 m-class telescope-like configuration. Notebook `10_noise_latency_gain_stability.ipynb` then stress-tests the control loop with compact noise, delay, and gain scans.

## Main features

* Circular pupil generation and Zernike / Zernike-like wavefront synthesis
* Atmospheric-like phase screens with Kolmogorov / von Karman-style spectral structure
* Geometric Shack-Hartmann WFS slope sensing
* Detector-level Shack-Hartmann WFS simulation:

  * lenslet diffraction spots
  * finite detector windows
  * photon noise, read noise, and background
  * centroiding and reference-subtracted centroid shifts
  * detector-level response matrices
* Modal least-squares reconstruction
* Singular-value diagnostics and TSVD / `rcond` regularization
* Sampling-density versus correction-order studies
* Gaussian-influence-function deformable mirror model
* Geometric and detector-level closed-loop AO demonstrations
* Loop-gain, loop-delay, and command-leak diagnostics
* Noise / latency / gain-stability error-budget notebook
* FFT-based PSF calculation and Strehl-ratio diagnostics
* J/H/K science-wavelength PSF metrics:

  * residual OPD RMS
  * Strehl ratio
  * FWHM
  * EE50 and EE80 encircled-energy radii
  * simple analytic field-performance roll-off
* Simplified Fourier-optics PWFS forward model:

  * pyramid-like phase mask
  * four re-imaged pupil images
  * modulation points
  * noisy detector measurement
  * normalized `Sx/Sy` signal maps
  * modal PWFS calibration diagnostics

## Representative figures

### High-order AO correction: NIR PSF sharpening

The latest high-order notebook connects actuator-space AO correction to science-facing NIR image quality. The figure compares diffraction-limited, open-loop seeing-limited, and high-order AO-corrected PSFs.

![High-order AO NIR PSF performance](figures/high_order_ao_jhk_psf.png)

### Closed-loop residual OPD RMS

The high-order SH-AO loop suppresses a full seeing-strength frozen-flow atmospheric sequence. The residual OPD RMS is evaluated after the initial loop-settling period.

![High-order AO residual RMS history](figures/high_order_ao_rms_history.png)

### Wavefront correction maps

The wavefront maps show the open-loop atmospheric phase, the DM correction, and the final closed-loop residual. This illustrates that the controller reconstructs actuator commands rather than only fitting low-order Zernike modes.

![High-order AO phase maps](figures/high_order_ao_phase_maps.png)

### Noise, latency, and loop-gain trade-offs

Notebook 10 scans photon flux, read noise, loop gain, and frame delay to expose the stability and residual-error trade-offs hidden by clean AO runs.

![Noise, latency, and gain stability](figures/noise_latency_gain_stability.png)

### Simplified PWFS detector images

The PWFS branch is an exploratory Fourier-optics extension. It demonstrates detector-plane pupil images and normalized slope-like signal maps, but it is not yet a fully validated PWFS AO control simulator.

![PWFS detector images](figures/PWFS_detector_images.png)

### Sampling density versus correction order

This diagnostic illustrates the trade-off between WFS sampling density, controlled modal order, reconstruction conditioning, and correction performance.

![Sampling versus correction order](figures/sampling_vs_correction_order.png)

### Detector-level Shack-Hartmann reconstruction

The detector-level SH-WFS branch explicitly simulates lenslet spots, finite detector windows, noise, centroiding, reference subtraction, and response-matrix-based reconstruction.

![Detector-level reconstruction](figures/detector_level_reconstruction.png)

## Repository structure

```text
.
├── .github/
│   └── workflows/
│       └── ci.yml
│
├── src/
│   ├── zernike.py
│   ├── phase_screen.py
│   ├── reconstruction.py
│   ├── shwfs_detector.py
│   ├── ao_closed_loop.py
│   ├── psf_tools.py
│   └── pwfs_forward.py
│
├── notebooks/
│   ├── 01_test_sh_wfs.ipynb
│   ├── 02_sh_wfs_experiment.ipynb
│   ├── 03_zernike_mode_scan.ipynb
│   ├── 04_sampling_vs_mode_order_heatmap.ipynb
│   ├── 05_tsvd_regularization.ipynb
│   ├── 06_detector_level_shwfs.ipynb
│   ├── 07_closed_loop_ao.ipynb
│   ├── 07_detector_level_closed_loop_ao.ipynb
│   ├── 08_pwfs_detector_level_atmospheric_turbulence.ipynb
│   ├── 09_ao_psf_instrument_performance_high_order_ao.ipynb
│   └── 10_noise_latency_gain_stability.ipynb
│
├── examples/
│   ├── run_shwfs_centroid_demo.py
│   └── run_psf_strehl_demo.py
│
├── tests/
│   ├── test_psf_tools.py
│   ├── test_shwfs_detector.py
│   ├── test_phase_screen.py
│   └── test_reconstruction.py
│
├── scripts/
│   └── README.md
│
├── figures/
│   ├── high_order_ao_jhk_psf.png
│   ├── high_order_ao_rms_history.png
│   ├── high_order_ao_phase_maps.png
│   ├── noise_latency_gain_stability.png
│   ├── controller_tuning_rcond_gain.png
│   ├── PWFS_detector_images.png
│   ├── sampling_vs_correction_order.png
│   ├── detector_level_reconstruction.png
│   └── generated/
│
├── pyproject.toml
├── requirements.txt
├── README.md
└── LICENSE
```

## Code architecture

| Module              | Main role                                                                                                                                                       |
| ------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `zernike.py`        | Pupil grid generation, named low-order Zernike-like modes, general real Zernike modes, modal synthesis, piston removal, and RMS calculation.                    |
| `phase_screen.py`   | Atmospheric-like phase screens, seeing / `r0` utilities, wavelength scaling, phase-to-OPD conversion, OPD-to-phase conversion, and frozen-flow shifting.        |
| `reconstruction.py` | Geometric SH-WFS subapertures, local slope measurement, response matrices, least-squares modal reconstruction, and residual metrics.                            |
| `shwfs_detector.py` | Detector-level SH-WFS simulation: lenslet diffraction spots, detector noise, centroiding, reference centroids, centroid shifts, and detector response matrices. |
| `ao_closed_loop.py` | Gaussian DM influence functions, DM-to-WFS response matrices, command updates, geometric closed-loop AO, detector-level closed-loop AO, and loop diagnostics.   |
| `psf_tools.py`      | FFT-based PSF calculation, Strehl ratio, radial profiles, Marechal approximation, and wavelength-dependent phase diagnostics.                                   |
| `pwfs_forward.py`   | Simplified Fourier-optics PWFS model, pyramid phase mask, pupil-image extraction, modulation, detector noise, `Sx/Sy` maps, and PWFS interaction matrices.      |

## Notebook sequence

| Notebook                                               | Purpose                                                                                                                                                                                                                                                                           |
| ------------------------------------------------------ | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `01_test_sh_wfs.ipynb`                                 | Initial SH-WFS sanity tests for deterministic and atmospheric-like phase inputs. Checks slope-field sign convention, modal reconstruction, residual phase, and Strehl improvement.                                                                                                |
| `02_sh_wfs_experiment.ipynb`                           | Broader baseline SH-WFS experiment with fixed and random wavefronts, detector spot checks, centroid validity, calibration amplitude tests, photon-noise sensitivity, and PSF diagnostics.                                                                                         |
| `03_zernike_mode_scan.ipynb`                           | Studies how reconstruction quality changes with maximum Zernike order and controlled mode number. Includes residual RMS, condition number, rank, singular spectra, and Strehl diagnostics.                                                                                        |
| `04_sampling_vs_mode_order_heatmap.ipynb`              | Maps the trade-off between lenslet sampling density and correction order. Includes noise extensions to show when higher-order reconstruction becomes counterproductive.                                                                                                           |
| `05_tsvd_regularization.ipynb`                         | Demonstrates TSVD regularization in an intentionally ill-conditioned reconstruction problem. Connects singular-value cutoffs to residual wavefront error and Strehl.                                                                                                              |
| `06_detector_level_shwfs.ipynb`                        | Detector-level SH-WFS pipeline: diffraction-limited lenslet spots, finite detector windows, noisy centroiding, response-matrix calibration, coefficient reconstruction, and centroid-noise scans.                                                                                 |
| `07_closed_loop_ao.ipynb`                              | Geometric closed-loop AO baseline with Gaussian DM influence functions, ideal SH-WFS slopes, integrator control, RMS tracking, and Strehl diagnostics.                                                                                                                            |
| `07_detector_level_closed_loop_ao.ipynb`               | Detector-level closed-loop AO using centroid-shift measurements instead of ideal slopes. Includes detector response matrices, loop gain, loop delay, command leak, residual RMS, and PSF/Strehl diagnostics.                                                                      |
| `08_pwfs_detector_level_atmospheric_turbulence.ipynb`  | Simplified detector-level PWFS extension with Fourier propagation, pyramid-like phase mask, four re-imaged pupils, detector noise, `Sx/Sy` maps, modal calibration, reconstruction, and PSF diagnostics.                                                                          |
| `09_ao_psf_instrument_performance_high_order_ao.ipynb` | High-order actuator-space SH-AO / NIR PSF performance demonstrator. Uses a seeing-strength frozen-flow atmosphere, 48 × 48 SH-WFS sampling, 49 × 49 nominal DM actuator grid, poke-matrix calibration, TSVD command reconstruction, loop-gain diagnostics, and J/H/K PSF metrics. |
| `10_noise_latency_gain_stability.ipynb`                | Engineering trade-off notebook for photon flux, read noise, centroid-noise proxy, loop gain, loop delay, residual OPD RMS, H-band Strehl, and a simple stability boundary.                                                                                                      |

## Installation

Clone the repository:

```bash
git clone https://github.com/xhvoid/Shack-Hartmann-AO-Simulation.git
cd Shack-Hartmann-AO-Simulation
```

Install the required packages:

```bash
python -m pip install -r requirements.txt
```

For development, examples, and tests, install the local project with test and notebook extras:

```bash
python -m pip install -e ".[test,notebooks]"
```

The project is intentionally lightweight. The core dependencies are mainly:

```text
numpy
scipy
matplotlib
pandas
jupyter
ipykernel
```

## Running tests

The repository includes focused unit tests for the numerical core rather than executing notebooks in CI:

```bash
pytest -q
```

The GitHub Actions workflow in `.github/workflows/ci.yml` runs the same test command on Python 3.11 and then executes the two command-line examples.

## Command-line examples

The examples provide quick entry points for reviewers who want reproducible outputs without opening Jupyter:

```bash
python examples/run_shwfs_centroid_demo.py
python examples/run_psf_strehl_demo.py
```

They write lightweight artifacts under `figures/generated/`:

```text
figures/generated/shwfs_centroid_demo.png
figures/generated/shwfs_centroid_demo.csv
figures/generated/psf_strehl_demo.png
figures/generated/psf_strehl_demo.csv
```

## Running the notebooks

Start Jupyter from the repository root:

```bash
jupyter notebook
```

Recommended reading order:

```text
01 → 02 → 03 → 04 → 05 → 06 → 07 → 07b → 08 → 09 → 10
```

For a short review, start with:

```text
06_detector_level_shwfs.ipynb
07_detector_level_closed_loop_ao.ipynb
08_pwfs_detector_level_atmospheric_turbulence.ipynb
09_ao_psf_instrument_performance_high_order_ao.ipynb
10_noise_latency_gain_stability.ipynb
```

For the high-order AO PSF result, start directly with:

```text
09_ao_psf_instrument_performance_high_order_ao.ipynb
```

For noise, latency, and loop-gain trade-offs, start with:

```text
10_noise_latency_gain_stability.ipynb
```

## Reproducibility and validation

The repository now has three reproducibility layers:

* Unit tests validate the numerical core: PSF normalization, Strehl sanity checks, phase/OPD conversion, detector centroiding edge cases, phase-screen RMS scaling, and modal reconstruction.
* Command-line examples regenerate small PNG and CSV artifacts from deterministic seeds.
* Notebooks remain the narrative entry points for the broader AO experiments, with notebook 10 explicitly separating clean-model AO performance from noise, latency, and controller-stability stress tests.

## Technical highlights

### 1. Detector-level SH-WFS measurement chain

The detector-level Shack-Hartmann model follows the measurement process more explicitly than a geometric slope model:

```text
local pupil phase
→ lenslet focal-plane spot
→ finite detector window
→ photon / read / background noise
→ centroid measurement
→ reference-subtracted centroid shift
→ calibrated response matrix
→ modal reconstruction
```

This is useful because real Shack-Hartmann sensors do not measure continuous slopes directly. They measure spot images on detector pixels. Centroiding error, finite sampling, photon statistics, read noise, spot clipping, and thresholding all enter before the reconstruction step.

### 2. Response-matrix conditioning and TSVD regularization

The reconstruction problem is treated as a linear inverse problem. Each WFS model produces an interaction matrix whose singular values determine which modal or actuator combinations are strongly or weakly sensed.

The TSVD notebooks show that keeping all singular directions is not always optimal. Weak singular directions can amplify noise or unstable command components, while overly aggressive truncation removes useful controllable modes. The useful cutoff is therefore a system-level trade-off between fitting error, noise amplification, command conditioning, and image quality.

### 3. Closed-loop AO correction

The closed-loop branch uses a simplified deformable mirror with Gaussian influence functions and an integrator controller. The simulations track residual phase RMS, Strehl ratio, command evolution, and gain stability over loop iterations.

The detector-level closed-loop experiment is deliberately simplified, but it no longer uses ideal analytic slopes. The loop is driven by centroid-shift measurements from simulated lenslet spots.

### 4. High-order actuator-space AO / NIR PSF performance

The latest notebook upgrades the previous low-order modal correction into a high-order actuator-space SH-AO closed loop.

The current high-order run uses:

```text
AO_QUALITY_MODE = "extreme"
N_SUBAP         = 48
N_ACT_ACROSS   = 49
N_PUPIL        = 384
TSVD rcond      = 2e-2
loop gain       = 0.65
```

For the adopted 0.8 arcsec seeing, the SH-WFS subaperture size is close to the Fried parameter at the WFS wavelength. The controller uses an actuator poke matrix and TSVD command reconstruction, then converts residual OPD into J/H/K science PSF metrics.

Representative clean-run results:

```text
open-loop median OPD RMS after settling    ≈ 2914 nm
closed-loop median OPD RMS after settling  ≈ 135 nm
correction factor                          ≈ 22×
```

Approximate NIR Strehl ratios:

```text
J band  ≈ 0.64
H band  ≈ 0.77
K band  ≈ 0.86
```

These values should be read as a clean-model performance demonstration rather than a calibrated prediction for a specific telescope or AO system.

This makes the project closer to astronomical instrumentation performance analysis rather than only wavefront-map reconstruction.

### 5. Noise, latency, and loop-gain stability

Notebook `10_noise_latency_gain_stability.ipynb` adds a compact engineering trade-off layer around the closed-loop AO model. It scans photon flux, read noise, a transparent centroid-noise proxy, loop gain, and frame delay, then reports residual OPD RMS and H-band Marechal Strehl.

This is not a full AO error budget. Its purpose is to make the controller trade-offs visible: noise floors limit the value of high gain, latency narrows the stable gain range, and clean-model Strehl can be optimistic when detector and timing effects are ignored.

### 6. PWFS extension

The pyramid-WFS branch uses a compact Fourier-optics model. The pupil field is propagated to the focal plane, multiplied by a pyramid-like phase mask, and propagated back to form four re-imaged pupil intensities. Normalized `Sx/Sy` maps are then used as the PWFS measurement vector.

This branch should be read as an exploratory extension. It demonstrates the PWFS measurement logic and modal calibration, but it does not yet include a fully validated PWFS modulation, control, and detector-readout model.

## Validation and limitations

The current implementation intentionally keeps several assumptions simple:

* The atmosphere is represented by compact Kolmogorov / von Karman-like phase screens.
* Some atmospheric screens are RMS-normalized for controlled tests.
* The deformable mirror uses idealized Gaussian influence functions.
* The detector models are simplified.
* The SH-WFS branch is more mature than the PWFS branch.
* The PWFS branch is a compact Fourier-optics demonstrator, not a fully validated PWFS control simulator.
* The high-order notebook uses a geometric SH-WFS slope model rather than detector-level spot centroiding.
* The high-order notebook does not include photon noise, read noise, centroiding error, detector nonlinearity, temporal delay, vibration, wind-shake, or a full servo-lag error budget.
* Notebook 10 explores photon/read noise and delay through a simplified proxy model rather than a calibrated detector and real-time-control simulator.
* The high-order notebook does not include multi-layer tomography, LGS cone effect, NCPA, chromaticity, throughput, sky background, or science-camera noise.
* The simulations are not calibrated to a specific telescope, instrument, guide-star configuration, or observatory AO system.

These limitations are intentional: the project is designed to make the modelling chain inspectable and educational rather than to hide assumptions inside a black-box simulator.

## Author

Xu Han
MSc Astrophysics, LMU Munich
MSc Photonics student, Hochschule München

Research interests: adaptive optics, astronomical instrumentation, detector-level wavefront sensing, PSF modelling, instrument-performance prediction, and astrophotonics.

## License

This project is released under the MIT License.

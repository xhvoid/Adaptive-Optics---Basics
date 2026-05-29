# Shack-Hartmann Wavefront Sensing and Adaptive Optics Simulation

This project is a modular Python simulation framework for Shack-Hartmann wavefront sensing and simplified adaptive-optics correction. It includes Zernike modal wavefront generation, geometric slope sensing, detector-level lenslet spot simulation, centroid-based wavefront reconstruction, atmospheric phase-screen generation, PSF/Strehl evaluation, and simplified closed-loop AO correction with a deformable mirror.

The goal of the project is to build an interpretable end-to-end simulation chain for learning and analyzing wavefront sensing, modal reconstruction, noise propagation, and AO correction performance. The implementation is intended as an educational and research-oriented portfolio project rather than a fully calibrated observatory-grade AO simulator.

## Features

- Circular pupil generation and Zernike modal wavefront synthesis
- Geometric Shack-Hartmann slope sensing
- Response-matrix-based modal reconstruction
- Sampling-vs-mode-order analysis
- Singular-value diagnostics and TSVD regularization
- Fourier atmospheric phase-screen generation
- PSF computation and Strehl-ratio evaluation
- Detector-level lenslet spot simulation and centroiding
- Photon/read-noise sensitivity analysis
- Simplified deformable-mirror model
- Frozen-flow atmosphere and closed-loop AO correction

## Repository Structure

```text
src/        Core Python modules
notebooks/  Step-by-step simulation notebooks
figures/    Representative output figures
```text

## Installation
git clone https://github.com/xhvoid/Shack-Hartmann-AO-Simulation.git
cd Shack-Hartmann-AO-Simulation
pip install -r requirements.txt

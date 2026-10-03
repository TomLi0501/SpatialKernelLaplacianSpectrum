# Spectral Analysis of Spatial Kernel Laplacian

UROP project, Imperial College London, Summer 2026.
Supervisor: Gunnar Pruessner

## Overview

Builds a Gaussian-kernel graph Laplacian on a 2D periodic square with uniformly
distributed particles, computes the eigenvalue spectrum and correlation matrix,
and identifies the factors affecting eigenmode correlations.

## Requirements

- Python 3
- numpy
- scipy
- tqdm

## Usage

Generate data:

    python movingparticles.py

Read and analyze results:

    python readresults.py

Other scripts are for quick analysis, graph generation, or independent tests.
They do not depend on `movingparticles.py`.

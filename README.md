# ECE415 — High-Performance Computing

![C](https://img.shields.io/badge/C-systems-blue)
![OpenMP](https://img.shields.io/badge/OpenMP-parallel-orange)
![CUDA](https://img.shields.io/badge/CUDA-GPU-green)
![HPC](https://img.shields.io/badge/HPC-performance-purple)
![Coursework](https://img.shields.io/badge/UTH-ECE415-teal)

High-performance computing coursework for **ECE415** at the **University of Thessaly**. The labs start with CPU-side profiling and loop-level optimization, then move through OpenMP, CUDA kernels, CUDA streams, memory-hierarchy tuning, and multi-GPU execution.

## Quick overview

| Area | What is included |
| --- | --- |
| CPU optimization | Sobel edge-detection optimization with loop transformations, strength reduction, inlining, and serial/OpenMP comparisons. |
| OpenMP parallelism | K-Means clustering experiments using OpenMP constructs, SIMD, atomics, critical sections, and timing comparisons. |
| CUDA kernels | 2D convolution and CLAHE implementations covering block sizing, memory access patterns, streams, events, and profiling. |
| Final application | A galaxy-style N-body simulator with CPU, OpenMP, CUDA, stream, shared-memory, coarsening, multi-GPU, and divergence-reduction versions. |

## Standout work: N-body simulator

The strongest part of the repository is **Lab 5**, where the final assignment builds an N-body simulator for independent galaxy-style particle systems. The lab starts from a sequential gravitational simulation and grows into a full CPU/OpenMP/CUDA performance study with multiple GPU implementations and correctness comparisons.

<p align="center">
  <img src="docs/images/lab5-nbody-simulation.gif" alt="N-body galaxy simulation visualizer animation" width="100%">
</p>

Key pieces of the Lab 5 implementation include:

- **CPU and OpenMP baselines:** sequential and OpenMP implementations provide reference outputs and performance baselines for the GPU work.
- **CUDA optimization ladder:** the GPU versions explore naive kernels, streams, Structure-of-Arrays layout, shared-memory tiling, fast math/FMA variants, thread coarsening, launch-bound tuning, divergence-reduction variants, and multi-GPU execution.
- **Measurement workflow:** run logs and CSV summaries are used to compare implementations with throughput plots instead of only raw terminal timings.
- **Extra visualizer work:** beyond the base simulator, the repository includes a custom dataset generator and an animation of the simulation output.

<p align="center">
  <img src="docs/images/lab5-throughput-comparison.png" alt="Lab 5 N-body throughput comparison across CPU and GPU versions" width="85%">
</p>

## Course contents

<p align="center">
  <img src="docs/images/ece415-lab-map.svg" alt="ECE415 lab map from CPU optimization and OpenMP to CUDA convolution, CLAHE, and N-body simulation" width="100%">
</p>

| Directory | Topic | Implementation summary |
| --- | --- | --- |
| `Lab1/` | Sobel CPU optimization | Serial C optimization passes for Sobel edge detection, including loop interchange, unrolling, fusion, inlining, common-subexpression elimination, strength reduction, and OpenMP comparison. |
| `Lab2/` | OpenMP K-Means | Parallel K-Means clustering with OpenMP, SIMD/critical/atomic experiments, timing scripts, and report plots. |
| `Lab3/` | CUDA convolution | CUDA 2D convolution kernels covering block sizing, precision behavior, image padding, divergence effects, and runtime comparisons. |
| `Lab4/` | CUDA CLAHE | Contrast Limited Adaptive Histogram Equalization with multiple kernels, privatized histograms, scans, coalesced memory access, streams, events, and profiling plots. |
| `Lab5/` | N-body simulation | CPU, OpenMP, CUDA, stream, shared-memory, coarsening, multi-GPU, and divergence-reduction versions of a galaxy N-body simulator. |

## Requirements

- Linux environment
- `gcc` and `make`
- Python 3 for data generation and plotting scripts
- OpenMP-capable GCC
- NVIDIA CUDA toolkit (`nvcc`) for Labs 3–5
- NVIDIA GPU hardware for CUDA execution
- Optional: NVIDIA Nsight Systems / Nsight Compute for profiling

Python plotting scripts use packages such as `numpy`, `pandas`, `matplotlib`, and `seaborn`.

## Build and run

Each lab keeps its own scripts and Makefiles. Typical entry points are:

```bash
cd Lab1 && ./run.sh
cd Lab2 && ./run.sh
cd Lab3 && ./run.sh
cd Lab4 && ./run.sh
cd Lab5 && ./run.sh --help
```

Lab 5 examples:

```bash
cd Lab5
./run.sh --iterations=10 --cpu=omp --gpu=on --file=06_coarsening_2.cu --input=galaxy_data.bin
./run.sh --iterations=5 --cpu=seq --gpu=off --input=galaxy_data.bin
python3 generate_custom_dataset.py 1 4096 galaxy_1x4096.bin
```

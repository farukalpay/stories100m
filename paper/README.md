# Building the Research Paper

This directory contains the LaTeX source for the paper regarding the **Stories100M Inference Engine**.

## Prerequisites
- TeX Live / MacTeX (including `pdflatex`)

## Build Command
```bash
pdflatex main.tex
```

## Abstract
The paper "Accelerating Autoregressive Transformer Inference via Bare-Metal SIMD Optimization on ARM64" details the memory layout strategies (SoA vs AoS) and the custom NEON GEMV kernel implementation used to achieve sub-4µs latency.

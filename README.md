# Mantis

[![docs:stable](https://img.shields.io/badge/docs-stable-purple)](https://mantisfem.github.io/Mantis.jl/stable/)
[![docs:dev](https://img.shields.io/badge/docs-dev-purple)](https://mantisfem.github.io/Mantis.jl/dev/)
[![Build Status](https://github.com/MantisFEM/Mantis.jl/actions/workflows/CIlts.yml/badge.svg?branch=main)](https://github.com/MantisFEM/Mantis.jl/actions/workflows/CIlts.yml?query=branch%3Amain)
[![Build Status](https://github.com/MantisFEM/Mantis.jl/actions/workflows/CIv1withcov.yml/badge.svg?branch=main)](https://github.com/MantisFEM/Mantis.jl/actions/workflows/CIv1withcov.yml?query=branch%3Amain)
[![Build Status](https://github.com/MantisFEM/Mantis.jl/actions/workflows/CIpre.yml/badge.svg?branch=main)](https://github.com/MantisFEM/Mantis.jl/actions/workflows/CIpre.yml?query=branch%3Amain)
[![codecov](https://codecov.io/gh/MantisFEM/Mantis.jl/graph/badge.svg?token=ZWA3YV3IB6)](https://codecov.io/gh/MantisFEM/Mantis.jl)
[![Code Style: Blue](https://img.shields.io/badge/code%20style-blue-4495d1.svg)](https://github.com/JuliaDiff/BlueStyle)

## Introduction

Welcome to the documentation for `Mantis`, a Julia package for high-order
structure-preserving finite element methods.

This package is designed based on the Finite Element Exterior Calculus (FEEC) framework,
which provides a rigorous foundation for designing structure-preserving discretisations for
PDEs, e.g., those arising in electromagnetism, fluid flows, and elasticity. Such
discretisations require finite element spaces which discretize the Hilbert complexes
associated to the PDEs, such as the de Rham complex for Maxwell's equations. `Mantis`
provides users with a flexible environment where they can implement FEEC using the natural
language of Exterior Calculus, allowing them to discretize PDEs using spaces of arbitrary
regularities. Some examples of supported finite element spaces are piecewise-polynomial
spaces, non-polynomial spaces (e.g., trigonometric, exponential, Tchebycheffian B-splines),
and adaptively-refinable spaces (e.g., hierarchical B-splines).

`Mantis` is free, open-source, and available under the
[EUPL licence](https://github.com/MantisFEM/Mantis.jl/blob/main/LICENSE).

## Authors

The `Mantis` package was created by

  - Diogo C. Cabanas,
  - Joey Dekker,
  - Artur Palha,
  - Deepesh Toshniwal,

from TU Delft's Institute of Applied Mathematics (DIAM).

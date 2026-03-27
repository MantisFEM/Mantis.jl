````@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: Mantis
  text: 
  tagline: Structure-preserving discretizations in Julia
  image:
    src: logo.png
    alt: Mantis
  actions:
    - theme: brand
      text: Getting Started
      link: /GettingStarted
    - theme: alt
      text: View on Github
      link: https://github.com/MantisFEM/Mantis.jl
    - theme: alt
      text: Examples
      link: /Examples/Introduction
---
````

# Home page

Welcome to the documentation for `Mantis`, a Julia package for high-order
structure-preserving finite element methods.

This package is designed based on the Finite Element Exterior Calculus (FEEC) framework
[Arnold2006](@cite), which provides a rigorous foundation for designing structure-preserving
discretisations for PDEs, e.g., those arising in electromagnetism, fluid flows, and
elasticity. Such discretisations require finite element spaces which discretize the Hilbert
complexes associated to the PDEs, such as the de Rham complex for Maxwell's equations.
`Mantis` provides users with a flexible environment where they can implement FEEC using
the natural language of Exterior Calculus, allowing them to discretize PDEs using spaces of
arbitrary regularities. Some examples of supported finite element spaces are
piecewise-polynomial spaces, non-polynomial spaces (e.g., trigonometric, exponential,
Tchebycheffian B-splines), and adaptively-refinable spaces (e.g., hierarchical B-splines).

`Mantis` is free, open-source, and available under the
[EUPL licence](https://github.com/MantisFEM/Mantis.jl/blob/main/LICENSE).

The `Mantis` package was created by
- Diogo C. Cabanas,
- Joey Dekker,
- Artur Palha,
- Deepesh Toshniwal,
from TU Delft's Institute of Applied Mathematics (DIAM).

!!! warning "Under development"
    `Mantis` is under active development and can still undergo large changes.

```@bibliography
Pages = []
Canonical = false

Arnold2006
```

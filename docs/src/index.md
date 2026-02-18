````@raw html
---
# https://vitepress.dev/reference/default-theme-home-page
layout: home

hero:
  name: Mantis.jl
  text: 
  tagline: Structure-preserving discretizations in Julia
  image:
    src: logo.png
    alt: Mantis.jl
  actions:
    - theme: brand
      text: Getting Mantis
      link: /Manual/InstallGuide
    - theme: alt
      text: View on Github
      link: https://github.com/MantisFEM/Mantis.jl
    - theme: alt
      text: Examples
      link: /Examples/Introduction
---
````

# Mantis.jl Documentation

## Introduction
Welcome to the documentation for `Mantis.jl`, a Julia package for high-order structure-preserving finite element methods.

This package is designed based on the Finite Element Exterior Calculus (FEEC) framework [Arnold2006](@cite) which provides a rigorous foundation for designing structure-preserving discretizations for PDEs, e.g., those arising in electromagnetism, fluid flows, and elasticity. Such discretizations require finite element spaces which discretize the Hilbert complexes associated to the PDEs, such as the de Rham complex for Maxwell's equations. `Mantis.jl` provides users with a flexible environment where they can implement FEEC using the natural language of Exterior Calculus, allowing them to discretize PDEs using spaces of arbitrary regularities. Some examples of supported finite element spaces are piecewise-polynomial spaces, non-polynomial spaces (e.g., trigonometric, exponential, Tchebycheffian B-splines), and adaptively-refinable spaces (e.g., hierarchical B-splines).

`Mantis.jl` is free and open-source
([MIT license](https://github.com/MantisFEM/Mantis.jl/blob/main/LICENSE)).

!!! warning "Under development"
    Mantis.jl is under active development and can still undergo large changes.

```@bibliography
Pages = []
Canonical = false

Arnold2006
```

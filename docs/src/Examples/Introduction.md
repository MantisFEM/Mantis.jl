# Examples Overview

These examples show how `Mantis` is used in practice. They are organised into three tiers, so
you can either follow them in order as a tutorial or jump straight to what you need.

## First steps

Small, single-concept examples. They build up the core objects one at a time, from the
geometry up to integration, and are the place to start if you are new to `Mantis`.

- [Constructing Cartesian geometries](@ref) builds lines, rectangles, boxes, and graded meshes.
- [One-dimensional mapped geometry](@ref) and [Two-dimensional mapped geometry](@ref) curve a
  domain with a mapping.
- [Tensor-product geometry](@ref) builds a box and a cylinder from simpler geometries.
- [Inspecting and visualizing a geometry](@ref) covers evaluating, differentiating, and
  plotting a geometry.
- [B-spline spaces and basis functions](@ref) introduces function spaces, degree, and
  regularity.
- [Form spaces and form fields](@ref) turns a function space into differential forms.
- [Differential-form operators](@ref) covers the exterior derivative, wedge, and Hodge star.
- [Quadrature and integrating a form](@ref) covers numerical integration and choosing a rule.

## Solving PDEs

Complete, end-to-end solves, ordered by increasing complexity.

- [L2 projection](@ref) is the smallest complete pipeline.
- [Hodge Laplacian](@ref) solves the Poisson problem in form language.
- [Heat Equation](@ref) solves a time-dependent problem.
- [Biharmonic](@ref) solves a higher-order problem and includes convergence studies.
- [Maxwell eigenvalue problem](@ref) solves a mixed, structure-preserving eigenproblem.

## Advanced features

Features specific to `Mantis`.

- [Non-polynomial function spaces](@ref) uses trigonometric and exponential B-splines.
- [Adaptive refinement](@ref) uses hierarchical B-splines driven by an error estimator.

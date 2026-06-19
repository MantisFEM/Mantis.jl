```@meta
CurrentModule = Mantis.Analysis
```
# Analysis

The `Analysis` module provides the post-processing tools used to *measure* a computed
solution: error norms against a reference solution, and the per-element error indicators that
drive adaptive refinement. It works entirely in the language of [Forms](@ref), so it applies
uniformly to forms of any rank, on any geometry.

## Measuring errors

The most common task is a convergence study: solve a problem on a sequence of meshes and check
that the error decreases at the expected rate. Because errors are just norms of the difference
of two forms, you compute them by subtracting an exact (analytical) solution from the computed
one and taking a norm:

```julia
using Mantis

# uₕ is the computed solution (a FormField)
# uₑ is the exact solution on the same geometry (an AnalyticalFormField)
# dΩ_error is an over-integrating quadrature rule (a StandardQuadrature)

err = Analysis.L2_norm(uₕ - uₑ, dΩ_error)
```

Two points are worth emphasising:

- **Use a finer quadrature rule for the error than for assembly.** Measuring the error with
  the same rule used to assemble the system can mask the true discretisation error behind
  quadrature error. The [Quadrature](@ref) module's
  [`get_canonical_quadrature_rules`](@ref Mantis.Quadrature.get_canonical_quadrature_rules)
  helper builds matched assembly/error rules.
- **The exact solution must live on the same geometry** as the computed one;
  [`compute_error_total`](@ref) and [`compute_error_per_element`](@ref) check this and warn (or
  error) otherwise.

## Available quantities

| Function | Returns | Typical use |
|---|---|---|
| [`L2_norm`](@ref) | scalar | quick ``L^2`` error/norm |
| [`compute_error_total`](@ref) | scalar | domain-wide ``L^2`` or ``L^\infty`` error |
| [`compute_error_per_element`](@ref) | vector | local error indicator for adaptivity |

The per-element version is what links analysis back to adaptivity: its output is exactly the
local indicator handed to Dörfler marking in the [Adaptive refinement](@ref) example. The
worked [Biharmonic](@ref) and [Maxwell eigenvalue problem](@ref) examples show `L2_norm` used
in full convergence studies.

## All docstrings from Mantis.Analysis
```@autodocs
Modules = [Main.Mantis.Analysis]
```

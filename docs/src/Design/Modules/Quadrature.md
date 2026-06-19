```@meta
CurrentModule = Mantis.Quadrature
```
# Quadrature

Every integral that `Mantis` computes (mass and stiffness matrix entries, load vectors, error
norms) is evaluated numerically. The `Quadrature` module provides the quadrature rules that
supply the evaluation points and weights for those integrals.

## Element rules vs. global rules

There are two layers, mirroring the canonical-vs-global distinction used for function spaces
(see the [FunctionSpaces](@ref) module):

- **Element (canonical) rules** ([`AbstractElementQuadratureRule`](@ref)) live on the canonical
  element ``[0, 1]^n``. A multivariate canonical rule is built as a
  [`tensor_product_rule`](@ref) of univariate rules. The available univariate rules are
  [`gauss_legendre`](@ref), [`gauss_lobatto`](@ref), [`clenshaw_curtis`](@ref) and
  [`newton_cotes`](@ref).
- **Global rules** ([`AbstractGlobalQuadratureRule`](@ref)) place an element rule on every
  element of the mesh. The common case is [`StandardQuadrature`](@ref), which applies the
  *same* canonical rule to each element.

### Terminology: evaluation elements

A global rule does not store the geometry; it only needs to know *how many* elements it should
place its canonical rule on. These are the **evaluation elements**, and their count is exactly
the `num_elements` you pass to [`StandardQuadrature`](@ref). The same term appears in the
[Forms](@ref) module: [`Integral`](@ref Mantis.Forms.Integral) reports its number of evaluation
elements through `get_num_evaluation_elements`. It always refers to the elements over which the
quadrature rule is laid out. In the common case this equals the number of mesh elements of the
geometry, which is why you usually write:

```julia
dΩ = Quadrature.StandardQuadrature(canonical_qrule, Geometry.get_num_elements(geometry))
```

## Choosing a rule

A canonical rule is created from the number of points per direction and a univariate rule
generator. As a rule of thumb, a Gauss-Legendre rule with `p + 1` points per direction
integrates the entries of a B-spline mass/stiffness matrix of degree `p` to round-off, so this
is the standard assembly choice:

```julia
using Mantis

p = (3, 3)
canonical_qrule = Quadrature.tensor_product_rule(p .+ 1, Quadrature.gauss_legendre)
dΩ = Quadrature.StandardQuadrature(canonical_qrule, prod((4, 4)))
```

For error computation (see [Analysis](@ref)) it is good practice to *over-integrate* with more
points than were used for assembly, so that the quadrature error does not pollute the measured
discretisation error. The helper [`get_canonical_quadrature_rules`](@ref) builds an assembly
rule and a (finer) error rule in one call.

## All docstrings from Mantis.Quadrature
```@autodocs
Modules = [Main.Mantis.Quadrature]
```

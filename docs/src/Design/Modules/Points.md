```@meta
CurrentModule = Mantis.Points
```
# Points

The `Points` module defines the *evaluation points*: the locations, in canonical
(reference-element) coordinates, at which geometries, function spaces, forms and quadrature
rules are sampled. Almost every `evaluate` call in `Mantis` takes an `element_idx` together
with an [`AbstractPoints`](@ref) object: the index selects which element's mapping to use, and
the points say where on the canonical element ``[0, 1]^n`` to evaluate. (See the
[One-dimensional mapped geometry](@ref) example for this convention in action.)

There are two concrete point types, differing only in how the points are laid out:

- [`PointSet`](@ref) is an explicit list of points. Use it to evaluate at a specific, possibly
  scattered, set of locations.
- [`CartesianPoints`](@ref) is a tensor-product grid of points, given as a tuple of
  per-direction coordinate vectors. Use it for structured sampling, for example to plot a
  solution on a regular grid over each element.

```julia
using Mantis

# Five points along the canonical 1D element:
xi = Points.CartesianPoints((LinRange(0.0, 1.0, 5),))

# The same five points given explicitly:
ps = Points.PointSet(([0.0, 0.25, 0.5, 0.75, 1.0],))
```

Both are subtypes of [`AbstractPoints`](@ref){`manifold_dim`, `T`}, so the manifold dimension
of the points is part of their type and is checked against the object being evaluated.

## All docstrings from Mantis.Points
```@autodocs
Modules = [Main.Mantis.Points]
```

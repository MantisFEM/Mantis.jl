```@meta
CurrentModule = Mantis.Points
```
# [Points](@id DocPointsModule)

```@docs
Points
```

## Overview

In Mantis, most object evaluations take place in the _canonical domain_ ``[0, 1]^n``, where
``n`` is the manifold dimension. Physical coordinates are then defined from the canonical
ones, according to the geometry being used, as described in [Geometry](@ref
DocGeometryModule). The `Points` module establishes an interface, common to every module, of
how points are interpreted, regardless of whether they are canonical, parametric, or
physical, ensuring consistency throughout the codebase.

## [The `AbstractPoints` Interface](@id PointsAbstract)

All point sets in Mantis are subtypes of `AbstractPoints`:

```@docs
AbstractPoints
```

The module requires every concrete subtype to implement `get_num_points`:

```@docs
get_num_points
```

From this, other methods can be automatically derived, without needing manual
implementation:

```@docs
get_manifold_dim
get_input_points
scale_and_shift_points
```

Every concrete `points` object is also iterable and indexable. Indexing `points` at a linear
index `i` returns an `NTuple{manifold_dim, T}` representing the coordinates of the `i`-th
point. This uniform interface means that other modules can simply index `points[i]`, or loop
over all points with `for p in points`, regardless of whether the underlying layout is
unstructured or tensor-product.

## [Concrete Subtypes](@id PointsConcrete)

The module provides two concrete subtypes of `AbstractPoints`, each suited to a different
structure of evaluation points.

### [PointSet](@id PointsPointSet)

```@docs
PointSet
```

The dimension-wise layout, as opposed to a list of points, allows geometry and
function-space evaluation routines to extract all coordinates along a given dimension with a
single array access, avoiding unnecessary allocations when iterating over large point sets.
As tensor-product structured geometries and function-spaces are very common, this ensures
such implementations can remain efficient, even with unstructured points.

### [TensorProductPoints](@id PointsTensorProduct)

```@docs
TensorProductPoints
```

Tensor-product points are the natural choice whenever the evaluation sites themselves have a
tensor-product structure — for instance, the nodes of a [Quadrature](@ref) rule on a
multidimensional element.

This concrete type has methods that help interface with [TensorProducts](@ref DocTensorProductsModule):

```@docs
get_factor_points
get_cart_num_points
get_lin_num_points
get_factor_num_points
get_iteration_order
get_permuted_cart_num_points
```

## [Relationships to Other Modules](@id PointsRelationships)

The `Points` module deliberately has minimal dependencies, only [TensorProducts](@ref
DocTensorProductsModule), and is used by every module that requires evaluation:

- **[Geometry](@ref DocGeometryModule)**: evaluates geometries by mapping points in the
  canonical domain to a set of points on a given element, in either parametric or physical
  coordinates.
- [FunctionSpaces](@ref): evaluates basis functions and their derivatives at `AbstractPoints` in
  the canonical domain of each element.
- **[Forms](@ref)**: in the most general case, evaluates from a canonical set of points,
  using both parametric information from the underlying function spaces, and information
  from the physical geometry for pullbacks.
- [Quadrature](@ref): quadrature rules store their nodes as `AbstractPoints`. The
  `TensorProductPoints` type is particularly useful here, as most multidimensional rules are
  formed from 1D factor rules.
- [Plot](@ref): uses points to build visualisations written to VTK output.

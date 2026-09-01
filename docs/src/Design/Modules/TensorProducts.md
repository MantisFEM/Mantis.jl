```@meta
CurrentModule = Mantis
```

# [TensorProducts](@id DocTensorProductsModule)

```@docs
TensorProducts
```

The module is intentionally minimal. Its only task is to take any tuple of factor sets,
pre-compute the bijection between a global linear index and the tuple of per-set indices,
and provide utilities to traverse and map over the factors. Modules such as [Geometry](@ref)
and [FunctionSpaces](@ref) implement a small interface and delegate all index bookkeeping to
[`TensorProducts`](@ref), keeping the tensor-product logic in one place rather than scattered
across the codebase.

```@meta
CurrentModule = Mantis.TensorProducts
```

## [The [`TensorProduct`](@ref) Type](@id TensorProductsType)

The central type in this module is [`TensorProduct`](@ref).

```@docs
TensorProduct
```

For convenience, the module exports the symbolic operator `⊗`:

```@docs
⊗
```

Two or more sets can also be freely composed. These compositions are automatically
flattened: `(A ⊗ B) ⊗ C` and `A ⊗ (B ⊗ C)` are both equivalent to `A ⊗ B ⊗ C`, producing a
single [`TensorProduct`](@ref) with three factors rather than a nested pair.

## [The `get_num_objects` requirement](@id TensorProductsRequirement)

For a set to be used in a [`TensorProduct`](@ref), it must implement one function:

```@docs
get_num_objects
```

This is the only requirement imposed by the module. Standard Julia containers (`Tuple` and
`AbstractArray`) satisfy it out of the box using `length`. Custom types — such as
[`Mantis.Geometry.AbstractGeometry`](@ref) or [`Mantis.FunctionSpaces.AbstractFESpace`](@ref) —
implement this function to expose their element or basis count, respectively. This keeps the
`TensorProducts` module free of any dependency on the rest of Mantis.

## [Index Bookkeeping](@id TensorProductsIndexing)

The central service provided by a [`TensorProduct`](@ref) is the efficient bidirectional mapping
between a single linear index and a tuple of per-factor indices. This mapping is
pre-computed once at construction time and stored internally. `Mantis` has a few helpers for
this purpose:

```@docs
get_cart_ids
get_lin_ids
get_factor_ids
get_factor_num_objects
get_num_factors
get_num_objects(tp::TensorProduct)
```

An example should make it clear.

```@repl TensorProductsIndexExample
using Mantis
tp = TensorProducts.TensorProduct(("a", "b"), ("x", "y", "z"))  # 2 × 3 product
TensorProducts.get_num_objects(tp)
TensorProducts.get_factor_ids(tp, 4)  # (2, 2): second of "a","b", second of "x","y","z"
TensorProducts.get_factor_num_objects(tp)
```

## [Accessing and Mapping factors](@id TensorProductsMapping)

The following functions provide access to the factor sets and allow applying operations to
them in a dimension-independent way.

```@docs
get_factors
mapfactors(f, tp::TensorProduct)
mapfactors(f, tp::TensorProduct, id)
Base.map(f, tp::TensorProduct, args...)
```

## [Usage in Other Modules](@id TensorProductsUsage)

`TensorProducts` is used by several modules in Mantis. The two most prominent use-cases are
[`Mantis.Geometry.TensorProductGeometry`](@ref) and
[`Mantis.FunctionSpaces.TensorProductSpace`](@ref).

### [Tensor-Product Geometry](@id TensorProductsGeometry)

In [Geometry](@ref), a [`Mantis.Geometry.TensorProductGeometry`](@ref) is formed by assembling
lower-dimensional geometries along independent coordinate directions. The interface with
[`TensorProducts`](@ref) is valid since [`Mantis.Geometry.AbstractGeometry`](@ref) implements
[`get_num_objects`](@ref) as an alias of [`Mantis.Geometry.get_num_elements`](@ref). As a
result, a global element ID in the product geometry can be decomposed into per-factor element
IDs, and geometry evaluation, Jacobian computation, and Hessian computation all reduce to
per-factor operations with no special-casing.

```@repl TensorProductsGeometryExample
using Mantis

line_x = Geometry.create_cartesian_box((0.0,), (2.0,), (4,))
line_y = Geometry.create_cartesian_box((0.0,), (1.0,), (3,))
geo = Geometry.TensorProductGeometry(line_x, line_y)

Geometry.get_num_elements(geo)
Geometry.get_factor_num_elements(geo)
```

### [Tensor-Product Function Spaces](@id TensorProductsSpaces)

In [FunctionSpaces](@ref), a [`Mantis.FunctionSpaces.TensorProductSpace`](@ref) assembles a
multi-dimensional finite element space from lower-dimensional ones. Here the
`TensorProducts` interface uses the number of basis functions, aliasing
[`get_num_objects`](@ref) with [`Mantis.FunctionSpaces.get_num_basis`](@ref). As such, a
global basis ID decomposes into per-factor basis IDs. 

The local basis functions on a product element are assembled via Kronecker products of the
per-factor local bases, and supports and extraction operators follow the same structure.

```@repl TensorProductsSpacesExample
using Mantis

Bx = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (4,), (3,), (2,))
By = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (2,), (3,), (2,))
Bz = FunctionSpaces.create_bspline_space((0.0,), (1.0,), (3,), (3,), (2,))

B3D = FunctionSpaces.TensorProductSpace(Bx, By, Bz)
FunctionSpaces.get_num_basis(B3D)
FunctionSpaces.get_factor_num_basis(B3D)
```

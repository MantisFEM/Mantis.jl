```@meta
CurrentModule = Mantis.FunctionSpaces
```
# FunctionSpaces

The `FunctionSpaces` module provides the finite-dimensional spaces that `Mantis` uses as the
*bases* for its [Form Spaces](@ref FormsSpaces). Where the [Forms](@ref) module decides how
an object *transforms* (i.e. its `form_rank`), the `FunctionSpaces` module decides which
scalar- or vector-valued functions are available to approximate with. It is the largest
module in `Mantis`; the high-regularity, non-polynomial, and adaptively-refinable spaces are
all defined here.

## Two kinds of spaces

Every space is a subtype of [`AbstractFunctionSpace`](@ref), and there are two fundamentally
different kinds:

- **Canonical spaces** ([`AbstractCanonicalSpace`](@ref)) are *element-local* bases. They are
  defined only on the canonical (reference) element ``[0, 1]`` and describe the shape
  functions on a single element. They carry no notion of a mesh or of inter-element
  continuity.
- **Finite element spaces** ([`AbstractFESpace`](@ref)) are *global* spaces. They are defined
  over an entire geometry by gluing together element-local canonical bases with a prescribed
  smoothness across element interfaces. These are the spaces you hand to a `FormSpace`.

```
AbstractFunctionSpace
├── AbstractCanonicalSpace           # element-local shape functions on [0,1]
│   ├── Bernstein
│   ├── AbstractLagrangePolynomials  → Lagrange
│   ├── AbstractEdgePolynomials      → Edge
│   └── AbstractECTSpaces            → GeneralizedTrigonometric,
│                                      GeneralizedExponential, Tchebycheff
└── AbstractFESpace{manifold_dim, num_components, num_patches}
    ├── BSplineSpace                 # univariate B-splines
    ├── TensorProductSpace          # tensor product of univariate FE spaces
    ├── DirectSumSpace              # vector-valued / multi-component spaces
    ├── HierarchicalFiniteElementSpace  # adaptively refinable
    ├── RationalFESpace             # rational (NURBS-like) spaces
    ├── PolarSplineSpace            # splines on polar/singular parametrisations
    └── GTBSplineSpace              # generalised (unstructured) B-splines
```

### Canonical spaces: the section spaces

A canonical space is sometimes called a *section space*, because it is the building block
repeated along each parametric direction. The default choice is [`Bernstein`](@ref) (which
yields Bézier/B-spline bases), but `Mantis` also supports non-polynomial bases through the
*extended complete Tchebycheffian* (`AbstractECTSpaces`) family:

- [`GeneralizedTrigonometric`](@ref) for trigonometric generalisations of B-splines,
- [`GeneralizedExponential`](@ref) for exponential generalisations of B-splines,
- [`Tchebycheff`](@ref) for general Tchebycheffian spaces,

as well as nodal [`Lagrange`](@ref) and [`Edge`](@ref) bases. To use non-polynomial finite
elements you change the section space; see the [Non-polynomial function spaces](@ref) example.

### Finite element spaces and extraction

A finite element space turns element-local canonical bases into a globally-defined basis with
the requested continuity. The bookkeeping that maps each element's canonical shape functions
onto the global degrees of freedom is the [`ExtractionOperator`](@ref) (Bézier-style
extraction). You rarely construct one by hand; the space constructors build it for you. Every
`evaluate` call on an `AbstractFESpace` goes through it.

The univariate building block is the [`BSplineSpace`](@ref), which is created from a 1D
geometry together with a polynomial degree `p` and an inter-element regularity `k` (with
``-1 \le k < p``). Multivariate spaces are then assembled as tensor products
([`TensorProductSpace`](@ref)) and stacked into vector-valued spaces with
[`DirectSumSpace`](@ref).

## Creating a function space

For the common case of a B-spline space on a Cartesian box there is a one-shot helper,
[`create_bspline_space`](@ref):

```julia
using Mantis

# A 2D maximally-smooth quadratic B-spline space on the unit square with 4×4 elements:
starting_point = (0.0, 0.0)
box_size       = (1.0, 1.0)
num_elements   = (4, 4)
p = (2, 2)   # polynomial degree per direction
k = (1, 1)   # regularity per direction (here C¹)

B = FunctionSpaces.create_bspline_space(starting_point, box_size, num_elements, p, k)
```

The univariate constructor is useful when you want full control, for example to build a
curvilinear space on a mapped geometry:

```julia
line_geo = Geometry.CartesianGeometry((LinRange(0.0, 1.0, 5),))
B1 = FunctionSpaces.BSplineSpace(line_geo, 2, 1)            # degree 2, C¹
```

Once a space exists you can query it with the usual getters, e.g.
[`get_num_basis`](@ref Mantis.Forms.get_num_basis) (the dimension of the space) and the
per-direction constituent spaces via `get_constituent_spaces`.

## Building a space from an extraction operator

This section looks under the hood at how a finite element space is defined, and shows how to
build a custom one. It is an advanced topic; for everyday use the constructors above are
enough.

Every [`AbstractFESpace`](@ref) is built from two parts: an element-local section space (here
[`Bernstein`](@ref) polynomials) and an [`ExtractionOperator`](@ref). On each element the
extraction operator stores a matrix `E` of *extraction coefficients*. A global basis function,
restricted to that element, is a linear combination of the local section-space functions, and
the columns of `E` hold the coefficients of those combinations. Evaluation is exactly the
matrix product `local_values * E`, mapped to the global basis indices the operator also stores.

### The standard example: a C¹ quadratic B-spline

Consider a degree-2, ``C^1`` B-spline space on two elements of ``[0, 1]``.

```@example extraction
using Mantis

geometry = Geometry.create_cartesian_box((0.0,), (1.0,), (2,))
B = FunctionSpaces.BSplineSpace(geometry, 2, 1)   # degree 2, regularity 1

(FunctionSpaces.get_num_basis(B), FunctionSpaces.get_num_elements(B))
```

The space has four global basis functions. Each element carries three quadratic Bernstein
functions, and the extraction operator says how the global functions are assembled from them.
Here is the extraction matrix on element 1, together with the global indices of the basis
functions it supports:

```@example extraction
(FunctionSpaces.get_extraction_coefficients(B, 1), collect(FunctionSpaces.get_basis_indices(B, 1)))
```

The columns are the supported global functions (indices ``1, 2, 3``) and the rows are the three
Bernstein functions. Reading the columns: global function 1 is the first Bernstein function,
global function 2 is the second plus half the third, and global function 3 is half the third.
The blending coefficient ``0.5`` is what makes the basis ``C^1`` across the element boundary.
Element 2 is the mirror image:

```@example extraction
(FunctionSpaces.get_extraction_coefficients(B, 2), collect(FunctionSpaces.get_basis_indices(B, 2)))
```

We can check the rule `local_values * E` directly. Evaluating the Bernstein functions on the
reference element and multiplying by `E` reproduces the spline values that `evaluate` returns:

```@example extraction
bernstein = FunctionSpaces.Bernstein(2)
ξ = Points.CartesianPoints(([0.0, 0.25, 0.5, 0.75, 1.0],))

bernstein_values = FunctionSpaces.evaluate(bernstein, ξ, 0)[1][1]      # 5 points × 3
spline_values, _ = FunctionSpaces.evaluate(B, 1, ξ, 0)

maximum(abs.(spline_values[1][1][1] .- bernstein_values * FunctionSpaces.get_extraction_coefficients(B, 1)))
```

The three local building blocks are the quadratic Bernstein functions on the reference element:

```@example extraction
using GLMakie
using DisplayAs # hide

ξ_plot = collect(LinRange(0.0, 1.0, 40))
bernstein_curves = FunctionSpaces.evaluate(bernstein, Points.CartesianPoints((ξ_plot,)), 0)[1][1]

fig = Figure(size = (480, 300))
ax = Axis(fig[1, 1]; title = "Quadratic Bernstein basis (reference element)",
          xlabel = "ξ", ylabel = "value")
for i in 1:3
    lines!(ax, ξ_plot, bernstein_curves[:, i]; label = "B$(i)")
end
axislegend(ax)
fig = DisplayAs.Text(DisplayAs.PNG(fig)) # hide
```

Combining these per-element building blocks with the extraction coefficients gives the four
global B-spline basis functions. The helper below plots each global function over the whole
mesh, with the element boundary at ``x = 0.5`` marked:

```@example extraction
function plot_global_basis(space; title)
    fig = Figure(size = (560, 320))
    ax = Axis(fig[1, 1]; title = title, xlabel = "x", ylabel = "value")
    ξ_pts = Points.CartesianPoints((collect(LinRange(0.0, 1.0, 40)),))
    palette = [:steelblue, :darkorange, :seagreen, :crimson, :purple]
    for e in 1:Geometry.get_num_elements(geometry)
        values, indices = FunctionSpaces.evaluate(space, e, ξ_pts, 0)
        x = Geometry.evaluate(geometry, e, ξ_pts)
        for (local_id, global_id) in enumerate(indices)
            lines!(ax, vec(x), values[1][1][1][:, local_id];
                   color = palette[global_id], label = "φ$(global_id)")
        end
    end
    vlines!(ax, [0.5]; color = :gray, linestyle = :dash)
    axislegend(ax; merge = true, unique = true)
    return fig
end

fig = plot_global_basis(B; title = "C¹ quadratic B-spline basis")
fig = DisplayAs.Text(DisplayAs.PNG(fig)) # hide
```

Function 1 is the single Bernstein function on element 1, while functions 2 and 3 span both
elements: each is built from the blended Bernstein functions whose ``0.5`` coefficients make the
join smooth.

### A custom space from a different extraction operator

To define a new space we supply our own extraction operator. A single-component
[`AbstractFESpace`](@ref) only needs an `extraction_op` field and a `get_local_basis` method;
the basis indices, extraction coefficients, and dimension are all read from the operator by
default.

```@example extraction
struct CustomSpace{S, E, G} <: FunctionSpaces.AbstractFESpace{1, 1, 1}
    section_space::S
    extraction_op::E
    geometry::G
end

function FunctionSpaces.get_local_basis(
    space::CustomSpace, element_id::Int, ξ::Points.AbstractPoints{1},
    nderivatives::Int, component_id::Int=1,
)
    section_eval = FunctionSpaces.evaluate(space.section_space, ξ, nderivatives)
    local_basis = Vector{Vector{Vector{Matrix{eltype(ξ)}}}}(undef, nderivatives + 1)
    for i in eachindex(section_eval, local_basis)
        local_basis[i] = [[section_eval[i][1]]]
    end
    return local_basis
end
nothing # hide
```

Now we choose the extraction coefficients. Taking the identity on each element, and letting the
two elements share only the degree of freedom at the joint, gives a ``C^0`` quadratic space
with five basis functions instead of four:

```@example extraction
identity3 = [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
extraction_coefficients = [(identity3,), (identity3,)]
basis_indices = [
    FunctionSpaces.Indices([1, 2, 3], ([1, 2, 3],)),   # element 1 → global 1,2,3
    FunctionSpaces.Indices([3, 4, 5], ([1, 2, 3],)),   # element 2 → global 3,4,5
]
custom_extraction = FunctionSpaces.ExtractionOperator(extraction_coefficients, basis_indices, 2, 5)
custom = CustomSpace(FunctionSpaces.Bernstein(2), custom_extraction, geometry)

FunctionSpaces.get_num_basis(custom)
```

The two choices produce different smoothness. The function shared across the joint has a
continuous derivative in the B-spline space (the ``0.5`` blend was chosen for exactly that),
but a jump in the custom ``C^0`` space. Comparing the derivative of the shared function from the
left and the right of the joint shows this:

```@example extraction
function derivative_jump(space, shared_basis)
    left,  il = FunctionSpaces.evaluate(space, 1, Points.CartesianPoints(([1.0],)), 1)
    right, ir = FunctionSpaces.evaluate(space, 2, Points.CartesianPoints(([0.0],)), 1)
    a = findfirst(==(shared_basis), collect(il))
    b = findfirst(==(shared_basis), collect(ir))
    return left[2][1][1][1, a] - right[2][1][1][1, b]
end

(derivative_jump(B, 3), derivative_jump(custom, 3))   # (C¹ B-spline, custom C⁰)
```

The B-spline value is zero (the derivative matches), while the custom space leaves a jump. The
extraction coefficients, not the section space, are what set the inter-element continuity.

The same plot for the custom space shows five functions, and the shared function at ``x = 0.5``
now has a corner instead of the smooth peak of the B-spline:

```@example extraction
fig = plot_global_basis(custom; title = "Custom C⁰ quadratic basis")
fig = DisplayAs.Text(DisplayAs.PNG(fig)) # hide
```

## De Rham complexes

Most structure-preserving problems do not use a single space but a *compatible sequence* of
spaces forming a discrete de Rham complex. Because these spaces are consumed as
[Form Spaces](@ref FormsSpaces), the constructors that build a whole complex at once live in
the [Forms](@ref) module; see [De Rham Complexes](@ref FormsComplexes). They assemble the
per-rank `FunctionSpaces` objects described above (for instance the hierarchical
complex stacks [`HierarchicalFiniteElementSpace`](@ref)s built from tensor-product B-splines).

## Adaptive refinement support

[`HierarchicalFiniteElementSpace`](@ref) is the adaptively refinable finite element space. It
maintains a hierarchy of nested B-spline levels and an active set of basis functions, and it
relies on two pieces of machinery in this module:

- **Two-scale relations**, which express coarse-level basis functions in terms of finer-level
  ones (the `TwoScaleRelations` submodule), and
- **Marking and refinement** (the `AdaptiveRefinement` submodule), including Dörfler-style
  marking and the L-chain admissibility fix used to keep refinements graded.

Refinement is driven through [`refine_space`](@ref) (and, at the complex level, through
[`update_hierarchical_de_rham_complex`](@ref Mantis.Forms.update_hierarchical_de_rham_complex)).
The active elements and basis of a hierarchical space are tracked using
[`ActiveInfo`](@ref Mantis.Hierarchy.ActiveInfo); see the [Hierarchy](@ref) module for that
bookkeeping, and the [Adaptive refinement](@ref) example for an end-to-end adaptive loop.

## All docstrings from Mantis.FunctionSpaces
```@autodocs
Modules = [Main.Mantis.FunctionSpaces]
```

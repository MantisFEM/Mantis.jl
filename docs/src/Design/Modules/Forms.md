```@meta
CurrentModule = Mantis.Forms
```

# Forms

As explained on the [differential form theory page](@ref TheoryForms), differential forms provide an elegant and general framework to deal with the discretisation of PDEs.
One of the most distinguishing features of `Mantis` is its ability to work with these differential forms.
The `Forms` module provides all the required tools to use differential forms in `Mantis`.

## What is a differential form in `Mantis`?

The top-level type within the `Forms` module is the `AbstractForm{manifold_dim, form_rank, expression_rank}` type. Every expression involving forms (see [Creating Forms](@ref FormsCreation)) and operations on forms (see [Operations on Forms](@ref FormsOperations)) will be an `AbstractForm`.
There are two exceptions to this rule.
The first exception is an operation that returns a real value, such as an integral, see [Operators returning a real value](@ref FormsRealValuedOperators).
The second exception is an operation that returns a vector, such as the sharp, see [Operators returning a vector](@ref FormsOperationsToVectors)

```@docs
AbstractForm
```

There are two aliases for `AbstractForm`, which are `AbstractFormField` and `AbstractFormSpace`.

```@docs
AbstractFormField
AbstractFormSpace
```

Every `AbstractForm` has three type parameters which say something about the form.
You can always call the following three methods on any `AbstractForm` to get these type parameters.

```@docs
get_manifold_dim
get_form_rank
get_expression_rank
```

The above abstract types are used in function signatures, but cannot be instantiated.
The concrete types that can be instantiated are discussed next.

## [Creating Forms](@id FormsCreation)

You can create two main types of `Forms`: [FormSpaces](@ref FormsSpaces) and [FormFields](@ref FormsFields).

### [FormSpaces](@id FormsSpaces)

A `FormSpace` allows you to distinguish between functions and forms.
A `FormSpace` is build on top of a [FunctionSpaces.AbstractFESpace](@ref), which acts as its basis.
However, it is the `FormSpace` that dictates the behaviour of the form.

```@docs
FormSpace
```

As explained on the [differential form theory page](@ref TheoryForms), differential forms are more expressive than functions.
By using a `FormSpace`, this expressiveness becomes available within your code.
For example, if we start by creating a simple 2D [FunctionSpaces.BSplineSpace](@ref) using the helper [FunctionSpaces.create\_bspline\_space](@ref) (on a unit square with ``4 \times 4`` elements, degree ``3`` and regularity ``2``),

```@repl CreatingFormSpaces
using Mantis
B = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (4, 4), (3, 3), (2, 2))
```

we can use this function space to create two different spaces: one for a ``0``-form ``\Lambda^0_h`` and one for a ``2``-form ``\Lambda^2_h`` (a top form in 2D).

```@repl CreatingFormSpaces
Λ⁰ₕ = Forms.FormSpace(0, B, "0-form")
Λ²ₕ = Forms.FormSpace(2, B, "2-form")
```

These two forms have the same basis `B`, but have different transformation properties.
This will result in the use of different pullbacks (see [How `FormSpaces` are evaluated](@ref FormsInternalEvaluateFormSpace) on how that is reflected in the implementation), and on the operations that you can apply to these forms (see [Operations on Forms](@ref FormsOperations)).

### [ConstantFormSpaces](@id FormsConstantSpaces)

Next to the conventional `FormSpace`, `Mantis` also provides a `ConstantFormSpace`.
A `ConstantFormSpace` can be instantiated as a ``0``- or `manifold_dim`-form (so a top form), and will always evaluate to ``1``.
This is often useful as Lagrange multiplier, where a `ConstantFormSpace` can act as the form basis for the real numbers ``\mathbb{R}``.
Note that, compared to the [FormSpaces](@ref FormsSpaces), the `ConstantFormSpace` does not require a function space but only the geometry.

```@docs
ConstantFormSpace
```

### [FormFields](@id FormsFields)

A `FormField` can be used to represent a differential form field (a combination of a basis with coefficients) or forms without an underlying basis.
The former is, for example, useful to represent solution fields or right hand sides, while the latter can be used with analytical expressions to, for example, represent exact solutions or forcings.

```@docs
FormField
AnalyticalFormField
```

Since a `FormField` has coefficients and an `AnalyticalFormField` has an analytical expression, you can inspect them using the following functions.

```@docs
get_coefficients
get_num_coefficients
get_expression
```

## [Evaluating Forms](@id FormsEvaluateFormSpace)

As with any object in `Mantis`, evaluating a form is a matter of calling the `evaluate`-function:

```@docs
evaluate(::AbstractForm{manifold_dim}, ::Int, ::Points.AbstractPoints{manifold_dim}) where {manifold_dim}
```

### [Internals: How a `FormSpace` is evaluated](@id FormsInternalEvaluateFormSpace)

!!! note "Internal behaviour"

    We explain how a `FormSpace` is evaluated. However, this is considered an implementational detail.

The evaluation of a `FormSpace` happens in the canonical domain and is done in two steps.
Firstly, the underlying function space is evaluated.
This evaluation gives us the function values and the basis indices.
Secondly, the function space evaluation is pulled-back to the canonical domain.
What this pullback looks like is dictated by the `form_rank`.
The evaluation then returns the pulled-back values and the basis indices (the indices for the form are the same as for the function space).
This behaviour is encoded using the following two internal functions.

```@docs
_evaluate_form_in_canonical_coordinates
_pullback_to_canonical_coordinates
```

### [Internals: How an `AnalyticalFormField` is evaluated](@id FormsInternalEvaluateAnalyticalFormField)

!!! note "Internal behaviour"

    We explain how an `AnalyticalFormField` is evaluated. However, this is considered an implementational detail.

A user has to define the expression used in the `AnalyticalFormField` in the physical domain. However, in `Mantis`, forms are always evaluated in the canonical domain. This means that any `AnalyticalFormField` must always be pulled-back before the result can be used in other computations.
These pull-backs are determined by the `form_rank` of the `AnalyticalFormField`.

```@docs
_evaluate(::AnalyticalFormField{manifold_dim, 0}, ::Int, ::Points.AbstractPoints{manifold_dim}) where {manifold_dim}
```

## [Operations on Forms](@id FormsOperations)

Now that we know how to create forms, we can look into the operators that we can use on these form objects.
We first look at operators that map forms to forms.

### [Exterior Derivative](@id FormsExteriorDerivative)

The exterior derivative is a generalised derivative, which maps `k`-forms to `k+1` forms, and is known by its alias `d`.
The exterior derivative is a metric-independent operation.
In ``\mathbb{R}^3``, the exterior derivative embodies the well-known gradient (when applied to ``0``-forms), curl (when applied to ``1``-forms), and divergence (when applied to ``2``-forms).

```@docs
ExteriorDerivative
d
```

### [Wedge](@id FormsWedge)

The wedge-operator is a generalisation of products. It takes in two forms (say a `k`-form and an `l`-form) and produces another form (a `k+l`-form).

```@docs
Wedge
∧
```

### [Hodge](@id FormsHodge)

The Hodge-star operator is a metric-dependent operator, which maps `k`-forms to `manifold_dim-k` forms.

```@docs
Hodge
★
```

### [Codifferential](@id FormsCodifferential)

The codifferential, often denoted ``d^{\star}`` or ``\delta``, is a differential operator mapping ``k``-forms to ``k-1``-forms.
On manifolds without boundaries, it is the ``L^2``-adjoint of the exterior derivative.
That is, ``(\alpha^{k-1}, \delta\beta^k) = (d\alpha^{k-1}, \beta^k)``, where ``(\cdot, \cdot)`` is an ``L^2`` inner-product.

```@docs
CoDifferential
dstar
δ
```

### [Algebraic](@id FormsAlgebraic)

The algebraic operators allow you to use operators like addition, subtraction, and multiplication by a scalar on any form.
These operations are implemented as `UnaryFormTransformation` or `BinaryFormTransformation`, depending on whether the operator is a unary or binary operator, respectively.

```@docs
UnaryFormTransformation
BinaryFormTransformation
```

## [Operators returning a vector](@id FormsOperationsToVectors)

Next to operators that map forms to forms, there are operators that map forms to vectors.
At the moment, `Mantis` does not have a type for vectors like it does for forms.
The result of the operators in this section are thus **not** a subtype of `AbstractForm`.

### [Sharp](@id FormsSharp)

The sharp operator takes a ``1``-form and returns the proxy vector field.

```@docs
Sharp
♯
```

The sharp operator also has its own evaluate function, which, like the [evaluate](@ref) on forms, evaluates in the canonical domain.

```@docs
evaluate(::Sharp{manifold_dim}, ::Int, ::Points.AbstractPoints{manifold_dim}) where {manifold_dim}
```

### [Pushforward](@id FormsPushforward)

As explained above, the [Sharp](@ref) turns a ``1``-form into a vector field, but its evaluate still returns values in the canonical domain.
To get values in the physical domain, the vector has to be pushforwarded.
Note that this is not a structure in `Mantis`, just a function.

```@docs
evaluate_pushforward
```

Because the [Sharp](@ref) and pushforward are often used in combination, there is a convenience function to call both operators directly.

```@docs
evaluate_sharp_pushforward
```

## [Operators returning a real value](@id FormsRealValuedOperators)

Another main class of operators are operators that return a value. These are all grouped under the `AbstractRealValuedOperator`-type.

```@docs
AbstractRealValuedOperator
```

### [Integrals](@id FormsIntegrals)

The most important `AbstractRealValuedOperator` is the integral.

```@docs
Integral
∫
```

The integral has its own evaluate function, which only takes the integral and an `element_id` as input, since the integral operator already stores a quadrature rule and thus the evaluation points.

```@docs
evaluate(::Integral{manifold_dim, F, Q}, ::Int) where {manifold_dim, form_rank, expression_rank, F <: AbstractForm{manifold_dim, form_rank, expression_rank}, Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim}}
```

You can retrieve the underlying quadrature rule and the underlying number of evaluation elements (see the docs page of [Quadrature](@ref) for this terminology) with the following functions.

```@docs
get_quadrature_rule
get_num_evaluation_elements
```

### [Algebraic Operations on Integrals](@id FormsAlgebraicReals)

```@docs
UnaryOperatorTransformation
BinaryOperatorTransformation
```

## [Basic Operations](@id FormsBasicOperations)

Next to the operators described in the previous section, you can also interact and inspect form objects using the following methods.

Every form in `Mantis` has a label.
You can retrieve this label using the following function.

```@docs
get_label
```

Most forms or form operators in `Mantis` are structs that contain another form.
For example, the [`ExteriorDerivative`](@ref) stores the form to which it is applied.
To retrieve the underlying form, you can use one of the following functions.

```@docs
get_form
get_forms
get_form_space_tree
```

Additionally, every form is defined on some geometry.
While this geometry is not stored in every form explicitly, it can always be retrieved using the following getter.

```@docs
get_geometry
```

It is also possible to immediately obtain the number of elements in the underlying geometry using the following method.

```@docs
get_num_elements
```

Most forms also have an underlying [FunctionSpaces.AbstractFESpace](@ref).
To obtain this function space, use the following getter.

```@docs
get_fe_space
```

It is also possible to directly obtain some useful information about the underlying function space using the following functions.

```@docs
get_estimated_nnz_per_elem
get_max_local_dim
get_num_basis
```

## [Helper Functions](@id FormsHelpers)

### [De Rham Complexes](@id FormsComplexes)

The De Rham complex (or any other complex for that matter) is an important construct which structure-preserving methods utilise.
As such, there are some (well-)known sequences of [Form Spaces](@ref FormsSpaces) that form a finite-dimensional De Rham complex.
`Mantis` provides some helper functions to easily create the spaces in such a complex.

```@docs
create_tensor_product_bspline_de_rham_complex
create_curvilinear_tensor_product_bspline_de_rham_complex
create_hierarchical_de_rham_complex
update_hierarchical_de_rham_complex
create_polar_spline_de_rham_complex
```

### [Boundary Conditions](@id FormsBCs)

In `Mantis`, boundary conditions are set during assembly (see the [assembly page](@ref DocAssemblyModule) for the details), but the following functions can help in specifying boundary conditions.

```@docs
set_dirichlet_boundary_conditions
trace_basis_idxs
```

### [Other Helper Functions](@id FormsOtherHelpers)

```@docs
get_basis_index_combinations
```

```@meta
CurrentModule = Mantis.Forms
```
# Forms

As explained in ..., differential forms provide an elegant and general framework to deal with the discretisation of PDEs.
One of the most distinguishing features of `Mantis` is its ability to work with these differential forms.
The `Forms` module provides all the required tools to use differential forms in numerical methods.

## What is a differential form in `Mantis`?
The top-level type within the `Forms` module is the `AbstractForm{manifold_dim, form_rank, expression_rank}` type. Every expression involving forms (see [Creating Forms](@ref FormsCreation)) and operations on forms (see [Operations on Forms](@ref FormsOperations)) will be `AbstractForm`s. The only exception are operations that return a real value, such as integrals, see [Operators returning a real value](@ref FormsRealValuedOperators).
```@docs
AbstractForm
```

There are two aliases for `AbstractForm`, which are `AbstractFormField` and `AbstractFormSpace`.
```@docs
AbstractFormField
AbstractFormSpace
```
These are the most common types of `AbstractForm`s.

The above abstract types are used in function signatures, but cannot be instantiated. 
The concrete types that can be instantiated are discussed next.


## [Creating Forms](@id FormsCreation)
You can create two main types of `Forms`: [FormSpaces](@ref FormsSpaces) and [FormFields](@ref FormsFields).

### [FormSpaces](@id FormsSpaces)
`FormSpace`s allow you to distinguish between functions and forms. 
A `FormSpace` is build on top of a [FunctionSpaces.AbstractFESpace](@ref), which acts as its basis.
However, it is the `FormSpace` that dicates the behaviour of the form.
```@docs
FormSpace
```
As explained in ..., differential forms are more expressive than functions.
By using a `FormSpace`, this expressiveness becomes available within your code.
For example, if we start by creating a simple 2D [FunctionSpaces.BSplineSpace](@ref) using the helper [FunctionSpaces.create\_bspline\_space](@ref) (on a unit square with ``4 \times 4`` elements, degree ``3`` and regularity ``2``),
```@repl CreatingFormSpaces
using Mantis
B = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (4, 4), (3, 3), (2, 2))
```
we can use this function space to create two different `FormSpace`s: one for a ``0``-form ``\Lambda^0_h`` and one for a ``2``-form ``\Lambda^2_h`` (a top form in 2D).
```@repl CreatingFormSpaces
Λ⁰ₕ = Forms.FormSpace(0, B, "0-form")
Λ²ₕ = Forms.FormSpace(2, B, "2-form")
```
These two forms have the same basis `B`, but have different transformation properties. 
This will result in the use of different pullbacks (see [How `FormSpaces` are evaluated](@ref FormsInternalEvaluateFormSpace) on how that is reflected in the implementation), and on the operations that you can apply to these forms (see [Operations on Forms](@ref FormsOperations)).

### [ConstantFormSpaces](@id FormsConstantSpaces)
Next to the conventional `FormSpace`, `Mantis` also provides a `ConstantFormSpace`. 
A `ConstantFormSpace` can be instantiated as ``0``- of ``manifold\_dim``-form (so a top form), and will always evaluate to ``1``. This is often usefull as Lagrange multiplier, where a `ConstantFormSpace` can act as the form basis for the real numbers ``\mathbb{R}``. Note that, compared to the [FormSpaces](@ref FormsSpaces), the `ConstantFormSpace` does not require a function space but only the geometry.
```@docs
ConstantFormSpace
```

### [FormFields](@id FormsFields)
`FormField`s are used to represent differential forms fields (a combination of a basis with coefficients) or forms without an underlying basis. The former is, for example, useful to represent solution fields or right hand sides, while the latter can be used with analytical expressions to, for example, represent exact solutions or forcings.
```@docs
FormField
AnalyticalFormField
```

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


### [Internals: How `FormSpace`s are evaluated](@id FormsInternalEvaluateFormSpace)
!!! note "Internal behaviour"
    We explain how `FormSpace`s are evaluated. However, this is considered an implementational detail.

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

### [Internals: How `AnalyticalFormField`s are evaluated](@id FormsInternalEvaluateAnalyticalFormField)
!!! note "Internal behaviour"
    We explain how `AnalyticalFormField`s are evaluated. However, this is considered an implementational detail.



## [Operations on Forms](@id FormsOperations)


### [Exterior Derivative](@id FormsExteriorDerivative)
```@docs
ExteriorDerivative
d
```

### [Hodge](@id FormsExteriorDerivative)
```@docs
Hodge
★
```

### [Codifferential](@id FormsExteriorDerivative)
```@docs
CoDifferential
codifferential
dstar
δ
```

### [Wedge](@id FormsExteriorDerivative)
```@docs
Wedge
∧
```

### [Algebraic](@id FormsExteriorDerivative)
```@docs
UnaryFormTransformation
BinaryFormTransformation
```

### [Sharp](@id FormsSharp)
Not like the others, it is not a form itself!
```@docs
Sharp
♯
evaluate(::Sharp{manifold_dim}, ::Int, ::Points.AbstractPoints{manifold_dim}) where {manifold_dim}
```

### [Pushforward](@id Pushforward)
```@docs
evaluate_pushforward
evaluate_sharp_pushforward
```

## [Operators returning a real value](@id FormsRealValuedOperators)
```@docs
AbstractRealValuedOperator
```

### [Integrals](@id FormsIntegrals)
```@docs
Integral
∫
evaluate(::Integral{manifold_dim, F, Q}, ::Int) where {manifold_dim, form_rank, expression_rank, F <: AbstractForm{manifold_dim, form_rank, expression_rank}, Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim}}
```

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
```@docs
get_manifold_dim
get_form_rank
get_expression_rank
```

```@docs
get_label
get_geometry
get_form
get_forms
```

```@docs
get_num_elements
get_estimated_nnz_per_elem
get_max_local_dim
get_num_basis
```

```@docs
get_fe_space
get_form_space
get_form_space_tree
```

## [Helper Functions](@id FormsHelpers)
```@docs
get_basis_index_combinations
create_tensor_product_bspline_de_rham_complex
create_curvilinear_tensor_product_bspline_de_rham_complex
create_hierarchical_de_rham_complex
update_hierarchical_de_rham_complex
create_polar_spline_de_rham_complex
set_dirichlet_boundary_conditions
trace_basis_idxs
```

## [Creating your own](@id FormsUserCreation)

### Form Operations

### Form Objects

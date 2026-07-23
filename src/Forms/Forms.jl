module Forms

using ..Points
using ..Hierarchy
using ..FunctionSpaces
using ..Geometry
using ..Quadrature

import Combinatorics
import LaTeXStrings
import LinearAlgebra
import SparseArrays

# These exports make these symbols/functions/etc. available outside the Forms module. Only
# if they are also exported in the Mantis.jl file will they be available when using Mantis.

# Abstract types
# The public keyword is only available in Julia 1.11 and up. Since we also support LTS
# (currently 1.10), we add the following line from the manual:
# https://docs.julialang.org/en/v1.12/manual/modules/#Export-lists
VERSION >= v"1.11.0-DEV.469" && eval(
    Meta.parse(
        "public AbstractForm, AbstractFormField, AbstractFormSpace, AbstractRealValuedOperator",
    ),
)

# Type parameter methods
export get_manifold_dim, get_form_rank, get_expression_rank
# Functions dealing with Forms-related quantities
export evaluate
export get_label, get_form
# Functions dealing with Geometry-related quantities
export get_geometry, get_num_elements
# Functions dealing with FunctionSpaces-related quantities
export get_max_local_dim, get_fe_space, get_num_basis

# FormExpressions
export ConstantFormSpace
export FormField,
    AnalyticalFormField, get_coefficients, get_num_coefficients, get_expression
export FormSpace
# FormOperators
export CoDifferential, dstar, δ
export ExteriorDerivative, d
export ★, Hodge
export Integral, ∫, get_quadrature_rule
export evaluate_pushforward, evaluate_sharp_pushforward
export ♯, Sharp
export ∧, Wedge, get_forms

############################################################################################
#                                      Abstract Types                                      #
############################################################################################

"""
    AbstractForm{manifold_dim, form_rank, expression_rank}

Supertype for all form expressions representing differential forms.

# Type parameters

  - `manifold_dim`: Dimension of the manifold on which the form lives. This will always be
    inherited from the underlying function space or geometry.
  - `form_rank`: The rank of the form, i.e. ``0``-form, ``1``-form, ``2``-form, etc.
  - `expression_rank`: The number of bases present in an expression. Is ``0`` if no bases are
    present, ``1`` for a single basis, and ``2`` for two bases. Expression ranks larger
    than ``2`` are not allowed.

!!! note "Non-zero expression rank does not mean that the full expression has a basis."

    If `expression_rank` is larger than ``0``, this means that the expression acts on
    at least one basis, but not necessarily that there is a basis for the total
    expression. For example, applying the exterior derivative to a `FormSpace` will
    result in a form with expression rank ``1``. However, while the `FormSpace` has a
    basis, this does not generate a basis for the exterior derivative (only a spanning
    set).
"""
abstract type AbstractForm{manifold_dim, form_rank, expression_rank} end

"""
    AbstractFormField{manifold_dim, form_rank}

Alias for an `AbstractForm` with expression rank 0, that is, a form expression without a
basis. See [`AbstractForm`](@ref) for more details.
"""
const AbstractFormField{manifold_dim, form_rank} = AbstractForm{manifold_dim, form_rank, 0}

"""
    AbstractFormSpace{manifold_dim, form_rank}

Alias for an `AbstractForm` with expression rank 1, that is, a form expression involving one
basis. See [`AbstractForm`](@ref) for more details.
"""
const AbstractFormSpace{manifold_dim, form_rank} = AbstractForm{manifold_dim, form_rank, 1}

"""
    AbstractRealValuedOperator{manifold_dim}

Supertype for all real-valued operators.
"""
abstract type AbstractRealValuedOperator{manifold_dim} end

############################################################################################
#                                  Type parameter methods                                  #
############################################################################################

"""
    get_manifold_dim(::AbstractForm{manifold_dim}) where {manifold_dim}

Returns the `manifold_dim` of the given form.
"""
get_manifold_dim(::AbstractForm{manifold_dim}) where {manifold_dim} = manifold_dim

function get_manifold_dim(::AbstractRealValuedOperator{manifold_dim}) where {manifold_dim}
    return manifold_dim
end

"""
    get_form_rank(
        ::AbstractForm{manifold_dim, form_rank}
    ) where {manifold_dim, form_rank}

Returns the `form_rank` of the given form.
"""
function get_form_rank(
    ::AbstractForm{manifold_dim, form_rank}
) where {manifold_dim, form_rank}
    return form_rank
end

"""
    get_expression_rank(
        ::AbstractForm{manifold_dim, form_rank, expression_rank}
    ) where {manifold_dim, form_rank, expression_rank}

Returns the `expression_rank` of the given form.
"""
function get_expression_rank(
    ::AbstractForm{manifold_dim, form_rank, expression_rank}
) where {manifold_dim, form_rank, expression_rank}
    return expression_rank
end

get_expression_rank(op::AbstractRealValuedOperator) = get_expression_rank(get_form(op))

############################################################################################
#                                   Getters within Forms                                   #
############################################################################################
"""
    get_label(form::AbstractForm)

Returns the label of the form expression. This is some `AbstractString`, but the concrete
type can vary. Unicode characters and LaTeX string are allowed.
"""
get_label(form::AbstractForm) = form.label

"""
    get_form(form::AbstractForm)
    get_form(op::AbstractRealValuedOperator)

Returns the form underlying the form expression or real-valued operator to which the given
operator is applied.
"""
get_form(form::AbstractForm) = form.form
get_form(op::AbstractRealValuedOperator) = op.form

"""
    get_form_space_tree(form::AbstractFormSpace)

Returns the list of forms of `expression_rank > 0` in the tree of the expression.

For example, if `form` represents `d((α ∧ β) ∧ γ)`, it returns the spaces of `α`, `β`,
and `γ`, if all have expression_rank > 1. If, e.g., `α` has expression_rank = 0, it only
returns the spaces of `β` and `γ`.

# Arguments

  - `form_space::AbstractFormSpace`: The AbstractFormSpace structure.

# Returns

  - `::Tuple(<:AbstractForm)`: The list of forms present in the tree of the expression.
"""
function get_form_space_tree(form::AbstractFormSpace)
    return get_form_space_tree(get_form(form))
end

function get_form_space_tree(op::AbstractRealValuedOperator)
    return get_form_space_tree(get_form(op))
end

function get_form_space_tree(::AbstractFormField)
    return ()
end

############################################################################################
#                              Getters dealing with Geometry                               #
############################################################################################
"""
    get_geometry(form::AbstractForm)
    get_geometry(op::AbstractRealValuedOperator)

Returns the geometry of the given form expression.

Will recurse using [get_form](@ref) to find the geometry.
"""
function get_geometry(form::AbstractForm)
    return get_geometry(get_form(form))
end
get_geometry(op::AbstractRealValuedOperator) = get_geometry(get_form(op))

"""
    get_geometry(
        single_form::AbstractForm, additional_forms::AbstractForm...
    )

If a single form is given, returns the geometry of that form. If additional forms are given,
checks if the number of elements is the same between them; throws an error if not.

!!! warning

    Even if the number of elements is the same, the geometries might be incompatible.
"""
function get_geometry(single_form::AbstractForm, additional_forms::AbstractForm...)
    foreach(form -> check_geometry(single_form, form), additional_forms)

    return get_geometry(single_form)
end

function check_geometry(form_1::AbstractForm, form_2::AbstractForm)
    if get_num_elements(form_1) != get_num_elements(form_2)
        throw(ArgumentError("The given forms have a different number of elements."))
    end
end

"""
    get_num_elements(form::AbstractForm)

Returns the number of elements in the geometry of the given form expression.

# Arguments

  - `form::AbstractForm`: The form expression.

# Returns

  - `Int`: The number of elements in the geometry of the form expression.
"""
function get_num_elements(form::AbstractForm)
    return Geometry.get_num_elements(get_geometry(form))
end

############################################################################################
#                           Getters dealing with FunctionSpaces                            #
############################################################################################
"""
    get_estimated_nnz_per_elem(form::AbstractForm)

Returns the estimated number of non-zero entries per element for the given form expression.

# Arguments

  - `form::AbstractForm`: The form expression.

# Returns

  - `::Int`: The estimated number of non-zero entries per element.
"""
function get_estimated_nnz_per_elem(form::AbstractForm)
    return prod(get_estimated_nnz_per_elem.(get_forms(form)))
end

function get_estimated_nnz_per_elem(::AbstractFormField)
    return 1
end

function get_estimated_nnz_per_elem(form::AbstractFormSpace)
    return get_max_local_dim(get_form(form))
end

"""
    get_max_local_dim(form_space::AbstractFormSpace)

Compute an upper bound of the element-local dimension of `form_space`. Note that this is not
necessarily a tight upper bound.

# Arguments

  - `form_space::AbstractFormSpace`: The form space.

# Returns

  - `::Int`: The element-local upper bound.
"""
function get_max_local_dim(form_space::AbstractFormSpace)
    return FunctionSpaces.get_max_local_dim(get_fe_space(form_space))
end

"""
    get_fe_space(form::FS) where {FS <: AbstractForm}

Returns the finite element space associated with the given form. Note that this function
recurses untill it finds a form (usually a [`FormSpace`](@ref)) which has an underlying
finite element space.

# Arguments

  - `form_space::AbstractForm`: The form space.

# Returns

  - `<:FunctionSpaces.AbstractFESpace`: The finite element space.
"""
function get_fe_space(form::FS) where {FS <: AbstractForm}
    if hasfield(FS, :fem_space)
        return form.fem_space
    end

    return get_fe_space(get_form(form))
end

"""
    get_num_basis(form_space::AbstractFormSpace)

Returns the number of basis functions of the function space associated with the given form
space.

# Arguments

  - `form_space::AbstractFormSpace`: The form space.

# Returns

  - `Int`: The number of basis functions of the function space.
"""
function get_num_basis(form_space::AbstractFormSpace)
    return get_num_basis(get_form(form_space))
end

"""
    get_num_basis(form_space::AbstractFormSpace, element_id::Int)

Returns the number of basis functions at the given element of the function space associated
the given form space.

# Arguments

  - `form_space::AbstractFormSpace`: The form space.

# Returns

  - `Int`: The number of basis functions at the given element.
"""
function get_num_basis(form_space::AbstractFormSpace, element_id::Int)
    return get_num_basis(get_form(form_space), element_id)
end

############################################################################################
#                                         Evaluate                                         #
############################################################################################
"""
    evaluate(
        form::AbstractForm{manifold_dim},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim}

Evaluate any form (expression) on the given `element_id` at the given points `xi`.

!!! note "Evaluation in the canonical domain."

    The evaluation of a form (expression) is always done in the canonical domain, not the
    physical domain. See the [documentation on the Geometry module](@ref DocGeometryModule)
    for more details on these domains.

# Arguments

  - `form::AbstractForm{manifold_dim}`: The differential form space.
  - `element_id::Int`: The global element id. See [Geometry](@ref) for the details.
  - `xi::Points.AbstractPoints{manifold_dim}`: The points in the canonical domain at which to
    evaluate the form. See [Geometry](@ref) and [Points](@ref) for more details on the
    canonical domain and point structure.

# Returns

  - `Vector{Array{Float64, expression_rank+1}}`: Vector of length equal to the number of
    components of the form, where each entry is a `Array{Float64, expression_rank+1}` (so,
    a `Vector` for `AbstractFormField`s and a `Matrix` for `AbstractFormSpace`s)  of size
    `(num_evaluation_points,)`, `(num_evaluation_points, num_basis_functions_on_element)`,
    respectively. For expressions involving two forms (such as the `wedge`), the entries
    will be of type `Array{Float64, 1 + expression_rank_1 + expression_rank_2}`
  - `form_basis_indices::Vector{Vector{Int}}`: The indices of the underlying function space
    that have been evaluated (the inner vector), per basis (the outer vector). For
    `AbstractFormField`s (things without a basis), this will always be [[1]].

# Examples

Evaluating a ``0``-form:

```jldoctest
julia> using Mantis

julia> B = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (2, 2), (2, 2), (1, 1));

julia> Λ⁰ₕ = Forms.FormSpace(0, B, "0-form");  # 0-form with B as basis.

julia> xi = Points.CartesianPoints((LinRange(0.0, 1.0, 2), LinRange(0.0, 1.0, 3)));

julia> Forms.evaluate(Λ⁰ₕ, 1, xi)
([[1.0 0.0 … 0.0 0.0; 0.0 0.5 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.25 0.25]], [[1, 2, 3, 5, 6, 7, 9, 10, 11]])
```

Evaluating a ``2``-form in 2D (a top form). Note how the result is scaled by the pullback
to the canonical domain.

```jldoctest
julia> using Mantis

julia> B = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (2, 2), (2, 2), (1, 1));

julia> Λ²ₕ = Forms.FormSpace(2, B, "2-form");  # 2-form with B as basis.

julia> xi = Points.CartesianPoints((LinRange(0.0, 1.0, 2), LinRange(0.0, 1.0, 3)));

julia> Forms.evaluate(Λ²ₕ, 1, xi)
([[0.25 0.0 … 0.0 0.0; 0.0 0.125 … 0.0 0.0; … ; 0.0 0.0 … 0.0 0.0; 0.0 0.0 … 0.0625 0.0625]], [[1, 2, 3, 5, 6, 7, 9, 10, 11]])
```
"""
function evaluate(
    form::AbstractForm{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    return throw(MethodError(evaluate, (form, element_id, xi)))
end

############################################################################################
#                                         Includes                                         #
############################################################################################

include("./FormExpressions/FormExpressions.jl")
include("./FormOperators/FormOperators.jl")
include("./FormsHelpers.jl")

end

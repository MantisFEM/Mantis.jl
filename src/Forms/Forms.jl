"""
    module Forms

Contains all definitions of forms, including form fields, form spaces, and form expressions.
"""
module Forms

using ..Points
using ..FunctionSpaces
using ..Geometry
using ..Quadrature

import LinearAlgebra
import SparseArrays
import Subscripts

############################################################################################
#                                         Exports                                          #
############################################################################################
include("FormsExports.jl")

############################################################################################
#                                      Abstract Types                                      #
############################################################################################

"""
    AbstractForm{manifold_dim, form_rank, G}

Supertype for all form expressions representing differential forms.

# Type parameters
- `manifold_dim`: Dimension of the manifold.
- `form_rank`: Rank of the differential form.
- `expression_rank`: Rank of the expression. Expressions without basis forms have rank 0,
    with one single set of basis forms have rank 1, with two sets of basis forms have rank
    2. Higher ranks are not possible.
- `G <: Geometry.AbstractGeometry{manifold_dim}`: Type of the underlying geometry.
"""
abstract type AbstractForm{manifold_dim, form_rank, expression_rank, G} end

"""
    AbstractFormField{manifold_dim, form_rank, G} = AbstractForm{
        manifold_dim, form_rank, 0, G

Alias for `AbstractForm`s with expression rank 0, that is, a form without a basis. See
[`AbstractForm`](@ref) for more details.
"""
const AbstractFormField{manifold_dim, form_rank, G} = AbstractForm{
    manifold_dim, form_rank, 0, G
}

"""
    AbstractFormField{manifold_dim, form_rank, G} = AbstractForm{
        manifold_dim, form_rank, 0, G

Alias for `AbstractForm`s with expression rank 1, that is, a form with a basis. See
[`AbstractForm`](@ref) for more details.
"""
const AbstractFormSpace{manifold_dim, form_rank, G} = AbstractForm{
    manifold_dim, form_rank, 1, G
}

"""
    AbstractRealValuedOperator{manifold_dim}

Supertype for all real-valued operators defined over a manifold.

# Type parameters
- `manifold_dim`: Dimension of the manifold.
"""
abstract type AbstractRealValuedOperator{manifold_dim} end

############################################################################################
#                                     Abstract Methods                                     #
############################################################################################

"""
    get_manifold_dim(::AbstractForm{manifold_dim}) where {manifold_dim}

Returns the manifold dimension of the given form.

# Arguments
- `::AbstractForm{manifold_dim}`: The form.

# Returns
- `manifold_dim::Int`: The manifold dimension.
"""
function get_manifold_dim(::AbstractForm{manifold_dim}) where {manifold_dim}
    return manifold_dim
end

function get_manifold_dim(::AbstractRealValuedOperator{manifold_dim}) where {manifold_dim}
    return manifold_dim
end

"""
    get_form_rank(
        ::AbstractForm{manifold_dim, form_rank, expression_rank, G}
    ) where {manifold_dim, form_rank, expression_rank, G}

Returns the form rank of the given form.

# Arguments
- `::FE`: The form.

# Returns
- `::Int`: The form rank of the form.
"""
function get_form_rank(
    ::AbstractForm{manifold_dim, form_rank, expression_rank, G}
) where {manifold_dim, form_rank, expression_rank, G}
    return form_rank
end

"""
    get_expression_rank(
        ::AbstractForm{manifold_dim, form_rank, expression_rank, G}
    ) where {manifold_dim, form_rank, expression_rank, G}

Returns the `expression_rank` of the given form.

# Arguments
- `::FE`: The form expression.

# Returns
- `::Int`: The expression rank of the form.
"""
function get_expression_rank(
    ::AbstractForm{manifold_dim, form_rank, expression_rank, G}
) where {manifold_dim, form_rank, expression_rank, G}
    return expression_rank
end

"""
    get_expression_rank(op::AbstractRealValuedOperator)

Returns the rank of the expression associated with the given operator.

# Arguments
- `op::AbstractRealValuedOperator`: The given operator.

# Returns
- `::Int`: The rank of the expression associated with the given operator.
"""
get_expression_rank(op::AbstractRealValuedOperator) = get_expression_rank(get_form(op))

"""
    get_label(form::AbstractForm)

Returns the label of the form expression.

# Arguments
- `form::AbstractForm`: The form expression.

# Returns
- `String`: The label of the form expression.
"""
get_label(form::AbstractForm) = form.label

"""
    get_geometry(form_expression::AbstractForm)

Returns the geometry of the given form expression.

# Arguments
- `form_expression::AbstractForm`: The form expression.

# Returns
- `<:Geometry.AbstractGeometry`: The geometry of the form expression.
"""
get_geometry(form_expression::AbstractForm) = form_expression.geometry

"""
    get_geometry(
        single_form::AbstractForm, additional_forms::AbstractForm...
    )

If a single form is given, returns the geometry of that form. If additional forms are
given, checks if all the geometries of the different forms refer to the same object in
memory, and then returns it.

# Arguments
- `single_form::AbstractForm`: The first form.
- `additional_forms::AbstractForm...`: Arbitrary number of additional forms.

# Returns
- `<:Geometry.AbstractGeometry`: The geometry of the given form(s).
"""
function get_geometry(single_form::AbstractForm, additional_forms::AbstractForm...)
    all_forms = tuple(single_form, additional_forms...)

    for i in 1:(length(all_forms) - 1)
        if !(get_geometry(all_forms[i]) == get_geometry(all_forms[i + 1]))
            msg1 = "Not all forms share a common geometry. "
            msg2 = "The geometries of form number(i) and form number(i+1) differ."
            throw(ArgumentError(msg1 * msg2))
        end
    end

    return get_geometry(single_form)
end

"""
    get_geometry(op::AbstractRealValuedOperator)

Returns the geometry associated to the form to which the given operator is applied.

# Arguments
- `op::AbstractRealValuedOperator`: The operator to which the form is applied.

# Returns
- `<:AbstractgeometryExpression`: The geometry where the form in the operator is defined.
"""
get_geometry(op::AbstractRealValuedOperator) = get_geometry(get_form(op))

"""
    get_form(op::AbstractRealValuedOperator)

Returns the form to which the given operator is applied.

# Arguments
- `op::AbstractRealValuedOperator`: The operator to which the form is applied.

# Returns
- `<:AbstractFormExpression`: The form to which the operator is applied.
"""
get_form(op::AbstractRealValuedOperator) = op.form

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

"""
    get_estimated_nnz_per_elem(
        form::AbstractForm{manifold_dim, form_rank, expression_rank}
    ) where {manifold_dim, form_rank, expression_rank}

Returns the estimated number of non-zero entries per element for the given form expression.

# Arguments
- `form::AbstractForm{manifold_dim, form_rank, expression_rank}`:
    The form expression.

# Returns
- `::Int`: The estimated number of non-zero entries per element.
"""
function get_estimated_nnz_per_elem(
    form::AbstractForm{manifold_dim, form_rank, expression_rank}
) where {manifold_dim, form_rank, expression_rank}
    return prod(get_estimated_nnz_per_elem.(get_forms(form)))
end

function get_estimated_nnz_per_elem(
    ::AbstractForm{manifold_dim, form_rank, 0}
) where {manifold_dim, form_rank}
    return 1
end

function get_estimated_nnz_per_elem(
    form::AbstractForm{manifold_dim, form_rank, 1}
) where {manifold_dim, form_rank}
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
    get_fe_space(form::FS) where {FS <: AbstractFormSpace}

Returns the finite element space associated with the given form space.

# Arguments
- `form_space::AbstractFormSpace`: The form space.

# Returns
- `<:FunctionSpaces.AbstractFESpace`: The finite element space.
"""
function get_fe_space(form::FS) where {FS <: AbstractFormSpace}
    if hasfield(FS, :fem_space)
        return form.fem_space
    end

    return get_fe_space(get_form(form))
end

"""
    get_form_space_tree(form_space::AbstractFormSpace)

Returns the list of spaces of forms of `expression_rank > 0` in the tree of the expression.
Since `AbstractFormSpace` has a single form of `expression_rank = 1` it returns a `Tuple`
with the space of the `AbstractFormSpace`.

# Arguments
- `form_space::AbstractFormSpace`: The AbstractFormSpace structure.

# Returns
- `::Tuple(<:AbstractForm)`: The list of forms present in the tree of the expression, in
    this case the form space.
"""
function get_form_space_tree(form_space::AbstractFormSpace)
    return (get_form(form_space),)
end

"""
    get_form_space_tree(form_field::AbstractFormField)

Returns the list of spaces of forms of `expression_rank > 0` in the tree of the expression.
Since `FormField` has a single form of `expression_rank = 0` it returns an empty `Tuple`.

# Arguments
- `form_field::AbstractFormField`: The AbstractFormField structure.

# Returns
- `Tuple(<:AbstractForm)`: The list of forms present in the tree of the expression, in this case empty.
"""
function get_form_space_tree(form_field::AbstractFormField)
    return ()
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
    return FunctionSpaces.get_num_basis(get_fe_space(form_space))
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
    return FunctionSpaces.get_num_basis(get_fe_space(form_space), element_id)
end

############################################################################################
#                                         Includes                                         #
############################################################################################

include("./FormExpressions/FormExpressions.jl")
include("./FormOperators/FormOperators.jl")
include("./FormsHelpers.jl")

end

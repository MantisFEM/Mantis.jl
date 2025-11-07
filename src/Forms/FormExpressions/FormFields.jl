############################################################################################
#                                        Structures                                        #
############################################################################################

"""
    FormField{manifold_dim, form_rank, G, FS} <:
    AbstractFormField{manifold_dim, form_rank, 0, G}

Represents a differential form field.

# Fields
- `geometry::G`: The geometry associated with this field.
- `form_space::FS`: The form space associated with this field.
- `coefficients::Vector{Float64}`: Coefficients of the form field.
- `label::String`: Label for the form field.

# Type parameters
- `manifold_dim`: Dimension of the manifold.
- `form_rank`: Rank of the differential form.
- `G`: Type of the geometry.
- `FS`: Type of the form space.

# Inner Constructors
- `FormField(form_space::FS, coefficients::Vector{Int} label::String)`: General constructor
    for form fields.
- `FormField(form_space::FS, label::String)`: Constructor with zero coefficients.
"""
struct FormField{manifold_dim, form_rank, G, FS} <:
       AbstractFormField{manifold_dim, form_rank, G}
    geometry::G
    form_space::FS
    coefficients::Vector{Float64}
    label::String

    """
        FormField(
            form_space::FS, label::String
        ) where {
            manifold_dim,
            form_rank,
            G <: Geometry.AbstractGeometry{manifold_dim},
            FS <: AbstractFormSpace{manifold_dim, form_rank, G},
        }

    Construct a FormField with zero coefficients.

    # Arguments
    - `form_space::FS`: The form space.
    - `coefficients::Vector{Float64}`: The form field coefficients.
    - `label::String`: Label for the form field.

    # Returns
    - `FormField`: A new FormField instance.
    """
    function FormField(
        form_space::FS, coefficients::Vector{Float64}, label::String
    ) where {
        manifold_dim,
        form_rank,
        G <: Geometry.AbstractGeometry{manifold_dim},
        FS <: AbstractFormSpace{manifold_dim, form_rank, G},
    }
        if length(coefficients) != get_num_basis(form_space)
            throw(ArgumentError("""\
                      The number of coefficients ($(length(coefficients))) must match the\
                      number of basis functions ($(get_num_basis(form_space))).
                      """))
        end

        return new{manifold_dim, form_rank, G, FS}(
            get_geometry(form_space), form_space, coefficients, label
        )
    end

    """
        FormField(form_space::FS, label::String)

    Construct a FormField with zero coefficients.

    # Arguments
    - `form_space::FS`: The form space.
    - `label::String`: Label for the form field.

    # Returns
    - `FormField`: A new FormField instance with zero coefficients.
    """
    function FormField(form_space::FS, label::String) where {FS}
        coefficients = zeros(get_num_basis(form_space))

        return FormField(form_space, coefficients, label)
    end
end

"""
    AnalyticalFormField{manifold_dim, form_rank, G, E} <:
    AbstractFormField{manifold_dim, form_rank, G}

Represents an analytical differential form field.

# Fields
- `geometry::G`: The geometry associated with this field.
- `expression::E`: The expression defining the form field.
- `label::String`: Label for the form field.

# Type parameters
- `manifold_dim`: Dimension of the manifold.
- `form_rank`: Rank of the differential form.
- `G`: Type of the geometry.
- `E`: Type of the expression.

# Inner Constructors
- `AnalyticalFormField(form_rank::Int, expression::E, geometry::G, label::String)`: General
    constructor for analytical form fields.

"""
struct AnalyticalFormField{manifold_dim, form_rank, G, E} <:
       AbstractFormField{manifold_dim, form_rank, G}
    geometry::G
    expression::E
    label::String

    """
        AnalyticalFormField(
            form_rank::Int, expression::E, geometry::G, label::String
        ) where {
            manifold_dim, E <: Function, G <: Geometry.AbstractGeometry{manifold_dim}
        }

    General constructor for analytical form fields.

    # Arguments
    - `form_rank::Int`: The rank of the form field.
    - `expression::E`: The expression defining the form field.
    - `geometry::G`: The geometry associated with this field.
    - `label::String`: The label for the form field.

    # Returns
    - `AnalyticalFormField`: The new analytical form field.
    """
    function AnalyticalFormField(
        form_rank::Int, expression::E, geometry::G, label::String
    ) where {manifold_dim, E <: Function, G <: Geometry.AbstractGeometry{manifold_dim}}
        return new{manifold_dim, form_rank, G, E}(geometry, expression, label)
    end
end

############################################################################################
#                                   Getters and setters                                    #
############################################################################################
"""
    get_form_space(form_field::FormField)

Returns the form space associated with the form field.

# Arguments
- `form_field::FormField`: The form field.

# Returns
- `<:AbstractFormSpace`: The form space associated with the form field.
"""
get_form_space(form_field::FormField) = form_field.form_space

"""
    get_form_space_tree(form_field::AbstractFormField)

Returns the list of spaces of forms of `expression_rank > 0` in the tree of the expression.
Since `FormField` has a single form of `expression_rank = 0` it returns an empty `Tuple`.

# Arguments
- `form_field::AbstractFormField`: The AbstractFormField structure.

# Returns
- `Tuple(<:AbstractFormExpression)`: The list of forms present in the tree of the expression, in this case empty.
"""
function get_form_space_tree(form_field::AbstractFormField)
    return ()  # return an empty Tuple because AbstractFormField has only one form of expression_rank = 0
end


"""
    get_coefficients(form_field::FormField)

Returns the coefficients of the form field.

# Arguments
- `form_field::FormField`: The form field.

# Returns
- `Vector{Float64}`: The coefficients of the form field.
"""
get_coefficients(form_field::FormField) = form_field.coefficients

"""
    get_num_coefficients(form_field::FormField)

Returns the number of coefficients of the form field.

# Arguments
- `form_field::FormField`: The form field.

# Returns
- `Int`: The number of coefficients (dofs) of the form field.
"""
get_num_coefficients(form_field::FormField) = size(form_field.coefficients, 1)


"""
    get_expression(form_field::AnalyticalFormField)

Returns the expression of the analytical form field.

# Arguments
- `form_field::AnalyticalFormField`: The analytical form field.

# Returns
- `<:Function`: The expression of the analytical form field.
"""

get_expression(form_field::AnalyticalFormField) = form_field.expression

############################################################################################
#                                    Evaluation methods                                    #
############################################################################################

"""
    evaluate(
        form_field::FormField{manifold_dim, form_rank, G, FS},
        element_idx::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {
        manifold_dim,
        form_rank,
        G <: Geometry.AbstractGeometry{manifold_dim},
        FS <: AbstractFormSpace{manifold_dim, form_rank, G},
    }

Evaluates a differential form field at given canonical points `xi` mapped to the parametric
element given by `element_idx`.

# Arguments
- `form::FS`: The differential form field.
- `element_idx::Int`: The parametric element identifier.
- `xi::NTuple{manifold_dim, Vector{Float64}`: The set of canonical points.

# Returns
- `Vector{Vector{Float64}}`: Vector of length equal to the number of components of the form,
    where each entry is a `Vector{Float64}`  of length `n_evaluation_points`.
- `Vector{Vector{Int}}`: This vector is always `[[1]]` because form fields have no basis.
"""
function evaluate(
    form_field::FormField{manifold_dim, form_rank, G, FS},
    element_idx::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {
    manifold_dim,
    form_rank,
    G <: Geometry.AbstractGeometry{manifold_dim},
    FS <: AbstractFormSpace{manifold_dim, form_rank, G},
}
    n_form_components = binomial(manifold_dim, form_rank)
    form_basis_eval, form_basis_indices = evaluate(
        get_form_space(form_field), element_idx, xi
    )
    form_eval = Vector{Vector{Float64}}(undef, n_form_components)
    form_field_coefficients = get_coefficients(form_field)
    for form_component_idx in 1:n_form_components
        form_eval[form_component_idx] =
            form_basis_eval[form_component_idx] *
            form_field_coefficients[form_basis_indices[1]]
    end

    # We need to wrap form_basis_indices in [] to return a vector of vector to allow
    # multi-indexed expressions, like wedges
    return form_eval, [[1]]
end

"""
    evaluate(
        form_field::AnalyticalFormField{manifold_dim},
        element_idx::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim}

Evaluates a differential form field at given canonical points `xi` mapped to the parametric
element given by `element_idx`.

# Arguments
- `form::FS`: The analytical, differential form field.
- `element_idx::Int`: The parametric element identifier.
- `xi::NTuple{manifold_dim, Vector{Float64}`: The set of canonical points.

# Returns
- `Vector{Vector{Float64}}`: Vector of length equal to the number of components of the form,
    where each entry is a `Vector{Float64}`  of length `n_evaluation_points`.
- `Vector{Vector{Int}}`: This vector is always `[[1]]` because form fields have no basis.
"""
function evaluate(
    form_field::AnalyticalFormField{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    return _evaluate(form_field, element_id, xi)
end

function _evaluate(
    form_field::AnalyticalFormField{manifold_dim, 0},
    element_idx::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    x = Geometry.evaluate(get_geometry(form_field), element_idx, xi)
    form_eval = get_expression(form_field)(x)

    # We need to wrap form_basis_indices in [] to return a vector of vector to allow
    # multi-indexed expressions, like wedges
    return form_eval, [[1]]
end

function _evaluate(
    form_field::AnalyticalFormField{manifold_dim, manifold_dim},
    element_idx::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    geometry = get_geometry(form_field)
    if Geometry.get_image_dim(geometry) != manifold_dim
        throw("Image manifold must have the same dimension as the domain manifold.")
    end

    x = Geometry.evaluate(geometry, element_idx, xi)
    J = Geometry.jacobian(geometry, element_idx, xi)  # Jₖⱼ = ∂Φᵏ\∂ξⱼ
    form_eval = get_expression(form_field)(x)
    form_eval[1][:] .*= LinearAlgebra.det.(eachslice(J; dims=1))

    # We need to wrap form_basis_indices in [] to return a vector of vector to allow
    # multi-indexed expressions, like wedges
    return form_eval, [[1]]
end

function _evaluate(
    form_field::AnalyticalFormField{manifold_dim, 1},
    element_idx::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    geometry = get_geometry(form_field)
    x = Geometry.evaluate(geometry, element_idx, xi)
    J = Geometry.jacobian(geometry, element_idx, xi)  # Jₖⱼ = ∂Φᵏ\∂ξⱼ
    form_eval = get_expression(form_field)(x) # size: num_points x image_dim
    num_eval_points = size(x, 1)
    image_dim = length(form_eval)
    form_pullback = Vector{Vector{Float64}}(undef, manifold_dim)
    for j in 1:manifold_dim
        form_pullback[j] = zeros(num_eval_points)
    end

    a = zeros(num_eval_points, manifold_dim)
    for j in 1:image_dim
        for i in 1:num_eval_points
            a[i, :] .+= form_eval[j][i] .* J[i, j, :]
        end
    end

    for j in 1:manifold_dim
        form_pullback[j] = a[:, j]
    end

    # We need to wrap form_basis_indices in [] to return a vector of vector to allow multi-indexed expressions, like wedges
    return form_pullback, [[1]]
end

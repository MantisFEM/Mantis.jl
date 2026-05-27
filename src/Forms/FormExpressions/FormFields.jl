############################################################################################
#                                        Structures                                        #
############################################################################################

"""
    FormField{manifold_dim, form_rank, FS, L} <: AbstractFormField{manifold_dim, form_rank}

Represents a differential form field, i.e., a differential form with `coefficients` and
`form_space`. Note that this is considered a field, and thus to **not** have a basis.

# Fields
- `form_space::FS`: The form space associated with this field.
- `coefficients::Vector{Float64}`: Coefficients of the form field.
- `label::AbstractString`: Label for the form field.

# Type parameters
- `manifold_dim`: Dimension of the manifold.
- `form_rank`: Rank of the differential form.
- `FS`: Type of the form space.

# Inner Constructors
- `FormField(
    form_space::AbstractFormSpace{manifold_dim, form_rank},
    coefficients::Vector{Float64}=zeros(get_num_basis(form_space)),
    label::AbstractString="field over "*get_label(form_space),
)`: General constructor for form fields. Note that the coefficients default to zero if not
    given, and that the label also has a default.
"""
struct FormField{manifold_dim, form_rank, FS, L} <:
       AbstractFormField{manifold_dim, form_rank}
    form_space::FS
    coefficients::Vector{Float64}
    label::L

    function FormField(
        form_space::FS,
        coefficients::Vector{Float64}=zeros(get_num_basis(form_space)),
        label::AbstractString="field over "*get_label(form_space),
    ) where {manifold_dim, form_rank, FS <: AbstractFormSpace{manifold_dim, form_rank}}
        if length(coefficients) != get_num_basis(form_space)
            throw(
                ArgumentError(
                    LazyString(
                        "The number of coefficients (",
                        length(coefficients),
                        ") must match the number of basis functions (",
                        get_num_basis(form_space),
                        ") in the space, but doesn't.",
                    )
                )
            )
        end

        return new{manifold_dim, form_rank, FS, typeof(label)}(
            form_space, coefficients, label
        )
    end
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
function FormField(form_space::FS, label::Union{AbstractString, Nothing}=nothing) where {FS}
    coefficients = zeros(get_num_basis(form_space))
    if isnothing(label)
        return FormField(form_space, coefficients, Forms.get_label(form_space))
    end

    return FormField(form_space, coefficients, label)
end

"""
    AnalyticalFormField{manifold_dim, form_rank, G, E} <:
    AbstractFormField{manifold_dim, form_rank}

Represents an analytical differential form field.

# Fields
- `geometry::G`: The geometry associated with this field.
- `expression::E`: The expression defining the form field.
- `label::AbstractString`: Label for the form field.

# Type parameters
- `manifold_dim`: Dimension of the manifold.
- `form_rank`: Rank of the differential form.
- `G`: Type of the geometry.
- `E`: Type of the expression.

# Inner Constructors
- `AnalyticalFormField(form_rank::Int, expression::E, geometry::G, label::AbstractString)`:
    General constructor for analytical form fields.

"""
struct AnalyticalFormField{manifold_dim, form_rank, G, E, L} <:
       AbstractFormField{manifold_dim, form_rank}
    geometry::G
    expression::E
    label::L

    """
        AnalyticalFormField(
            form_rank::Int, expression::E, geometry::G, label::AbstractString
        ) where {
            manifold_dim, E <: Function, G <: Geometry.AbstractGeometry{manifold_dim}
        }

    General constructor for analytical form fields.

    # Arguments
    - `form_rank::Int`: The rank of the form field.
    - `expression::E`: The expression defining the form field.
    - `geometry::G`: The geometry associated with this field.
    - `label::AbstractString`: The label for the form field.

    # Returns
    - `AnalyticalFormField`: The new analytical form field.
    """
    function AnalyticalFormField(
        form_rank::Int, expression::E, geometry::G, label::AbstractString
    ) where {manifold_dim, E <: Function, G <: Geometry.AbstractGeometry{manifold_dim}}
        return new{manifold_dim, form_rank, G, E, typeof(label)}(
            geometry, expression, label
        )
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

get_form(form_field::FormField) = form_field.form_space

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

get_geometry(form_field::AnalyticalFormField) = form_field.geometry

############################################################################################
#                                    Evaluation methods                                    #
############################################################################################

function evaluate(
    form_field::FormField{manifold_dim, form_rank, FS},
    element_idx::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank, FS}
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

function evaluate(
    form_field::AnalyticalFormField{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    return _evaluate(form_field, element_id, xi)
end

"""
    _evaluate(
        form_field::AnalyticalFormField{manifold_dim, form_rank},
        element_idx::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim}

Internal function to evaluate an analytical form field, by first pulling back the form to
the canonical domain. The used pull-back is dictated by the `form_rank`.

# Arguments
- See [evaluate](@ref) for the details.

# Returns
- See [evaluate](@ref) for the details.
"""
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
    form_eval[1][:] .*= LinearAlgebra.det.(J)

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
    form_eval = get_expression(form_field)(x)
    num_eval_points = size(x, 1)
    image_dim = Geometry.get_image_dim(geometry)
    form_pullback = Vector{Vector{Float64}}(undef, manifold_dim)
    for j in 1:manifold_dim
        form_pullback[j] = zeros(num_eval_points)
    end

    a = zeros(num_eval_points, manifold_dim)
    for j in 1:image_dim
        for i in 1:num_eval_points
            a[i, :] .+= form_eval[j][i] .* J[i][j, :]
        end
    end

    for j in 1:manifold_dim
        form_pullback[j] = a[:, j]
    end

    # We need to wrap form_basis_indices in [] to return a vector of vector to allow multi-indexed expressions, like wedges
    return form_pullback, [[1]]
end

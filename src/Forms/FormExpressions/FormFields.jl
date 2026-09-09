############################################################################################
#                                        Structures                                        #
############################################################################################

"""
    FormField{manifold_dim, form_rank, FS, L} <: AbstractFormField{manifold_dim, form_rank}

Represents a differential form field, i.e., a differential form with `coefficients` and
`form_space`. Note that this is considered a field, and thus to **not** have a basis.

# Constructors
- `FormField(
        form_space::FS,
        coefficients::Vector{Float64}=zeros(get_num_basis(form_space)),
        label::AbstractString=get_label(form_space),
    )`: General constructor for form fields. Note that the coefficients default to zero if
    not given, and that the label also has a default.

# Example
```jldoctest
julia> using Mantis

julia> B = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (4, 4), (3, 3), (2,2));

julia> Λ⁰ₕ = Forms.FormSpace(0, B, "0-form");  # 0-form space with B as basis.

julia> coefficients = ones(Forms.get_num_basis(Λ⁰ₕ)); # Create some coefficients

julia> α⁰ₕ = Forms.FormField(Λ⁰ₕ, coefficients, "0-form-field");  # 0-form field with Λ⁰ₕ as basis and all ones as coefficients.

julia> β⁰ₕ = Forms.FormField(Λ⁰ₕ);  # 0-form field with Λ⁰ₕ as basis and all zero coefficients.
```

# Fields
- `form_space::FS`: The form space associated with this field.
- `coefficients::Vector{Float64}`: Coefficients of the form field.
- `label::AbstractString`: Label for the form field.

# Type parameters
- `manifold_dim`: Dimension of the manifold.
- `form_rank`: Rank of the differential form.
- `FS`: Type of the form space.
- `L`: Type of the label (an `AbstractString`).
"""
struct FormField{manifold_dim, form_rank, FS, L} <:
       AbstractFormField{manifold_dim, form_rank}
    form_space::FS
    coefficients::Vector{Float64}
    label::L

    function FormField(
        form_space::FS,
        coefficients::Vector{Float64}=zeros(get_num_basis(form_space)),
        label::AbstractString=get_label(form_space),
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
                    ),
                ),
            )
        end

        return new{manifold_dim, form_rank, FS, typeof(label)}(
            form_space, coefficients, label
        )
    end
end

"""
    AnalyticalFormField{manifold_dim, form_rank, PB, S, G, E, L} <:
    AbstractFormField{manifold_dim, form_rank}

Represents an analytical differential form field.

The pullback and source location can be changed. By default, these is the standard
[`FormPullback`](@ref) from the physical domain. See the
[documentation on the Geometry module](@ref DocGeometryModule) for the difference between
the domains.

# Constructors
- `AnalyticalFormField(
        form_rank::Int,
        expression::E,
        geometry::G,
        label::AbstractString,
        ::Type{PB}=FormPullback,
        ::Type{S}=Physical,
    ) where {
        manifold_dim, E <: Function, G <: Geometry.AbstractGeometry{manifold_dim}, PB, S
    }`: General constructor for analytical form fields.

# Example
```jldoctest
julia> using Mantis

julia> function my_form_expression(input)
           x = input[:, 1]
           y = input[:, 2]
           return [@. sin(x) * sin(y)]
       end;

julia> geometry = Geometry.create_cartesian_box((0.0, 0.0), (1.0, 1.0), (4, 4));

julia> α⁰ₕ = Forms.AnalyticalFormField(0, my_form_expression, geometry, "Analytical 0-form");

julia> α²ₕ = Forms.AnalyticalFormField(2, my_form_expression, geometry, "Analytical 2-form");
```

# Fields
- `geometry::G`: The geometry associated with this field.
- `expression::E`: The expression defining the form field.
- `label::AbstractString`: Label for the form field.

# Type parameters
- `manifold_dim`, `form_rank`, `expression_rank`: See [`AbstractForm`](@ref).
- `PB`: Type of the pullback used, see [`AbstractPullback`](@ref).
- `S`: Type of the domain used, see [`AbstractPullbackLocation`](@ref).
- `G`: Type of the geometry.
- `E`: Type of the expression.
- `L`: Type of the label (an `AbstractString`).
"""
struct AnalyticalFormField{manifold_dim, form_rank, PB, S, G, E, L} <:
       AbstractFormField{manifold_dim, form_rank}
    geometry::G
    expression::E
    label::L

    function AnalyticalFormField(
        form_rank::Int,
        expression::E,
        geometry::G,
        label::AbstractString,
        ::Type{PB}=FormPullback,
        ::Type{S}=Physical,
    ) where {
        manifold_dim, E <: Function, G <: Geometry.AbstractGeometry{manifold_dim}, PB, S
    }
        return new{manifold_dim, form_rank, PB, S, G, E, typeof(label)}(
            geometry, expression, label
        )
    end
end

############################################################################################
#                                   Getters and setters                                    #
############################################################################################

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

Returns the expression of the analytical form field. Remember that the expression is
defined in the physical domain. See [`AnalyticalFormField`](@ref) for the details.

# Arguments
- `form_field::AnalyticalFormField`: The analytical form field.

# Returns
- `<:Function`: The expression of the analytical form field.
"""
get_expression(form_field::AnalyticalFormField) = form_field.expression

get_geometry(form_field::AnalyticalFormField) = form_field.geometry

function get_pullback_type(
    form::AnalyticalFormField{manifold_dim, form_rank, PB}
) where {manifold_dim, form_rank, PB}
    return PB
end

function get_pullback(
    form::FormField{manifold_dim, form_rank, FS}, ::Type{D}=Canonical
) where {
    manifold_dim, form_rank, PB, S, D, F, FS <: FormSpace{manifold_dim, form_rank, PB, S, F}
}
    return PB(form, S, D)
end

function get_pullback(
    form::AnalyticalFormField{manifold_dim, form_rank, PB, S}, ::Type{D}=Canonical
) where {manifold_dim, form_rank, PB, S, D}
    return PB(form, S, D)
end

############################################################################################
#                                    Evaluation methods                                    #
############################################################################################

function evaluate(
    form_field::FormField{manifold_dim, form_rank, FS},
    element_idx::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank, FS}
    n_form_components = binomial(manifold_dim, form_rank)
    form_basis_eval, form_basis_indices = evaluate(get_form(form_field), element_idx, xi)
    TT = eltype(eltype(form_basis_eval))
    form_eval = Vector{Vector{TT}}(undef, n_form_components)
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
    form::AnalyticalFormField{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    return evaluate(get_pullback(form), element_id, xi)
end

function evaluate(
    form::FormPullback{manifold_dim, form_rank, expression_rank, S, D, F},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank, expression_rank, S, D, F <: AnalyticalFormField}
    form_field = get_form(form)
    x = Geometry.evaluate(get_geometry(form_field), element_id, xi)
    evaluations = get_expression(form_field)(x)

    pullback!(evaluations, form, element_id, xi)

    return evaluations, [[1]]
end

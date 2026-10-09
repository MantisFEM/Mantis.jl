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
    AnalyticalFormField{manifold_dim, form_rank, G, E, L} <:
    AbstractFormField{manifold_dim, form_rank}

Represents an analytical differential form field.

The analytical `expression` should be a Julia function defining the form in the *physical*
domain. See the [documentation on the Geometry module](@ref DocGeometryModule) for the
difference between the domains used in `Mantis`.

The `expression` takes the physical points as a matrix with one row per point and one
column per coordinate, and returns a vector with one vector of values per component of the
form, in the physical basis:
- ``0``-forms ``f``: `[f]`;
- ``1``-forms ``f_1\\,dx^1 + \\dots + f_m\\,dx^m``: `[f₁, …, fₘ]`, where `m` is the image
    dimension of the geometry;
- ``2``-forms in 3D
    ``f_1\\,dx^2\\wedge dx^3 + f_2\\,dx^3\\wedge dx^1 + f_3\\,dx^1\\wedge dx^2``:
    `[f₁, f₂, f₃]`;
- top forms ``f\\,dx^1\\wedge\\dots\\wedge dx^n``: `[f]`.
Top forms and ``2``-forms in 3D require the image dimension to be equal to the manifold
dimension. When evaluated, the form is pulled back to the canonical domain of each element.

# Constructors
- `AnalyticalFormField(form_rank::Int, expression::E, geometry::G, label::AbstractString)`:
    General constructor for analytical form fields.

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
- `manifold_dim`, `form_rank`, `expression_rank`: See [`AbstractForm`](@ref) for the details.
- `G`: Type of the geometry.
- `E`: Type of the expression.
- `L`: Type of the label (an `AbstractString`).
"""
struct AnalyticalFormField{manifold_dim, form_rank, G, E, L} <:
       AbstractFormField{manifold_dim, form_rank}
    geometry::G
    expression::E
    label::L

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
        get_form(form_field), element_idx, xi
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
- See [`evaluate`](@ref) for the details.

# Returns
- See [`evaluate`](@ref) for the details.
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

function _evaluate(
    form_field::AnalyticalFormField{3, 2}, element_idx::Int, xi::Points.AbstractPoints{3}
)
    # We need to check that the image dim is not larger than 3
    # This is still possible to setup but we do not have the operators
    # generalized to larger dimensions, once we do that, this function must
    # be generalized and this check removed.
    geometry = get_geometry(form_field)
    if Geometry.get_image_dim(geometry) != 3
        throw(
            ArgumentError(
                LazyString(
                    "Analytical 2-forms in 3D require a geometry with image dimension 3, ",
                    "but the image dimension is ",
                    Geometry.get_image_dim(geometry),
                    ".",
                ),
            ),
        )
    end

    # Evaluate geometric data
    x = Geometry.evaluate(geometry, element_idx, xi)
    J = Geometry.jacobian(geometry, element_idx, xi)  # Jₖⱼ = ∂Φᵏ\∂ξⱼ

    # Components of f₁dx²∧dx³ + f₂dx³∧dx¹ + f₃dx¹∧dx² in the physical basis.
    form_eval = get_expression(form_field)(x)
    num_eval_points = size(x, 1)
    form_pullback = [zeros(num_eval_points) for _ in 1:3]
    
    # The pullback has the components α₁dξ²∧dξ³ + α₂dξ³∧dξ¹ + α₃dξ¹∧dξ². With the tangent
    # vectors tᵢ = ∂Φ/∂ξᵢ (the columns of J) and f = (f₁, f₂, f₃), they are
    #   α₁ = f ⋅ (t₂ × t₃),  α₂ = f ⋅ (t₃ × t₁),  α₃ = f ⋅ (t₁ × t₂),
    # that is, α = det(J) J⁻¹ f.
    for point in 1:num_eval_points
        f = (form_eval[1][point], form_eval[2][point], form_eval[3][point])
        t₁, t₂, t₃ = J[point][:, 1], J[point][:, 2], J[point][:, 3]
        for (component, (tᵢ, tⱼ)) in enumerate(((t₂, t₃), (t₃, t₁), (t₁, t₂)))
            normal = LinearAlgebra.cross(tᵢ, tⱼ)
            form_pullback[component][point] =
                f[1] * normal[1] + f[2] * normal[2] + f[3] * normal[3]
        end
    end

    # We need to wrap form_basis_indices in [] to return a vector of vector to allow
    # multi-indexed expressions, like wedges.
    return form_pullback, [[1]]
end

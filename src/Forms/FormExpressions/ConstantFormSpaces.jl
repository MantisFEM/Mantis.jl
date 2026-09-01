############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    ConstantFormSpace{manifold_dim, form_rank, G, L} <:
    AbstractFormSpace{manifold_dim, form_rank}

Constant scalar differential form.

This can, for instance, be used as a Lagrange multiplier enforcing a zero-average
constraint on another differential form.

# Constructors
- `ConstantFormSpace(form_rank::Int, geometry::G, label::L)`: Generic constructor.

# Example
```jldoctest
julia> using Mantis

julia> geometry = Geometry.create_cartesian_box((0.0, 0.0), (1.0, 1.0), (4, 4));

julia> Λ⁰ₕ = Forms.ConstantFormSpace(0, geometry, "0-form");  # 0-form constant on geometry.

julia> Λ²ₕ = Forms.ConstantFormSpace(2, geometry, "2-form");  # 2-form constant on geometry.
```

# Fields
- `geometry::G`: The geometry [`Geometry.AbstractGeometry`](@ref) on which the
    `ConstantFormSpace` should be created. The `manifold_dim` will be inherited from this
    geometry.
- `label::L`: Label for the constant form space. This will be used in export and plotting
    functions to easily identify the form.

# Type parameters
- `manifold_dim`: Dimension of the manifold.
- `form_rank`: Rank of the differential form.
- `G`: Type of the geometry (a [`Geometry.AbstractGeometry`](@ref)).
- `L`: Type of the label (an `AbstractString`).
"""
struct ConstantFormSpace{manifold_dim, form_rank, G, L} <:
       AbstractFormSpace{manifold_dim, form_rank}
    geometry::G
    label::L

    function ConstantFormSpace(
        form_rank::Int, geometry::G, label::AbstractString
    ) where {manifold_dim, G <: Geometry.AbstractGeometry{manifold_dim}}
        if (form_rank ∉ Set([0, manifold_dim]))
            throw(
                ArgumentError(
                    "Mantis.Forms.ConstantFormSpace: form_rank = $form_rank with " *
                    "manifold_dim = $manifold_dim requires form_rank to be 0 or " *
                    "manifold_dim.",
                ),
            )
        end
        return new{manifold_dim, form_rank, G, typeof(label)}(geometry, label)
    end
end

############################################################################################
#                                   Getters and setters                                    #
############################################################################################

get_num_basis(::ConstantFormSpace) = 1

get_num_basis(::ConstantFormSpace, ::Int) = 1

get_max_local_dim(::ConstantFormSpace) = 1

get_estimated_nnz_per_elem(::ConstantFormSpace) = 1

get_form(form::ConstantFormSpace) = form

get_form_space_tree(form::ConstantFormSpace) = (get_form(form_space),)

get_geometry(form::ConstantFormSpace) = form.geometry

function get_fe_space(::ConstantFormSpace)
    throw(
        ArgumentError(
            "ConstantFormSpace does not have an associated finite element space.",
        ),
    )
end

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

function evaluate(
    ::ConstantFormSpace{manifold_dim, 0},
    ::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    num_evaluation_points = Points.get_num_points(xi)
    return [ones(Float64, num_evaluation_points, 1)], [[1]]
end

function evaluate(
    form_space::ConstantFormSpace{manifold_dim, manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    num_evaluation_points = Points.get_num_points(xi)
    _, sqrt_g = Geometry.metric(get_geometry(form_space), element_id, xi)  # Jₖⱼ = ∂Φᵏ\\∂ξⱼ
    form_eval = [ones(Float64, num_evaluation_points, 1)]
    form_eval[1][:] .*= sqrt_g

    return form_eval, [[1]]
end

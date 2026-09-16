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
- `ConstantFormSpace(
        form_rank::Int,
        geometry::G,
        label::AbstractString,
        ::Type{PB}=FormPullback,
        ::Type{S}=Physical,
    ) where {manifold_dim, PB, S, G <: Geometry.AbstractGeometry{manifold_dim}}`: Generic
        constructor.

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
- `manifold_dim`, `form_rank`: See [`AbstractForm`](@ref).
- `PB`: Type of the pullback used, see [`AbstractPullback`](@ref).
- `S`: Type of the domain used, see [`AbstractPullbackLocation`](@ref).
- `G`: Type of the geometry (a [`Geometry.AbstractGeometry`](@ref)).
- `L`: Type of the label (an `AbstractString`).
"""
struct ConstantFormSpace{manifold_dim, form_rank, PB, S, G, L} <:
       AbstractFormSpace{manifold_dim, form_rank}
    geometry::G
    label::L

    function ConstantFormSpace(
        form_rank::Int,
        geometry::G,
        label::AbstractString,
        ::Type{PB}=FormPullback,
        ::Type{S}=Physical,
    ) where {manifold_dim, PB, S, G <: Geometry.AbstractGeometry{manifold_dim}}
        if (form_rank != 0 && form_rank != manifold_dim)
            throw(
                ArgumentError(
                    "Mantis.Forms.ConstantFormSpace: form_rank = $form_rank with " *
                    "manifold_dim = $manifold_dim requires form_rank to be 0 or " *
                    "manifold_dim.",
                ),
            )
        end
        return new{manifold_dim, form_rank, PB, S, G, typeof(label)}(geometry, label)
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

get_form_space_tree(form::ConstantFormSpace) = (form,)

get_geometry(form::ConstantFormSpace) = form.geometry

function get_fe_space(::ConstantFormSpace)
    return throw(
        ArgumentError("ConstantFormSpace does not have an associated finite element space.")
    )
end

function get_pullback_type(
    form::ConstantFormSpace{manifold_dim, form_rank, PB}
) where {manifold_dim, form_rank, PB}
    return PB
end

function get_pullback(
    form::ConstantFormSpace{manifold_dim, form_rank, PB, S}, ::Type{D}=Canonical
) where {manifold_dim, form_rank, PB, S, D}
    return PB(form, S, D)
end

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

function evaluate(
    form::ConstantFormSpace{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    return evaluate(get_pullback(form), element_id, xi)
end

function evaluate(
    form::FormPullback{manifold_dim, form_rank, expression_rank, S, D, F},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank, expression_rank, S, D, F <: ConstantFormSpace}
    num_evaluation_points = Points.get_num_points(xi)
    evaluations = [ones(Float64, num_evaluation_points, 1)]

    pullback!(evaluations, form, element_id, xi)

    return evaluations, [[1]]
end

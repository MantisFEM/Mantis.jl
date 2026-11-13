############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    ConstantFormSpace{manifold_dim, form_rank, S, T, G, L} <:
    AbstractFormSpace{manifold_dim, form_rank, S}

Constant scalar differential form that evaluates to 1 in domain `S`. By default, `S` is the
[`Physical`](@ref) domain.

This can, for instance, be used as a Lagrange multiplier enforcing a zero-average
constraint on another differential form.

The type of the value can be changed by specifying the type in the constructor. This
defaults to `Float64`.

# Constructors
- `ConstantFormSpace(
        form_rank::Int,
        geometry::G,
        label::AbstractString,
        ::Type{S}=Physical,
        ::Type{T}=Float64,
    ) where {manifold_dim, S, G <: Geometry.AbstractGeometry{manifold_dim}, T}`: Generic
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
- `manifold_dim`, `form_rank`, `S`: See [`AbstractForm`](@ref).
- `T`: Eltype of the returned values of [`evaluate`](@ref). Defaults to `Float64`.
- `G`: Type of the geometry (a [`Geometry.AbstractGeometry`](@ref)).
- `L`: Type of the label (an `AbstractString`).
"""
struct ConstantFormSpace{manifold_dim, form_rank, S, T, G, L} <:
       AbstractFormSpace{manifold_dim, form_rank, S}
    geometry::G
    label::L

    function ConstantFormSpace(
        form_rank::Int,
        geometry::G,
        label::AbstractString,
        ::Type{S}=Physical,
        ::Type{T}=Float64,
    ) where {manifold_dim, S, G <: Geometry.AbstractGeometry{manifold_dim}, T <: Number}
        if (form_rank != 0 && form_rank != manifold_dim)
            throw(
                ArgumentError(
                    "Mantis.Forms.ConstantFormSpace: form_rank = $form_rank with " *
                    "manifold_dim = $manifold_dim requires form_rank to be 0 or " *
                    "manifold_dim.",
                ),
            )
        end
        return new{manifold_dim, form_rank, S, T, G, typeof(label)}(geometry, label)
    end
end

############################################################################################
#                                   Getters and setters                                    #
############################################################################################

function Base.eltype(
    ::Type{ConstantFormSpace{manifold_dim, form_rank, S, T, G, L}}
) where {manifold_dim, form_rank, S, T, G, L}
    return T
end

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

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

function evaluate(
    form::ConstantFormSpace{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    num_evaluation_points = Points.get_num_points(xi)
    return [ones(eltype(form), num_evaluation_points, 1)], [[1]]
end

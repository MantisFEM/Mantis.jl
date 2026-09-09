############################################################################################
#                               Differential form pullbacks                                #
############################################################################################
"""
    FormPullback{manifold_dim, form_rank, expression_rank, S, D, F} <:
    AbstractPullback{manifold_dim, form_rank, expression_rank, S, D}

Standard differential form pullback which depends on the `form_rank` of the form to which
it is applied.

# Constructors
- `FormPullback(
    form::F, ::Type{S}, ::Type{D}
) where {
    manifold_dim,
    form_rank,
    expression_rank,
    F <: AbstractForm{manifold_dim, form_rank, expression_rank},
    S <: AbstractPullbackLocation,
    D <: AbstractPullbackLocation,
}`: General constructor.

# Fields
- `form::F`: The form to which the pullback is applied.

# Type parameters
- `manifold_dim`, `form_rank`, `expression_rank`: See [`AbstractForm`](@ref).
- `S`, `D`: See [`AbstractPullback`](@ref).
"""
struct FormPullback{manifold_dim, form_rank, expression_rank, S, D, F} <:
       AbstractPullback{manifold_dim, form_rank, expression_rank, S, D}
    form::F

    function FormPullback(
        form::F, ::Type{S}, ::Type{D}
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        F <: AbstractForm{manifold_dim, form_rank, expression_rank},
        S <: AbstractPullbackLocation,
        D <: AbstractPullbackLocation,
    }
        return new{manifold_dim, form_rank, expression_rank, S, D, F}(form)
    end
end

function evaluate(
    form::FormPullback{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    evaluations, basis_indices = Forms.evaluate(get_form(form), element_id, xi)

    pullback!(evaluations, form, element_id, xi)

    return evaluations, [basis_indices]
end

# If the source and destination are the same, the pullback does nothing.
function pullback!(
    evaluations::Vector{T},
    form::FormPullback{manifold_dim, form_rank, expression_rank, S, S},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank, expression_rank, S, T}
    return evaluations
end

# The 0-form FormPullback works the same irrespective of the source and destination.
# Covers outputs of FESpace (T = Vector{Vector{Matrix{T}}}), FormSpace (T = Matrix{<:Real})
# and FormField (T = Vector{<:Real}).
function pullback!(
    evaluations::Vector{T},
    form::FormPullback{manifold_dim, 0},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, T}
    return evaluations
end

# Pullbacks of top-forms.
# Parametric -> Canonical
function pullback!(
    evaluations::Vector{T},
    form::FormPullback{manifold_dim, manifold_dim, expression_rank, S, D},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, expression_rank, T, S <: Parametric, D <: Canonical}
    form = get_form(form)
    geometry = get_geometry(form)
    element_lengths = Geometry.get_element_lengths(geometry, element_id)
    element_volume = prod(element_lengths)
    for c in eachindex(evaluations)
        for p in eachindex(evaluations[c])
            evaluations[c][p] *= element_volume
        end
    end

    return evaluations
end
# Physical -> Canonical
function pullback!(
    evaluations::Vector{T},
    form::FormPullback{manifold_dim, manifold_dim, expression_rank, S, D},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, expression_rank, T, S <: Physical, D <: Canonical}
    geometry = get_geometry(form)
    _, sqrt_g = Geometry.metric(geometry, element_id, xi)
    for c in eachindex(evaluations)
        for p in eachindex(sqrt_g)
            LinearAlgebra.lmul!(sqrt_g[p], view(evaluations[c], p, :))
        end
    end

    return evaluations
end

# Pullbacks of one-forms.
# Parametric -> Canonical
function pullback!(
    evaluations::Vector{T},
    form::FormPullback{manifold_dim, 1, expression_rank, S, D},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, expression_rank, T, S <: Parametric, D <: Canonical}
    geometry = get_geometry(form)
    element_lengths = Geometry.get_element_lengths(geometry, element_id)
    for c in eachindex(evaluations)
        element_dimension = element_lengths[c]
        for p in eachindex(evaluations[c])
            evaluations[c][p] *= element_dimension
        end
    end

    return evaluations
end
# Physical -> Canonical
function pullback!(
    evaluations::Vector{T},
    form::Union{
        FormPullback{2, 1, expression_rank, S, D}, FormPullback{3, 1, expression_rank, S, D}
    },
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, expression_rank, T, S <: Physical, D <: Canonical}
    geometry = get_geometry(form)
    J = Geometry.jacobian(geometry, element_id, xi)
    num_eval_points = Points.get_num_points(xi)
    TT = eltype(eltype(evaluations))

    a = zeros(TT, num_eval_points, manifold_dim)
    for j in eachindex(evaluations)
        for i in eachindex(evaluations[j])
            a[i, :] .+= evaluations[j][i] .* J[i][j, :]
        end
    end

    for j in 1:manifold_dim
        evaluations[j] = a[:, j]
    end

    return evaluations
end

# Pullbacks of two-forms.
# Parametric -> Canonical
function pullback!(
    evaluations::Vector{T},
    form::FormPullback{3, 2, expression_rank, S, D},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank, T, S <: Parametric, D <: Canonical}
    geometry = get_geometry(form)
    element_lengths = Geometry.get_element_lengths(geometry, element_id)
    area_12 = element_lengths[1] * element_lengths[2]
    area_23 = element_lengths[2] * element_lengths[3]
    area_13 = element_lengths[1] * element_lengths[3]
    for p in eachindex(evaluations[1])
        evaluations[1][p] *= area_23
    end
    for p in eachindex(evaluations[2])
        evaluations[2][p] *= area_13
    end
    for p in eachindex(evaluations[3])
        evaluations[3][p] *= area_12
    end

    return evaluations
end
# Physical -> Canonical

############################################################################################
#                                 Component-wise pullbacks                                 #
############################################################################################
"""
    ComponentWisePullback{manifold_dim, form_rank, expression_rank, S, D, F} <:
    AbstractPullback{manifold_dim, form_rank, expression_rank, S, D}

Pullback a 'form' by component-wise composition, irrespective of the form rank.

# Constructors
- `ComponentWisePullback(
    form::F, ::Type{S}, ::Type{D}
) where {
    manifold_dim,
    form_rank,
    expression_rank,
    F <: AbstractForm{manifold_dim, form_rank, expression_rank},
    S <: AbstractPullbackLocation,
    D <: AbstractPullbackLocation,
}`: General constructor.

# Fields
- `form::F`: The form to which the pullback is applied.

# Type parameters
- `manifold_dim`, `form_rank`, `expression_rank`: See [`AbstractForm`](@ref).
- `S`, `D`: See [`AbstractPullback`](@ref).
"""
struct ComponentWisePullback{manifold_dim, form_rank, expression_rank, S, D, F} <:
       AbstractPullback{manifold_dim, form_rank, expression_rank, S, D}
    form::F

    function ComponentWisePullback(
        form::F, ::Type{S}, ::Type{D}
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        F <: AbstractForm{manifold_dim, form_rank, expression_rank},
        S <: AbstractPullbackLocation,
        D <: AbstractPullbackLocation,
    }
        return new{manifold_dim, form_rank, expression_rank, S, D, F}(form)
    end
end

# The ComponentWisePullback works the same irrespective of the source and destination.
# Covers outputs of FESpace (T = Vector{Vector{Matrix{T}}}), FormSpace (T = Matrix{<:Real})
# and FormField (T = Vector{<:Real}).
function pullback!(
    evaluations::Vector{T},
    form::ComponentWisePullback{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, T}
    return evaluations
end

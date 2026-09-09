############################################################################################
#                               Differential form pullbacks                                #
############################################################################################
"""
    FormPullback{manifold_dim, form_rank, expression_rank, D, S, F} <:
    AbstractPullback{manifold_dim, form_rank, expression_rank, D}

Standard differential form pullback which depends on the `form_rank` of the form to which
it is applied. Performs the pullback from `S` (dictated by the input `form`) to `D` (given).

Whenever this pullback is applied to an operator with a commutative/distributive property
w.r.t. the pullback, this property is used to propagete the pullback into the expression.
For example, writing `FormPullback(ExteriorDerivative(FormSpace), D)` will return
`ExteriorDerivative(FormPullback(FormSpace, D))`. Additionally, the source locations form a
hierarchy (see [`AbstractPullbackLocation`](@ref)), so consecutively applied pullbacks are
squashed.

# Constructors
- `FormPullback(
    form::F, ::Type{D}
) where {
    manifold_dim,
    form_rank,
    expression_rank,
    S <: AbstractPullbackLocation,
    F <: AbstractForm{manifold_dim, form_rank, expression_rank, S},
    D <: AbstractPullbackLocation,
}`: General constructor.

# Fields
- `form::F`: The form to which the pullback is applied.

# Type parameters
- `manifold_dim`, `form_rank`, `expression_rank`, `D`: See [`AbstractForm`](@ref). Note
    that [`AbstractForm`](@ref) indicates `D` as `S`.
- `S`, `F`: Source location and type of `form`.
"""
struct FormPullback{manifold_dim, form_rank, expression_rank, D, S, F, L} <:
       AbstractPullback{manifold_dim, form_rank, expression_rank, D}
    form::F
    label::L

    function FormPullback(
        form::F, ::Type{D}
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        S <: AbstractPullbackLocation,
        F <: AbstractForm{manifold_dim, form_rank, expression_rank, S},
        D <: AbstractPullbackLocation,
    }
        new_label = convert(
            typeof(get_label(form)), LaTeXStrings.L"\Phi^*" * "(" * get_label(form) * ")"
        )

        return new{manifold_dim, form_rank, expression_rank, D, S, F, typeof(new_label)}(
            form, new_label
        )
    end
end

# Creating the pullback to a destination that is already the source does nothing. This
# method prevents the nesting of more and more of such pullbacks. This way, other
# operators can simply propagate pullbacks without adding noise.
function FormPullback(
    form::F, ::Type{S}
) where {
    manifold_dim,
    form_rank,
    expression_rank,
    S <: AbstractPullbackLocation,
    F <: AbstractForm{manifold_dim, form_rank, expression_rank, S},
}
    return form
end

# Since there is a hierarchy in the locations, we squash the pullbacks together if they
# are applied consecutively.
function FormPullback(
    form::F, ::Type{Canonical}
) where {
    manifold_dim,
    form_rank,
    expression_rank,
    F <: FormPullback{manifold_dim, form_rank, expression_rank, Parametric, Physical},
}
    return FormPullback(get_form(form), Canonical)
end

############################################################################################
#                                     Evaluate methods                                     #
############################################################################################

function evaluate(
    form::FormPullback{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    evaluations, basis_indices = Forms.evaluate(get_form(form), element_id, xi)

    pullback!(evaluations, form, element_id, xi)

    return evaluations, basis_indices
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
    form::FormPullback{manifold_dim, manifold_dim, expression_rank, D, S},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, expression_rank, T, D <: Canonical, S <: Parametric}
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
    form::FormPullback{manifold_dim, manifold_dim, expression_rank, D, S},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, expression_rank, T, D <: Canonical, S <: Physical}
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
    form::FormPullback{manifold_dim, 1, expression_rank, D, S},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, expression_rank, T, D <: Canonical, S <: Parametric}
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
        FormPullback{2, 1, expression_rank, D, S}, FormPullback{3, 1, expression_rank, D, S}
    },
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, expression_rank, T, D <: Canonical, S <: Physical}
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
    form::FormPullback{3, 2, expression_rank, D, S},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank, T, D <: Canonical, S <: Parametric}
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
function exterior_power(J::StaticArrays.SMatrix{3, 3})
    return StaticArrays.SMatrix{3, 3}(
        J[1, 1]*J[2, 2] - J[2, 1]*J[1, 2],
        J[1, 1]*J[3, 2] - J[3, 1]*J[1, 2],
        J[2, 1]*J[3, 2] - J[3, 1]*J[2, 2],
        J[1, 1]*J[2, 3] - J[2, 1]*J[1, 3],
        J[1, 1]*J[3, 3] - J[3, 1]*J[1, 3],
        J[2, 1]*J[3, 3] - J[3, 1]*J[2, 3],
        J[1, 2]*J[2, 3] - J[2, 2]*J[1, 3],
        J[1, 2]*J[3, 3] - J[3, 2]*J[1, 3],
        J[2, 2]*J[3, 3] - J[3, 2]*J[2, 3],
    )
end
function pullback!(
    evaluations::Vector{T},
    form::FormPullback{3, 2, expression_rank, D, S},
    element_id::Int,
    xi::Points.AbstractPoints{3},
) where {expression_rank, T, D <: Canonical, S <: Physical}
    geometry = get_geometry(form)
    J = Geometry.jacobian(geometry, element_id, xi)
    TT = eltype(eltype(evaluations))
    temp = [
        zeros(TT, size(evaluations[1])...),
        zeros(TT, size(evaluations[2])...),
        zeros(TT, size(evaluations[3])...),
    ]

    for p in eachindex(J)
        partial_det_J = _partial_determinants(J[p])

        for b in axes(evaluations[1], 2)
            temp[1][p, b] =
                evaluations[1][p, b] * partial_det_J[1] +
                evaluations[2][p, b] * partial_det_J[2] +
                evaluations[3][p, b] * partial_det_J[3]

            temp[2][p, b] =
                evaluations[1][p, b] * partial_det_J[4] +
                evaluations[2][p, b] * partial_det_J[5] +
                evaluations[3][p, b] * partial_det_J[6]

            temp[3][p, b] =
                evaluations[1][p, b] * partial_det_J[7] +
                evaluations[2][p, b] * partial_det_J[8] +
                evaluations[3][p, b] * partial_det_J[9]
        end
    end

    for j in eachindex(evaluations, temp)
        for i in eachindex(evaluations[j], temp[j])
            evaluations[j][i] = temp[j][i]
        end
    end

    return evaluations
end

############################################################################################
#                                 Component-wise pullbacks                                 #
############################################################################################
"""
    ComponentWisePullback{manifold_dim, form_rank, expression_rank, D, S, F} <:
    AbstractPullback{manifold_dim, form_rank, expression_rank, D}

'Pullback' a form by component-wise composition, irrespective of the form rank. Performs
the pullback from `S` (dictated by the input `form`) to `D` (given).

# Constructors
- `ComponentWisePullback(
    form::F, ::Type{D}
) where {
    manifold_dim,
    form_rank,
    expression_rank,
    S <: AbstractPullbackLocation,
    F <: AbstractForm{manifold_dim, form_rank, expression_rank, S},
    D <: AbstractPullbackLocation,
}`: General constructor.

# Fields
- `form::F`: The form to which the pullback is applied.

# Type parameters
- `manifold_dim`, `form_rank`, `expression_rank`, `D`: See [`AbstractForm`](@ref). Note
    that [`AbstractForm`](@ref) indicates `D` as `S`.
- `S`, `F`: Source location and type of `form`.
- `L <: AbstractString`: The type of the label. Since a "Φ" is added to the label, this
    type may differ from the label type of the underlying form.
"""
struct ComponentWisePullback{manifold_dim, form_rank, expression_rank, D, S, F, L} <:
       AbstractPullback{manifold_dim, form_rank, expression_rank, D}
    form::F
    label::L

    function ComponentWisePullback(
        form::F, ::Type{D}
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        S <: AbstractPullbackLocation,
        F <: AbstractForm{manifold_dim, form_rank, expression_rank, S},
        D <: AbstractPullbackLocation,
    }
        new_label = convert(
            typeof(get_label(form)), LaTeXStrings.L"\Phi" * "(" * get_label(form) * ")"
        )

        return new{manifold_dim, form_rank, expression_rank, D, S, F, typeof(new_label)}(
            form, new_label
        )
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

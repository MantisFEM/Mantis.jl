############################################################################################
#                               Differential form pullbacks                                #
############################################################################################
"""
    FormPullback{manifold_dim, form_rank, expression_rank, D, S, F, L} <:
    AbstractPullback{manifold_dim, form_rank, expression_rank, D}

Standard differential form pullback which depends on the `form_rank` of the form to which
it is applied. Performs the pullback from `S` (dictated by the input `form`) to `D` (given).

Whenever this pullback is applied to an operator with a commutative/distributive property
w.r.t. the pullback, this property is used to propagate the pullback into the expression.
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
- `L`: Type of the label.
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

        # We only have one direction for the mappings, so
        # canonical -> parametric -> physical
        # meaning that the pullback will go from right to left. Pulling back the other way
        # around will fail, because the pullback! methods have not been implemented.
        if (D <: Parametric && S <: Canonical) ||
            (D <: Physical && (S <: Canonical || S <: Parametric))
            throw(
                ArgumentError(
                    LazyString(
                        "You tried to create a Pullback from ",
                        S,
                        " to ",
                        D,
                        ". This can only be done the other way around. Otherwise, use a ",
                        "pushforward.",
                    ),
                ),
            )
        end

        # Now the locations are in the correct order. However, we currently cannot pullback
        # from Physical to Parametric, because we cannot access that jacobian. Until this is
        # fixed, we throw an error.
        if D <: Parametric
            throw(
                ArgumentError(
                    LazyString(
                        "You tried to create a pullback from Physical to Parametric.",
                        " This has not been implemented.",
                    ),
                ),
            )
        end

        return new{manifold_dim, form_rank, expression_rank, D, S, F, typeof(new_label)}(
            form, new_label
        )
    end
end

get_pullback_type(::FormPullback) = AbstractFormPullback

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

# The pullback implementation uses the the cofactor expansion of the jacobian.
function _cofactor_matrix(
    J::StaticArrays.SMatrix{image_dim, manifold_dim}, ::Val{1}
) where {image_dim, manifold_dim}
    return StaticArrays.SMatrix{manifold_dim, image_dim}(StaticArrays.transpose(J))
end
function _cofactor_matrix(J::StaticArrays.SMatrix{2, 2}, ::Val{2})
    return StaticArrays.SMatrix{1, 1}(J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1])
end
function _cofactor_matrix(J::StaticArrays.SMatrix{3, 2}, ::Val{2})
    return StaticArrays.SMatrix{1, 3}(
        J[2, 1] * J[3, 2] - J[2, 2] * J[3, 1],
        J[3, 1] * J[1, 2] - J[3, 2] * J[1, 1],
        J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1],
    )
end
function _cofactor_matrix(J::StaticArrays.SMatrix{3, 3}, ::Val{2})
    # SMatrix is created per column. CofactorMatrix[j,i] is for output component j and
    # input component i.
    return StaticArrays.SMatrix{3, 3}(
        J[2, 2] * J[3, 3] - J[2, 3] * J[3, 2],
        J[2, 3] * J[3, 1] - J[2, 1] * J[3, 3],
        J[2, 1] * J[3, 2] - J[2, 2] * J[3, 1],
        J[3, 2] * J[1, 3] - J[3, 3] * J[1, 2],
        J[3, 3] * J[1, 1] - J[3, 1] * J[1, 3],
        J[3, 1] * J[1, 2] - J[3, 2] * J[1, 1],
        J[1, 2] * J[2, 3] - J[1, 3] * J[2, 2],
        J[1, 3] * J[2, 1] - J[1, 1] * J[2, 3],
        J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1],
    )
end
function _cofactor_matrix(J::StaticArrays.SMatrix{3, 3}, ::Val{3})
    return StaticArrays.SMatrix{1, 1}(StaticArrays.det(J))
end

# The pullback! functions cover outputs of FESpace (T = Vector{Vector{Matrix{T}}}),
# FormSpace (T = Matrix{<:Real}) and FormField (T = Vector{<:Real}) in one go.
# 0-forms don't need to modify the evaluations.
function pullback!(
    evaluations::Vector{T},
    J::Vector{S},
    ::Val{0},
    ::Val{num_source_components},
    ::Val{num_destination_components},
    ::Type{PB},
) where {
    T,
    S <: StaticArrays.SMatrix,
    num_source_components,
    num_destination_components,
    PB <: AbstractFormPullback,
}
    return evaluations
end

# k-forms
function pullback!(
    evaluations::Vector{T},
    J::Vector{S},
    ::Val{form_rank},
    ::Val{num_source_components},
    ::Val{num_destination_components},
    ::Type{PB},
) where {
    T,
    S <: StaticArrays.SMatrix,
    form_rank,
    num_source_components,
    num_destination_components,
    PB <: AbstractFormPullback,
}
    ET = eltype(T)
    for p in axes(evaluations[1], 1) # loop over the points
        # For 1-forms: C = J^T, so C[j, i] = ∂Φᵢ/∂ξⱼ
        # For 2-forms: C = det_{J,I}(J^T)
        C = _cofactor_matrix(J[p], Val(form_rank))

        for b in axes(evaluations[1], 2) # loop of the number of basis
            # Get the input evaluations at point p for basis b, before overwriting.
            evals_pb = ntuple(i -> evaluations[i][p, b], num_source_components)
            # Now we can safely overwrite the evaluations.
            # We first zero the value to ensure that the later sum is correct.
            for i in 1:num_source_components
                evaluations[i][p, b] = zero(ET)
            end
            # Now we can write to the evaluations as if they were the output components.
            for j in 1:num_destination_components
                for i in 1:num_source_components
                    evaluations[j][p, b] += C[j, i] * evals_pb[i]
                end
            end
        end
    end

    # We keep only the output components that were used.
    resize!(evaluations, num_destination_components)

    return evaluations
end
# Physical -> Canonical
function _cofactor_matrix(J::StaticArrays.SMatrix{3, 3})
    return StaticArrays.SMatrix{3, 3}(
        J[2, 2] * J[3, 3] - J[2, 3] * J[3, 2],
        J[3, 2] * J[1, 3] - J[3, 3] * J[1, 2],
        J[1, 2] * J[2, 3] - J[1, 3] * J[2, 2],
        J[2, 3] * J[3, 1] - J[2, 1] * J[3, 3],
        J[3, 3] * J[1, 1] - J[3, 1] * J[1, 3],
        J[1, 3] * J[2, 1] - J[1, 1] * J[2, 3],
        J[2, 1] * J[3, 2] - J[2, 2] * J[3, 1],
        J[3, 1] * J[1, 2] - J[3, 2] * J[1, 1],
        J[1, 1] * J[2, 2] - J[1, 2] * J[2, 1],
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

    for p in axes(evaluations[1], 1)
        C = _cofactor_matrix(J[p])

        for b in axes(evaluations[1], 2)
            e1 = evaluations[1][p, b]
            e2 = evaluations[2][p, b]
            e3 = evaluations[3][p, b]
            for k in 1:3
                evaluations[k][p, b] = e1 * C[1, k] + e2 * C[2, k] + e3 * C[3, k]
            end
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

get_pullback_type(::ComponentWisePullback) = AbstractComponentWisePullback

# Creating the pullback to a destination that is already the source does nothing. This
# method prevents the nesting of more and more of such pullbacks. This way, other
# operators can simply propagate pullbacks without adding noise.
function ComponentWisePullback(
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
function ComponentWisePullback(
    form::F, ::Type{Canonical}
) where {
    manifold_dim,
    form_rank,
    expression_rank,
    F <:
    ComponentWisePullback{manifold_dim, form_rank, expression_rank, Parametric, Physical},
}
    return ComponentWisePullback(get_form(form), Canonical)
end

# The ComponentWisePullback works the same irrespective of the source and destination.
function pullback!(
    evaluations::Vector{T},
    J::Vector{S},
    ::Val{form_rank},
    ::Val{num_source_components},
    ::Val{num_destination_components},
    ::Type{PB},
) where {
    T,
    S <: StaticArrays.SMatrix,
    form_rank,
    num_source_components,
    num_destination_components,
    PB <: AbstractComponentWisePullback,
}
    return evaluations
end

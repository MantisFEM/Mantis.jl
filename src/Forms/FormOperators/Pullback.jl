############################################################################################
#                                    AbstractPullbacks                                     #
############################################################################################
function evaluate(
    form::AbstractPullback{manifold_dim, form_rank, expression_rank},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, form_rank, expression_rank}
    evaluations, basis_indices = Forms.evaluate(get_form(form), element_id, xi)

    source = get_source_location(get_form(form)) # source of the original form
    dest = get_source_location(form) # source of the pullback = destination of original form
    num_source_components = get_num_components(
        Val(get_dim(form, source)), Val(get_form_rank(form))
    )
    num_dest_components = get_num_components(
        Val(get_dim(form, dest)), Val(get_form_rank(form))
    )
    geometry = get_geometry(form, source)
    jacobian = Geometry.jacobian(geometry, element_id, xi)

    return _pullback(
        evaluations,
        basis_indices,
        jacobian,
        Val(form_rank),
        Val(num_source_components),
        Val(num_dest_components),
        Val(expression_rank),
        get_pullback_type(form),
    )
end

# 0-forms are not modified by the form pullback.
function _pullback(
    evaluations,
    indices,
    J,
    ::Val{0},
    ::Val{num_source_components},
    ::Val{num_destination_components},
    ::Val{expression_rank},
    ::Type{PB},
) where {
    num_source_components,
    num_destination_components,
    expression_rank,
    PB <: AbstractFormPullback,
}
    return evaluations, indices
end

# Each component of the pulled-back k-form is a point-wise linear combination of the source
# components (weights given by the cofactor matrix). Since the source components can have
# different basis indices, each destination component lives on the union of the indices of
# the source components that contribute to it.
function _pullback(
    evaluations,
    indices,
    J,
    ::Val{form_rank},
    ::Val{num_source_components},
    ::Val{num_destination_components},
    ::Val{expression_rank},
    ::Type{PB},
) where {
    form_rank,
    num_source_components,
    num_destination_components,
    expression_rank,
    PB <: AbstractFormPullback,
}
    # Only combine the source components with a non-zero contribution, so that, e.g.,
    # diagonal jacobians (Parametric -> Canonical) keep the indices of each component.
    uses = zeros(Bool, num_destination_components, num_source_components)
    for p in eachindex(J)
        C = _cofactor_matrix(J[p], Val(form_rank))
        for j in 1:num_destination_components, i in 1:num_source_components
            uses[j, i] |= !iszero(C[j, i])
        end
    end

    T = eltype(eltype(evaluations))
    num_points = length(J)
    pb_evals = Vector{Array{T, 1 + expression_rank}}(undef, num_destination_components)
    pb_indices = Vector{Vector{Vector{Int}}}(undef, num_destination_components)
    column_maps = Vector{Vector{NTuple{expression_rank, Vector{Int}}}}(
        undef, num_destination_components
    )
    for j in 1:num_destination_components
        components = [i for i in 1:num_source_components if uses[j, i]]
        combined_indices, column_maps[j] = _combine_indices(
            indices, components, Val(expression_rank)
        )
        pb_evals[j] = _allocate_indices(num_points, combined_indices; T=T)
        pb_indices[j] = _tuple_to_vec_indices(combined_indices)
    end

    for p in eachindex(J)
        C = _cofactor_matrix(J[p], Val(form_rank))
        for j in 1:num_destination_components, i in 1:num_source_components
            uses[j, i] || continue
            column_map = column_maps[j][i]
            local_basis = map(e -> 1:length(e), column_map)
            view(pb_evals[j], p, column_map...) .+=
                C[j, i] .* view(evaluations[i], p, local_basis...)
        end
    end

    return pb_evals, pb_indices
end

############################################################################################
#                               Differential form pullbacks                                #
############################################################################################
"""
    FormPullback{manifold_dim, form_rank, expression_rank, D, S, F, L} <:
    AbstractPullback{manifold_dim, form_rank, expression_rank, D}

Standard differential form pullback which depends on the `form_rank` of the form to which it
is applied. Performs the pullback from the source `S` (dictated by the input `form`) to the
destination `D` (given).

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
        # around will fail, because the `_pullback` methods have not been implemented.
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

        # TODO: Now the locations are in the correct order. However, we currently cannot
        # pullback from Physical to Parametric, because we cannot access that jacobian.
        # Until this is fixed, we throw an error.
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

# Trait type to allow the `_pullback`-methods to specialise on the type of the pullback.
# This will enable the specialisation for this `_pullback`-methods, and it will prevent
# overspecialisation which would happen if the `_pullback`-methods accepted a full pullback
# type.
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

function _pullback(
    evaluations,
    indices,
    J,
    ::Val{form_rank},
    ::Val{num_source_components},
    ::Val{num_destination_components},
    ::Val{expression_rank},
    ::Type{PB},
) where {
    form_rank,
    num_source_components,
    num_destination_components,
    expression_rank,
    PB <: AbstractComponentWisePullback,
}
    return evaluations, indices
end

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

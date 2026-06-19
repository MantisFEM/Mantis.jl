"""
    L2_norm(u, dΩ)

Computes the ``L^2`` norm ``\\sqrt{\\int u \\wedge \\star u}`` of a form `u` using the global
quadrature rule `dΩ`.

To compute an error, pass the difference of two forms, e.g. `L2_norm(uₕ - uₑ, dΩ)`, where
`uₕ` is the computed solution and `uₑ` an [`AnalyticalFormField`](@ref Mantis.Forms.AnalyticalFormField)
holding the exact solution. For accurate results, `dΩ` should over-integrate relative to the
rule used for assembly (see the [Quadrature](@ref) module).

# Arguments
- `u`: The form whose ``L^2`` norm is computed.
- `dΩ`: The global quadrature rule used to evaluate the integral.

# Returns
- `::Float64`: The ``L^2`` norm of `u`.
"""
function L2_norm(u, dΩ)
    norm = 0.0
    inner_prod = ∫(u ∧ ★(u), dΩ)
    for el_id in 1:Forms.get_num_elements(u)
        norm += Forms.evaluate(inner_prod, el_id)[1][1]
    end

    return sqrt(norm)
end

function _compute_square_error_per_element(
    computed_sol::TF1, exact_sol::TF2, quad_rule::Q, norm="L2"
) where {
    manifold_dim,
    form_rank,
    expression_rank_1,
    expression_rank_2,
    TF1 <: Forms.AbstractForm{manifold_dim, form_rank, expression_rank_1},
    TF2 <: Forms.AbstractForm{manifold_dim, form_rank, expression_rank_2},
    Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
}
    num_elements = Quadrature.get_num_base_elements(quad_rule)
    result = Vector{Float64}(undef, num_elements)
    difference = computed_sol - exact_sol
    if norm == "L2"
        integral = ∫((difference) ∧ ★(difference), quad_rule)
    end

    for elem_id in 1:1:num_elements
        if norm == "L2"
            result[elem_id] = Forms.evaluate(integral, elem_id)[1][1]
        elseif norm == "H1"
            throw(ArgumentError("Computing the H1 norm still needs to be updated."))
            # d_difference = Forms.ExteriorDerivative(difference)
            # result[elem_id] = sum(
            #     Forms.evaluate_inner_product(
            #         d_difference, d_difference, elem_id, quad_rule
            #     )[3],
            # )
        elseif norm == "Linf"
            result[elem_id] = maximum(
                abs.(
                    Forms.evaluate(difference, elem_id, Quadrature.get_nodes(quad_rule))[1][1]
                ),
            )
        else
            throw(
                ArgumentError(
                    "Unknown norm '$norm'. Only 'L2', 'Linf', and 'H1' are accepted inputs."
                ),
            )
        end
    end

    return result
end

"""
    compute_error_per_element(computed_sol, exact_sol, quad_rule, norm="L2")

Returns a vector with the error between `computed_sol` and `exact_sol` on each evaluation
element of `quad_rule`. This is the per-element quantity used to drive adaptive refinement
(for example as the local error indicator fed to Dörfler marking; see the
[Adaptive refinement](@ref) example).

# Arguments
- `computed_sol`: The computed (discrete) solution form.
- `exact_sol`: The reference solution form (e.g. an
    [`AnalyticalFormField`](@ref Mantis.Forms.AnalyticalFormField)).
- `quad_rule`: The global quadrature rule used to evaluate the error.
- `norm="L2"`: The norm to use. Accepts `"L2"` and `"Linf"`; `"H1"` is not yet implemented.

# Returns
- `::Vector{Float64}`: The error contributed by each evaluation element.
"""
function compute_error_per_element(
    computed_sol::TF1, exact_sol::TF2, quad_rule::Q, norm="L2"
) where {
    manifold_dim,
    form_rank,
    expression_rank_1,
    expression_rank_2,
    TF1 <: Forms.AbstractForm{manifold_dim, form_rank, expression_rank_1},
    TF2 <: Forms.AbstractForm{manifold_dim, form_rank, expression_rank_2},
    Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
}
    if !(Forms.get_geometry(computed_sol) == Forms.get_geometry(exact_sol))
        @warn(
            "Trying to compute the error between two forms " *
                "whose geometries don't point to the same object in memory. " *
                "The geometries might not be compatible."
        )
    end

    partial_result = _compute_square_error_per_element(
        computed_sol, exact_sol, quad_rule, norm
    )
    if norm == "Linf"
        return partial_result
    elseif norm == "L2" || norm == "H1"
        return sqrt.(partial_result)
    else
        throw(
            ArgumentError(
                "Unknown norm '$norm'. Only 'L2', 'Linf', and 'H1' are accepted inputs."
            ),
        )
    end
end

"""
    compute_error_total(computed_sol, exact_sol, quad_rule, norm="L2")

Returns the total (domain-wide) error between `computed_sol` and `exact_sol` as a single
scalar. For the `"L2"` norm this equals [`L2_norm`](@ref)`(computed_sol - exact_sol, quad_rule)`;
`"Linf"` returns the maximum error over all evaluation points.

# Arguments
- `computed_sol`: The computed (discrete) solution form.
- `exact_sol`: The reference solution form (e.g. an
    [`AnalyticalFormField`](@ref Mantis.Forms.AnalyticalFormField)).
- `quad_rule`: The global quadrature rule used to evaluate the error.
- `norm="L2"`: The norm to use. Accepts `"L2"` and `"Linf"`; `"H1"` is not yet implemented.

# Returns
- `::Float64`: The total error over the whole domain.
"""
function compute_error_total(
    computed_sol::TF1, exact_sol::TF2, quad_rule::Q, norm="L2"
) where {
    manifold_dim,
    form_rank,
    expression_rank_1,
    expression_rank_2,
    TF1 <: Forms.AbstractForm{manifold_dim, form_rank, expression_rank_1},
    TF2 <: Forms.AbstractForm{manifold_dim, form_rank, expression_rank_2},
    Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
}
    if !(Forms.get_geometry(computed_sol) == Forms.get_geometry(exact_sol))
        throw(
            ArgumentError(
                LazyString(
                    "The two forms must have the same geometry. Instead they have ",
                    Forms.get_geometry(computed_sol),
                    " for the first form, and ",
                    Forms.get_geometry(exact_sol),
                    " for the second form.",
                ),
            ),
        )
    end

    partial_result = _compute_square_error_per_element(
        computed_sol, exact_sol, quad_rule, norm
    )
    if norm == "Linf"
        return maximum(partial_result)
    elseif norm == "L2" || norm == "H1"
        return sqrt(sum(partial_result))
    else
        throw(
            ArgumentError(
                "Unknown norm '$norm'. Only 'L2', 'Linf', and 'H1' are accepted inputs."
            ),
        )
    end
end

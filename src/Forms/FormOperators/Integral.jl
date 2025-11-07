############################################################################################
#                                        Structure                                         #
############################################################################################
"""
    Integral{manifold_dim, F, Q} <: AbstractRealValuedOperator{manifold_dim}

Structure representing the integral of a form over a manifold.

# Fields
- `form::F`: The form expression to be integrated.
- `quad_rule::Quadrature.AbstractGlobalQuadratureRule{manifold_dim}`: The quadrature rule
    used for the integral.

# Type Parameters
- `manifold_dim::Int`: The dimension of the manifold.
- `F`: The type of the form expression.
# Inner Constructors
- `Integral(form::AbstractFormExpression{manifold_dim})`: General constructor.
# Outer Constructors
- `∫`: Symbolic wrapper for the integral operator.
"""
struct Integral{manifold_dim, F, Q} <: AbstractRealValuedOperator{manifold_dim}
    form::F
    quad_rule::Q
    function Integral(
        form::F, quad_rule::Q
    ) where {
        manifold_dim,
        F <: AbstractFormExpression{manifold_dim, manifold_dim},
        Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
    }
        geom = get_geometry(form)
        if Geometry.get_num_elements(geom) != Quadrature.get_num_base_elements(quad_rule)
            throw(
                ArgumentError(
                    """The number of elements in the geometry and quadrature rule must \
                    match. The geometry has $(Geometry.get_num_elements(geom)) elements \
                    and the quadrature rule has \
                    $(Quadrature.get_num_base_elements(quad_rule)) elements."""
                ),
            )
        end

        return new{manifold_dim, F, Q}(form, quad_rule)
    end
end

"""
    ∫(form::AbstractFormExpression)

Symbolic wrapper for the integral operator. The unicode character command is `\\int`.

# Arguments
- `form::AbstractFormExpression`: The form to be integrated.
- `quad_rule::Quadrature.AbstractGlobalQuadratureRule`: The quadrature rule to be used.

# Returns
- `Integral`: The integral operator.
"""
function ∫(form::AbstractFormExpression, quad_rule::Quadrature.AbstractGlobalQuadratureRule)
    return Integral(form, quad_rule)
end

############################################################################################
#                                         Getters                                          #
############################################################################################

"""
    get_form(integral::Integral)

Returns the form associated with the integral operator.

# Arguments
- `integral::Integral`: The integral operator.

# Returns
- `<: AbstractFormExpression`: The form associated with the integral operator.
"""
get_form(integral::Integral) = integral.form

"""
    get_form_space_tree(integral::Integral)

Returns the spaces of forms of `expression_rank` > 0 appearing in the tree of the integrand of integral, e.g., for
`∫c*((α ∧ β) + γ)`, it returns the spaces of `α`, `β`, and `γ`, if all have expression_rank > 1. 
If `α` has expression_rank = 0, it returns only the spaces of `β` and `γ`.

# Arguments
- `integral::Integral`: The Integral structure.

# Returns
- `Tuple(<:AbstractFormExpression)`: The list of form spaces present in the tree of the integrand of integral.
"""
function get_form_space_tree(integral::Integral)
    return get_form_space_tree(get_form(integral))
end

"""
    get_expression_rank(integral::Integral)

Returns the rank of the expression associated with the integral operator.

# Arguments
- `integral::Integral`: The integral operator.

# Returns
- `::Int`: The rank of the expression associated with the integral operator.
"""
get_expression_rank(integral::Integral) = get_expression_rank(get_form(integral))

"""
    get_quadrature_rule(integral::Integral)

Returns the quadrature rule associated with the integral operator.

# Arguments
- `integral::Integral`: The integral operator.

# Returns
- `<:Quadrature.AbstractGlobalQuadratureRule`: Returns the quadrature rule associated with
    the integral operator.
"""
get_quadrature_rule(integral::Integral) = integral.quad_rule

"""
    get_num_elements(integral::Integral)

Returns the number of elements in the geometry associated with the integral operator.

# Arguments
- `integral::Integral`: The integral operator.

# Returns
- `::Int`: The number of elements associated with the integral operator.
"""
function get_num_elements(integral::Integral)
    return Quadrature.get_num_base_elements(get_quadrature_rule(integral))
end

"""
    get_num_evaluation_elements(integral::Integral)

Returns the number of evaluation elements in the quadrature rule associated with the
integral operator.

# Arguments
- `integral::Integral`: The integral operator.

# Returns
- `::Int`: The number of evaluation elements associated with the integral operator.
"""
function get_num_evaluation_elements(integral::Integral)
    return Quadrature.get_num_evaluation_elements(get_quadrature_rule(integral))
end

"""
    get_estimated_nnz_per_elem(integral::Integral)

Returns the estimated number of non-zero entries per element for the integral operator.

# Arguments
- `integral::Integral`: The integral operator.

# Returns
- `::Int`: The estimated number of non-zero entries per element associated with the integral
    operator.
"""
function get_estimated_nnz_per_elem(integral::Integral)
    return get_estimated_nnz_per_elem(get_form(integral))
end
############################################################################################
#                                        Evaluate                                          #
############################################################################################

"""
    evaluate(
        integral::Integral{manifold_dim, F, Q},
        global_element_id::Int,
    ) where {
        manifold_dim,
        form_rank,
        expression_rank,
        F <: AbstractFormExpression{manifold_dim, form_rank, expression_rank},
    }

Evaluates the integral of a form over a given global element using a specified quadrature
rule.

# Arguments
- `integral::Integral{manifold_dim, F, Q}`: The integral operator to evaluate.
- `global_element_id::Int`: The global element over which to evaluate the integral.

# Returns
- `integral_eval::Vector{Float64}`: The evaluated integral.
- `integral_indices::Vector{Vector{Int}}`: The indices of the evaluated integral. The length
    of the outer vector depends on the `expression_rank` of the form expression.
"""
function evaluate(
    integral::Integral{manifold_dim, F, Q}, global_element_id::Int
) where {
    manifold_dim,
    form_rank,
    expression_rank,
    F <: AbstractFormExpression{manifold_dim, form_rank, expression_rank},
    Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
}
    quad_rule = get_quadrature_rule(integral)
    quadrature_elements = Quadrature.get_element_idxs(quad_rule, global_element_id)
    if isempty(quadrature_elements)
        array_dim = max(expression_rank, 1)
        return Array{Float64, array_dim}(undef, ntuple(_ -> 0, array_dim)), Vector{Int}[]
    end

    form = get_form(integral)
    element_quad_rule = Quadrature.get_element_quadrature_rule(
        quad_rule, quadrature_elements[1]
    )
    weights = Quadrature.get_weights(element_quad_rule)

    form_eval, basis_indices = evaluate(
        form, global_element_id, Quadrature.get_nodes(element_quad_rule)
    )
    num_basis_indices = ntuple(i -> length(basis_indices[i]), max(expression_rank, 1))
    integral_vals = zeros(num_basis_indices)
    integral_vals = add_integral_contribution!(
        integral_vals, num_basis_indices, form_eval, weights
    )
    for quad_element_id in quadrature_elements[2:end]
        element_quad_rule = Quadrature.get_element_quadrature_rule(
            quad_rule, quad_element_id
        )
        weights .= Quadrature.get_weights(element_quad_rule)
        form_eval .= evaluate(
            form, global_element_id, Quadrature.get_nodes(element_quad_rule)
        )[1]
        integral_vals = add_integral_contribution!(
            integral_vals, num_basis_indices, form_eval, weights
        )
    end

    return integral_vals, basis_indices
end

function add_integral_contribution!(integral_vals, num_basis_indices, form_eval, weights)
    for ord_id in CartesianIndices(num_basis_indices)
        for node_id in axes(form_eval[1], 1)
            integral_vals[ord_id] += weights[node_id] * form_eval[1][node_id, ord_id]
        end
    end

    return integral_vals
end

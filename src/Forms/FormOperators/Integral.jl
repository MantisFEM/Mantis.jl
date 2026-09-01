############################################################################################
#                                        Structure                                         #
############################################################################################
"""
    Integral{manifold_dim, F, Q} <: AbstractRealValuedOperator{manifold_dim}

Integral of a form over a manifold.

# Constructors
- `Integral(
        form::F, quad_rule::Q
    ) where {
        manifold_dim,
        F <: AbstractForm{manifold_dim, manifold_dim},
        Q <: Quadrature.AbstractGlobalQuadratureRule{manifold_dim},
    }`: General constructor.

# Examples
Basic syntax:
```jldoctest
julia> using Mantis

julia> B = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (2, 2), (2, 2), (1, 1));

julia> Λ²ₕ = Forms.FormSpace(2, B, "2-form");  # 2-form space with B as basis.

julia> canonical_qrule = Quadrature.tensor_product_rule((3, 3), Quadrature.gauss_legendre);

julia> dΩ = Quadrature.StandardQuadrature(canonical_qrule, 4);

julia> integral = ∫(Λ²ₕ, dΩ);

julia> isa(integral, Forms.Integral{2})
true

julia> isa(Forms.get_form(integral), Forms.FormSpace{2, 2})
true

```

The `Integral` is more commonly used to represent inner products in combination with the
[`Wedge`](@ref) and [`Hodge`](@ref) operators:
```jldoctest
julia> using Mantis

julia> B = FunctionSpaces.create_bspline_space((0.0, 0.0), (1.0, 1.0), (2, 2), (2, 2), (1, 1));

julia> Λ⁰ₕ = Forms.FormSpace(0, B, "0-form");  # 0-form space with B as basis.

julia> canonical_qrule = Quadrature.tensor_product_rule((3, 3), Quadrature.gauss_legendre);

julia> dΩ = Quadrature.StandardQuadrature(canonical_qrule, 4);

julia> integral = ∫(Λ⁰ₕ ∧ ★(Λ⁰ₕ), dΩ);

julia> isa(integral, Forms.Integral{2})
true

julia> isa(Forms.get_form(integral), Forms.Wedge{2, 2})
true

```

# Fields
- `form::F`: The form expression to be integrated.
- `quad_rule::Quadrature.AbstractGlobalQuadratureRule{manifold_dim}`: The quadrature rule
    used for the integral.

# Type Parameters
- `manifold_dim::Int`: The dimension of the manifold.
- `F`: The type of the form expression.
- `Q`: The type of the quadrature expression.
"""
struct Integral{manifold_dim, F, Q} <: AbstractRealValuedOperator{manifold_dim}
    form::F
    quad_rule::Q
    function Integral(
        form::F, quad_rule::Q
    ) where {
        manifold_dim,
        F <: AbstractForm{manifold_dim, manifold_dim},
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
    ∫

Symbolic wrapper for the integral operator. The unicode character command is `\\int`. See
[`Integral`](@ref) for the details.
"""
const ∫ = Integral

############################################################################################
#                                         Getters                                          #
############################################################################################

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
        F <: AbstractForm{manifold_dim, form_rank, expression_rank},
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
    F <: AbstractForm{manifold_dim, form_rank, expression_rank},
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

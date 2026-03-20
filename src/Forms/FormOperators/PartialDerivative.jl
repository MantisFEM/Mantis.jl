############################################################################################
#                                        Structure                                         #
############################################################################################

"""
	PartialDerivative{manifold_dim, form_rank, expression_rank, F, L} <:
	AbstractForm{manifold_dim, form_rank, expression_rank}

`PartialDerivative` wraps an [`AbstractForm`](@ref) and applies dimension-wise partial
derivatives, as specified by `orders`.

# Fields
- `form::F`: The form to differentiate.
- `orders::NTuple{manifold_dim, Int}`: The order of the partial derivative in each
	dimension.
- `label::L`: This is a concatenation of `"∂"` with the label of `form`.
"""
struct PartialDerivative{manifold_dim, form_rank, expression_rank, F, L} <:
       AbstractForm{manifold_dim, form_rank, expression_rank}
    form::F
    orders::NTuple{manifold_dim, Int}
    label::L

    function PartialDerivative(
        form::F, orders::NTuple{manifold_dim, Int}
    ) where {
        manifold_dim, expression_rank, F <: AbstractForm{manifold_dim, 0, expression_rank}
    }
        if any(i -> i < 0, orders)
            throw(
                ArgumentError(
                    "Only positive derivative orders are valid. " *
                    "Orders $(orders) were given.",
                ),
            )
        end

        if !(typeof(get_geometry(form)) <: Geometry.CartesianGeometry)
            throw(
                ArgumentError(
                    "Only 'CartesianGeometry' is allowed. " *
                    "A geometry of type '$(nameof(typeof(get_geometry(form))))' was given.",
                ),
            )
        end

        old_label = get_label(form)
        new_label = convert(typeof(old_label), "∂(" * old_label * ")")

        return new{manifold_dim, 0, expression_rank, F, typeof(new_label)}(
            form, orders, new_label
        )
    end
end

"""
    ∂

Symbolic wrapper for the partial derivative operator. See [`PartialDerivative`](@ref) for
the details.
"""
const ∂ = PartialDerivative

############################################################################################
#                                         Getters                                          #
############################################################################################

"""
	get_orders(partial_der::PartialDerivative)

Returns the dimension-wise derivative orders of `partial_der`.
"""
get_orders(partial_der::PartialDerivative) = partial_der.orders

"""
	get_form(partial_der::PartialDerivative)

Returns the form to which the partial derivative is applied.
"""
get_form(partial_der::PartialDerivative) = partial_der.form

############################################################################################
#                                        Evaluation                                        #
############################################################################################

function evaluate(
    partial_der::PartialDerivative{manifold_dim, 0, 1},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    form = get_form(partial_der)
    partial_orders = get_orders(partial_der)
    num_derivatives = sum(partial_orders)
    form_eval, form_indices = _evaluate_form_in_canonical_coordinates(
        form, element_id, xi, num_derivatives
    )
    der_idx = FunctionSpaces.get_derivative_idx([partial_orders...])
    partial_der_eval = form_eval[num_derivatives + 1][der_idx]
    partial_der_eval = _add_geometric_scaling!(
        partial_der_eval, form, element_id, xi, partial_orders
    )

    return partial_der_eval, form_indices
end

function evaluate(
    partial_der::PartialDerivative{manifold_dim, 0, 0},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    form = get_form(partial_der)
    partial_orders = get_orders(partial_der)
    num_derivatives = sum(partial_orders)
    form_eval, form_indices = _evaluate_form_in_canonical_coordinates(
        get_form_space(form), element_id, xi, num_derivatives
    )
    der_idx = FunctionSpaces.get_derivative_idx([partial_orders...])
    partial_der_eval = [
        form_eval[num_derivatives + 1][der_idx][1] *
        view(get_coefficients(form), form_indices[1]),
    ]
    partial_der_eval = _add_geometric_scaling!(
        partial_der_eval, form, element_id, xi, partial_orders
    )

    return partial_der_eval, [[1]]
end

"""
	_add_geometric_scaling!(
	    partial_der_eval, form::AbstractForm{manifold_dim}, element_id, xi, partial_orders
	)

Scales `partial_der_eval` from canonical to physical coordinates by applying the coordinate
transformation
	∂/∂xᵢ = Σⱼ (J⁻¹)ᵢⱼ ∂/∂ξⱼ.
[`PartialDerivative`](@ref) only supports `Geometry.CartesianGeometry`, so the
transformation simplifies to
	∂/∂xᵢ = 1/Jᵢᵢ ∂/∂ξᵢ.
"""
function _add_geometric_scaling!(
    partial_der_eval, form::AbstractForm{manifold_dim}, element_id, xi, partial_orders
) where {manifold_dim}
    jacobian = Geometry.jacobian(get_geometry(form), element_id, xi)
    for point in axes(partial_der_eval[1], 1), dim in 1:manifold_dim
        scaling = jacobian[point][dim, dim]^partial_orders[dim]
        for component in eachindex(partial_der_eval)
            partial_der_eval[component][point, :] ./= scaling
        end
    end

    return partial_der_eval
end

############################################################################################
#                                        Structure                                         #
############################################################################################

struct PartialOrder{manifold_dim, total_order}
    orders::NTuple{manifold_dim, Int}
    function PartialOrder(
        orders::NTuple{manifold_dim, Int}, ::Val{total_order}
    ) where {manifold_dim, total_order}
        if any(i -> i < 0, orders)
            throw(
                ArgumentError(
                    LazyString(
                        "Only positive derivative orders are valid. Orders ",
                        orders,
                        " were given.",
                    ),
                ),
            )
        end

        if sum(orders) != total_order
            throw(
                ArgumentError(
                    LazyString(
                        "The given derivative orders and total order do not match. Got ",
                        sum(orders),
                        " from the former, and ",
                        total_order,
                        " from the latter.",
                    ),
                ),
            )
        end

        return new{manifold_dim, total_order}(orders)
    end
end

PartialOrder(orders) = PartialOrder(orders, Val(sum(orders)))

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
struct PartialDerivative{manifold_dim, form_rank, expression_rank, G, F, PO, L} <:
       AbstractForm{manifold_dim, form_rank, expression_rank}
    form::F
    orders::PO
    label::L

    function PartialDerivative(
        form::F, orders::PO
    ) where {
        manifold_dim,
        expression_rank,
        total_order,
        F <: AbstractForm{manifold_dim, 0, expression_rank},
        PO <: PartialOrder{manifold_dim, total_order},
    }
        geo_type = typeof(get_geometry(form))
        # if !(geo_type <: Geometry.CartesianGeometry) && total_order > 1
        #     throw(
        #         ArgumentError(
        #             LazyString(
        #                 "Only 'CartesianGeometry' is supported for total derivative order ",
        #                 total_order,
        #                 "Got geometry of type ",
        #                 nameof(typeof(get_geometry(form))),
        #             ),
        #         ),
        #     )
        # end

        old_label = get_label(form)
        new_label = convert(typeof(old_label), "∂(" * old_label * ")")

        return new{manifold_dim, 0, expression_rank, geo_type, F, PO, typeof(new_label)}(
            form, orders, new_label
        )
    end
end

PartialDerivative(form, orders) = PartialDerivative(form, PartialOrder(orders))

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
get_orders(partial_der::PartialDerivative) = partial_der.orders.orders

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
    der_idx = FunctionSpaces.get_derivative_idx(partial_orders)
    partial_der_eval = form_eval[num_derivatives + 1][der_idx]
    partial_der_eval = _add_geometric_scaling!(
        partial_der_eval, partial_der, element_id, xi, partial_orders
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
        get_form(form), element_id, xi, num_derivatives
    )
    der_idx = FunctionSpaces.get_derivative_idx(partial_orders)
    partial_der_eval = [
        form_eval[num_derivatives + 1][der_idx][1] *
        view(get_coefficients(form), form_indices[1]),
    ]
    partial_der_eval = _add_geometric_scaling!(
        partial_der_eval, partial_der, element_id, xi, partial_orders
    )

    return partial_der_eval, [[1]]
end

"""
	_add_geometric_scaling!(
		partial_der_eval, partial_der::PartialDerivative{manifold_dim}, element_id, xi, _
	) where {manifold_dim}

Scales `partial_der_eval` from canonical to physical coordinates by applying the coordinate
transformation
	∂/∂xᵢ = Σⱼ (g⁻¹Jᵀ)ⱼᵢ ∂/∂ξⱼ.
"""
function _add_geometric_scaling!(
    partial_der_eval, partial_der::PartialDerivative{manifold_dim}, element_id, xi, _
) where {manifold_dim}
    throw(ArgumentError("Incorrect implementation."))
    geometry = get_geometry(partial_der)
    image_dim = Geometry.get_image_dim(geometry)
    J_T = transpose.(Geometry.jacobian(geometry, element_id, xi))
    inv_g, _, _ = Geometry.inv_metric(geometry, element_id, xi)
    scaling = inv_g .* J_T
    for point in axes(partial_der_eval[1], 1)
        point_scaling = scaling[point]
        for dim in 1:image_dim
            dim_scaling = sum(view(point_scaling, :, dim))
            for component in eachindex(partial_der_eval)
                partial_der_eval[component][point, :] ./= dim_scaling
            end
        end
    end

    return partial_der_eval
end

"""
	_add_geometric_scaling!(
		partial_der_eval,
		partial_der::PartialDerivative{manifold_dim, 0, expression_rank, G},
		element_id,
		xi,
		partial_orders,
	) where {manifold_dim, expression_rank, G <: Geometry.CartesianGeometry}

Scales `partial_der_eval` from canonical to physical coordinates by applying the coordinate
transformation
	∂/∂xᵢ = Σⱼ (g⁻¹Jᵀ)ⱼᵢ ∂/∂ξⱼ.
Because the geometry is a `Geometry.CartesianGeometry`, the transformation simplifies to
	∂ⁿ/∂xᵢⁿ = (1/Jᵢᵢ)ⁿ ∂/∂ξᵢ.
"""
function _add_geometric_scaling!(
    partial_der_eval,
    partial_der::PartialDerivative{manifold_dim, 0, expression_rank, G},
    element_id,
    xi,
    partial_orders,
) where {manifold_dim, expression_rank, G <: Geometry.CartesianGeometry}
    J = Geometry.jacobian(get_geometry(partial_der), element_id, xi)
    for point in axes(partial_der_eval[1], 1), dim in 1:manifold_dim
        scaling = J[point][dim, dim]^partial_orders[dim]
        for component in eachindex(partial_der_eval)
            partial_der_eval[component][point, :] ./= scaling
        end
    end

    return partial_der_eval
end

function _add_geometric_scaling!(
    partial_der_eval,
    partial_der::PartialDerivative{manifold_dim, 0, expression_rank, HG},
    element_id,
    xi,
    partial_orders,
) where {
    manifold_dim,
    expression_rank,
    image_dim,
    num_patches,
    num_levels,
    TG <: NTuple{num_levels, <:Geometry.CartesianGeometry},
    HG <: Geometry.HierarchicalGeometry{manifold_dim, image_dim, num_patches, TG},
}
    J = Geometry.jacobian(get_geometry(partial_der), element_id, xi)
    for point in axes(partial_der_eval[1], 1), dim in 1:manifold_dim
        scaling = J[point][dim, dim]^partial_orders[dim]
        for component in eachindex(partial_der_eval)
            partial_der_eval[component][point, :] ./= scaling
        end
    end

    return partial_der_eval
end

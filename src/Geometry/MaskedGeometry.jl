############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    MaskedGeometry{manifold_dim, G, M}

A masked geometry obtained by a pseudo-refinement of another geometry. The
geometry `base_geometry` is the original geometry and it is composed with the evaluation
mask `eval_mask`.

# Fields

  - `base_geometry::G`: The base geometry.
  - `eval_mask::AbstractEvaluationMask`: The evaluation mask.
"""
struct MaskedGeometry{manifold_dim, image_dim, num_patches, G, M} <:
       AbstractGeometry{manifold_dim, image_dim, num_patches}
    base_geometry::G
    eval_mask::M

    function MaskedGeometry(
        base_geometry::G, eval_mask::M
    ) where {
        manifold_dim,
        image_dim,
        num_patches,
        G <: AbstractGeometry{manifold_dim, image_dim, num_patches},
        M <: AbstractEvaluationMask{manifold_dim},
    }
        if get_num_elements(base_geometry) != get_num_elements_base(eval_mask)
            throw(
                ArgumentError(
                    "Number of elements in geometry and evaluation mask do not match."
                ),
            )
        end

        return new{manifold_dim, image_dim, num_patches, G, M}(base_geometry, eval_mask)
    end
end

############################################################################################
#                                         Getters                                          #
############################################################################################

get_base_geometry(geometry::MaskedGeometry) = geometry.base_geometry
get_evaluation_mask(geometry::MaskedGeometry) = geometry.eval_mask

function get_num_elements(geometry::MaskedGeometry)
    return get_num_elements(get_evaluation_mask(geometry))
end

function get_element_lengths(geometry::MaskedGeometry, element_id::Int)
    element_id_base = get_base_element(get_evaluation_mask(geometry), element_id)
    lengths_base = get_element_lengths(get_base_geometry(geometry), element_id_base)
    lengths = lengths_base .* get_scaling(get_evaluation_mask(geometry), element_id)

    return lengths
end

function get_element_measure(geometry::MaskedGeometry, element_id::Int)
    element_id_base = get_base_element(get_evaluation_mask(geometry), element_id)
    volume_base = get_element_measure(get_base_geometry(geometry), element_id_base)
    volume_base *= get_scaling(get_evaluation_mask(geometry), element_id)

    return volume_base
end

function get_element_vertices(
    geometry::MaskedGeometry{manifold_dim}, element_id::Int
) where {manifold_dim}
    eval_mask = get_evaluation_mask(geometry)
    element_id_base = get_base_element(eval_mask, element_id)
    element_vertices_base = get_element_vertices(
        get_base_geometry(geometry), element_id_base
    )

    return transform_element_vertices(eval_mask, element_vertices_base, element_id)
end

############################################################################################
#                                        Evaluation                                        #
############################################################################################

function evaluate(
    geometry::MaskedGeometry, element_id::Int, xi::Points.AbstractPoints{manifold_dim}
) where {manifold_dim}
    element_id_base, xi_base, _ = transform_evaluation_points(
        get_evaluation_mask(geometry), element_id, xi
    )
    x = evaluate(get_base_geometry(geometry), element_id_base, xi_base)

    return x
end

function jacobian(
    geometry::MaskedGeometry, element_id::Int, xi::Points.AbstractPoints{manifold_dim}
) where {manifold_dim}
    # map the canonical points to the base geometry
    element_id_base, xi_base, scaling = transform_evaluation_points(
        get_evaluation_mask(geometry), element_id, xi
    )
    # evaluate the base geometry
    J_base = jacobian(get_base_geometry(geometry), element_id_base, xi_base)
    diag_scaling = Diagonal([scaling...])
    J_scaled = [Jp * diag_scaling for Jp in J_base]

    return J_scaled
end

function hessian(
    geometry::MaskedGeometry, element_id::Int, xi::Points.AbstractPoints{manifold_dim}
) where {manifold_dim}
    element_id_base, xi_base, scaling = transform_evaluation_points(
        get_evaluation_mask(geometry), element_id, xi
    )
    hess_base = hessian(get_base_geometry(geometry), element_id_base, xi_base)
    scaling_mat = [scaling[i] * scaling[j] for i in 1:manifold_dim, j in 1:manifold_dim]
    for point in eachindex(hess_base), k in eachindex(hess_base[point])
        hess_base[point][k] .*= scaling_mat
    end

    return hess_base
end

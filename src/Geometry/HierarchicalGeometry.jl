############################################################################################
#                                        Structure                                         #
############################################################################################

struct HierarchicalGeometry{manifold_dim, image_dim, H} <: AbstractGeometry{manifold_dim, image_dim, 1}
    hier_space::H

    function HierarchicalGeometry(
        hier_space::FunctionSpaces.HierarchicalFiniteElementSpace{manifold_dim, S, T}
    ) where {manifold_dim, S, T}
        return new{
            manifold_dim, manifold_dim, FunctionSpaces.HierarchicalFiniteElementSpace{manifold_dim, S, T}
        }(
            hier_space
        )
    end
end

############################################################################################
#                                      Basic Getters                                       #
############################################################################################

function get_fe_space(geometry::HierarchicalGeometry)
    return geometry.hier_space
end

function get_num_elements(geometry::HierarchicalGeometry)
    return FunctionSpaces.get_num_elements(get_fe_space(geometry))
end

function get_num_levels(geometry::HierarchicalGeometry)
    return FunctionSpaces.get_num_levels(get_fe_space(geometry))
end

function get_num_subdivisions(geometry::HierarchicalGeometry)
    return FunctionSpaces.get_num_subdivisions(get_fe_space(geometry))
end

# This functions will need to be updated, since it currently assumes a single patch
# geometry. This will also change when the dependency on FunctionSpaces is flipped.
function get_parametric_geometry(geometry::HierarchicalGeometry, patch_id::Int)
    @warn "The `patch_id` argument is currently ignored."
    return compute_parametric_geometry(get_fe_space(geometry))
end

############################################################################################
#                                      Other methods                                       #
############################################################################################

function evaluate(
    geometry::HierarchicalGeometry{manifold_dim, H},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, H}
    element_vertices = FunctionSpaces.get_element_vertices(get_fe_space(geometry), element_id)
    A = zeros(Float64, manifold_dim, manifold_dim)
    b = zeros(Float64, manifold_dim)
    for k in 1:manifold_dim
        A[k, k] = (element_vertices[k][2] - element_vertices[k][1])
        b[k] = element_vertices[k][1]
    end

    mapped_points = affine_map(xi, A, b)
    mapped_matrix = vec_tuple_to_matrix(mapped_points)

    return mapped_matrix
end

function jacobian(
    geometry::HierarchicalGeometry{manifold_dim, H},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, H}
    element_vertices = FunctionSpaces.get_element_vertices(get_fe_space(geometry), element_id)
    delta = zeros(Float64, manifold_dim)
    for k in 1:manifold_dim
        delta[k] = (element_vertices[k][2] - element_vertices[k][1])
    end

    num_points = Points.get_num_points(xi)
    J = zeros(Float64, num_points, manifold_dim, manifold_dim)
    for k in range(1, manifold_dim)
        for point in 1:num_points
            J[point, k, k] = delta[k]
        end
    end

    return J
end

function get_element_vertices(geometry::HierarchicalGeometry, element_id::Int)
    level, element_level_id = FunctionSpaces.convert_to_element_level_and_level_id(
        get_fe_space(geometry), element_id
    )

    return FunctionSpaces.get_element_vertices(
        FunctionSpaces.get_space(get_fe_space(geometry), level), element_level_id
    )
end

function get_element_lengths(
    geometry::HierarchicalGeometry{manifold_dim}, element_id::Int
) where {manifold_dim}
    element_vertices = get_element_vertices(geometry, element_id)
    element_lengths = ntuple(manifold_dim) do k
        return element_vertices[k][2] - element_vertices[k][1]
    end

    return element_lengths
end

function get_element_measure(geometry::HierarchicalGeometry, element_id::Int)
    return prod(get_element_lengths(geometry, element_id))
end

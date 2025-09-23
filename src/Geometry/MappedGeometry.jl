
struct Mapping{M, dM}
    dimensions::NTuple{2, Int}
    mapping::M
    dmapping::dM

    function Mapping(
        dimensions::NTuple{2, Int}, mapping::M, dmapping::dM
    ) where {M <: Function, dM <: Function}
        return new{M, dM}(dimensions, mapping, dmapping)
    end
end

"""
    get_manifold_dim(mapping::Mapping)

Returns the dimension of the domain manifold of the mapping.

# Arguments
- `mapping::Mapping`: The mapping structure.

# Returns
- `::Int`: The dimension of the domain manifold.
"""
function get_manifold_dim(mapping::Mapping)
    return mapping.dimensions[1]
end

"""
    get_image_dim(mapping::Mapping)

Returns the dimension of the image manifold of the mapping.

# Arguments
- `mapping::Mapping`: The mapping structure.

# Returns
- `::Int`: The dimension of the image manifold.
"""
function get_image_dim(mapping::Mapping)
    return mapping.dimensions[2]
end

"""
    evaluate(mapping::Mapping, x::Matrix{Float64})

Evaluates the mappping of the points `x` from the parametric space to physical space.

# Arguments
- `mapping::Mapping`: The mapping defining the transformation of the points `x`.
- `x::Matrix{Float64}`: The points in parametric space to be mapped.

# Returns
- `::Matrix{Float64}`: The mapped points in physical space. The size of the matrix is
    `(num_points, image_dim)`, where `num_points` is the number of rows in `x` and
    `image_dim` is the dimension of the mapped points.
"""
function evaluate(mapping::Mapping, x::Matrix{Float64})
    image_dim = get_image_dim(mapping)
    num_points = size(x, 1)
    eval = zeros(num_points, image_dim)

    for point in 1:num_points
        eval[point, :] .= mapping.mapping(view(x, point, :))
    end

    return eval
end

"""
    jacobian(mapping::Mapping, x::Matrix{Float64})

Evaluates the jacobian at the physical points mapped from the parametric points `x`.

# Arguments
- `mapping::Mapping`: The mapping defining the transformation of the points `x`.
- `x::Matrix{Float64}`: The points in parametric space to be mapped.

# Returns
- `::Matrix{Float64}`: The mapped points in physical space. The size of the matrix is
    `(num_points, image_dim, manifold_dim)`, where `num_points` is the number of rows in
    `x`, `image_dim` is the dimension of the mapped points and `manifold_dim` is the number
    of columns in `x`.
"""
function jacobian(
    mapping::Mapping{M, dM}, x::Matrix{Float64}
) where {M <: Function, dM <: Function}
    manifold_dim = get_manifold_dim(mapping)
    image_dim = get_image_dim(mapping)
    num_points = size(x, 1)
    J = zeros(num_points, image_dim, manifold_dim)

    for i in 1:num_points
        # Compute Jacobian for each input point
        J[i, :, :] .= mapping.dmapping(view(x, i, :))
    end

    return J
end

struct MappedGeometry{manifold_dim, num_patches, G, Map} <: AbstractGeometry{manifold_dim, num_patches}
    geometry::G
    mapping::Map
    num_elements::Int
    num_elements_per_patch::NTuple{num_patches, Int}

    # Constructor for multiple mappings, one per patch.
    function MappedGeometry(
        geometry::G, mapping::M
    ) where {manifold_dim, num_patches, G <: AbstractGeometry{manifold_dim, num_patches}, M <: NTuple{num_patches, Mapping}}
        num_elements_per_patch = get_num_elements_per_patch(geometry)

        return new{manifold_dim, num_patches, G, M}(
            geometry, mapping, sum(num_elements_per_patch), Tuple(num_elements_per_patch)
        )
    end

    # Constructor for a single mapping for all patches.
    function MappedGeometry(
        geometry::G, mapping::Map
    ) where {manifold_dim, num_patches, G <: AbstractGeometry{manifold_dim, num_patches}, Map <: Mapping}
        num_elements_per_patch = get_num_elements_per_patch(geometry)

        return new{manifold_dim, num_patches, G, Map}(
            geometry, mapping, sum(num_elements_per_patch), Tuple(num_elements_per_patch)
        )
    end
end

function get_image_dim(geometry::MappedGeometry)
    return geometry.mapping.dimensions[2]
end

function get_element_lengths(geometry::MappedGeometry, element_id::Int)
    return get_element_lengths(geometry.geometry, element_id)
end

function get_element_measure(geometry::MappedGeometry, element_id::Int)
    return get_element_measure(geometry.geometry, element_id)
end

function evaluate(
    geometry::MappedGeometry, element_idx::Int, xi::Points.AbstractPoints{manifold_dim}
) where {manifold_dim}
    x = evaluate(geometry.geometry, element_idx, xi)
    x_mapped = evaluate(geometry.mapping, x)

    return x_mapped
end

function jacobian(
    geometry::MappedGeometry, element_idx::Int, xi::Points.AbstractPoints{manifold_dim}
) where {manifold_dim}
    # the Jacobian for the mapping from the elements to base geometry image
    J_1 = jacobian(geometry.geometry, element_idx, xi)
    x = evaluate(geometry.geometry, element_idx, xi)
    # the mapping from the image of the  base geometry to the image of the mapping
    J_2 = jacobian(geometry.mapping, x)

    num_points = size(x, 1)
    image_dim = get_image_dim(geometry)
    J_1_image_dim = get_image_dim(geometry.geometry)

    J = zeros(num_points, image_dim, manifold_dim)
    for k_im_1 in 1:J_1_image_dim
        for cart_id in CartesianIndices(J)
            (point, k_im, k_mani) = Tuple(cart_id)
            J[point, k_im, k_mani] += J_2[point, k_im, k_im_1] * J_1[point, k_im_1, k_mani]
        end
    end

    return J
end

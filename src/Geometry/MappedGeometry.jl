
struct Mapping{manifold_dim, image_dim, M, dM}
    dimensions::NTuple{2, Int}
    mapping::M
    dmapping::dM

    function Mapping(
        dimensions::NTuple{2, Int}, mapping::M, dmapping::dM
    ) where {M <: Function, dM <: Function}
        return new{dimensions[1], dimensions[2], M, dM}(dimensions, mapping, dmapping)
    end

    function Mapping(
        ::Val{manifold_dim}, ::Val{image_dim}, mapping::M, dmapping::dM
    ) where {manifold_dim, image_dim, M <: Function, dM <: Function}
        return new{manifold_dim, image_dim, M, dM}(dimensions, mapping, dmapping)
    end
end

"""
    get_manifold_dim(mapping::Mapping{manifold_dim, image_dim, M, dM})

Returns the dimension of the domain manifold of the mapping.

# Arguments
- `::Mapping{manifold_dim, image_dim, M, dM}`: The mapping structure.

# Returns
- `::Int`: The dimension of the domain manifold.
"""
function get_manifold_dim(::Mapping{manifold_dim, image_dim, M, dM}) where {
    manifold_dim, image_dim, M, dM
}
    return manifold_dim
end

"""
    get_image_dim(mapping::Mapping{manifold_dim, image_dim, M, dM})

Returns the dimension of the image manifold of the mapping.

# Arguments
- `::Mapping{manifold_dim, image_dim, M, dM}`: The mapping structure.

# Returns
- `::Int`: The dimension of the image manifold.
"""
function get_image_dim(::Mapping{manifold_dim, image_dim, M, dM}) where {
    manifold_dim, image_dim, M, dM
}
    return image_dim
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
    mapping::Mapping{manifold_dim, image_dim, M, dM}, x::Matrix{Float64}
) where {manifold_dim, image_dim, M <: Function, dM <: Function}
    num_points = size(x, 1)
    J = zeros(num_points, image_dim, manifold_dim)

    for i in 1:num_points
        # Compute Jacobian for each input point
        J[i, :, :] .= mapping.dmapping(view(x, i, :))
    end

    return J
end

struct MappedGeometry{manifold_dim, image_dim, num_patches, G, Map} <: AbstractGeometry{manifold_dim, image_dim, num_patches}
    geometry::G
    mapping::Map
    num_elements::Int
    num_elements_per_patch::NTuple{num_patches, Int}

    # Constructor for multiple mappings, one per patch, with each parametric geometry being
    # a different object for each patch.
    function MappedGeometry(
        geometry::G, mapping::M
    ) where {
        manifold_dim,
        image_dim_base,
        num_patches,
        G <: NTuple{num_patches, AbstractGeometry{manifold_dim, image_dim_base, 1}},
        M <: NTuple{num_patches, Mapping}
    }
        num_elements_per_patch = get_num_elements_per_patch(geometry)

        image_dim = get_image_dim(mapping[1])
        for i in eachindex(mapping)
            if get_image_dim(mapping[i]) != image_dim
                throw(ArgumentError(LazyString(
                    "All mappings must have the same image dim. Mapping ",i,
                    " has an image dim of ",get_image_dim(mapping[i])," instead of ",
                    image_dim
                )))
            end
        end

        return new{manifold_dim, image_dim, num_patches, G, M}(
            geometry, mapping, sum(num_elements_per_patch), Tuple(num_elements_per_patch)
        )
    end

    # Constructor for multiple mappings, one per patch, with the parametric geometry being
    # only one object. If this is a single patch geometry, this one geometry will be used
    # for every patch. If it is a multi-patch geometry, it must have the same number of
    # patches as the number of mappings.
    function MappedGeometry(
        geometry::G, mapping::M
    ) where {
        manifold_dim,
        image_dim_base,
        num_patches,
        G <: AbstractGeometry{manifold_dim, image_dim_base, num_patches_G},
        M <: NTuple{num_patches, Mapping}
    }
        num_elements_per_patch = get_num_elements_per_patch(geometry)
        image_dim = get_image_dim(mapping[1])
        for i in eachindex(mapping)
            if get_image_dim(mapping[i]) != image_dim
                throw(ArgumentError(LazyString(
                    "All mappings must have the same image dim. Mapping ",i,
                    " has an image dim of ",get_image_dim(mapping[i])," instead of ",
                    image_dim
                )))
            end
        end

        if num_patches_G != 1 || num_patches_G != num_patches
            throw(ArgumentError(LazyString(
                "The underlying geometry must have the same number of patches as the ",
                "number of mappings, but the geometry has ",num_patches_G,
                " patches, while there are ",num_patches," mappings."
            )))
        end

        return new{manifold_dim, image_dim, num_patches, G, M}(
            geometry, mapping, sum(num_elements_per_patch), Tuple(num_elements_per_patch)
        )
    end

    # Constructor for a single mapping for all patches.
    function MappedGeometry(
        geometry::G, mapping::Map
    ) where {
        manifold_dim,
        image_dim_base,
        num_patches,
        G <: NTuple{num_patches, AbstractGeometry{manifold_dim, image_dim_base, 1}},
        Map <: Mapping
    }
        num_elements_per_patch = get_num_elements_per_patch(geometry)

        image_dim = get_image_dim(mapping)

        return new{manifold_dim, image_dim, num_patches, G, Map}(
            geometry, mapping, sum(num_elements_per_patch), Tuple(num_elements_per_patch)
        )
    end

    # Constructor for a single mapping for a single patch.
    function MappedGeometry(
        geometry::G, mapping::Map
    ) where {
        manifold_dim,
        image_dim_base,
        G <: AbstractGeometry{manifold_dim, image_dim_base, 1},
        Map <: Mapping
    }
        num_elements_per_patch = get_num_elements_per_patch(geometry)

        image_dim = get_image_dim(mapping)

        return new{manifold_dim, image_dim, 1, G, Map}(
            geometry, mapping, sum(num_elements_per_patch), Tuple(num_elements_per_patch)
        )
    end
end

function get_parametric_geometry(
    geometry::MappedGeometry{manifold_dim, image_dim, num_patches, G, Map}
) where {manifold_dim, image_dim, num_patches, G <: AbstractGeometry, Map}
    return get_parametric_geometry(geometry.geometry)
end

function get_parametric_geometry(
    geometry::MappedGeometry{manifold_dim, image_dim, num_patches, G, Map}
) where {
    manifold_dim, image_dim, num_patches, G <: NTuple{num_patches, AbstractGeometry}, Map
}
    return CartesianGeometry(get_breakpoints.(get_parametric_geometry.(geometry.geometry)))
end

function get_parametric_geometry(
    geometry::MappedGeometry{manifold_dim, image_dim, num_patches, G, Map}, patch_id::Int
) where {manifold_dim, image_dim, num_patches, G <: AbstractGeometry, Map}
    return get_parametric_geometry(geometry.geometry, patch_id)
end

function get_parametric_geometry(
    geometry::MappedGeometry{manifold_dim, image_dim, num_patches, G, Map}, patch_id::Int
) where {
    manifold_dim, image_dim, num_patches, G <: NTuple{num_patches, AbstractGeometry}, Map
}
    return get_parametric_geometry(geometry.geometry[patch_id], patch_id)
end

function get_mapping(
    geometry::MappedGeometry{manifold_dim, image_dim, num_patches, G, Map}, patch_id::Int=1
) where {manifold_dim, image_dim, num_patches, G, Map <: Mapping}
    return geometry.mapping
end

function get_mapping(
    geometry::MappedGeometry{manifold_dim, image_dim, num_patches, G, Map}, patch_id::Int=1
) where {manifold_dim, image_dim, num_patches, G, Map <: NTuple{num_patches, Mapping}}
    return geometry.mapping[patch_id]
end


function get_element_lengths(geometry::MappedGeometry, element_id::Int)
    patch_id, local_element_id = get_patch_and_local_element_id(geometry, element_id)
    return get_element_lengths(
        get_parametric_geometry(geometry, patch_id), local_element_id
    )
end

function get_element_measure(geometry::MappedGeometry, element_id::Int)
    patch_id, local_element_id = get_patch_and_local_element_id(geometry, element_id)
    return get_element_measure(
        get_parametric_geometry(geometry, patch_id), local_element_id
    )
end

function evaluate(
    geometry::MappedGeometry{manifold_dim, image_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches}
    patch_id, local_element_id = get_patch_and_local_element_id(geometry, element_id)
    x = evaluate(get_parametric_geometry(geometry, patch_id), local_element_id, xi)
    x_mapped = evaluate(get_mapping(geometry, patch_id), x)

    return x_mapped
end

function jacobian(
    geometry::MappedGeometry{manifold_dim, image_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches}
    patch_id, local_element_id = get_patch_and_local_element_id(geometry, element_id)
    # the Jacobian for the mapping from the elements to base geometry image
    J_1 = jacobian(get_parametric_geometry(geometry, patch_id), local_element_id, xi)
    x = evaluate(get_parametric_geometry(geometry, patch_id), local_element_id, xi)
    # the mapping from the image of the  base geometry to the image of the mapping
    J_2 = jacobian(get_mapping(geometry, patch_id), x)

    num_points = size(x, 1)
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

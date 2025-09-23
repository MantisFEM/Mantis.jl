"""
    CartesianGeometry{manifold_dim, num_patches, T, CI} <: AbstractAnalyticalGeometry{manifold_dim, num_patches}

A structure representing a Cartesian grid geometry in `manifold_dim` dimensions. Can have
multiple patches, even though each patch is still a Cartesian grid. Note that the patches
are not required to have a matching grid.

# Fields
- `breakpoints::NTuple{manifold_dim, AbstractVector{T}}`: A tuple of vectors defining the
    grid points in each dimension.
- `cart_num_elements::CI`: A (tuple of) `CartesianIndices` representing the indices of
    elements in the grid for each patch. Used to convert from linear to cartesian indexing.
"""
struct CartesianGeometry{manifold_dim, num_patches, T, CI} <: AbstractAnalyticalGeometry{manifold_dim, num_patches}
    breakpoints::T
    cart_num_elements::CI

    function CartesianGeometry(
        breakpoints::T
    ) where {
        manifold_dim,
        num_patches,
        NT <: Number,
        T <: NTuple{num_patches, NTuple{manifold_dim, AbstractVector{NT}}}
    }
        cart_num_elements = ntuple(num_patches) do i
            return CartesianIndices(
                ntuple(dim -> length(breakpoints[i][dim]) - 1, manifold_dim)
            )
        end

        return new{manifold_dim, num_patches, T, typeof(cart_num_elements)}(
            breakpoints, cart_num_elements
        )
    end

    # Convenience constructor for single patch geometries.
    function CartesianGeometry(
        breakpoints::NTuple{manifold_dim, AbstractVector{NT}}
    ) where {manifold_dim, NT <: Number}
        return CartesianGeometry((breakpoints,))
    end
end

get_cart_num_elements(geometry::CartesianGeometry, patch_id::Int=1) = geometry.cart_num_elements[patch_id]
get_breakpoints(geometry::CartesianGeometry, patch_id::Int=1) = geometry.breakpoints[patch_id]
get_image_dim(geometry::CartesianGeometry) = get_manifold_dim(geometry)

function get_num_elements(geometry::CartesianGeometry)
    return mapreduce(prod, +, get_constituent_num_elements(geometry))
end

function get_num_elements_per_patch(geometry::AbstractGeometry)
    return (prod(get_constituent_num_elements(geometry)[i]) for i in 1:get_num_patches(geometry))
end

function get_constituent_element_id(geometry::CartesianGeometry, element_id::Int)
    patch_id, local_element_id = get_patch_and_local_element_id(geometry, element_id)
    return get_cart_num_elements(geometry, patch_id)[local_element_id], patch_id
end

function get_constituent_num_elements(
    geometry::CartesianGeometry{manifold_dim, num_patches}
) where {manifold_dim, num_patches}
    return ntuple(num_patches) do i
        return Tuple(maximum(get_cart_num_elements(geometry, i)))
    end
end

function evaluate(
    geometry::CartesianGeometry{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    const_element_id, patch_id = get_constituent_element_id(geometry, element_id)
    breakpoints = get_breakpoints(geometry, patch_id)
    scaling = ntuple(
        dim ->
            breakpoints[dim][const_element_id[dim] + 1] -
            breakpoints[dim][const_element_id[dim]],
        manifold_dim,
    )
    offset = ntuple(dim -> breakpoints[dim][const_element_id[dim]], manifold_dim)
    num_points = Points.get_num_points(xi)
    eval = zeros(num_points, manifold_dim)
    for (i, point) in enumerate(xi)
        for dim in axes(eval, 2)
            eval[i, dim] += affine_map(point[dim], scaling[dim], offset[dim])
        end
    end

    return eval
end

function jacobian(
    geometry::CartesianGeometry{manifold_dim},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim}
    const_element_id, patch_id = get_constituent_element_id(geometry, element_id)
    breakpoints = get_breakpoints(geometry, patch_id)
    scaling = ntuple(
        dim ->
            breakpoints[dim][const_element_id[dim] + 1] -
            breakpoints[dim][const_element_id[dim]],
        manifold_dim,
    )
    # Generate the Jacobian for the Cartesian grid
    # Per point, it's a diagonal matrix multiplied by the cell spacings in each direction
    num_points = Points.get_num_points(xi)
    J = zeros(num_points, manifold_dim, manifold_dim)
    for dim in axes(J, 3)
        for point in axes(J, 1)
            J[point, dim, dim] = scaling[dim]
        end
    end

    return J
end

function get_element_vertices(
    geometry::CartesianGeometry{manifold_dim}, element_id::Int
) where {manifold_dim}
    const_element_id, patch_id = get_constituent_element_id(geometry, element_id)
    breakpoints = get_breakpoints(geometry, patch_id)
    element_vertices = ntuple(manifold_dim) do dim
        vertex_1 = breakpoints[dim][const_element_id[dim]]
        vertex_2 = breakpoints[dim][const_element_id[dim] + 1]

        return [vertex_1, vertex_2]
    end

    return element_vertices
end

function get_element_lengths(
    geometry::CartesianGeometry{manifold_dim}, element_id::Int
) where {manifold_dim}
    element_vertices = get_element_vertices(geometry, element_id)
    element_lengths = ntuple(manifold_dim) do dim
        return element_vertices[dim][2] - element_vertices[dim][1]
    end

    return element_lengths
end

function get_element_measure(geometry::CartesianGeometry, element_id::Int)
    element_lengths = get_element_lengths(geometry, element_id)

    return prod(element_lengths)
end

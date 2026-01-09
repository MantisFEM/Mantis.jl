"""
    CartesianGeometry{manifold_dim, image_dim, num_patches, T, CI} <: AbstractGeometry{
        manifold_dim, image_dim, num_patches
    }

A structure representing a Cartesian grid geometry in `manifold_dim` dimensions. Can have
multiple patches, even though each patch is still a Cartesian grid. Note that the patches
are not required to have a matching grid.

# Fields
- `breakpoints::T`: A tuple of vectors defining the grid points in each dimension.
- `cart_num_elements::CI`: A (tuple of) `CartesianIndices` representing the indices of
    elements in the grid for each patch.

# Constructors
- `CartesianGeometry(
        breakpoints::T
    ) where {
        manifold_dim,
        num_patches,
        NT <: Number,
        T <: NTuple{num_patches, NTuple{manifold_dim, AbstractVector{NT}}},
    }`: General constructor.
- `CartesianGeometry(
        breakpoints::NTuple{manifold_dim, AbstractVector{NT}}
    ) where {manifold_dim, NT <: Number}`: Single-patch convenience constructor.
"""
struct CartesianGeometry{manifold_dim, image_dim, num_patches, T, CI, LI} <:
       AbstractGeometry{manifold_dim, image_dim, num_patches}
    breakpoints::T
    cart_num_elements::CI
    lin_num_elements::LI

    function CartesianGeometry(
        breakpoints::T
    ) where {
        manifold_dim,
        num_patches,
        NT <: Number,
        T <: NTuple{num_patches, NTuple{manifold_dim, AbstractVector{NT}}},
    }
        foreach(breakpoints) do patch_breakpoints
            for k in 1:manifold_dim
                if !isequal(patch_breakpoints[k], unique(patch_breakpoints[k]))
                    throw(ArgumentError("Breakpoints should be unique."))
                elseif !isequal(patch_breakpoints[k], sort(patch_breakpoints[k]))
                    throw(ArgumentError("Breakpoints should be stricly increasing."))
                end
            end
        end

        cart_num_elements = ntuple(Val(num_patches)) do i
            return CartesianIndices(
                ntuple(dim -> length(breakpoints[i][dim]) - 1, manifold_dim)
            )
        end

        lin_num_elements = ntuple(
            patch -> LinearIndices(cart_num_elements[patch]), Val(num_patches)
        )

        return new{
            manifold_dim,
            manifold_dim,
            num_patches,
            T,
            typeof(cart_num_elements),
            typeof(lin_num_elements),
        }(
            breakpoints, cart_num_elements, lin_num_elements
        )
    end

    # Convenience constructor for single patch geometries.
    function CartesianGeometry(
        breakpoints::NTuple{manifold_dim, AbstractVector{NT}}
    ) where {manifold_dim, NT <: Number}
        return CartesianGeometry((breakpoints,))
    end

    # Convenience constructor for 1D, single patch geometries.
    function CartesianGeometry(breakpoints::AbstractVector{NT}) where {NT <: Number}
        return CartesianGeometry(((breakpoints,),))
    end
end

# Get properties.
get_breakpoints(geometry::CartesianGeometry, patch_id::Int=1) =
    geometry.breakpoints[patch_id]
get_breakpoints_per_dim(geometry::CartesianGeometry, patch_id::Int=1, dim::Int=1) =
    geometry.breakpoints[patch_id][dim]
get_breakpoint(geometry::CartesianGeometry, patch_id::Int=1, dim::Int=1, point::Int=1) =
    geometry.breakpoints[patch_id][dim][point]

"""
	get_cart_num_elements(geometry::CartesianGeometry, patch_id::Int=1)

Returns a CartesianIndices iterator of all elements in the patch indicated by `patch_id`.
"""
get_cart_num_elements(geometry::CartesianGeometry, patch_id::Int=1) =
    geometry.cart_num_elements[patch_id]

"""
	get_lin_num_elements(geometry::CartesianGeometry, patch_id::Int=1)

Returns a LinearIndices iterator of all elements in the patch indicated by `patch_id`.
"""
get_lin_num_elements(geometry::CartesianGeometry, patch_id::Int=1) =
    geometry.lin_num_elements[patch_id]

# Getters for consituents.
function get_constituent_element_id(geometry::CartesianGeometry, element_id::Int)
    patch_id, local_element_id = get_patch_and_local_element_id(geometry, element_id)
    return get_cart_num_elements(geometry, patch_id)[local_element_id], patch_id
end

function get_constituent_num_elements(geometry::CartesianGeometry, patch_id::Int)
    # The cartesian number of elements is always ordered and created with the number of
    # elements in each constituent. So, its last entry is the total number of elements per
    # constituent. This means we don't have to search for its maximum.
    return Tuple(last(get_cart_num_elements(geometry, patch_id)))
end
function get_constituent_num_elements(geometry::CartesianGeometry)
    return (get_constituent_num_elements(geometry, i) for i in 1:get_num_patches(geometry))
end

# Getters for geometries.
function get_geometry(geometry::CartesianGeometry, patch_id::Int)
    return CartesianGeometry((geometry.breakpoints[patch_id],))
end

# function get_parametric_geometry(geometry::CartesianGeometry)
#     return geometry
# end

# function get_parametric_geometry(geometry::CartesianGeometry, patch_id::Int)
#     return get_geometry(geometry, patch_id)
# end

# Getters for numbers, sizes, shapes, lengths, etc.
function get_num_elements(geometry::CartesianGeometry, patch_id::Int)
    return length(get_cart_num_elements(geometry, patch_id))
end
function get_num_elements(geometry::CartesianGeometry)
    num_elements = 0
    for patch_id in 1:get_num_patches(geometry)
        num_elements += get_num_elements(geometry, patch_id)
    end
    return num_elements
end

function get_num_elements_per_patch(geometry::CartesianGeometry)
    return ntuple(Val(get_num_patches(geometry))) do patch_id
        return get_num_elements(geometry, patch_id)
    end
end

function get_element_vertices(geometry::CartesianGeometry, element_id::Int)
    const_element_id, patch_id = get_constituent_element_id(geometry, element_id)
    element_vertices = ntuple(get_manifold_dim(geometry)) do dim
        vertex_1 = get_breakpoint(geometry, patch_id, dim, const_element_id[dim])
        vertex_2 = get_breakpoint(geometry, patch_id, dim, const_element_id[dim] + 1)

        return (vertex_1, vertex_2)
    end

    return element_vertices
end

function get_element_lengths(geometry::CartesianGeometry, element_id::Int)
    # Directly compute the element lengths without calling `get_element_vertices`. This
    # avoids the overhead of computing the vertices explicitly.
    const_element_id, patch_id = get_constituent_element_id(geometry, element_id)
    element_lengths = ntuple(get_manifold_dim(geometry)) do dim
        return get_breakpoint(geometry, patch_id, dim, const_element_id[dim] + 1) -
               get_breakpoint(geometry, patch_id, dim, const_element_id[dim])
    end

    return element_lengths
end

function get_element_measure(geometry::CartesianGeometry, element_id::Int)
    return prod(get_element_lengths(geometry, element_id))
end

# Evaluations and derivatives.
function evaluate(
    geometry::CartesianGeometry{manifold_dim, image_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches}
    const_element_id, patch_id = get_constituent_element_id(geometry, element_id)
    scaling = get_element_lengths(geometry, element_id)
    offset = ntuple(manifold_dim) do dim
        return get_breakpoint(geometry, patch_id, dim, const_element_id[dim])
    end
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
    geometry::CartesianGeometry{manifold_dim, image_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches}
    # Per point, the jacobian is a diagonal matrix with the cell spacings in each dimension.
    scaling = get_element_lengths(geometry, element_id)
    return [
        SMatrix{manifold_dim, manifold_dim}(LinearAlgebra.I) .* scaling for
        _ in 1:Points.get_num_points(xi)
    ]
end

function hessian(
    geometry::CartesianGeometry{manifold_dim, image_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches}
    # The Hessian is zero for Cartesian geometries.
    num_points = Points.get_num_points(xi)
    return [
        ntuple(image_dim) do _
            return zeros(SMatrix{manifold_dim, manifold_dim})
        end for _ in 1:num_points
    ]
end

"""
    module Geometry

Contains all geometry structure definitions and related methods.
"""
module Geometry

using LinearAlgebra
using StaticArrays

import ..Points
import ..GeneralHelpers
import ..Topology

abstract type AbstractGeometry{manifold_dim, image_dim, num_patches} end

"""
    get_manifold_dim(::AbstractGeometry{manifold_dim, image_dim, num_patches})

Returns the dimensions of the domain manifold of a given geometry.

# Arguments
- `::AbstractGeometry{manifold_dim, image_dim, num_patches}`: The geometry being used.

# Returns
- `::Int`: The domain manifold dimension.
"""
get_manifold_dim(
    ::AbstractGeometry{manifold_dim, image_dim, num_patches}
) where {manifold_dim, image_dim, num_patches} = manifold_dim

"""
    get_image_dim(::AbstractGeometry{manifold_dim, image_dim, num_patches})

Returns the dimensions of the image manifold of a given geometry.

# Arguments
- `::AbstractGeometry{manifold_dim, image_dim, num_patches}`: The geometry being used.

# Returns
- `::Int`: The image manifold dimension.
"""
get_image_dim(
    ::AbstractGeometry{manifold_dim, image_dim, num_patches}
) where {manifold_dim, image_dim, num_patches} = image_dim

"""
    get_num_patches(::AbstractGeometry{manifold_dim, image_dim, num_patches})

Returns the number of patches in a given geometry.

# Arguments
- `::AbstractGeometry{manifold_dim, image_dim, num_patches}`: The geometry being used.

# Returns
- `::Int`: The number of patches in the geometry.
"""
get_num_patches(
    ::AbstractGeometry{manifold_dim, image_dim, num_patches}
) where {manifold_dim, image_dim, num_patches} = num_patches

get_topology(geometry::AbstractGeometry) = geometry.topology

"""
    get_patch_id(geometry::AbstractGeometry, element_id::Int)

Get the ID of the patch to which the specified global element belongs.

# Arguments
- `geometry::AbstractGeometry`: The (multi-patch) geometry.
- `element_id::Int`: The global element ID.

# Returns
- `patch_id::Int`: ID of the patch to which the element belongs.

# Exceptions
- ArgumentError: This error is thrown if the given `element_id` is larger than the total
    number of elements found in the `geometry`.
- Error "no field 'num_elements'": This error is thrown if the number of elements is not
    stored in a field called `num_elements` and if no specific `get_num_elements` method is
    defined for the given `geometry`.
- Error "no field 'num_elements_per_patch'": This error is thrown if the number of elements
    per patch is not stored in a field called `num_elements_per_patch` and if no specific
    `get_num_elements`-method (with `patch_id` argument) is defined for the given
    `geometry`.
"""
function get_patch_id(geometry::AbstractGeometry, element_id::Int)
    num_cumulative_elements = 0
    for patch_id in 1:get_num_patches(geometry)
        num_cumulative_elements += get_num_elements(geometry, patch_id)
        if element_id <= num_cumulative_elements
            return patch_id
        end
    end
    throw(
        ArgumentError(
            LazyString(
                "Element ID ",
                element_id,
                " exceeds the total number of elements (",
                num_cumulative_elements,
                ") in the geometry.",
            ),
        ),
    )
end

function get_patch_id(
    geometry::AbstractGeometry{manifold_dim, image_dim, 1}, element_id::Int
) where {manifold_dim, image_dim}
    if element_id <= get_num_elements(geometry)
        return 1
    else
        throw(
            ArgumentError(
                LazyString(
                    "Element ID ",
                    element_id,
                    " exceeds the total number of elements (",
                    get_num_elements(geometry),
                    ") in the geometry.",
                ),
            ),
        )
    end
end

"""
    get_patch_and_local_element_id(geometry::AbstractGeometry, element_id::Int)

Get the constituent patch ID and local element ID for the specified global element ID.

# Arguments
- `geometry::AbstractGeometry`: The multi-patch geometry.
- `element_id::Int`: The global element ID.

# Returns
- `patch_id::Int`: The patch ID
- `local_element_id::Int`: The local element ID.

# Exceptions
- ArgumentError: This error is thrown if the given `element_id` is larger than the total
    number of elements found in the `geometry`.
- Error "no field 'num_elements'": This error is thrown if the number of elements is not
    stored in a field called `num_elements` and if no specific `get_num_elements` method is
    defined for the given `geometry`.
- Error "no field 'num_elements_per_patch'": This error is thrown if the number of elements
    per patch is not stored in a field called `num_elements_per_patch` and if no specific
    `get_num_elements`-method (with `patch_id` argument) is defined for the given
    `geometry`.
"""
function get_patch_and_local_element_id(geometry::AbstractGeometry, element_id::Int)
    num_cumulative_elements = 0
    local_element_id = element_id
    for patch_id in 1:get_num_patches(geometry)
        num_element_patch_i = get_num_elements(geometry, patch_id)
        num_cumulative_elements += num_element_patch_i

        if element_id <= num_cumulative_elements
            return patch_id, local_element_id
        end

        # Only subtract the number of elements on the current patch after we're sure that
        # the element is not on this patch. Otherwise we will miscount.
        local_element_id -= num_element_patch_i
    end

    throw(
        ArgumentError(
            LazyString(
                "Element ID ",
                element_id,
                " exceeds the total number of elements (",
                num_cumulative_elements,
                ") in the geometry.",
            ),
        ),
    )
end

function get_patch_and_local_element_id(
    geometry::AbstractGeometry{manifold_dim, image_dim, 1}, element_id::Int
) where {manifold_dim, image_dim}
    if element_id <= get_num_elements(geometry)
        return 1, element_id
    else
        throw(
            ArgumentError(
                LazyString(
                    "Element ID ",
                    element_id,
                    " exceeds the total number of elements (",
                    get_num_elements(geometry),
                    ") in the geometry.",
                ),
            ),
        )
    end
end

"""
    get_global_element_id(geometry::AbstractGeometry, patch_id::Int, local_element_id::Int)

Get the global element ID for the specified constituent patch ID and local element ID.

# Arguments
- `geometry::AbstractGeometry`: A (multi-)patch geometry.
- `patch_id::Int`: The constituent patch ID.
- `local_element_id::Int`: The local element ID.

# Returns
- `::Int`: The global element ID.
"""
function get_global_element_id(
    geometry::AbstractGeometry, patch_id::Int, local_element_id::Int
)
    if patch_id > get_num_patches(geometry)
        throw(
            ArgumentError(
                LazyString(
                    "The given patch_id, ",
                    patch_id,
                    ", exceeds the total number of patches on this geometry (",
                    get_num_patches(geometry),
                    ").",
                ),
            ),
        )
    end
    if local_element_id > get_num_elements(geometry)
        throw(
            ArgumentError(
                LazyString(
                    "The given local_element_id, ",
                    local_element_id,
                    ", exceeds the total number of element on this geometry (",
                    get_num_elements(geometry),
                    ").",
                ),
            ),
        )
    end
    if local_element_id > get_num_elements(geometry, patch_id)
        throw(
            ArgumentError(
                LazyString(
                    "The given local_element_id, ",
                    local_element_id,
                    ", exceeds the total number of element on this patch (",
                    get_num_elements(geometry, patch_id),
                    ").",
                ),
            ),
        )
    end
    global_element_id = 0
    for i in 1:(patch_id - 1)
        global_element_id += get_num_elements(geometry, i)
    end
    return global_element_id + local_element_id
end

"""
    get_num_elements(geometry::AbstractGeometry)
    get_num_elements(geometry::AbstractGeometry, patch_id::Int)
    get_num_elements(geometry::AbstractGeometry, patch_id::Int, local_object_id, geometric_dim)

Returns the number of elements in `geometry`. If only a `patch_id` is given, return the
number of elements in the patch. If a `patch_id`, `local_object_id`, and `geometric_dim`
are given, return the number of elements on the patch located on the requested topological
object.

# Arguments
- `geometry::AbstractGeometry`: The geometry being used.
- `patch_id::Int`:

# Returns
- `::Int`: The number of elements.

# Notes
This method is used as a fallback and assumes that the number of elements (per patch) are
explicitly stored.

# Exceptions
- Error "no field 'num_elements'": This error is thrown if the number of elements is not
    stored in a field called `num_elements` and if no specific `get_num_elements` method is
    defined for the given `geometry`.
- Error "no field 'num_elements_per_patch'": This error is thrown if the number of elements
    per patch is not stored in a field called `num_elements_per_patch` and if no specific
    `get_num_elements_per_patch`- or `get_num_elements`-method (with `patch_id` argument) is
    defined for the given `geometry`.
"""
function get_num_elements(geometry::AbstractGeometry)
    return geometry.num_elements
end
function get_num_elements(geometry::AbstractGeometry, patch_id::Int)
    return get_num_elements_per_patch(geometry)[patch_id]
end
function get_num_elements(geometry::AbstractGeometry, patch_id::Int, local_object_id, geometric_dim)
    return prod(size(get_elements(geometry, patch_id, local_object_id, geometric_dim)))
end

"""
    get_num_elements_per_patch(geometry::AbstractGeometry)

Returns the number of elements in each patch of `geometry`.

# Arguments
- `geometry::AbstractGeometry`: The geometry being used.

# Returns
- `::NTuple{num_patches, Int}`: The number of elements per patch in the geometry.

# Notes
This method is used as a fallback if there isn't a more specific method to be used. The
latter should only be implemented explicitly if necessary.

# Exceptions
- Error "no field 'num_elements_per_patch'": This error is thrown if the number of elements
    per patch is not stored in a field called `num_elements_per_patch` and if no specific
    `get_num_elements_per_patch` method is defined for the given `geometry`.
"""
function get_num_elements_per_patch(geometry::AbstractGeometry)
    return geometry.num_elements_per_patch
end

"""
    get_element_measure(geometry::AbstractGeometry, element_id::Int)

Computes the measure of the element given by `element_id` in `geometry`.

# Arguments
- 'geometry::AbstractGeometry': The geometry being used.
- 'element_id::Int': Index of the element being considered.

# Returns
- '<:Number': The measure of the element.

# Notes
There is no generic fallback for this method. It should be implemented for each concrete
geometry type.
"""
function get_element_measure(geometry::AbstractGeometry, element_id::Int)
    throw(MethodError(get_element_measure, (geometry, element_id)))
end

"""
    get_element_lengths(
        geometry::AbstractGeometry{manifold_dim, image_dim, num_patches}, element_id::Int,
    ) where {manifold_dim, image_dim, num_patches}

Computes the length, in each manifold dimension, of the element given by `element_id` in
`geometry`.

# Arguments
- 'geometry::AbstractGeometry': The geometry being used.
- 'element_id::Int': Index of the element being considered.

# Returns
- '<:NTuple{manifold_dim, Number}': The element's lengths.

# Notes
There is no generic fallback for this method. It should be implemented for each concrete
geometry type.
"""
function get_element_lengths(
    geometry::AbstractGeometry{manifold_dim, image_dim, num_patches}, element_id::Int
) where {manifold_dim, image_dim, num_patches}
    throw(MethodError(get_element_lengths, (geometry, element_id)))
end

"""
    get_element_vertices(
        geometry::AbstractGeometry{manifold_dim, image_dim, num_patches}, element_id::Int,
    ) where {manifold_dim, image_dim, num_patches}

Computes the vertices, in each manifold dimension, of the element given by `element_id` in
`geometry`.

# Arguments
- 'geometry::AbstractGeometry': The geometry being used.
- 'element_id::Int': Index of the element being considered.

# Returns
- '<:NTuple{manifold_dim, NTuple{2, Number}}': The element's vertices per manifold dim. For
    example, for the unit cube [0.0, 1.0]^3 this will be:
    ((0.0, 1.0), (0.0, 1.0), (0.0, 1.0)).

# Notes
There is no generic fallback for this method. It should be implemented for each concrete
geometry type.
"""
function get_element_vertices(
    geometry::AbstractGeometry{manifold_dim, image_dim, num_patches}, element_id::Int
) where {manifold_dim, image_dim, num_patches}
    throw(MethodError(get_element_vertices, (geometry, element_id)))
end

"""
    evaluate(
        geometry::AbstractGeometry{manifold_dim, image_dim, num_patches},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim, image_dim, num_patches}

Computes the coordinates of the physical points, given the canonical points `xi`, on
the element identified by `element_id` of a given `geometry`.

# Arguments
- `geometry::AbstractGeometry{manifold_dim, image_dim, num_patches}`: The geometry being evaluated.
- `element_id::Int`: The global element id.
- `xi::Points.AbstractPoints{manifold_dim}`: The points in the canonical domain at which to
    evaluate the geometry.

# Returns
- `::Matrix{Float64}`: The physical coordinates of `xi` on element `element_id`. The
    size of the matrix is `(num_eval_points, image_dim)`.

# Notes
There is no generic fallback for this method. It should be implemented for each concrete
geometry type.
"""
function evaluate(
    geometry::AbstractGeometry{manifold_dim, image_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches}
    throw(MethodError(evaluate, (geometry, element_id, xi)))
end

"""
    jacobian(
        geometry::AbstractGeometry{manifold_dim, image_dim, num_patches},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim, image_dim, num_patches}

Computes the jacobian at the physical points, given the canonical points `xi,` on the
element identified by `element_id` of a given `geometry`.

# Arguments
- `geometry::AbstractGeometry{manifold_dim, image_dim, num_patches}`: The geometry being used.
- `element_id::Int`: The global element id.
- `xi::Points.AbstractPoints{manifold_dim}`: The points in the canonical domain to evaluate
    the jacobian at.

# Returns
- `::Vector{SMatrix{image_dim, manifold_dim, Float64}}`: The jacobian at the physical points
    corresponding to the canonical points `xi` on element `element_id`. The length of the
    vector equals the number of evaluation points.

# Notes
There is no generic fallback for this method. It should be implemented for each concrete
geometry type.
"""
function jacobian(
    geometry::AbstractGeometry{manifold_dim, image_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches}
    throw(MethodError(jacobian, (geometry, element_id, xi)))
end

"""
    hessian(
        geometry::AbstractGeometry{manifold_dim, image_dim, num_patches},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim, image_dim, num_patches}

Computes the hessian at the physical points, given the canonical points `xi,` on the
element identified by `element_id` of a given `geometry`.

# Arguments
- `geometry::AbstractGeometry{manifold_dim, image_dim, num_patches}`: The geometry being used.
- `element_id::Int`: The global element id.
- `xi::Points.AbstractPoints{manifold_dim}`: The points in the canonical domain to evaluate
    the hessian at.

# Returns
- `::Vector{NTuple{image_dim, SMatrix{manifold_dim, manifold_dim}}`: The hessian at the
    physical points corresponding to the canonical points `xi` on element `element_id`. The
    length of the vector equals the number of evaluation points. Per points, there are
    `image_dim` many hessian metrices, hence the tuple.

# Notes
There is no generic fallback for this method. It should be implemented for each concrete
geometry type.
"""
function hessian(
    geometry::AbstractGeometry{manifold_dim, image_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
) where {manifold_dim, image_dim, num_patches}
    throw(MethodError(hessian, (geometry, element_id, xi)))
end

# Methods to obtain coordinates of vertices, edges, etc. based on topological ids.
"""
    get_canonical_point(geometry::AbstractGeometry, local_vertex_id::Int)
    get_canonical_point(
        ::Type{T}, geometry::AbstractGeometry, local_vertex_id::Int
    ) where {T}

Create a [`CartesianPoints`](@ref)-object with the canonical coordinate for the vertex with
`local_vertex_id`. Uses `eltype(geometry)` to determine the output coordinate type if none
is provided.
"""
function get_canonical_points(
    geometry::AbstractGeometry,
    local_object_id::Int,
    geometric_dim::Int,
    points_per_dim::Int=5,
)
    return get_canonical_points(
        eltype(geometry), geometry, local_object_id, geometric_dim, points_per_dim
    )
end
function get_canonical_points(
    ::Type{T},
    geometry::AbstractGeometry,
    local_object_id::Int,
    geometric_dim::Int,
    points_per_dim::Int=5,
) where {T}
    position = Topology.id2position(
        get_manifold_dim(geometry), geometric_dim, local_object_id
    )
    return Points.CartesianPoints(
        ntuple(get_manifold_dim(geometry)) do i
            if position[i] == 1
                return LinRange(one(T), one(T), 1)
            elseif position[i] == -1
                return LinRange(zero(T), zero(T), 1)
            else
                return LinRange(zero(T), one(T), points_per_dim)
            end
        end,
    )
end

# only for edges
# function get_tangent_vector(
#     ::Type{T}, geometry::AbstractGeometry, local_object_id::Int, geometric_dim::Int
# ) where {T}
#     position = Topology.id2position(
#         get_manifold_dim(geometry), geometric_dim, local_object_id
#     )
#     return ntuple(get_manifold_dim(geometry)) do i
#         if position[i] == 1
#             return zero(T)
#         elseif position[i] == -1
#             return zero(T)
#         else
#             return one(T)
#         end
#     end
# end

"""
    get_elements(geometry::AbstractGeometry, patch_id, local_object_id, geometric_dim)

Compute the elements located on the given topological object `local_object_id` (vertices
`geometric_dim`=0, edges `geometric_dim`=1, surfaces `geometric_dim`=2, etc.) on patch
`patch_id`.
"""
function get_elements(geometry::AbstractGeometry, patch_id, local_object_id, geometric_dim)
    throw(MethodError(get_elements, (geometry, patch_id, local_object_id, geometric_dim)))
end

"""
    get_vertex_coordinates(geometry::AbstractGeometry)
    get_vertex_coordinates(geometry::AbstractGeometry, patch_id::Int)
    get_vertex_coordinates(geometry::AbstractGeometry, patch_id::Int, local_vertex_id::Int)
    get_vertex_coordinates(::Type{VT}, geometry::AbstractGeometry) where {VT}
    get_vertex_coordinates(::Type{VT}, geometry::AbstractGeometry, patch_id::Int) where {VT}
    get_vertex_coordinates(
        ::Type{VT}, geometry::AbstractGeometry, patch_id::Int, local_vertex_id::Int
    ) where {VT}

Compute the coordinate of the patch-vertex. If only `geometry` is given, computes the
coordinates for all patch-vertices. If a `patch_id` is also given, computes the coordinates
for the patch-vertices of patch `patch_id`. If both `patch_id` and `local_vertex_id` are
also given, compute only the coordinate of the requested vertex.

Converts the output type of the coordinates to `VT` if given. `VT` defaults to
`NTuple{get_image_dim(geometry), eltype(geometry)}`.
"""
function get_vertex_coordinates(
    geometry::AbstractGeometry, patch_id::Int, local_vertex_id::Int
)
    return get_vertex_coordinates(
        NTuple{get_image_dim(geometry), eltype(geometry)},
        geometry,
        patch_id,
        local_vertex_id,
    )
end
function get_vertex_coordinates(
    ::Type{VT}, geometry::AbstractGeometry, patch_id::Int, local_vertex_id::Int
) where {VT}
    element_id = get_elements(geometry, patch_id, local_vertex_id, 0)[1]
    xi_vertex = get_canonical_points(eltype(VT), geometry, local_vertex_id, 0)
    coord = NTuple{get_image_dim(geometry), eltype(VT)}(
        vec(evaluate(geometry, element_id, xi_vertex))
    )
    return convert(VT, coord)
end
function get_vertex_coordinates(geometry::AbstractGeometry, patch_id::Int)
    return get_vertex_coordinates(
        NTuple{get_image_dim(geometry), eltype(geometry)}, geometry, patch_id
    )
end
function get_vertex_coordinates(
    ::Type{VT}, geometry::AbstractGeometry, patch_id::Int
) where {VT}
    num_local_vertices = Topology.get_local_size(get_topology(geometry), 1)

    return ntuple(num_local_vertices) do local_vertex_id
        return get_vertex_coordinates(VT, geometry, patch_id, local_vertex_id)
    end
end
function get_vertex_coordinates(geometry::AbstractGeometry)
    return get_vertex_coordinates(
        NTuple{get_image_dim(geometry), eltype(geometry)}, geometry
    )
end
function get_vertex_coordinates(::Type{VT}, geometry::AbstractGeometry) where {VT}
    topology = get_topology(geometry)
    num_vertices = size(topology, 1)

    manifold_dim = get_manifold_dim(geometry)
    return [
        get_vertex_coordinates(
            VT,
            geometry,
            topology[1, manifold_dim + 1][vertex_id][1], # The patch_id of a support patch.
            Topology.get_local_id(
                topology, topology[1, manifold_dim + 1][vertex_id][1], vertex_id, 0
            ), # local vertex id
        ) for vertex_id in 1:num_vertices
    ]
end

"""
    get_edge_coordinates(geometry::AbstractGeometry)
    get_edge_coordinates(geometry::AbstractGeometry, patch_id::Int)
    get_edge_coordinates(geometry::AbstractGeometry, patch_id::Int, local_edge_id::Int)
    get_edge_coordinates(::Type{VT}, geometry::AbstractGeometry) where {VT}
    get_edge_coordinates(::Type{VT}, geometry::AbstractGeometry, patch_id::Int) where {VT}
    get_edge_coordinates(
        ::Type{VT}, geometry::AbstractGeometry, patch_id::Int, local_edge_id::Int
    ) where {VT}

Compute the coordinates of the start and end vertex of the patch-edges. If only `geometry`
is given, computes the coordinates for all patch-edges. If only a `patch_id` is given,
computes the coordinates for the patch-edges of patch `patch_id`. If both `patch_id` and
`local_edge_id` are given, compute only the coordinate of the requested edge.

Converts the output type of the coordinates to `VT` if given. `VT` defaults to
`NTuple{get_image_dim(geometry), eltype(geometry)}`.
"""
function get_edge_coordinates(geometry::AbstractGeometry, patch_id::Int, local_edge_id::Int)
    return get_edge_coordinates(
        NTuple{get_image_dim(geometry), eltype(geometry)}, geometry, patch_id, local_edge_id
    )
end
function get_edge_coordinates(
    ::Type{VT}, geometry::AbstractGeometry, patch_id::Int, local_edge_id::Int
) where {VT}
    topology = get_topology(geometry)
    if get_image_dim(geometry) == 1
        # The global and local ids are the same, and the edges are the patches.
        global_edge_id = patch_id
    else
        global_edge_id = Topology.get_global_id(topology, patch_id, abs(local_edge_id), 1)
    end
    global_vertices = topology[2, 1][abs(global_edge_id)]
    starting_vertex_coordinate = get_vertex_coordinates(
        VT,
        geometry,
        patch_id,
        Topology.get_local_id(topology, patch_id, global_vertices[1], 0),
    )
    final_vertex_coordinate = get_vertex_coordinates(
        VT,
        geometry,
        patch_id,
        Topology.get_local_id(topology, patch_id, global_vertices[2], 0),
    )
    return starting_vertex_coordinate, final_vertex_coordinate
end
function get_edge_coordinates(geometry::AbstractGeometry, patch_id::Int)
    return get_edge_coordinates(
        NTuple{get_image_dim(geometry), eltype(geometry)}, geometry, patch_id
    )
end
function get_edge_coordinates(
    ::Type{VT}, geometry::AbstractGeometry, patch_id::Int
) where {VT}
    num_local_edges = Topology.get_local_size(get_topology(geometry), 2)

    return ntuple(num_local_edges) do local_edge_id
        return get_edge_coordinates(VT, geometry, patch_id, local_edge_id)
    end
end
function get_edge_coordinates(geometry::AbstractGeometry)
    return get_edge_coordinates(NTuple{get_image_dim(geometry), eltype(geometry)}, geometry)
end
function get_edge_coordinates(::Type{VT}, geometry::AbstractGeometry) where {VT}
    topology = get_topology(geometry)
    num_edges = size(topology, 2)

    manifold_dim = get_manifold_dim(geometry)
    edge_coordinates = Vector{NTuple{2, VT}}(undef, num_edges)
    for edge_id in eachindex(edge_coordinates)
        # Get a patch_id of a patch on which this edge is supported. A patch is a
        # (manifold_dim+1)-dimensional geometric object. We can simply take the first patch
        # in this list, because the coordinate of the edge will not change.
        if get_image_dim(geometry) == 1
            # The edges are the patches, so no conversion needed
            patch_id = edge_id
            local_edge_id = 1
        else
            patch_id = topology[2, manifold_dim + 1][edge_id][1]
            local_edge_id = abs(Topology.get_local_id(topology, patch_id, edge_id, 1))
        end
        edge_coordinates[edge_id] = get_edge_coordinates(
            VT, geometry, patch_id, local_edge_id
        )
    end

    return edge_coordinates
end

include("CartesianGeometry.jl")
include("DiscreteGeometry.jl")
include("MappedGeometry.jl")
include("TensorProductGeometry.jl")
include("EvaluationMask/EvaluationMask.jl")
include("MaskedGeometry.jl")
include("UnstructuredGeometry.jl")
include("Metric.jl")
include("SkeletonGeometry.jl")

include("./GeometryConversions.jl")

# helper functions for convenience
include("./GeometryHelpers.jl")

end

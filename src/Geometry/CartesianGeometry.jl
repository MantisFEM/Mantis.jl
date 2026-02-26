"""
    CartesianGeometry{manifold_dim, image_dim, num_patches, B, CI} <: AbstractGeometry{
        manifold_dim, image_dim, num_patches
    }

A structure representing a Cartesian grid geometry in `manifold_dim` dimensions. Can have
multiple patches, even though each patch is still a Cartesian grid. Note that the patches
are not required to have a matching grid.

# Fields
- `breakpoints::B`: A tuple of vectors defining the grid points in each dimension.
- `cart_num_elements::CI`: A (tuple of) `CartesianIndices` representing the indices of
    elements in the grid for each patch.

# Constructors
- `CartesianGeometry(
        breakpoints::B
    ) where {
        manifold_dim,
        num_patches,
        NT <: Number,
        B <: NTuple{num_patches, NTuple{manifold_dim, AbstractVector{NT}}},
    }`: General constructor.
- `CartesianGeometry(
        breakpoints::NTuple{manifold_dim, AbstractVector{NT}}
    ) where {manifold_dim, NT <: Number}`: Single-patch convenience constructor.
"""
struct CartesianGeometry{manifold_dim, image_dim, num_patches, B, CI, LI, T, NT} <:
       AbstractGeometry{manifold_dim, image_dim, num_patches}
    topology::T
    breakpoints::B
    cart_num_elements::CI
    lin_num_elements::LI

    function CartesianGeometry(
        breakpoints::B, topology::T
    ) where {
        manifold_dim,
        incidence_relations_dim,
        num_patches,
        NT <: Number,
        B <: NTuple{num_patches, NTuple{manifold_dim, AbstractVector{NT}}},
        T <: Topology.MeshTopology{manifold_dim, incidence_relations_dim, num_patches},
    }
        foreach(breakpoints) do patch_breakpoints
            unique_breakpoints = map(unique, patch_breakpoints)
            are_unique = map(isequal, patch_breakpoints, unique_breakpoints)
            if !all(are_unique)
                index = findfirst(x -> x == false, are_unique)
                throw(
                    ArgumentError(
                        LazyString(
                            "Breakpoints should be unique, but the breakpoints in ",
                            "direction ",
                            index,
                            " are ",
                            patch_breakpoints[index],
                            ".",
                        ),
                    ),
                )
            end
            sorted_breakpoints = map(sort, patch_breakpoints)
            are_sorted = map(isequal, patch_breakpoints, sorted_breakpoints)
            if !all(are_sorted)
                index = findfirst(x -> x == false, are_sorted)
                throw(
                    ArgumentError(
                        LazyString(
                            "Breakpoints should be stricly increasing, but the ",
                            "breakpoints in direction ",
                            index,
                            " are ",
                            patch_breakpoints[index],
                            ".",
                        ),
                    ),
                )
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
            B,
            typeof(cart_num_elements),
            typeof(lin_num_elements),
            T,
            NT,
        }(
            topology, breakpoints, cart_num_elements, lin_num_elements
        )
    end

    # Convenience constructor for single patch geometries.
    function CartesianGeometry(
        breakpoints::NTuple{manifold_dim, AbstractVector{NT}}
    ) where {manifold_dim, NT <: Number}
        topo = Topology.MeshTopology([collect(1:(2^manifold_dim))])
        return CartesianGeometry((breakpoints,), topo)
    end

    # Convenience constructor for 1D, single patch geometries.
    function CartesianGeometry(breakpoints::AbstractVector{NT}) where {NT <: Number}
        topo = Topology.MeshTopology([[1, 2]])
        return CartesianGeometry(((breakpoints,),), topo)
    end
end

# Get properties.
get_breakpoints(geometry::CartesianGeometry, patch_id::Int=1) =
    geometry.breakpoints[patch_id]
get_breakpoints_per_dim(geometry::CartesianGeometry, patch_id::Int=1, dim::Int=1) =
    geometry.breakpoints[patch_id][dim]
get_breakpoint(geometry::CartesianGeometry, patch_id::Int=1, dim::Int=1, point::Int=1) =
    geometry.breakpoints[patch_id][dim][point]
get_topology(geometry::CartesianGeometry) = geometry.topology
function get_number_type(
    geometry::CartesianGeometry{manifold_dim, image_dim, num_patches, B, CI, LI, T, NT}
) where {manifold_dim, image_dim, num_patches, B, CI, LI, T, NT}
    return NT
end

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

function get_vertex_coordinate(
    geometry::CartesianGeometry, patch_id::Int, local_vertex_id::Int
)
    return get_vertex_coordinate(
        NTuple{get_image_dim(geometry), get_number_type(geometry)},
        geometry,
        patch_id,
        local_vertex_id,
    )
end
function get_vertex_coordinate(
    ::Type{VT}, geometry::CartesianGeometry, patch_id::Int, local_vertex_id::Int
) where {VT}
    image_dim = get_image_dim(geometry)
    coord = ntuple(image_dim) do dim
        position = Topology.id2position(image_dim, 0, local_vertex_id)
        if position[dim] == -1
            point_id = 1
        else # position[dim] == 1
            cart_num_elements = get_cart_num_elements(geometry, patch_id)
            point_id = Tuple(last(cart_num_elements))[dim] + 1
        end
        return get_breakpoint(geometry, patch_id, dim, point_id)
    end
    return convert(VT, coord)
end

function get_vertex_coordinates(geometry::CartesianGeometry, patch_id::Int)
    return get_vertex_coordinates(
        NTuple{get_image_dim(geometry), get_number_type(geometry)}, geometry, patch_id
    )
end
function get_vertex_coordinates(
    ::Type{VT}, geometry::CartesianGeometry, patch_id::Int
) where {VT}
    num_local_vertices = Topology.get_local_size(get_topology(geometry), 1)

    return ntuple(num_local_vertices) do local_vertex_id
        return get_vertex_coordinate(VT, geometry, patch_id, local_vertex_id)
    end
end

function get_vertex_coordinates(geometry::CartesianGeometry)
    return get_vertex_coordinates(
        NTuple{get_image_dim(geometry), get_number_type(geometry)}, geometry
    )
end
function get_vertex_coordinates(::Type{VT}, geometry::CartesianGeometry) where {VT}
    topology = get_topology(geometry)
    num_vertices = size(topology, 1)

    manifold_dim = get_manifold_dim(geometry)
    return [
        get_vertex_coordinate(
            VT,
            geometry,
            topology[1, manifold_dim + 1][vertex_id][1], # The patch_id of a support patch.
            findfirst(
                isequal(vertex_id),
                topology[manifold_dim + 1, 1][topology[1, manifold_dim + 1][vertex_id][1]],
            ),
        ) for vertex_id in 1:num_vertices
    ]
end

function get_edge_coordinates(
    geometry::CartesianGeometry, patch_id::Int, local_edge_id::Int
)
    return get_edge_coordinates(
        NTuple{get_image_dim(geometry), get_number_type(geometry)},
        geometry,
        patch_id,
        local_edge_id,
    )
end
function get_edge_coordinates(
    ::Type{VT}, geometry::CartesianGeometry, patch_id::Int, local_edge_id::Int
) where {VT}
    topology = get_topology(geometry)
    if get_image_dim(geometry) == 1
        # The global and local ids are the same, and the edges are the patches.
        global_edge_id = patch_id
    else
        global_edge_id = Topology.get_global_id(topology, patch_id, local_edge_id, 2)
    end
    global_vertices = topology[2, 1][global_edge_id]
    starting_vertex_coordinate = get_vertex_coordinate(
        VT,
        geometry,
        patch_id,
        abs(Topology.get_local_id(topology, patch_id, global_vertices[1], vertex_dim)),
    )
    final_vertex_coordinate = get_vertex_coordinate(
        VT,
        geometry,
        patch_id,
        abs(Topology.get_local_id(topology, patch_id, global_vertices[2], vertex_dim)),
    )
    return starting_vertex_coordinate, final_vertex_coordinate
end

function get_edge_coordinates(geometry::CartesianGeometry, patch_id::Int)
    return get_edge_coordinates(
        NTuple{get_image_dim(geometry), get_number_type(geometry)}, geometry, patch_id
    )
end
function get_edge_coordinates(
    ::Type{VT}, geometry::CartesianGeometry, patch_id::Int
) where {VT}
    num_local_edges = Topology.get_local_size(get_topology(geometry), 2)

    return ntuple(num_local_edges) do local_edge_id
        return get_edge_coordinates(VT, geometry, patch_id, local_edge_id)
    end
end

function get_edge_coordinates(geometry::CartesianGeometry)
    return get_edge_coordinates(
        NTuple{get_image_dim(geometry), get_number_type(geometry)}, geometry
    )
end
function get_edge_coordinates(::Type{VT}, geometry::CartesianGeometry) where {VT}
    topology = get_topology(geometry)
    num_edges = size(topology, 2)

    manifold_dim = get_manifold_dim(geometry)
    edge_coordinates = Vector{NTuple{2, VT}}(undef, num_edges)
    edge_dim = 1  # the geometric dimension of the edge is 1
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
            local_edge_id = Topology.get_local_id(topology, patch_id, edge_id, 2)
        end
        edge_coordinates[edge_id] = get_edge_coordinates(geometry, patch_id, local_edge_id)
    end

    return edge_coordinates
end

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

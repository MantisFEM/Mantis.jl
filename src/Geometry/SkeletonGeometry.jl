"""
    SkeletonGeometry{manifold_dim, image_dim, num_patches, T, PG} <:
           AbstractGeometry{manifold_dim, image_dim, num_patches}

Contains the geometry of the skeleton of its `parent_geometry`, as well as the `topology`
of the skeleton.

# Fields
- `topology::T`: Skeleton topology, see [`SkeletonTopology`](@ref)
- `parent_geometry::PG`: The provided geometry of which to obtain the skeleton.

# Constructors
- `SkeletonGeometry(
        parent_geometry::PG
    ) where {
        parent_manifold_dim,
        image_dim,
        parent_num_patches,
        PG <: AbstractGeometry{parent_manifold_dim, image_dim, parent_num_patches},
    }`: General constructor.
"""
struct SkeletonGeometry{manifold_dim, image_dim, num_patches, T, PG} <:
       AbstractGeometry{manifold_dim, image_dim, num_patches}
    topology::T
    parent_geometry::PG

    function SkeletonGeometry(
        parent_geometry::PG
    ) where {
        parent_manifold_dim,
        image_dim,
        parent_num_patches,
        PG <: AbstractGeometry{parent_manifold_dim, image_dim, parent_num_patches},
    }
        manifold_dim = parent_manifold_dim - 1
        topology = Topology.SkeletonTopology(Geometry.get_topology(parent_geometry)) 
        num_patches = Topology.get_num_patches(topology)  # TODO: this is type unstable, either we
                                                          # use vectors or we must add 
                                                          # (n_faces, n_patches)
                                                          # to the type parameters instead of only n_patches

        return new{manifold_dim, image_dim, num_patches, typeof(topology), PG}(
            topology, parent_geometry
        )
    end
end

function get_topology(geometry::SG) where {
    SG <: SkeletonGeometry
}
    return geometry.topology
end

function get_parent_geometry(geometry::SG) where {
    SG <: SkeletonGeometry
}
    return geometry.parent_geometry
end

function get_num_elements(geometry::SkeletonGeometry)
    return sum(get_num_elements_per_patch(geometry))
end

function get_num_elements(
    geometry::SkeletonGeometry{manifold_dim, image_dim, num_patches},
    patch_id::Int
) where {
    manifold_dim,
    image_dim,
    num_patches
}
    # Get the current patch parents, i.e., the patches in the parent geometry that contain
    # the current patch. Since all parents must have the same elements on the current patch 
    # of the skeleton, we can just pick any parent patch. We pick the first one.
    parent_patch = Topology.get_patch_parents(get_topology(geometry), patch_id)[:, 1]

    # Now we get the number of elements on the local geometric object of dimension
    # patch_dim that coincides with our current patch, since the current patch must have the
    # same element distribution.
    parent_patch_id = parent_patch[1]
    local_patch_id = parent_patch[2]

    return get_num_elements(get_parent_geometry(geometry), parent_patch_id;
        local_object_id=local_patch_id, 
        geometric_dim=manifold_dim
    )
end

function get_parent_elements(
    geometry::SkeletonGeometry{patch_dim, image_dim, num_patches},
    patch_id::Int,
    local_element_id::Int
) where {
    patch_dim,
    image_dim,
    num_patches
}
    # Get the parents of the patch, i.e., the patches in the parent geometry
    # where the current patch with patch_id lies
    patch_parents = Topology.get_patch_parents(get_topology(geometry), patch_id)
    num_parents = size(patch_parents, 2)

    parent_elements_ids = Vector{Int}(undef, num_parents)

    for k_parent in 1:num_parents
        parent_elements_ids[k_parent] = get_elements(
            get_parent_geometry(geometry), 
            patch_parents[1, k_parent], 
            patch_parents[2, k_parent], 
            patch_dim;
            rotation=patch_parents[3, k_parent],
            orientation=patch_parents[4, k_parent]
        )[local_element_id]
    end
    return parent_elements_ids, patch_parents
end

function get_parent_elements(
    geometry::SkeletonGeometry{patch_dim, image_dim, num_patches},
    element_id::Int
) where {
    patch_dim,
    image_dim,
    num_patches
}
    patch_id, local_element_id = get_patch_and_local_element_id(geometry, element_id)

    return get_parent_elements(geometry, patch_id, local_element_id)
end

function get_num_elements_per_patch(
    geometry::SkeletonGeometry{manifold_dim, image_dim, num_patches}
) where {
    manifold_dim,
    image_dim,
    num_patches
}
    return ntuple(num_patches) do patch_id
        get_num_elements(geometry, patch_id)
    end 
end

# function get_element_vertices(
#     geometry::AbstractGeometry{manifold_dim, image_dim, num_patches}, element_id::Int
# ) where {manifold_dim, image_dim, num_patches}
#     throw(MethodError(get_element_vertices, (geometry, element_id)))
# end



"""
    evaluate(
        geometry::SkeletonGeometry{manifold_dim, image_dim, num_patches},
        element_id::Int,
        xi::Points.AbstractPoints{manifold_dim},
    ) where {manifold_dim, image_dim, num_patches}

Computes the coordinates of the physical points, given the canonical points `xi`, on
the element identified by `element_id` of a given `geometry`.

# Arguments
- `geometry::SkeletonGeometry{manifold_dim, image_dim, num_patches}`: The geometry being evaluated.
- `element_id::Int`: The global element id.
- `xi::Points.AbstractPoints{manifold_dim}`: The points in the canonical domain at which to
    evaluate the geometry.

# Returns
- `::Matrix{Float64}`: The physical coordinates of `xi` on element `element_id`. The
    size of the matrix is `(num_eval_points, image_dim)`.
"""
function evaluate(
    geometry::SkeletonGeometry{2, image_dim, num_patches},
    element_id::Int,
    xi::Points.AbstractPoints{2},
) where {image_dim, num_patches}
    # Identify the parent elements (i.e., in the parent geometry) so that we 
    # know which element to evaluate. Since for evaluation all parents are the same
    # we simply choose the first one. Additionally we also collect the rotation
    # information that allow us to know how we should translate evaluation of 
    # the skeleton element (the current element) into evaluation of the parent.
    parent_elements_ids, patch_parents = get_parent_elements(geometry, element_id)

    # Evaluate the parent geometry at the location of the skeleton element
    xi_parent = skeleton_element_to_parent_element_coords(
        xi, patch_parents[2, 1], patch_parents[3, 1], patch_parents[4, 1])
    
    parent_geometry = get_parent_geometry(geometry)

    eval_geometry = evaluate(parent_geometry, parent_elements_ids[1], xi_parent)

    return eval_geometry
end

function skeleton_element_to_parent_element_coords(
    skeleton_points::P, 
    local_geometric_object::Int,
    rotation::Int, 
    orientation::Int,
) where {P <: Points.CartesianPoints{2}}
    manifold_dim = 2

    println(rotation)
    println(orientation)
    println("\n\n\n")

    # Get the constituent points
    skeleton_σ_τ = Points.get_constituent_points(skeleton_points)
   
    # Step 1: Apply orientation
    iteration_order_rotation = collect(1:manifold_dim)
    if orientation == -1
        iteration_order_rotation[1], iteration_order_rotation[2] = iteration_order_rotation[2], iteration_order_rotation[1]
        σ_τ = (skeleton_σ_τ[2], skeleton_σ_τ[1])
    elseif orientation == 1
        σ_τ = skeleton_σ_τ
    else
        throw(ArgumentError("orientation must be 1 or -1, got $orientation"))
    end


    # Step 2: Apply rotation
    σ_τ = if rotation == 0
        σ_τ
        # No iteration order swap needed
    elseif rotation == 1
        (1.0 .- σ_τ[2], σ_τ[1])
        iteration_order_rotation[1], iteration_order_rotation[2] = iteration_order_rotation[2], iteration_order_rotation[1] 
    elseif rotation == 2
        (1.0 .- σ_τ[1], 1.0 .- σ_τ[2])
        # No iteration order swap needed
    elseif rotation == 3
        (σ_τ[2], 1.0 .- σ_τ[1])
        iteration_order_rotation[1], iteration_order_rotation[2] = iteration_order_rotation[2], iteration_order_rotation[1]
    else
        throw(ArgumentError("rotation must be 0, 1, 2, or 3, got $orientation"))
    end
    display(σ_τ)
    # Step 3: Map 2D surface coords to 3D element coords based on local face
    # The logic here is the following
    # position contains for example (0, 1, 0), this means that in the parent element
    # the skeleton element spans the ξ and ζ axis at the position η = 1.
    # Hence, a point (σ, τ) in the skeleton element (after rotation and orientation taken
    # into account) will be evaluated as (σ, 1, τ) in the parent element.
    position = Topology.id2position(manifold_dim + 1, manifold_dim, local_geometric_object)
    
    constituent_points = Vector{eltype(σ_τ)}(undef, manifold_dim + 1)

    iteration_order = zeros(Int, manifold_dim + 1)  # allocate the memory to store the final iteration order
    skeleton_position_id = 1
    for position_id in 1:(manifold_dim + 1)
        if position[position_id] == 0
            iteration_order[position_id] = iteration_order_rotation[skeleton_position_id]
            constituent_points[position_id] = σ_τ[skeleton_position_id]
            skeleton_position_id += 1
        else 
            iteration_order[position_id] = 3
            # Rescale to 0 or 1 instead of -1 or 1 as in position
            constituent_points[position_id] = 
                single_coordinate_of_type(
                    (position[position_id] + 1.0)/2.0,
                    σ_τ[1]
                )
        end
    end
    
    # Step 4: Construct new Points of the same type, but evaluatable in parent geometry
    return Points.CartesianPoints(constituent_points...; iteration_order=Tuple(iteration_order))
end


function single_coordinate_of_type(
    coordinate::F, object_of_type::LinRange{F, Int}
) where {F<:AbstractFloat}
    return LinRange(coordinate, coordinate, 1)
end

function single_coordinate_of_type(
    coordinate::F, object_of_type::Vector{F}
) where {F<:Real}
    return [coordinate]
end


############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    HierarchicalGeometry{manifold_dim, image_dim, num_patches, H} <:
    AbstractGeometry{manifold_dim, image_dim, num_patches}

Represents a hierarchy of nested `geometries`. Which elements are active or inactive is
determined by `hierarchy`.

See also [`Hierarchical.Hierarchy`](@ref).

# Fields
- `hierarchy::H`: A hierarchy of geometries, determining which elements are active form each
    level, and how they are related.
"""
struct HierarchicalGeometry{manifold_dim, image_dim, num_patches, H} <:
       AbstractGeometry{manifold_dim, image_dim, num_patches}
    hierarchy::H

    function HierarchicalGeometry(hierarchy::H) where {H <: Hierarchical.NestedHierarchy}
        geometries = Hierarchical.get_sets(hierarchy)
        for geo in geometries
            if !(isa(geo, AbstractGeometry))
                throw(
                    ArgumentError(
                        "Hierarchy must contain only geometries. Got type $(typeof(geo)) " *
                        "in $(map(g -> Base.typename(typeof(g)).wrapper, geometries)).",
                    ),
                )
            end
        end

        first_geo = first(geometries)
        manifold_dim = get_manifold_dim(first_geo)
        image_dim = get_image_dim(first_geo)
        num_patches = get_num_patches(first_geo)
        for geo in Base.tail(geometries)
            md = get_manifold_dim(geo)
            id = get_image_dim(geo)
            np = get_num_patches(geo)
            if md != manifold_dim
                throw(
                    ArgumentError(
                        "Expect manifold dimension $(manifold_dim). " *
                        "Got $(md) for $(typeof(geo)).",
                    ),
                )
            end

            if id != image_dim
                throw(
                    ArgumentError(
                        "Expect image dimension $(image_dim). Got $(id) for $(typeof(geo))."
                    ),
                )
            end

            if np != num_patches
                throw(
                    ArgumentError(
                        "Expect $(num_patches) patches. Got $(np) for $(typeof(geo))."
                    ),
                )
            end
        end

        return new{manifold_dim, image_dim, num_patches, H}(hierarchy)
    end
end

function HierarchicalGeometry(active_info::Hierarchical.ActiveInfo, scalings)
    return HierarchicalGeometry(Hierarchical.NestedHierarchy(active_info, scalings))
end

############################################################################################
#                                         Getters                                          #
############################################################################################

"""
    get_hierarchy(geometry::HierarchicalGeometry)

Return the `hierarchy` field in `geometry`.
"""
get_hierarchy(geometry::HierarchicalGeometry) = geometry.hierarchy

"""
	get_geometries(geometry::HierarchicalGeometry)

Returns the tuple of level-wise geometries defining the hierarchical `geometry`.
"""
function get_geometries(geometry::HierarchicalGeometry)
    return Hierarchical.get_sets(get_hierarchy(geometry))
end

"""
	get_active_elements(geometry::HierarchicalGeometry)

Returns the `Hierarchical.ActiveInfo` object defining the active elements of the
hierarchical `geometry`.

See also [`Hierarchical.ActiveInfo`](@ref).
"""
function get_active_elements(geometry::HierarchicalGeometry)
    return Hierarchical.get_active_info(get_hierarchy(geometry))
end

"""
	get_level_geometry(geometry::HierarchicalGeometry, level::Int)

Returns the full geometry associated with the given `level`.
"""
function get_level_geometry(geometry::HierarchicalGeometry, level::Int)
    return get_geometries(geometry)[level]
end

"""
	get_num_levels(geometry::HierarchicalGeometry)

Returns the number of levels of the hierarchical `geometry`.
"""
function get_num_levels(geometry::HierarchicalGeometry)
    return Hierarchical.get_num_levels(get_hierarchy(geometry))
end

function get_num_elements(geometry::HierarchicalGeometry)
    return Hierarchical.get_num_objects(get_hierarchy(geometry))
end

function get_element_measure(geometry::HierarchicalGeometry, element_id::Int)
    level, level_id = convert_to_level_and_level_id(geometry, element_id)

    return get_element_measure(get_level_geometry(geometry, level), level_id)
end

function get_element_lengths(geometry::HierarchicalGeometry, element_id::Int)
    level, level_id = convert_to_level_and_level_id(geometry, element_id)

    return get_element_lengths(get_level_geometry(geometry, level), level_id)
end

function get_element_vertices(geometry::HierarchicalGeometry, element_id::Int)
    level, level_id = convert_to_level_and_level_id(geometry, element_id)

    return get_element_vertices(get_level_geometry(geometry, level), level_id)
end

function get_ancestors(geometry::HierarchicalGeometry, element_id::Int)
    level, current = convert_to_level_and_level_id(geometry, element_id)
    ancestors = Vector{Int}(undef, level-1)
    hierarchy = get_hierarchy(geometry)
    for l in level:-1:2
        scaling = Hierarchical.get_scaling(hierarchy, l-1)
        # We use `first` since each element in a hier geometry only has 1 parent.
        next = first(Hierarchical.get_parents(scaling, current))
        ancestors[l - 1] = next
        current = next
    end

    return ancestors
end

function get_nested_active(geometry::HierarchicalGeometry, level::Int, level_id::Int)
    hierarchy = get_hierarchy(geometry)
    # If the element is active, we return its hierarchical element id.
    level_set = Hierarchical.get_level_set(hierarchy, level)
    if level_id in level_set
        return [Hierarchical.convert_to_hier_id(hierarchy, level, level_id)]
    end

    # Else, we recursively look for active children.
    active_children = Int[]
    curr_inactive = [level_id]
    next_inactive = Int[]
    l = level
    while !isempty(curr_inactive)
        # For each parent, we check its children.
        next_l_set = Hierarchical.get_level_set(hierarchy, l+1)
        for p in curr_inactive,
            c in Hierarchical.get_children(Hierarchical.get_scaling(hierarchy, l), p)

            if c in next_l_set
                # Child is active.
                push!(active_children, Hierarchical.convert_to_hier_id(hierarchy, l+1, c))
            else
                # Child is inactive.
                push!(next_inactive, c)
            end
        end

        # Swap inactive vectors, so the curr_inactive become the previous inactive children.
        curr_inactive, next_inactive = next_inactive, curr_inactive
        # Clear next_inactive to get a clean slate.
        empty!(next_inactive)
        # Increment to next level
        l += 1
    end

    return active_children
end

############################################################################################
#                                        Conversion                                        #
############################################################################################

"""
	convert_to_level_and_level_id(geometry::HierarchicalGeometry, element_id::Int)

Returns the `level` and `level_id` of the `element_id` in hierarchical indexing.

See also [`Hierarchical.convert_to_level_and_level_id`](@ref).
"""
function convert_to_level_and_level_id(geometry::HierarchicalGeometry, element_id::Int)
    level, level_id = Hierarchical.convert_to_level_and_level_id(
        get_hierarchy(geometry), element_id
    )

    return level, level_id
end

############################################################################################
#                                        Evaluation                                        #
############################################################################################

function evaluate(
    geometry::HierarchicalGeometry, element_id::Int, xi::Points.AbstractPoints
)
    level, level_id = convert_to_level_and_level_id(geometry, element_id)

    return evaluate(get_level_geometry(geometry, level), level_id, xi)
end

function jacobian(
    geometry::HierarchicalGeometry, element_id::Int, xi::Points.AbstractPoints
)
    level, level_id = convert_to_level_and_level_id(geometry, element_id)

    return jacobian(get_level_geometry(geometry, level), level_id, xi)
end

function hessian(geometry::HierarchicalGeometry, element_id::Int, xi::Points.AbstractPoints)
    level, level_id = convert_to_level_and_level_id(geometry, element_id)

    return hessian(get_level_geometry(geometry, level), level_id, xi)
end

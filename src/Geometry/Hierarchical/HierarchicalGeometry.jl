############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    HierarchicalGeometry{manifold_dim, image_dim, num_patches, H} <:
    AbstractGeometry{manifold_dim, image_dim, num_patches}

Represents a tree-like hierarchy of geometries.

See also [`Hierarchical.Hierarchy`](@ref) and [`TreeHierarchy`](@ref).

# Fields
- `hierarchy::H`: A hierarchy of geometries, determining which elements are active form each
    level, and how they are related.

# Outer Constructors
- [`HierarchicalGeometry(active_info::Hierarchical.ActiveInfo, scalings)`](@ref)
"""
struct HierarchicalGeometry{manifold_dim, image_dim, num_patches, H} <:
       AbstractGeometry{manifold_dim, image_dim, num_patches}
    hierarchy::H

    function HierarchicalGeometry(hierarchy::H) where {H <: Hierarchical.TreeHierarchy}
        geometries = Hierarchical.get_sets(hierarchy)
        for geo in geometries
            if !(isa(geo, AbstractGeometry))
                return throw(
                    ArgumentError(
                        LazyString(
                            "Hierarchy must contain only geometries. ",
                            "Got type ",
                            typeof(geo),
                            " in ",
                            map(g -> Base.typename(typeof(g)).wrapper, geometries),
                        ),
                    ),
                )
            end
        end

        # Extract parametres from the first geometry
        first_geo = first(geometries)
        manifold_dim = get_manifold_dim(first_geo)
        image_dim = get_image_dim(first_geo)
        num_patches = get_num_patches(first_geo)
        # Check if the remaining geometries match
        for geo in Base.tail(geometries)
            md = get_manifold_dim(geo)
            id = get_image_dim(geo)
            np = get_num_patches(geo)
            if md != manifold_dim
                throw(
                    ArgumentError(
                        LazyString(
                            "Expect manifold dimension ",
                            manifold_dim,
                            ". Got ",
                            md,
                            " for ",
                            typeof(geo),
                            ".",
                        ),
                    ),
                )
            end

            if id != image_dim
                throw(
                    ArgumentError(
                        LazyString(
                            "Expect image dimension ",
                            image_dim,
                            ". Got ",
                            id,
                            " for ",
                            typeof(geo),
                            ".",
                        ),
                    ),
                )
            end

            if np != num_patches
                throw(
                    ArgumentError(
                        LazyString(
                            "Expect ",
                            num_patches,
                            " patches. Got ",
                            np,
                            " for ",
                            typeof(geo),
                            ".",
                        ),
                    ),
                )
            end
        end

        return new{manifold_dim, image_dim, num_patches, H}(hierarchy)
    end
end

"""
    HierarchicalGeometry(active_info::Hierarchical.ActiveInfo, scalings)

Helper constructor from active element information and scalings.
"""
function HierarchicalGeometry(active_info::Hierarchical.ActiveInfo, scalings)
    return HierarchicalGeometry(Hierarchical.TreeHierarchy(active_info, scalings))
end

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

Returns the `Hierarchical.ActiveInfo` defining the active elements of the hierarchical
`geometry`.

See also [`Hierarchical.ActiveInfo`](@ref).
"""
function get_active_elements(geometry::HierarchicalGeometry)
    return Hierarchical.get_active_info(get_hierarchy(geometry))
end

"""
	get_level_geometry(geometry::HierarchicalGeometry, level::Int)

Returns the geometry from the given `level`.
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

"""
    get_ancestors(geometry::HierarchicalGeometry, element_id::Int)

Return the all the successive parents, or ancestors, starting from the level before that of
the give element up to the first level.

!!!note
    Will return an empty vector if the element is from the first level.
"""
function get_ancestors(geometry::HierarchicalGeometry, element_id::Int)
    # The element at the given level is the child from which we will iterate backwards over
    # the levels
    level, child = convert_to_level_and_level_id(geometry, element_id)
    # Empty if level=1
    ancestors = Vector{Int}(undef, level-1)
    hierarchy = get_hierarchy(geometry)
    for l in level:-1:2
        scaling = Hierarchical.get_scaling(hierarchy, l-1)
        # We use `first` since each element in a hierarchial geometry only has 1 parent.
        parent = first(Hierarchical.get_parents(scaling, child))
        ancestors[l - 1] = parent
        # current parent is the next child since we iterate backwards over the levels
        child = parent
    end

    return ancestors
end

"""
    get_active_descendants(geometry::HierarchicalGeometry, level::Int, level_id::Int)

Return all the _active_ successive children, or descendants, of the element indexed by
`level` and `level_id`, including the current `level`.

!!!note
    Unlike most hierarchical methods, `level` and `level_id` is not assumed to refer to an
    active element.
"""
function get_active_descendants(geometry::HierarchicalGeometry, level::Int, level_id::Int)
    hierarchy = get_hierarchy(geometry)
    # If the element is active, we return its hierarchical element id.
    level_set = Hierarchical.get_level_set(hierarchy, level)
    if level_id in level_set
        return [Hierarchical.convert_to_hier_id(hierarchy, level, level_id)]
    end

    # Else, we walk all the levels looking for the active children of any inactive parents.
    active_children = Int[]
    inactive_parents = [level_id]
    inactive_children = Int[]
    l = level
    while !isempty(inactive_parents)
        level_scaling = Hierarchical.get_scaling(hierarchy, l)
        # For each parent, we check its children to determine if they are active or not.
        children_set = Hierarchical.get_level_set(hierarchy, l+1)
        for parent in inactive_parents,
            child in Hierarchical.get_children(level_scaling, parent)

            if child in children_set # Child is active.
                push!(
                    active_children, Hierarchical.convert_to_hier_id(hierarchy, l+1, child)
                )
            else # Child is inactive.
                push!(inactive_children, child)
            end
        end

        # Swap inactive vectors, so the current children become the next parents.
        inactive_parents, inactive_children = inactive_children, inactive_parents
        # Clear the children to get a clean slate for the next level
        empty!(inactive_children)
        # We have check all the inactive_parents of the current level, so we move on
        l += 1
    end

    return active_children
end

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

############################################################################################
#                                          Update                                          #
############################################################################################

"""
    refine(
        geometry::HierarchicalGeometry,
        level::Int,
        remove::Vector{Int},
        scaling::Hierarchical.AbstractScaling,
    )

See [`Hierarchical.refine`](@ref).
"""
function refine(
    geometry::HierarchicalGeometry,
    level::Int,
    remove::Vector{Int},
    scaling::Hierarchical.AbstractScaling,
)
    hierarchy = Hierarchical.refine(get_hierarchy(geometry), level, remove, scaling)

    return HierarchicalGeometry(hierarchy)
end

"""
    update(
        geometry::HierarchicalGeometry,
        level::Int,
        remove::Vector{Int},
        add::Vector{Int},
        scaling::Hierarchical.AbstractScaling,
    )

See [`Hierarchical.update`](@ref).
"""
function update(
    geometry::HierarchicalGeometry,
    level::Int,
    remove::Vector{Int},
    add::Vector{Int},
    scaling::Hierarchical.AbstractScaling,
)
    hierarchy = Hierarchical.update(get_hierarchy(geometry), level, remove, add, scaling)

    return HierarchicalGeometry(hierarchy)
end

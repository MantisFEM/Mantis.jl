############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    HierarchicalGeometry{manifold_dim, image_dim, num_patches, G} <:
    AbstractGeometry{manifold_dim, image_dim, num_patches}

Represents a hierarchy of nested `geometries`. Which elements are active or inactive is
determined by `active_elements`.

See also [`Hierarchy.ActiveInfo`](@ref).

# Fiels

  - `geometries::G`: A tuple `G` such that `G <: NTuple{num_levels, AbstractGeometry}`, where
    `num_levels` is the number of levels in the hierarchy. As a consequence, `num_levels` is
    also the number of distinct level-wise geometries.
  - `active_elements::Hierarchy.ActiveInfo`: Information about which elements are active or
    inactive at each level and geometry.
"""
struct HierarchicalGeometry{manifold_dim, image_dim, num_patches, G} <:
       AbstractGeometry{manifold_dim, image_dim, num_patches}
    geometries::G
    active_elements::Hierarchy.ActiveInfo

    function HierarchicalGeometry(
        geometries::G, active_elements::Hierarchy.ActiveInfo
    ) where {
        manifold_dim,
        image_dim,
        num_patches,
        num_levels,
        G <: NTuple{num_levels, AbstractGeometry{manifold_dim, image_dim, num_patches}},
    }
        active_num_levels = Hierarchy.get_num_levels(active_elements)
        if active_num_levels != num_levels
            throw(
                ArgumentError(
                    "Number of geometries and number of levels in 'active_elements' " *
                    "must match. Got $(num_levels) and $(active_num_levels), respectively.",
                ),
            )
        end

        return new{manifold_dim, image_dim, num_patches, G}(geometries, active_elements)
    end
end

############################################################################################
#                                         Getters                                          #
############################################################################################

"""
    get_geometries(geometry::HierarchicalGeometry)

Returns the tuple of level-wise geometries defining the hierarchical `geometry`.
"""
function get_geometries(geometry::HierarchicalGeometry)
    return geometry.geometries
end

"""
    get_active_elements(geometry::HierarchicalGeometry)

Returns the `Hierarchy.ActiveInfo` object defining the active elements of the hierarchical
`geometry`.

See also [`Hierarchy.ActiveInfo`](@ref).
"""
function get_active_elements(geometry::HierarchicalGeometry)
    return geometry.active_elements
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
    return length(get_geometries(geometry))
end

function get_num_elements(geometry::HierarchicalGeometry)
    return Hierarchy.get_num_objects(get_active_elements(geometry))
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

############################################################################################
#                                        Conversion                                        #
############################################################################################

"""
    convert_to_level_and_level_id(geometry::HierarchicalGeometry, element_id::Int)

Returns the `level` and `level_id` of the `element_id` in hierarchical indexing.

See also [`Hierarchy.convert_to_level_and_level_id`](@ref).

# Arguments

  - `geometry::HierarchicalGeometry`: The hierarchical geometry.
  - `element_id::Int`: The hierarchical indexing of the given element.

# Returns

  - `level::Int`: The level to which the element given by `element_id` corresponds to.
  - `level_id::Int`: The index to which the element given by `element_id` corresponds to in
    geometry at the determined `level`.
"""
function convert_to_level_and_level_id(geometry::HierarchicalGeometry, element_id::Int)
    level, level_id = Hierarchy.convert_to_level_and_level_id(
        get_active_elements(geometry), element_id
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

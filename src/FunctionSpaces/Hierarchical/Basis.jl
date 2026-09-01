
############################################################################################
#                                         Options                                          #
############################################################################################

"""
    SelectionAlgorithm

Abstract type for algorithms that determine the active basis functions at each level of a
hierarchical function space.
"""
abstract type SelectionAlgorithm end

"""
    SelectionStandard

Select active functions by testing their supports against the active domains at each level.
"""
abstract type SelectionStandard <: SelectionAlgorithm end

"""
    SelectionSimple

Select active functions by propagating children of inactive functions through the hierarchy.
"""
abstract type SelectionSimple <: SelectionAlgorithm end

############################################################################################
#                            Methods for active basis selection                            #
############################################################################################

"""
    create_basis(
        geometry::Geometry.HierarchicalGeometry,
        scalings::NTuple{LS, Hierarchical.AbstractScaling},
        selection_type::Type{S},
    ) where {LS, S <: SelectionAlgorithm}

Construct a hierarchical basis for `geometry` from the function space `scalings`, using
`selection_type` to determine the active basis functions at each level.
"""
function create_basis(
    geometry::Geometry.HierarchicalGeometry,
    scalings::NTuple{LS, Hierarchical.AbstractScaling},
    ::Type{S},
) where {LS, S <: SelectionAlgorithm}
    return throw(MethodError(create_basis, (geometry, scalings, S)))
end

# Standard

function create_basis(
    geometry::Geometry.HierarchicalGeometry,
    scalings::NTuple{LS, Hierarchical.AbstractScaling},
    ::Type{SelectionStandard},
) where {LS}
    # TODO: assert an equal number of els per level
    L = LS + 1 # Number of levels. (1 Scaling has 2 levels.)
    geometry_hierarchy = Geometry.get_hierarchy(geometry)
    spaces = Hierarchical.get_sets(scalings)
    level_sets = [Set(1:get_num_basis(first(spaces)))]
    foreach(_ -> push!(level_sets, Set{Int}()), 2:L)
    # TODO: Remove space indexing
    for l in 1:(L - 1)
        # Remove inactive from current level.
        parent_space = spaces[l]
        parent_level_set = Hierarchical.get_level_set(geometry_hierarchy, l)
        for p in 1:get_num_basis(parent_space)
            supp = get_support(parent_space, p)
            # None of the elements in the support are active
            if isdisjoint(supp, parent_level_set)
                setdiff!(level_sets[l], p)
            end
        end

        # Add active from next level.
        child_space = spaces[l + 1]
        # This support-containment test is sufficient because the tree ids are gathered from
        # the tree below the refined domain rather than only its active elements.
        # This is why we use `get_tree_ids` and not `get_level_ids`.
        child_level_set = Hierarchical.get_tree_ids(geometry_hierarchy, l+1)
        for c in 1:get_num_basis(child_space)
            supp = get_support(child_space, c)
            if issubset(supp, child_level_set)
                push!(level_sets[l + 1], c)
            end
        end
    end
    
    active_info = Hierarchical.ActiveInfo(level_sets)

    return Hierarchical.Hierarchy(active_info, scalings...)
end

# Simple

function create_basis(
    geometry::Geometry.HierarchicalGeometry,
    scalings::NTuple{LS, Hierarchical.AbstractScaling},
    ::Type{SelectionSimple},
) where {LS}
    # Number of levels. (1 Scaling has 2 levels.)
    L = LS + 1
    geometry_hierarchy = Geometry.get_hierarchy(geometry)
    spaces = Hierarchical.get_sets(scalings)
    level_sets = [Set(1:get_num_basis(first(spaces)))]
    foreach(_ -> push!(level_sets, Set{Int}()), 2:L)
    # We create `level_ids` already to iterate faster
    level_ids = [collect(first(level_sets))]
    for (level, scaling) in enumerate(scalings)
        parent_level_set = Hierarchical.get_level_set(geometry_hierarchy, level)
        space = Hierarchical.get_parent(scaling)
        for p in level_ids[level]
            supp = get_support(space, p)
            # None of the elements in the support are active
            if isdisjoint(supp, parent_level_set)
                # Remove inactive from current level
                setdiff!(level_sets[level], p)
                setdiff!(level_ids[level], p)
                # Add children as active
                union!(level_sets[level + 1], Hierarchical.get_children(scaling, p))
            end
        end

        push!(level_ids, collect(level_sets[level + 1]))
    end

    # ids were created from sets, so no checks are needed
    active_info = Hierarchical.ActiveInfo(
        level_ids, level_sets; check_unique=false, check_sets=true
    )

    return Hierarchical.Hierarchy(active_info, scalings...)
end

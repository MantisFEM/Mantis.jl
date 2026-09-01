
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

"""
    _check_levels(geometry::Geometry.HierarchicalGeometry, spaces::Tuple)

Throw an `ArgumentError` if `geometry` and `spaces` do not have the same number of levels,
or if the geometry of a level is not the same object as the parametric geometry of the space
on that level.
"""
function _check_levels(geometry::Geometry.HierarchicalGeometry, spaces::Tuple)
    num_geometry_levels = Geometry.get_num_levels(geometry)
    num_space_levels = length(spaces)
    if num_geometry_levels != num_space_levels
        return throw(
            ArgumentError(
                LazyString(
                    "The geometry and the spaces should have the same number of levels. ",
                    "Got ",
                    num_geometry_levels,
                    " levels for the geometry and ",
                    num_space_levels,
                    " levels for the spaces.",
                ),
            ),
        )
    end

    for (l, space) in enumerate(spaces)
        # The elements of the hierarchical geometry index the elements of the spaces, so
        # both have to be built on the same level-wise geometries.
        if !(get_parametric_geometry(space) === Geometry.get_level_geometry(geometry, l))
            return throw(
                ArgumentError(
                    LazyString(
                        "The parametric geometry of the space and the geometry on level ",
                        l,
                        " should be the same object in memory.",
                    ),
                ),
            )
        end
    end

    return nothing
end

# Standard

function create_basis(
    geometry::Geometry.HierarchicalGeometry,
    scalings::NTuple{LS, Hierarchical.AbstractScaling},
    ::Type{SelectionStandard},
) where {LS}
    _check_levels(geometry, Hierarchical.get_sets(scalings))
    L = LS + 1 # Number of levels. (1 Scaling has 2 levels.)
    geometry_hierarchy = Geometry.get_hierarchy(geometry)
    level_sets = [Set(1:get_num_basis(Hierarchical.get_parent(first(scalings))))]
    foreach(_ -> push!(level_sets, Set{Int}()), 2:L)
    for (l, scaling) in enumerate(scalings)
        # Remove inactive from current level.
        parent_space = Hierarchical.get_parent(scaling)
        parent_level_set = Hierarchical.get_level_set(geometry_hierarchy, l)
        for p in 1:get_num_basis(parent_space)
            supp = get_support(parent_space, p)
            # None of the elements in the support are active
            if isdisjoint(supp, parent_level_set)
                setdiff!(level_sets[l], p)
            end
        end

        # Add active from next level.
        child_space = Hierarchical.get_child(scaling)
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
    _check_levels(geometry, Hierarchical.get_sets(scalings))
    # Number of levels. (1 Scaling has 2 levels.)
    L = LS + 1
    geometry_hierarchy = Geometry.get_hierarchy(geometry)
    spaces = Hierarchical.get_sets(scalings)
    level_sets = [Set(1:get_num_basis(first(spaces)))]
    foreach(_ -> push!(level_sets, Set{Int}()), 2:L)
    # We create `level_ids` already to iterate faster
    level_ids = [sort!(collect(first(level_sets)))]
    for (level, scaling) in enumerate(scalings)
        parent_level_set = Hierarchical.get_level_set(geometry_hierarchy, level)
        space = Hierarchical.get_parent(scaling)
        for p in copy(level_ids[level])
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

        push!(level_ids, sort!(collect(level_sets[level + 1])))
    end

    # ids were created from sets, so no checks are needed
    active_info = Hierarchical.ActiveInfo(
        level_ids, level_sets; check_unique=false, check_sets=true
    )

    return Hierarchical.Hierarchy(active_info, scalings...)
end

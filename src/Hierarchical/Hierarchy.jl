############################################################################################
#                                    AbstractHierarchy                                     #
############################################################################################

"""
    AbstractHierarchy{S}

Supertype for all hierarchies. A hierarchy is a collection of sets, each set corresponding
to a level, such that at each level there is scaling information relating parents and
children, and information about which objects are active.

# Type Parameters
- `S`: The concrete type shared by all scalings of the hierarchy.
"""
abstract type AbstractHierarchy{S} end

"""
    get_scalings(hierarchy::AbstractHierarchy)

Return the scaling relations defining `hierarchy`.
"""
get_scalings(hierarchy::AbstractHierarchy) = hierarchy.scalings

"""
    get_scaling(hierarchy::AbstractHierarchy, level::Int)

Return the scaling relation at `level`.
"""
get_scaling(hierarchy::AbstractHierarchy, level::Int) = get_scalings(hierarchy)[level]

"""
    get_active_info(hierarchy::AbstractHierarchy)

Return the active object information associated with `hierarchy`.
"""
get_active_info(hierarchy::AbstractHierarchy) = hierarchy.active_info

"""
    get_num_levels(hierarchy::AbstractHierarchy)

Return the number of hierarchical levels of the given hierarchy.
"""
get_num_levels(hierarchy::AbstractHierarchy) = get_num_levels(get_active_info(hierarchy))

"""
    get_descendants(
        hierarchy::AbstractHierarchy, id::Int; last_level=get_num_levels(hierarchy)
    )

Return all the descendants of object `id`, up-to and including `last_level` (defaults to the
number of levels of the hierarchy.)
"""
function get_descendants(
    hierarchy::AbstractHierarchy, id::Int; last_level=get_num_levels(hierarchy)
)
    level, level_id = convert_to_level_and_level_id(hierarchy, id)
    num_descendants = last_level - level
    num_descendants < 1 && (return Int[])
    descendants = Vector{Vector{Int}}(undef, num_descendants)
    # The first set of descendants is just the children of the current object
    descendants[1] = get_children(get_scaling(hierarchy, level), level_id)
    # We walk all the descendants until `last_level` - `level`.
    # Note that we skip the first set of descendants since they have already been assigned.
    # This is to avoid having to call mapreducevcat on a single collection.
    for i in
        Iterators.dropwhile(i -> i == 1 || level + i > last_level, eachindex(descendants))

        # The current set of descendants is the "union" of all the descendants at the
        # previous level
        descendants[i] = mapreduce(
            d -> get_children(get_scaling(hierarchy, level + i - 1), d),
            vcat,
            descendants[i - 1];
            init=Int[],
        )
        unique!(descendants[i])
    end

    return descendants
end

"""
    get_sets(hierarchy::AbstractHierarchy)

Return the sets of all levels of `hierarchy`, i.e. the sets of its scalings.
"""
get_sets(hierarchy::AbstractHierarchy) = get_sets(get_scalings(hierarchy))

"""
    _check_concrete_scalings(scalings::AbstractVector{<:AbstractScaling})

Throw an `ArgumentError` if the element type of `scalings` is not concrete.

Scalings with different parent or child types, or built from different relation methods
(each function has its own type), have different concrete types.
"""
function _check_concrete_scalings(scalings::AbstractVector{<:AbstractScaling})
    if !isconcretetype(eltype(scalings))
        throw(
            ArgumentError(
                LazyString(
                    "All scalings must have the same concrete type. Got element type ",
                    eltype(scalings),
                ),
            ),
        )
    end

    return nothing
end

"""
    _check_consecutive_scalings(scalings::AbstractVector{<:AbstractScaling})

Throw an `ArgumentError` if the child set of a scaling is not the parent set of the next
scaling (they must be the same object in memory).
"""
function _check_consecutive_scalings(scalings::AbstractVector{<:AbstractScaling})
    for l in 1:(length(scalings) - 1)
        if !(get_child(scalings[l]) === get_parent(scalings[l + 1]))
            throw(
                ArgumentError(
                    LazyString(
                        "Consecutive scalings must have matching child/parent objects. ",
                        "Failed at level ",
                        l,
                    ),
                ),
            )
        end
    end

    return nothing
end

"""
    _check_num_levels(active_info::ActiveInfo, scalings::AbstractVector{<:AbstractScaling})

Throw an `ArgumentError` if the number of levels of `active_info` is not one more than the
number of `scalings`, since each scaling relates two consecutive levels.
"""
function _check_num_levels(
    active_info::ActiveInfo, scalings::AbstractVector{<:AbstractScaling}
)
    if length(scalings) + 1 != get_num_levels(active_info)
        throw(
            ArgumentError(
                LazyString(
                    "Number of levels in `active_info` does not match number of scalings. ",
                    "Got ",
                    get_num_levels(active_info),
                    " levels and ",
                    length(scalings),
                    " scalings.",
                ),
            ),
        )
    end

    return nothing
end

"""
    get_set(hierarchy::AbstractHierarchy, level::Int)

Return the object set at `level`.
"""
get_set(hierarchy::AbstractHierarchy, level::Int) = get_sets(hierarchy)[level]

# NOTE: The following methods serve as an interface with `ActiveInfo`

"""
    get_level_ids(hierarchy::AbstractHierarchy)

See [`get_level_ids(active_info::ActiveInfo)`](@ref).
"""
get_level_ids(hierarchy::AbstractHierarchy) = get_level_ids(get_active_info(hierarchy))

"""
    get_level_ids(hierarchy::AbstractHierarchy, level::Int)

See [`get_level_ids(active_info::ActiveInfo, level::Int)`](@ref).
"""
get_level_ids(hierarchy::AbstractHierarchy, level::Int) =
    get_level_ids(get_active_info(hierarchy), level)

"""
    get_level_cum_num_ids(hierarchy::AbstractHierarchy)

See [`get_level_cum_num_ids(active_info::ActiveInfo)`](@ref).
"""
get_level_cum_num_ids(hierarchy::AbstractHierarchy) =
    get_level_cum_num_ids(get_active_info(hierarchy))

"""
    get_level_cum_num_ids(hierarchy::AbstractHierarchy, level::Int)

See [`get_level_cum_num_ids(active_info::ActiveInfo, level::Int)`](@ref).
"""
get_level_cum_num_ids(hierarchy::AbstractHierarchy, level::Int) =
    get_level_cum_num_ids(get_active_info(hierarchy), level)

"""
    get_level_lookup(hierarchy::AbstractHierarchy)

See [`get_level_lookup(active_info::ActiveInfo)`](@ref).
"""
get_level_lookup(hierarchy::AbstractHierarchy) =
    get_level_lookup(get_active_info(hierarchy))

"""
    get_level_lookup(hierarchy::AbstractHierarchy, level::Int)

See [`get_level_lookup(active_info::ActiveInfo, level)`](@ref).
"""
get_level_lookup(hierarchy::AbstractHierarchy, level::Int) =
    get_level_lookup(get_active_info(hierarchy), level)

"""
    get_level_sets(hierarchy::AbstractHierarchy)

See [`get_level_sets(active_info::ActiveInfo)`](@ref).
"""
get_level_sets(hierarchy::AbstractHierarchy) = get_level_sets(get_active_info(hierarchy))

"""
    get_level_set(hierarchy::AbstractHierarchy, level::Int)

See [`get_level_set(active_info::ActiveInfo, level)`](@ref).
"""
get_level_set(hierarchy::AbstractHierarchy, level::Int) =
    get_level_set(get_active_info(hierarchy), level)

"""
    get_level_num_ids(hierarchy::AbstractHierarchy, level::Int)

See [`get_level_num_ids(active_info::ActiveInfo, level::Int)`](@ref).
"""
get_level_num_ids(hierarchy::AbstractHierarchy, level::Int) =
    get_level_num_ids(get_active_info(hierarchy), level)

"""
    get_num_objects(hierarchy::AbstractHierarchy)

See [`get_num_objects(active_info::ActiveInfo)`](@ref).
"""
get_num_objects(hierarchy::AbstractHierarchy) = get_num_objects(get_active_info(hierarchy))

"""
    get_level(hierarchy::AbstractHierarchy, hier_id::Int)

See [`get_level(active_info::ActiveInfo, hier_id::Int)`](@ref).
"""
get_level(hierarchy::AbstractHierarchy, hier_id::Int) =
    get_level(get_active_info(hierarchy), hier_id)

"""
    convert_to_level_id(hierarchy::AbstractHierarchy, hier_id::Int)

See [`convert_to_level_id(active_info::ActiveInfo, hier_id::Int)`](@ref).
"""
convert_to_level_id(hierarchy::AbstractHierarchy, hier_id::Int) =
    convert_to_level_id(get_active_info(hierarchy), hier_id)

"""
    convert_to_level_and_level_id(hierarchy::AbstractHierarchy, hier_id::Int)

See [`convert_to_level_and_level_id(active_info::ActiveInfo, hier_id::Int)`](@ref).
"""
convert_to_level_and_level_id(hierarchy::AbstractHierarchy, hier_id::Int) =
    convert_to_level_and_level_id(get_active_info(hierarchy), hier_id)

"""
    convert_to_hier_id(hierarchy::AbstractHierarchy, level::Int, level_id::Int)

See [`convert_to_hier_id(active_info::ActiveInfo, level::Int, level_id::Int)`](@ref).
"""
convert_to_hier_id(hierarchy::AbstractHierarchy, level::Int, level_id::Int) =
    convert_to_hier_id(get_active_info(hierarchy), level, level_id)

# TODO: Implement a `coarsen` method as well.

"""
    refine(
        hierarchy::AbstractHierarchy,
        level::Int,
        remove::Vector{Int},
        scaling::AbstractScaling,
    )

Return a refined version of `hierarchy` by deactivating the objects in `remove` and
activating their children. If `level` is the finest level, a new level is created using
`scaling`.

See also [`update`](@ref).
"""
function refine(
    hierarchy::AbstractHierarchy, level::Int, remove::Vector{Int}, scaling::AbstractScaling
)
    is_finest = level == get_num_levels(hierarchy)
    relevant_scaling = is_finest ? scaling : get_scaling(hierarchy, level)
    add = mapreduce(
        p -> collect(get_children(relevant_scaling, p)), vcat, remove; init=Int[]
    )

    return update(hierarchy, level, remove, add, scaling)
end

"""
    update(
        hierarchy::AbstractHierarchy,
        level::Int,
        remove::Vector{Int},
        add::Vector{Int},
        scaling::AbstractScaling,
    )

Similar to [`update(active_info::ActiveInfo, level::Int, remove::Vector{Int},
add::Vector{Int})`](@ref), but if a new level needs to be added, then `scaling` is appended
to the scalings of the hierarchy. The scalings of `hierarchy` are not modified.
"""
function update(
    hierarchy::H,
    level::Int,
    remove::Vector{Int},
    add::Vector{Int},
    scaling::AbstractScaling,
) where {H <: AbstractHierarchy}
    active_info = update(get_active_info(hierarchy), level, remove, add)
    scalings = get_scalings(hierarchy)
    if get_num_levels(active_info) > get_num_levels(hierarchy)
        scalings = vcat(scalings, scaling)
    end

    # This gets the name of the concrete type `H`
    constructor = Base.typename(H).wrapper

    return constructor(active_info, scalings)
end

############################################################################################
#                                        Hierarchy                                         #
############################################################################################

"""
    Hierarchy{S} <: AbstractHierarchy{S}

Stores a hierarchical construction consisting of a sequence of scaling relations together
with the active objects on each level.

The scaling on level `l` relates the objects on level `l` to those on level `l+1`, while the
associated `ActiveInfo` specifies which objects are active on each level.

# Fields
- `active_info::ActiveInfo`: See [`ActiveInfo`](@ref).
- `scalings::Vector{S}`: One scaling per pair of consecutive levels; see
    [`AbstractScaling`](@ref).

# Type Parameters
- `S`: The concrete type shared by all scalings.
"""
struct Hierarchy{S} <: AbstractHierarchy{S}
    active_info::ActiveInfo
    scalings::Vector{S}

    """
        Hierarchy(active_info::ActiveInfo, scalings::Vector{S}) where {S <: AbstractScaling}

    Throws an `ArgumentError` if `S` is not concrete, if consecutive scalings do not share
    their child/parent sets, or if the number of levels of `active_info` is not
    `length(scalings) + 1`.
    """
    function Hierarchy(
        active_info::ActiveInfo, scalings::Vector{S}
    ) where {S <: AbstractScaling}
        _check_concrete_scalings(scalings)
        _check_consecutive_scalings(scalings)
        _check_num_levels(active_info, scalings)

        return new{S}(active_info, scalings)
    end
end

############################################################################################
#                                     TreeHierarchy                                      #
############################################################################################

"""
    TreeHierarchy{S} <: AbstractHierarchy{S}

Hierarchy meant for single-parent objects; essentially a tree. Useful for creating, for
example, a hierarchical geometry, where it is _not_ meaningful to have both a parent and its
children active.

See also [`Hierarchy`](@ref).

# Fields
- `active_info::ActiveInfo`: See [`ActiveInfo`](@ref).
- `scalings::Vector{S}`:  See [`AbstractScaling`](@ref).
- `tree_ids::Vector{Set{Int}}`: Collection of level-wise indices indicating whether a
    given object is either active or has active children. In general, this is _not_ the same
    as the active level-wise indices. For a hierarchical geometry, these are used, for
    example, to select the active basis functions of the hierarchical spaces built on it.

# Type Parameters
- `S`: The concrete type shared by all scalings.
"""
struct TreeHierarchy{S} <: AbstractHierarchy{S}
    active_info::ActiveInfo
    scalings::Vector{S}
    tree_ids::Vector{Set{Int}}

    """
        TreeHierarchy(
            active_info::ActiveInfo, scalings::Vector{S}; check_tree=true
        ) where {S <: AbstractScaling}

    Similar to the constructor for `Hierarchy`, with the default option of checking whether a
    tree-like structure is satisfied.

    # Arguments
    - `active_info::ActiveInfo`: See [`ActiveInfo`](@ref).
    - `scalings::Vector{S}`: See [`AbstractScaling`](@ref).

    # Keyword arguments
    - `check_tree=true`: Checks whether `active_info` satisfies a tree-like structure for
        the given `scalings`. Enabled by default.
    """
    function TreeHierarchy(
        active_info::ActiveInfo, scalings::Vector{S}; check_tree=true
    ) where {S <: AbstractScaling}
        _check_concrete_scalings(scalings)
        _check_consecutive_scalings(scalings)
        _check_num_levels(active_info, scalings)
        check_tree && _check_tree(scalings, active_info)
        tree_ids = _tree_ids(scalings, active_info)

        return new{S}(active_info, scalings, tree_ids)
    end
end

"""
    _check_tree(scalings, active_info)

Throw an `ArgumentError` if an active object in `active_info` has an active descendant, on
any finer level, according to `scalings`. Return `true` otherwise.
"""
function _check_tree(scalings, active_info)
    L = get_num_levels(active_info)
    for level in 1:(L - 1)
        parents = copy(get_level_ids(active_info, level))
        children = similar(parents, 0)
        # Check if an active parent has active children
        for l in (level + 1):L
            # Clear previous parents, which are now the children
            empty!(children)
            children_set = get_level_set(active_info, l)
            for p in parents, c in get_children(scalings[l - 1], p)
                # Child is active
                if c in children_set
                    throw(
                        ArgumentError(
                            LazyString(
                                "Not a tree. Child ",
                                c,
                                " on level ",
                                l,
                                " is active and has an active parent ",
                                p,
                                " on level ",
                                level,
                            ),
                        ),
                    )
                end

                # Store the children to be the next level's parents.
                # Note that we do not simply loop over the parents of a given level and
                # check them against their children, since this would fail to detect a
                # parent from level `l` having active children at level `l+2`, or  `l+3`,...
                push!(children, c)
            end

            # Swap to descend into nested children
            parents, children = children, parents
        end
    end

    return true
end

"""
    _tree_ids(scalings, active_info)

Return, for each level, the level-wise ids of the objects in `active_info` that are either
active or ancestors of active objects, according to `scalings`.

See also [`get_tree_ids`](@ref).
"""
function _tree_ids(scalings, active_info)
    # `tree_ids` starts as a clone of the level-wise sets, but then we append to each such
    # set at level `l` the parents of objects in `l+1`.
    tree_ids = deepcopy(get_level_sets(active_info))
    L = get_num_levels(active_info)
    for l in L:-1:2, child in tree_ids[l]
        union!(tree_ids[l - 1], get_parents(scalings[l - 1], child))
    end

    return tree_ids
end

"""
    get_tree_ids(hierarchy::TreeHierarchy)

Return the `tree_ids` of `hierarchy`.
"""
get_tree_ids(hierarchy::TreeHierarchy) = hierarchy.tree_ids

"""
    get_tree_ids(hierarchy::TreeHierarchy, level::Int)

Return the `tree_ids` of `hierarchy` at `level`.
"""
get_tree_ids(hierarchy::TreeHierarchy, level::Int) = get_tree_ids(hierarchy)[level]

############################################################################################
#                                    AbstractHierarchy                                     #
############################################################################################

"""
    AbstractHierarchy{L, S}

Supertype for all hierarchies. An hierarchy is a collection of sets, each set corresponding
to a level, such that at each level there is scaling information relating parents and
children, and information about which objects are active.
"""
abstract type AbstractHierarchy{L, S} end

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
    get_num_levels(::AbstractHierarchy{L}) where {L}

Return the number of hierarchical levels of the given hierarchy.
"""
get_num_levels(::AbstractHierarchy{L}) where {L} = L

"""
    get_descendants(hierarchy::AbstractHierarchy{L}, id::Int; last_level=L) where {L}

Return all the descendants of object `id`, up-to and including `last_level` (defaults to the
number of levels of the hierarchy.)
"""
function get_descendants(hierarchy::AbstractHierarchy{L}, id::Int; last_level=L) where {L}
    level, level_id = convert_to_level_and_level_id(hierarchy, id)
    # The finest level has no descendants
    level == L && (return Int[])

    # `last_level` - `level` is the number of descendants we have
    descendants = Vector{Vector{Int}}(undef, last_level - level)
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
            descendants[i - 1],
        )
        unique!(descendants[i])
    end

    return descendants
end

# TODO: These might not be needed
# function Base.iterate(hierarchy::AbstractHierarchy{L}) where {L}
#     scalings = get_scalings(hierarchy)
#
#     return (get_parent(first(scalings)), Val(2))
# end
#
# function Base.iterate(hierarchy::AbstractHierarchy{L}, ::Val{l}) where {L, l}
#     scalings = get_scalings(hierarchy)
#     isnothing(parent) && return nothing
#
#     return (get_parent(scalings[l]), Val(l + 1))
# end
#
# function Base.iterate(hierarchy::AbstractHierarchy{L}, ::Val{L}) where {L}
#     scalings = get_scalings(hierarchy)
#     child = get_child(scalings[L - 1])
#     isnothing(child) && return nothing
#
#     return (child, Val(0))
# end
#
# function Base.iterate(::AbstractHierarchy{L}, ::Val{0}) where {L}
#     return nothing
# end
#
# Base.length(hierarchy::AbstractHierarchy) = get_num_levels(hierarchy)

"""
    get_sets(hierarchy::AbstractHierarchy)

See [`get_sets(scalings::S) where {LS, S <: NTuple{LS, AbstractScaling}}`](@ref).
"""
get_sets(hierarchy::AbstractHierarchy) = get_sets(get_scalings(hierarchy))

function _warn_stability(sets)
    # STABILITY: We issue this warning because heterogeneously-typed tuples make runtime
    # indexing unstable, as the output type is unknown.
    if !(sets isa NTuple)
        @warn LazyString(
            "The sets provided in scalings do not form an homogeneously-typed tuple. ",
            "Type-stability will be affected. ",
            "Got type ",
            typeof(sets),
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

get_level_ids(hierarchy::AbstractHierarchy) = get_level_ids(get_active_info(hierarchy))

get_level_ids(hierarchy::AbstractHierarchy, level::Int) =
    get_level_ids(get_active_info(hierarchy), level)

get_level_cum_num_ids(hierarchy::AbstractHierarchy) =
    get_level_cum_num_ids(get_active_info(hierarchy))

get_level_cum_num_ids(hierarchy::AbstractHierarchy, level::Int) =
    get_level_cum_num_ids(get_active_info(hierarchy), level)

get_level_lookup(hierarchy::AbstractHierarchy) =
    get_level_lookup(get_active_info(hierarchy))

get_level_lookup(hierarchy::AbstractHierarchy, level::Int) =
    get_level_lookup(get_active_info(hierarchy), level)

get_level_sets(hierarchy::AbstractHierarchy) = get_level_sets(get_active_info(hierarchy))

get_level_set(hierarchy::AbstractHierarchy, level::Int) =
    get_level_set(get_active_info(hierarchy), level)

get_level_num_ids(hierarchy::AbstractHierarchy, level::Int) =
    get_level_num_ids(get_active_info(hierarchy), level)

get_num_objects(hierarchy::AbstractHierarchy) = get_num_objects(get_active_info(hierarchy))

get_level(hierarchy::AbstractHierarchy, hier_id::Int) =
    get_level(get_active_info(hierarchy), hier_id)

convert_to_level_id(hierarchy::AbstractHierarchy, hier_id::Int) =
    convert_to_level_id(get_active_info(hierarchy), hier_id)

convert_to_level_and_level_id(hierarchy::AbstractHierarchy, hier_id::Int) =
    convert_to_level_and_level_id(get_active_info(hierarchy), hier_id)

convert_to_hier_id(hierarchy::AbstractHierarchy, level::Int, level_id::Int) =
    convert_to_hier_id(get_active_info(hierarchy), level, level_id)

# TODO: Implement a `coarsen` method as well.

"""
    refine(
        hierarchy::AbstractHierarchy{L},
        level::Int,
        remove::Vector{Int},
        scaling::AbstractScaling,
    ) where {L}

Return a refined version of `hierarchy` by deactivating the objects in `remove` and
activating their children.

See also [`update`](@ref).
"""
function refine(
    hierarchy::AbstractHierarchy{L},
    level::Int,
    remove::Vector{Int},
    scaling::AbstractScaling,
) where {L}
    # STABILITY: obtaining the relevant scaling here is inherently unstable. Either the
    # level already exists, in which case we have to index `scalings[level]`, or the level
    # is new and we have to use the provided `scaling`.  Nonetheless, `add` should be
    # correctly inferred as a `Vector{Int}`.
    relevant_scaling = (level == L ? scaling : get_scaling(hierarchy, level))
    add = mapreduce(p -> collect(get_children(relevant_scaling, p)), vcat, remove)

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

Similar [`update(active_info::ActiveInfo, level::Int, remove::Vector{Int},
add::Vector{Int})`](@ref), but if a new level needs to be added, then `scaling` is appended
to the scalings of the hierarchy.

!!!warning
    This is type unstable since the output type depends on the runtime value of `level`.
"""
function update(
    hierarchy::H,
    level::Int,
    remove::Vector{Int},
    add::Vector{Int},
    scaling::AbstractScaling,
) where {H <: AbstractHierarchy}
    active_info = update(get_active_info(hierarchy), level, remove, add)
    # STABILITY: This method is type unstable. See method's comments
    scalings = _maybe_add_scaling(hierarchy, scaling, active_info)
    # This gets the name of the concrete type `H`
    constructor = Base.typename(H).wrapper

    return constructor(active_info, scalings...)
end

function add_scaling(hierarchy::AbstractHierarchy, scaling::AbstractScaling)
    return (get_scalings(hierarchy)..., scaling)
end

@noinline function _maybe_add_scaling(
    hierarchy::H, scaling::AbstractScaling, active_info::ActiveInfo
) where {L, H <: AbstractHierarchy{L}}
    # STABILITY: This operation is inherently type unstable as it depends on the runtime
    # value of `get_num_levels(active_info)`. However, it is limited to 2 options:
    #     1. NTuple{L, <: AbstractScaling}
    #     2. NTuple{L+1, <: AbstractScaling}
    #
    # The `@noinline` hopefully stops this instability from propagating. `update` callers
    # should barrier `update` as much as possible. 
    # See the example below this method.

    scalings = get_scalings(hierarchy)
    if get_num_levels(active_info) > L
        # The new scaling needs to be added
        scalings = (scalings..., scaling)
    end

    return scalings
end

#= 

This method does _not_ barrier `update`, so `last_obj` will be `Union{SomeType{L,...},
SomeType{L+1,...}}`. As a consequence, `n` can not infer `L` or `L+1` from the `Union` type,
resulting in `n::Any`, and an output of `::Val`.

function nbarrier(hierarchy, scaling)
    hierarchy = update(hierarchy, 1, Int[], Int[], scaling)
    obj = get_sets(hierarchy)
    last_obj = last(obj)
    n = first(typeof(last_obj).parameters)

    return Val(n)
end


This method barriers `update` correctly. As before, `update` gets inferred as
`Union{AbstractHierarchy{L,...}, AbstractHierarchy{L+1,...}}`. However, since this
immediately dispatched to the type stable `_wbarrier`, each branch of the `Union` gets
correctly inferred as `Val{L}` and `Val{L+1}`, and so the output type is `Union{Val{L},
Val{L+1}}`.

function wbarrier(hierarchy, scaling)
    hierarchy = update(hierarchy, 1, Int[], Int[], scaling)
    return _wbarrier(hierarchy)
end

function _wbarrier(hierarchy)
    obj = get_sets(hierarchy)
    last_obj = last(obj)
    n = first(typeof(last_obj).parameters)

    return Val(n)
end

=#

############################################################################################
#                                        Hierarchy                                         #
############################################################################################

"""
    Hierarchy{L, S} <: AbstractHierarchy{L, S}

Stores a hierarchical construction consisting of a sequence of scaling relations together
with the active objects on each level.

The scaling on level `l` relates the objects on level `l-1` to those on level `l`, while the
associated `ActiveInfo` specifies which objects are active on each level.

# Fields
- `active_info::ActiveInfo`: See [`ActiveInfo`](@ref).
- `scalings::S`: A tuple of `L+1` abstract scalings; see [`AbstractScaling`](@ref).

# Type Parameters
- `L`: The number of hierarchical levels.
- `S`: The type of the tuple of scalings.
"""
struct Hierarchy{L, S} <: AbstractHierarchy{L, S}
    active_info::ActiveInfo
    scalings::S

    function Hierarchy(
        active_info::ActiveInfo, scalings::Vararg{AbstractScaling, LS}
    ) where {LS}
        for i in 1:(LS - 1)
            if !(get_child(scalings[i]) === get_parent(scalings[i + 1]))
                throw(
                    ArgumentError(
                        LazyString(
                            "Consecutive scalings must have matching child/parent objects. ",
                            "Failed at level ",
                            i,
                        ),
                    ),
                )
            end
        end

        # Each scaling corresponds to two consecutive levels
        L = LS + 1
        if L != get_num_levels(active_info)
            throw(
                ArgumentError(
                    LazyString(
                        "Number of levels in `active_info` does not match number of scalings. ",
                        " Got ",
                        get_num_levels(active_info),
                        " and ",
                        LS,
                    ),
                ),
            )
        end

        _warn_stability(get_sets(scalings))

        return new{L, typeof(scalings)}(active_info, scalings)
    end
end

############################################################################################
#                                     TreeHierarchy                                      #
############################################################################################

"""
    TreeHierarchy{L, S} <: AbstractHierarchy{L, S}

Hierarchy meant for single-parent objects; essentially a tree. Useful for creating, for
example, a hierarchical geometry, where it is _not_ meaningful to have both a parent and its
children active.

See also [`Hierarchy`](@ref).

# Fields
- `active_info::ActiveInfo`: See [`ActiveInfo`](@ref).
- `scalings::S`:  See [`AbstractScaling`](@ref).
- `tree_ids::Vector{Set{Int}}`: Collection of level-wise indices indicating whether a
    given object is either active or has active children. In general, this is _not_ the same
    as the active level-wise indices. Useful in basis selection algorithms for hierarchical
    spaces, for example
"""
struct TreeHierarchy{L, S} <: AbstractHierarchy{L, S}
    active_info::ActiveInfo
    scalings::S
    tree_ids::Vector{Set{Int}}

    """
        TreeHierarchy(
            active_info::ActiveInfo, scalings::Vararg{AbstractScaling, LS}; check_tree=true
        ) where {LS}

    Similar the constructor for `Hierarchy`, with the default option of checking whether a
    tree-like structure is satisfied.

    # Arguments
    - `active_info::ActiveInfo`: See [`ActiveInfo`](@ref).
    - `scalings::Vararg{AbstractScaling, LS}`: See [`AbstractScaling`](@ref).

    # Keyword arguments
    - `check_tree=true`: Checks whether `active_info` satisfies a tree-like structure for
        the given `scalings`. Enabled by default.
    """
    function TreeHierarchy(
        active_info::ActiveInfo, scalings::Vararg{AbstractScaling, LS}; check_tree=true
    ) where {LS}
        for l in 1:(LS - 1)
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

        # Each scaling corresponds to two levels
        L = LS + 1
        if L != get_num_levels(active_info)
            throw(
                ArgumentError(
                    LazyString(
                        "Number of levels in `active_info` does not match ",
                        "number of scalings. Got ",
                        get_num_levels(active_info),
                        " and ",
                        L,
                        " respectively.",
                    ),
                ),
            )
        end

        tree_ids = _tree_ids(scalings, active_info)
        check_tree && _check_tree(scalings, active_info)

        _warn_stability(get_sets(scalings))

        return new{L, typeof(scalings)}(active_info, scalings, tree_ids)
    end
end

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

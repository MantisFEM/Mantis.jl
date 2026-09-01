############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    ActiveInfo

Stores the active objects in a hierarchical construction.

Active objects are grouped by refinement level. Within each level, objects retain their
level-wise numbering, while the hierarchical numbering is obtained by concatenating the
active objects from all levels.

Besides the level-wise indices, `ActiveInfo` stores auxiliary lookup tables to efficiently
convert between hierarchical and level-wise numbering.

# Fields
- `level_ids::Vector{Vector{Int}}`: Active level-wise object ids for each level. Useful for
    fast iteration.
- `level_sets::Vector{Set{Int}}`: Set representation of `level_ids` for fast membership
    testing.
- `level_lookup::Vector{Dict{Int, Int}}`: Maps a level-wise id to its position within
    `level_ids[level]`.
- `level_cum_num_ids`: Cumulative number of active objects per level, used for converting
    between hierarchical and level-wise numbering.

# Outer Constructors
- [`ActiveInfo(level_ids::Vector{Vector{Int}})`](@ref)
"""
struct ActiveInfo
    level_ids::Vector{Vector{Int}}
    level_sets::Vector{Set{Int}}
    level_lookup::Vector{Dict{Int, Int}}
    level_cum_num_ids::Vector{Int}

    """
        ActiveInfo(
            level_ids::Vector{Vector{Int}},
            level_sets::Vector{Set{Int}};
            check_unique=true,
            check_sets=true,
        )

    # Arguments
    - `level_ids::Vector{Vector{Int}}`: The level-wise indices of active objects.
    - `level_sets::Vector{Set{Int}}`: The set representation of `level_ids`.

    # Keyword Arguments
    - `check_unique=true`: Checks of all entries of `level_ids` are unique.
    - `check_sets=true`: Checks if `level_ids` and `level_sets` represent the same objects.
    """
    function ActiveInfo(
        level_ids::Vector{Vector{Int}},
        level_sets::Vector{Set{Int}};
        check_unique=true,
        check_sets=true,
    )
        if check_unique
            for l in eachindex(level_ids)
                if !allunique(level_ids[l])
                    return throw(
                        ArgumentError(
                            LazyString("Level indices must be unique. Failed on level ", l)
                        ),
                    )
                end
            end
        end

        if check_sets
            for l in eachindex(level_ids)
                if Set(level_ids[l]) != level_sets[l]
                    return throw(
                        ArgumentError(
                            LazyString(
                                "Incompatible `level_ids` and `level_sets`. ",
                                "Failed on level ",
                                l,
                            ),
                        ),
                    )
                end
            end
        end

        level_cum_num_ids = Vector{Int}(undef, length(level_ids) + 1)
        level_cum_num_ids[1] = 0
        for (l, ids) in enumerate(level_ids)
            level_cum_num_ids[l + 1] = level_cum_num_ids[l] + length(ids)
        end

        # `level_lookup` is a dictionary for each level. It stores what position at
        # `level_ids[level]` a given `level_id` has. 
        # With this we can avoid having to find where `level_id` is in `level_ids[level]`,
        # which is needed for converting `(level, level_id)` to `hier_id`.
        level_lookup = map(
            ids ->
                Dict{Int, Int}(level_id => count for (count, level_id) in enumerate(ids)),
            level_ids,
        )

        return new(level_ids, level_sets, level_lookup, level_cum_num_ids)
    end
end

"""
    ActiveInfo(level_ids::Vector{Vector{Int}})

Constructor from a collection of active `level_ids`.
"""
function ActiveInfo(level_ids::Vector{Vector{Int}})
    level_sets = map(Set, level_ids)

    # We don't need to check the sets since they are constructed from `level_ids`
    return ActiveInfo(level_ids, level_sets; check_sets=false)
end

"""
    ActiveInfo(level_sets::Vector{Set{Int}})

Constructor from a collection of active `level_sets`.
"""
function ActiveInfo(level_sets::Vector{Set{Int}})
    level_ids = map(collect, level_sets)

    # We don't any checks since `level_ids` are constructed form `level_sets`, so they match
    # and are necessarily unique.
    return ActiveInfo(level_ids, level_sets; check_unique=false, check_sets=false)
end

"""
    get_level_ids(active_info::ActiveInfo)

Return the active level-wise ids for all levels.
"""
function get_level_ids(active_info::ActiveInfo)
    return active_info.level_ids
end

"""
    get_level_ids(active_info::ActiveInfo, level::Int)

Return the active level-wise ids on `level`.
"""
function get_level_ids(active_info::ActiveInfo, level::Int)
    return get_level_ids(active_info)[level]
end

"""
    get_level_cum_num_ids(active_info::Int)

Return the cumulative number of active objects per level.

The returned vector has length `num_levels + 1` and the first entry is _always_ zero.
"""
function get_level_cum_num_ids(active_info::ActiveInfo)
    return active_info.level_cum_num_ids
end

"""
    get_level_cum_num_ids(active_info::ActiveInfo, level::Int)

Return the cumulative number of active objects up to and _including_ `level`.
"""
function get_level_cum_num_ids(active_info::ActiveInfo, level::Int)
    return get_level_cum_num_ids(active_info)[level + 1]
end

"""
    get_level_lookup(active_info::ActiveInfo)

Return the lookup tables mapping level-wise ids to their local ordering within each level.
"""
get_level_lookup(active_info::ActiveInfo) = active_info.level_lookup

"""
    get_level_lookup(active_info::ActiveInfo, level)

Return the lookup tables mapping level-wise ids to their local ordering on `level`.
"""
get_level_lookup(active_info::ActiveInfo, level) = get_level_lookup(active_info)[level]

"""
    get_level_sets(active_info)

Return the set of active level-wise ids for each level.
"""
get_level_sets(active_info::ActiveInfo) = active_info.level_sets

"""
    get_level_set(active_info::ActiveInfo, level)

Return the set of active level-wise ids on `level`.
"""
get_level_set(active_info::ActiveInfo, level) = get_level_sets(active_info)[level]

"""
    get_level_num_ids(active_info::ActiveInfo, level::Int)

Return the number of active objects on `level`.
"""
function get_level_num_ids(active_info::ActiveInfo, level::Int)
    level_cum_num_ids = get_level_cum_num_ids(active_info)

    return level_cum_num_ids[level + 1] - level_cum_num_ids[level]
end

"""
    get_num_levels(active_info::ActiveInfo)

Return the number of hierarchical levels of `active_info`.
"""
function get_num_levels(active_info::ActiveInfo)
    return length(get_level_ids(active_info))
end

"""
    get_num_objects(active_info::ActiveInfo)

Return the total number of active objects across all levels.
"""
function get_num_objects(active_info::ActiveInfo)
    return active_info.level_cum_num_ids[end]
end

"""
    get_level(active_info::ActiveInfo, hier_id::Int)

Return the level containing the hierarchical object indexed by `hier_id`.
"""
function get_level(active_info::ActiveInfo, hier_id::Int)
    # hierarchical ids are cumulatively counted from the coarsest level, so to determine the
    # level of `hier_id`, we find the entry in the level-wise cumulative count that is
    # smaller or equal to `hier_id`.
    level = searchsortedlast(get_level_cum_num_ids(active_info), hier_id - 1)
    if iszero(level) || level > get_num_levels(active_info)
        throw(BoundsError(active_info, hier_id))
    end

    return level
end

############################################################################################
#                                       Conversions                                        #
############################################################################################

"""
    convert_to_level_id(active_info, hier_id)

Convert hierarchical index `hier_id` to the corresponding level-wise id.
"""
function convert_to_level_id(active_info::ActiveInfo, hier_id::Int)
    object_level = get_level(active_info, hier_id)

    return _convert_to_level_id(active_info, hier_id, object_level)
end

"""
	convert_to_level_and_level_id(active_info::ActiveInfo, hier_id::Int)

Returns the `level` and `level_id` that correspond to hierarchical index `hier_id`.
"""
function convert_to_level_and_level_id(active_info::ActiveInfo, hier_id::Int)
    object_level = get_level(active_info, hier_id)

    return object_level, _convert_to_level_id(active_info, hier_id, object_level)
end

function _convert_to_level_id(active_info::ActiveInfo, hier_id, object_level)
    level_ids = get_level_ids(active_info, object_level)
    prev_level_cum_num_ids = get_level_cum_num_ids(active_info, object_level - 1)
    # We subtract from `hier_id` the number of active objects from previous levels.
    return level_ids[hier_id - prev_level_cum_num_ids]
end

"""
    convert_to_hier_id(active_info::ActiveInfo, level::Int, level_id::Int)

Convert the level-wise indexing `level_id` on `level` to its hierarchical index.
"""
function convert_to_hier_id(active_info::ActiveInfo, level::Int, level_id::Int)
    level_id_count = get_level_lookup(active_info, level)[level_id]
    prev_level_cum_num_ids = get_level_cum_num_ids(active_info, level - 1)
    # Because we store the position of `level_id` within `level_ids[level]`, we know that
    # the corresponding hierarchical index is just the number of active objects from all
    # previous levels plus that position.
    return prev_level_cum_num_ids + level_id_count
end

############################################################################################
#                                         Update                                           #
############################################################################################

"""
    update(active_info::ActiveInfo, level::Int, remove::Vector{Int}, add::Vector{Int})

Returns a clone of `active_info` after updating the active object identifiers.

The objects in `remove` are removed from `level`, while the objects in `add` become active
on `level + 1`. If `level` is the finest level and `add` is non-empty, a new refinement
level is created automatically. The cumulative numbering is updated accordingly.
"""
function update(active_info::ActiveInfo, level::Int, remove::Vector{Int}, add::Vector{Int})
    # MUTABILITY: Deep-copy to protect the original `active_info` from the mutation
    # happening in the rest of the method
    active_info = deepcopy(active_info)

    num_levels = get_num_levels(active_info)
    if level == num_levels && !isempty(add)
        active_info = add_level!(active_info)
        num_levels += 1
    end

    # Levels `level` and `level+1` are updated with `remove` and `add`
    setdiff!(get_level_ids(active_info, level), remove)
    setdiff!(get_level_set(active_info, level), remove)
    union!(get_level_ids(active_info, level + 1), add)
    union!(get_level_set(active_info, level + 1), add)

    # The previous changes to `level` and `level+1` require updating the cumulative count
    # and level lookups for all subsequent levels
    level_cum_num_ids = get_level_cum_num_ids(active_info)
    for l in level:num_levels
        level_cum_num_ids[l + 1] =
            level_cum_num_ids[l] + length(get_level_ids(active_info)[l])
    end

    get_level_lookup(active_info)[level] = Dict(
        id => i for (i, id) in enumerate(get_level_ids(active_info, level))
    )
    get_level_lookup(active_info)[level + 1] = Dict(
        id => i for (i, id) in enumerate(get_level_ids(active_info, level + 1))
    )

    return active_info
end

"""
    add_level!(active_info::ActiveInfo)

Returns a _mutated_ version of `active_info` after appending an empty level.
"""
function add_level!(active_info::ActiveInfo)
    # MUTABILITY: using `push!` and `append!` will alter the fields of the provided
    # `active_info`
    push!(get_level_ids(active_info), Int[])
    push!(get_level_sets(active_info), Set{Int}())
    push!(get_level_lookup(active_info), Dict{Int, Int}())
    append!(get_level_cum_num_ids(active_info), get_level_cum_num_ids(active_info)[end])

    return active_info
end

"""
    add_level(active_info::ActiveInfo)

Mutation-safe version of [`add_level!`](@ref). Preferable in most cases.
"""
function add_level(active_info::ActiveInfo)
    active_info = deepcopy(active_info)
    active_info = add_level!(active_info)

    return active_info
end

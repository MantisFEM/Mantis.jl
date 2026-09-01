############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    ActiveInfo(level_ids)

Stores the active objects in a hierarchical construction.

Active objects are grouped by refinement level. Within each level, objects retain their
level-wise numbering, while the hierarchical numbering is obtained by concatenating the
active objects from all levels.

Besides the level-wise indices, `ActiveInfo` stores auxiliary lookup tables to efficiently
convert between hierarchical and level-wise numbering.

# Fields
- `level_ids`: Active level-wise object ids for each level.
- `level_sets`: Set representation of `level_ids` for fast membership queries.
- `level_lookup`: Maps a level-wise id to its position within `level_ids[level]`.
- `level_cum_num_ids`: Cumulative number of active objects per level, used for converting
    between hierarchical and level-wise numbering.
"""
struct ActiveInfo
    level_ids::Vector{Vector{Int}}
    level_sets::Vector{Set{Int}}
    level_lookup::Vector{Dict{Int, Int}}
    level_cum_num_ids::Vector{Int}

    function ActiveInfo(
        level_ids::Vector{Vector{Int}}, level_sets::Vector{Set{Int}}; check_unique=true
    )
        if check_unique
            for l in eachindex(level_ids)
                if !allunique(level_ids[l])
                    throw(
                        ArgumentError("Level indices must be unique. Failed on level $(l).")
                    )
                end
            end
        end

        level_cum_num_ids = Vector{Int}(undef, length(level_ids) + 1)
        level_cum_num_ids[1] = 0
        for (l, ids) in enumerate(level_ids)
            level_cum_num_ids[l+1] = level_cum_num_ids[l] + length(ids)
        end

        level_lookup = [
            Dict{Int, Int}(id => i for (i, id) in enumerate(ids)) for ids in level_ids
        ]

        return new(level_ids, level_sets, level_lookup, level_cum_num_ids)
    end
end

function ActiveInfo(level_ids::Vector{Vector{Int}})
    level_sets = map(Set, level_ids)

    return ActiveInfo(level_ids, level_sets)
end

function ActiveInfo(level_sets::Vector{Set{Int}})
    level_ids = map(collect, level_sets)

    return ActiveInfo(level_ids, level_sets; check_unique=false)
end

############################################################################################
#                                         Getters                                          #
############################################################################################

"""
    get_level_ids(active_info)

Return the active level-wise ids for all levels.
"""
function get_level_ids(active_info::ActiveInfo)
    return active_info.level_ids
end

"""
    get_level_ids(active_info, level)

Return the active level-wise ids on `level`.
"""
function get_level_ids(active_info::ActiveInfo, level::Int)
    return get_level_ids(active_info)[level]
end

"""
    get_level_cum_num_ids(active_info)

Return the cumulative number of active objects per level.

The returned vector has length `num_levels + 1`, where the first entry is zero.
"""
function get_level_cum_num_ids(active_info::ActiveInfo)
    return active_info.level_cum_num_ids
end

"""
    get_level_cum_num_ids(active_info, level)

Return the cumulative number of active objects up to and including `level`.
"""
function get_level_cum_num_ids(active_info::ActiveInfo, level::Int)
    return get_level_cum_num_ids(active_info)[level + 1]
end

"""
    get_level_lookup(active_info)

Return the lookup tables mapping level-wise ids to their local ordering within each level.
"""
get_level_lookup(active_info::ActiveInfo) = active_info.level_lookup

"""
    get_level_lookup(active_info, level)

Return the lookup tables mapping level-wise ids to their local ordering on `level`.
"""
get_level_lookup(active_info::ActiveInfo, level) = get_level_lookup(active_info)[level]

"""
    get_level_sets(active_info)

Return the set of active level-wise ids for each level.
"""
get_level_sets(active_info::ActiveInfo) = active_info.level_sets

"""
    get_level_set(active_info, level)

Return the set of active level-wise ids on `level`.
"""
get_level_set(active_info::ActiveInfo, level) = get_level_sets(active_info)[level]

"""
    get_level_num_ids(active_info, level)

Return the number of active objects on `level`.
"""
function get_level_num_ids(active_info::ActiveInfo, level::Int)
    level_cum_num_ids = get_level_cum_num_ids(active_info)

    return level_cum_num_ids[level + 1] - level_cum_num_ids[level]
end

"""
    get_num_levels(active_info)

Return the number of hierarchical levels.
"""
function get_num_levels(active_info::ActiveInfo)
    return length(get_level_ids(active_info))
end

"""
    get_num_objects(active_info)

Return the total number of active objects across all levels.
"""
function get_num_objects(active_info::ActiveInfo)
    return active_info.level_cum_num_ids[end]
end

"""
    get_level(active_info, hier_id)

Return the level containing the hierarchical object `hier_id`.
"""
function get_level(active_info::ActiveInfo, hier_id::Int)
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

    return level_ids[hier_id - prev_level_cum_num_ids]
end

"""
    convert_to_hier_id(active_info, level, level_id)

Convert the level-wise object `level_id` on `level` to its hierarchical index.
"""
function convert_to_hier_id(active_info::ActiveInfo, level::Int, level_id::Int)
    level_id_count = get_level_lookup(active_info, level)[level_id]
    prev_level_cum_num_ids = get_level_cum_num_ids(active_info, level - 1)

    return prev_level_cum_num_ids + level_id_count
end

############################################################################################
#                                      Field changes                                       #
############################################################################################

"""
    update!(active_info, level, remove, add)

Update the active objects after refining `level`.

The objects in `remove` are removed from `level`, while the objects in `add` become active
on `level + 1`. If `level` is the finest level and `add` is non-empty, a new refinement
level is created automatically.

The cumulative numbering is updated accordingly.

Returns the modified `active_info`.
"""
function update!(active_info::ActiveInfo, level::Int, remove::Vector{Int}, add::Vector{Int})
    num_levels = get_num_levels(active_info)
    if level == num_levels && !isempty(add)
        add_level!(active_info)
        num_levels += 1
    end

    setdiff!(get_level_ids(active_info, level), remove)
    setdiff!(get_level_set(active_info, level), remove)
    union!(get_level_ids(active_info, level + 1), add)
    union!(get_level_set(active_info, level + 1), add)

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
    add_level!(active_info)

Returns `active_info` after appending to it an empty level.
"""
function add_level!(active_info::ActiveInfo)
    push!(get_level_ids(active_info), Int[])
    push!(get_level_sets(active_info), Set{Int}())
    push!(get_level_lookup(active_info), Dict{Int, Int}())
    append!(get_level_cum_num_ids(active_info), get_level_cum_num_ids(active_info)[end])

    return active_info
end

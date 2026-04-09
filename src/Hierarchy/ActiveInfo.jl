############################################################################################
#                                        Structure                                         #
############################################################################################

"""
    struct ActiveInfo

Contains information about active objects in a hierarchical construction. The indexing in
the hierarchical space is such that the index of an object in level `l-1` is always smaller
than that of an object in level `l`.

# Fields
- `level_ids::Vector{Vector{Int}}`: Per level collection of active objects.
    `level_ids[l][i]` gives the id in level `l` of the object indicated by `i`,
    not the hierarchical id of the overall set of objects.
- `level_cum_num_ids::Vector{Int}`: Total number of active objects up to a certain level,
    i.e. `level_cum_num_ids[l]=sum(length.(level_ids[1:l-1]))`. First entry is always 0 for
    ease of use.
"""
struct ActiveInfo
    level_ids::Vector{Vector{Int}}
    level_cum_num_ids::Vector{Int}

    function ActiveInfo(level_ids::Vector{Vector{Int}})
        level_cum_num_ids = [0; cumsum(map(length, level_ids))]

        return new(level_ids, level_cum_num_ids)
    end
end

############################################################################################
#                                         Getters                                          #
############################################################################################

function get_level_ids(active_info::ActiveInfo)
    return active_info.level_ids
end

function get_level_ids(active_info::ActiveInfo, level::Int)
    return get_level_ids(active_info)[level]
end

function get_level_cum_num_ids(active_info::ActiveInfo)
    return active_info.level_cum_num_ids
end

function get_level_cum_num_ids(active_info::ActiveInfo, level::Int)
    return get_level_cum_num_ids(active_info)[level + 1]
end

function get_level_num_ids(active_info::ActiveInfo, level::Int)
    level == 0 ? (return 0) : nothing

    level_cum_num_ids = get_level_cum_num_ids(active_info)

    return level_cum_num_ids[level + 1] - level_cum_num_ids[level]
end

function get_num_levels(active_info::ActiveInfo)
    return length(get_level_ids(active_info))
end

function get_num_objects(active_info::ActiveInfo)
    return sum(length, get_level_ids(active_info))
end

function get_num_active(active_info::ActiveInfo)
    return active_info.level_cum_num_ids[end]
end

function get_level(active_info::ActiveInfo, hier_id::Int)
    return findlast(x -> x < hier_id, get_level_cum_num_ids(active_info))
end

############################################################################################
#                                       Conversions                                        #
############################################################################################

function convert_to_level_id(active_info::ActiveInfo, hier_id::Int)
    object_level = get_level(active_info, hier_id)

    return get_level_ids(active_info, object_level)[hier_id - get_level_cum_num_ids(
        active_info, object_level - 1
    )]
end

"""
	convert_to_level_and_level_id(active_info::ActiveInfo, hier_id::Int)

Returns the `level` and `level_id` that correspond to hierarchical index `hier_id`.
"""
function convert_to_level_and_level_id(active_info::ActiveInfo, hier_id::Int)
    level = get_level(active_info, hier_id)
    level_id = get_level_ids(active_info, level)[hier_id - get_level_cum_num_ids(
        active_info, level - 1
    )]

    return level, level_id
end

function convert_to_level_ids(active_info::ActiveInfo)
    num_levels = get_num_levels(active_info)

    level_ids = [Int[] for _ in 1:num_levels]

    for i in 1:get_num_objects(active_info)
        level, level_id = convert_to_level_and_level_id(active_info, i)

        append!(level_ids[level], level_id)
    end

    return level_ids
end

function convert_to_hier_id(active_info::ActiveInfo, level::Int, level_id::Int)
    level_id_count = findfirst(id -> id == level_id, get_level_ids(active_info, level))

    return level_id_count + get_level_cum_num_ids(active_info, level - 1)
end

############################################################################################
#                                      Field changes                                       #
############################################################################################

function update!(active_info::ActiveInfo, level::Int, remove::Vector{Int}, add::Vector{Int})
    num_levels = get_num_levels(active_info)
    if level == num_levels && !isempty(add)
        add_level!(active_info)
    end

    setdiff!(get_level_ids(active_info)[level], remove)
    union!(get_level_ids(active_info)[level + 1], add)
    level_cum_num_ids = get_level_cum_num_ids(active_info)
    for l in level:(level + 1)
        level_cum_num_ids[l + 1] =
            level_cum_num_ids[l] + length(get_level_ids(active_info)[l])
    end

    return active_info
end

function add_level!(active_info::ActiveInfo)
    push!(get_level_ids(active_info), Int[])
    append!(get_level_cum_num_ids(active_info), get_level_cum_num_ids(active_info)[end])

    return active_info
end

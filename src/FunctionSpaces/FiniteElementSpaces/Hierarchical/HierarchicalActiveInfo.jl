"""
    struct HierarchicalActiveInfo

Contains information about active objects in a hierarchical construction. The indexing in
the hierarchical space is such that the index of an object in level l-1 is always less than
that of an object in level l.

# Fields
- `level_ids::Vector{Vector{Int}}`: Per level collection of active objects.
    'level_ids[l][i]' gives the id in level 'l' of the object indicated by 'i',
    not the hierarchical id of the overall set of objects.
- `level_cum_num_ids::Vector{Int}`: Total number of active objects up to a certain level,
    i.e. 'level_cum_num_ids[l]=sum(length.(level_ids[1:l-1]))'. First entry is always 0 for
    ease of use.
"""
mutable struct HierarchicalActiveInfo
    level_ids::Vector{Vector{Int}}
    level_cum_num_ids::Vector{Int}

    function HierarchicalActiveInfo(level_ids::Vector{Vector{Int}})
		level_cum_num_ids = [0; cumsum(map(length, level_ids))]

        return new(level_ids, level_cum_num_ids)
    end
end

# Basis getters for HierarchicalActiveInfo

function get_level_ids(active_info::HierarchicalActiveInfo)
    return active_info.level_ids
end

function get_level_ids(active_info::HierarchicalActiveInfo, level::Int)
    return get_level_ids(active_info)[level]
end

function get_level_cum_num_ids(active_info::HierarchicalActiveInfo)
    return active_info.level_cum_num_ids
end

function get_level_cum_num_ids(active_info::HierarchicalActiveInfo, level::Int)
    return get_level_cum_num_ids(active_info)[level + 1]
end

function get_level_num_ids(active_info::HierarchicalActiveInfo, level::Int)
    level == 0 ? (return 0) : nothing

    level_cum_num_ids = get_level_cum_num_ids(active_info)

    return level_cum_num_ids[level + 1] - level_cum_num_ids[level]
end

function get_num_levels(active_info::HierarchicalActiveInfo)
    return length(get_level_ids(active_info))
end

function get_num_objects(active_info::HierarchicalActiveInfo)
    return sum(length.(get_level_ids(active_info)))
end

function get_num_active(active_info::HierarchicalActiveInfo)
    return active_info.level_cum_num_ids[end]
end

function get_level(active_info::HierarchicalActiveInfo, hier_id::Int)
    return findlast(x -> x < hier_id, get_level_cum_num_ids(active_info))
end

# Other basic functionality

function convert_to_level_id(active_info::HierarchicalActiveInfo, hier_id::Int)
    object_level = get_level(active_info, hier_id)

    return get_level_ids(active_info, object_level)[hier_id - get_level_cum_num_ids(
        active_info, object_level - 1
    )]
end

function convert_to_level_and_level_id(active_info::HierarchicalActiveInfo, hier_id::Int)
	level = get_level(active_info, hier_id)
	level_id = get_level_ids(active_info, level)[hier_id - get_level_cum_num_ids(
        active_info, level - 1
    )]
	

    return level, level_id
end

function convert_to_level_ids(active_info::HierarchicalActiveInfo)
    num_levels = get_num_levels(active_info)

    level_ids = [Int[] for _ in 1:num_levels]

    for i in 1:get_num_objects(active_info)
        level, level_id = convert_to_level_and_level_id(active_info, i)

        append!(level_ids[level], level_id)
    end

    return level_ids
end

function convert_to_hier_id(active_info::HierarchicalActiveInfo, level::Int, level_id::Int)
    level_id_count = findfirst(id -> id == level_id, get_level_ids(active_info, level))

    return level_id_count + get_level_cum_num_ids(active_info, level - 1)
end

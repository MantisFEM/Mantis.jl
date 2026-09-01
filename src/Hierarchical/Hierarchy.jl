############################################################################################
#                                    AbstractHierarchy                                     #
############################################################################################

abstract type AbstractHierarchy{L, S} end

############################################################################################
#                                         Getters                                          #
############################################################################################

"""
    get_scalings(hierarchy)

Return the scaling relations defining the hierarchy.
"""
get_scalings(hierarchy::AbstractHierarchy) = hierarchy.scalings

"""
    get_scaling(hierarchy, level)

Return the scaling relation on `level`.
"""
get_scaling(hierarchy::AbstractHierarchy, level::Int) = get_scalings(hierarchy)[level]

"""
    get_active_info(hierarchy)

Return the active object information associated with the hierarchy.
"""
get_active_info(hierarchy::AbstractHierarchy) = hierarchy.active_info

"""
    get_num_levels(::AbstractHierarchy{L})

Return the number of hierarchical levels.
"""
get_num_levels(::AbstractHierarchy{L}) where {L} = L

function get_descendants(hierarchy::AbstractHierarchy{L}, id::Int; last_level=L) where {L}
    level, level_id = convert_to_level_and_level_id(hierarchy, id)
    level == L && (return Int[])
    descendants = Vector{Vector{Int}}(undef, last_level - level)
    descendants[1] = get_children(get_scaling(hierarchy, level), level_id)
    for i in
        Iterators.dropwhile(i -> i == 1 || level + i > last_level, eachindex(descendants))
        descendants[i] = mapreduce(
            d -> get_children(get_scaling(hierarchy, level + i - 1), d),
            vcat,
            descendants[i - 1],
        )
        unique!(descendants[i])
    end

    return descendants
end

############################################################################################
#                                        Iteration                                         #
############################################################################################

function Base.iterate(hierarchy::AbstractHierarchy{L}) where {L}
    scalings = get_scalings(hierarchy)

    return (get_parent(first(scalings)), Val(2))
end

function Base.iterate(hierarchy::AbstractHierarchy{L}, ::Val{l}) where {L, l}
    scalings = get_scalings(hierarchy)
    isnothing(parent) && return nothing

    return (get_parent(scalings[l]), Val(l + 1))
end

function Base.iterate(hierarchy::AbstractHierarchy{L}, ::Val{L}) where {L}
    scalings = get_scalings(hierarchy)
    child = get_child(scalings[L - 1])
    isnothing(child) && return nothing

    return (child, Val(0))
end

function Base.iterate(::AbstractHierarchy{L}, ::Val{0}) where {L}
    return nothing
end

Base.length(hierarchy::AbstractHierarchy) = get_num_levels(hierarchy)

get_sets(hierarchy::AbstractHierarchy) = get_sets(get_scalings(hierarchy))

get_set(hierarchy::AbstractHierarchy, i::Int) = get_sets(hierarchy)[i]

############################################################################################
#                                   ActiveInfo Interface                                   #
############################################################################################

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

function update!(
    hierarchy::AbstractHierarchy,
    level::Int,
    remove::Vector{Int},
    add::Vector{Int},
    scaling::AbstractScaling,
)
    active_info = update!(get_active_info(hierarchy), level, remove, add)

    return _maybe_add_level!(hierarchy, scaling, active_info)
end

function add_level!(hierarchy::H, scaling::AbstractScaling) where {H <: AbstractHierarchy}
    active_info = add_level!(get_active_info(hierarchy))
    scalings = add_scaling(hierarchy, scaling)
    constructor = Base.typename(H).wrapper

    return constructor(active_info, scalings...)
end

function add_level!(
    hierarchy::H, scaling::AbstractScaling, active_info::ActiveInfo
) where {H <: AbstractHierarchy}
    scalings = add_scaling(hierarchy, scaling)
    constructor = Base.typename(H).wrapper

    return constructor(active_info, scalings...)
end

function add_scaling(hierarchy::AbstractHierarchy, scaling::AbstractScaling)
    return (get_scalings(hierarchy)..., scaling)
end

#=

# WARNING:
  This operation is inherently type unstable. However, it is limited to a Union of 2
  options:
      1. AbstractHierarchy{L, ...}
      2. AbstractHierarchy{L+1, ...}
  The @noinline hopefully stops this instability from propagating. `update!` callers should
  barrier `update!` as much as possible.
  See the example below.
=#
@noinline function _maybe_add_level!(
    hierarchy::AbstractHierarchy{L}, scaling::AbstractScaling, active_info::ActiveInfo
) where {L}
    if L < get_num_levels(active_info)
        return add_level!(hierarchy, scaling, active_info)
    else
        return hierarchy
    end
end

#= 

This method does not barrier `update!`, so `last_obj` will be `Union{SomeType{L,...},
SomeType{L+1,...}}`. As a consequence, `n` can not infer `L` or `L+1` from the `Union` type,
resulting in `n::Any`, and an output of `::Val`.

function nbarrier(hierarchy, scaling)
    hierarchy = update!(hierarchy, 1, Int[], Int[], scaling)
    obj = get_sets(hierarchy)
    last_obj = last(obj)
    n = first(typeof(last_obj).parameters)

    return Val(n)
end


This method barriers `update!` correctly. As before, `update!` gets inferred as
`Union{AbstractHierarchy{L,...}, AbstractHierarchy{L+1,...}}`. However, since this
immediately dispatched to the type-stable `_wbarrier`, each branch of the `Union` gets
correctly inferred as `Val{L}` and `Val{L+1}`, and so the output type is `Union{Val{L},
Val{L+1}}`.

function wbarrier(hierarchy, scaling)
    hierarchy = update!(hierarchy, 1, Int[], Int[], scaling)
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

The scaling on level `i` relates the objects on level `i-1` to those on level `i`, while
the associated `ActiveInfo` specifies which objects are active on each level.

# Fields
- `scalings::S`: See [`AbstractScaling`](@ref).
- `active_info::ActiveInfo`: See [`ActiveInfo`](@ref).
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
                        "Consecutive scalings must have matching child/parent objects. " *
                        "Failed at level $(i).",
                    ),
                )
            end
        end

        # Each scaling corresponds to two levels
        L = LS + 1
        if L != get_num_levels(active_info)
            throw(
                ArgumentError(
                    "Number of levels in `active_info` does not match number of scalings." *
                    " Got $(get_num_levels(active_info)) and $(L), respectively.",
                ),
            )
        end

        # TODO: Add warning for unstable scalings

        return new{L, typeof(scalings)}(active_info, scalings)
    end
end

############################################################################################
#                                     NestedHierarchy                                      #
############################################################################################

struct NestedHierarchy{L, S} <: AbstractHierarchy{L, S}
    active_info::ActiveInfo
    scalings::S
    nested_ids::Vector{Set{Int}}

    function NestedHierarchy(
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

        nested_ids = _nested_ids(scalings, active_info)
        check_tree && _check_tree(scalings, active_info)

        return new{L, typeof(scalings)}(active_info, scalings, nested_ids)
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

                push!(children, c)
            end

            # Swap to descend into nested children
            parents, children = children, parents
        end
    end

    return true
end

function _nested_ids(scalings, active_info)
    nested_ids = deepcopy(get_level_sets(active_info))
    L = get_num_levels(active_info)
    for l in L:-1:2, B in nested_ids[l]
        union!(nested_ids[l - 1], get_parents(scalings[l - 1], B))
    end

    return nested_ids
end

get_nested_ids(hierarchy::NestedHierarchy) = hierarchy.nested_ids
get_nested_ids(hierarchy::NestedHierarchy, level::Int) = get_nested_ids(hierarchy)[level]

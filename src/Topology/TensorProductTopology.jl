############################################################################################
#                                        Structure                                         #
############################################################################################
"""
    ProductBlock{num_factors}

The geometric objects of a [`TensorProductTopology`](@ref) that are products of factor
objects of fixed dimensions.

A geometric object of dimension `d` of a tensor-product topology is the product of one
geometric object per factor, whose dimensions sum to `d`. For instance, an edge of the
product of two 1D topologies is either an (edge × vertex) or a (vertex × edge) product. All
objects sharing the same split of dimensions over the factors form a block, and the objects
of dimension `d` are numbered block after block.

# Fields
- `object_dims::NTuple{num_factors, Int}`: The dimension of the object in each factor.
- `offset::Int`: The number of objects of the same dimension in all preceding blocks.
- `cartesian_ids::CartesianIndices{num_factors}`: Converts the id of an object within the
    block into the global ids of its factor objects.
- `linear_ids::LinearIndices{num_factors}`: The inverse of `cartesian_ids`.
"""
struct ProductBlock{num_factors}
    object_dims::NTuple{num_factors, Int}
    offset::Int
    cartesian_ids::CartesianIndices{num_factors, NTuple{num_factors, Base.OneTo{Int}}}
    linear_ids::LinearIndices{num_factors, NTuple{num_factors, Base.OneTo{Int}}}
end

Base.length(block::ProductBlock) = length(block.cartesian_ids)

"""
    TensorProductTopology{
        manifold_dim, ir_dim, num_patches, PT, num_factors, F
    } <: AbstractTopology{manifold_dim, ir_dim, num_patches, PT}

Topology of the tensor product of the topologies `factors`, evaluated lazily.

The incidence relations are never assembled. Instead, `topology[i, k]` returns a
[`TensorProductIncidenceRelation`](@ref) whose rows are computed from the rows of the factor
topologies when they are requested. Constructing a tensor-product topology therefore takes
time and memory independent of the size of its factors.

# Structure
Every geometric object of the product is the product of one object per factor, and its
dimension is the sum of their dimensions. An object contains another object if and only if
each of its factor objects contains (or is) the corresponding factor object of the other.
The objects of a given dimension are numbered block by block, see [`ProductBlock`](@ref),
with the ids of the factor objects varying fastest for the first factor. In particular, the
patches form a single block: patch `p` is the product of the factor patches
`Tuple(CartesianIndices(map(get_num_patches, factors))[p])`, as in a
`Geometry.TensorProductGeometry`.

# Local numbering
Each object of dimension `d` is shaped like the tensor-product patch of dimension `d` (see
[`tensor_product_patch`](@ref)), and the row `topology[d + 1, k][id]` with `k < d + 1` lists
the objects contained in object `id` in the local order of that patch. The position (see
[`id_to_position`](@ref)) of a contained object is the concatenation of its positions within
the factor objects, so that the axes of a product object follow the order of the factors.
This requires the factors to follow the same convention, which [`MeshTopology`](@ref),
[`SkeletonTopology`](@ref) and `TensorProductTopology` itself all do. The rows
`topology[i, k][id]` with `k > i` are sorted in increasing order.

All stored ids are positive: how a patch traverses a shared object is given by the order of
the vertices in its rows, see [`compute_neighbours`](@ref).

# Constructors
- `TensorProductTopology(factors::NTuple{num_factors, AbstractTopology})`: General
    constructor. The factors must be made of tensor-product patches, and their manifold
    dimensions must sum to at most 3.
- `TensorProductTopology(factors::AbstractTopology...)`: Same as above.

# Fields
- `topological_patch::PT`: The [`AbstractTensorProductPatch`](@ref) of dimension
    `manifold_dim`.
- `factors::F`: The tuple of factor topologies.
- `blocks::NTuple{ir_dim, Vector{ProductBlock{num_factors}}}`: The blocks of objects of
    each dimension.
- `num_geometric_objects::NTuple{ir_dim, Int}`: Total number of global geometric objects
    per topological dimension.
- `local_id_tables::NTuple{ir_dim, NTuple{ir_dim, Matrix{Array{Int, num_factors}}}}`: For
    dimension ids `i > k` and blocks `b_i` and `b_k` of these dimensions, the table
    `local_id_tables[i][k][b_i, b_k]` returns the local id of a contained object from the
    local ids of its factor objects within the factor objects of the container. See
    [`_local_id_table`](@ref).
"""
struct TensorProductTopology{manifold_dim, ir_dim, num_patches, PT, num_factors, F} <:
       AbstractTopology{manifold_dim, ir_dim, num_patches, PT}
    topological_patch::PT
    factors::F
    blocks::NTuple{ir_dim, Vector{ProductBlock{num_factors}}}
    num_geometric_objects::NTuple{ir_dim, Int}
    local_id_tables::NTuple{ir_dim, NTuple{ir_dim, Matrix{Array{Int, num_factors}}}}

    function TensorProductTopology(
        factors::F
    ) where {num_factors, F <: NTuple{num_factors, AbstractTopology}}
        _check_factors(factors)

        manifold_dim = sum(get_manifold_dim, factors)
        ir_dim = manifold_dim + 1
        num_patches = prod(get_num_patches, factors)
        topological_patch = tensor_product_patch(Val(manifold_dim))

        blocks = _product_blocks(factors, Val(ir_dim))
        num_geometric_objects = map(dim_blocks -> sum(length, dim_blocks), blocks)
        local_id_tables = _local_id_tables(blocks)

        return new{
            manifold_dim, ir_dim, num_patches, typeof(topological_patch), num_factors, F
        }(
            topological_patch, factors, blocks, num_geometric_objects, local_id_tables
        )
    end
end

TensorProductTopology(factors::AbstractTopology...) = TensorProductTopology(factors)

"""
    _check_factors(factors::Tuple)

Throw an `ArgumentError` if `factors` is empty or if one of the factors is not made of
[`AbstractTensorProductPatch`](@ref)es, whose local numbering is needed to number the
objects of the product.
"""
function _check_factors(factors::Tuple)
    if isempty(factors)
        throw(
            ArgumentError("A tensor-product topology needs at least one factor topology.")
        )
    end
    for (factor_id, factor) in enumerate(factors)
        patch = get_topological_patch(factor)
        if !(patch isa AbstractTensorProductPatch)
            throw(
                ArgumentError(
                    LazyString(
                        "Factor ",
                        factor_id,
                        " is made of patches of type ",
                        typeof(patch),
                        ", but a tensor-product topology can only be formed from ",
                        "topologies made of tensor-product patches.",
                    ),
                ),
            )
        end
    end

    return nothing
end

"""
    _product_blocks(factors, ::Val{ir_dim})

Return, for each dimension, the vector of [`ProductBlock`](@ref)s of the objects of that
dimension of the tensor product of `factors`.

The blocks of a dimension are ordered by their split of dimensions over the factors, with
the dimension of the first factor varying fastest.
"""
function _product_blocks(
    factors::NTuple{num_factors, AbstractTopology}, ::Val{ir_dim}
) where {num_factors, ir_dim}
    factor_dim_ranges = map(factor -> 0:get_manifold_dim(factor), factors)

    return ntuple(Val(ir_dim)) do dim_id
        blocks = ProductBlock{num_factors}[]
        offset = 0
        # Iterators.product varies the first factor fastest, like CartesianIndices.
        for object_dims in Iterators.product(factor_dim_ranges...)
            sum(object_dims) == dim_id - 1 || continue

            num_objects = map((factor, dim) -> size(factor, dim + 1), factors, object_dims)
            cartesian_ids = CartesianIndices(num_objects)
            push!(
                blocks,
                ProductBlock(
                    object_dims, offset, cartesian_ids, LinearIndices(cartesian_ids)
                ),
            )
            offset += length(cartesian_ids)
        end

        return blocks
    end
end

"""
    _is_contained(target_dims, container_dims)

Return `true` if the objects of a block with factor dimensions `target_dims` can be
contained in the objects of a block with factor dimensions `container_dims`, i.e. if each
factor object of the former is at most as high-dimensional as that of the latter.
"""
function _is_contained(
    target_dims::NTuple{num_factors, Int}, container_dims::NTuple{num_factors, Int}
) where {num_factors}
    return all(map(<=, target_dims, container_dims))
end

"""
    _num_sub_objects(container_dim::Int, target_dim::Int)

Return the number of objects of dimension `target_dim` contained in a tensor-product patch
of dimension `container_dim`, which is one if both dimensions are equal.
"""
function _num_sub_objects(container_dim::Int, target_dim::Int)
    return binomial(container_dim, target_dim) * 2^(container_dim - target_dim)
end

"""
    _sub_object_position(container_dim::Int, target_dim::Int, local_id::Int)

Return the position of the `local_id`-th object of dimension `target_dim` of the
tensor-product patch of dimension `container_dim`, see [`id_to_position`](@ref).
"""
function _sub_object_position(container_dim::Int, target_dim::Int, local_id::Int)
    # An object only contains itself, which spans all directions.
    container_dim == target_dim && return ntuple(_ -> 0, container_dim)

    return id_to_position(tensor_product_patch(Val(container_dim)), target_dim, local_id)
end

"""
    _local_id_table(container_dims, target_dims)

Return the array whose entry `[j_1, ..., j_n]` is the local id, within a product object with
factor dimensions `container_dims`, of the product of the `j_f`-th objects of dimension
`target_dims[f]` of its factor objects.

The factor objects number the objects they contain as their tensor-product patch does (see
[`TensorProductTopology`](@ref)), so this table is the same for all product objects of a
block and is computed once, at construction.
"""
function _local_id_table(
    container_dims::NTuple{num_factors, Int}, target_dims::NTuple{num_factors, Int}
) where {num_factors}
    container_patch = tensor_product_patch(Val(sum(container_dims)))
    num_sub_objects = map(_num_sub_objects, container_dims, target_dims)

    table = Array{Int, num_factors}(undef, num_sub_objects)
    for factor_local_ids in CartesianIndices(num_sub_objects)
        factor_positions = map(
            _sub_object_position, container_dims, target_dims, Tuple(factor_local_ids)
        )
        position = foldl((a, b) -> (a..., b...), factor_positions; init=())
        table[factor_local_ids] = position_to_id(container_patch, position)
    end

    return table
end

"""
    _local_id_tables(blocks::NTuple{ir_dim, Vector{ProductBlock{num_factors}}})

Return the tables of [`_local_id_table`](@ref) for all pairs of blocks, see the field
`local_id_tables` of [`TensorProductTopology`](@ref). Pairs of blocks whose objects cannot
contain each other get an empty table.
"""
function _local_id_tables(
    blocks::NTuple{ir_dim, Vector{ProductBlock{num_factors}}}
) where {ir_dim, num_factors}
    no_table = Array{Int, num_factors}(undef, ntuple(_ -> 0, Val(num_factors)))

    return ntuple(Val(ir_dim)) do container_dim_id
        return ntuple(Val(ir_dim)) do target_dim_id
            container_blocks = blocks[container_dim_id]
            target_blocks = blocks[target_dim_id]
            tables = fill(no_table, length(container_blocks), length(target_blocks))
            # Only the relations listing contained objects need a local numbering.
            target_dim_id < container_dim_id || return tables

            for (container_block_id, container_block) in enumerate(container_blocks),
                (target_block_id, target_block) in enumerate(target_blocks)

                container_dims = container_block.object_dims
                target_dims = target_block.object_dims
                _is_contained(target_dims, container_dims) || continue

                tables[container_block_id, target_block_id] = _local_id_table(
                    container_dims, target_dims
                )
            end

            return tables
        end
    end
end

############################################################################################
#                                         Getters                                          #
############################################################################################
"""
    get_factor_topologies(topology::TensorProductTopology)

Return the tuple of factor topologies of `topology`.
"""
get_factor_topologies(topology::TensorProductTopology) = topology.factors

"""
    get_factor_object_ids(
        topology::TensorProductTopology, object_dim::Int, object_id::Int
    )

Return the tuple `(object_dims, factor_object_ids)` describing the object of dimension
`object_dim` and global id `object_id` of `topology` as the product of the objects of
dimension `object_dims[f]` and global id `factor_object_ids[f]` of each factor `f`.

This is the inverse of [`get_product_object_id`](@ref).
"""
function get_factor_object_ids(
    topology::TensorProductTopology, object_dim::Int, object_id::Int
)
    if !(1 ≤ object_dim + 1 ≤ get_incidence_relations_dim(topology)) ||
        !(1 ≤ object_id ≤ size(topology, object_dim + 1))
        throw(BoundsError(topology, (object_dim, object_id)))
    end

    return _factor_object_ids(topology.blocks[object_dim + 1], object_id)
end

"""
    get_product_object_id(
        topology::TensorProductTopology,
        object_dims::NTuple{num_factors, Int},
        factor_object_ids::NTuple{num_factors, Int},
    )

Return the global id of the object of `topology` that is the product of the objects of
dimension `object_dims[f]` and global id `factor_object_ids[f]` of each factor `f`.

This is the inverse of [`get_factor_object_ids`](@ref).
"""
function get_product_object_id(
    topology::TensorProductTopology{manifold_dim, ir_dim, num_patches, PT, num_factors},
    object_dims::NTuple{num_factors, Int},
    factor_object_ids::NTuple{num_factors, Int},
) where {manifold_dim, ir_dim, num_patches, PT, num_factors}
    object_dim_id = sum(object_dims) + 1
    if 1 ≤ object_dim_id ≤ ir_dim
        for block in topology.blocks[object_dim_id]
            if block.object_dims == object_dims
                return _product_object_id(block, factor_object_ids)
            end
        end
    end

    return throw(
        ArgumentError(
            LazyString(
                "The factors of the topology have no objects of dimensions ",
                object_dims,
                "; the factor manifold dimensions are ",
                map(get_manifold_dim, get_factor_topologies(topology)),
                ".",
            ),
        ),
    )
end

"""
    _find_block(blocks::Vector{<:ProductBlock}, object_id::Int)

Return the index of the block of `blocks` that contains the object with global id
`object_id`.
"""
function _find_block(blocks::Vector{<:ProductBlock}, object_id::Int)
    for block_id in eachindex(blocks)
        block = blocks[block_id]
        object_id ≤ block.offset + length(block) && return block_id
    end

    return throw(BoundsError(blocks, object_id))
end

"""
    _factor_object_ids(blocks::Vector{<:ProductBlock}, object_id::Int)

Return the factor dimensions and the factor ids of the object with global id `object_id`
among the objects described by `blocks`.
"""
function _factor_object_ids(blocks::Vector{<:ProductBlock}, object_id::Int)
    block = blocks[_find_block(blocks, object_id)]
    return block.object_dims, Tuple(block.cartesian_ids[object_id - block.offset])
end

"""
    _product_object_id(block::ProductBlock, factor_object_ids::Tuple)

Return the global id of the object of `block` with factor ids `factor_object_ids`.
"""
function _product_object_id(
    block::ProductBlock{num_factors}, factor_object_ids::NTuple{num_factors, Int}
) where {num_factors}
    return block.offset + block.linear_ids[factor_object_ids...]
end

############################################################################################
#                                        Equality                                          #
############################################################################################
# A tensor-product topology is fully determined by its factors. The type is hashed as well
# so that, e.g., a tensor-product topology with one factor differs from that factor.
function Base.hash(topology::TensorProductTopology, h::UInt)
    return hash(get_factor_topologies(topology), hash(TensorProductTopology, h))
end

function Base.:(==)(a::TensorProductTopology, b::TensorProductTopology)
    return get_factor_topologies(a) == get_factor_topologies(b)
end

############################################################################################
#                                   Incidence relations                                    #
############################################################################################
"""
    TensorProductIncidenceRelation{TPT <: TensorProductTopology} <:
        AbstractVector{Vector{Int}}

The incidence relation `topology[container_dim_id, target_dim_id]` of a
[`TensorProductTopology`](@ref), evaluated lazily.

Indexing it with the id of an object of dimension `container_dim_id - 1` computes the row
listing the incident objects of dimension `target_dim_id - 1` from the rows of the factor
topologies; see [`TensorProductTopology`](@ref) for the order of the rows. Use `collect` to
assemble the whole relation.
"""
struct TensorProductIncidenceRelation{TPT <: TensorProductTopology} <:
       AbstractVector{Vector{Int}}
    topology::TPT
    container_dim_id::Int
    target_dim_id::Int
end

Base.IndexStyle(::Type{<:TensorProductIncidenceRelation}) = IndexLinear()

function Base.size(relation::TensorProductIncidenceRelation)
    # As for a MeshTopology, objects are not related to the objects of their own dimension.
    relation.container_dim_id == relation.target_dim_id && return (0,)

    return (size(relation.topology, relation.container_dim_id),)
end

function Base.getindex(relation::TensorProductIncidenceRelation, container_id::Int)
    @boundscheck checkbounds(relation, container_id)

    (; topology, container_dim_id, target_dim_id) = relation
    if target_dim_id < container_dim_id
        return _sub_objects(topology, container_dim_id, target_dim_id, container_id)
    end

    return _super_objects(topology, container_dim_id, target_dim_id, container_id)
end

# Placeholder row for the factors in which the container and the target share their object,
# so that no row of the factor is needed.
const NO_FACTOR_OBJECTS = Int[]

"""
    _factor_rows(factors, container_dims, target_dims, container_factor_ids)

Return, for each factor, the row listing the objects of dimension `target_dims[f]` incident
to the factor object `container_factor_ids[f]` of dimension `container_dims[f]`.
"""
function _factor_rows(
    factors::NTuple{num_factors, AbstractTopology},
    container_dims::NTuple{num_factors, Int},
    target_dims::NTuple{num_factors, Int},
    container_factor_ids::NTuple{num_factors, Int},
) where {num_factors}
    return map(
        factors, container_dims, target_dims, container_factor_ids
    ) do factor, container_dim, target_dim, factor_id
        container_dim == target_dim && return NO_FACTOR_OBJECTS
        return factor[container_dim + 1, target_dim + 1][factor_id]::Vector{Int}
    end
end

"""
    _target_factor_ids(
        factor_rows, container_dims, target_dims, container_factor_ids, row_ids
    )

Return the global ids of the factor objects of the target object made of the `row_ids[f]`-th
entry of each of the `factor_rows`.
"""
function _target_factor_ids(
    factor_rows::NTuple{num_factors, Vector{Int}},
    container_dims::NTuple{num_factors, Int},
    target_dims::NTuple{num_factors, Int},
    container_factor_ids::NTuple{num_factors, Int},
    row_ids::NTuple{num_factors, Int},
) where {num_factors}
    return map(
        factor_rows, container_dims, target_dims, container_factor_ids, row_ids
    ) do row, container_dim, target_dim, factor_id, row_id
        container_dim == target_dim && return factor_id
        # The sign of a stored id encodes orientation, not identity.
        return abs(row[row_id])
    end
end

"""
    _sub_objects(
        topology::TensorProductTopology,
        container_dim_id::Int,
        target_dim_id::Int,
        container_id::Int,
    )

Return the global ids of the objects of dimension `target_dim_id - 1` contained in the object
`container_id` of dimension `container_dim_id - 1`, in the local order of the tensor-product
patch of dimension `container_dim_id - 1`.
"""
function _sub_objects(
    topology::TensorProductTopology,
    container_dim_id::Int,
    target_dim_id::Int,
    container_id::Int,
)
    container_blocks = topology.blocks[container_dim_id]
    container_block_id = _find_block(container_blocks, container_id)
    container_block = container_blocks[container_block_id]
    container_dims = container_block.object_dims
    container_factor_ids = Tuple(
        container_block.cartesian_ids[container_id - container_block.offset]
    )

    row = Vector{Int}(undef, _num_sub_objects(container_dim_id - 1, target_dim_id - 1))
    tables = topology.local_id_tables[container_dim_id][target_dim_id]
    for (target_block_id, target_block) in enumerate(topology.blocks[target_dim_id])
        table = tables[container_block_id, target_block_id]
        # The objects of this block are not contained in those of the container block.
        isempty(table) && continue

        target_dims = target_block.object_dims
        factor_rows = _factor_rows(
            topology.factors, container_dims, target_dims, container_factor_ids
        )
        for row_ids in CartesianIndices(table)
            target_factor_ids = _target_factor_ids(
                factor_rows,
                container_dims,
                target_dims,
                container_factor_ids,
                Tuple(row_ids),
            )
            row[table[row_ids]] = _product_object_id(target_block, target_factor_ids)
        end
    end

    return row
end

"""
    _super_objects(
        topology::TensorProductTopology,
        container_dim_id::Int,
        target_dim_id::Int,
        container_id::Int,
    )

Return the sorted global ids of the objects of dimension `target_dim_id - 1` that contain the
object `container_id` of dimension `container_dim_id - 1`.
"""
function _super_objects(
    topology::TensorProductTopology,
    container_dim_id::Int,
    target_dim_id::Int,
    container_id::Int,
)
    container_dims, container_factor_ids = _factor_object_ids(
        topology.blocks[container_dim_id], container_id
    )

    row = Int[]
    for target_block in topology.blocks[target_dim_id]
        target_dims = target_block.object_dims
        _is_contained(container_dims, target_dims) || continue

        factor_rows = _factor_rows(
            topology.factors, container_dims, target_dims, container_factor_ids
        )
        row_lengths = map(factor_rows, container_dims, target_dims) do factor_row, c, t
            return c == t ? 1 : length(factor_row)
        end
        for row_ids in CartesianIndices(row_lengths)
            target_factor_ids = _target_factor_ids(
                factor_rows,
                container_dims,
                target_dims,
                container_factor_ids,
                Tuple(row_ids),
            )
            push!(row, _product_object_id(target_block, target_factor_ids))
        end
    end

    return sort!(row)
end

############################################################################################
#                                         Indexing                                         #
############################################################################################
function Base.getindex(topology::TensorProductTopology, i::Int, k::Int)
    @boundscheck begin
        ir_dim = get_incidence_relations_dim(topology)
        if !(1 ≤ i ≤ ir_dim && 1 ≤ k ≤ ir_dim)
            throw(BoundsError(topology, (i, k)))
        end
    end

    return TensorProductIncidenceRelation(topology, i, k)
end

############################################################################################
#                                          Sizes                                           #
############################################################################################
Base.size(topology::TensorProductTopology) = topology.num_geometric_objects
function Base.size(topology::TensorProductTopology, geometric_dim_id::Int)
    return size(topology)[geometric_dim_id]
end

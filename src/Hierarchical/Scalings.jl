
############################################################################################
#                                     AbstractScaling                                      #
############################################################################################

"""
    AbstractScaling{P,C,R}

Abstract supertype representing a scaling relation between a parent space and a child space.

A scaling stores the parent and child sets together with relations that allow traversal
between parent and child objects.
"""
abstract type AbstractScaling{P, C, R} end

"""
    get_parent(scaling)

Return the parent set.
"""
get_parent(scaling::AbstractScaling) = scaling.parent

"""
    get_child(scaling)

Return the child set.
"""
get_child(scaling::AbstractScaling) = scaling.child

"""
    get_relations(scaling)

Return the parent-child relations associated with `scaling`.
"""
get_relations(scaling::AbstractScaling) = scaling.relations

"""
    get_children(scaling, id)

Return the children of parent object `id`.
"""
get_children(scaling::AbstractScaling, id) = get_children(get_relations(scaling), id)

"""
    get_parents(scaling, id)

Return the parents of child object `id`.
"""
get_parents(scaling::AbstractScaling, id) = get_parents(get_relations(scaling), id)

@generated function get_sets(scalings::S) where {LS, S <: NTuple{LS, AbstractScaling}}
    ex = Expr(:tuple)

    # First set
    push!(ex.args, :(get_parent(first(scalings))))

    # Intermediate sets
    for i in 2:LS
        push!(ex.args, :(get_parent(scalings[$i])))
    end

    # Last set
    type = last(S.parameters).parameters[2] # C in AbstractScaling{P, C, R}
    if type !== Nothing
        push!(ex.args, :(get_child(last(scalings))))
    end

    return ex
end

############################################################################################
#                                        Relations                                         #
############################################################################################

abstract type RelationType end

struct PC <: RelationType end
struct CP <: RelationType end

abstract type RelationMethod{RT <: RelationType} end

struct RelationEmpty{RT} <: RelationMethod{RT} end

function (::RelationEmpty)(::Int)
    return ()
end

struct RelationExplicit{RT, F <: Function} <: RelationMethod{RT}
    method::F
    function RelationExplicit{RT}(method::F) where {RT, F}
        return new{RT, F}(method)
    end
end

function (relation::RelationExplicit)(id::Int)
    return relation.method(id)
end

"""
    Relations{PC, CP}

Stores the parent-child connectivity of a scaling.

# Fields
- `parent_to_children::PC`: Method relating parent objects to their corresponding children.
- `child_to_parents::CP`: Method relating child objects to their corresponding parents.
"""
struct Relations{RPC <: RelationMethod{PC}, RCP <: RelationMethod{CP}}
    parent_to_children::RPC
    child_to_parents::RCP
end

"""
    get_parent_to_children(relations)

Return the parent-to-children mapping.
"""
get_parent_to_children(relations::Relations) = relations.parent_to_children

"""
    get_child_to_parents(relations)

Return the child-to-parents mapping.
"""
get_child_to_parents(relations::Relations) = relations.child_to_parents

"""
    get_children(relations, id)

Return the children of parent object `id`.
"""
get_children(relations::Relations, id) = get_parent_to_children(relations)(id)

"""
    get_parents(relations, id)

Return the parents of child object `id`.
"""
get_parents(relations::Relations, id) = get_child_to_parents(relations)(id)

############################################################################################
#                                         Scaling                                          #
############################################################################################

"""
    Scaling{P, C, R <: Relations} <: AbstractScaling{P, C, R}

Represents a scaling relation between parent and child tensor-products.

Besides storing the parent and child tensor-products, a `Scaling` provides mappings between
parent and child object indices through its associated `Relations`.
"""
struct Scaling{P, C, R <: Relations} <: AbstractScaling{P, C, R}
    parent::P
    child::C
    relations::R
end

"""
    Scaling(parent)

Construct a placeholder scaling with no child space or scaling relations.

This constructor is primarily intended for one-dimensional spaces that have no scaling
information available.
"""
function Scaling(parent)
    return Scaling(parent, nothing, Relations(RelationEmpty{PC}(), RelationEmpty{CP}()))
end

"""
    Scaling(parent, child, relations...)

Returns a tensor-product scaling relation from the factor scaling relations.
"""
function Scaling(
    parent, child, scalings::NTuple{num_scalings, AbstractScaling}
) where {num_scalings}
    parent_factors = TensorProducts.get_factors(parent)
    child_factors = TensorProducts.get_factors(child)
    for i in eachindex(parent_factors, child_factors)
        if !(parent_factors[i] === get_parent(scalings[i]))
            throw(
                ArgumentError(
                    "factor parent sets must match parent sets in scalings. " *
                    "Failed for index $(i).",
                ),
            )
        elseif !(child_factors[i] === get_child(scalings[i]))
            throw(
                ArgumentError(
                    "factor child sets must match child sets in scalings. " *
                    "Failed for index $(i).",
                ),
            )
        end
    end

    function parent_to_children(id)
        factor_ids = TensorProducts.get_factor_ids(parent, id)
        factor_children = ntuple(
            i -> get_children(scalings[i], factor_ids[i]), num_scalings
        )
        product = Iterators.product(factor_children...)
        child_lin_ids = TensorProducts.get_lin_ids(child)
        children = Iterators.flatten(Iterators.map(c -> child_lin_ids[c...], product))

        return children
    end

    function child_to_parents(id)
        factor_ids = TensorProducts.get_factor_ids(child, id)
        factor_parents = ntuple(i -> get_parents(scalings[i], factor_ids[i]), num_scalings)
        product = Iterators.product(factor_parents...)
        parent_lin_ids = TensorProducts.get_lin_ids(parent)
        parents = Iterators.flatten(Iterators.map(p -> parent_lin_ids[p...], product))

        return parents
    end

    return Scaling(
        parent,
        child,
        Relations(
            RelationExplicit{PC}(parent_to_children), RelationExplicit{CP}(child_to_parents)
        ),
    )
end

function Scaling(parent, child, R::NTuple{num_rs, Relations}) where {num_rs}
    parent_factors = TensorProducts.get_factors(parent)
    child_factors = TensorProducts.get_factors(child)
    scalings = ntuple(num_rs) do i
        return Scaling(parent_factors[i], child_factors[i], R[i])
    end

    return Scaling(parent, child, scalings)
end

############################################################################################
#                                      MatrixScaling                                       #
############################################################################################

struct MatrixScaling{P, C, R <: Relations, M <: AbstractMatrix} <: AbstractScaling{P, C, R}
    parent::P
    child::C
    relations::R
    scaling_matrix::M
end

function MatrixScaling(parent, child, scaling_method)
    scaling_matrix = build_scaling_matrix(parent, child, scaling_method)
    relations = Relations(scaling_matrix)

    return MatrixScaling(parent, child, relations, scaling_matrix)
end

function build_scaling_matrix(parent, child, scaling_method)
    return throw(MethodError(build_scaling_matrix, (parent, child, scaling_method)))
end

function MatrixScaling(
    sets::NTuple{num_sets, Any}, fs::Vararg{Function, num_fs}
) where {num_sets, num_fs}
    num_sets == (num_fs + 1) || throw(
        ArgumentError(
            LazyString(
                "Incorrect number of scaling methods. ",
                "Expected ",
                num_sets - 1,
                ", but got ",
                num_fs,
                ".",
            ),
        ),
    )
    set_1, curr = Iterators.peel(sets)
    set_2, next = Iterators.peel(curr)
    f_1, rem_f = Iterators.peel(fs)
    scaling_matrix = build_scaling_matrix(set_1, set_2, f_1)
    for (set_curr, set_next, f) in zip(curr, next, rem_f)
        scaling_matrix = build_scaling_matrix(set_curr, set_next, f) * scaling_matrix
    end

    relations = Relations(scaling_matrix)

    return MatrixScaling(set_1, last(sets), relations, scaling_matrix)
end

function MatrixScaling(parent, child, fs::NTuple{num_methods, Function}) where {num_methods}
    parent_factors = TensorProducts.get_factors(parent)
    child_factors = TensorProducts.get_factors(child)
    num_factors = length(parent_factors)
    if num_factors != num_methods
        throw(
            ArgumentError(
                LazyString(
                    "Incorrect number of scaling methods. Expected ",
                    num_factors,
                    ", but got ",
                    num_methods,
                    ".",
                ),
            ),
        )
    end

    factor_scalings = ntuple(
        i -> MatrixScaling(parent_factors[i], child_factors[i], fs[i]), num_factors
    )
    scaling = Scaling(parent, child, factor_scalings)
    relations = get_relations(scaling)
    scaling_matrix = LinearAlgebra.kron(
        (get_scaling_matrix(factor_scalings[i]) for i in num_factors:-1:1)...
    )

    return MatrixScaling(parent, child, relations, scaling_matrix)
end

get_scaling_matrix(scaling::MatrixScaling) = scaling.scaling_matrix

function view_scaling_matrix(
    scaling::MatrixScaling, child_indices::AbstractVector, parent_indices::AbstractVector
)
    return view(get_scaling_matrix(scaling), child_indices, parent_indices)
end

"""
    Relations(scaling_matrix)

Construct parent-child relations from a scaling matrix.

The sparsity pattern of `scaling_matrix` encodes which child objects are generated by each
parent object. In particular, the non-zero rows for a given column `j` dictate the children
of parent object `j`.
"""
function Relations(scaling_matrix::AbstractMatrix)
    parent_to_children_vec = Vector{Vector{Int}}(undef, size(scaling_matrix, 2))
    child_to_parents_vec = Vector{Vector{Int}}(undef, size(scaling_matrix, 1))

    @inbounds begin
        for j in axes(scaling_matrix, 2)
            parent_to_children_vec[j] = findall(!iszero, view(scaling_matrix, :, j))
        end

        for i in axes(scaling_matrix, 1)
            child_to_parents_vec[i] = findall(!iszero, view(scaling_matrix, i, :))
        end
    end

    parent_to_children(id) = parent_to_children_vec[id]
    child_to_parents(id) = child_to_parents_vec[id]

    return Relations(
        RelationExplicit{PC}(parent_to_children), RelationExplicit{CP}(child_to_parents)
    )
end

function Relations(scaling_matrix::M) where {M <: SparseArrays.SparseMatrixCSC}
    #=
    The `scaling_matrix` gives the parent_to_children relations, so the transpose gives
    child_to_parents. We transpose instead of doing row lookup for efficiency.
    =#
    matrix_data = SparseArrays.findnz(scaling_matrix)
    transpose_matrix = SparseArrays.sparse(matrix_data[2], matrix_data[1], matrix_data[3])

    # From parent id, find children (non-zeros of `scaling_matrix`)
    parent_to_children(id) =
        view(scaling_matrix.rowval, SparseArrays.nzrange(scaling_matrix, id))
    # From child id, find parents (non-zeros of `transpose_matrix`)
    child_to_parents(id) =
        view(transpose_matrix.rowval, SparseArrays.nzrange(transpose_matrix, id))

    return Relations(
        RelationExplicit{PC}(parent_to_children), RelationExplicit{CP}(child_to_parents)
    )
end

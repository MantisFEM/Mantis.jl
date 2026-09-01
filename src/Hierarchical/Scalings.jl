
############################################################################################
#                                        Relations                                         #
############################################################################################

"""
    RelationType

Supertype indicating the direction of a parent-child relation. Possible values are
[`PC`](@ref) and [`CP`](@ref).
"""
abstract type RelationType end

"""
    PC

Singleton indicating a relation from parent objects to their children.
"""
struct PC <: RelationType end

"""
    CP

Singleton indicating a relation from child objects to their parents.
"""
struct CP <: RelationType end
"""
    RelationMethod{RT <: RelationType}

Supertype for all relation methods. Concrete subtypes store any methods or state necessary
to produce parent-child relations. The order of the direction is dictated by `RT`.

# Type Parameters
- `RT <: RelationType`: Indicates the direction of the parent-child relation.
"""
abstract type RelationMethod{RT <: RelationType} end

"""
    RelationExplicit{RT, F <: Function} <: RelationMethod{RT}

A relation method in closed form, depending only on its underlying expression.

!!!note
    This struct is callable on an `id::Int` to produce the given parents or children of
    the object indexed by `id`.

# Fields
- `method::F`: The method defining the relation.

# Type Parameters
- `RT <: RelationType`: Indicates the direction of the parent-child relation.
"""
struct RelationExplicit{RT, F <: Function} <: RelationMethod{RT}
    method::F
    function RelationExplicit{RT}(method::F) where {RT, F}
        return new{RT, F}(method)
    end
end

"""
    (relation::RelationExplicit)(id::Int)

Return either the parents or children of the object indexed by `id`, as dictate by the
associated relation type.
"""
function (relation::RelationExplicit)(id::Int)
    return relation.method(id)
end

"""
    Relations{RPC <: RelationMethod{PC}, RCP <: RelationMethod{CP}}

Stores the full parent-child connectivity of a scaling.

# Fields
- `parent_to_children::RPC`: Method relating parent objects to their corresponding children.
- `child_to_parents::RCP`: Method relating child objects to their corresponding parents.
"""
struct Relations{RPC <: RelationMethod{PC}, RCP <: RelationMethod{CP}}
    parent_to_children::RPC
    child_to_parents::RCP
end

"""
    get_parent_to_children(relations::Relations)

Return the parent-to-children mapping.
"""
get_parent_to_children(relations::Relations) = relations.parent_to_children

"""
    get_child_to_parents(relations::Relations)

Return the child-to-parents mapping.
"""
get_child_to_parents(relations::Relations) = relations.child_to_parents

"""
    get_children(relations::Relations, id)

Return the children of the parent object indexed by `id`.
"""
get_children(relations::Relations, id) = get_parent_to_children(relations)(id)

"""
    get_parents(relations, id)

Return the parents of the child object indexed by `id`.
"""
get_parents(relations::Relations, id) = get_child_to_parents(relations)(id)

############################################################################################
#                                     AbstractScaling                                      #
############################################################################################

"""
    AbstractScaling{P,C,R}

Supertype representing a scaling relation between parent and child objects.

A scaling stores the parent and child sets together with relations that allow traversal
between parent and child objects.

# Type Parameters
- `P`: Type of the parent set.
- `C`: Type of the child set.
- `R`: Type of the relations between parent and child objects.
"""
abstract type AbstractScaling{P, C, R} end

"""
    get_parent(scaling::AbstractScaling)

Return the parent set of `scaling`.
"""
get_parent(scaling::AbstractScaling) = scaling.parent

"""
    get_child(scaling::AbstractScaling)

Return the child set of `scaling`.
"""
get_child(scaling::AbstractScaling) = scaling.child

"""
    get_relations(scaling::AbstractScaling)

Return the parent-child relations associated with `scaling`.
"""
get_relations(scaling::AbstractScaling) = scaling.relations

"""
    get_children(scaling::AbstractScaling, id)

Return the children of the parent object indexed by `id`.
"""
get_children(scaling::AbstractScaling, id) = get_children(get_relations(scaling), id)

"""
    get_parents(scaling::AbstractScaling, id)

Return the parents of the child object indexed by `id`.
"""
get_parents(scaling::AbstractScaling, id) = get_parents(get_relations(scaling), id)

"""
    get_sets(scalings::S) where {LS, S <: NTuple{LS, AbstractScaling}}

Return a tuple whose length is the number of levels of all `scalings` combined, and each
entry the the object set for that level. 
"""
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
#                                         Scaling                                          #
############################################################################################

"""
    Scaling{P, C, R <: Relations} <: AbstractScaling{P, C, R}

Represents a scaling relation between parent and child sets.

Besides storing the parent and child sets, a `Scaling` provides mappings between parent and
child objects through its associated `Relations`.

# Fields
- `parent::P`: The parent set.
- `child::C`: The child set.
- `relations::R`: The parent-child relations.

# Outer Constructors
- [`Scaling(parent)`](@ref)
- [`Scaling(parent, child, scalings::NTuple{num_scalings, AbstractScaling})`](@ref)
- [`Scaling(parent, child, R::NTuple{num_rs, Relations})`](@ref)
"""
struct Scaling{P, C, R <: Relations} <: AbstractScaling{P, C, R}
    parent::P
    child::C
    relations::R
end

"""
    Scaling(parent)

Construct a scaling with no child space, and therefore empty scaling relations.
"""
function Scaling(parent)
    return Scaling(
        parent,
        nothing,
        Relations(RelationExplicit{PC}(_ -> ()), RelationExplicit{CP}(_ -> ())),
    )
end

"""
    Scaling(parent, child, scalings::NTuple{num_scalings, AbstractScaling})

Returns a tensor-product scaling relation from the factor scaling relations.

!!!note
    `parent` and `child` are not required to be of type [`TensorProduct`](@ref), but they
    _must_ implement [`TensorProducts.get_factors`](@ref),
    [`TensorProducts.get_factor_ids`](@ref), and [`TensorProducts.get_lind_ids`](@ref).

# Arguments
- `parent`: The parent set.
- `child`: The child set.
- `scalings::NTuple{num_scalings, AbstractScaling}`: The scaling relations between each
    factor parent and child set.
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

"""
    Scaling(parent, child, R::NTuple{num_rs, Relations}) where {num_rs}

See [`Scaling(parent, child, scalings::NTuple{num_scalings, AbstractScaling})`](@ref).
"""
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

"""
    MatrixScaling{P, C, R, M} <: AbstractScaling{P, C, R}

A parent-child scaling whose relations are provided by a matrix. Besides providing
parent-child indexing of objects, the scaling matrix can also be used to define a change of
basis between the parent and child sets. In particular, if ``P`` are coefficients for a
linear combinations of parent objects, we can express them in terms of child objects as ``P
= CS`` where ``C`` are the children coefficients and ``S`` is the scaling matrix.

See also [`Scaling`](@ref), [`Relations(scaling_matrix::AbstractMatrix)`](@ref).

# Fields
- `parent::P`: The parent set.
- `child::C`: The child set.
- `relations::R`: The parent-child relations.
- `scaling_matrix::M`: The matrix providing the parent-child relations. The matrix has size
    ``dP x dC`` where ``dP`` and ``dC`` are the dimensions of `parent` and `child`,
    respectively.

# Outer Constructors
- [`MatrixScaling(parent, child, scaling_method)`](@ref) 
- [`MatrixScaling(sets::NTuple{num_sets, Any}, scaling_methods::Vararg{Function, num_methods})`](@ref)
- [`MatrixScaling(parent, child, scaling_methods::NTuple{num_methods, Function})`](@ref)
"""
struct MatrixScaling{P, C, R, M} <: AbstractScaling{P, C, R}
    parent::P
    child::C
    relations::R
    scaling_matrix::M
    function MatrixScaling(
        parent::P, child::C, relations::R, scaling_matrix::M
    ) where {P, C, R <: Relations, M <: AbstractMatrix}
        parent_dim = get_num_objects(parent)
        child_dim = get_num_objects(child)
        if size(scaling_matrix, 1) != child_dim
            return throw(
                ArgumentError(
                    LazyString(
                        "Incompatible scaling matrix. ",
                        "The number of rows in `scaling_matrix` must equal the dimension of
                        `child`. ",
                        "Got",
                        size(scaling_matrix, 1),
                        " and ",
                        child_dim,
                    ),
                ),
            )
        end

        if size(scaling_matrix, 2) != parent_dim
            return throw(
                ArgumentError(
                    LazyString(
                        "Incompatible scaling matrix. ",
                        "The number of columns in `scaling_matrix` must equal the dimension
                        of `parent`. ",
                        "Got",
                        size(scaling_matrix, 2),
                        " and ",
                        parent_dim,
                    ),
                ),
            )
        end

        return new{P, C, R, M}(parent, child, relations, scaling_matrix)
    end
end

"""
    MatrixScaling(parent, child, scaling_method)

Return a `MatrixScaling` where the scaling matrix is constructed by calling `scaling_method`
on `parent` and `child`.
"""
function MatrixScaling(parent, child, scaling_method)
    scaling_matrix = build_scaling_matrix(parent, child, scaling_method)
    relations = Relations(scaling_matrix)

    return MatrixScaling(parent, child, relations, scaling_matrix)
end

"""
    MatrixScaling(
        sets::NTuple{num_sets, Any}, scaling_methods::Vararg{Function, num_methods}
    ) where {num_sets, num_fs}

Return a `MatrixScaling` where the scaling matrix between the first element of `sets` and
the last is constructed by iteratively composing `scaling_methods`.
"""
function MatrixScaling(
    sets::NTuple{num_sets, Any}, scaling_methods::Vararg{Function, num_methods}
) where {num_sets, num_methods}
    if num_sets != (num_methods + 1)
        return throw(
            ArgumentError(
                LazyString(
                    "Incorrect number of scaling methods. ",
                    "Expected ",
                    num_sets - 1,
                    ", got ",
                    num_methods,
                ),
            ),
        )
    end

    # We create the first scaling matrix explicity
    parent, parents = Iterators.peel(sets)
    child, children = Iterators.peel(parents)
    scal_method, scaling_methods = Iterators.peel(scaling_methods)
    scaling_matrix = build_scaling_matrix(parent, child, scal_method)
    # Then, we iterate over the remaining sets and scaling methods, and build the
    # composition of scaling methods via matrix multiplication.
    # Note that `parents` and `children` are iterators over the same objects, but `children`
    # is missing the first element. `zip` will therefore iterator over the parent-child
    # pairs, and stop when we are at the last element of `children`.
    for (parent, child, scal_method) in zip(parents, children, scaling_methods)
        scaling_matrix = build_scaling_matrix(parent, child, scal_method) * scaling_matrix
    end

    relations = Relations(scaling_matrix)

    return MatrixScaling(parent, last(sets), relations, scaling_matrix)
end

"""
    MatrixScaling(
        parent, child, scaling_methods::NTuple{num_methods, Function}
    ) where {num_methods}

Return a `MatrixScaling` where the scaling matrix is constructed by tensoring the scaling
matrices resulting from the factor `scaling_methods`.

!!!note
    `parent` and `child` are not required to be of type [`TensorProduct`](@ref), but they
    _must_ implement [`TensorProducts.get_factors`](@ref).
"""
function MatrixScaling(
    parent, child, scaling_methods::NTuple{num_methods, Function}
) where {num_methods}
    parent_factors = TensorProducts.get_factors(parent)
    child_factors = TensorProducts.get_factors(child)
    num_factors = length(parent_factors)
    if num_factors != num_methods
        return throw(
            ArgumentError(
                LazyString(
                    "Incorrect number of scaling methods. Expected ",
                    num_factors,
                    ", got ",
                    num_methods,
                ),
            ),
        )
    end

    if num_factors != length(child_factors)
        return throw(
            ArgumentError(
                LazyString(
                    "Incompatible `parent` and `child` sets. ",
                    "`parent` has ",
                    num_factors,
                    " factors, but `child` has ",
                    length(child_factors),
                ),
            ),
        )
    end

    factor_scalings = ntuple(
        i -> MatrixScaling(parent_factors[i], child_factors[i], scaling_methods[i]),
        num_factors,
    )
    scaling = Scaling(parent, child, factor_scalings)
    relations = get_relations(scaling)
    scaling_matrix = LinearAlgebra.kron(
        (get_scaling_matrix(factor_scalings[i]) for i in num_factors:-1:1)...
    )

    return MatrixScaling(parent, child, relations, scaling_matrix)
end

"""
    build_scaling_matrix(parent, child, scaling_method)

Return the scaling matrix between `parent` and `child` as defined by `scaling_method`.

!!!note
    Must be implement for every concrete type of `parent` and `child`.
"""
function build_scaling_matrix(parent, child, scaling_method)
    return throw(MethodError(build_scaling_matrix, (parent, child, scaling_method)))
end


"""
    get_scaling_matrix(scaling::MatrixScaling)

Return the scaling matrix of `scaling`.
"""
get_scaling_matrix(scaling::MatrixScaling) = scaling.scaling_matrix

"""
    view_scaling_matrix(
    scaling::MatrixScaling, child_indices::AbstractVector, parent_indices::AbstractVector
)

Return a view over the scaling matrix of `scaling` at the given `child_indices` and
`parent_indices`.
"""
function view_scaling_matrix(
    scaling::MatrixScaling, child_indices::AbstractVector, parent_indices::AbstractVector
)
    return view(get_scaling_matrix(scaling), child_indices, parent_indices)
end

"""
    Relations(scaling_matrix::AbstractMatrix)

Construct parent-child relations from a scaling matrix.

The sparsity pattern of `scaling_matrix` encodes which child objects are generated by each
parent object. In particular, the non-zero rows for a given column `j` dictate the children
of parent object `j`, and vice-versa.
"""
function Relations(scaling_matrix::AbstractMatrix)
    # We choose to compute all the parent-child relations in advance and then store, as
    # oppose to computing them on-the-fly.
    # This has the advantage that we only compute them once, but the downside that relations
    # that might be unnecessary are also computed (and stored.)
    # TODO: Might be reasonable at some point to provide a choice between the two options
    # above.
    parent_to_children_vec = Vector{Vector{Int}}(undef, size(scaling_matrix, 2))
    child_to_parents_vec = Vector{Vector{Int}}(undef, size(scaling_matrix, 1))

    @inbounds begin
        for j in axes(scaling_matrix, 2)
            # children of `j` are the non-zero entries of column `j`.
            parent_to_children_vec[j] = findall(!iszero, view(scaling_matrix, :, j))
        end

        for i in axes(scaling_matrix, 1)
            # parents of `i` are the non-zero entries of row `i`.
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
    # The `scaling_matrix` gives the parent_to_children relations, so the transpose gives
    # child_to_parents. We transpose instead of doing row lookup for efficiency.
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

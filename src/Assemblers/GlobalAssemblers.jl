
"""
    assemble(
        weak_form::WeakForm{manifold_dim, LHS, RHS, I},
        dirichlet_bcs::Dict{Int, Float64}=Dict{Int, Float64}();
        lhs_type::Type=spa.SparseMatrixCSC{Float64, Int},
        rhs_type::Type=Vector{Float64},
    ) where {
        manifold_dim,
        num_rows,
        lhs_num_cols,
        rhs_num_cols,
        LHS <: NTuple{
            num_rows, NTuple{lhs_num_cols, Union{Int, Forms.AbstractRealValuedOperator}}
        },
        RHS <: NTuple{
            num_rows, NTuple{rhs_num_cols, Union{Int, Forms.AbstractRealValuedOperator}}
        },
        I,
    }

Assemble the left- and right-hand sides of a discrete Petrov-Galerkin problem for the given
weak-formulation and Dirichlet boundary conditions.

# Arguments

  - `weak_form::WeakForm{manifold_dim, LHS, RHS, I}`: The weak form to assemble.
  - `dirichlet_bcs::Dict{Int, Float64}`: A dictionary containing the Dirichlet boundary
    conditions, where the key is the index of the boundary condition and the value is the
    boundary condition value.
  - `lhs_type::Type`: The type of the left-hand side array. Default is
    `SparseMatrixCSC{Float64, Int}`.
  - `rhs_type::Type`: The type of the right-hand side array. Default is `Vector{Float64}`.

# Returns

  - `lhs::lhs_type`: The assembled left-hand side array.
  - `rhs::rhs_type`: The assembled right-hand side vector.
"""
function assemble(
    weak_form::WeakForm{manifold_dim, LHS, RHS, I},
    dirichlet_bcs::Dict{Int, Float64}=Dict{Int, Float64}();
    lhs_type::Type=spa.SparseMatrixCSC{Float64, Int},
    rhs_type::Type=Vector{Float64},
) where {
    manifold_dim,
    num_rows,
    lhs_num_cols,
    rhs_num_cols,
    LHS <:
    NTuple{num_rows, NTuple{lhs_num_cols, Union{Int, Forms.AbstractRealValuedOperator}}},
    RHS <:
    NTuple{num_rows, NTuple{rhs_num_cols, Union{Int, Forms.AbstractRealValuedOperator}}},
    I,
}
    lhs_expressions = get_lhs_expressions(weak_form)
    rhs_expressions = get_rhs_expressions(weak_form)
    test_offsets, trial_offsets = get_test_offsets(weak_form), get_trial_offsets(weak_form)
    lhs_rows, lhs_cols, lhs_vals = get_pre_allocation(weak_form, "lhs", lhs_type)
    rhs_rows, rhs_cols, rhs_vals = get_pre_allocation(weak_form, "rhs", rhs_type)
    lhs_counts, rhs_counts = 0, 0
    # TODO: The loop over elements should also handle boundary integrals.
    # PERF: We might what to make the loop over elements the inner most one; that way we can
    # skip the 0 expressions instead of checking them everytime.
    for elem_id in 1:get_num_elements(weak_form), row in 1:num_rows
        for col in 1:lhs_num_cols
            lhs_rows, lhs_cols, lhs_vals, lhs_counts = add_expression_contributions!(
                lhs_rows,
                lhs_cols,
                lhs_vals,
                lhs_counts,
                lhs_expressions[row][col],
                elem_id,
                test_offsets[row],
                trial_offsets[col],
            )
        end

        for col in 1:rhs_num_cols
            rhs_rows, rhs_cols, rhs_vals, rhs_counts = add_expression_contributions!(
                rhs_rows,
                rhs_cols,
                rhs_vals,
                rhs_counts,
                rhs_expressions[row][col],
                elem_id,
                test_offsets[row],
                trial_offsets[col],
            )
        end
    end

    zero_rows!(lhs_vals, rhs_vals, lhs_rows, rhs_rows, dirichlet_bcs)
    lhs_size = get_lhs_size(weak_form)
    rhs_size = get_rhs_size(weak_form)
    lhs = build_array(
        lhs_type,
        lhs_rows[1:lhs_counts],
        lhs_cols[1:lhs_counts],
        lhs_vals[1:lhs_counts],
        lhs_size,
    )
    rhs = build_array(
        rhs_type,
        rhs_rows[1:rhs_counts],
        rhs_cols[1:rhs_counts],
        rhs_vals[1:rhs_counts],
        rhs_size,
    )
    lhs, rhs = set_diagonal!(lhs, rhs, dirichlet_bcs)

    return lhs, rhs
end

"""
    get_pre_allocation(
        weak_form::WeakForm, side::String, ::Type{A}
    ) where {T, A <: AbstractArray{T}}

Returns pre-allocated row, column, and value vectors for the left-hand side (lhs) or
right-hand side (rhs) array.

# Arguments

  - `weak_form::WeakForm`: The weak form to use for the pre-allocation.
  - `side::String`: The side of the array to pre-allocate. Must be either "lhs" or "rhs".

# Returns

  - `rows::Vector{Int}`: The pre-allocated row indices.
  - `cols::Vector{Int}`: The pre-allocated column indices.
  - `vals::Vector{T}`: The pre-allocated values.
"""
function get_pre_allocation(
    weak_form::WeakForm, side::String, ::Type{A}
) where {T, A <: AbstractArray{T}}
    nnz_elem = get_estimated_nnz_per_elem(weak_form)
    if side == "lhs"
        nvals = nnz_elem[1] * get_num_evaluation_elements(weak_form)
    elseif side == "rhs"
        nvals = nnz_elem[2] * get_num_evaluation_elements(weak_form)
    else
        throw(ArgumentError("Invalid side: $(side). Must be 'lhs' or 'rhs'."))
    end

    rows = Vector{Int}(undef, nvals)
    if side == "rhs" && A <: AbstractVector
        cols = ones(Int, nvals)
    else
        cols = Vector{Int}(undef, nvals)
    end

    vals = Vector{eltype(A)}(undef, nvals)

    return rows, cols, vals
end

"""
    add_expression_contributions!(
        rows::Vector{Int},
        cols::Vector{Int},
        vals::AbstractVector,
        counts::Int,
        expression,
        element_id::Int,
        test_offset::Int,
        trial_offset::Int,
    )

Updates the row, column, and value vectors with contributions from the specified real-valued
expression at the element given by `element_id`.

# Arguments

  - `rows::Vector{Int}`: The row indices of the array.
  - `cols::Vector{Int}`: The column indices of the array.
  - `vals::AbstractVector`: The values of the array.
  - `counts::Int`: The current count of non-zero entries in the array.
  - `expressions`: The expression to evaluate.
  - `element_id::Int`: The identifier of the element.
  - `test_offsets::Int`: The offset for the test function.
  - `trial_offsets::Int`: The offset for the trial function.

# Returns

  - `rows::Vector{Int}`: The updated row indices of the array.
  - `cols::Vector{Int}`: The updated column indices of the array.
  - `vals::AbstractVector`: The updated values of the array.
  - `counts::Int`: The updated count of non-zero entries in the array.
"""
function add_expression_contributions!(
    rows::Vector{Int},
    cols::Vector{Int},
    vals::AbstractVector,
    counts::Int,
    expression,
    element_id::Int,
    test_offset::Int,
    trial_offset::Int,
)
    if expression == 0
        return rows, cols, vals, counts
    end

    block_eval, block_indices = Forms.evaluate(expression, element_id)
    for ord_id in CartesianIndices(block_eval)
        counts += 1
        for exp_id in eachindex(block_indices)
            if exp_id == 1
                rows[counts] = block_indices[1][ord_id[1]] + test_offset
            elseif exp_id == 2
                cols[counts] = block_indices[2][ord_id[2]] + trial_offset
            end
        end

        vals[counts] = block_eval[ord_id]
    end

    return rows, cols, vals, counts
end

"""
    zero_rows!(
        lhs_vals::AbstractVector{T},
        rhs_vals::AbstractVector{T},
        lhs_rows::Vector{Int},
        rhs_rows::Vector{Int},
        dirichlet_bcs::Dict{Int, T},
    ) where {T}

For each index `i` that is a key of `dirichlet_bcs`, set the corresponding values from that
row to zero, both in `lhs_vals` and `rhs_vals`.

# Examples

```jldoctest
using Mantis

Assemblers.zero_rows!(
    [1.0, 1.0, 1.0], [1.0, 2.0, 3.0], [1, 2, 3], [1, 3, 2], Dict(2 => 42.0)
)

# output

([1.0, 0.0, 1.0], [1.0, 2.0, 0.0])
```
"""
function zero_rows!(
    lhs_vals::AbstractVector{T},
    rhs_vals::AbstractVector{T},
    lhs_rows::Vector{Int},
    rhs_rows::Vector{Int},
    dirichlet_bcs::Dict{Int, T},
) where {T}
    zero = Base.zero(T)
    if !isempty(dirichlet_bcs)
        for id in eachindex(lhs_rows, lhs_vals)
            # Check if the row index is also a boundary index.
            if haskey(dirichlet_bcs, lhs_rows[id])
                lhs_vals[id] = zero
            end
        end

        for id in eachindex(rhs_rows, rhs_vals)
            # Check if the row index is also a boundary index.
            if haskey(dirichlet_bcs, rhs_rows[id])
                rhs_vals[id] = zero
            end
        end
    end

    return lhs_vals, rhs_vals
end

"""
    add_bc!(
        lhs::AbstractMatrix{T}, rhs::AbstractVector{T}, dirichlet_bcs::Dict{Int, T}
    ) where {T}

Set the rows to 0 and diagonals to 1 in `lhs`, and rows of `rhs` to a value, as specified by
the row indices (keys) and values of `dirichlet_bcs`.

# Examples

```jldoctest
using Mantis

Assemblers.add_bc!([2.0 2.0 2.0; 2.0 2.0 2.0; 2.0 2.0 2.0], zeros(3), Dict(2 => 42.0))

# output

([2.0 2.0 2.0; 0.0 1.0 0.0; 2.0 2.0 2.0], [0.0, 42.0, 0.0])
```
"""
function add_bc!(
    lhs::AbstractMatrix{T}, rhs::AbstractVector{T}, dirichlet_bcs::Dict{Int, T}
) where {T}
    one = Base.one(T)
    zero = Base.zero(T)
    for i in keys(dirichlet_bcs)
        view(lhs, i, :) .= zero
        lhs[i, i] = one
        rhs[i] = dirichlet_bcs[i]
    end

    return lhs, rhs
end

"""
    add_bc!(lhs::AbstractMatrix, rhs::AbstractMatrix, dirichlet_bcs::Dict)

Remove the rows and columns of `lhs` and `rhs` as specified by the keys of `dirichlet_bcs`.
See also [`set_diagonal!`](@ref).

```
```
"""
function add_bc!(lhs::AbstractMatrix, rhs::AbstractMatrix, dirichlet_bcs::Dict)
    return set_diagonal!(lhs, rhs, dirichlet_bcs)
end

"""
    set_diagonal!(
        lhs::AbstractMatrix{T}, rhs::AbstractVector{T}, dirichlet_bcs::Dict{Int, T}
    ) where {T}

Set the diagonals of `lhs` to 1 and rows of `rhs` to a value, as specified by the row
indices (keys) and values of `dirichlet_bcs`. See also [`add_bc!`](@ref).

# Examples

```jldoctest
using Mantis

Assemblers.set_diagonal!([2.0 2.0 2.0; 2.0 2.0 2.0; 2.0 2.0 2.0], zeros(3), Dict(2 => 42.0))

# output

([2.0 2.0 2.0; 2.0 1.0 2.0; 2.0 2.0 2.0], [0.0, 42.0, 0.0])
```
"""
function set_diagonal!(
    lhs::AbstractMatrix{T}, rhs::AbstractVector{T}, dirichlet_bcs::Dict{Int, T}
) where {T}
    one = Base.one(T)
    for i in keys(dirichlet_bcs)
        lhs[i, i] = one
        rhs[i] = dirichlet_bcs[i]
    end

    return lhs, rhs
end

"""
    set_diagonal!(lhs::AbstractMatrix, rhs::AbstractMatrix, dirichlet_bcs::Dict)

Remove the rows and columns of `lhs` and `rhs` as specified by the keys of `dirichlet_bcs`.

# Examples

```jldoctest
using Mantis

Assemblers.set_diagonal!(
    [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0],
    [2.0 0.0 0.0; 0.0 2.0 0.0; 0.0 0.0 2.0],
    Dict(2 => 42.0),
)

# output

([2.0 0.0; 0.0 2.0], [2.0 0.0; 0.0 2.0])
```
"""
function set_diagonal!(lhs::AbstractMatrix, rhs::AbstractMatrix, dirichlet_bcs::Dict)
    ks = keys(dirichlet_bcs)
    lhs_size = size(lhs)
    rhs_size = size(rhs)
    lhs = lhs[setdiff(1:lhs_size[1], ks), setdiff(1:lhs_size[2], ks)]
    rhs = rhs[setdiff(1:rhs_size[1], ks), setdiff(1:rhs_size[2], ks)]

    return lhs, rhs
end

"""
    build_array(
        array_type::Type{A},
        rows::Vector{Int},
        cols::Vector{Int},
        vals::AbstractVector,
        size::Tuple{Int, Int},
    ) where {A <: AbstractArray}

Returns a array of the specified type with the given row and column indices and values.

# Arguments

  - `array_type::Type{AbstractArray}`: The type of array to build.
  - `rows::Vector{Int}`: The row indices of the array.
  - `cols::Vector{Int}`: The column indices of the array.
  - `vals::AbstractVector`: The values of the array.
  - `size::Tuple{Int, Int}`: The size of the array.

# Returns

  - `::array_type`: The constructed array of the specified type.
"""
function build_array(
    array_type::Type{A},
    rows::Vector{Int},
    cols::Vector{Int},
    vals::AbstractVector,
    size::Tuple{Int, Int},
) where {A <: AbstractArray}
    return throw(
        ArgumentError("Assembly of array type `$(array_type)` not currently implemented.")
    )
end

function build_array(
    ::Type{SM},
    rows::Vector{Int},
    cols::Vector{Int},
    vals::AbstractVector,
    size::Tuple{Int, Int},
) where {SM <: spa.AbstractSparseMatrix}
    return spa.sparse(rows, cols, vals, size...)
end

function build_array(
    ::Type{M},
    rows::Vector{Int},
    cols::Vector{Int},
    vals::AbstractVector{T},
    size::Tuple{Int, Int},
) where {T, M <: AbstractMatrix{T}}
    array = zeros(T, size)
    for (row, col, val) in zip(rows, cols, vals)
        array[row, col] += val
    end

    return array
end

function build_array(
    ::Type{V}, rows::Vector{Int}, cols::Vector{Int}, vals::V, size::Tuple{Int, Int}
) where {T, V <: AbstractVector{T}}
    any(!=(1), cols) && throw(ArgumentError("`cols` must contain only the value 1."))
    array = zeros(T, first(size))
    for (row, val) in zip(rows, vals)
        array[row] += val
    end

    return array
end

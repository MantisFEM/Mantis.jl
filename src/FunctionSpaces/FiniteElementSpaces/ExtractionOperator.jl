
"""
    Indices{num_output_components, TI}

Stores the global basis indices on a single element for an `ExtractionOperator`.

# Fields
- `I::TI`: Global (multi-component) basis indices supported on the element. This is the
    union of the per-component indices `I_c`. Should be a vector-like type with `Int`
    elements and without duplicates.
- `I_c::NTuple{num_output_components, TI}`: Global basis indices per output component.
    `I_c[co]` lists the global basis functions whose `co`-th component is non-zero on the
    element. Each `I_c[co]` must be a subset of `I`. Note that a global basis function may
    appear in more than one `I_c[co]`.

# Constructors
- `Indices(I::TI, I_c::NTuple{num_output_components, TI})`: General constructor.
- `Indices(I::TI)`: Single-component constructor; equivalent to `Indices(I, (I,))`.
"""
struct Indices{num_output_components, TI}
    I::TI
    I_c::NTuple{num_output_components, TI}

    function Indices(
        I::TI, I_c::NTuple{num_output_components, TI}
    ) where {num_output_components, TI}
        if eltype(I) != Int
            throw(ArgumentError("The elements in `I` must be of type Int."))
        end

        if !allunique(I)
            throw(ArgumentError("The elements in `I` must be unique."))
        end

        for co in 1:num_output_components
            if !issubset(I_c[co], I)
                throw(
                    ArgumentError(
                        LazyString(
                            "The indices of output component ",
                            co,
                            " must be a subset of `I`.",
                        ),
                    ),
                )
            end
        end

        return new{num_output_components, TI}(I, I_c)
    end
end

Indices(I::TI) where {TI} = Indices(I, (I,))

get_num_components(::Indices{num_output_components, TI}) where {num_output_components, TI} =
    num_output_components
get_basis_indices(indices::Indices) = indices.I
get_basis_indices(indices::Indices, component_id::Int) = indices.I_c[component_id]

"""
    ExtractionOperator{
        num_input_components, num_output_components, num_blocks, TE, TI
    }

Stores the extraction coefficients and basis indices of a function space.

On each element, the extraction maps the local bases of the input components,
``B_1, \\dots, B_{n_i}``, to the output components ``M_1, \\dots, M_{n_o}`` of the space as
```math
\\left[M_1 \\cdots M_{n_o}\\right]
=
\\left[B_1 \\cdots B_{n_i}\\right]
\\begin{bmatrix}
E_{11} & \\cdots & E_{1n_o} \\\\
\\vdots & \\ddots & \\vdots \\\\
E_{n_i1} & \\cdots & E_{n_in_o}
\\end{bmatrix},
\\qquad\\text{i.e.}\\qquad
M_{c_o} = \\sum_{c_i} B_{c_i} E_{c_i c_o}.
```
The rows of a block ``E_{c_i c_o}`` correspond to the local basis functions of input
component ``c_i``, and its columns to the global basis indices of output component ``c_o``
on that element (see [`Indices`](@ref)). The number of input components may differ from
the number of output components.

Only the structurally non-zero blocks are stored. Which `(input, output)` component pairs
are stored is described by `block_pattern`, which is shared by all elements. All stored
blocks have the same type `TE`, so that iterating over them is type stable.

# Examples
- A single-component space has `block_pattern = ((1, 1),)` and one block per element.
- A direct sum of two single-component spaces has `block_pattern = ((1, 1), (2, 2))`, with
    (for example) `LinearAlgebra.I` blocks. The zero off-diagonal blocks are not stored.

# Fields
- `block_pattern::NTuple{num_blocks, Tuple{Int, Int}}`: The `(input, output)` component
    pair of each stored block, in storage order.
- `extraction_coefficients::Vector{NTuple{num_blocks, TE}}`: The stored blocks per element,
    in the order given by `block_pattern`. `TE` should be matrix-like.
- `basis_indices::Vector{Indices{num_output_components, TI}}`: The basis indices per
    element. See [`Indices`](@ref) for more details.
- `num_elements::Int`: The number of elements.
- `num_basis::Int`: The (total) dimension of the function space.

# Constructors
- `ExtractionOperator{num_input_components}(block_pattern, extraction_coefficients,
    basis_indices, num_elements, num_basis)`: General constructor. `num_input_components`
    must be the largest input component in `block_pattern`, and every input and output
    component must appear in at least one block. It is given as a type parameter so that
    the constructed type is inferable from the types of the inputs.
- `ExtractionOperator(extraction_coefficients::Vector{TE}, basis_indices::Vector{TI},
    num_elements, num_basis)`: Single-component constructor, using
    `block_pattern = ((1, 1),)`.
"""
struct ExtractionOperator{num_input_components, num_output_components, num_blocks, TE, TI}
    block_pattern::NTuple{num_blocks, Tuple{Int, Int}}
    extraction_coefficients::Vector{NTuple{num_blocks, TE}}
    basis_indices::Vector{Indices{num_output_components, TI}}
    num_elements::Int
    num_basis::Int

    function ExtractionOperator{num_input_components}(
        block_pattern::NTuple{num_blocks, Tuple{Int, Int}},
        extraction_coefficients::Vector{NTuple{num_blocks, TE}},
        basis_indices::Vector{Indices{num_output_components, TI}},
        num_elements::Int,
        num_basis::Int,
    ) where {num_input_components, num_blocks, num_output_components, TE, TI}
        if length(extraction_coefficients) != num_elements
            throw(
                ArgumentError(
                    "Number of extraction coefficient tuples must match number of elements."
                ),
            )
        end

        if length(basis_indices) != num_elements
            throw(
                ArgumentError(
                    "Number of basis index objects must match number of elements."
                ),
            )
        end

        pattern_num_input_components = _check_block_pattern(
            block_pattern, num_output_components
        )
        if num_input_components != pattern_num_input_components
            throw(
                ArgumentError(
                    LazyString(
                        "Expected ",
                        pattern_num_input_components,
                        " input components from the block pattern. Got ",
                        num_input_components,
                        ".",
                    ),
                ),
            )
        end

        if TE <: AbstractMatrix
            _check_block_sizes(
                block_pattern, extraction_coefficients, basis_indices, num_input_components
            )
        end

        return new{num_input_components, num_output_components, num_blocks, TE, TI}(
            block_pattern, extraction_coefficients, basis_indices, num_elements, num_basis
        )
    end
end

function ExtractionOperator(
    extraction_coefficients::Vector{TE},
    basis_indices::Vector{TI},
    num_elements::Int,
    num_basis::Int,
) where {TE, TI <: AbstractVector{Int}}
    return ExtractionOperator{1}(
        ((1, 1),),
        [(coeffs,) for coeffs in extraction_coefficients],
        [Indices(indices) for indices in basis_indices],
        num_elements,
        num_basis,
    )
end

"""
    _check_block_pattern(
        block_pattern::NTuple{num_blocks, Tuple{Int, Int}}, num_output_components::Int
    ) where {num_blocks}

Validates `block_pattern` and returns the number of input components.

# Exceptions
- `ArgumentError`: If the pattern is empty, contains components out of range, contains
    duplicate blocks, or if some input or output component is not part of any block.
"""
function _check_block_pattern(
    block_pattern::NTuple{num_blocks, Tuple{Int, Int}}, num_output_components::Int
) where {num_blocks}
    if num_blocks == 0
        throw(ArgumentError("The block pattern must contain at least one block."))
    end

    for (ci, co) in block_pattern
        if ci < 1
            throw(
                ArgumentError(
                    LazyString("Input component ", ci, " in block pattern is not positive.")
                ),
            )
        end
        if !(1 <= co <= num_output_components)
            throw(
                ArgumentError(
                    LazyString(
                        "Output component ",
                        co,
                        " in block pattern is not in 1:",
                        num_output_components,
                        ".",
                    ),
                ),
            )
        end
    end

    if !allunique(block_pattern)
        throw(ArgumentError("The block pattern must not contain duplicate blocks."))
    end

    num_input_components = maximum(first, block_pattern)
    for ci in 1:num_input_components
        if !any(block -> block[1] == ci, block_pattern)
            throw(
                ArgumentError(
                    LazyString("Input component ", ci, " is not part of any block.")
                ),
            )
        end
    end

    for co in 1:num_output_components
        if !any(block -> block[2] == co, block_pattern)
            throw(
                ArgumentError(
                    LazyString("Output component ", co, " is not part of any block.")
                ),
            )
        end
    end

    return num_input_components
end

"""
    _check_block_sizes(
        block_pattern::NTuple{num_blocks, Tuple{Int, Int}},
        extraction_coefficients::Vector{NTuple{num_blocks, TE}},
        basis_indices::Vector{Indices{num_output_components, TI}},
        num_input_components::Int,
    ) where {num_blocks, TE <: AbstractMatrix, num_output_components, TI}

Checks, on every element, that the number of columns of each block matches the number of
basis indices of its output component, and that all blocks of the same input component
have the same number of rows.

# Exceptions
- `ArgumentError`: If any of the size checks fails.
"""
function _check_block_sizes(
    block_pattern::NTuple{num_blocks, Tuple{Int, Int}},
    extraction_coefficients::Vector{NTuple{num_blocks, TE}},
    basis_indices::Vector{Indices{num_output_components, TI}},
    num_input_components::Int,
) where {num_blocks, TE <: AbstractMatrix, num_output_components, TI}
    num_rows = Vector{Int}(undef, num_input_components)
    for element_id in eachindex(extraction_coefficients, basis_indices)
        fill!(num_rows, -1)
        blocks = extraction_coefficients[element_id]
        indices = basis_indices[element_id]
        for (block_id, (ci, co)) in enumerate(block_pattern)
            block_rows, block_cols = size(blocks[block_id])
            if block_cols != length(get_basis_indices(indices, co))
                throw(
                    ArgumentError(
                        LazyString(
                            "Block ",
                            (ci, co),
                            " on element ",
                            element_id,
                            " has ",
                            block_cols,
                            " columns, but output component ",
                            co,
                            " has ",
                            length(get_basis_indices(indices, co)),
                            " basis indices.",
                        ),
                    ),
                )
            end

            # All blocks of the same input component must have the same number of rows on
            # a given element, so we store the number of rows of the first block of each
            # input component, and compare the other blocks of that component against it.
            if num_rows[ci] == -1
                num_rows[ci] = block_rows
            elseif num_rows[ci] != block_rows
                throw(
                    ArgumentError(
                        LazyString(
                            "Blocks of input component ",
                            ci,
                            " on element ",
                            element_id,
                            " have inconsistent numbers of rows.",
                        ),
                    ),
                )
            end
        end
    end

    return nothing
end

get_num_input_components(
    ::ExtractionOperator{num_input_components, num_output_components, num_blocks, TE, TI}
) where {num_input_components, num_output_components, num_blocks, TE, TI} =
    num_input_components

get_num_output_components(
    ::ExtractionOperator{num_input_components, num_output_components, num_blocks, TE, TI}
) where {num_input_components, num_output_components, num_blocks, TE, TI} =
    num_output_components

get_num_components(extraction_op::ExtractionOperator) =
    get_num_output_components(extraction_op)

get_num_blocks(
    ::ExtractionOperator{num_input_components, num_output_components, num_blocks, TE, TI}
) where {num_input_components, num_output_components, num_blocks, TE, TI} = num_blocks

get_EI_types(
    ::ExtractionOperator{num_input_components, num_output_components, num_blocks, TE, TI}
) where {num_input_components, num_output_components, num_blocks, TE, TI} = (TE, TI)

get_extraction_type(
    ::ExtractionOperator{num_input_components, num_output_components, num_blocks, TE, TI}
) where {num_input_components, num_output_components, num_blocks, TE, TI} = TE

get_index_type(
    ::ExtractionOperator{num_input_components, num_output_components, num_blocks, TE, TI}
) where {num_input_components, num_output_components, num_blocks, TE, TI} = TI

get_block_pattern(extraction_op::ExtractionOperator) = extraction_op.block_pattern

function _get_index_operator(extraction_op::ExtractionOperator, element_id::Int)
    return extraction_op.basis_indices[element_id]
end

"""
    _get_block_id(
        block_pattern::NTuple{num_blocks, Tuple{Int, Int}},
        input_component_id::Int,
        output_component_id::Int,
    ) where {num_blocks}

Returns the storage position of the block `(input_component_id, output_component_id)` in
`block_pattern`.

# Exceptions
- `ArgumentError`: If the block is not in the pattern, i.e. it is structurally zero.
"""
function _get_block_id(
    block_pattern::NTuple{num_blocks, Tuple{Int, Int}},
    input_component_id::Int,
    output_component_id::Int,
) where {num_blocks}
    for (block_id, block) in enumerate(block_pattern)
        if block == (input_component_id, output_component_id)
            return block_id
        end
    end

    return throw(
        ArgumentError(
            LazyString(
                "Block ",
                (input_component_id, output_component_id),
                " is not stored in the block pattern (it is structurally zero).",
            ),
        ),
    )
end

function _get_block_id(
    extraction_op::ExtractionOperator, input_component_id::Int, output_component_id::Int
)
    return _get_block_id(
        get_block_pattern(extraction_op), input_component_id, output_component_id
    )
end

"""
    get_extraction_blocks(extraction_op::ExtractionOperator, element_id::Int)

Returns all stored extraction coefficient blocks on element `element_id`, in the order
given by [`get_block_pattern`](@ref).
"""
function get_extraction_blocks(extraction_op::ExtractionOperator, element_id::Int)
    return extraction_op.extraction_coefficients[element_id]
end

"""
    get_extraction_coefficients(
        extraction_op::ExtractionOperator, element_id::Int, block_id::Int
    )

Returns the `block_id`-th stored extraction coefficient block on element `element_id`.
"""
function get_extraction_coefficients(
    extraction_op::ExtractionOperator, element_id::Int, block_id::Int
)
    return get_extraction_blocks(extraction_op, element_id)[block_id]
end

"""
    get_extraction_coefficients(
        extraction_op::ExtractionOperator{1, 1}, element_id::Int
    )

Returns the only extraction coefficient block of a single-input, single-output component
space.
"""
function get_extraction_coefficients(
    extraction_op::ExtractionOperator{1, 1}, element_id::Int
)
    return only(get_extraction_blocks(extraction_op, element_id))
end

"""
    get_extraction_coefficients(
        extraction_op::ExtractionOperator,
        element_id::Int,
        input_component_id::Int,
        output_component_id::Int,
    )

Returns the extraction coefficient block ``E_{c_i c_o}`` on element `element_id` that maps
input component `input_component_id` to output component `output_component_id`.

# Exceptions
- `ArgumentError`: If the block is structurally zero (not stored). Use
    [`get_block_pattern`](@ref) to iterate over the stored blocks instead.
"""
function get_extraction_coefficients(
    extraction_op::ExtractionOperator,
    element_id::Int,
    input_component_id::Int,
    output_component_id::Int,
)
    block_id = _get_block_id(extraction_op, input_component_id, output_component_id)

    return get_extraction_coefficients(extraction_op, element_id, block_id)
end

function get_basis_indices(extraction_op::ExtractionOperator, element_id::Int)
    return get_basis_indices(_get_index_operator(extraction_op, element_id))
end

function get_basis_indices(
    extraction_op::ExtractionOperator, element_id::Int, component_id::Int
)
    return get_basis_indices(_get_index_operator(extraction_op, element_id), component_id)
end

get_num_basis(extraction_op::ExtractionOperator) = extraction_op.num_basis

function get_num_basis(extraction_op::ExtractionOperator, element_id::Int)
    return length(get_basis_indices(extraction_op, element_id))
end

function get_num_basis(
    extraction_op::ExtractionOperator, element_id::Int, component_id::Int
)
    return length(get_basis_indices(extraction_op, element_id, component_id))
end

get_num_elements(extraction_op::ExtractionOperator) = extraction_op.num_elements

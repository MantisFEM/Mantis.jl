"""
Functions for 2-scale relations between elements and functions.

"""

# Getters for elements

"""

    element_ranges_to_tuple_list(finer_ranges::Vector{Vector{Int}}, subdivisions::NTuple{n, Int}) where {n}

Converts `finer_ranges::Vector{Vector{Int}}` to a `Vector{NTuple{n, Int}}` containing all 
permutations of numbers in `finer_ranges` as `NTuple{n, Int}`.

For example, `element_ranges_to_tuple_list([[1, 2], [3, 4]], (2,2))` returns `[(1, 3), (2, 3), (1, 4), (2, 4)]`.
# Arguments
- `finer_ranges::Vector{Vector{Int}}`: The numbers to be used in the permutations.
- `subdivisions::NTuple{n, Int})`: The number of subdivisions in each dimension.
# Returns
- `finer_elements::Vector{NTuple{n, Int}}`: Vector containing all permutations as `NTuple{n, Int}`.
"""
function element_ranges_to_tuple_list(finer_ranges::Vector{Vector{Int}}, subdivisions::NTuple{n, Int}) where {n}
    finer_elements = Vector{NTuple{n, Int}}(undef, prod(subdivisions))

    el_count = 1
    @inbounds for el_id in Iterators.product(finer_ranges...)
        finer_elements[el_count] = el_id
        el_count += 1
    end
    
    return finer_elements
end

"""
    get_finer_elements(coarse_element_id::Int, subdivisions::Int)

Returns the finer elements contained inside `coarse_element_id`, according to the number of
`subdivisions`.

# Arguments
- `coarse_element_id::Int`: The coarse element where the finer elements are contained.
- `subdivisions::Int`: The number of subdivisions.
# Returns
- `::Vector{Int}`: The element ids of the finer elements.
"""
function get_finer_elements(coarse_element_id::Int, subdivisions::Int)
    return [get_finer_elements((coarse_element_id,), (subdivisions,))[i][1] for i in 1:subdivisions]
end

"""
    get_finer_elements(coarse_element_id::NTuple{n, Int}, subdivisions::NTuple{n, Int}) where {n}

Returns the finer elements contained inside `coarse_element_id`, according to the number of
`subdivisions`.

# Arguments
- `coarse_element_id::NTuple{n, Int}`: The coarse element where the finer elements are contained.
- `subdivisions::NTuple{n, Int})`: The number of subdivisions in each dimension.
# Returns
- `::Vector{NTuple{n, Int}}`: The element ids of the finer elements.
"""
function get_finer_elements(coarse_element_id::NTuple{n, Int}, subdivisions::NTuple{n, Int}) where {n}
    finer_ranges = Vector{Vector{Int}}(undef, n)

    for d in 1:1:n
        finer_ranges[d] = collect((1:subdivisions[d]) .+ (coarse_element_id[d]-1)*subdivisions[d]) 
    end

    return element_ranges_to_tuple_list(finer_ranges, subdivisions)
end

"""
    get_coarser_elements(fine_element_id::Int, subdivisions::Int)

Returns the coarser element where `fine_element_id` is contained.

# Arguments
- `fine_element_id::Int`: The fine element where coarser element containment is checked. 
- `subdivisions::Int`: The number of subdivisions in each dimension.
# Returns
- `::Int`: The coarser element where `fine_element_id` is contained.
"""
function get_coarser_element(fine_element_id::Int, subdivisions::Int)
    return floor(Int, (fine_element_id-1)/subdivisions + 1)
end

"""
    get_coarser_elements(fine_element_id::NTuple{n, Int}, subdivisions::NTuple{n, Int}) where {n}

Returns the coarser element where `fine_element_id` is contained.

# Arguments
- `fine_element_id::NTuple{n, Int}`: The fine element where coarser element containment is checked. 
- `subdivisions::NTuple{n, Int})`: The number of subdivisions in each dimension.
# Returns
- `::NTuple{n, Int}`: The coarser element where `fine_element_id` is contained.
"""
function get_coarser_element(fine_element_id::NTuple{n, Int}, subdivisions::NTuple{n, Int}) where {n}
    return ntuple(d -> get_coarser_element(fine_element_id[d], subdivisions[d]), n)
end

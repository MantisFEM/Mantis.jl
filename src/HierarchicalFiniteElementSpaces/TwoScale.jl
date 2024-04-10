"""
Functions for 2-scale relations between elements and functions.

"""

# Getters for elements

"""

    element_ranges_to_tuple_list(finer_ranges::Vector{Vector{Int}}, nsubdivisions::NTuple{n, Int}) where {n}

Converts `finer_ranges::Vector{Vector{Int}}` to a `Vector{NTuple{n, Int}}` containing all 
permutations of numbers in `finer_ranges` as `NTuple{n, Int}`.

# Examples
```julia-repl
julia> element_ranges_to_tuple_list([[1, 2], [3, 4]], (2,2))
[(1, 3), (2, 3), (1, 4), (2, 4)]
```
# Arguments
- `finer_ranges::Vector{Vector{Int}}`: The numbers to be used in the permutations.
- `nsubdivisions::NTuple{n, Int})`: The number of subdivisions in each dimension.
# Returns
- `finer_elements::Vector{NTuple{n, Int}}`: Vector containing all permutations as `NTuple{n, Int}`.
"""
function element_ranges_to_tuple_list(finer_ranges::Vector{Vector{Int}}, nsubdivisions::NTuple{n, Int}) where {n}
    finer_elements = Vector{NTuple{n, Int}}(undef, prod(nsubdivisions))

    el_count = 1
    @inbounds for el_id in Iterators.product(finer_ranges...)
        finer_elements[el_count] = el_id
        el_count += 1
    end
    
    return finer_elements
end

"""
    get_finer_elements(coarse_element_id::Int, nsubdivisions::Int)

Returns the child elements contained inside `coarse_element_id`, according to the number of
`nsubdivisions`.

# Arguments
- `coarse_element_id::Int`: The parent element where the child elements are contained.
- `nsubdivisions::Int`: The number of subdivisions.
# Returns
- `::Vector{Int}`: The element ids of the child elements.
"""
function get_finer_elements(coarse_element_id::Int, nsubdivisions::Int)
    return [get_finer_elements((coarse_element_id,), (nsubdivisions,))[i][1] for i in 1:nsubdivisions]
end

"""
    get_finer_elements(coarse_element_id::NTuple{n, Int}, nsubdivisions::NTuple{n, Int}) where {n}

Returns the child elements contained inside `coarse_element_id`, according to the number of
`nsubdivisions`.

# Arguments
- `coarse_element_id::NTuple{n, Int}`: The parent element where the child elements are contained.
- `nsubdivisions::NTuple{n, Int})`: The number of subdivisions in each dimension.
# Returns
- `::Vector{NTuple{n, Int}}`: The element ids of the child elements.
"""
function get_finer_elements(coarse_element_id::NTuple{n, Int}, nsubdivisions::NTuple{n, Int}) where {n}
    finer_ranges = Vector{Vector{Int}}(undef, n)

    for d in 1:1:n
        finer_ranges[d] = collect((1:nsubdivisions[d]) .+ (coarse_element_id[d]-1)*nsubdivisions[d]) 
    end

    return element_ranges_to_tuple_list(finer_ranges, nsubdivisions)
end

"""
    get_coarser_elements(fine_element_id::Int, nsubdivisions::Int)

Returns the parent element where `fine_element_id` is contained.

# Arguments
- `fine_element_id::Int`: The child element where parent element containment is checked. 
- `nsubdivisions::Int`: The number of subdivisions in each dimension.
# Returns
- `::Int`: The parent element where `fine_element_id` is contained.
"""
function get_coarser_element(fine_element_id::Int, nsubdivisions::Int)
    return floor(Int, (fine_element_id-1)/nsubdivisions + 1)
end

"""
    get_coarser_elements(fine_element_id::NTuple{n, Int}, nsubdivisions::NTuple{n, Int}) where {n}

Returns the parent element where `fine_element_id` is contained.

# Arguments
- `fine_element_id::NTuple{n, Int}`: The child element where parent element containment is checked. 
- `nsubdivisions::NTuple{n, Int})`: The number of subdivisions in each dimension.
# Returns
- `::NTuple{n, Int}`: The parent element where `fine_element_id` is contained.
"""
function get_coarser_element(fine_element_id::NTuple{n, Int}, nsubdivisions::NTuple{n, Int}) where {n}
    return ntuple(d -> get_coarser_element(fine_element_id[d], nsubdivisions[d]), n)
end

# Getters for basis splines

"""
    get_finer_basis_id(coarse_bspline::FiniteElementSpaces.BSplineSpace, basis_id::Int, nsubdivisions::Int)

Returns the ids of the child B-splines, given `nsubdivisions`, whose support intersection with
the support of the B-spline identified by `basis_id` is non-empty.

# Arguments
- `coarse_bspline::FiniteElementSpaces.BSplineSpace`: Parent B-spline.
- `coarse_basis_id::Int`: Id of the parent B-spline.
- `nsubdivisions::NTuple{n, Int})`: The number of subdivisions in each dimension.
# Returns
- `::Vector{Int}`: Ids of the child B-splines.
"""
function get_finer_basis_id(coarse_bspline::FiniteElementSpaces.BSplineSpace, basis_id::Int, nsubdivisions::Int)
    p = coarse_bspline.knot_vector.polynomial_degree
    
    first_coarse_brk_idx = FiniteElementSpaces.get_breakpoint_index(coarse_bspline.knot_vector, basis_id) 
    last_coarse_brk_idx = FiniteElementSpaces.get_breakpoint_index(coarse_bspline.knot_vector,basis_id+p+1) 

    first_coarse_knot_idx = FiniteElementSpaces.get_first_knot_index(coarse_bspline.knot_vector, first_coarse_brk_idx)
    last_coarse_knot_idx = FiniteElementSpaces.get_first_knot_index(coarse_bspline.knot_vector, last_coarse_brk_idx)

    first_fine_knot_idx = first_coarse_knot_idx + (first_coarse_brk_idx - 1)*(nsubdivisions - 1)
    last_fine_knot_idx = last_coarse_knot_idx + (last_coarse_brk_idx - 1)*(nsubdivisions - 1)

    first_fine_basis_id = maximum((1, first_fine_knot_idx - p + coarse_bspline.knot_vector.multiplicity[first_coarse_brk_idx] - 1))
    last_fine_basis_id = last_fine_knot_idx - 1

    return collect(first_fine_basis_id:last_fine_basis_id)
end

"""
    get_finer_basis_id(coarse_basis_id::Int, refinement_operator::FiniteElementSpaces.RefinementOperator)

Returns the ids of the child B-splines of `coarse_basis_id`, in terms of the change of basis
provided by `refinement_operator`.

# Arguments
- `basis_id::Int`: Id of the parent B-spline.
- `refinement_operator::FiniteElementSpaces.RefinementOperator`: The refinement operator for
the change of basis.
# Returns
- `::Vector{Int}`: Ids of the child B-splines.
"""
function get_finer_basis_id(coarse_basis_id::Int, refinement_operator::FiniteElementSpaces.RefinementOperator)
    return refinement_operator.parent_child[coarse_basis_id]
end
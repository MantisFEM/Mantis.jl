"""
Functions and algorithms used for two scale relations.

"""

"""
    AbstractTwoScaleOperator

Supertype for all two scale relations.
"""
abstract type AbstractTwoScaleOperator end

# TwoScaleOperator structure

"""
    struct TwoScaleOperator

Two-scale operator for a change of basis between two finite element spaces.

# Fields
- `coarse_space::F1`: Coarse finite element space.
- `fine_space::F2`: Fine finite element space.
- `global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int}`: Change of basis matrix.
- `coarse_to_fine::Vector{Vector{Int}}`: Relation between coarser and finer basis functions.
"""
struct TwoScaleOperator{n, S} <: AbstractTwoScaleOperator
    coarse_space::S
    fine_space::S
    global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int}
    coarse_to_fine_elements::Vector{Vector{Int}}
    fine_to_coarse_elements::Vector{Int}
    coarse_to_fine_functions::Vector{Vector{Int}}
    fine_to_coarse_functions::Vector{Vector{Int}}

    function TwoScaleOperator(coarse_space::S, fine_space::S, global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int}, coarse_to_fine_elements::Vector{Vector{Int}}, fine_to_coarse_elements::Vector{Int}) where {n, S<:AbstractFiniteElementSpace{n}}
        dims = size(global_subdiv_matrix)
        coarse_to_fine_functions = Vector{Vector{Int}}(undef, dims[2])
        fine_to_coarse_functions = Vector{Vector{Int}}(undef, dims[1])

        gm_data = SparseArrays.findnz(global_subdiv_matrix)
        transpose_matrix = SparseArrays.sparse(gm_data[2],gm_data[1],gm_data[3])

        for i in 1:1:dims[2]
            coarse_to_fine_functions[i] = global_subdiv_matrix.rowval[SparseArrays.nzrange(global_subdiv_matrix, i)]
        end
        for i in 1:1:dims[1]
            fine_to_coarse_functions[i] = global_subdiv_matrix.rowval[SparseArrays.nzrange(transpose_matrix, i)]
        end


        new{n, S}(coarse_space, fine_space, global_subdiv_matrix, coarse_to_fine_elements, fine_to_coarse_elements, coarse_to_fine_functions, fine_to_coarse_functions)
    end
end

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

function get_ancestor_element(two_scale_operators, child_element_id::Int, child_level::Int, num_ancestor_levels::Int)
    ancestor_id = child_element_id
    for level ∈ child_level:-1:child_level-num_ancestor_levels+1
        ancestor_id = get_coarser_element(two_scale_operators[level-1], ancestor_id)
    end

    return ancestor_id 
end

function get_ancestor_element(two_scale_operator, child_element_ids, child_level::Int, num_ancestor_levels::Int)
    return union(get_ancestor_element.((two_scale_operator,), child_element_ids, (child_level,), (num_ancestor_levels,))...)
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

# Getters for AbstractTwoScaleOperator

function get_coarse_space(twoscale_operator::T) where {T<:AbstractTwoScaleOperator}
    return twoscale_operator.coarse_space
end

function get_fine_space(twoscale_operator::T) where {T<:AbstractTwoScaleOperator}
    return twoscale_operator.fine_space
end

function get_coarse_spaces(twoscale_opeators::NTuple{m, T}) where {m, T<:AbstractTwoScaleOperator}
    return get_coarse_space.(twoscale_opeators)
end

function get_fine_spaces(twoscale_opeators::NTuple{m, T}) where {m, T<:AbstractTwoScaleOperator}
    return get_fine_space.(twoscale_opeators)
end

"""
    subdivide_coeffs(coarse_basis_coeffs::Vector{Float64}, twoscale_operator::TwoScaleOperator)

Returns the spline coefficients in a refined basis from coefficients in a coarser B-spline basis.

# Arguments
- `coarse_basis_coeffs::Vector{Float64}`: Coefficients in the coarser basis.
- `twoscale_operator::TwoScaleOperator`: Two-scale operator the change of basis matrix.
# Returns
-`::Vector{Float64}`: Coefficients in the finer basis.
"""
function subdivide_coeffs(coarse_basis_coeffs::Vector{Float64}, twoscale_operator::T) where {T<:AbstractTwoScaleOperator}
    return twoscale_operator.global_subdiv_matrix * coarse_basis_coeffs
end

"""
    get_local_subdiv_matrix(twoscale_operator::TwoScaleOperator, fine_el_id::Int)

Returns the local subdivision matrix necessary to represent functions from a coarser space 
in terms of finer functions on element `fine_el_id`.

# Arguments 
- `twoscale_operator::TwoScaleOperator`: Two-scale operator relating two B-spline spaces.
- `fine_el_id::Int`: Id of the fine element.
# Returns
- `::@views Array{Float64, 2}`: Local refinement matrix.
"""
function get_local_subdiv_matrix(twoscale_operator::T, coarse_el_id::Int, fine_el_id::Int) where {T<:AbstractTwoScaleOperator}
    _, fine_basis_indices = get_extraction(twoscale_operator.fine_space, fine_el_id)
    _, coarse_basis_indices = get_extraction(twoscale_operator.coarse_space, coarse_el_id)

    return @view twoscale_operator.global_subdiv_matrix[fine_basis_indices, coarse_basis_indices]
end

"""
    get_finer_basis_id(coarse_basis_id::Int, twoscale_operator::FunctionSpaces.TwoScaleOperator)

Returns the ids of the child B-splines of `coarse_basis_id`, in terms of the change of basis
provided by `twoscale_operator`.

# Arguments
- `basis_id::Int`: Id of the parent B-spline.
- `twoscale_operator::TwoScaleOperator`: The two-scale operator for
the change of basis.
# Returns
- `::@view Vector{Int}`: Ids of the child B-splines.
"""
function get_finer_basis_id(twoscale_operator::TwoScaleOperator{n, S}, basis_id::Int) where {n, S<:AbstractFiniteElementSpace{n}}
    return twoscale_operator.coarse_to_fine_functions[basis_id]
end

function get_finer_elements(twoscale_operator::TwoScaleOperator{n, S}, el_id::Int) where {n, S<:AbstractFiniteElementSpace{n}}
    return twoscale_operator.coarse_to_fine_elements[el_id]    
end

function get_finer_elements(twoscale_operator::TwoScaleOperator{n, S}, el_ids) where {n, S<:AbstractFiniteElementSpace{n}}
    return reduce(vcat, get_finer_elements.(Ref(twoscale_operator), el_ids))  
end

function get_coarser_element(twoscale_operator::TwoScaleOperator{n, S}, el_id::Int) where {n, S<:AbstractFiniteElementSpace{n}}
    return twoscale_operator.fine_to_coarse_elements[el_id]    
end

function get_coarser_elements(twoscale_operator::TwoScaleOperator{n, S}, el_ids) where {n, S<:AbstractFiniteElementSpace{n}}
    if el_ids == Int[]
        return Int[]
    end

    return reduce(union, get_coarser_element.(Ref(twoscale_operator), el_ids))  
end

function get_coarser_basis_id(twoscale_operator::TwoScaleOperator{n, S}, basis_id::Int) where {n, S<:AbstractFiniteElementSpace{n}}
    return twoscale_operator.fine_to_coarse_functions[basis_id]
end

# Getters for basis splines

function get_coarser_element(two_scale_operators::Vector{T}, coarse_level::Int, finer_element::Int, finer_level::Int) where {T<:AbstractTwoScaleOperator}

    current_coarse = finer_element
    current_fine = finer_element
    for level ∈ finer_level:-1:coarse_level+1
        current_coarse = get_coarser_element(two_scale_operators[level-1], current_fine)

        current_fine = current_coarse
    end

    return current_coarse
end

function get_contained_knot_vector(supp_intersection::StepRange{Int, Int}, ts::T, fine_space::BSplineSpace) where {T <: AbstractTwoScaleOperator}
    if supp_intersection == []
        breakpoint_idxs = get_finer_elements(ts, first(supp_intersection))[1]
    else
        element_idxs = Int[]

        for element ∈ supp_intersection
            append!(element_idxs, get_finer_elements(ts, element))
        end
        
        breakpoint_idxs = minimum(element_idxs):(maximum(element_idxs)+1)
    end

    breakpoints = get_patch(fine_space).breakpoints[breakpoint_idxs]
    multiplicity = get_multiplicity_vector(fine_space)[breakpoint_idxs]

    return KnotVector(Mesh.Patch1D(breakpoints), get_polynomial_degree(fine_space), multiplicity)
end

function get_contained_knot_vector(supp_intersection::StepRange{Int, Int}, fem_space::BSplineSpace)
    if supp_intersection == []
        breakpoint_idxs = first(supp_intersection)
    else
        breakpoint_idxs = minimum(supp_intersection):maximum(supp_intersection)+1
    end

    breakpoints = get_patch(fem_space).breakpoints[breakpoint_idxs]
    multiplicity = get_multiplicity_vector(fem_space)[breakpoint_idxs]

    return KnotVector(Mesh.Patch1D(breakpoints), get_polynomial_degree(fem_space), multiplicity)
end

# Includes for concrete two scale relations

include("UnivariateBSplineTwoScaleRelations.jl")
include("TensorProductTwoScaleRelations.jl")
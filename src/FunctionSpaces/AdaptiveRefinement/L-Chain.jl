"""
Algorithms related with L-chain creation.

"""

using Combinatorics
using Graphs

function proper_range(start::Int, finish::Int)
    if start>finish
        return range(start, finish, step=:(-1))
    end

    return range(start, finish, step=:1)
end

function ordered_range(start::Int, finish::Int)
    if start>finish
        return proper_range(finish, start)
    end

    return proper_range(start, finish)
end

@doc raw"""
    get_corner_basis_ids(fe_space::AbstractFiniteElementSpace{n}, basis_ids::Vector{Int}) where {n}

Returns the basis indices of basis functions in the "corners" of a 
set of `basis_ids` with non-empty support in a given element.

# Arguments

- `fe_space::AbstractFiniteElementSpace{n}`: finite element space at a given level.
- `basis_ids::Vector{Int}`: set of basis indices with non-empty support in a given element.

# Returns 

- `::NTuple{4, Int}`: corner basis function indices in the order "UR", "LR", "LL", "UL".
"""
function get_corner_basis_ids(fe_space::AbstractFiniteElementSpace{n}, basis_ids::Vector{Int}) where {n}
    max_ind_basis = _get_num_basis_per_space(fe_space)
    lower_left_bspline = minimum(basis_ids)
    upper_right_bspline = maximum(basis_ids)

    lower_left_bspline_per_dim = linear_to_ordered_index(lower_left_bspline, max_ind_basis)
    upper_right_bspline_per_dim = linear_to_ordered_index(upper_right_bspline, max_ind_basis)

    lower_right_bspline = ordered_to_linear_index((upper_right_bspline_per_dim[1], lower_left_bspline_per_dim[2]), max_ind_basis)
    upper_left_bspline = ordered_to_linear_index((lower_left_bspline_per_dim[1], upper_right_bspline_per_dim[2]), max_ind_basis)

    return upper_right_bspline, lower_right_bspline, lower_left_bspline, upper_left_bspline
end

@doc raw"""
    initiate_basis_to_check(fe_space::AbstractFiniteElementSpace{n}, marked_elements::Vector{Int}) where {n}

Computes which basis functions are need for L-chain checks.

# Arguments

- `fe_space::AbstractFiniteElementSpace{n}`: finite element space at a given level.
- `marked_elements::Vector{Int}`: marked elements at a given level.

# Returns

- `::Vector{Int}`: basis that need to be checked.
- `::Vector{Int}`: all new basis what will be deactivated.
"""
function initiate_basis_to_check(fe_space::AbstractFiniteElementSpace{n}, marked_elements::Vector{Int}) where {n}
    all_basis_ids = Int[]
    curr_basis_ids = Int[]

    all_corner_basis_ids = Int[]
    all_interior_basis_ids = Int[]

    for el ∈ marked_elements
        _, curr_basis_ids = get_extraction(fe_space, el) 
        curr_corner_basis_ids = get_corner_basis_ids(fe_space, curr_basis_ids)
        curr_interior_basis_ids = setdiff(curr_basis_ids, curr_corner_basis_ids)

        append!(all_corner_basis_ids, curr_corner_basis_ids)
        append!(all_interior_basis_ids, curr_interior_basis_ids)
        append!(all_basis_ids, curr_basis_ids)
    end

    return setdiff(all_corner_basis_ids, all_interior_basis_ids), unique!(all_basis_ids)
end

@doc raw"""
    check_nl_intersection(hspace::HierarchicalFiniteElementSpace{n, S, T}, level::Int, basis_pair, new_operator::T) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Checks whether a pair of basis functions has an (n-1, l+1)-intersection.

# Arguments

- `hspace::HierarchicalFiniteElementSpace{n, S, T}`: hierarchical finite element space.
- `level::Int`: current level.
- `basis_pair::`: pair of basis functions from which the L-chain is contructed.
- `new_operator<:AbstractTwoScaleOperator`: operator to be used when a new level needs to be checked.

# Returns 

- `::Bool`: whether there is an (n-1, l+1)-intersection.
"""
function check_nl_intersection(hspace::HierarchicalFiniteElementSpace{n, S, T}, level::Int, basis_pair, new_operator::T) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    
    basis_supp_per_dim1 = _get_support_per_dim(hspace.spaces[level], basis_pair[1])
    basis_supp_per_dim2 = _get_support_per_dim(hspace.spaces[level], basis_pair[2])

    for k ∈ 1:n
        if first(basis_supp_per_dim2[k]) - last(basis_supp_per_dim1[k]) > 1 || first(basis_supp_per_dim1[k]) - last(basis_supp_per_dim2[k]) > 1 
            return false
        end
    end

    if level == get_num_levels(hspace)
        operator = new_operator
    else
        operator = hspace.two_scale_operators[level]
    end

    p_fine = get_polynomial_degree_per_dim(operator.fine_space)

    length_flag = Vector{Bool}(undef, n)

    for k ∈ 1:n
        if k ==1
            coarse_space = operator.coarse_space.function_space_1
            fine_space = operator.fine_space.function_space_1
            ts = operator.twoscale_operator_1
        elseif k==2 
            coarse_space = operator.coarse_space.function_space_2
            fine_space = operator.fine_space.function_space_2
            ts = operator.twoscale_operator_2
        end

        basis_supp_intersection = StepRange(intersect(basis_supp_per_dim1[k], basis_supp_per_dim2[k]))
        I_k = get_contained_knot_vector(basis_supp_intersection, ts, fine_space)
        
        length_flag[k] = get_knot_vector_length(I_k) > p_fine[k]
    end
    
    return sum(length_flag) >= n-1 ? true : false
end

@doc raw"""
    _get_basis_pair_graph(max_id_basis, basis_per_dim, diff_basis_per_dim, inactive_basis, n::Int)

Creates a local graph of active basis functions in the grid determined by pair of basis functions.

# Arguments

- `max_id_basis::`: dimension wise number of degrees of freedom in the finite element space at a given level.
- `basis_per_dim::`: dimension wise indices of the basis function pair.
- `diff_basis_per_dim::`: dimension wise difference of indices of basis functions pair.
- `inactive_basis::`: the indices of all deactivated basis in `level`.
- `n::Int`: dimension of the finite element space.

# Returns 

- `basis_pair_graph::`:  graph of active basis functions.
"""
function _get_basis_pair_graph(max_id_basis, basis_per_dim, diff_basis_per_dim, inactive_basis, n::Int)
    basis_between_pair_per_dim = [ordered_range(basis_per_dim[1][k], basis_per_dim[2][k]) for k ∈ 1:n]

    basis_pair_graph = Graphs.SimpleGraphs.grid(abs.(diff_basis_per_dim) .+ 1)
    
    basis_to_remove = Int[]
    for (local_idx, ordered_id) ∈ enumerate(Iterators.product(basis_between_pair_per_dim...))
        basis_id = ordered_to_linear_index(ordered_id, max_id_basis)
        if basis_id ∉ inactive_basis
            append!(basis_to_remove, local_idx)
        end
    end
    
    Graphs.SimpleGraphs.rem_vertices!(basis_pair_graph, basis_to_remove, keep_order=true)

    return basis_pair_graph
end

@doc raw"""
    check_shortest_chain(hspace::HierarchicalFiniteElementSpace{n, S, T}, level::Int, basis_pair, inactive_basis) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Checks whether a pair of basis functions has a shortest chain between them.

# Arguments

- `hspace::HierarchicalFiniteElementSpace{n, S, T}`: hierarchical finite element space.
- `level::Int`: current level.
- `basis_pair::`: pair of basis functions from which the L-chain is contructed.
- `inactive_basis::`: the indices of all deactivated basis in `level`.

# Returns 

- `::Bool`: whether there is a shortest chain.
"""
function check_shortest_chain(hspace::HierarchicalFiniteElementSpace{n, S, T}, level::Int, basis_pair, inactive_basis) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    max_id_basis = _get_num_basis_per_space(get_space(hspace, level))
    basis_per_dim = [linear_to_ordered_index(basis_pair[k], max_id_basis) for k ∈ 1:2]
    diff_basis_per_dim = -(basis_per_dim...)

    any(diff_basis_per_dim .== 0) ? (return true) : nothing # check for trivial shortest chain
    
    basis_pair_graph = _get_basis_pair_graph(max_id_basis, basis_per_dim, diff_basis_per_dim, inactive_basis, n)

    Graphs.has_path(basis_pair_graph, 1, Graphs.nv(basis_pair_graph)) ? shortest_chain = true : shortest_chain = false

    return shortest_chain
end

@doc raw"""
    check_problematic_pair(hspace::HierarchicalFiniteElementSpace{n, S, T}, level::Int, basis_pair, inactive_basis, new_operator::T) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Checks whether a pair of basis functions is problematic. I.e., if there is an (n-1, l+1)-intersection and no shortest chain.

# Arguments

- `hspace::HierarchicalFiniteElementSpace{n, S, T}`: hierarchical finite element space.
- `level::Int`: current level.
- `basis_pair::`: pair of basis functions from which the L-chain is contructed.
- `inactive_basis::`: the indices of all deactivated basis in `level`.
- `new_operator<:AbstractTwoScaleOperator`: operator to be used when a new level needs to be checked.

# Returns 

- `problematic_pair::Bool`: whether the pair is problematic.
"""
function check_problematic_pair(hspace::HierarchicalFiniteElementSpace{n, S, T}, level::Int, basis_pair, inactive_basis, new_operator::T) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator} 
    nl_intersection = check_nl_intersection(hspace, level, basis_pair, new_operator)

    if !nl_intersection
        return false
    end

    shortest_chain = check_shortest_chain(hspace, level, basis_pair, inactive_basis)

    problematic_pair = nl_intersection && !shortest_chain

    return problematic_pair
end

@doc raw"""
    build_L_chain(hspace::HierarchicalFiniteElementSpace{n, S, T}, level::Int, basis_pair, chain_type="LR") where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Returns the basis indices of basis functions in the L-chain between the basis in `basis_pair`.

# Arguments

- `hspace::HierarchicalFiniteElementSpace{n, S, T}`: hierarchical finite element space.
- `level::Int`: current level.
- `basis_pair::`: pair of basis functions from which the L-chain is contructed.
- `chain_type::String`: determines the shape of the L-chain. Either "LR" or "UL".

# Returns

- `L_chain::Vector{Int}`: the indices of basis functions in the L-chain, excluding the endpoints.
- `corner_basis::Int`: the index of the basis function in the corner of the L-chain.
"""
function build_L_chain(hspace::HierarchicalFiniteElementSpace{n, S, T}, level::Int, basis_pair, chain_type="LR") where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    L_chain = Int[]
    max_id_basis = _get_num_basis_per_space(get_space(hspace, level))
    basis_per_dim = [linear_to_ordered_index(basis_pair[k], max_id_basis) for k ∈ 1:2]

    # Lower right L-chain
    if chain_type=="LR"
        for first_index ∈ proper_range(basis_per_dim[1][1],basis_per_dim[2][1])
            append!(L_chain, ordered_to_linear_index( (first_index, basis_per_dim[1][2]), max_id_basis))
        end
        for second_index ∈ proper_range(basis_per_dim[1][2], basis_per_dim[2][2])[2:end]
            append!(L_chain, ordered_to_linear_index( (basis_per_dim[2][1], second_index), max_id_basis))
        end
        corner_basis = ordered_to_linear_index( (basis_per_dim[2][1], basis_per_dim[1][2]), max_id_basis)
    # Upper left L-chain
    elseif chain_type=="UL" 
        for second_index ∈ proper_range(basis_per_dim[1][2], basis_per_dim[2][2])
            append!(L_chain, ordered_to_linear_index( (basis_per_dim[1][1], second_index), max_id_basis))
        end
        for first_index ∈ proper_range(basis_per_dim[1][1], basis_per_dim[2][1])
            append!(L_chain, ordered_to_linear_index( (first_index, basis_per_dim[2][2]), max_id_basis))
        end
        corner_basis = ordered_to_linear_index( (basis_per_dim[1][1], basis_per_dim[2][2]), max_id_basis)
    else
        throw(ArgumentError("Invalid chain type. Supported types are \"LR\" or \"UL\" and \"$chain_type\" was given."))
    end

    return L_chain[2:end-1], corner_basis
end

@doc raw"""
    get_level_marked_basis(hspace::HierarchicalFiniteElementSpace{n, S, T}, level::Int, marked_elements_per_level::Vector{Vector{Int}}, new_operator::T) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator} 

Returns the  basis functions in `level` that will contribute to the marked domains. These are given by the basis functions with non-empty support on `marked_elements` togheter with the ones introduced by L-chains.

# Arguments

- `hspace::HierarchicalFiniteElementSpace{n, S, T}`: hierarchical finite element space.
- `level::Int`: current level.
- `marked_elements::Vector{Int}`: marked elements from error analysis.
- `new_two_scale_operator::AbstractTwoScaleOperator`: operator to be used when a new level needs to be created or checked.

# Returns

- `new_inactive_basis::Vector{Int}`: basis used to contrsuct the marked domains for refinement.
"""
function get_level_marked_basis(hspace::HierarchicalFiniteElementSpace{n, S, T}, level::Int, marked_elements_per_level::Vector{Vector{Int}}, new_operator::T) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator} 
    if marked_elements_per_level[level] == Int[]
        return Int[]
    end 

    checked_pairs = Vector{Int}[]
    new_basis_to_check, new_inactive_basis = initiate_basis_to_check(get_space(hspace, level), marked_elements_per_level[level])
    previous_inactive_basis = get_basis_contained_in_next_level(hspace, level)
    
    basis_to_check = union(previous_inactive_basis, new_basis_to_check)
    inactive_basis = union(previous_inactive_basis, new_inactive_basis)
    
    combinations_to_check = Combinatorics.combinations(basis_to_check, 2)
    
    check_count = 1
    while check_count > 0 # Add L-chains until there are no more problematic intersections
        check_count = 0
        
        for basis_pair ∈ combinations_to_check # Loop over unchecked pairs of B-splines
            problematic_pair = check_problematic_pair(hspace, level, basis_pair, inactive_basis, new_operator)
            
            if problematic_pair
                L_chain, corner_basis = build_L_chain(hspace, level, basis_pair) # Choose chain type here default is "LR" 
                append!(basis_to_check, corner_basis)
                append!(inactive_basis, L_chain)
                check_count += 1
            end
        end

        append!(checked_pairs, combinations_to_check)
        combinations_to_check = setdiff(Combinatorics.combinations(basis_to_check, 2), checked_pairs)
    end
    
    return inactive_basis
end

@doc raw"""
    get_marked_element_padding(hspace::HierarchicalFiniteElementSpace{n, S, T}, marked_elements_per_level::Vector{Vector{Int}}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Returns all the elements in the support of basis functions supported on `marked_elements_per_level`.

# Arguments

- `hspace::HierarchicalFiniteElementSpace{n, S, T}`: hierarchical finite element space.
- `marked_elements_per_level::Vector{Vector{Int}}`: marked elements, separated by level.

# Returns

- `element_padding::Vector{Vector{Int}}`: padding of marked elements.
"""
function get_marked_element_padding(hspace::HierarchicalFiniteElementSpace{n, S, T}, marked_elements_per_level::Vector{Vector{Int}}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    num_levels = get_num_levels(hspace)

    element_padding = Vector{Vector{Int}}(undef, num_levels)
    

    for level ∈ 1:num_levels
        if marked_elements_per_level[level] == Int[]
            element_padding[level]= Int[]
            continue
        end
        basis_in_marked_elements = reduce(union, get_basis_indices.(Ref(hspace.spaces[level]), marked_elements_per_level[level]))
        element_padding[level] = reduce(union, get_support.(Ref(hspace.spaces[level]), basis_in_marked_elements))
    end

    return element_padding
end

@doc raw"""
    get_marked_domains(hspace::HierarchicalFiniteElementSpace{n, S, T}, marked_elements::Vector{Int}, new_two_scale_operator::T, L_chain::Bool=false) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Returns a set of domains for a hierarchical space construction based on a set of `marked_elements` from error analysis. When `L_chain` is false the marked domains are given by `get_marked_element_padding`. If `L_chain` is true L-chains will be added when needed aside from the element padding.

# Arguments
- `hspace::HierarchicalFiniteElementSpace{n, S, T}`: hierarchical finite element space.
- `marked_elements::Vector{Int}`: marked elements from error analysis.
- `new_two_scale_operator::AbstractTwoScaleOperator`: operator to be used when a new level needs to be created or checked.
- `L_chain::Bool`: flag for whether L-chains are added or not.
# Returns

- `marked_domains::Vector{Vector{Int}}`: domains for hierarchical space construction.
"""
function get_marked_domains(hspace::HierarchicalFiniteElementSpace{n, S, T}, marked_elements::Vector{Int}, new_two_scale_operator::T, L_chain::Bool=false) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    num_levels = get_num_levels(hspace)
    marked_elements_per_level = convert_element_vector_to_elements_per_level(hspace, marked_elements)
    
    marked_domains = Vector{Vector{Int}}(undef, num_levels)
    marked_domains[1] = Int[]
    if !L_chain
        element_padding = get_marked_element_padding(hspace, marked_elements_per_level)

        for level ∈ 1:1:num_levels
            if level<num_levels
                if element_padding[level] == Int[]
                    marked_domains[level+1] = get_level_domain(hspace, level+1)
                else
                    marked_domains[level+1] = union(get_level_domain(hspace, level+1), get_finer_elements(new_two_scale_operator, element_padding[level]))
                end
            elseif element_padding[level] != Int[]
                push!(marked_domains, get_finer_elements(new_two_scale_operator, element_padding[level]))
            end
        end
    else
        for level ∈ 1:1:num_levels
            level_marked_basis = get_level_marked_basis(hspace, level, marked_elements_per_level, new_two_scale_operator)

            if level<num_levels
                if level_marked_basis == Int[]
                    marked_domains[level+1] = get_level_domain(hspace, level+1)
                else
                    basis_supports = reduce(union, get_support.(Ref(hspace.spaces[level]), level_marked_basis))
                    marked_domains[level+1] = union(get_level_domain(hspace, level+1), basis_supports)
                end
            elseif level_marked_basis != Int[]
                basis_supports = reduce(union, get_support.(Ref(hspace.spaces[level]), level_marked_basis))
                push!(marked_domains, get_finer_elements(new_two_scale_operator, basis_supports))
            end
        end
    end
    

    return marked_domains
end


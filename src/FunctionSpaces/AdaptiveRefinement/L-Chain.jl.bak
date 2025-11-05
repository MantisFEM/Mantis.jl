function proper_range(start::Int, finish::Int)
    if start > finish
        return range(start, finish; step=:(-1))
    end

    return range(start, finish; step=:1)
end

function ordered_range(start::Int, finish::Int)
    if start > finish
        return proper_range(finish, start)
    end

    return proper_range(start, finish)
end

@doc raw"""
    get_corner_basis_ids(fe_space::AbstractFESpace{manifold_dim}, basis_ids::Vector{Int}) where {manifold_dim}

Returns the basis indices of basis functions in the "corners" of a
set of `basis_ids` with non-empty support in a given element.

# Arguments

- `fe_space::AbstractFESpace{manifold_dim}`: finite element space at a given level.
- `basis_ids::Vector{Int}`: set of basis indices with non-empty support in a given element.

# Returns

- `::NTuple{4, Int}`: corner basis function indices in the order "UR", "LR", "LL", "UL".
"""
function get_corner_basis_ids(
    fe_space::AbstractFESpace{manifold_dim}, basis_ids::Vector{Int}
) where {manifold_dim}
    max_ind_basis = get_constituent_num_basis(fe_space)
    lower_left_bspline = minimum(basis_ids)
    upper_right_bspline = maximum(basis_ids)

    lower_left_bspline_per_dim = linear_to_ordered_index(lower_left_bspline, max_ind_basis)
    upper_right_bspline_per_dim = linear_to_ordered_index(
        upper_right_bspline, max_ind_basis
    )

    lower_right_bspline = ordered_to_linear_index(
        (upper_right_bspline_per_dim[1], lower_left_bspline_per_dim[2]), max_ind_basis
    )
    upper_left_bspline = ordered_to_linear_index(
        (lower_left_bspline_per_dim[1], upper_right_bspline_per_dim[2]), max_ind_basis
    )

    return upper_right_bspline, lower_right_bspline, lower_left_bspline, upper_left_bspline
end

@doc raw"""
    initiate_basis_to_check(fe_space::AbstractFESpace{manifold_dim}, marked_element_ids::Vector{Int}) where {manifold_dim}

Computes which basis functions are need for L-chain checks.

# Arguments

- `fe_space::AbstractFESpace{manifold_dim}`: finite element space at a given level.
- `marked_element_ids::Vector{Int}`: marked elements at a given level.

# Returns

- `::Vector{Int}`: basis that need to be checked.
- `::Vector{Int}`: all new basis what will be deactivated.
"""
function initiate_basis_to_check(
    fe_space::AbstractFESpace{manifold_dim}, marked_element_ids::Vector{Int}
) where {manifold_dim}
    function _basis_contained_in_domain(fe_space, marked_element_ids)
        supported_basis_ids = Int[]
        for basis_id in 1:get_num_basis(fe_space)
            basis_support = get_support(fe_space, basis_id)
            if all(map(∈, basis_support, marked_element_ids))
                append!(supported_basis_ids, basis_id)
            end
        end
        return supported_basis_ids
    end

    all_basis_ids = Int[]
    curr_basis_ids = Int[]

    all_corner_basis_ids = Int[]
    all_interior_basis_ids = Int[]

    for el in marked_element_ids
        curr_basis_ids = get_basis_indices(fe_space, el)
        curr_corner_basis_ids = get_corner_basis_ids(fe_space, curr_basis_ids)
        curr_basis_ids = get_basis_indices(fe_space, el)
        curr_interior_basis_ids = setdiff(curr_basis_ids, curr_corner_basis_ids)

        append!(all_corner_basis_ids, curr_corner_basis_ids)
        append!(all_interior_basis_ids, curr_interior_basis_ids)
        append!(all_basis_ids, curr_basis_ids)
    end

    return setdiff(all_corner_basis_ids, all_interior_basis_ids), unique!(all_basis_ids)
end

function initiate_check_and_inactive_basis(
    hier_space::HierarchicalFiniteElementSpace{2}, level::Int, marked_elements::Set{Int}
)
    # marked_elements here should already be union of marked_elements from estimator and inactive elements on level

    inactive_basis = Set{Int}()
    sizehint!(inactive_basis, length(marked_elements))

    # 1. loop over marked elements to get all basis that have support on the elements

    for element_id in marked_elements
        # we want the indices in the indexing of the underlying space, not the hierarchical indexing!
        element_basis_ids = get_basis_indices(get_space(hier_space, level), element_id)
        for basis_id in element_basis_ids
            if issubset(
                Set(get_support(get_space(hier_space, level), basis_id)), marked_elements
            )
                push!(inactive_basis, basis_id)
            end
        end
    end

    # 2. check side configurations of inactive basis (only works in 2D!)

    basis_to_check = Set{Int}()
    sizehint!(basis_to_check, length(inactive_basis))

    max_ind_basis = get_constituent_num_basis(get_space(hier_space, level))
    left_basis = [0, 0]
    right_basis = [0, 0]
    down_basis = [0, 0]
    up_basis = [0, 0]
    for basis in inactive_basis
        basis_id_per_dim = linear_to_ordered_index(basis, max_ind_basis)
        # check if (±, x) and skip basis it is true
        @. left_basis = basis_id_per_dim - (1, 0)
        @. right_basis = basis_id_per_dim + (1, 0)
        if ordered_to_linear_index(left_basis, max_ind_basis) ∈ inactive_basis &&
            ordered_to_linear_index(right_basis, max_ind_basis) ∈ inactive_basis
            continue
        end
        # check if (x, ±)
        @. down_basis = basis_id_per_dim - (0, 1)
        @. up_basis = basis_id_per_dim + (0, 1)
        if ordered_to_linear_index(down_basis, max_ind_basis) ∈ inactive_basis &&
            ordered_to_linear_index(up_basis, max_ind_basis) ∈ inactive_basis
            continue
        end
        push!(basis_to_check, basis)
    end

    return basis_to_check, inactive_basis
end

@doc raw"""
    check_nl_intersection(hier_space::HierarchicalFiniteElementSpace{manifold_dim, S, T}, level::Int, basis_pair, new_operator::T) where {manifold_dim, S<:AbstractFESpace{manifold_dim}, T<:AbstractTwoScaleOperator}

Checks whether a pair of basis functions has an (manifold_dim-1, l+1)-intersection.

# Arguments

- `hier_space::HierarchicalFiniteElementSpace{manifold_dim, S, T}`: hierarchical finite element space.
- `level::Int`: current level.
- `basis_pair::`: pair of basis functions from which the L-chain is contructed.
- `new_operator<:AbstractTwoScaleOperator`: operator to be used when a new level needs to be checked.

# Returns

- `::Bool`: whether there is an (manifold_dim-1, l+1)-intersection.
"""
function check_nl_intersection(
    hier_space::HierarchicalFiniteElementSpace{2, num_components, num_patches, S, T},
    level::Int,
    basis_pair,
    new_operator::T,
) where {
    num_components, num_patches, S <: AbstractFESpace{2}, T <: AbstractTwoScaleOperator
}
    level_space = get_space(hier_space, level)
    basis_supp_per_dim1 = get_constituent_support(level_space, basis_pair[1])
    basis_supp_per_dim2 = get_constituent_support(level_space, basis_pair[2])

    for k in 1:2
        if first(basis_supp_per_dim2[k]) - last(basis_supp_per_dim1[k]) > 1 ||
            first(basis_supp_per_dim1[k]) - last(basis_supp_per_dim2[k]) > 1
            return false
        end
    end

    if level == get_num_levels(hier_space)
        operator = new_operator
    else
        operator = hier_space.two_scale_operators[level]
    end

    p_fine = get_constituent_polynomial_degree(get_child_space(operator))

    length_flag = Vector{Bool}(undef, 2)
    const_twoscale_operators = get_constituent_twoscale_operators(operator)

    for k in 1:2
        ts = const_twoscale_operators[k]
        fine_space = get_child_space(ts)
        min_basis_1 = minimum(basis_supp_per_dim1[k])
        max_basis_1 = maximum(basis_supp_per_dim1[k])
        min_basis_2 = minimum(basis_supp_per_dim2[k])
        max_basis_2 = maximum(basis_supp_per_dim2[k])
        intersection_boundary_breakpoints = (
            maximum((min_basis_1, min_basis_2)), minimum((max_basis_1, max_basis_2))
        )

        I_k = get_contained_knot_vector(intersection_boundary_breakpoints, ts, fine_space)

        length_flag[k] = get_knot_vector_length(I_k) > p_fine[k]
    end

    return sum(length_flag) >= 1 ? true : false
end

function get_contained_knot_vector(
    boundary_breakpoints::NTuple{2, Int}, ts::T, fine_space::BSplineSpace
) where {T <: AbstractTwoScaleOperator}
    if boundary_breakpoints[1] == boundary_breakpoints[2]
        breakpoint_idxs = get_element_children(ts, boundary_breakpoints[1])[1]
    else
        element_idxs = Int[]
        # A breakpoint i is associated to element [ξᵢ, ξᵢ₊₁]
        for element in boundary_breakpoints
            append!(element_idxs, get_element_children(ts, element))
        end

        breakpoint_idxs = minimum(element_idxs):(maximum(element_idxs) + 1)
    end

    breakpoints = get_patch(fine_space).breakpoints[breakpoint_idxs]
    multiplicity = get_multiplicity_vector(fine_space)[breakpoint_idxs]

    return KnotVector(
        Mesh.Patch1D(breakpoints), get_polynomial_degree(fine_space), multiplicity
    )
end

@doc raw"""
    _get_basis_pair_graph(max_id_basis, basis_per_dim, diff_basis_per_dim, inactive_basis, manifold_dim::Int)

Creates a local graph of active basis functions in the grid determined by pair of basis functions.

# Arguments

- `max_id_basis::`: dimension wise number of degrees of freedom in the finite element space at a given level.
- `basis_per_dim::`: dimension wise indices of the basis function pair.
- `diff_basis_per_dim::`: dimension wise difference of indices of basis functions pair.
- `inactive_basis::`: the indices of all deactivated basis in `level`.
- `manifold_dim::Int`: dimension of the finite element space.

# Returns

- `basis_pair_graph::`:  graph of active basis functions.
"""
function _get_basis_pair_graph(
    max_id_basis, basis_per_dim, diff_basis_per_dim, inactive_basis, manifold_dim::Int
)
    basis_between_pair_per_dim = [
        ordered_range(basis_per_dim[1][k], basis_per_dim[2][k]) for k in 1:manifold_dim
    ]

    basis_pair_graph = Graphs.SimpleGraphs.grid(abs.(diff_basis_per_dim) .+ 1)

    basis_to_remove = Int[]
    for (local_idx, ordered_id) in
        enumerate(Iterators.product(basis_between_pair_per_dim...))
        basis_id = ordered_to_linear_index(ordered_id, max_id_basis)
        if basis_id ∉ inactive_basis
            append!(basis_to_remove, local_idx)
        end
    end

    Graphs.SimpleGraphs.rem_vertices!(basis_pair_graph, basis_to_remove; keep_order=true)

    return basis_pair_graph
end

@doc raw"""
    check_shortest_chain(hier_space::HierarchicalFiniteElementSpace{manifold_dim, S, T}, level::Int, basis_pair, inactive_basis) where {manifold_dim, S<:AbstractFESpace{manifold_dim}, T<:AbstractTwoScaleOperator}

Checks whether a pair of basis functions has a shortest chain between them.

# Arguments

- `hier_space::HierarchicalFiniteElementSpace{manifold_dim, S, T}`: hierarchical finite element space.
- `level::Int`: current level.
- `basis_pair::`: pair of basis functions from which the L-chain is contructed.
- `inactive_basis::`: the indices of all deactivated basis in `level`.

# Returns

- `::Bool`: whether there is a shortest chain.
"""
function check_shortest_chain(
    hier_space::HierarchicalFiniteElementSpace{2}, level::Int, basis_pair, inactive_basis
)
    max_id_basis = get_constituent_num_basis(get_space(hier_space, level))
    basis_per_dim = [linear_to_ordered_index(basis_pair[k], max_id_basis) for k in 1:2]
    diff_basis_per_dim = -(basis_per_dim...)

    any(diff_basis_per_dim .== 0) ? (return true) : nothing # check for trivial shortest chain

    basis_pair_graph = _get_basis_pair_graph(
        max_id_basis, basis_per_dim, diff_basis_per_dim, inactive_basis, 2
    )

    if Graphs.has_path(basis_pair_graph, 1, Graphs.nv(basis_pair_graph))
        shortest_chain = true
    else
        shortest_chain = false
    end

    return shortest_chain
end

@doc raw"""
    check_problematic_pair(hier_space::HierarchicalFiniteElementSpace{manifold_dim, S, T}, level::Int, basis_pair, inactive_basis, new_operator::T) where {manifold_dim, S<:AbstractFESpace{manifold_dim}, T<:AbstractTwoScaleOperator}

Checks whether a pair of basis functions is problematic. I.e., if there is an (manifold_dim-1, l+1)-intersection and no shortest chain.

# Arguments

- `hier_space::HierarchicalFiniteElementSpace{manifold_dim, S, T}`: hierarchical finite element space.
- `level::Int`: current level.
- `basis_pair::`: pair of basis functions from which the L-chain is contructed.
- `inactive_basis::`: the indices of all deactivated basis in `level`.
- `new_operator<:AbstractTwoScaleOperator`: operator to be used when a new level needs to be checked.

# Returns

- `problematic_pair::Bool`: whether the pair is problematic.
"""
function check_problematic_pair(
    hier_space::HierarchicalFiniteElementSpace{2, num_components, num_patches, S, T},
    level::Int,
    basis_pair,
    inactive_basis,
    new_operator::T,
) where {
    num_components, num_patches, S <: AbstractFESpace{2}, T <: AbstractTwoScaleOperator
}
    nl_intersection = check_nl_intersection(hier_space, level, basis_pair, new_operator)

    if !nl_intersection
        return false
    end

    shortest_chain = check_shortest_chain(hier_space, level, basis_pair, inactive_basis)

    return !shortest_chain
end

@doc raw"""
    build_Lchain(hier_space::HierarchicalFiniteElementSpace{manifold_dim, S, T}, level::Int, basis_pair, chain_type="LR") where {manifold_dim, S<:AbstractFESpace{manifold_dim}, T<:AbstractTwoScaleOperator}

Returns the basis indices of basis functions in the L-chain between the basis in `basis_pair`.

# Arguments

- `hier_space::HierarchicalFiniteElementSpace{manifold_dim, S, T}`: hierarchical finite element space.
- `level::Int`: current level.
- `basis_pair::`: pair of basis functions from which the L-chain is contructed.
- `chain_type::String`: determines the shape of the L-chain. Either "LR" or "UL".

# Returns

- `Lchain::Vector{Int}`: the indices of basis functions in the L-chain, excluding the endpoints.
- `corner_basis::Int`: the index of the basis function in the corner of the L-chain.
"""
function build_Lchain(
    hier_space::HierarchicalFiniteElementSpace{2}, level::Int, basis_pair, chain_type="LR"
)
    Lchain = Int[]
    max_id_basis = get_constituent_num_basis(get_space(hier_space, level))
    basis_per_dim = [linear_to_ordered_index(basis_pair[k], max_id_basis) for k in 1:2]

    # Lower right L-chain
    if chain_type == "LR"
        for first_index in proper_range(basis_per_dim[1][1], basis_per_dim[2][1])
            append!(
                Lchain,
                ordered_to_linear_index((first_index, basis_per_dim[1][2]), max_id_basis),
            )
        end
        for second_index in proper_range(basis_per_dim[1][2], basis_per_dim[2][2])[2:end]
            append!(
                Lchain,
                ordered_to_linear_index((basis_per_dim[2][1], second_index), max_id_basis),
            )
        end
        corner_basis = ordered_to_linear_index(
            (basis_per_dim[2][1], basis_per_dim[1][2]), max_id_basis
        )
        # Upper left L-chain
    elseif chain_type == "UL"
        for second_index in proper_range(basis_per_dim[1][2], basis_per_dim[2][2])
            append!(
                Lchain,
                ordered_to_linear_index((basis_per_dim[1][1], second_index), max_id_basis),
            )
        end
        for first_index in proper_range(basis_per_dim[1][1], basis_per_dim[2][1])
            append!(
                Lchain,
                ordered_to_linear_index((first_index, basis_per_dim[2][2]), max_id_basis),
            )
        end
        corner_basis = ordered_to_linear_index(
            (basis_per_dim[1][1], basis_per_dim[2][2]), max_id_basis
        )
    else
        throw(
            ArgumentError(
                "Invalid chain type. Supported types are \"LR\" or \"UL\" and \"$chain_type\" was given.",
            ),
        )
    end

    return Lchain[2:(end - 1)], corner_basis
end

function _compute_Lchain_basis(
    hier_space::HierarchicalFiniteElementSpace{2, num_components, num_patches, S, T},
    level::Int,
    marked_elements::Vector{Int},
    new_operator::T,
) where {
    num_components, num_patches, S <: AbstractFESpace{2}, T <: AbstractTwoScaleOperator
}
    if marked_elements == Int[]
        return Int[]
    end

    num_levels = get_num_levels(hier_space)

    if level < num_levels
        # basis_to_check are all the inactive basis of the current level that have a side configuration
        # which requires checking i.e. different from (±,x) or (x, ±) where x can be anything.
        # inactive_basis are all the inactive basis of the current level, used for checking wheter a
        # shortest chain exists or not
        # Note: there is an implicit assumption that the marked_elements are given as the supports of 0-forms

        level_inactive_elements = map(
            element ->
                get_element_parent(get_twoscale_operator(hier_space, level), element),
            get_level_domain(hier_space, level + 1),
        )
        inactive_elements = Set([marked_elements; level_inactive_elements])

        basis_to_check, inactive_basis = initiate_check_and_inactive_basis(
            hier_space, level, inactive_elements
        )
    else
        basis_to_check, inactive_basis = initiate_check_and_inactive_basis(
            hier_space, level, Set(marked_elements)
        )
    end

    basis_to_check = collect(basis_to_check)
    inactive_basis = collect(inactive_basis)

    combinations_to_check = Combinatorics.combinations(basis_to_check, 2)

    checked_pairs = Vector{Int}[]
    Lchain_basis_ids = Int[]

    check_count = 1
    while check_count > 0 # Add L-chains until there are no more problematic intersections
        check_count = 0

        for basis_pair in combinations_to_check # Loop over unchecked pairs of B-splines
            problematic_pair = check_problematic_pair(
                hier_space, level, basis_pair, inactive_basis, new_operator
            )

            if problematic_pair
                # Choose chain type. Default is "LR"
                Lchain, corner_basis = build_Lchain(hier_space, level, basis_pair)
                push!(basis_to_check, corner_basis)
                append!(inactive_basis, Lchain)
                append!(Lchain_basis_ids, Lchain)
                check_count += 1
            end
        end

        append!(checked_pairs, combinations_to_check)
        combinations_to_check = setdiff(
            Combinatorics.combinations(basis_to_check, 2), checked_pairs
        )
    end

    return Lchain_basis_ids
end

# TODO: This file should be changed to match the L-chain paper algorithm.

function add_Lchains_supports!(
    marked_elements_per_level::Vector{Vector{Int}},
    hier_space::HierarchicalFiniteElementSpace{2, num_components, num_patches, S, T},
    new_operator::T,
) where {
    num_components, num_patches, S <: AbstractFESpace{2}, T <: AbstractTwoScaleOperator
}
    num_levels = get_num_levels(hier_space)

    for level in 1:num_levels
        Lchain_basis_ids = _compute_Lchain_basis(
            hier_space, level, marked_elements_per_level[level], new_operator
        )
        if Lchain_basis_ids == Int[]
            continue
        end
        Lchain_supports = union(
            map(
                basis -> get_support(get_space(hier_space, level), basis), Lchain_basis_ids
            )...,
        )
        union!(marked_elements_per_level[level], Lchain_supports)
    end

    return marked_elements_per_level
end

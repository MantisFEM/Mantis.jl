"""
	update_space_with_lchains!(
	    space::HierarchicalFiniteElementSpace{2}, marked_els::Vector{Vector{Int}}
	)

Takes as input a `space` corresponding to an exact de Rham complex and a set of `marked_els`
for refinement, and returns  a refined space corresponding to an exact complex by adding
L-chains where needed. We refer the reader to [Cabanas2025](@cite) for further details.

# Arguments
- `space::HierarchicalFiniteElementSpace{2}`: The space to be updated.
- `marked_els::Vector{Vector{Int}}`: The marked elements.

# Returns
- `space::HierarchicalFiniteElementSpace{2}`: The refined space, with no problematic pairs.
"""
function update_space_with_lchains!(
    space::HierarchicalFiniteElementSpace{2}, marked_els::Vector{Vector{Int}}
)
    previous_parents = Int[]
    L = get_num_levels(space)
    for level in L:-1:1
        level_marked_els = marked_els[level]
        if isempty(level_marked_els)
            continue
        end

        level_space = get_space(space, level)
        refine_mesh!(space, level, level_marked_els)
        Blk = get_Blk(space, level)
        unchecked_pairs = initiate_pairs(space, level, Blk, level_marked_els)
        problematic_mesh = true
        level_corners = Int[]
        while problematic_mesh
            problematic_mesh = false
            current_corners = Int[]
            for (βᵢ, βⱼ) in unchecked_pairs
                if is_problematic(space, level, Blk, (βᵢ, βⱼ))
                    corner = get_lchain_corner(space, level, Blk, (βᵢ, βⱼ))
                    append!(current_corners, corner)
                    problematic_mesh = true
                end
            end

            if !problematic_mesh
                break
            end

            refine_mesh!(
                space,
                level,
                mapreduce(c -> get_support(level_space, c), union, current_corners),
            )
            Blk = get_Blk(space, level)
            unchecked_pairs = get_local_pairs(space, level, Blk, current_corners)
            union!(level_corners, current_corners)
        end

        if level == 1
            continue
        end

        level_corners = level_corners ∪ previous_parents
        refined_elements = marked_els[level - 1] ∪ get_level_domain(space, level - 1)
        previous_parents = Int[]
        pl_space = get_space(space, level - 1)
        for βᵢ in level_corners
            if !isempty(get_support(level_space, βᵢ) ∩ refined_elements)
                parent = get_parent_function(space, level, βᵢ)
                push!(previous_parents, parent)
                union!(marked_els[level - 1], get_support(pl_space, parent))
            end
        end
    end

    return update_basis!(space)
end

"""
	get_Blk(space::HierarchicalFiniteElementSpace, l::Int, k::Int)

Returns the basis indices the original space at level `l` whose support is contained
in the domain `Ωₖ`.

# Returns
- `Vector{Int}`: The basis contained in the domain `Ωₖ`.
"""
function get_Blk(space::HierarchicalFiniteElementSpace, l::Int, k::Int)
    if k < l
        throw(ArgumentError("The given argument k=$(k) should be higher than l=$(l)."))
    end

    Blk = Int[]
    L = get_num_levels(space)
    if l == L
        return Blk
    end

    Ωₖ = get_level_domain(space, k)
    ops = ntuple(lvl -> get_twoscale_operator(space, l - 1 + lvl), (k - l))
    for βᵢ in 1:get_num_basis(get_space(space, l))
        supp_βᵢ = get_support(get_space(space, l), βᵢ)
        for lvl in 1:(k - l)
            supp_βᵢ = mapreduce(e -> get_element_children(ops[lvl], e), vcat, supp_βᵢ)
        end

        if all(e ∈ Ωₖ for e in supp_βᵢ)
            push!(Blk, βᵢ)
        end
    end

    return Blk
end

get_Blk(space::HierarchicalFiniteElementSpace, l::Int) = get_Blk(space, l, l + 1)

"""
	initiate_pairs(
	    space::HierarchicalFiniteElementSpace{2},
	    level::Int,
	    Blk::Vector{Int},
	    marked_els::Vector{Int},
	)

Generates all the possibly problematic pairs at `level` that need to be checked for
problems.

# Returns
- `Vector{Tuple{Int, Int}}`: The pairs that need to be checked for problems.

See also [`update_space_with_lchains!`](@ref), [`get_Blk`](@ref) and [`get_local_pairs`](@ref).
"""
function initiate_pairs(
    space::HierarchicalFiniteElementSpace{2},
    level::Int,
    Blk::Vector{Int},
    marked_els::Vector{Int},
)
    level_space = get_space(space, level)
    unchecked = Int[]
    for βᵢ in Blk
        if !is_resolved(space, level, Blk, βᵢ) &&
            !isempty(get_support(level_space, βᵢ) ∩ marked_els)
            append!(unchecked, βᵢ)
        end
    end

    unchecked_pairs = get_local_pairs(space, level, Blk, unchecked)

    return unchecked_pairs
end

"""
	is_resolved(
	    space::HierarchicalFiniteElementSpace{2}, level::Int, Blk::Vector{Int}, βᵢ::Int
	)

Checks whether the basis function `βᵢ` at `level` is resolved or not.

# Returns
- `Bool`: Whether `βᵢ` is resolved.
"""
function is_resolved(
    space::HierarchicalFiniteElementSpace{2}, level::Int, Blk::Vector{Int}, βᵢ::Int
)
    level_space = get_space(space, level)
    const_βᵢ = get_constituent_basis_id(level_space, βᵢ)
    const_num_basis = get_constituent_num_basis(level_space)
    lin_num_basis = get_lin_num_basis(level_space)
    for k in 1:2
        const_left_βᵢ = const_βᵢ .- (k .== (1, 2))
        const_right_βᵢ = const_βᵢ .+ (k .== (1, 2))
        if const_βᵢ[k] == 1
            right_βᵢ = lin_num_basis[const_right_βᵢ...]
            if right_βᵢ ∈ Blk
                return true
            else
                continue
            end
        elseif const_βᵢ[k] == const_num_basis[k]
            left_βᵢ = lin_num_basis[const_left_βᵢ...]
            if left_βᵢ ∈ Blk
                return true
            else
                continue
            end
        end

        left_βᵢ = lin_num_basis[const_left_βᵢ...]
        right_βᵢ = lin_num_basis[const_right_βᵢ...]
        if left_βᵢ ∈ Blk && right_βᵢ ∈ Blk
            return true
        end
    end

    return false
end

"""
	get_local_pairs(
	    space::HierarchicalFiniteElementSpace{2},
	    level::Int,
	    Blk::Vector{Int},
	    unchecked::Vector{Int},
	)

Returns a list of pairs of basis functions that need to be checked for problems from a set
of `unchecked` basis functions.

# Arguments
- `unchecked::Vector{Int}`: A list of unresolved basis functions.

# Returns
- `Vector{Tuple{Int, Int}}`: The list of possibly problematic pairs.

See also [`initiate_pairs`](@ref) and [`is_resolved`](@ref).
"""
function get_local_pairs(
    space::HierarchicalFiniteElementSpace{2},
    level::Int,
    Blk::Vector{Int},
    unchecked::Vector{Int},
)
    pairs = Tuple{Int, Int}[]
    for βᵢ in unchecked
        for βⱼ in get_interaction_box(space, level, Blk, βᵢ)
            if !is_resolved(space, level, Blk, βⱼ) && βᵢ != βⱼ
                # We do this so that we can call unique! after
                if βᵢ < βⱼ
                    push!(pairs, (βᵢ, βⱼ))
                else
                    push!(pairs, (βⱼ, βᵢ))
                end
            end
        end
    end

    return unique!(pairs)
end

"""
	get_interaction_box(
	    space::HierarchicalFiniteElementSpace{2}, level::Int, Blk::Vector{Int}, βᵢ::Int
	)

Returns a list of basis functions that are at most `p[k]+1` away from `βᵢ` in index space
for each manifold dimension `k`, where `p[k]` is the polynomial degree.

# Returns
- `Vector{Int}`: The list of basis functions interacting with `βᵢ`.
"""
function get_interaction_box(
    space::HierarchicalFiniteElementSpace{2}, level::Int, Blk::Vector{Int}, βᵢ::Int
)
    level_space = get_space(space, level)
    p = get_constituent_polynomial_degree(level_space)
    const_num_basis = get_constituent_num_basis(level_space)
    const_βᵢ = get_constituent_basis_id(level_space, βᵢ)
    lin_num_basis = get_lin_num_basis(level_space)
    inter_box = Int[]
    for offset_1 in (-(p[1] + 1)):(p[1] + 1), offset_2 in (-(p[2] + 1)):(p[2] + 1)
        if offset_1 == 0 && offset_2 == 0
            continue
        end

        lj = const_βᵢ[1] + offset_1
        rj = const_βᵢ[2] + offset_2
        if lj < 1 || lj > const_num_basis[1] || rj < 1 || rj > const_num_basis[2]
            continue
        end

        βⱼ = lin_num_basis[lj, rj]
        if βⱼ ∈ Blk
            push!(inter_box, βⱼ)
        end
    end

    return inter_box
end

"""
	is_problematic(
	    space::HierarchicalFiniteElementSpace{2},
	    level::Int,
	    Blk::Vector{Int},
	    (βᵢ, βⱼ)::Tuple{Int, Int},
	)

Checks whether a `(βᵢ, βⱼ)` is a problematic pair.

# Returns
- `Bool`: Whether the pair is problematic.
"""
function is_problematic(
    space::HierarchicalFiniteElementSpace{2},
    level::Int,
    Blk::Vector{Int},
    (βᵢ, βⱼ)::Tuple{Int, Int},
)
    return has_minimal_intersection(space, level, (βᵢ, βⱼ)) &&
           !has_shortest_chain(space, level, Blk, (βᵢ, βⱼ))
end

"""
	has_minimal_intersection(
	    space::HierarchicalFiniteElementSpace{2}, level::Int, (βᵢ, βⱼ)::Tuple{Int, Int}
	)

Checks whether a `(βᵢ, βⱼ)` share a minimal-(l+1) intersection.

# Returns
- `Bool`: Whether the pair shares a minimal intersection.
"""
function has_minimal_intersection(
    space::HierarchicalFiniteElementSpace{2}, level::Int, (βᵢ, βⱼ)::Tuple{Int, Int}
)
    level_space = get_space(space, level)
    const_supp_1 = get_constituent_support(level_space, βᵢ)
    const_supp_2 = get_constituent_support(level_space, βⱼ)
    operator = get_twoscale_operator(space, level)
    p_fine = get_constituent_polynomial_degree(get_child_space(operator))
    const_twoscale_operators = get_constituent_twoscale_operators(operator)
    for k in 1:2
        ts = const_twoscale_operators[k]
        child_space = get_child_space(ts)
        lv1, rv1 = const_supp_1[k][1], const_supp_1[k][end]
        lv2, rv2 = const_supp_2[k][1], const_supp_2[k][end]
        intersection_boundary_breakpoints = (maximum((lv1, lv2)), minimum((rv1, rv2)) + 1)
        I_k = get_contained_knot_vector(intersection_boundary_breakpoints, ts, child_space)
        if get_knot_vector_length(I_k) > p_fine[k]
            return true
        end
    end

    return false
end

"""
	get_contained_knot_vector(
	    boundary_breakpoints::NTuple{2, Int},
	    ts::AbstractTwoScaleOperator,
	    fine_space::BSplineSpace,
	)

Returns a `KnotVector` corresponding to the largest subset of the knot-vector defining
`fine_space` that is contained between `boundary_breakpoints`.

# Returns
- `KnotVector`: The largest knot-vector contained between `boundary_breakpoints`.
"""
function get_contained_knot_vector(
    boundary_breakpoints::NTuple{2, Int},
    ts::AbstractTwoScaleOperator,
    fine_space::BSplineSpace,
)
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

"""
	has_shortest_chain(
	    space::HierarchicalFiniteElementSpace{2},
	    level::Int,
	    Blk::Vector{Int},
	    (βᵢ, βⱼ)::Tuple{Int, Int},
	)

Checks whether a `(βᵢ, βⱼ)` have a shortest chain between them.

# Returns
- `Bool`: Whether `(βᵢ, βⱼ)` have a shortest chain between them.
"""
function has_shortest_chain(
    space::HierarchicalFiniteElementSpace{2},
    level::Int,
    Blk::Vector{Int},
    (βᵢ, βⱼ)::Tuple{Int, Int},
)
    level_space = get_space(space, level)
    lin_num_basis = get_lin_num_basis(level_space)
    const_βᵢ = get_constituent_basis_id(level_space, βᵢ)
    const_βⱼ = get_constituent_basis_id(level_space, βⱼ)
    const_diff = const_βᵢ .- const_βⱼ
    if any(map(iszero, const_diff))
        return true
    end

    inter_box_i = get_interaction_box(space, level, Blk, βᵢ)
    inter_box_j = get_interaction_box(space, level, Blk, βⱼ)
    inter_box_ij = inter_box_i ∩ inter_box_j
    verts_to_rmv = Int[]
    sign_1 = sign(const_diff[1])
    sign_2 = sign(const_diff[2])
    for (v, offset) in
        enumerate(CartesianIndices((0:abs(const_diff[1]), 0:abs(const_diff[2]))))
        βₜ = lin_num_basis[
            const_βᵢ[1] - sign_1 * offset[1], const_βᵢ[2] - sign_2 * offset[2]
        ]
        if βₜ ∈ (βᵢ, βⱼ)
            continue
        end

        if βₜ ∉ inter_box_ij
            push!(verts_to_rmv, v)
        end
    end

    graph = Graphs.SimpleGraphs.grid(abs.(const_diff) .+ 1)
    Graphs.SimpleGraphs.rem_vertices!(graph, verts_to_rmv; keep_order=true)

    return Graphs.has_path(graph, 1, Graphs.nv(graph))
end

"""
	get_lchain_corner(
	    space::HierarchicalFiniteElementSpace{2},
	    level::Int,
	    Blk::Vector{Int},
	    (βᵢ, βⱼ)::Tuple{Int, Int},
	)

Returns a corner basis function of an L-chain between `βᵢ` and `βⱼ`, with preference for
resolved corners.

# Returns
- `Int`: The id of the corner basis function.
"""
function get_lchain_corner(
    space::HierarchicalFiniteElementSpace{2},
    level::Int,
    Blk::Vector{Int},
    (βᵢ, βⱼ)::Tuple{Int, Int},
)
    level_space = get_space(space, level)
    lin_num_basis = get_lin_num_basis(level_space)
    const_βᵢ = get_constituent_basis_id(level_space, βᵢ)
    const_βⱼ = get_constituent_basis_id(level_space, βⱼ)
    corner = lin_num_basis[const_βᵢ[1], const_βⱼ[2]]
    if is_resolved(space, level, Blk, corner)
        return corner
    end

    return lin_num_basis[const_βⱼ[1], const_βᵢ[2]]
end

"""
	get_parent_function(space::HierarchicalFiniteElementSpace, level::Int, βᵢ::Int)

Returns the first basis function of `βᵢ`.

# Returns
- `Int`: The id of the parent basis function.
"""
function get_parent_function(space::HierarchicalFiniteElementSpace, level::Int, βᵢ::Int)
    operator = get_twoscale_operator(space, level - 1)
    parents = get_basis_parents(operator, βᵢ)

    return parents[1]
end

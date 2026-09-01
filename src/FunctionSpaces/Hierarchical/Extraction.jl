
############################################################################################
#                                         Options                                          #
############################################################################################

abstract type BasisType end

struct HB <: BasisType end
struct THB <: BasisType end

# HB does modify refinenment or coefficients
_mod_refinement!(R::AbstractMatrix, ::Type{HB}, ::BitVector) = R
_mod_coeffs_and_ids!(E::AbstractMatrix, I::AbstractVector, ::Type{HB}) = E, I

function _mod_refinement!(R::AbstractMatrix, ::Type{THB}, active_indices::BitVector)
    A = LinearAlgebra.Diagonal(active_indices)
    # Truncate R
    R = (LinearAlgebra.I - A) * R

    return R
end

function _mod_coeffs_and_ids!(E::AbstractMatrix, I::AbstractVector, ::Type{THB})
    keep = trues(size(E, 2))
    @inbounds for c in axes(E, 2)
        # Remove fully trimmed basis functions
        keep[c] = !all(iszero, @view E[:, c])
    end

    return E[:, keep], I[keep]
end

############################################################################################
#                                       Coefficients                                       #
############################################################################################

function get_coarsest_level(geometry, basis, element_id::Int)
    #=
    We start by gathering all the ancestors from level 1 up to (and not including) the
    element level
    =#
    ancestors = Geometry.get_ancestors(geometry, element_id)
    for (l, id) in enumerate(ancestors)
        space = Hierarchical.get_set(basis, l)
        # Get all basis indices supported on the ancestor
        local_indices = get_basis_indices(space, id)
        l_set = Hierarchical.get_level_set(basis, l)
        # At least one basis function is active, so we return its level.
        if !isdisjoint(local_indices, l_set)
            return l
        end
    end

    #=
    No active basis functions from coarser levels are supported on the element, so we
    return its level
    =#
    return length(ancestors) + 1
end

function _get_coarsest_level_with_ancestors(geometry, basis, element_id)
    ancestors = Geometry.get_ancestors(geometry, element_id)
    spaces = Hierarchical.get_sets(basis)
    for (l, (id, space)) in enumerate(zip(ancestors, spaces))
        local_indices = get_basis_indices(space, id)
        l_set = Hierarchical.get_level_set(basis, l)
        # true if ancestor has active supported basis functions, false otherwise.
        if !isdisjoint(local_indices, l_set)
            return l, ancestors
        end
    end

    return length(ancestors) + 1, ancestors
end

############################################################################################
#                                         Indices                                          #
############################################################################################

function get_active_indices(
    geometry::Geometry.HierarchicalGeometry, basis::Hierarchical.Hierarchy, element_id::Int
)
    level, level_id = Geometry.convert_to_level_and_level_id(geometry, element_id)

    return get_active_indices(basis, level, level_id)
end

function get_active_indices(basis::Hierarchical.Hierarchy, level::Int, level_id::Int)
    space = Hierarchical.get_set(basis, level)
    active_set = Hierarchical.get_level_set(basis, level)
    # All basis indices from that level support on the element
    local_indices = get_basis_indices(space, level_id)
    active_indices = BitVector(undef, length(local_indices))
    for (i, B) in enumerate(local_indices)
        # 1 if active, 0 otherwise
        active_indices[i] = B in active_set
    end

    return active_indices
end

############################################################################################
#                                          Main!                                           #
############################################################################################

function build_extraction_operator(
    geometry, basis::Hierarchical.Hierarchy, ::Type{B}
) where {B <: BasisType}
    # Initialise parameters
    spaces = Hierarchical.get_sets(basis)
    num_elements = Geometry.get_num_elements(geometry)
    num_basis = Hierarchical.get_num_objects(basis)
    num_components = get_num_components(first(spaces))
    extraction_coefficients = Vector{NTuple{num_components, Matrix{Float64}}}(
        undef, num_elements
    )
    basis_indices = Vector{Indices{num_components, Vector{Int}, UnitRange{Int}}}(
        undef, num_elements
    )

    NC = Val(num_components)
    # Compute extraction and indicies for each element 
    Threads.@threads for element_id in eachindex(extraction_coefficients, basis_indices)
        E, I = build_local_extraction(geometry, basis, B, NC, element_id)
        extraction_coefficients[element_id] = E
        basis_indices[element_id] = I
    end

    return ExtractionOperator(
        extraction_coefficients, basis_indices, num_elements, num_basis
    )
end

function build_local_extraction(
    geometry::Geometry.HierarchicalGeometry,
    basis::Hierarchical.Hierarchy,
    ::Type{B},
    ::Val{1}, # only one component
    element_id,
) where {B <: BasisType}
    level, level_id = Geometry.convert_to_level_and_level_id(geometry, element_id)
    # Construct extraction matrix with only active basis function from element level
    spaces = Hierarchical.get_sets(basis)
    child_space = spaces[level]
    child_indices = get_basis_indices(child_space, level_id)
    ancestor_level, ancestors = _get_coarsest_level_with_ancestors(
        geometry, basis, element_id
    )
    # No coarser basis functions are supported on the element
    if ancestor_level == level
        child_active_ids = get_active_indices(basis, level, level_id)
        # Drop inactive basis functions
        basis_indices = map(
            s -> Hierarchical.convert_to_hier_id(basis, level, s),
            view(child_indices, child_active_ids),
        )
        extraction_coefficients = get_extraction_coefficients(child_space, level_id)
        # Avoid slicing an full matrix
        if all(child_active_ids)
            E = (extraction_coefficients,)
        else
            E = (extraction_coefficients[:, child_active_ids],)
        end

        I = Indices(basis_indices, (1:length(basis_indices),))

        return E, I
    end

    #After this point, there _are_ ancestor basis functions supported on the element

    #= 
    We get all ancestors active indices to figure out the full size of
    extraction_coefficients and basis_indices
    =#
    num_ancestors = level - ancestor_level
    ancestor_active_ids = Vector{BitVector}(undef, num_ancestors)
    cum_num_active = Vector{Int}(undef, num_ancestors + 2)
    cum_num_active[1] = 0
    child_active_ids = get_active_indices(basis, level, level_id)
    cum_num_active[2] = sum(child_active_ids)
    #=
    Refinement happens in reverse order, so we reverse the ancestors, starting from the
    first non-trivial ancestor level.
    =#
    ancestor_iterator = Iterators.reverse(@view ancestors[ancestor_level:end])
    for (i, e) in enumerate(ancestor_iterator)
        # The ancestor count decrements the finest level
        l = level - i
        ancestor_active_ids[i] = get_active_indices(basis, l, e)
        cum_num_active[i + 2] = cum_num_active[i + 1] + sum(ancestor_active_ids[i])
    end

    num_active = last(cum_num_active)
    E = get_extraction_coefficients(child_space, level_id)
    extraction_coefficients = Matrix{Float64}(undef, size(E, 1), num_active)
    basis_indices = Vector{Int}(undef, num_active)
    # Copy the child level information
    ids_range = (cum_num_active[1] + 1):cum_num_active[2]
    @views extraction_coefficients[:, ids_range] .= E[:, child_active_ids]
    for (i, b) in enumerate(view(child_indices, child_active_ids))
        basis_indices[i] = Hierarchical.convert_to_hier_id(basis, level, b)
    end

    # Initialise the refinement matrix as an identity matrix
    R = Matrix(LinearAlgebra.I, length(child_indices), length(child_indices))
    scalings = Hierarchical.get_scalings(basis)
    for (i, e) in enumerate(ancestor_iterator)
        l = level - i
        scal = scalings[l]
        # Retrieve local scaling matrix
        parent_indices = get_basis_indices(Hierarchical.get_parent(scal), e)
        curr_R = Hierarchical.view_scaling_matrix(scal, child_indices, parent_indices)
        # Modify curr_R based on basis type
        parent_active_ids = get_active_indices(basis, l, e)
        curr_R = _mod_refinement!(curr_R, B, child_active_ids)
        # Propagate previous refinements
        R *= curr_R
        # Add active indices and coefficients
        ids_range = (cum_num_active[i + 1] + 1):cum_num_active[i + 2]
        @views LinearAlgebra.mul!(
            extraction_coefficients[:, ids_range], E, R[:, parent_active_ids]
        )
        for (j, b) in enumerate(view(parent_indices, parent_active_ids))
            basis_indices[cum_num_active[i + 1] + j] = Hierarchical.convert_to_hier_id(
                basis, l, b
            )
        end

        # We are iterating in reverse, so next children are the current parents
        child_indices = parent_indices
        child_active_ids = parent_active_ids
    end

    # Modify coefficients based on basis type
    extraction_coefficients, basis_indices = _mod_coeffs_and_ids!(
        extraction_coefficients, basis_indices, B
    )

    return (extraction_coefficients,), Indices(basis_indices, (1:length(basis_indices),))
end

function build_local_extraction(
    geometry::Geometry.HierarchicalGeometry,
    basis::Hierarchical.Hierarchy,
    ::Type{B},
    ::Val{num_components},
    element_id,
) where {num_components, B <: BasisType}
    # TODO: Generalise to multiple components!
    return error("todo!")
end


############################################################################################
#                                        BasisTypes                                        #
############################################################################################

abstract type BasisType end

"""
    HB

Abstract basis type for the (untruncated) hierarchical B-spline representation.
"""
abstract type HB <: BasisType end

"""
    THB

Abstract basis type for the truncated hierarchical B-spline representation.
"""
abstract type THB <: BasisType end

"""
    _mod_refinement!(R::AbstractMatrix, ::Type{HB}, active_indices::BitVector)

Return `R` unchanged for the untruncated hierarchical basis.
"""
_mod_refinement!(R::AbstractMatrix, ::Type{HB}, ::BitVector) = R

"""
    _mod_coeffs_and_ids!(T::AbstractMatrix, I::AbstractVector, ::Type{HB})

Return the multilevel refinement matrix `T` and hierarchical basis ids `I` unchanged for the
untruncated hierarchical basis.
"""
_mod_coeffs_and_ids!(T::AbstractMatrix, I::AbstractVector, ::Type{HB}) = T, I

"""
    _mod_refinement!(R::AbstractMatrix, ::Type{THB}, active_indices::BitVector)

Zero the rows of `R` corresponding to active child basis functions before propagating
refinement with truncation.
"""
function _mod_refinement!(R::AbstractMatrix, ::Type{THB}, active_indices::BitVector)
    A = LinearAlgebra.Diagonal(active_indices)
    # Truncate R
    R = (LinearAlgebra.I - A) * R

    return R
end

"""
    _mod_coeffs_and_ids!(T::AbstractMatrix, I::AbstractVector, ::Type{THB})

Remove the columns of the multilevel refinement matrix `T` of fully truncated basis
functions, together with their hierarchical ids in `I`.
"""
function _mod_coeffs_and_ids!(T::AbstractMatrix, I::AbstractVector, ::Type{THB})
    keep = trues(size(T, 2))
    @inbounds for b in axes(T, 2)
        # NOTE: An exact comparison is intended. Truncation zeroes the rows of active
        # children exactly, so fully truncated basis functions have exactly zero
        # coefficients.
        keep[b] = !all(iszero, @view T[:, b])
    end

    return T[:, keep], I[keep]
end

############################################################################################
#                                       Coefficients                                       #
############################################################################################

"""
    get_coarsest_level(geometry, basis, element_id::Int)

Return the coarsest level with an active basis function supported on `element_id`, or the
element's own level if no coarser active basis function supports it.
"""
function get_coarsest_level(geometry, basis, element_id::Int)
    # We start by gathering all the ancestors from level 1 up to (and not including) the
    # element level
    ancestors = Geometry.get_ancestors(geometry, element_id)
    for (level, id) in enumerate(ancestors)
        space = Hierarchical.get_set(basis, level)
        # Get all basis indices supported on the ancestor
        local_indices = get_basis_indices(space, id)
        l_set = Hierarchical.get_level_set(basis, level)
        # If at least one basis function is active, we return its level.
        if !isdisjoint(local_indices, l_set)
            return level
        end
    end

    # No active basis functions from coarser levels are supported on the element, so we
    # return its level, which must be one higher than the number of ancestors.
    return length(ancestors) + 1
end

"""
    _get_coarsest_level_with_ancestors(geometry, basis, element_id)

Return the coarsest level with an active basis function supported on `element_id`, together
with the element's ancestors.

See [`get_coarsest_level`](@ref).
"""
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

"""
    get_active_indices(
        geometry::Geometry.HierarchicalGeometry,
        basis::Hierarchical.Hierarchy,
        element_id::Int,
    )

Return a bit vector marking which level-local basis functions supported on `element_id` are
active.
"""
function get_active_indices(
    geometry::Geometry.HierarchicalGeometry, basis::Hierarchical.Hierarchy, element_id::Int
)
    level, level_id = Geometry.convert_to_level_and_level_id(geometry, element_id)

    return get_active_indices(basis, level, level_id)
end

"""
    get_active_indices(basis::Hierarchical.Hierarchy, level::Int, level_id::Int)

Return a bit vector marking which basis functions supported on the element indexed by
`level` and `level_id` are active in `basis`.
"""
function get_active_indices(basis::Hierarchical.Hierarchy, level::Int, level_id::Int)
    space = Hierarchical.get_set(basis, level)

    return _get_active_indices(basis, level, get_basis_indices(space, level_id))
end

"""
    get_active_indices(
        basis::Hierarchical.Hierarchy, level::Int, level_id::Int, component_id::Int
    )

Return a bit vector marking which basis functions whose `component_id`-th component is
non-zero on the element indexed by `level` and `level_id` are active in `basis`.
"""
function get_active_indices(
    basis::Hierarchical.Hierarchy, level::Int, level_id::Int, component_id::Int
)
    space = Hierarchical.get_set(basis, level)

    return _get_active_indices(
        basis, level, get_basis_indices(space, level_id, component_id)
    )
end

function _get_active_indices(
    basis::Hierarchical.Hierarchy, level::Int, local_indices::AbstractVector{Int}
)
    # NOTE: The `local_indices` may refer to the multi- or component-wise indices.
    active_set = Hierarchical.get_level_set(basis, level)
    active_indices = BitVector(undef, length(local_indices))
    for (i, b) in enumerate(local_indices)
        active_indices[i] = b in active_set
    end

    return active_indices
end

############################################################################################
#                                        Extraction                                        #
############################################################################################

"""
    build_extraction_operator(
        geometry, basis::Hierarchical.Hierarchy, ::Type{B}
    ) where {B <: BasisType}

Build the element-wise extraction operator for `basis`, applying the truncation rules
selected by the basis type `B`.

The extraction operator has the block pattern of the level-wise spaces, which are assumed to
share it. On an element of level `L`, each block `(ci, co)` is the corresponding block of
the level-`L` space multiplied by the multilevel refinement matrix of output component `co`.
See [`_build_multilevel_refinement`](@ref).
"""
function build_extraction_operator(
    geometry, basis::Hierarchical.Hierarchy, ::Type{B}
) where {B <: BasisType}
    spaces = Hierarchical.get_sets(basis)
    first_space = first(spaces)
    block_pattern = get_block_pattern(first_space)
    num_elements = Geometry.get_num_elements(geometry)
    num_basis = Hierarchical.get_num_objects(basis)
    num_components = get_num_components(first_space)
    extraction_coefficients = Vector{NTuple{length(block_pattern), Matrix{Float64}}}(
        undef, num_elements
    )
    basis_indices = Vector{Indices{num_components, Vector{Int}}}(undef, num_elements)
    Threads.@threads for element_id in eachindex(extraction_coefficients, basis_indices)
        E, I = build_local_extraction(geometry, basis, B, block_pattern, element_id)
        extraction_coefficients[element_id] = E
        basis_indices[element_id] = I
    end

    return ExtractionOperator{get_num_input_components(first_space)}(
        block_pattern, extraction_coefficients, basis_indices, num_elements, num_basis
    )
end

"""
    build_local_extraction(
        geometry::Geometry.HierarchicalGeometry,
        basis::Hierarchical.Hierarchy,
        ::Type{B},
        block_pattern::NTuple{num_blocks, Tuple{Int, Int}},
        element_id::Int,
    ) where {B <: BasisType, num_blocks}

Build the extraction coefficient blocks, in the order of `block_pattern`, and the
hierarchical basis indices on element `element_id`.

The rows of each block `(ci, co)` are the local basis functions of input component `ci` of
the element's level space, and its columns are the active (and, for `THB`, not fully
truncated) hierarchical basis functions whose `co`-th component is non-zero on the element.
"""
function build_local_extraction(
    geometry::Geometry.HierarchicalGeometry,
    basis::Hierarchical.Hierarchy,
    ::Type{B},
    block_pattern::NTuple{num_blocks, Tuple{Int, Int}},
    element_id::Int,
) where {B <: BasisType, num_blocks}
    level, level_id = Geometry.convert_to_level_and_level_id(geometry, element_id)
    space = Hierarchical.get_set(basis, level)
    level_blocks = get_extraction_blocks(space, level_id)
    ancestor_level, ancestors = _get_coarsest_level_with_ancestors(
        geometry, basis, element_id
    )
    num_components = get_num_components(space)
    if ancestor_level == level
        # No coarser basis functions are supported on the element, so we only need to drop
        # the inactive basis functions of the element's level.
        active_ids = ntuple(
            co -> get_active_indices(basis, level, level_id, co), num_components
        )
        component_indices = ntuple(num_components) do co
            local_indices = view(get_basis_indices(space, level_id, co), active_ids[co])

            return map(b -> Hierarchical.convert_to_hier_id(basis, level, b), local_indices)
        end
        blocks = map(block_pattern, level_blocks) do (_, co), E
            return _select_active_columns(E, active_ids[co])
        end
    else
        refinements = ntuple(num_components) do co
            return _build_multilevel_refinement(
                basis, B, level, level_id, ancestor_level, ancestors, co
            )
        end
        component_indices = map(last, refinements)
        blocks = map(block_pattern, level_blocks) do (_, co), E
            return convert(Matrix{Float64}, E * first(refinements[co]))
        end
    end

    return blocks, Indices(union(component_indices...), component_indices)
end

"""
    _select_active_columns(E, active_ids::BitVector)

Return the columns of the level extraction block `E` marked by `active_ids` as a
`Matrix{Float64}`. `E` may be a `LinearAlgebra.UniformScaling`, in which case it is a square
block of size `length(active_ids)`.
"""
function _select_active_columns(E, active_ids::BitVector)
    E = convert(Matrix{Float64}, _as_extraction_matrix(E, length(active_ids)))
    # Avoid slicing a full matrix.
    all(active_ids) && return E

    return E[:, active_ids]
end

"""
    _build_multilevel_refinement(
        basis::Hierarchical.Hierarchy,
        ::Type{B},
        level::Int,
        level_id::Int,
        ancestor_level::Int,
        ancestors,
        component_id::Int,
    ) where {B <: BasisType}

Build the multilevel refinement matrix `T` of output component `component_id` on the element
indexed by `level` and `level_id`, together with the hierarchical ids of its columns.

The rows of `T` are the level-`level` basis functions whose `component_id`-th component is
non-zero on the element. Its columns express the active basis functions of all levels from
`ancestor_level` to `level` in terms of those rows: first the active functions of `level`
(identity columns), followed by the active functions of each coarser level, obtained by
chaining the (possibly truncated) scaling matrices. Since the `component_id`-th component of
a parent is the scaled sum of the `component_id`-th components of its children, only the
children with a non-zero `component_id`-th component on the element are needed.
"""
function _build_multilevel_refinement(
    basis::Hierarchical.Hierarchy,
    ::Type{B},
    level::Int,
    level_id::Int,
    ancestor_level::Int,
    ancestors,
    component_id::Int,
) where {B <: BasisType}
    child_indices = get_basis_indices(
        Hierarchical.get_set(basis, level), level_id, component_id
    )
    child_active_ids = get_active_indices(basis, level, level_id, component_id)
    # Refinement happens in reverse order over the levels, so we reverse the ancestors.
    # As such, the first entry corresponds to `level-1`.
    ancestor_iterator = Iterators.reverse(@view ancestors[ancestor_level:end])
    # We get all ancestors active indices first to know the full size of `T` and `ids`,
    # instead of enlarging them once per level.
    num_ancestors = level - ancestor_level
    ancestor_active_ids = Vector{BitVector}(undef, num_ancestors)
    cum_num_active = Vector{Int}(undef, num_ancestors + 2)
    cum_num_active[1] = 0
    cum_num_active[2] = count(child_active_ids)
    for (i, e) in enumerate(ancestor_iterator)
        ancestor_active_ids[i] = get_active_indices(basis, level - i, e, component_id)
        cum_num_active[i + 2] = cum_num_active[i + 1] + count(ancestor_active_ids[i])
    end

    num_rows = length(child_indices)
    T = zeros(Float64, num_rows, last(cum_num_active))
    ids = Vector{Int}(undef, last(cum_num_active))
    for (j, i) in enumerate(findall(child_active_ids))
        T[i, j] = 1.0
        ids[j] = Hierarchical.convert_to_hier_id(basis, level, child_indices[i])
    end

    R = Matrix{Float64}(LinearAlgebra.I, num_rows, num_rows)
    scalings = Hierarchical.get_scalings(basis)
    for (i, e) in enumerate(ancestor_iterator)
        l = level - i
        scal = scalings[l]
        parent_indices = get_basis_indices(Hierarchical.get_parent(scal), e, component_id)
        curr_R = Hierarchical.view_scaling_matrix(scal, child_indices, parent_indices)
        curr_R = _mod_refinement!(curr_R, B, child_active_ids)
        # Propagate previous refinements
        R *= curr_R
        parent_active_ids = ancestor_active_ids[i]
        ids_range = (cum_num_active[i + 1] + 1):cum_num_active[i + 2]
        @views T[:, ids_range] .= R[:, parent_active_ids]
        for (j, b) in enumerate(view(parent_indices, parent_active_ids))
            ids[cum_num_active[i + 1] + j] = Hierarchical.convert_to_hier_id(basis, l, b)
        end

        # We are iterating in reverse, so the next children are the current parents
        child_indices, child_active_ids = parent_indices, parent_active_ids
    end

    return _mod_coeffs_and_ids!(T, ids, B)
end

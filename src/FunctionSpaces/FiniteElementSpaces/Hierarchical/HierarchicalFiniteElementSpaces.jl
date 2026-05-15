############################################################################################
#                                     Structure                                            #
############################################################################################

"""
	HierarchicalFiniteElementSpace{
		manifold_dim, num_components, num_patches, S, T, G, GP
	} <: AbstractFESpace{manifold_dim, num_components, num_patches}

A hierarchical space that is built from a nested hierarchy of `manifold_dim`-variate
function `spaces` and `nested_domains`. At each level of the hierarchy, a certain set of
elements and basis functions are deemed as active, depending on the `nested_domains`, while
the rest is inactive.

See [Giannelli2013](@cite) for more information.

# Fields
- `geometry::G`: The hierarchical geometry associated with the hierarchical space. Will be a
    subtype of [`Geometry.HierarchicalGeometry`](@ref).
- `parametric_geometry::GP`: The parametic hierarchical geometry associated with the
    hierarchy of parametic geometries from each level. Will be a subtype of
    [`Geometry.HierarchicalGeometry`](@ref).
- `spaces::Vector{S} `: Collection of `L` `manifold_dim`-variate function spaces, where `L`
    is the total number of levels.
- `two_scale_operators::Vector{T}`: Collection of `L-1` two-scale operators relating each
    consecutive pair of finite element spaces, where `L` is the total number of levels. See
    [`AbstractTwoScaleOperator`](@ref).
- `active_elements::Hierarchy.ActiveInfo`: Information about the active elements at each
    level. See [`Hierarchy.ActiveInfo`](@ref).
- `active_basis::Hierarchy.ActiveInfo`: Information about the active basis at each level.
    See [`Hierarchy.ActiveInfo`](@ref).
- `nested_domains::Hierarchy.ActiveInfo`: Information about the nested domains at each
    level. This is the usual definition of Ωₗ in the literature. See
    [`Hierarchy.ActiveInfo`](@ref).
- `multilevel_elements::SparseArrays.SparseVector{Int, Int}`: Elements where basis from
    multiple levels have non-empty support.
- `multilevel_extraction_coeffs::Vector{NTuple{num_components, Matrix{Float64}}}`:
    Extraction coefficients of active basis functions in `multilevel_elements`.
- `multilevel_basis_indices::Vector{Vector{Int}}`: Indices of active basis in
    `multilevel_elements`, in hierarchical indexing.
- `num_subdivisions::NTuple{manifold_dim, Int}`: Number of subdivisions per `manifold_dim`,
    per level for the hierarchical mesh.
- `truncated::Bool`: Flag for truncated hierarchical spaces.
- `simplified::Bool`: Flag for simplified hierarchical spaces.
- `dof_partition::Vector{Vector{Vector{Int}}}`: The degree-of-freedom partitioning of the
    hierarchical space, in hierarchical indexing.
"""
struct HierarchicalFiniteElementSpace{
    manifold_dim, num_components, num_patches, S, T, G, GP
} <: AbstractFESpace{manifold_dim, num_components, num_patches}
    geometry::G
    parametric_geometry::GP
    spaces::Vector{S}
    two_scale_operators::Vector{T}
    active_elements::Hierarchy.ActiveInfo
    active_basis::Hierarchy.ActiveInfo
    nested_domains::Hierarchy.ActiveInfo
    multilevel_elements::SparseArrays.SparseVector{Int, Int}
    multilevel_extraction_coeffs::Vector{NTuple{num_components, Matrix{Float64}}}
    multilevel_basis_indices::Vector{Vector{Int}}
    num_subdivisions::NTuple{manifold_dim, Int}
    truncated::Bool
    simplified::Bool
    dof_partition::Vector{Vector{Vector{Int}}}

    function HierarchicalFiniteElementSpace(
        geometry::G,
        parametric_geometry::GP,
        spaces::Vector{S},
        two_scale_operators::Vector{T},
        active_elements,
        active_basis,
        nested_domains,
        multilevel_elements,
        multilevel_extraction_coeffs,
        multilevel_basis_indices,
        num_subdivisions::NTuple{manifold_dim},
        truncated::Bool,
        simplified::Bool,
        dof_partition,
    ) where {
        manifold_dim,
        num_components,
        num_patches,
        G <: Geometry.AbstractGeometry,
        GP <: Geometry.AbstractGeometry,
        S <: AbstractFESpace{manifold_dim, num_components, num_patches},
        T <: AbstractTwoScaleOperator,
    }
        num_levels = length(spaces)
        # Checks for incompatible arguments
        if num_levels < 1
            throw(ArgumentError("At least 1 level is required, but 0 were given."))
        elseif length(two_scale_operators) != num_levels - 1
            msg1 = "Number of two-scale operators should be `num_levels - 1`. "
            msg2 = "Got $(length(two_scale_operators)) and $(num_levels - 1)."
            throw(ArgumentError(msg1 * msg2))
        elseif Hierarchy.get_num_levels(nested_domains) != num_levels
            msg1 = "Number of nested domains should be the same as the number of levels. "
            msg2 = "Got $(num_levels) and $(Hierarchy.get_num_levels(nested_domains))."
            throw(ArgumentError(msg1 * msg2))
        end

        for level in 1:(num_levels - 1)
            if !(spaces[level] === get_parent_space(two_scale_operators[level]))
                throw(
                    ArgumentError(
                        "Space at level $(level) is different from the corresponding " *
                        "space in the two-scale operator.",
                    ),
                )
            end
        end

        if !(spaces[num_levels] === get_child_space(two_scale_operators[num_levels - 1]))
            throw(
                ArgumentError(
                    "Space at level $(num_levels) is different from the corresponding " *
                    "space in the two-scale operator.",
                ),
            )
        end

        return new{manifold_dim, num_components, num_patches, S, T, G, GP}(
            geometry,
            parametric_geometry,
            spaces,
            two_scale_operators,
            active_elements,
            active_basis,
            nested_domains,
            multilevel_elements,
            multilevel_extraction_coeffs,
            multilevel_basis_indices,
            num_subdivisions,
            truncated,
            simplified,
            dof_partition,
        )
    end
end

"""
	HierarchicalFiniteElementSpace(
		spaces::Vector{S},
		two_scale_operators::Vector{T},
		domains::Hierarchy.ActiveInfo,
		num_subdivisions::NTuple{manifold_dim, Int},
		truncated::Bool=true,
		simplified::Bool=false,
	) where {
		manifold_dim,
		num_components,
		num_patches,
		S <: AbstractFESpace{manifold_dim, num_components, num_patches},
		T <: AbstractTwoScaleOperator,
	}

Constructor that generates multilevel information.
"""
function HierarchicalFiniteElementSpace(
    spaces::Vector{S},
    two_scale_operators::Vector{T},
    domains::Hierarchy.ActiveInfo,
    num_subdivisions::NTuple{manifold_dim, Int},
    truncated::Bool=true,
    simplified::Bool=false;
) where {
    manifold_dim,
    num_components,
    num_patches,
    S <: AbstractFESpace{manifold_dim, num_components, num_patches},
    T <: AbstractTwoScaleOperator,
}
    num_levels = length(spaces)
    # Computes necessary hierarchical information
    active_elements, active_basis, nested_domains = get_active_objects_and_nested_domains(
        spaces, two_scale_operators, domains, simplified
    )
    multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_indices = get_multilevel_extraction(
        spaces, two_scale_operators, active_elements, active_basis, truncated
    )
    dof_partition = compute_dof_partition(spaces, active_basis, num_levels)
    parametric_geometries = ntuple(l -> get_parametric_geometry(spaces[l]), num_levels)
    geometries = ntuple(l -> get_geometry(spaces[l]), num_levels)
    parametric_geometry = Geometry.HierarchicalGeometry(
        parametric_geometries, active_elements
    )
    geometry = Geometry.HierarchicalGeometry(geometries, active_elements)

    return HierarchicalFiniteElementSpace(
        geometry,
        parametric_geometry,
        spaces,
        two_scale_operators,
        active_elements,
        active_basis,
        nested_domains,
        multilevel_elements,
        multilevel_extraction_coeffs,
        multilevel_basis_indices,
        num_subdivisions,
        truncated,
        simplified,
        dof_partition,
    )
end

"""
	HierarchicalFiniteElementSpace(
		spaces::Vector{S},
		two_scale_operators::Vector{T},
		domains_per_level::Vector{Vector{Int}},
		num_subdivisions::NTuple{manifold_dim, Int},
		truncated::Bool=true,
		simplified::Bool=false,
	) where {
		manifold_dim,
		num_components,
		num_patches,
		S <: AbstractFESpace{manifold_dim, num_components, num_patches},
		T <: AbstractTwoScaleOperator,
	}

Constructor for domains given in a per-level vector.
"""
function HierarchicalFiniteElementSpace(
    spaces::Vector{S},
    two_scale_operators::Vector{T},
    domains_per_level::Vector{Vector{Int}},
    num_subdivisions::NTuple{manifold_dim, Int},
    truncated::Bool=true,
    simplified::Bool=false,
) where {
    manifold_dim,
    num_components,
    num_patches,
    S <: AbstractFESpace{manifold_dim, num_components, num_patches},
    T <: AbstractTwoScaleOperator,
}
    domains = Hierarchy.ActiveInfo(domains_per_level)

    return HierarchicalFiniteElementSpace(
        spaces, two_scale_operators, domains, num_subdivisions, truncated, simplified
    )
end

"""
	HierarchicalFiniteElementSpace(
		space::S,
		num_subdivisions::NTuple{manifold_dim, Int},
		truncated::Bool=true,
		simplified::Bool=false,
	) where {manifold_dim, S <: AbstractFESpace{manifold_dim}}

Constructor for a Hierarchical space with no refinement. This is useful for initializating a
hierarchical space that will later be refined.
"""
function HierarchicalFiniteElementSpace(
    space::S,
    num_subdivisions::NTuple{manifold_dim, Int},
    truncated::Bool=true,
    simplified::Bool=false,
) where {manifold_dim, S <: AbstractFESpace{manifold_dim}}
    TS, ref_space = build_two_scale_operator(space, num_subdivisions)
    domains = [collect(1:get_num_elements(space)), Int[]]

    return HierarchicalFiniteElementSpace(
        [space, ref_space], [TS], domains, num_subdivisions, truncated, simplified
    )
end

############################################################################################
#                              Structure initialization                                    #
############################################################################################

"""
    get_active_objects_and_nested_domains(
		spaces::Vector{S},
		two_scale_operators::Vector{T},
		domains::Hierarchy.ActiveInfo,
		simplified::Bool,
	) where {S <: AbstractFESpace, T <: AbstractTwoScaleOperator}

Computes the active elements and basis on each level based on `spaces`,
`two_scale_operators` and the set of nested `domains`.

The construction loops over the `domains` on each level, deactivates basis functions fully
supported on the next level's domain, and then selects the active basis in the next level as
either the basis functions supported on the domain of the next level, or the children of
deactivated basis, based on wheter the space is `simplified` or not. A similar logic is
applied to determine the active elements.

# Arguments
- `spaces::Vector{AbstractFESpace{manifold_dim, num_components, num_patches}}`: Finite
	element spaces at each level.
- `two_scale_operators::Vector{AbstractTwoScaleOperator}`: Two scale operators relating the
    finite element spaces at each level.
- `domains::Hierarchy.ActiveInfo`: Nested domains where the support of active basis is
    determined.

# Returns
- `active_elements::Hierarchy.ActiveInfo`: Active elements at each level.
- `active_basis::Hierarchy.ActiveInfo`: Active basis at each level.
- `nested_domains::Hierarchy.ActiveInfo`: Information about the nested domains at each
    level. This is the usual definition of Ωₗ in the literature. See
    [`Hierarchy.ActiveInfo`](@ref).
"""
function get_active_objects_and_nested_domains(
    spaces::Vector{S},
    two_scale_operators::Vector{T},
    domains::Hierarchy.ActiveInfo,
    simplified::Bool,
) where {S <: AbstractFESpace, T <: AbstractTwoScaleOperator}
    num_levels = Hierarchy.get_num_levels(domains)
    active_elements_per_level = [collect(1:get_num_elements(spaces[1]))]
    active_basis_per_level = [collect(1:get_num_basis(spaces[1]))]
    nested_domains_per_level = [collect(1:get_num_elements(spaces[1]))]
    # If the hierarchical space is not simplified, we need to ensure that the
    # refinement domains contain proper subsets of a given element's
    # children as refined.
    if !simplified
        new_domains = [Hierarchy.get_level_ids(domains, level) for level in 1:num_levels]
        for level in num_levels:-1:2
            if isempty(new_domains[level])
                continue
            end

            parents = mapreduce(
                child -> get_element_parent(two_scale_operators[level - 1], child),
                union,
                new_domains[level],
            )
            children = mapreduce(
                parent -> get_element_children(two_scale_operators[level - 1], parent),
                vcat,
                parents,
            )
            new_domains[level] = children
            union!(new_domains[level - 1], parents)
        end

        domains = Hierarchy.ActiveInfo(new_domains)
    end

    for level in 1:(num_levels - 1)
        next_level_domain = Set(Hierarchy.get_level_ids(domains, level + 1))
        elements_to_remove = Int[]
        elements_to_add = Int[]
        basis_to_remove = Int[]
        basis_to_add = Int[]
        if !simplified
            for parent_basis in active_basis_per_level[level]
                # Gets the support of Ni on current level and the next one
                support = get_support(spaces[level], parent_basis)
                support_children = [
                    child for parent in support for
                    child in get_element_children(two_scale_operators[level], parent)
                ]
                # Updates elements and basis to add and remove
                if issubset(support_children, next_level_domain)
                    append!(elements_to_remove, support)
                    append!(basis_to_remove, parent_basis)
                    basis_children = get_basis_children(
                        two_scale_operators[level], parent_basis
                    )
                    append!(basis_to_add, basis_children)
                    append!(
                        elements_to_add,
                        mapreduce(
                            child -> get_support(spaces[level + 1], child),
                            union,
                            basis_children,
                        ),
                    )
                end
            end

            for child_basis in setdiff(1:get_num_basis(spaces[level + 1]), basis_to_add)
                support = get_support(spaces[level + 1], child_basis)
                if issubset(support, next_level_domain)
                    parents = mapreduce(
                        child -> get_element_parent(two_scale_operators[level], child),
                        union,
                        support,
                    )
                    append!(elements_to_remove, parents)
                    append!(elements_to_add, support)
                    append!(basis_to_add, child_basis)
                end
            end
        else
            for parent_basis in active_basis_per_level[level]
                # Gets the support of Ni on current level and the next one
                support = get_support(spaces[level], parent_basis)
                support_children = [
                    child for parent in support for
                    child in get_element_children(two_scale_operators[level], parent)
                ]
                # Updates elements and basis to add and remove
                if issubset(support_children, next_level_domain)
                    append!(elements_to_remove, support)
                    append!(basis_to_remove, parent_basis)
                    append!(elements_to_add, support_children)
                    append!(
                        basis_to_add,
                        get_basis_children(two_scale_operators[level], parent_basis),
                    )
                end
            end
        end

        # Remove inactive elements and basis on current level
        setdiff!(active_elements_per_level[level], elements_to_remove)
        setdiff!(active_basis_per_level[level], basis_to_remove)
        # Add active elements and basis on next level
        unique!(elements_to_add)
        unique!(basis_to_add)
        push!(active_elements_per_level, elements_to_add)
        push!(active_basis_per_level, basis_to_add)
        # Store nested domains Ωₗ
        push!(nested_domains_per_level, copy(elements_to_add))
    end

    map(elements -> sort!(elements), active_elements_per_level)
    map(basis -> sort!(basis), active_basis_per_level)
    active_elements = Hierarchy.ActiveInfo(active_elements_per_level)
    active_basis = Hierarchy.ActiveInfo(active_basis_per_level)
    nested_domains = Hierarchy.ActiveInfo(nested_domains_per_level)

    return active_elements, active_basis, nested_domains
end

"""
    get_multilevel_extraction(
        spaces::Vector{S},
        two_scale_operators::Vector{T},
        active_elements::Hierarchy.ActiveInfo,
        active_basis::Hierarchy.ActiveInfo,
        truncated::Bool,
    ) where {manifold_dim, num_components, num_patches, S <: AbstractFESpace{manifold_dim, num_components, num_patches}, T <: AbstractTwoScaleOperator}

Computes which elements are multilevel elements, i.e. elements for which basis from
multiple levels have non-emtpy support, as well as their extraction coefficients matrices
and active basis indices.

The extraction coefficients depend on whether the hierarchical space is `truncated` or not.

# Arguments
- `spaces::Vector{AbstractFESpace{manifold_dim, num_components, num_patches}}`: finite element spaces at each level.
- `two_scale_operators::Vector{AbstractTwoScaleOperator}`: two scale operators relating the
    finite element spaces at each level.
- `active_elements::Hierarchy.ActiveInfo`: active elements on each level.
- `active_basis::Hierarchy.ActiveInfo`: active basis on each level.
- `truncated`: flag for a truncated hierarchical space.

# Returns
- `multilevel_elements::SparseArrays.SparseVector{Int, Int}`: elements where basis from
    multiple levels have non-empty support.
- `multilevel_extraction_coeffs::Vector{Matrix{Float64}}`: extraction coefficients of
    active basis in `multilevel_elements`.
- `multilevel_basis_indices::Vector{Vector{Int}}`: indices of active basis in
    `multilevel_elements`.
"""
function get_multilevel_extraction(
    spaces::Vector{S},
    two_scale_operators::Vector{T},
    active_elements::Hierarchy.ActiveInfo,
    active_basis::Hierarchy.ActiveInfo,
    truncated::Bool,
) where {
    manifold_dim,
    num_components,
    S <: AbstractFESpace{manifold_dim, num_components},
    T <: AbstractTwoScaleOperator,
}
    # First, we determines which elements contain basis functions from multiple levels.
    multilevel_information = get_multilevel_information(
        spaces, two_scale_operators, active_elements, active_basis
    )
    multilevel_keys = keys(multilevel_information)
    num_multilevel_elements = length(multilevel_keys)
    # Skip trivial case
    if num_multilevel_elements == 0
        return SparseArrays.spzeros(Int, Hierarchy.get_num_objects(active_elements)),
        NTuple{num_components, Matrix{Float64}}[],
        [Int[]]
    end

    # Next, we construct the extraction coefficients for the multilevel elements.
    multilevel_element_indices = Vector{Int}(undef, num_multilevel_elements)
    multilevel_extraction_coeffs = Vector{NTuple{num_components, Matrix{Float64}}}(
        undef, num_multilevel_elements
    )
    multilevel_basis_ids = Vector{Vector{Int}}(undef, num_multilevel_elements)
    ml_id_count = 1
    # TODO: Change storage to go over all the elements of a given level at the same time
    for (level, element_level_id) in multilevel_keys
        # Create multilevel element specific extraction coefficients per component
        basis_level_ids = get_basis_indices(spaces[level], element_level_id)
        # Subset of an identity matrix with size (num_basis, num_active_basis)
        # where num_basis is the number of basis functions supported at the
        # element on the original level l space
        active_basis_matrix, active_local_ids = get_active_basis_matrix(
            spaces[level], element_level_id, level, active_basis
        )
        # Multi-component refinement matrix
        refinement_matrix, multilevel_basis_hier_ids = get_refinement_data(
            active_basis_matrix,
            active_local_ids,
            spaces,
            two_scale_operators,
            active_basis,
            element_level_id,
            level,
            multilevel_information,
            truncated,
        )
        # Convert and store multi-level basis ids
        basis_hier_ids = map(
            basis_level_id ->
                Hierarchy.convert_to_hier_id(active_basis, level, basis_level_id),
            basis_level_ids[active_local_ids],
        )
        multilevel_basis_ids[ml_id_count] = append!(
            basis_hier_ids, multilevel_basis_hier_ids
        )
        # Store component-wise extraction coefficients from the refinement matrix
        multilevel_extraction_coeffs[ml_id_count] = ntuple(num_components) do component_id
            level_coeffs, J = get_extraction(spaces[level], element_level_id, component_id)
            #TODO: This should be optimized to also include child permutations
            return level_coeffs * view(refinement_matrix, J, :)
        end

        # Add multilevel element specific index
        multilevel_element_indices[ml_id_count] = Hierarchy.convert_to_hier_id(
            active_elements, level, element_level_id
        )
        ml_id_count += 1
    end

    multilevel_elements = SparseArrays.sparsevec(
        multilevel_element_indices,
        1:num_multilevel_elements,
        Hierarchy.get_num_objects(active_elements),
    )

    return multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_ids
end

"""
    get_multilevel_information(
        spaces::Vector{S},
        two_scale_operators::Vector{T},
        active_elements::Hierarchy.ActiveInfo,
        active_basis::Hierarchy.ActiveInfo,
    ) where {manifold_dim, num_components, num_patches, S <: AbstractFESpace{manifold_dim, num_components, num_patches}, T <: AbstractTwoScaleOperator}

Computes which active elements are multilevel elements, i.e. elements where basis from
multiple levels have non-empty support, as well as which basis from parentr levels are
active on those elements.

# Arguments
- `spaces::Vector{AbstractFESpace{manifold_dim, num_components, num_patches}}`: finite element spaces at each level.
- `two_scale_operators::Vector{AbstractTwoScaleOperator}`: two scale operators relating the
    finite element spaces at each level.
- `active_elements::Hierarchy.ActiveInfo`: active elements on each level.
- `active_basis::Hierarchy.ActiveInfo`: active basis on each level.

# Returns
- `multilevel_information::Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}`: information
    about multilevel elements. The key's two indices indicate the multilevel element's
    level and id and the and the key's value is a vector of tuples where the indices are
    the basis level and id (from parentr levels), respectively.
"""
function get_multilevel_information(
    spaces::Vector{S},
    two_scale_operators::Vector{T},
    active_elements::Hierarchy.ActiveInfo,
    active_basis::Hierarchy.ActiveInfo,
) where {S <: AbstractFESpace, T <: AbstractTwoScaleOperator}
    L = Hierarchy.get_num_levels(active_elements)
    multilevel_information = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
    # Above, first tuple is level and id of ml element, second tuple
    # is level and id of ml basis in that element
    # TODO: Change the storage so that entry of a given key stores all the basis
    # for a given level.
    for level in 1:(L - 1)
        level_active_elements = Set(Hierarchy.get_level_ids(active_elements, level))
        level_active_basis = Hierarchy.get_level_ids(active_basis, level)
        for basis in level_active_basis
            support = get_support(spaces[level], basis)
            inactive_elements = [!(element ∈ level_active_elements) for element in support]
            for inactive_element in support[inactive_elements]
                active_children = get_element_active_children(
                    active_elements, level, inactive_element, two_scale_operators
                )
                for (child_level, child_id) in active_children
                    if haskey(multilevel_information, (child_level, child_id))
                        push!(
                            multilevel_information[(child_level, child_id)], (level, basis)
                        )
                    else
                        multilevel_information[(child_level, child_id)] = [(level, basis)]
                    end
                end
            end
        end
    end

    return multilevel_information
end

function get_active_basis_matrix(space, element_level_id, level, active_basis)
    full_level_indices = get_basis_indices(space, element_level_id)
    # Find which basis functions of the space are active at the element
    active_local_ids = findall(
        basis_id -> basis_id in Hierarchy.get_level_ids(active_basis, level),
        full_level_indices,
    )
    num_basis = length(full_level_indices)
    active_basis_matrix = Matrix{Float64}(LinearAlgebra.I, (num_basis, num_basis))
    active_basis_matrix = active_basis_matrix[:, active_local_ids]

    return active_basis_matrix, active_local_ids
end

function get_refinement_data(
    active_basis_matrix,
    active_local_ids,
    fe_spaces,
    two_scale_operators,
    active_basis,
    element_id,
    element_level,
    multilevel_information,
    truncated,
)
    active_basis_size = size(active_basis_matrix)
    num_multilevel_basis = length(multilevel_information[(element_level, element_id)])
    # Multi-component refinement matrix
    refinement_matrix = hcat(
        active_basis_matrix, zeros(active_basis_size[1], num_multilevel_basis)
    )
    multilevel_basis_hier_ids = Vector{Int}(undef, num_multilevel_basis)
    ml_basis_count = 1
    for (basis_level, basis_level_id) in multilevel_information[(element_level, element_id)]
        #TODO: Change this to perform the multilevel_evaluation for all the
        # basis of one level at the same time
        refinement_matrix[:, active_basis_size[2] + ml_basis_count] .= get_multilevel_basis_evaluation(
            fe_spaces,
            two_scale_operators,
            active_basis,
            basis_level,
            basis_level_id,
            element_level,
            element_id,
            truncated,
        )
        multilevel_basis_hier_ids[ml_basis_count] = Hierarchy.convert_to_hier_id(
            active_basis, basis_level, basis_level_id
        )
        ml_basis_count += 1
    end

    if truncated
        refinement_matrix = truncate_refinement_matrix!(refinement_matrix, active_local_ids)
    end

    return refinement_matrix, multilevel_basis_hier_ids
end

function get_multilevel_basis_evaluation(
    fe_spaces,
    two_scale_operators,
    active_basis,
    basis_level,
    basis_id,
    element_level,
    element_level_id,
    truncated::Bool,
)
    local_subdiv_matrix = LinearAlgebra.I
    current_child_element = element_level_id
    for level in element_level:-1:(basis_level + 1)
        current_parent_element = get_element_parent(
            two_scale_operators[level - 1], current_child_element
        )
        current_subdiv_matrix = get_local_subdiv_matrix(
            two_scale_operators[level - 1], current_parent_element, current_child_element
        )
        if truncated
            full_level_indices = get_basis_indices(fe_spaces[level], current_child_element)
            active_indices = findall(
                basis_id -> basis_id in Hierarchy.get_level_ids(active_basis, level),
                full_level_indices,
            )
            current_subdiv_matrix[active_indices, :] .= 0.0
        end

        local_subdiv_matrix *= current_subdiv_matrix
        current_child_element = current_parent_element
    end

    level_diff = element_level - basis_level
    basis_element_level_id = get_element_ancestor(
        two_scale_operators, element_level_id, element_level, level_diff
    )
    lowest_level_basis_indices = get_basis_indices(
        fe_spaces[basis_level], basis_element_level_id
    )
    basis_local_id = findfirst(local_id -> local_id == basis_id, lowest_level_basis_indices)

    return @view local_subdiv_matrix[:, basis_local_id]
end

"""
    truncate_refinement_matrix!(refinement_matrix, active_indices::Vector{Int})

Updates `refinement_matrix` by the rows of `active_indices` to zeros in lower level basis
functions.

# Arguments
- `refinement_matrix`: the refinement matrix to be updated.
- `active_indices::Vector{Int}`: element local indices of active basis functions from the
    highest refinement level.

# Returns
- `refinement_matrix`: truncated refinement matrix.
"""
function truncate_refinement_matrix!(refinement_matrix, active_indices::Vector{Int})
    active_length = length(active_indices)
    refinement_matrix[active_indices, (active_length + 1):end] .= 0.0

    return refinement_matrix
end

function compute_dof_partition(spaces, active_basis, L)
    level_partition = map(get_dof_partition, spaces)
    n_patches = length(level_partition[1])
    n_partitions = [length(level_partition[1][i]) for i in 1:n_patches]
    dof_partition = Vector{Vector{Vector{Int}}}(undef, n_patches)
    for patch in eachindex(dof_partition)
        dof_partition[patch] = Vector{Vector{Int}}(undef, n_partitions[patch])
        for level in 1:L
            level_active_basis = Set(Hierarchy.get_level_ids(active_basis, level))
            for partition in eachindex(dof_partition[patch])
                active_level_dofs = filter(
                    basis -> basis in level_active_basis,
                    level_partition[level][patch][partition],
                )
                dof_ids = map(
                    og_id -> Hierarchy.convert_to_hier_id(active_basis, level, og_id),
                    active_level_dofs,
                )
                if level == 1
                    dof_partition[patch][partition] = dof_ids
                else
                    append!(dof_partition[patch][partition], dof_ids)
                end
            end
        end
    end

    return dof_partition
end

############################################################################################
#                                        Extraction                                        #
############################################################################################

function get_local_basis(
    space::HierarchicalFiniteElementSpace{manifold_dim},
    hier_id::Int,
    xi::Points.AbstractPoints{manifold_dim},
    nderivatives::Int,
    component_id::Int=1,
) where {manifold_dim}
    element_level, element_level_id = convert_to_element_level_and_level_id(space, hier_id)

    return get_local_basis(
        get_space(space, element_level), element_level_id, xi, nderivatives, component_id
    )
end

function get_extraction(
    space::HierarchicalFiniteElementSpace, hier_id::Int, component_id::Int=1
)
    if get_multilevel_id(space, hier_id) == 0
        element_level, element_level_id = convert_to_element_level_and_level_id(
            space, hier_id
        )
        coeffs, J = get_extraction(
            get_space(space, element_level), element_level_id, component_id
        )
        basis_indices = collect(
            get_basis_indices(get_space(space, element_level), element_level_id)
        )
        basis_indices .=
            convert_to_basis_hier_id.(Ref(space), Ref(element_level), basis_indices)
    else
        element_level, element_level_id = convert_to_element_level_and_level_id(
            space, hier_id
        )
        multilevel_id = get_multilevel_id(space, hier_id)
        coeffs = space.multilevel_extraction_coeffs[multilevel_id][component_id]
        basis_indices = copy(space.multilevel_basis_indices[multilevel_id])
        J = 1:length(basis_indices)
    end

    return coeffs, J
end

function get_extraction_coefficients(
    space::HierarchicalFiniteElementSpace, hier_id::Int, component_id::Int=1
)
    if get_multilevel_id(space, hier_id) == 0
        element_level, element_level_id = convert_to_element_level_and_level_id(
            space, hier_id
        )
        coeffs = get_extraction_coefficients(
            get_space(space, element_level), element_level_id, component_id
        )
    else
        multilevel_id = get_multilevel_id(space, hier_id)
        coeffs = space.multilevel_extraction_coeffs[multilevel_id][component_id]
    end

    return coeffs
end

function get_basis_permutation(
    space::HierarchicalFiniteElementSpace, hier_id::Int, component_id::Int=1
)
    if get_multilevel_id(space, hier_id) == 0
        element_level, element_level_id = convert_to_element_level_and_level_id(
            space, hier_id
        )
        J = get_basis_permutation(
            get_space(space, element_level), element_level_id, component_id
        )
    else
        J = 1:length(get_basis_indices(space, hier_id))
    end

    return J
end

function get_basis_indices(space::HierarchicalFiniteElementSpace, element_id::Int)
    if iszero(get_multilevel_id(space, element_id))
        element_level, element_level_id = convert_to_element_level_and_level_id(
            space, element_id
        )
        basis_indices = collect(
            get_basis_indices(get_space(space, element_level), element_level_id)
        )
        map!(
            basis_id -> convert_to_basis_hier_id(space, element_level, basis_id),
            basis_indices,
            basis_indices,
        )
    else
        multilevel_id = get_multilevel_id(space, element_id)
        basis_indices = space.multilevel_basis_indices[multilevel_id]
    end

    return basis_indices
end

############################################################################################
#                                       Update Space                                       #
############################################################################################	

"""
	refine_space(space::HierarchicalFiniteElementSpace, domains::Hierarchy.ActiveInfo)

Returns a refined hierarchical space, given an original `space` and a set of hierarchically
nested `domains`.

# Arguments
- `space::HierarchicalFiniteElementSpace`: The original hierarchical space.
- `domains::Hierarchy.ActiveInfo`: Information about the nested domains at each level. This
	is the usual definition of Ωₗ in the literature. See [`Hierarchy.ActiveInfo`](@ref).

# Returns
- `refine_space::HierarchicalFiniteElementSpace`: A refined hierarchical space.
"""
function refine_space(space::HierarchicalFiniteElementSpace, domains::Hierarchy.ActiveInfo)
    L = Hierarchy.get_num_levels(domains)
    spaces = get_spaces(space)
    two_scale_operators = get_two_scale_operators(space)
    while L > length(spaces)
        new_ts, new_space = build_two_scale_operator(
            spaces[end], get_num_subdivisions(space)
        )
        push!(spaces, new_space)
        push!(two_scale_operators, new_ts)
    end

    refined_space = HierarchicalFiniteElementSpace(
        spaces,
        two_scale_operators,
        domains,
        get_num_subdivisions(space),
        is_truncated(space),
        is_simplified(space),
    )

    return refined_space
end

"""
	refine_space(
	    space::HierarchicalFiniteElementSpace, marked_elements_per_level::Vector{Vector{Int}}
	)

Returns a refined hierarchical space, given an original `space` and a set of
`marked_elements_per_level`. The elements are used for refinement at each level by
extracting their children, and consequently updating the `nested_domains` of `space`.

# Arguments
- `space::HierarchicalFiniteElementSpace`: The original hierarchical space.
- `marked_elements_per_level::Vector{Vector{Int}}`: The elements marked for refinement at
	each level, which will be used to refine the nested domains.

# Returns
- `refine_space::HierarchicalFiniteElementSpace`: A refined hierarchical space.
"""
function refine_space(
    space::HierarchicalFiniteElementSpace, marked_elements_per_level::Vector{Vector{Int}}
)
    L = get_num_levels(space)
    spaces = get_spaces(space)
    two_scale_operators = get_two_scale_operators(space)
    domains = get_nested_domains(space)
    if !isempty(marked_elements_per_level[L])
        new_ts, new_space = build_two_scale_operator(
            spaces[end], get_num_subdivisions(space)
        )
        push!(spaces, new_space)
        push!(two_scale_operators, new_ts)
        Hierarchy.add_level!(domains)
    end

    refine_domains!(domains, two_scale_operators, marked_elements_per_level)
    refined_space = HierarchicalFiniteElementSpace(
        spaces,
        two_scale_operators,
        get_nested_domains(space),
        get_num_subdivisions(space),
        is_truncated(space),
        is_simplified(space),
    )

    return refined_space
end

"""
	refine_domains!(
	    domains::Hierarchy.ActiveInfo,
	    two_scale_operators,
	    marked_elements_per_level::Vector{Vector{Int}},
	)

Refines `domains` in-place, by extracting the children of the `marked_elements_per_level`
using `two_scale_operators`.

# Arguments
- `domains::Hierarchy.ActiveInfo`: The hierarchical domains to be refined.
- `two_scale_operators`: The two-scale operators used to extract the children of the marked
	elements.
- `marked_elements_per_level::Vector{Vector{Int}}`: The elements marked for refinement at
	each level, which will be used to refine the nested domains.

# Returns
- `domains::Hierarchy.ActiveInfo`: The refined `domains`.
"""
function refine_domains!(
    domains::Hierarchy.ActiveInfo,
    two_scale_operators,
    marked_elements_per_level::Vector{Vector{Int}},
)
    for level in eachindex(marked_elements_per_level)
        marked_elements = marked_elements_per_level[level]
        if !isempty(marked_elements)
            refine_domains!(domains, two_scale_operators, marked_elements, level)
        end
    end

    return domains
end

"""
	refine_domains!(domains, two_scale_operators, marked_elements, level)

Refines `domains` in-place, by extracting the children of the `marked_elements` using
`two_scale_operators` at the given `level`.
"""
function refine_domains!(domains, two_scale_operators, marked_elements, level)
    if isempty(marked_elements)
        return domains
    end

    refined_elements = mapreduce(
        el -> get_element_children(two_scale_operators[level], el), vcat, marked_elements
    )
    Hierarchy.update!(domains, level, marked_elements, refined_elements)

    return domains
end

############################################################################################
#                                     Getters                                              #
############################################################################################

function get_active_elements(space::HierarchicalFiniteElementSpace)
    return space.active_elements
end

function get_active_basis(space::HierarchicalFiniteElementSpace)
    return space.active_basis
end

function get_nested_domains(space::HierarchicalFiniteElementSpace)
    return space.nested_domains
end

function get_spaces(space::HierarchicalFiniteElementSpace)
    return space.spaces
end

function get_two_scale_operators(space::HierarchicalFiniteElementSpace)
    return space.two_scale_operators
end

function is_truncated(space::HierarchicalFiniteElementSpace)
    return space.truncated
end

function is_simplified(space::HierarchicalFiniteElementSpace)
    return space.simplified
end

function get_num_levels(space::HierarchicalFiniteElementSpace)
    return Hierarchy.get_num_levels(get_active_elements(space))
end

function get_num_elements(space::HierarchicalFiniteElementSpace)
    return Hierarchy.get_num_objects(get_active_elements(space))
end

function get_num_basis(space::HierarchicalFiniteElementSpace)
    return Hierarchy.get_num_objects(get_active_basis(space))
end

function get_num_subdivisions(space::HierarchicalFiniteElementSpace)
    return space.num_subdivisions
end

function get_num_basis(space::HierarchicalFiniteElementSpace, hier_id::Int)
    return length(get_basis_indices(space, hier_id))
end

function get_max_local_dim(space::HierarchicalFiniteElementSpace)
    return get_max_local_dim(space.spaces[1]) * 2
end

function get_element_level(space::HierarchicalFiniteElementSpace, hier_id::Int)
    return Hierarchy.get_level(get_active_elements(space), hier_id)
end

function get_basis_level(space::HierarchicalFiniteElementSpace, hier_id::Int)
    return Hierarchy.get_level(get_active_basis(space), hier_id)
end

function get_element_level_id(space::HierarchicalFiniteElementSpace, hier_id::Int)
    return Hierarchy.get_level_ids(get_active_elements(space), hier_id)
end

function get_basis_level_id(space::HierarchicalFiniteElementSpace, hier_id::Int)
    return Hierarchy.get_level_ids(get_active_basis(space), hier_id)
end

function get_space(space::HierarchicalFiniteElementSpace, level::Int)
    return space.spaces[level]
end

function get_twoscale_operator(space::HierarchicalFiniteElementSpace, level::Int)
    return space.two_scale_operators[level]
end

function get_level_element_ids(space::HierarchicalFiniteElementSpace, level::Int)
    return Hierarchy.get_level_ids(get_active_elements(space), level)
end

function get_level_basis_ids(space::HierarchicalFiniteElementSpace, level::Int)
    return Hierarchy.get_level_ids(get_active_basis(space), level)
end

function get_level_domain(space::HierarchicalFiniteElementSpace, level::Int)
    return Hierarchy.get_level_ids(get_nested_domains(space), level)
end

function get_multilevel_id(space::HierarchicalFiniteElementSpace, hier_id::Int)
    return space.multilevel_elements[hier_id]
end

function get_element_active_children(
    active_elements::Hierarchy.ActiveInfo,
    level::Int,
    level_id::Int,
    two_scale_operators::Vector{T},
) where {T <: AbstractTwoScaleOperator}
    active_children = NTuple{2, Int}[]
    current_level_ids = [level_id]
    current_level = level
    all_active_check = false
    while !all_active_check
        all_active_check = true
        inactive_children = Int[]
        for level_id in current_level_ids
            children = get_element_children(two_scale_operators[current_level], level_id)
            children_check =
                children .∈ [Hierarchy.get_level_ids(active_elements, current_level + 1)]
            for child_level_id in children[children_check]
                push!(active_children, (current_level + 1, child_level_id))
            end

            append!(inactive_children, children[map(!, children_check)])
            all_active_check = all_active_check && all(children_check)
        end

        current_level_ids = inactive_children
        current_level += 1
    end

    return active_children
end

############################################################################################
#                                Numbering Conversions                                     #
############################################################################################

function convert_to_element_hier_id(
    space::HierarchicalFiniteElementSpace, level::Int, level_id::Int
)
    return Hierarchy.convert_to_hier_id(get_active_elements(space), level, level_id)
end

function convert_to_element_level_id(space::HierarchicalFiniteElementSpace, hier_id::Int)
    return Hierarchy.convert_to_level_id(get_active_elements(space), hier_id)
end

function convert_to_element_level_and_level_id(
    space::HierarchicalFiniteElementSpace, hier_id::Int
)
    return Hierarchy.convert_to_level_and_level_id(get_active_elements(space), hier_id)
end

function convert_to_basis_hier_id(
    space::HierarchicalFiniteElementSpace, level::Int, level_id::Int
)
    return Hierarchy.convert_to_hier_id(get_active_basis(space), level, level_id)
end

function convert_to_basis_level_id(space::HierarchicalFiniteElementSpace, hier_id::Int)
    return Hierarchy.convert_to_level_id(get_active_basis(space), hier_id)
end

function convert_to_basis_level_and_level_id(
    space::HierarchicalFiniteElementSpace, hier_id::Int
)
    return Hierarchy.convert_to_level_and_level_id(get_active_basis(space), hier_id)
end

"""
	convert_element_vector_to_elements_per_level(
	    space::HierarchicalFiniteElementSpace, hier_ids::Vector{Int}
	)

Separates a vector of `hier_ids` in hierarchical indexing into a set of level-wise indices.

# Arguments
- `space::HierarchicalFiniteElementSpace`: The hierarchical finite element space.
- `hier_ids::Vector{Int}`: The list of hierarchical indices.

# Returns
- `Vector{Vector{Int}}`: The level-wise indices. The length of the outer vector is the
	number of levels of `space`.
"""
function convert_element_vector_to_elements_per_level(
    space::HierarchicalFiniteElementSpace, hier_ids::Vector{Int}
)
    L = get_num_levels(space)
    element_ids_per_level = [Int[] for _ in 1:L]

    # Separate the marked elements per level
    for hier_id in hier_ids
        element_level, element_level_id = convert_to_element_level_and_level_id(
            space, hier_id
        )
        append!(element_ids_per_level[element_level], element_level_id)
    end

    return element_ids_per_level
end

############################################################################################
#                                Geometry (STBD)                                           #
############################################################################################

function get_element_vertices(space::HierarchicalFiniteElementSpace, hier_id::Int)
    element_level, element_level_id = convert_to_element_level_and_level_id(space, hier_id)

    return get_element_vertices(space.spaces[element_level], element_level_id)
end

function get_element_measure(space::HierarchicalFiniteElementSpace, hier_id::Int)
    element_level, element_level_id = convert_to_element_level_and_level_id(space, hier_id)

    return get_element_measure(space.spaces[element_level], element_level_id)
end

function _compute_thb_parametric_geometry_coeffs(
    space::HierarchicalFiniteElementSpace{manifold_dim}
) where {manifold_dim}
    num_levels = get_num_levels(space)

    coeffs = Matrix{Float64}(undef, get_num_basis(space), manifold_dim)

    id_count = 1
    for level in 1:num_levels
        greville_points = get_greville_points(get_space(space, level))

        level_active_basis = get_level_basis_ids(space, level)

        for (point_id, point) in enumerate(Iterators.product(greville_points...))
            if point_id ∈ level_active_basis
                coeffs[id_count, :] .= point
                id_count += 1
            end
        end
    end

    return coeffs
end

function _compute_parametric_geometry_coeffs(
    space::HierarchicalFiniteElementSpace{manifold_dim, num_components, num_patches}
) where {manifold_dim, num_components, num_patches}
    if space.truncated
        return _compute_thb_parametric_geometry_coeffs(space)
    end

    return invoke(
        _compute_parametric_geometry_coeffs,
        Tuple{AbstractFESpace{manifold_dim, num_components, num_patches}},
        space,
    )
end

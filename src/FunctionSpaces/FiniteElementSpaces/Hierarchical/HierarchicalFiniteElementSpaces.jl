############################################################################################
#                                     Structure                                            #
############################################################################################

"""
    HierarchicalFiniteElementSpace{
        manifold_dim, num_components, num_patches, S, T, G, GP
    } <: AbstractFESpace{manifold_dim, num_components, num_patches}

A hierarchical space that is built from nested hierarchies of `manifold_dim`-variate
function spaces and domains; see [Giannelli2013](@cite).

# Fields
- `spaces::Vector{S} `: Collection of `manifold_dim`- variate function spaces.
- `two_scale_operators::Vector{T}`: Collection of two-scale operators relating each
	consecutive pair of finite element spaces. See [`AbstractTwoScaleOperator`](@ref).
- `active_elements::HierarchicalActiveInfo`: Information about the active elements in each
	level.
- `active_basis::HierarchicalActiveInfo`: Information about the active basis in each level.
- `nested_domains::HierarchicalActiveInfo`: Information about the nested domains in each
	level. This is the usual definition of Ωₗ in the literature.
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
mutable struct HierarchicalFiniteElementSpace{
    manifold_dim, num_components, num_patches, S, T, G, GP
} <: AbstractFESpace{manifold_dim, num_components, num_patches}
    geometry::G
    parametric_geometry::GP
    spaces::Vector{S}
    two_scale_operators::Vector{T}
    active_elements::HierarchicalActiveInfo
    active_basis::HierarchicalActiveInfo
    nested_domains::HierarchicalActiveInfo
    multilevel_elements::SparseArrays.SparseVector{Int, Int}
    multilevel_extraction_coeffs::Vector{NTuple{num_components, Matrix{Float64}}}
    multilevel_basis_indices::Vector{Vector{Int}}
    num_subdivisions::NTuple{manifold_dim, Int}
    truncated::Bool
    simplified::Bool
    dof_partition::Vector{Vector{Vector{Int}}}

    """
    	HierarchicalFiniteElementSpace(
    		spaces::Vector{S},
    		two_scale_operators::Vector{T},
    		domains::HierarchicalActiveInfo,
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

    Main constructor with compatibility checks.
    """
    function HierarchicalFiniteElementSpace(
        spaces::Vector{S},
        two_scale_operators::Vector{T},
        domains::HierarchicalActiveInfo,
        num_subdivisions::NTuple{manifold_dim, Int},
        truncated::Bool=true,
        simplified::Bool=false;
        mapping::Union{Nothing, Function}=nothing,
        dmapping::Union{Nothing, Function}=nothing,
    ) where {
        manifold_dim,
        num_components,
        num_patches,
        S <: AbstractFESpace{manifold_dim, num_components, num_patches},
        T <: AbstractTwoScaleOperator,
    }
        num_levels = length(spaces)
        # Checks for incompatible arguments
        if num_levels < 1
            throw(ArgumentError("At least 1 level is required, but 0 were given."))
        elseif length(two_scale_operators) != num_levels - 1
            msg1 = "Number of two-scale operators should be `num_levels - 1`. "
            msg2 = "Evaluated $(length(two_scale_operators)) and $(num_levels - 1)."
            throw(ArgumentError(msg1 * msg2))
        elseif get_num_levels(domains) != num_levels
            msg1 = "Number of nested domains should be the same as the number of levels. "
            msg2 = "Evaluated $(num_levels) and $(get_num_levels(domains))."
            throw(ArgumentError(msg1 * msg2))
        end

        # Computes necessary hierarchical information
        active_elements, active_basis, nested_domains = get_active_objects_and_nested_domains(
            spaces, two_scale_operators, domains, simplified
        )
        multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_indices = get_multilevel_extraction(
            spaces, two_scale_operators, active_elements, active_basis, truncated
        )
        dof_partition = compute_dof_partition(spaces, active_basis, num_levels)
        parametric_geometry = create_masked_geometry(
            spaces, two_scale_operators, active_elements
        )
        if isnothing(mapping)
            geometry = parametric_geometry
        else
            geometry = Geometry.MappedGeometry(
                parametric_geometry,
                Geometry.Mapping((manifold_dim, manifold_dim), mapping, dmapping),
            )
        end

        G = typeof(geometry)
        GP = typeof(parametric_geometry)
        # Creates the structure
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

Helper constructor for domains given in a per-level vector.
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
    domains = HierarchicalActiveInfo(domains_per_level)

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

Constructor for a Hierarchical space with no refinement
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
		domains::HierarchicalActiveInfo,
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
- `domains::HierarchicalActiveInfo`: Nested domains where the support of active basis is
    determined.

# Returns
- `active_elements::HierarchicalActiveInfo`: Active elements on each level.
- `active_basis::HierarchicalActiveInfo`: Active basis on each level.
"""
function get_active_objects_and_nested_domains(
    spaces::Vector{S},
    two_scale_operators::Vector{T},
    domains::HierarchicalActiveInfo,
    simplified::Bool,
) where {S <: AbstractFESpace, T <: AbstractTwoScaleOperator}
    num_levels = get_num_levels(domains)
    active_elements_per_level = [collect(1:get_num_elements(spaces[1]))]
    active_basis_per_level = [collect(1:get_num_basis(spaces[1]))]
    nested_domains_per_level = [collect(1:get_num_elements(spaces[1]))]
    # If the hierarchical space is not simplified, we need to ensure that the
    # refinement domains contain proper subsets of a given element's
    # children as refined.
    if !simplified
        new_domains = [get_level_ids(domains, level) for level in 1:num_levels]
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

        domains = HierarchicalActiveInfo(new_domains)
    end

    for level in 1:(num_levels - 1)
        next_level_domain = Set(get_level_ids(domains, level + 1))
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
    active_elements = HierarchicalActiveInfo(active_elements_per_level)
    active_basis = HierarchicalActiveInfo(active_basis_per_level)
    nested_domains = HierarchicalActiveInfo(nested_domains_per_level)

    return active_elements, active_basis, nested_domains
end

"""
    get_multilevel_extraction(
        spaces::Vector{S},
        two_scale_operators::Vector{T},
        active_elements::HierarchicalActiveInfo,
        active_basis::HierarchicalActiveInfo,
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
- `active_elements::HierarchicalActiveInfo`: active elements on each level.
- `active_basis::HierarchicalActiveInfo`: active basis on each level.
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
    active_elements::HierarchicalActiveInfo,
    active_basis::HierarchicalActiveInfo,
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
        return SparseArrays.spzeros(Int, get_num_objects(active_elements)),
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
            basis_level_id -> convert_to_hier_id(active_basis, level, basis_level_id),
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
        multilevel_element_indices[ml_id_count] = convert_to_hier_id(
            active_elements, level, element_level_id
        )
        ml_id_count += 1
    end

    multilevel_elements = SparseArrays.sparsevec(
        multilevel_element_indices,
        1:num_multilevel_elements,
        get_num_objects(active_elements),
    )

    return multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_ids
end

"""
    get_multilevel_information(
        spaces::Vector{S},
        two_scale_operators::Vector{T},
        active_elements::HierarchicalActiveInfo,
        active_basis::HierarchicalActiveInfo,
    ) where {manifold_dim, num_components, num_patches, S <: AbstractFESpace{manifold_dim, num_components, num_patches}, T <: AbstractTwoScaleOperator}

Computes which active elements are multilevel elements, i.e. elements where basis from
multiple levels have non-empty support, as well as which basis from parentr levels are
active on those elements.

# Arguments
- `spaces::Vector{AbstractFESpace{manifold_dim, num_components, num_patches}}`: finite element spaces at each level.
- `two_scale_operators::Vector{AbstractTwoScaleOperator}`: two scale operators relating the
    finite element spaces at each level.
- `active_elements::HierarchicalActiveInfo`: active elements on each level.
- `active_basis::HierarchicalActiveInfo`: active basis on each level.

# Returns
- `multilevel_information::Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}`: information
    about multilevel elements. The key's two indices indicate the multilevel element's
    level and id and the and the key's value is a vector of tuples where the indices are
    the basis level and id (from parentr levels), respectively.
"""
function get_multilevel_information(
    spaces::Vector{S},
    two_scale_operators::Vector{T},
    active_elements::HierarchicalActiveInfo,
    active_basis::HierarchicalActiveInfo,
) where {S <: AbstractFESpace, T <: AbstractTwoScaleOperator}
    L = get_num_levels(active_elements)
    multilevel_information = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
    # Above, first tuple is level and id of ml element, second tuple
    # is level and id of ml basis in that element
    # TODO: Change the storage so that entry of a given key stores all the basis
    # for a given level.
    for level in 1:(L - 1)
        level_active_elements = Set(get_level_ids(active_elements, level))
        level_active_basis = get_level_ids(active_basis, level)
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
        basis_id -> basis_id in get_level_ids(active_basis, level), full_level_indices
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
        multilevel_basis_hier_ids[ml_basis_count] = convert_to_hier_id(
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
                basis_id -> basis_id in get_level_ids(active_basis, level),
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
            level_active_basis = Set(get_level_ids(active_basis, level))
            for partition in eachindex(dof_partition[n_patches])
                active_level_dofs = filter(
                    basis -> basis in level_active_basis,
                    level_partition[level][patch][partition],
                )
                dof_ids = map(
                    og_id -> convert_to_hier_id(active_basis, level, og_id),
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

function create_masked_geometry(spaces, two_scale_operators, active_elements)
    eval_mask = create_evaluation_mask(spaces, two_scale_operators, active_elements)
    base_geoemtry = get_geometry(first(spaces))

    return Geometry.MaskedGeometry(base_geoemtry, eval_mask)
end

function create_evaluation_mask(
    spaces::Vector{S}, two_scale_operators, active_elements
) where {manifold_dim, S <: AbstractFESpace{manifold_dim}}
    num_elements_base = get_num_elements(first(spaces))
    base_geometry = get_parametric_geometry(first(spaces))
    base_element_vertices = [
        Geometry.get_element_vertices(base_geometry, i) for i in 1:num_elements_base
    ]
    num_elements = get_num_objects(active_elements)
    element_id_map = zeros(Int, num_elements)
    eval_element_vertices = Vector{NTuple{manifold_dim, NTuple{2, Float64}}}(
        undef, num_elements
    )
    for i in 1:num_elements
        element_level, element_level_id = convert_to_level_and_level_id(active_elements, i)
        eval_element_vertices[i] = get_element_vertices(
            spaces[element_level], element_level_id
        )
        element_id_map[i] = get_element_ancestor(
            two_scale_operators, element_level_id, element_level, element_level - 1
        )
    end

    translations = Vector{NTuple{manifold_dim, Float64}}(undef, num_elements)
    scalings = Vector{NTuple{manifold_dim, Float64}}(undef, num_elements)
    for element_id in 1:num_elements
        element_id_base = element_id_map[element_id]
        base_el_verts = base_element_vertices[element_id_base]
        eval_el_verts = eval_element_vertices[element_id]
        scalings[element_id] = ntuple(manifold_dim) do k
            eval_scaling = eval_el_verts[k][2] - eval_el_verts[k][1]
            base_scaling = base_el_verts[k][2] - base_el_verts[k][1]

            return eval_scaling / base_scaling
        end

        translations[element_id] = ntuple(manifold_dim) do k
            return (eval_el_verts[k][1] - base_el_verts[k][1]) /
                   (base_el_verts[k][2] - base_el_verts[k][1])
        end
    end

    eval_mask = Geometry.AffineEvaluationMask(
        num_elements, num_elements_base, element_id_map, translations, scalings
    )

    return eval_mask
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
	update_space!(
	    space::HierarchicalFiniteElementSpace, marked_elements_per_level::Vector{Vector{Int}}
	)

Updates the given hierarchical `space` based on  `marked_elements_per_level`.

# Returns
- `space::HierarchicalFiniteElementSpace`: The updated space.

See also [`refine_mesh!`](@ref) and [`update_basis!`](@ref).
"""
function update_space!(
    space::HierarchicalFiniteElementSpace, marked_elements_per_level::Vector{Vector{Int}}
)
    refine_mesh!(space, marked_elements_per_level)
    update_basis!(space)

    return space
end

"""
	refine_mesh!(
	    space::HierarchicalFiniteElementSpace, marked_elements_per_level::Vector{Vector{Int}}
	)

Updates the hierarchical mesh underlying `space` based on `marked_elements_per_level`.

# Returns
- `space::HierarchicalFiniteElementSpace`: Space with refined mesh.
"""
function refine_mesh!(
    space::HierarchicalFiniteElementSpace, marked_elements_per_level::Vector{Vector{Int}}
)
    L = get_num_levels(space)
    if !isempty(marked_elements_per_level[L])
        add_level!(space)
    end

    for level in 1:L
        marked_elements = marked_elements_per_level[level]
        if !isempty(marked_elements)
            refine_mesh!(space, level, marked_elements)
        end
    end

    return space
end

"""
	add_level!(space::HierarchicalFiniteElementSpace)

Adds an empty level to `space`.

# Returns
- `space::HierarchicalFiniteElementSpace`: Space with an extra refinement level.
"""
function add_level!(space::HierarchicalFiniteElementSpace)
    L = get_num_levels(space)
    new_ts, new_space = build_two_scale_operator(
        get_space(space, L), get_num_subdivisions(space)
    )
    push!(get_spaces(space), new_space)
    push!(get_two_scale_operators(space), new_ts)
    push!(get_level_ids(get_active_elements(space)), Int[])
    append!(
        get_level_cum_num_ids(get_active_elements(space)),
        last(get_level_cum_num_ids(get_active_elements(space))),
    )
    push!(get_level_ids(get_nested_domains(space)), Int[])
    append!(
        get_level_cum_num_ids(get_nested_domains(space)),
        last(get_level_cum_num_ids(get_nested_domains(space))),
    )
    push!(get_level_ids(get_active_basis(space)), Int[])
    append!(
        get_level_cum_num_ids(get_active_basis(space)),
        last(get_level_cum_num_ids(get_active_basis(space))),
    )

    return space
end

"""
	refine_mesh!(
	    space::HierarchicalFiniteElementSpace, level::Int, marked_elements::Vector{Int}
	)

Updates the hierarchical mesh underlying `space` based on `marked_elements_per_level` at
`level`.

# Returns
- `space::HierarchicalFiniteElementSpace`: Space with refined mesh at `level`.
"""
function refine_mesh!(
    space::HierarchicalFiniteElementSpace, level::Int, marked_elements::Vector{Int}
)
    if isempty(marked_elements)
        return space
    end

    L = get_num_levels(space)
    if level == L
        add_level!(space)
    end

    active_elements = get_active_elements(space)
    nested_domains = get_nested_domains(space)
    setdiff!(get_level_ids(active_elements, level), marked_elements)
    ts = get_twoscale_operator(space, level)
    refined_elements = mapreduce(el -> get_element_children(ts, el), vcat, marked_elements)
    union!(get_level_ids(active_elements, level + 1), refined_elements)
    union!(get_level_ids(nested_domains, level + 1), refined_elements)
    setfield!(
        space, :active_elements, HierarchicalActiveInfo(get_level_ids(active_elements))
    )
    setfield!(space, :nested_domains, HierarchicalActiveInfo(get_level_ids(nested_domains)))
    parametric_geometry = create_masked_geometry(
        get_spaces(space), get_two_scale_operators(space), get_active_elements(space)
    )
    if typeof(get_geometry(space)) <: Geometry.MappedGeometry
        geometry = Geometry.MappedGeometry(
            parametric_geometry, Geometry.get_mapping(get_geometry(space))
        )
    else
        geometry = parametric_geometry
    end

    setfield!(space, :geometry, geometry)
    setfield!(space, :parametric_geometry, parametric_geometry)

    return space
end

"""
	update_basis!(space::HierarchicalFiniteElementSpace)

Updates the hierarchical basis underlying `space`, after the latter has had mesh refinement.

# Returns
- `space::HierarchicalFiniteElementSpace`: Space with updated hierarchical basis.
"""
function update_basis!(space::HierarchicalFiniteElementSpace)
    L = get_num_levels(space)
    active_basis = get_active_basis(space)
    nested_domains = get_nested_domains(space)
    simplified = is_simplified(space)
    for level in 1:(L - 1)
        level_space = get_space(space, level)
        next_level_space = get_space(space, level + 1)
        level_ts = get_twoscale_operator(space, level)
        next_level_domain = Set(get_level_ids(nested_domains, level + 1))
        basis_to_remove = Int[]
        basis_to_add = Int[]
        if !simplified
            for parent_basis in get_level_ids(active_basis, level)
                parent_support = get_support(level_space, parent_basis)
                if all(
                    child -> child in next_level_domain,
                    mapreduce(
                        el -> get_element_children(level_ts, el), vcat, parent_support
                    ),
                )
                    append!(basis_to_remove, parent_basis)
                end
            end

            for child_basis in 1:get_num_basis(next_level_space)
                child_support = get_support(next_level_space, child_basis)
                if all(child -> child in next_level_domain, child_support)
                    union!(basis_to_add, child_basis)
                end
            end
        else
            for parent_basis in get_level_ids(active_basis, level)
                parent_support = get_support(level_space, parent_basis)
                if all(
                    child -> child in next_level_domain,
                    mapreduce(
                        el -> get_element_children(level_ts, el), vcat, parent_support
                    ),
                )
                    append!(basis_to_remove, parent_basis)
                    union!(basis_to_add, get_basis_children(level_ts, parent_basis))
                end
            end
        end

        setdiff!(get_level_ids(active_basis, level), basis_to_remove)
        union!(get_level_ids(active_basis, level + 1), basis_to_add)
    end

    setfield!(
        active_basis,
        :level_cum_num_ids,
        [0; cumsum(map(length, get_level_ids(active_basis)))],
    )
    multilevel_els, multilevel_coeffs, multilevel_indices = get_multilevel_extraction(
        get_spaces(space),
        get_two_scale_operators(space),
        get_active_elements(space),
        get_active_basis(space),
        is_truncated(space),
    )
    dof_partition = compute_dof_partition(
        get_spaces(space), active_basis, get_num_levels(space)
    )
    setfield!(space, :multilevel_elements, multilevel_els)
    setfield!(space, :multilevel_extraction_coeffs, multilevel_coeffs)
    setfield!(space, :multilevel_basis_indices, multilevel_indices)
    setfield!(space, :dof_partition, dof_partition)

    return space
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
    return get_num_levels(space.active_elements)
end

function get_num_elements(space::HierarchicalFiniteElementSpace)
    return get_num_objects(space.active_elements)
end

function get_num_basis(space::HierarchicalFiniteElementSpace)
    return get_num_objects(space.active_basis)
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
    return get_level(space.active_elements, hier_id)
end

function get_basis_level(space::HierarchicalFiniteElementSpace, hier_id::Int)
    return get_level(space.active_basis, hier_id)
end

function get_element_level_id(space::HierarchicalFiniteElementSpace, hier_id::Int)
    return get_level_ids(space.active_elements, hier_id)
end

function get_basis_level_id(space::HierarchicalFiniteElementSpace, hier_id::Int)
    return get_level_ids(space.active_basis, hier_id)
end

function get_space(space::HierarchicalFiniteElementSpace, level::Int)
    return space.spaces[level]
end

function get_twoscale_operator(space::HierarchicalFiniteElementSpace, level::Int)
    return space.two_scale_operators[level]
end

function get_level_element_ids(space::HierarchicalFiniteElementSpace, level::Int)
    return get_level_ids(space.active_elements, level)
end

function get_level_basis_ids(space::HierarchicalFiniteElementSpace, level::Int)
    return get_level_ids(space.active_basis, level)
end

function get_level_domain(space::HierarchicalFiniteElementSpace, level::Int)
    return get_level_ids(space.nested_domains, level)
end

function get_multilevel_id(space::HierarchicalFiniteElementSpace, hier_id::Int)
    return space.multilevel_elements[hier_id]
end

function get_element_active_children(
    active_elements::HierarchicalActiveInfo,
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
            children_check = children .∈ [get_level_ids(active_elements, current_level + 1)]
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
    return convert_to_hier_id(get_active_elements(space), level, level_id)
end

function convert_to_element_level_id(space::HierarchicalFiniteElementSpace, hier_id::Int)
    return convert_to_level_id(get_active_elements(space), hier_id)
end

function convert_to_element_level_and_level_id(
    space::HierarchicalFiniteElementSpace, hier_id::Int
)
    return convert_to_level_and_level_id(get_active_elements(space), hier_id)
end

function convert_to_basis_hier_id(
    space::HierarchicalFiniteElementSpace, level::Int, level_id::Int
)
    return convert_to_hier_id(get_active_basis(space), level, level_id)
end

function convert_to_basis_level_id(space::HierarchicalFiniteElementSpace, hier_id::Int)
    return convert_to_level_id(get_active_basis(space), hier_id)
end

function convert_to_basis_level_and_level_id(
    space::HierarchicalFiniteElementSpace, hier_id::Int
)
    return convert_to_level_and_level_id(get_active_basis(space), hier_id)
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

@doc raw"""
    struct HierarchicalFiniteElementSpace{n, S, T} <: AbstractFiniteElementSpace{n}

A hierarchical space that is built from nested hierarchies of `n`-variate function spaces and domains. 

# Fields
- `spaces::Vector{AbstractFiniteElementSpace{n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}} `: collection of `n`-variate function spaces.
- `two_scale_operators::Vector{AbstractTwoScaleOperator}`: collection of two-scale operators relating each consecutive pair of finite element spaces.
- `active_elements::HierarchicalActiveInfo`: information about the active elements in each level.
- `active_basis::HierarchicalActiveInfo`: information about the active basis in each level.
- `multilevel_elements::SparseArrays.SparseVector{Int, Int}`: elements where basis from multiple levels have non-empty support.
- `multilevel_extraction_coeffs::Vector{Matrix{Float64}}`: extraction coefficients of active basis in `multilevel_elements`.
- `multilevel_basis_indices::Vector{Vector{Int}}`: indices of active basis in `multilevel_elements`.
"""
mutable struct HierarchicalFiniteElementSpace{n, S, T} <: AbstractFiniteElementSpace{n}
    spaces::Vector{S}   
    two_scale_operators::Vector{T}
    active_elements::HierarchicalActiveInfo
    active_basis::HierarchicalActiveInfo
    nested_domains::HierarchicalActiveInfo
    multilevel_elements::SparseArrays.SparseVector{Int, Int}
    multilevel_extraction_coeffs::Vector{Matrix{Float64}}
    multilevel_basis_indices::Vector{Vector{Int}}
    truncated::Bool
    dof_partition::Vector{Vector{Vector{Int}}}

    # Constructor that builds the space
    function HierarchicalFiniteElementSpace(spaces::Vector{S}, two_scale_operators::Vector{T}, domains::HierarchicalActiveInfo, truncated::Bool=false) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

        function _compute_dof_partition(spaces, active_basis)

            level_partition = get_dof_partition.(spaces)
            n_patches = length(level_partition[1])
            n_partitions = [length(level_partition[1][i]) for i ∈ 1:n_patches]
            dof_partition = Vector{Vector{Vector{Int}}}(undef, n_patches)

            for patch ∈ 1:n_patches
                dof_partition[patch] = Vector{Vector{Int}}(undef, n_partitions[patch])
                for partition ∈ 1:n_partitions[patch]
                    for level ∈ 1:num_levels
                        level_active_basis = [get_level_ids(active_basis, level)]
                        active_dof_partition_checks = level_partition[level][patch][partition] .∈ level_active_basis
                        dof_ids = convert_to_hier_id.(Ref(active_basis), level, level_partition[level][patch][partition][active_dof_partition_checks])

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

        num_levels = length(spaces)

        # Checks for incompatible arguments
        if num_levels<1
            throw(ArgumentError("At least 1 level is required, but 0 were given."))
        elseif length(two_scale_operators) != num_levels - 1
            msg1 = "Number of two-scale operators should be one less than the number of levels. "
            msg2 = "$num_levels refinement levels and $(length(two_scale_operators)) two-scale operators were given."
            throw(ArgumentError(msg1*msg2))
        elseif get_num_levels(domains) != num_levels
            msg1 = "Number of nested domains should be the same as the number of levels. "
            msg2 = "$num_levels refinement levels and $(get_num_levels(domains)) domains were given."
            throw(ArgumentError(msg1*msg2))
        end

        # Computes necessary hierarchical information
        active_elements, active_basis, nested_domains = get_active_objects_and_nested_domains(spaces, two_scale_operators, domains)
        multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_indices = get_multilevel_extraction(spaces, two_scale_operators, active_elements, active_basis, truncated)

        dof_partition = _compute_dof_partition(spaces, active_basis)

        # Creates the structure
        return new{n, S, T}(spaces, two_scale_operators, active_elements, active_basis, nested_domains,
                            multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_indices,
                            truncated, dof_partition)
    end

    # Helper constructor for domains given in a per-level vector.
    function HierarchicalFiniteElementSpace(spaces::Vector{S}, two_scale_operators::Vector{T}, domains_per_level::Vector{Vector{Int}}, truncated::Bool=false) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

        domains = HierarchicalActiveInfo(domains_per_level)
        
        return HierarchicalFiniteElementSpace(spaces, two_scale_operators, domains, truncated)
    end
end

# Basis getters for HierarchicalFiniteElementSpace

function get_num_levels(hier_space::HierarchicalFiniteElementSpace{n, S, T}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_num_levels(hier_space.active_elements)
end

function get_num_elements(hier_space::HierarchicalFiniteElementSpace{n, S, T}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_num_objects(hier_space.active_elements)
end

function get_num_basis(hier_space::HierarchicalFiniteElementSpace{n, S, T}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_num_objects(hier_space.active_basis)
end

function get_max_local_dim(hier_space::HierarchicalFiniteElementSpace{n, S, T}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_max_local_dim(hier_space.spaces[1])*2
end

function get_dof_partition(hier_space::HierarchicalFiniteElementSpace{n, S, T}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return hier_space.dof_partition
end

function get_element_level(hier_space::HierarchicalFiniteElementSpace{n, S, T}, hier_id::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_level(hier_space.active_elements, hier_id)
end

function get_basis_level(hier_space::HierarchicalFiniteElementSpace{n, S, T}, hier_id::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_level(hier_space.active_basis, hier_id)
end

function get_element_level_id(hier_space::HierarchicalFiniteElementSpace{n, S, T}, hier_id::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_level_id(hier_space.active_elements, hier_id)
end

function get_basis_level_id(hier_space::HierarchicalFiniteElementSpace{n, S, T}, hier_id::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_level_id(hier_space.active_basis, hier_id)
end

function get_space(hier_space::HierarchicalFiniteElementSpace{n, S, T}, level::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return hier_space.spaces[level]
end

function get_twoscale_operator(hier_space::HierarchicalFiniteElementSpace{n, S, T}, level::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return hier_space.two_scale_operators[level]
end

function get_level_element_ids(hier_space::HierarchicalFiniteElementSpace{n, S, T}, level::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_level_ids(hier_space.active_elements, level)
end

function get_level_basis_ids(hier_space::HierarchicalFiniteElementSpace{n, S, T}, level::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_level_ids(hier_space.active_basis, level)
end

# Other basic functionality

function convert_to_element_hier_id(hier_space::HierarchicalFiniteElementSpace{n, S, T}, level::Int, level_id::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return convert_to_hier_id(hier_space.active_elements, level, level_id)
end

function convert_to_element_level_id(hier_space::HierarchicalFiniteElementSpace{n, S, T}, hier_id::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return convert_to_level_id(hier_space.active_elements, hier_id)
end

function convert_to_element_level_and_level_id(hier_space::HierarchicalFiniteElementSpace{n, S, T}, hier_id::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return convert_to_level_and_level_id(hier_space.active_elements, hier_id)
end

function convert_to_basis_hier_id(hier_space::HierarchicalFiniteElementSpace{n, S, T}, level::Int, level_id::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return convert_to_hier_id(hier_space.active_basis, level, level_id)
end

function convert_to_basis_level_id(hier_space::HierarchicalFiniteElementSpace{n, S, T}, hier_id::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return convert_to_level_id(hier_space.active_basis, hier_id)
end

function convert_to_basis_level_and_level_id(hier_space::HierarchicalFiniteElementSpace{n, S, T}, hier_id::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return convert_to_level_and_level_id(hier_space.active_basis, hier_id)
end

function get_element_active_children(active_elements::HierarchicalActiveInfo, level::Int, level_id::Int, two_scale_operators::Vector{T}) where {T<:AbstractTwoScaleOperator}
    active_children = NTuple{2, Int}[]

    current_level_ids = [level_id]
    current_level = level

    all_active_check = false
    while !all_active_check
        all_active_check = true
        inactive_children = Int[]

        for level_id ∈ current_level_ids 
            children = get_element_children(two_scale_operators[current_level], level_id) 
            children_check = children .∈ [get_level_ids(active_elements, current_level+1)]
            for child_level_id ∈ children[children_check]
                push!(active_children, (current_level+1, child_level_id))
            end
            append!(inactive_children, children[map(!, children_check)])

            all_active_check = all_active_check && all(children_check)
        end
        current_level_ids = inactive_children
        current_level += 1
    end

    return active_children
end

function convert_element_vector_to_elements_per_level(hier_space::HierarchicalFiniteElementSpace{n, S, T}, hier_ids::Vector{Int}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    L = get_num_levels(hier_space)
    element_ids_per_level = [Int[] for _ ∈ 1:L]
    
    # Separate the marked elements per level
    for hier_id ∈ hier_ids
        element_level, element_level_id = convert_to_element_level_and_level_id(hier_space, hier_id)
        append!(element_ids_per_level[element_level], element_level_id)
    end

    return element_ids_per_level
end

function get_level_domain(hier_space::HierarchicalFiniteElementSpace{n, S, T}, level::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    return get_level_ids(hier_space.nested_domains, level)
end

# Methods for hierarchical space constructor

@doc raw"""
    get_active_objects(spaces::Vector{S}, two_scale_operators::Vector{T}, domains::HierarchicalActiveInfo) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Computes the active elements and basis on each level based on `spaces`, `two_scale_operators` and the set of nested `domains`.

The construction loops over the `domains` on each level and selects the active basis in the next level as the children of deactivated basis, based on their supports,
in the current level. The active elments in the next level are then given as the union of support of said basis in the next level. 
This differs slightly from the usual algorithm for generating the hierarchical space, where basis in the next level are only determined by whether their support is fully contained
in the next level domain, regardless of whether their parent basis are active or not.


# Arguments
- `spaces::Vector{AbstractFiniteElementSpace{n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}}`: finite element spaces at each level. 
- `two_scale_operators::Vector{AbstractTwoScaleOperator}`: two scale operators relating the finite element spaces at each level.
- `domains::HierarchicalActiveInfo`: nested domains where the support of active basis is determined.

# Returns
- `active_elements::HierarchicalActiveInfo`: active elements on each level.
- `active_basis::HierarchicalActiveInfo`: active basis on each level.
"""
function get_active_objects_and_nested_domains(spaces::Vector{S}, two_scale_operators::Vector{T}, domains::HierarchicalActiveInfo) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    num_levels = get_num_levels(domains)

    # Initialize active basis and elements on first level
    active_elements_per_level = [collect(1:get_num_elements(spaces[1]))]
    active_basis_per_level = [collect(1:get_num_basis(spaces[1]))]
    nested_domains_per_level = [collect(1:get_num_elements(spaces[1]))]

    for level in 1:num_levels-1 # Loop over levels
        next_level_domain = [get_level_ids(domains, level+1)]

        elements_to_remove = Int[]
        elements_to_add = Int[]
        basis_to_remove = Int[]
        basis_to_add = Int[]

        for Ni ∈ active_basis_per_level[level] # Loop over active basis on current level
            # Gets the support of Ni on current level and the next one
            support = get_support(spaces[level], Ni)
            element_children = [child for parent in support for child in get_element_children(two_scale_operators[level], parent)]
            check_in_next_domain = element_children .∈ next_level_domain # checks if the support is contained in the next level domain

            # Updates elements and basis to add and remove based on check_in_next_domain
            if all(check_in_next_domain)
                append!(elements_to_remove, support)
                append!(basis_to_remove, Ni)
                append!(elements_to_add, element_children)
                append!(basis_to_add, get_basis_children(two_scale_operators[level], Ni))
            end
        end
        
        # Remove inactive elements and basis on current level
        active_elements_per_level[level] = setdiff(active_elements_per_level[level], elements_to_remove)
        active_basis_per_level[level] = setdiff(active_basis_per_level[level], basis_to_remove)
        # Add active elements and basis on next level
        push!(active_elements_per_level, unique(elements_to_add))
        push!(active_basis_per_level, unique(basis_to_add))
        # Store nested domains Ωˡ
        push!(nested_domains_per_level, elements_to_add)
    end

    map(x -> sort!(x), active_elements_per_level)
    map(x -> sort!(x), active_basis_per_level)

    active_elements = HierarchicalActiveInfo(active_elements_per_level)
    active_basis = HierarchicalActiveInfo(active_basis_per_level)
    nested_domains = HierarchicalActiveInfo(nested_domains_per_level)
    
    return active_elements, active_basis, nested_domains
end

@doc raw"""
    get_multilevel_extraction(spaces::Vector{S}, two_scale_operators::Vector{T}, active_elements::HierarchicalActiveInfo, active_basis::HierarchicalActiveInfo, truncated::Bool) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Computes which elements are multilevel elements, i.e. elements for which basis from multiple levels have non-emtpy support, as well as their extraction coefficients matrices and active basis indices.

The extraction coefficients depend on whether the hierarchical space is `truncated` or not.

# Arguments
- `spaces::Vector{AbstractFiniteElementSpace{n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}}`: finite element spaces at each level. 
- `two_scale_operators::Vector{AbstractTwoScaleOperator}`: two scale operators relating the finite element spaces at each level.
- `active_elements::HierarchicalActiveInfo`: active elements on each level.
- `active_basis::HierarchicalActiveInfo`: active basis on each level.
- `truncated`: flag for a truncated hierarchical space.

# Returns 
- `multilevel_elements::SparseArrays.SparseVector{Int, Int}`: elements where basis from multiple levels
    have non-empty support.
- `multilevel_extraction_coeffs::Vector{Matrix{Float64}}`: extraction coefficients of active basis in `multilevel_elements`.
- `multilevel_basis_indices::Vector{Vector{Int}}`: indices of active basis in `multilevel_elements`.
"""
function get_multilevel_extraction(spaces::Vector{S}, two_scale_operators::Vector{T}, active_elements::HierarchicalActiveInfo, active_basis::HierarchicalActiveInfo, truncated::Bool) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

    function _get_active_basis_matrix(space, element_level_id, level, active_basis)
        full_coeffs, full_level_indices = get_extraction(space, element_level_id)
        active_local_ids = findall(x -> x in get_level_ids(active_basis, level), full_level_indices)
    
        return Matrix{Float64}(LinearAlgebra.I, size(full_coeffs))[:, active_local_ids], active_local_ids
    end

    multilevel_information = get_multilevel_information(spaces, two_scale_operators, active_elements, active_basis)
    
    num_multilevel_elements = length(keys(multilevel_information))

    if num_multilevel_elements == 0 # Skip trivial case (first adaptive iteration step)
        return SparseArrays.spzeros(Int, get_num_objects(active_elements)), Matrix{Float64}[], [Int[]]
    end

    multilevel_element_indices = Vector{Int}(undef, num_multilevel_elements)
    multilevel_extraction_coeffs = Vector{Matrix{Float64}}(undef, num_multilevel_elements)
    multilevel_basis_ids = Vector{Vector{Int}}(undef, num_multilevel_elements)

    ml_id_count = 1
    for (level, element_level_id) ∈ keys(multilevel_information)
        # Create multilevel element specific extraction coefficients
        active_basis_matrix, active_local_ids = _get_active_basis_matrix(spaces[level], element_level_id, level, active_basis)

        refinement_matrix, multilevel_basis_hier_ids = get_refinement_data(active_basis_matrix, active_local_ids, spaces, two_scale_operators, active_basis, element_level_id, level, multilevel_information, truncated)

        # Convert active local ids to hierarchical ids
        level_coeffs, basis_level_ids = get_extraction(spaces[level], element_level_id)
        basis_hier_ids = convert_to_hier_id.(Ref(active_basis), Ref(level), basis_level_ids[active_local_ids])

        # Add multilevel extraction data
        multilevel_extraction_coeffs[ml_id_count] = level_coeffs * refinement_matrix
        multilevel_basis_ids[ml_id_count] = append!(basis_hier_ids, multilevel_basis_hier_ids)

        # Add multilevel element specific index
        multilevel_element_indices[ml_id_count] = convert_to_hier_id(active_elements, level, element_level_id)
        ml_id_count += 1
    end

    multilevel_elements = SparseArrays.sparsevec(multilevel_element_indices, 1:num_multilevel_elements, get_num_objects(active_elements))

    return multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_ids
end

@doc raw"""
    get_multilevel_information(spaces::Vector{S}, two_scale_operators::Vector{T}, active_elements::HierarchicalActiveInfo, active_basis::HierarchicalActiveInfo) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

Computes which active elements are multilevel elements, i.e. elements where basis from multiple levels have non-empty support, as well as which basis from coarser levels are active on those elements.

# Arguments

- `spaces::Vector{AbstractFiniteElementSpace{n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}}`: finite element spaces at each level. 
- `two_scale_operators::Vector{AbstractTwoScaleOperator}`: two scale operators relating the finite element spaces at each level.
- `active_elements::HierarchicalActiveInfo`: active elements on each level.
- `active_basis::HierarchicalActiveInfo`: active basis on each level.

# Returns
- `multilevel_information::Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}`: information about multilevel elements. The key's two indices indicate the multilevel element's level and id and the and the key's value is a vector of tuples where the indices are the basis level and id (from coarser levels), respectively.
"""
function get_multilevel_information(spaces::Vector{S}, two_scale_operators::Vector{T}, active_elements::HierarchicalActiveInfo, active_basis::HierarchicalActiveInfo) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    L = get_num_levels(active_elements)
    multilevel_information = Dict{Tuple{Int, Int}, Vector{Tuple{Int, Int}}}()
    # Above, first tuple is level and id of ml element, second tuple
    # is level and id of ml basis in that element
    for level ∈ 1:L-1
        level_active_elements = [get_level_ids(active_elements, level)]
        level_active_basis = get_level_ids(active_basis, level)

        for basis ∈ level_active_basis
            support = get_support(spaces[level], basis)

            active_support_checks = support .∈ level_active_elements
            for inactive_element ∈ support[map(!, active_support_checks)]
                active_children = get_element_active_children(active_elements, level, inactive_element, two_scale_operators)
                for (child_level, child_id) ∈ active_children
                    if haskey(multilevel_information, (child_level, child_id))
                        push!(multilevel_information[(child_level, child_id)], (level, basis))
                    else
                        multilevel_information[(child_level, child_id)] = [(level, basis)]
                    end
                end
            end
        end
    end

    return multilevel_information
end

function get_refinement_data(active_basis_matrix, active_local_ids, fe_spaces, two_scale_operators, active_basis, element_id, element_level, multilevel_information, truncated)
    active_basis_size = size(active_basis_matrix)

    num_multilevel_basis = length(multilevel_information[(element_level, element_id)])

    refinement_matrix = zeros(active_basis_size .+ (0, num_multilevel_basis))

    refinement_matrix[:,1:active_basis_size[2]] .= active_basis_matrix

    multilevel_basis_hier_ids = Vector{Int}(undef, num_multilevel_basis)
    
    ml_basis_count = 1
    for (basis_level, basis_level_id) ∈ multilevel_information[(element_level, element_id)]
        refinement_matrix[:,active_basis_size[2]+ml_basis_count] .= get_multilevel_basis_evaluation(fe_spaces, two_scale_operators, active_basis, basis_level, basis_level_id, element_level, element_id, truncated)

        multilevel_basis_hier_ids[ml_basis_count] = convert_to_hier_id(active_basis, basis_level, basis_level_id)

        ml_basis_count += 1
    end
    if truncated
        refinement_matrix = truncate_refinement_matrix!(refinement_matrix, active_local_ids)
    end

    return refinement_matrix, multilevel_basis_hier_ids
end

function get_multilevel_basis_evaluation(fe_spaces, two_scale_operators, active_basis, basis_level, basis_id, element_level, element_level_id, truncated::Bool)
    local_subdiv_matrix = LinearAlgebra.I
    current_child_element = element_level_id

    for level ∈ element_level:-1:basis_level+1
        current_parent_element = get_element_parent(two_scale_operators[level-1], current_child_element)

        current_subdiv_matrix = get_local_subdiv_matrix(two_scale_operators[level-1], current_parent_element, current_child_element)

        if truncated
            _, full_level_indices = get_extraction(fe_spaces[level], current_child_element)
            active_indices = findall(x -> x in get_level_ids(active_basis, level), full_level_indices)
            current_subdiv_matrix[active_indices, :] .= 0.0
        end
        
        if level==element_level
            local_subdiv_matrix = local_subdiv_matrix * current_subdiv_matrix
        else
            local_subdiv_matrix .= local_subdiv_matrix * current_subdiv_matrix
        end
        current_child_element = current_parent_element
    end

    level_diff = element_level-basis_level
    basis_element_level_id = get_element_ancestor(two_scale_operators, element_level_id, element_level, level_diff)
    _, lowest_level_basis_indices = get_extraction(fe_spaces[basis_level], basis_element_level_id)
    basis_local_id = findfirst(x -> x == basis_id, lowest_level_basis_indices)

    return @view local_subdiv_matrix[:, basis_local_id]
end

@doc raw"""
    truncate_refinement_matrix!(refinement_matrix, active_indices::Vector{Int})

Updates `refinement_matrix` by the rows of `active_indices` to zeros in lower level basis functions.

# Arguments

- `refinement_matrix`: the refinement matrix to be updated.
- `active_indices::Vector{Int}`: element local indices of active basis functions from the highest refinement level.

# Returns

- `refinement_matrix`: truncated refinement matrix.
"""
function truncate_refinement_matrix!(refinement_matrix, active_indices::Vector{Int})
    active_length = length(active_indices)
    refinement_matrix[active_indices, active_length+1:end] .= 0.0

    return refinement_matrix
end

# Extraction method for hierarchical space

function get_local_basis(hier_space::HierarchicalFiniteElementSpace{n, S, T}, hier_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    element_level, element_level_id = convert_to_element_level_and_level_id(hier_space, hier_id)

    return get_local_basis(get_space(hier_space, element_level), element_level_id, xi, nderivatives)
end

function get_extraction(hier_space::HierarchicalFiniteElementSpace{n, S, T}, hier_id::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    if hier_space.multilevel_elements[hier_id] == 0
        element_level, element_level_id = convert_to_element_level_and_level_id(hier_space, hier_id)
        coeffs, basis_level_ids = get_extraction(get_space(hier_space, element_level), element_level_id)

        # Convert level space basis indices to hierarchical space basis indices
        basis_indices = convert_to_basis_hier_id.(Ref(hier_space), Ref(element_level), basis_level_ids)
    else
        multilevel_id = hier_space.multilevel_elements[hier_id]
        coeffs = hier_space.multilevel_extraction_coeffs[multilevel_id]
        basis_indices = hier_space.multilevel_basis_indices[multilevel_id]
    end

    return coeffs, basis_indices
end

# Methods for updating the hierarchical space

# Needs to be fixed
#=
function update_hierarchical_space!(hier_space::HierarchicalFiniteElementSpace{n, S, T}, domains::HierarchicalActiveInfo, new_operator::T, new_space::S) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    function _extend_levels!(hier_space, new_operator, new_space)
        push!(hier_space.two_scale_operators, new_operator)
        push!(hier_space.spaces, new_space)

        active_element_ids = push!(hier_space.active_elements.level_ids, Int[])
        active_basis_ids = push!(hier_space.active_basis.level_ids, Int[])

        hier_space.active_elements = HierarchicalActiveInfo(active_element_ids)
        hier_space.active_basis = HierarchicalActiveInfo(active_basis_ids)

        return hier_space
    end

    function _update_active_objects!(hier_space::HierarchicalFiniteElementSpace{n, S, T}, domains::HierarchicalActiveInfo) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}

        function _update_elements!(hier_space, level, elements_to_remove, elements_to_add)
            # remove elements
            hier_space.active_elements.level_ids[level] = setdiff(get_level_element_ids(hier_space, level), elements_to_remove)
            # add elements
            hier_space.active_elements.level_ids[level+1] = union(get_level_domain(hier_space, level+1), elements_to_add)
            
            return hier_space
        end
    
        function _update_basis!(hier_space, level, basis_to_remove, basis_to_add)
            # remove basis
            hier_space.active_basis.level_ids[level] = setdiff(get_level_basis_ids(hier_space, level), basis_to_remove)
            # add basis
            hier_space.active_basis.level_ids[level+1] = union(get_basis_contained_in_next_level(hier_space, level+1), basis_to_add)
    
            return hier_space
        end
    
        num_levels = get_num_levels(domains)
    
        # Initialize active basis and elements on first level
    
        for level in 1:num_levels-1 # Loop over levels
            next_level_domain = [union(get_level_domain(hier_space, level+1), get_level_ids(domains, level+1))]
    
            elements_to_remove = Int[]
            elements_to_add = Int[]
            basis_to_remove = Int[]
            basis_to_add = Int[]
    
            for Ni ∈ get_level_basis_ids(hier_space, level) # Loop over active basis on current level
                # Gets the support of Ni on current level and the next one
                support = get_support(get_space(hier_space, level), Ni)
                support_children = get_element_children(get_twoscale_operator(hier_space, level), support)
                check_in_next_domain = support_children .∈ next_level_domain # checks if the support is contained in the next level domain
    
                # Updates elements and basis to add and remove based on check_in_next_domain
                if all(check_in_next_domain)
                    append!(elements_to_remove, support)
                    append!(elements_to_add, support_children)
                    append!(basis_to_remove, Ni)
                    append!(basis_to_add, get_basis_children(get_twoscale_operator(hier_space, level), Ni))
                end
            end
            
            # Remove inactive elements and basis on current level
            _update_elements!(hier_space, level, elements_to_remove, elements_to_add)
            _update_basis!(hier_space, level, basis_to_remove, basis_to_add)
        end
    
        map(x -> sort!(x), hier_space.active_elements.level_ids)
        map(x -> sort!(x), hier_space.active_basis.level_ids)

        hier_space.active_elements = HierarchicalActiveInfo(hier_space.active_elements.level_ids)
        hier_space.active_basis = HierarchicalActiveInfo(hier_space.active_basis.level_ids)
    
        return hier_space
    end

    if get_num_levels(domains)==get_num_levels(hier_space)+1
        _extend_levels!(hier_space, new_operator, new_space)
    elseif get_num_levels(domains)>get_num_levels(hier_space)+1
        throw(ArgumentError("It is only possible to update by one level at a time."))
    end

    _update_active_objects!(hier_space, domains) # updates active elements and basis

    multilevel_elements, multilevel_extraction_coeffs, multilevel_basis_indices = get_multilevel_extraction(hier_space.spaces, hier_space.two_scale_operators, hier_space.active_elements, hier_space.active_basis, hier_space.truncated)

    hier_space.multilevel_elements = multilevel_elements
    hier_space.multilevel_extraction_coeffs = multilevel_extraction_coeffs
    hier_space.multilevel_basis_indices = multilevel_basis_indices

    return hier_space
end
=#

# Needs to be fixed
function update_hierarchical_space!(hier_space::HierarchicalFiniteElementSpace{n, S, T}, domains::Vector{Vector{Int}}, new_operator::T, new_space::S, t=false) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    num_levels = get_num_levels(hier_space)
    for level ∈ 1:num_levels
        union!(domains[level], get_level_domain(hier_space, level))
    end

    if length(domains)>num_levels
        return HierarchicalFiniteElementSpace(vcat(hier_space.spaces,new_space), vcat(hier_space.two_scale_operators,new_operator), domains, hier_space.truncated)
    end

    return HierarchicalFiniteElementSpace(hier_space.spaces, hier_space.two_scale_operators, domains, hier_space.truncated)
end

# Useful for refinement 

function get_level_inactive_domain(hier_space::HierarchicalFiniteElementSpace{n, S, T}, level::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    inactive_basis = setdiff(1:get_num_elements(hier_space.spaces[level]), get_level_element_ids(hier_space, level))
    if level > 1
        inactive_basis = setdiff(inactive_basis, get_level_element_ids(hier_space, level-1))
    end

    return inactive_basis
end

function get_basis_contained_in_next_level_domain(hier_space::HierarchicalFiniteElementSpace{n, S, T}, level::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    basis_contained_in_next_level = Int[]
    next_level_domain = [get_level_domain(hier_space, level+1)]

    for basis ∈ setdiff(1:get_num_basis(hier_space.spaces[level]), get_level_basis_ids(hier_space, level))
        basis_support = get_support(hier_space.spaces[level], basis)
        basis_support_children = get_element_children(get_twoscale_operator(hier_space, level), basis_support)

        if all(basis_support_children .∈ next_level_domain)
            append!(basis_contained_in_next_level, basis)
        end
    end

    return basis_contained_in_next_level
end

# Geometry methods

function _get_element_size(hier_space::HierarchicalFiniteElementSpace{n, S, T}, hier_id::Int) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    element_level, element_level_id = convert_to_element_level_and_level_id(hier_space, hier_id)

    return _get_element_size(hier_space.spaces[element_level], element_level_id)
end

function _compute_thb_parametric_geometry_coeffs(hspace::HierarchicalFiniteElementSpace{manifold_dim, S, T}) where {manifold_dim, S<:AbstractFiniteElementSpace{manifold_dim}, T<:AbstractTwoScaleOperator}
    num_levels = get_num_levels(hspace)
    
    coeffs = Matrix{Float64}(undef, get_num_basis(hspace), manifold_dim)

    id_count = 1
    for level ∈ 1:num_levels
        greville_points = get_greville_points(get_space(hspace, level))
        
        level_active_basis = get_level_basis_ids(hspace, level)

        for (point_id, point) ∈ enumerate(Iterators.product(greville_points...))
            if point_id ∈ level_active_basis
                coeffs[id_count, :] .= point
                id_count += 1
            end
        end
    end

    return coeffs
end

function _compute_parametric_geometry_coeffs(hspace::HierarchicalFiniteElementSpace{n, S, T}) where {n, S<:AbstractFiniteElementSpace{n}, T<:AbstractTwoScaleOperator}
    if hspace.truncated
        return _compute_thb_parametric_geometry_coeffs(hspace)
    end

    return invoke(_compute_parametric_geometry_coeffs, Tuple{AbstractFiniteElementSpace{n}}, hspace)
end


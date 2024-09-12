using LinearAlgebra

"""
    UnstructuredSpace{n,m} <: AbstractFiniteElementSpace{n}

An `n`-variate multi-patch space with `m` patches, representing an unstructured finite element space.

# Fields
- `function_spaces::NTuple{m, AbstractFiniteElementSpace{n}}`: Collection of `m` (uni or multivariate) function spaces.
- `extraction_op::ExtractionOperator`: Extraction operator that specifies how to combine functions from the `m` spaces into functions for the unstructured space.
- `dof_partition::Vector{Vector{Int}}`: Partition of degrees of freedom.
- `us_config::Dict`: Dictionary that stores helper functionality (e.g., connectivity) for the unstructured space.
- `data::Dict`: Any auxiliary data that the user wants to store for this unstructured space.
"""
struct UnstructuredSpace{n,m} <: AbstractFiniteElementSpace{n}
    function_spaces::NTuple{m, AbstractFiniteElementSpace{n}}
    extraction_op::ExtractionOperator
    dof_partition::Vector{Vector{Int}}
    us_config::Dict
    data::Dict

    function UnstructuredSpace(function_spaces::NTuple{m,AbstractFiniteElementSpace{n}}, extraction_op::ExtractionOperator, dof_partition::Vector{Vector{Int}}, us_config::Dict, data::Dict) where {n,m}
        # Initialize with empty dof partitioning
        new{n,m}(function_spaces, extraction_op, dof_partition, us_config, data)
    end

    function UnstructuredSpace(function_spaces::NTuple{m,AbstractFiniteElementSpace{1}}, extraction_op::ExtractionOperator, data::Dict) where {m}
        UnstructuredSpace(function_spaces, extraction_op, 1, 1, data)
    end

    function UnstructuredSpace(function_spaces::NTuple{m,AbstractFiniteElementSpace{1}}, extraction_op::ExtractionOperator, n_dofs_left::Int, n_dofs_right::Int, data::Dict) where {m}
        # Build 1D topology
        patch_neighbours = [-1 (1:m-1)...
                            (2:m)... -1]
        # Number of elements per patch
        patch_nels = [0; cumsum([get_num_elements(function_spaces[i]) for i = 1:m])]

        # Assemble patch config in a dictionary
        us_config = Dict("patch_neighbours" => patch_neighbours, "patch_nels" => patch_nels)

        # Allocate memory for degree of freedom partitioning
        dof_partition = Vector{Vector{Int}}(undef,3)
        # First, store the left dofs ...
        dof_partition[1] = collect(1:n_dofs_left)
        # ... then the interior dofs ...
        dof_partition[2] = collect(n_dofs_left+1:get_num_basis(extraction_op)-n_dofs_right)
        # ... and then finally the right dofs.
        dof_partition[3] = collect(get_num_basis(extraction_op)-n_dofs_right+1:get_num_basis(extraction_op))

        new{1,m}(function_spaces, extraction_op, dof_partition, us_config, data)
    end
end

"""
    get_num_basis(us_space::UnstructuredSpace)

Get the dimension of the unstructured function space.

# Arguments
- `us_space::UnstructuredSpace`: The unstructured function space.

# Returns
- `::Int`: The dimension of the space.
"""
function get_num_basis(us_space::UnstructuredSpace)
    return get_num_basis(us_space.extraction_op)
end

"""
    get_num_elements(us_space::UnstructuredSpace)

Get the total number of elements for the partition over which the function space is defined.

# Arguments 
- `us_space::UnstructuredSpace`: The unstructured space.

# Returns
- `::Int`: Total number of elements.
"""
function get_num_elements(us_space::UnstructuredSpace)
    return get_num_elements(us_space.extraction_op)
end

"""
    get_polynomial_degree(us_space::UnstructuredSpace, element_id::Int)

Get the polynomial degree of the specified element in the unstructured space.

# Arguments
- `us_space::UnstructuredSpace`: The unstructured space.
- `element_id::Int`: The global element ID.

# Returns
- `::Int`: The polynomial degree of the specified element.
"""
function get_polynomial_degree(us_space::UnstructuredSpace, element_id::Int)
    # Find the space ID and local element ID
    space_id = get_space_id(us_space, element_id)
    space_element_id = element_id - us_space.us_config["patch_nels"][space_id]
    
    # Get the polynomial degree from the corresponding function space
    return get_polynomial_degree(us_space.function_spaces[space_id], space_element_id)
end

"""
    get_extraction(us_space::UnstructuredSpace, element_id::Int)

Get extraction coefficients and global basis indices for the specified global element ID.

# Arguments 
- `us_space::UnstructuredSpace`: The unstructured space.
- `element_id::Int`: The global element ID.

# Returns
- `extraction_coefficients::Array{Float64}`: Extraction coefficients.
- `basis_indices::Vector{Int}`: Indices of global basis functions.
"""
function get_extraction(us_space::UnstructuredSpace, element_id::Int)
    return get_extraction(us_space.extraction_op, element_id)
end

"""
    get_space_id(us_space::UnstructuredSpace, element_id::Int)

Get the ID of the constituent space to which the specified element belongs.

# Arguments 
- `us_space::UnstructuredSpace`: The unstructured space.
- `element_id::Int`: The global element ID.

# Returns
- `::Int`: ID of the constituent space to which the element belongs.
"""
function get_space_id(us_space::UnstructuredSpace, element_id::Int)
    return findlast(us_space.us_config["patch_nels"] .< element_id)
end

"""
    get_max_local_dim(us_space::UnstructuredSpace)

Get the maximum local dimension across all constituent function spaces.

# Arguments
- `us_space::UnstructuredSpace`: The unstructured space.

# Returns
- `::Int`: The maximum local dimension.
"""
function get_max_local_dim(us_space::UnstructuredSpace)
    return maximum(get_max_local_dim.(us_space.function_spaces))
end

"""
    get_dof_partition(us_space::UnstructuredSpace{1,m}) where {m}

Get the dof partitioning for a 1D unstructured space.

# Arguments
- `us_space::UnstructuredSpace{1,m}`: The 1D unstructured space.

# Returns
- `::Vector{Vector{Int}}`: The dof partition.
"""
function get_dof_partition(us_space::UnstructuredSpace{1,m}) where {m}
    return us_space.dof_partition
end

"""
    get_local_basis(us_space::UnstructuredSpace{n,m}, element_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n,m}

Evaluate the local basis functions for a given element in the unstructured space.

# Arguments 
- `us_space::UnstructuredSpace`: The unstructured space.
- `element_id::Int`: The global element ID.
- `xi::NTuple{n,Vector{Float64}}`: Vector of element-normalized points (i.e., in [0,1]) where basis needs to be evaluated.
- `nderivatives::Int`: Number of derivatives to evaluate.

# Returns
- `::Array{Float64}`: Array of evaluated local basis (size: num_eval_points × num_funcs × (nderivatives+1)).
- `::Vector{Int}`: Vector of local basis indices (size: num_funcs).
"""
function get_local_basis(us_space::UnstructuredSpace{n,m}, element_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n,m}
    # Find the space ID and local element ID
    space_id = get_space_id(us_space, element_id)
    space_element_id = element_id - us_space.us_config["patch_nels"][space_id]

    # Evaluate the basis functions in the corresponding function space
    return evaluate(us_space.function_spaces[space_id], space_element_id, xi, nderivatives)[1]
end

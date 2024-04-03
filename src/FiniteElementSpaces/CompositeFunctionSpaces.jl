# UnstructuredSpace constructors

@doc raw"""
    UnstructuredSpace{n,m} <: AbstractFiniteElementSpace{n}

`n`-variate multi-patch space with `m` patches.

# Fields
- `function_spaces::NTuple{m, F} where {m, F <: AbstractFiniteElementSpace}`: collection of `m` (uni or multivariate) function spaces.
- `extraction_op::ExtractionOperator`: extraction operator that specifies how to combine functions from the `m` spaces into functions for the unstructure space.
- `us_config::Dict`: dictionary that stores some helper functionality (e.g., connectivity) for the unstructured space.
- `data::Dict`: any auxilliary data that the user wants to store for this unstructured space.
"""
struct UnstructuredSpace{n,m} <: AbstractFiniteElementSpace{n}
    function_spaces::NTuple{m, AbstractFiniteElementSpace{n}}
    extraction_op::ExtractionOperator
    us_config::Dict
    data::Dict

    function UnstructuredSpace(function_spaces::NTuple{m,AbstractFiniteElementSpace{n}}, extraction_op::ExtractionOperator, us_config::Dict, data::Dict) where {n,m}
        new{n,m}(function_spaces, extraction_op, us_config, data)
    end

    function UnstructuredSpace(function_spaces::NTuple{m,AbstractFiniteElementSpace{1}}, extraction_op::ExtractionOperator, data::Dict) where {m}
        # build 1D topology
        patch_neighbours = [-1 (1:m-1)...
                            (2:m)... -1]
        # number of elements per patch
        patch_nels = [0; cumsum([get_num_elements(function_spaces[i]) for i = 1:m])]

        # assemble patch config in a dictionary
        us_config = Dict("patch_neighbours" => patch_neighbours, "patch_nels" => patch_nels)

        new{1,m}(function_spaces, extraction_op, us_config, data)
    end
end

@doc raw"""
    get_num_elements(us_space::UnstructuredSpace)

Returns number of total number of elements for the partition over which the function space is defined.

# Arguments 
- `us_space::UnstructuredSpace`: unstructured space

# Returns
- `nel::Int`: total number of elements
"""
function get_num_elements(us_space::UnstructuredSpace)
    return get_num_elements(us_space.extraction_op)
end

@doc raw"""
    get_extraction(us_space::UnstructuredSpace, element_id::Int)

Get extraction coefficients and global basis indices for the global element id `element_id`. Note that the extraction coefficients define the unstructured space functions in terms of functions of the input spaces.

# Arguments 
- `us_space::UnstructuredSpace`: unstructured space
- `element_id::Int`: global element id

# Returns
- `extraction_coefficients::Array{Float64}`: extraction coefficients
- `basis_indices::Vector{Int}`: indies of global basis functions
"""
function get_extraction(us_space::UnstructuredSpace, element_id::Int)
    return get_extraction(us_space.extraction_op, element_id)
end

@doc raw"""
    get_space_id(us_space::UnstructuredSpace, element_id::Int)

Get id of contituent space to which `element_id` belongs.

# Arguments 
- `us_space::UnstructuredSpace`: unstructured space
- `element_id::Int`: global element id

# Returns
- `::Int`: id of constituent space to which `element_id`` belongs
"""
function get_space_id(us_space::UnstructuredSpace, element_id::Int)
    return findlast(us_space.us_config["patch_nels"] .< element_id)
end

@doc raw"""
    get_local_basis(us_space::UnstructuredSpace{1,m}, element_id::Int, xi::Vector{Float64}, nderivatives::Int) where {m}

For given global element id `element_id` for a given 1D unstructured space, find the right constituent space, evaluate and return it's basis.

# Arguments 
- `us_space::UnstructuredSpace`: unstructured space
- `element_id::Int`: global element id
- `xi::Vector{Float64}`: vector of element-normalized points (i.e., in [0,1]) where basis needs to be evaluated
- `nderivatives::Int`: number of derivatives to evaluate

# Returns
- `::Array{Float64}`: array of evaluated local basis (size: num_eval_points x num_funcs x nderivatives+1)
- `::Vector{Int}`: vector of local basis indices (size: num_funcs)
"""
function get_local_basis(us_space::UnstructuredSpace{1,m}, element_id::Int, xi::Vector{Float64}, nderivatives::Int) where {m}
    space_id = get_space_id(us_space, element_id)
    space_element_id = element_id - us_space.us_config["patch_nels"][space_id]
    return evaluate(us_space.function_spaces[space_id], space_element_id, xi, nderivatives)
end

@doc raw"""
    evaluate(us_space::UnstructuredSpace{1,m}, element_id::Int, xi::Vector{Float64}, nderivatives::Int) where {m}

For given global element id `element_id` for a given 1D unstructured space, evaluate the unstructured basis functions and return.

# Arguments 
- `us_space::UnstructuredSpace`: unstructured space
- `element_id::Int`: global element id
- `xi::Vector{Float64}`: vector of element-normalized points (i.e., in [0,1]) where basis needs to be evaluated
- `nderivatives::Int`: number of derivatives to evaluate

# Returns
- `::Array{Float64}`: array of evaluated global basis (size: num_eval_points x num_funcs x nderivatives+1)
- `::Vector{Int}`: vector of global basis indices (size: num_funcs)
"""
function evaluate(us_space::UnstructuredSpace{1,m}, element_id::Int, xi::Vector{Float64}, nderivatives::Int) where {m}
    extraction_coefficients, basis_indices = get_extraction(us_space, element_id)
    local_basis, _ = get_local_basis(us_space, element_id, xi, nderivatives)
    for r = 0:nderivatives
        local_basis[:,:,r+1] .= @views local_basis[:,:,r+1] * extraction_coefficients
    end

    return local_basis, basis_indices
end

function evaluate(us_space::UnstructuredSpace{1,m}, element_id::Int, xi::Float64, nderivatives::Int) where {m}
    return evaluate(us_space, element_id, [xi], nderivatives)
end

# TensorProductSpace constructors

"""
    TensorProductSpace{n} <: AbstractFiniteElementSpace{n} 

`n`-variate tensor-product space.

# Fields
- `patch::Patch{n}`: Patch on which the tensor product space is defined.
- `function_spaces::NTuple{m, F} where {m, F <: AbstractFiniteElementSpace}`: collection of uni or multivariate function spaces.
"""
struct TensorProductSpace{n} <: AbstractFiniteElementSpace{n} 
    patch::Mesh.Patch{n}
    function_spaces::NTuple{m, AbstractFiniteElementSpace} where {m}
    function TensorProductSpace(patch::Mesh.Patch{n}, function_spaces::NTuple{m, AbstractFiniteElementSpace}) where {n,m}
        if sum([get_n(function_spaces[i]) for i in 1:1:m]) != n
            throw(ArgumentError("The sum of the dimensions of the input spaces does not match the dimension of the patch!"))
        end
        new{n}(patch, function_spaces)
    end
end
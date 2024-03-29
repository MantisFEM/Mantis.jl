# MultiPatchSpace constructors

"""
    MultiPatchSpace{n, m} <: AbstractFunctionSpace{n}

`n`-variate multi-patch space with m patches

# Fields
- `function_spaces::NTuple{m, F} where {m, F <: AbstractFunctionSpace}`: collection of uni or multivariate function spaces.
"""
struct MultiPatchSpace{n,m} <: AbstractFunctionSpace{n} where {m}
    function_spaces::NTuple{m, AbstractFunctionSpace{n}}
    extraction_op::ExtractionOperator
    mp_config::Dict
    data::Dict

    function MultiPatchSpace(function_spaces::NTuple{m,AbstractFunctionSpace{1}}, extraction_op::ExtractionOperator, data::Dict) where {m}
        # build 1D topology
        patch_neighbours = [-1 (1:m-1)...
                            (2:m)... -1]
        # number of elements per patch
        patch_nels = [0; cumsum([get_num_elements(function_spaces[i]) for i = 1:m])]

        # assemble patch config in a dictionary
        mp_config = Dict("patch_neighbours" => patch_neighbours, "patch_nels" => patch_nels)

        new{1,m}(function_spaces, extraction_op, mp_config, data)
    end
end

# get total number of elements in multi-patch space
function get_num_elements(mp_space::MultiPatchSpace)
    return get_num_elements(mp_space.extraction_op)
end

# get extraction operator on a given element
function get_extraction(mp_space::MultiPatchSpace, element_id::Int)
    return get_extraction(mp_space.extraction_op, element_id)
end

# get patch_id for a given global element_id
function get_patch_id(mp_space::MultiPatchSpace, element_id::Int)
    return findlast(mp_space.mp_config["patch_nels"] .< element_id)
end

# get patch_local basis evaluated for a given global element_id
function get_local_basis(mp_space::MultiPatchSpace{1,m}, element_id::Int, xi::Vector{Float64}, nderivatives::Int) where {m}
    patch_id = get_patch_id(mp_space, element_id)
    patch_element_id = element_id-mp_space.mp_config["patch_nels"][patch_id]
    return evaluate(mp_space.function_spaces[patch_id], patch_element_id, xi, nderivatives)
end

# evaluate the multi-patch basis for a given global element_id
function evaluate(mp_space::MultiPatchSpace{1,m}, element_id::Int, xi::Vector{Float64}, nderivatives::Int) where {m}
    extraction_coefficients, basis_indices = get_extraction(mp_space, element_id)
    local_basis, _ = get_local_basis(mp_space, element_id, xi, nderivatives)
    for r = 0:nderivatives
        local_basis[:,:,r+1] .= @views local_basis[:,:,r+1] * extraction_coefficients
    end

    return local_basis, basis_indices
end

function evaluate(mp_space::MultiPatchSpace{1,m}, element_id::Int, xi::Float64, nderivatives::Int) where {m}
    return evaluate(mp_space, element_id, [xi], nderivatives)
end

# TensorProductSpace constructors

"""
    TensorProductSpace{n} <: AbstractFunctionSpace{n} 

`n`-variate tensor-product space.

# Fields
- `patch::Patch{n}`: Patch on which the tensor product space is defined.
- `function_spaces::NTuple{m, F} where {m, F <: AbstractFunctionSpace}`: collection of uni or multivariate function spaces.
"""
struct TensorProductSpace{n} <: AbstractFunctionSpace{n} 
    patch::Mesh.Patch{n}
    function_spaces::NTuple{m, AbstractFunctionSpace} where {m}
    function TensorProductSpace(patch::Mesh.Patch{n}, function_spaces::NTuple{m, AbstractFunctionSpace}) where {n,m}
        if sum([get_n(function_spaces[i]) for i in 1:1:m]) != n
            throw(ArgumentError("The sum of the dimensions of the input spaces does not match the dimension of the patch!"))
        end
        new{n}(patch, function_spaces)
    end
end
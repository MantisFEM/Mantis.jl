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
    data::Dict
end

# methods for multipatch space
# 1. iterating over elements
# 2. getters for extraction operators
# 3. getters for local basis

function get_num_elements(mp_space::MultiPatchSpace)
    return get_num_elements(mp_space.extraction_op)
end

function get_extraction(mp_space::MultiPatchSpace, element_id::Int)
    return get_extraction(mp_space.extraction_op, element_id)
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

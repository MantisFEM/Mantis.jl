"""
    TensorProductSpace{m} <: AbstractFiniteElementSpace

A tensor-product space that is built as the tensor-product of `m` input spaces. The functions in this space are `n`-variate where `n` is equal to the sum of all `n_i` such that the i-th input space is `n_i`-variate.

# Fields
- `function_spaces::NTuple{m,AbstractFiniteElementSpace} where {m}`: collection of uni or multivariate function spaces
- `data::Dict`: any relevant data that the user wants to store
"""
struct TensorProductSpace{n,m} <: AbstractFiniteElementSpace{n} where {m}
    function_spaces::NTuple{m, AbstractFiniteElementSpace}
    data::Dict

    function TensorProductSpace(function_spaces::NTuple{m, AbstractFiniteElementSpace}, data::Dict) where {m}
        n = get_n.(function_spaces)
        new{n,m}(function_spaces, data)
    end
end

@doc raw"""
    get_num_elements(tp_space::TensorProductSpace)

Returns number of total number of elements for the partition over which the function space is defined.

# Arguments 
- `tp_space::TensorProductSpace`: tensor-product space

# Returns
- `nel::Int`: total number of elements
"""
function get_num_elements(tp_space::TensorProductSpace)
    return prod(get_num_elements.(tp_space.function_spaces))
end

"""
    get_dim(tp_space::TensorProductSpace)

Returns the dimension of the tensor-product function space `tp_space`.

# Arguments
- `tp_space::TensorProductSpace)`: Tensor-product function space.
# Returns
- `::Int`: The dimension of the space.
"""
function get_dim(tp_space::TensorProductSpace)
    return prod(get_dim.(tp_space.function_spaces))
end

@doc raw"""
    ordered_to_linear_index(ord_ind::Vector{Int}, max_ind::Vector{Int})

Convert given ordered pair index to linear index:
linear_index = ordered_pair[1] + (ord_ind[2]-1)*max_ind[1] + ...

# Arguments 
- `ord_ind::Vector{Int}`: ordered index pair
- `max_ind::Vector{Int}`: maximum index in each direction

# Returns
- `lin_ind::Int`: computed linear index
"""
function ordered_to_linear_index(ord_ind::Vector{Int}, max_ind::Vector{Int})
    m = length(max_ind)
    @assert length(ord_ind)==m "Dimension mismatch."
    max_ind .= cumprod(max_ind)
    lin_ind = ord_ind[1]
    for i = 2:m
        lin_ind += (ord_ind[i]-1)*max_ind[i-1]
    end

    return lin_ind
end

function ordered_to_linear_index(ord_ind::Tuple{m,Int}, max_ind::Tuple{m,Int}) where {m}
    return ordered_to_linear_index(collect(ord_ind), collect(max_ind))
end

@doc raw"""
    linear_to_ordered_index(, max_ind::Vector{Int})

Convert given linear index to ordered index pair such that:
linear_index = ordered_pair[1] + (ord_ind[2]-1)*max_ind[1] + ...

# Arguments 
- `lin_ind::Int`: computed linear index
- `max_ind::Vector{Int}`: maximum index in each direction

# Returns
- `ord_ind::Vector{Int}`: ordered index pair
"""
function linear_to_ordered_index(lin_ind::Int, max_ind::Vector{Int})
    m = length(max_ind)
    ord_ind = zeros(Int,m)
    ord_ind[1] = lin_ind - floor((lin_ind-1)/max_ind[1])*max_ind[1]
    lin_ind = (lin_ind - ord_ind[1])/max_ind[1]
    for i = 2:m
        ord_ind[i] = lin_ind - floor(lin_ind/max_ind[i])*max_ind[i] + 1
        lin_ind = (lin_ind - ord_ind[i] + 1)/max_ind[i]
    end
    @assert lin_ind==0 "Ordered pair computation failed."
    
    return Int.(ord_ind)
end

function linear_to_ordered_index(lin_ind::Int, max_ind::Tuple{m,Int}) where {m}
    return linear_to_ordered_index(lin_ind, collect(max_ind))
end

@doc raw"""
    get_local_basis(tp_space::TensorProductSpace{n,m}, element_id::Int, xi::NTuple{m,Vector{Float64}}, nderivatives::Vector{Int}) where {n,m}

For given global element id `element_id` for a given tensor-product space, evaluate and return the constituent basis functions.

# Arguments
- `tp_space::TensorProductSpace`: tensor-product space
- `element_id::Int`: global element index
- `xi::NTuple{m,Vector{Float64}}`: vector of element-normalized points (i.e., in [0,1]) where basis needs to be evaluated
- `nderivatives::Vector{Int}`: number of derivatives to evaluate in each direction

# Returns
- `::NTuple{m,F}`: where `F` is a Tuple containing evaluated local basis and local basis indices
"""
function get_local_basis(tp_space::TensorProductSpace{n,m}, element_id::Int, xi::NTuple{m,Vector{Float64}}, nderivatives::Vector{Int}) where {n,m}
    # convert linear index ordered index pair
    ord_element_id = linear_to_ordered_index(element_id, get_num_elements.(tp_space.function_spaces))
    
    return evaluate.(tp_space.function_spaces, ord_element_id, xi, nderivatives)
end

@doc raw"""
    evaluate(tp_space::TensorProductSpace{n,m}, element_id::Int, xi::NTuple{m,Vector{Float64}}) where {n,m}

For given global element id `element_id` for a given tensor-product space, evaluate the tensor-product basis functions and return.

# Arguments 
- `tp_space::TensorProductSpace`: tensor-product space
- `element_id::Int`: global element id
- `xi::NTuple{m,Vector{Float64}}`: vector of element-normalized points (i.e., in [0,1]) where basis needs to be evaluated

# Returns
- `::Array{Float64}`: array of evaluated global basis (size: num_eval_points x num_funcs)
- `::Vector{Int}`: vector of global basis indices (size: num_funcs)
"""
function evaluate(tp_space::TensorProductSpace{n,m}, element_id::Int, xi::NTuple{m,Vector{Float64}}) where {n,m}
    # kronecker product of constituent basis functions
    all_local_basis = get_local_basis(tp_space, element_id, xi, zeros(Int,m))
    local_basis = all_local_basis[1][1][:,:,1]
    for i = 2:m
        local_basis = kron(all_local_basis[i][1][:,:,1], local_basis)
    end
    
    # get dimension of each constituent space
    space_dims = get_dim.(tp_space.function_spaces)
    # linear indexing for basis function indices
    basis_indices = Iterators.product((all_local_basis[i][2] for i = 1:m)...)
    basis_indices = reshape([ordered_to_linear_index(binds, space_dims) for binds in basis_indices], length(basis_indices))

    return local_basis, basis_indices
end
"""
    AbstractFiniteElementSpace

Supertype for all scalar finite element spaces.
"""
abstract type AbstractFiniteElementSpace{manifold_dim} <: AbstractFunctionSpace end

# Getters for the function spaces
get_manifold_dim(_::AbstractFiniteElementSpace{manifold_dim}) where {manifold_dim} = manifold_dim

"""
    get_basis_indices(space::AbstractFiniteElementSpace, element_id::Int)

Get the global indices of the basis functions of the finite element space `space` for the element with index `element_id`.

# Arguments
- `space::AbstractFiniteElementSpace`: Finite element space
- `element_id::Int`: Index of the element

# Returns
- `::Vector{Int}`: Global indices of the basis functions
"""
function get_basis_indices(space::AbstractFiniteElementSpace, element_id::Int)
    return get_basis_indices(space.extraction_op, element_id)
end

@doc raw"""
    evaluate(space::S, element_id::Int, xi::Vector{Float64}, nderivatives::Int) where {S<:AbstractFiniteElementSpace}

For given global element id `element_id` for a given finite element `space`, evaluate the local basis functions and return.

# Arguments 
- `space<:AbstractFiniteElementSpace`: finite element space.
- `element_id::Int`: global element id.
- `xi::Vector{Float64}`: vector of element-normalized points (i.e., in [0,1]) where basis needs to be evaluated.
- `nderivatives::Int`: number of derivatives to evaluate.

# Returns
- `::Matrix{Float64}`: array of evaluated global basis (size: num_eval_points x num_funcs x nderivatives+1)
- `::Vector{Int}`: vector of global basis indices (size: num_funcs).

# See also [`_get_derivative_idx(der_key::Vector{Int})`] to understand how evaluations are stored
"""
function evaluate(space::AbstractFiniteElementSpace{n}, element_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int) where {n}
    extraction_coefficients, basis_indices = get_extraction(space, element_id)
    local_basis = get_local_basis(space, element_id, xi, nderivatives)

    for j = 0:nderivatives
        for k = 1:length(local_basis[j+1])
            if isassigned(local_basis[j+1],k)
                local_basis[j+1][k] = @views local_basis[j+1][k] * extraction_coefficients
            end
        end
    end

    return local_basis, basis_indices
end

function evaluate(space::AbstractFiniteElementSpace{n}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n}
    return evaluate(space, element_id, xi, 0)
end

function evaluate(space::AbstractFiniteElementSpace{n}, element_id::Int, xi::NTuple{n,Vector{Float64}}, nderivatives::Int, coeffs::Vector{Float64}) where {n}
    local_basis, basis_indices = evaluate(space, element_id, xi, nderivatives) 
    evaluation = Vector{Vector{Vector{Float64}}}(undef, nderivatives + 1)
    for j = 0:nderivatives
        n_ders = length(local_basis[j+1])
        evaluation[j+1] = Vector{Vector{Float64}}(undef, n_ders)
    end

    for j = 0:nderivatives
        for k = 1:length(local_basis[j+1])
            if isassigned(local_basis[j+1],k)
                evaluation[j+1][k] = @views local_basis[j+1][k] * coeffs[basis_indices]
            end
        end
    end

    return evaluation
end

"""
    _evaluate_all_at_point(fem_space::AbstractFiniteElementSpace{1}, element_id::Int, xi::Float64, nderivatives::Int)

Evaluates all derivatives upto order `nderivatives` for all basis functions of `fem_space` at a given point `xi` in the element `element_id`.

# Arguments
- `fem_space::AbstractFiniteElementSpace{1}`: A univariate FEM space.
- `element_id::Int`: The id of the element.
- `xi::Float64`: The point where all global basis functiuons are evaluated.
- `nderivatives::Int`: The order upto which derivatives need to be computed.
# Returns
- `::SparseMatrixCSC{Float64}`: Global basis functions, size = n_dofs x nderivatives+1
"""
function _evaluate_all_at_point(fem_space::AbstractFiniteElementSpace{1}, element_id::Int, xi::Float64, nderivatives::Int)
    local_basis, basis_indices = evaluate(fem_space, element_id, ([xi],), nderivatives)
    nloc = length(basis_indices)
    ndofs = get_num_basis(fem_space)
    I = zeros(Int, nloc * (nderivatives + 1))
    J = zeros(Int, nloc * (nderivatives + 1))
    V = zeros(Float64, nloc * (nderivatives + 1))
    count = 0
    for r = 0:nderivatives
        for i = 1:nloc
            I[count+1] = basis_indices[i]
            J[count+1] = r+1
            V[count+1] = local_basis[r+1][1][1, i]
            count += 1
        end
    end

    return SparseArrays.sparse(I,J,V,ndofs,nderivatives+1)
end
@doc raw"""
    CanonicalFiniteElementSpace <: AbstractFiniteElementSpace{1}

Wrapper that allows treating a canonical space as a finite element space.

# Fields
- `canonical_space::AbstractCanonicalSpace` : canonical space.
"""
struct CanonicalFiniteElementSpace <: AbstractFiniteElementSpace{1}
    canonical_space::AbstractCanonicalSpace
end

function get_polynomial_degree(space::CanonicalFiniteElementSpace)
    return space.canonical_space.p
end

function get_num_elements(::CanonicalFiniteElementSpace)
    return 1
end

function get_dim(space::CanonicalFiniteElementSpace)
    return space.canonical_space.p+1
end

"""
evaluate_all_at_point(canonical_space::CanonicalFiniteElementSpace, xi::Float64, nderivatives::Int)

Evaluates all derivatives upto order `nderivatives` for all `canonical_space` basis functions at a given point `xi`.

# Arguments
- `canonical_space::AbstractCanonicalSpace`: A canonical function space.
- `xi::Float64`: The point where all global basis functiuons are evaluated.
- `nderivatives::Int`: The order upto which derivatives need to be computed.
# Returns
- `::SparseMatrixCSC{Float64}`: Global basis functions, size = n_dofs x nderivatives+1
"""
function evaluate_all_at_point(space::CanonicalFiniteElementSpace, xi::Float64, nderivatives::Int)
    local_basis = evaluate(space.canonical_space, xi, nderivatives)
    ndofs = get_dim(space)
    basis_indices = 1:ndofs
    nloc = length(basis_indices)
    I = zeros(Int, nloc * (nderivatives + 1))
    J = zeros(Int, nloc * (nderivatives + 1))
    V = zeros(Float64, nloc * (nderivatives + 1))
    count = 0
    for r = 0:nderivatives
        for i = 1:nloc
            I[count+1] = basis_indices[i]
            J[count+1] = r+1
            V[count+1] = local_basis[1, i, r+1]
            count += 1
        end
    end
    return SparseArrays.sparse(I,J,V,ndofs,nderivatives+1)
end

function evaluate_all_at_point(space::CanonicalFiniteElementSpace, ::Int, xi::Float64, nderivatives::Int)
    return evaluate_all_at_point(space, xi, nderivatives)
end

function evaluate(space::CanonicalFiniteElementSpace, ::Int, xi::Vector{Float64}, nderivatives::Int)
    return evaluate(space.canonical_space, xi, nderivatives), collect(1:get_dim(space))
end
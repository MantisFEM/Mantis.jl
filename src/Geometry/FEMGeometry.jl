struct FEMGeometry{n,m} <: AbstractGeometry{n, m}
    geometry_coeffs::Array{Float64,2}
    fem_space::FunctionSpaces.AbstractFiniteElementSpace{n}
    n_elements::Int

    function FEMGeometry(fem_space::FunctionSpaces.AbstractFiniteElementSpace{n}, geometry_coeffs::Array{Float64,2}) where {n}
        m = size(geometry_coeffs,2)
        n_elements = FunctionSpaces.get_num_elements(fem_space)
        return new{n,m}(geometry_coeffs, fem_space, n_elements)
    end
end

function get_num_elements(geometry::FEMGeometry{n,m}) where {n,m}
    return geometry.n_elements
end

function get_domain_dim(::FEMGeometry{n,m}) where {n, m}
    return n
end

function get_image_dim(::FEMGeometry{n,m}) where {n, m}
    return m
end

# # evaluate at matrix containing n-dimensional parametric points
# function evaluate(geometry::FEMGeometry{n,m}, element_idx::Int, ξ::Matrix{Float64}) where {n,m}
#     return evaluate.(geometry, element_idx, ntuple(i -> ntuple(j -> [ξ[i,j]], n), size(ξ,1)))
# end

# evaluate at a single n-dimensional parametric point
function evaluate(geometry::FEMGeometry{n,m}, element_idx::Int, ξ::Vector{Float64}) where {n,m}
    @assert length(ξ) == n "Dimension mismatch"
    return vec(evaluate(geometry, element_idx, ntuple(i -> [ξ[i]], n)))
end

function evaluate(geometry::FEMGeometry{1,m}, element_idx::Int, ξ::Float64) where {m}
    return evaluate(geometry, element_idx, [ξ])
end

# evaluate in each direction at the specific points in the ntuple
function evaluate(geometry::FEMGeometry{n,m}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n,m}
    # evaluate fem space
    fem_basis, fem_basis_indices = FunctionSpaces.evaluate(geometry.fem_space, element_id, xi, 0)
    # combine with coefficients and return
    key = Tuple(zeros(Float64,n))
    return fem_basis[key...] * geometry.geometry_coeffs[fem_basis_indices,:]
end

function jacobian(geometry::FEMGeometry{n,m}, element_idx::Int, ξ::Matrix{Float64}) where {n,m}
    return jacobian.(geometry, element_idx, ntuple(i -> ntuple(j -> [ξ[i,j]], n), size(ξ,1)))
end

function jacobian(geometry::FEMGeometry{n,m}, element_idx::Int, ξ::Vector{Float64}) where {n,m}
    @assert length(ξ) == n "Dimension mismatch"
    return collect(jacobian(geometry, element_idx, ntuple(i -> [ξ[i]], n))[1])
end

import LinearAlgebra

function jacobian(geometry::FEMGeometry{n,m}, element_id::Int, xi::NTuple{n,Vector{Float64}}) where {n,m}
    # evaluate fem space
    fem_basis, fem_basis_indices = FunctionSpaces.evaluate(geometry.fem_space, element_id, xi, 1)
    # combine with coefficients and return
    keys = Matrix{Float64}(LinearAlgebra.I,n,n)
    n_eval_points = prod(length.(xi))
    J = zeros(n_eval_points, m, n)
    for k = 1:n 
        J[:, :, k] .= fem_basis[Tuple(keys[k,:])...] * geometry.geometry_coeffs[fem_basis_indices,:]
    end
    return J #, fem_basis[Tuple(zeros(Float64,n))...] * geometry.geometry_coeffs[fem_basis_indices,:]
end
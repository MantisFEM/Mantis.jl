struct CartesianGeometry{n,n} <: AbstractAnalGeometry
    n_elements::NTuple{n,Int}
    breakpoints::NTuple{n,Vector{Float64}}
    cartesian_idxs::CartesianIndices

    function CartesianGeometry(breakpoints::NTuple{n,Vector{Float64}}) where {n}
        n_elements = length.(breakpoints) .- 1
        cartesian_idxs = CartesianIndices(n_elements)
        return new{n,n}(n_elements, breakpoints, cartesian_idxs)
    end

end

function evaluate(geometry::CartesianGeometry{n,n}, element_idx::Int, ξ::Matrix{Float64}) where {n}
    return evaluate.(geometry, element_idx, ntuple(i -> ntuple(j -> [ξ[i,j]], n), size(ξ,1)))
end

function evaluate(geometry::CartesianGeometry{n,n}, element_idx::Int, ξ::Vector{Float64}) where {n}
    @assert length(ξ) == n "Dimension mismatch"
    return collect(evaluate(geometry, element_idx, ntuple(i -> [ξ[i]], n))[1])
end

function evaluate(geometry::CartesianGeometry{n,n}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n}
    ordered_idx = Tuple(geometry.cartesian_idxs[element_idx])
    univariate_points = ntuple( k -> (1 .- ξ[k]) .* geometry.breakpoints[k][ordered_idx[k]] + ξ[k] .* geometry.breakpoints[k][ordered_idx[k]+1], n)
    return collect(Iterators.product(univariate_points...))
end

function jacobian(geometry::CartesianGeometry{n,n}, element_idx::Int, ξ::Matrix{Float64}) where {n}
    return jacobian(geometry, element_idx, ntuple(i -> ntuple(j -> [ξ[i,j]], n), size(ξ,1)))
end

function jacobian(geometry::CartesianGeometry{n,n}, element_idx::Int, ξ::Vector{Float64}) where {n}
    @assert length(ξ) == n "Dimension mismatch"
    return collect(jacobian(geometry, element_idx, ntuple(i -> [ξ[i]], n))[1])
end

function jacobian(geometry::CartesianGeometry{n,n}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n}
    ordered_idx = Tuple(geometry.cartesian_idxs[element_idx])
    dx = prod(geometry.breakpoints[k][ordered_idx[k]+1] - geometry.breakpoints[k][ordered_idx[k]] for k = 1:n)
    return dx * ones(Float64, prod(length.(ξ)))
end
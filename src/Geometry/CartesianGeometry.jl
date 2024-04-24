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

function evaluate(geometry::CartesianGeometry{n,n}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n}
    ordered_idx = Tuple(geometry.cartesian_idxs[element_idx])
    univariate_points = ((1 - ξ[k]) .* geometry.breakpoints[ordered_idx[k]] + ξ[k] .* geometry.breakpoints[ordered_idx[k]+1] for k = 1:n)
    return collect(Iterators.product(univariate_points...))
end

function jacobian(geometry::CartesianGeometry{n,n}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n}
    ordered_idx = Tuple(geometry.cartesian_idxs[element_idx])
    dx = 1.0
    for k = 1:n
        dx *= geometry.breakpoints[k][ordered_idx[k]+1] - geometry.breakpoints[k][ordered_idx[k]]
    end
    return dx * ones(Float64, prod(length.(ξ)))
end
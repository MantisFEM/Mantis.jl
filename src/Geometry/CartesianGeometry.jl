struct CartesianGeometry{n,n} <: AbstractAnalGeometry{n, n}
    n_elements::NTuple{n,Int}
    breakpoints::NTuple{n,Vector{Float64}}
    cartesian_idxs::CartesianIndices

    function CartesianGeometry(breakpoints::NTuple{n,Vector{Float64}}) where {n}
        n_elements = length.(breakpoints) .- 1
        cartesian_idxs = CartesianIndices(n_elements)
        return new{n,n}(n_elements, breakpoints, cartesian_idxs)
    end

end

function get_num_elements(geometry::CartesianGeometry{n,n}) where {n}
    return prod(geometry.n_elements)
end

function get_domain_dim(geometry::CartesianGeometry{n,n}) where {n}
    return n
end

function get_image_dim(geometry::CartesianGeometry{n,n}) where {n}
    return n
end

function evaluate(geometry::CartesianGeometry{n,n}, element_idx::Int, ξ::Matrix{Float64}) where {n}
    return evaluate.(geometry, element_idx, ntuple(i -> ntuple(j -> [ξ[i,j]], n), size(ξ,1)))
end

function evaluate(geometry::CartesianGeometry{n,n}, element_idx::Int, ξ::Vector{Float64}) where {n}
    @assert length(ξ) == n "Dimension mismatch"
    return reshape(evaluate(geometry, element_idx, ntuple(i -> [ξ[i]], n)), :)  # return as a vector since there is only one point (one column)
end

function evaluate(geometry::CartesianGeometry{n,n}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n}
    ordered_idx = Tuple(geometry.cartesian_idxs[element_idx])
    univariate_points = ntuple( k -> (1 .- ξ[k]) .* geometry.breakpoints[k][ordered_idx[k]] + ξ[k] .* geometry.breakpoints[k][ordered_idx[k]+1], n)
    
    points_tensor_product_idx = CartesianIndices(size.(univariate_points, 1))  # get the multidimensional indices to index the tensor product points

    # Loop over the points and compute their coordinates as the tensor product of the unidimensional points
    n_points = prod(size.(ξ, 1))  # the total number of points to evaluate, it is a tensor product of the coordinates to sample in each direction
    x = zeros(Float64, n, n_points)
    for (point_idx, point_cartesian_idx) in enumerate(points_tensor_product_idx)
        for component_idx in 1:n 
            x[component_idx, point_idx] = univariate_points[component_idx][point_cartesian_idx[component_idx]]
        end    
    end

    return x
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
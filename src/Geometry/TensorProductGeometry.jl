
struct TensorProductGeometry{n, m} <: AbstractGeometry{n, m}
    geometry_1::AbstractGeometry
    geometry_2::AbstractGeometry
    domain_dims::NTuple{2, Int}
    image_dims::NTuple{2, Int}
    n_elements_per_geometry::NTuple{2, Int}
    n_elements::Int
    cartesian_indices::CartesianIndices
    
    function TensorProductGeometry(geometry_1::AbstractGeometry{n_1, m_1}, geometry_2::AbstractGeometry{n_2, m_2}) where {n_1, m_1, n_2, m_2}
        n_elements_per_geometry = (get_num_elements(geometry_1), get_num_elements(geometry_2))
        n_elements = prod(n_elements_per_geometry)
        domain_dims = (get_domain_dim(geometry_1), get_domain_dim(geometry_2))
        image_dims = (get_image_dim(geometry_1), get_image_dim(geometry_2))
        n = sum(domain_dims)
        m = sum(image_dims) 
        cartesian_indices = CartesianIndices(n_elements_per_geometry)
        return new{n, m}(geometry_1, geometry_2, domain_dims, image_dims, n_elements_per_geometry, n_elements, cartesian_indices)
    end
end

function get_num_elements(geometry::TensorProductGeometry{n, m}) where {n, m}
    return geometry.n_elements
end

function get_domain_dim(::TensorProductGeometry{n, m}) where {n, m}
    return n
end

function get_image_dim(::TensorProductGeometry{n, m}) where {n, m}
    return m
end

function evaluate(geometry::TensorProductGeometry{n, m}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n, m}
    element_geometries_idx = geometry.cartesian_indices[element_idx]
    
    n_points = prod(size.(ξ, 1))  # the total number of points to evaluate, it is a tensor product of the coordinates to sample in each direction
    x = zeros(Float64, n_points, m)

    x[:, 1:geometry.image_dims[1]] .= evaluate(geometry.geometry_1, element_geometries_idx[1], ξ[1:geometry.domain_dims[1]])
    x[:, (geometry.image_dims[1]+1):end] .= evaluate(geometry.geometry_2, element_geometries_idx[2], ξ[(geometry.domain_dims[1]+1):end])
    
    return x
end

function evaluate(geometry::TensorProductGeometry{n, m}, element_idx::Int, ξ::Vector{Float64}) where {n, m}
    return evaluate(geometry, element_idx, ntuple(i -> [ξ[i]], n))[1, :]
end

function jacobian(geometry::TensorProductGeometry{n, m}, element_idx::Int, ξ::NTuple{n,Vector{Float64}}) where {n, m}
    element_geometries_idx = geometry.cartesian_indices[element_idx]

    n_points = prod(size.(ξ, 1))  # the total number of points to evaluate, it is a tensor product of the coordinates to sample in each direction
    J = zeros(Float64, n_points, m, n)

    J[:, 1:geometry.image_dims[1], 1:geometry.domain_dims[1]] .= jacobian(geometry.geometry_1, element_geometries_idx[1], ξ[1:geometry.domain_dims[1]])  # the Jacobian contribution from geometry 1
    J[:, (geometry.image_dims[1]+1):end, (geometry.domain_dims[1]+1):end] .= jacobian(geometry.geometry_2, element_geometries_idx[2], ξ[(geometry.domain_dims[1]+1):end])  # the Jacobian contribution from geometry 2
    
    return J
end

function jacobian(geometry::TensorProductGeometry{n, m}, element_idx::Int, ξ::Vector{Float64}) where {n, m}
    return jacobian(geometry, element_idx, ntuple(i -> [ξ[i]], n))
end

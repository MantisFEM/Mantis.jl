
struct TensorProductGeometry{n, m} <: AbstractGeometry{n, m}
    geometries::Vector{AbstractGeometry}
    domain_dims::Vector{Int}
    image_dims::Vector{Int}
    n_elements_per_geometry::Vector{Int}
    n_elements::Int
    
    function TensorProductGeometry(geometries::Vector{AbstractGeometry})
        n_elements_per_geometry = get_num_elements.(geometries)
        n_elements = prod(n_elements_per_geometry)
        domain_dims = get_domain_dim.(geometries)
        image_dims = get_image_dim.(geometries)
        n = sum(domain_dims)
        m = sum(image_dims)
        
        return new{n, m}(geometries, domain_dims, image_dims, n_elements_per_geometry, n_elements)
    end
end

function get_num_elements(geometry::TensorProductGeometry{n, m}) where {n, m}
    return n_elements
end

function get_domain_dim(geometry::TensorProductGeometry{n, m}) where {n, m}
    return n
end

function get_image_dim(geometry::TensorProductGeometry{n, m}) where {n, m}
    return m
end

# function evaluate(geometry::TensorProductGeometry{n, m}, element_idx::Int, ξ::Vector{Float64}) where {n, m}
#     x = evaluate(geometry.geometry, element_idx, ξ)
#     x_mapped = evaluate(geometry.mapping, x)
#     return x_mapped
# end

# function jacobian(geometry::MappedGeometry{n, m}, element_idx::Int, ξ::Vector{Float64}) where {n, m}
#     J_1 = jacobian(geometry.geometry, element_idx, ξ)  # the Jacobian for the mapping from the elements to base geometry image
#     x = evaluate(geometry, element_idx, ξ)
#     J_2 = jacobian(geometry.mapping, x)  # the mapping from the image of the  base geometry to the image of the mapping
#     return J_2 * J_1
# end


struct Mapping{n, m}
    mapping::Function
    dmapping::Function
end

function evaluate(mapping::Mapping{n, m}, x::Vector{Float64}) where {n, m}
    return mapping.mapping(x)
end

function jacobian(mapping::Mapping{n, m}, x::Vector{Float64}) where {n, m}
    return mapping.dmapping(x)
end

struct MappedGeometry{n, m} <: AbstractGeometry{n, m}
    geometry::AbstractGeometry
    mapping::Mapping
    n_elements::Int
end

function MappedGeometry(geometry::AbstractGeometry{n, k}, mapping::Mapping{k, m}) where {n, k, m}
    n_elements = geometry.n_elements
    return MappedGeometry{n, m}(geometry, mapping, n_elements)
end

function evaluate(geometry::MappedGeometry{n, m}, element_idx::Int, ξ::Vector{Float64}) where {n, m}
    x = evaluate(geometry.geometry, element_idx, ξ)
    x_mapped = evaluate(geometry.mapping, x)
    return x_mapped
end

function jacobian(geometry::MappedGeometry{n, m}, element_idx::Int, ξ::Vector{Float64}) where {n, m}
    J_1 = jacobian(geometry.geometry, element_idx, ξ)  # the Jacobian for the mapping from the elements to base geometry image
    x = evaluate(geometry, element_idx, ξ)
    J_2 = jacobian(geometry.mapping, x)  # the mapping from the image of the  base geometry to the image of the mapping
    return J_2 * J_1
end

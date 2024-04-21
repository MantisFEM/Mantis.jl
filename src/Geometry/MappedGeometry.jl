
struct Mapping
    dimension::NTuple{2, Int}
    mapping::Function
    dmapping::Function
end

function evaluate(mapping::Mapping, x::Vector{Float64})
    return mapping.mapping(x)
end

function jacobian(mapping::Mapping, x::Vector{Float64})
    return mapping.dmapping(x)
end

struct MappedGeometry <: AbstractGeometry
    dimension::NTuple{2, Int}
    geometry::AbstractGeometry
    mapping::Mapping
    n_elements::Int
end

function MappedGeometry(geometry::AbstractGeometry, mapping::Mapping)
    dimension = (geometry.dimension[1], mapping.dimension[2])  # maps from the canonical element to the image of mapping
    n_elements = geometry.n_elements
    return MappedGeometry(dimension, geometry, mapping, n_elements)
end

function evaluate(geometry::MappedGeometry, element_idx::Int, ξ::Vector{Float64})
    x = evaluate(geometry.geometry, element_idx, ξ)
    x_mapped = evaluate(geometry.mapping, x)
    return x_mapped
end

function jacobian(geometry::MappedGeometry, element_idx::Int, ξ::Vector{Float64})
    J_1 = jacobian(geometry.geometry, element_idx, ξ)  # the Jacobian for the mapping from the elements to base geometry image
    x = evaluate(geometry, element_idx, ξ)
    J_2 = jacobian(geometry.mapping, x)  # the mapping from the image of the  base geometry to the image of the mapping
    return J_2 * J_1
end

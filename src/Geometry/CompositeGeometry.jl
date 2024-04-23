
struct CompositeGeometry <: AbstractGeometry
    dimension::NTuple{2, Int}
    geometries::NTuple{2, AbstractGeometry}
    n_elements::Int
end

function CompositeGeometry(Φ_1::AbstractGeometry, Φ_2::AbstractGeometry)
    # Φ = Φ_2  ̊ Φ_1
    dimension = (Φ_1.dimension[1], Φ_2.dimension[2])
    n_elements = Φ_1.n_elements  # the number of elements need to be the same
    return CompositeGeometry(dimension, (Φ_1, Φ_2), n_elements)
end

function evaluate(geometry::CompositeGeometry, element_idx::Int, ξ::Vector{Float64})
    x = evaluate(Φ_2.geometry, element_idx, ξ)
    return x
end

function jacobian(geometry::CompositeGeometry, element_idx::Int, ξ::Vector{Float64})
    J_1 = jacobian(geometry.Φ_1, element_idx, ξ)  # the Jacobian for the mapping from the elements to base geometry image
    J_2 = jacobian(geometry.Φ_2, element_idx, ξ)  # the mapping from the image of the  base geometry to the image of the mapping
    return J_2 * J_1
end

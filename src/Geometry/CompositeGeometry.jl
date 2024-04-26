
struct CompositeGeometry{n, m} <: AbstractGeometry{n, m}
    dimension::NTuple{2, Int}
    Φ_1::AbstractGeometry
    Φ_2::AbstractGeometry
    n_elements::Int

    function CompositeGeometry{n, m}(Φ_1::AbstractGeometry{n, k}, Φ_2::AbstractGeometry{k, m}) where {n, k, m}
        # Φ = Φ_2  ̊ Φ_1
        dimension = (Φ_1.dimension[1], Φ_2.dimension[2])
        n_elements = Φ_1.n_elements  # the number of elements need to be the same
        return new{n, m}(dimension, Φ_1, Φ_2, n_elements)
    end

    function CompositeGeometry(Φ_1::AbstractGeometry{n, k}, Φ_2::AbstractGeometry{k, m}) where {n, k, m}
        # Φ = Φ_2  ̊ Φ_1
        dimension = (Φ_1.dimension[1], Φ_2.dimension[2])
        n_elements = Φ_1.n_elements  # the number of elements need to be the same
        return new{n, m}(dimension, Φ_1, Φ_2, n_elements)
    end
end

function evaluate(geometry::CompositeGeometry{n, m}, element_idx::Int, ξ::Vector{Float64}) where {n, m}
    # We only need to evaluate the last geometry because it is required that Φ₂ is defined as
    # Φ₂ = Φ̃₂ ̊Φ₁⁻¹
    # with
    # Φ₂: Ω₁ ↦ Ω₂
    # Φ₁: Ω₀ ↦ Ω₁
    # Φ̃₂: Ω₀ ↦ Ω₂  
    x = evaluate(Φ_2.geometry, element_idx, ξ)
    return x
end

function jacobian(geometry::CompositeGeometry{n, m}, element_idx::Int, ξ::Vector{Float64}) where {n, m}
    J_1 = jacobian(geometry.Φ_1, element_idx, ξ)  # the Jacobian for the mapping from the elements to base geometry image
    J_2 = jacobian(geometry.Φ_2, element_idx, ξ)  # the mapping from the image of the  base geometry to the image of the mapping
    return J_2 * J_1
end

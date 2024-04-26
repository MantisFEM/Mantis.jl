
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

# I would rename this to: ∂Φ_∂ξ
function jacobian_in_canonical_coordinates(geometry::CompositeGeometry{n, m}, element_idx::Int, ξ::Vector{Float64}) where {n, m}
    return jacobian_in_canonical_coordinates(geometry.Φ_2, element_idx, ξ)  # We only need the final geometry because it already accounts 
                                                                            # for the whole transformation because we require that geometries
                                                                            # share the same base geometry to be composable
end

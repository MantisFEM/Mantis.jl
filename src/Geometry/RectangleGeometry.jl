
struct Rectangle <: AbstractAnalGeometry 
    n_elements::NTuple{2, Int64}
    xy_start::Vector{Float64}
    xy_end::Vector{Float64}
    dxy::Vector{Float64}
    cartesian_idxs::CartesianIndices
end

function Rectangle(n_elements, xy_start, xy_end)
    dxy = (xy_end .- xy_start) ./ n_elements
    cartesian_idxs = CartesianIndices(n_elements)
    return Rectangle(n_elements, xy_start, xy_end, dxy, cartesian_idxs)
end

function evaluate(geometry::Rectangle, element_idx::Int, ξ::Vector{Float64})
    xy_idx = Tuple(geometry.cartesian_idxs[element_idx])
    x = (xy_idx .- 1) .* geometry.dxy .+ geometry.xy_start + ξ .* geometry.dxy
    return x
end
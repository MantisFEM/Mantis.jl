

struct MappedRectangle <: AbstractAnalGeometry 
    n_elements::NTuple{2, Int64}
    xy_start::Vector{Float64}
    xy_end::Vector{Float64}
    dxy::Vector{Float64}
    cartesian_idxs::CartesianIndices
    mapping::Function
end

function MappedRectangle(n_elements, xy_start, xy_end, mapping)
    dxy = (xy_end .- xy_start) ./ n_elements
    cartesian_idxs = CartesianIndices(n_elements)
    return MappedRectangle(n_elements, xy_start, xy_end, dxy, cartesian_idxs, mapping)
end

function evaluate(geometry::MappedRectangle, element_idx::Int, ξ::Vector{Float64})
    xy_idx = Tuple(geometry.cartesian_idxs[element_idx])
    x_rectangle = (xy_idx .- 1) .* geometry.dxy .+ geometry.xy_start + ξ .* geometry.dxy
    x_mapped = geometry.mapping(x_rectangle)
    return x_mapped
end
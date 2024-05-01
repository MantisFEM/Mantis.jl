
struct Rectangle{n, n} <: AbstractAnalGeometry{n, n}
    dimension::NTuple{2, Int64}
    n_elements::Int
    n_elements_xy::NTuple{2, Int64}
    xy_start::Vector{Float64}
    xy_end::Vector{Float64}
    dxy::Vector{Float64}
    cartesian_idxs::CartesianIndices
end

function Rectangle{2, 2}(n_elements_xy, xy_start, xy_end)
    dimension = (2, 2)
    dxy = (xy_end .- xy_start) ./ n_elements_xy
    cartesian_idxs = CartesianIndices(n_elements_xy)
    n_elements = prod(n_elements_xy)
    return Rectangle{2, 2}(dimension, n_elements, n_elements_xy, xy_start, xy_end, dxy, cartesian_idxs)
end

function get_num_elements(geometry::Rectangle{2,2})
    return geometry.n_elements
end

function get_domain_dim(geometry::Rectangle{2,2})
    return 2
end

function get_image_dim(geometry::Rectangle{2,2})
    return 2
end

function evaluate(geometry::Rectangle{2, 2}, element_idx::Int, ξ::Vector{Float64})
    xy_idx = Tuple(geometry.cartesian_idxs[element_idx])
    x = (xy_idx .- 1) .* geometry.dxy .+ geometry.xy_start + ξ .* geometry.dxy
    return x
end

function jacobian(geometry::Rectangle{2, 2}, element_idx::Int, ξ::Vector{Float64})
    J = [geometry.dxy[1] 0.0; 0.0 geometry.dxy[2]]
    return J
end
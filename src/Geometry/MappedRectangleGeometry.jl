

struct MappedRectangle{n, n} <: AbstractAnalGeometry{n, n}
    dimension::NTuple{2, Int64}
    n_elements::Int
    n_elements_xy::NTuple{2, Int64}
    xy_start::Vector{Float64}
    xy_end::Vector{Float64}
    dxy::Vector{Float64}
    cartesian_idxs::CartesianIndices
    mapping::Function
    dmapping::Function
end

function get_num_elements(geometry::MappedRectangle{n,m}) where {n,m}
    return geometry.n_elements
end

function get_domain_dim(geometry::MappedRectangle{n,m}) where {n, m}
    return n
end

function get_image_dim(geometry::MappedRectangle{n,m}) where {n, m}
    return m
end

function MappedRectangle{2, 2}(n_elements_xy, xy_start, xy_end, mapping, dmapping)
    dimension = (2, 2)
    dxy = (xy_end .- xy_start) ./ n_elements_xy
    cartesian_idxs = CartesianIndices(n_elements_xy)
    n_elements = prod(n_elements_xy)
    return MappedRectangle{2, 2}(dimension, n_elements, n_elements_xy, xy_start, xy_end, dxy, cartesian_idxs, mapping, dmapping)
end

function evaluate(geometry::MappedRectangle{2, 2}, element_idx::Int, ξ::Vector{Float64})
    xy_idx = Tuple(geometry.cartesian_idxs[element_idx])
    x_rectangle = (xy_idx .- 1) .* geometry.dxy .+ geometry.xy_start + ξ .* geometry.dxy
    x_mapped = geometry.mapping(x_rectangle)
    return x_mapped
end

function jacobian(geometry::MappedRectangle{2, 2}, element_idx::Int, ξ::Vector{Float64})
    J_1 = [dxy[1] 0.0; 0.0 dx[2]]  # the Jacobian for the mapping from the elements to the rectangle
    x = _evaluate_x_unmapped(geometry::MappedRectangle{2, 2}, element_idx::Int, ξ::Vector{Float64})
    J_2 = dmapping(x)  # the mapping from the rectangle to the mapped rectangle
    return J_2 * J_1
end

function _evaluate_x_unmapped(geometry::MappedRectangle{2, 2}, element_idx::Int, ξ::Vector{Float64})
    xy_idx = Tuple(geometry.cartesian_idxs[element_idx])
    x_unmapped = (xy_idx .- 1) .* geometry.dxy .+ geometry.xy_start + ξ .* geometry.dxy
    return x_unmapped
end
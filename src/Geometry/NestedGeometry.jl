"""
    NestedMapping{N, dN}

A nested mapping is a pre-mapping from a domain geometry with `num_elements_domain` elements
to a range geometry with `num_elements_range < num_elements_domain` elements. It is assumed
that the domain geometry is obtained by a refinement of the range geometry. Then, the map
`mapping` maps points from the domain to the range geometry and tells us which element of
the range geometry the points are in. It is assumed that this is an affine mapping. The
`dmapping` function gives the constant dim-wise jacobian of the mapping.

# Fields
- `num_elements_domain::Int`: The number of elements in the domain.
- `num_elements_range::Int`: The number of elements in the range.
- `mapping::N`: The mapping from the domain to the range.
- `dmapping::dN`: The mapping from the domain to the length scales of the range.
"""
struct NestedMapping{N, dN}
    num_elements_domain::Int
    num_elements_range::Int
    mapping::N
    dmapping::dNf
end

"""
    get_num_elements(nested_mapping::NestedMapping)

Get the number of elements in the domain geometry.

# Arguments
- `nested_mapping::NestedMapping`: The nested mapping.

# Returns
- `num_elements_domain::Int`: The number of elements in the domain.
"""
function get_num_elements(nested_mapping::NestedMapping)
    return nested_mapping.num_elements_domain
end

"""
    get_element_length_scales(nested_mapping::NestedMapping, element_idx_domain::Int)

Get the ratio of the length scales of the domain and range geometries for a given element.

# Arguments
- `nested_mapping::NestedMapping`: The nested mapping.
- `element_idx_domain::Int`: The element index in the domain geometry.

# Returns
- `length_scales::Vector{Float64}`: The ratio of the length scales.
"""
function get_element_length_scales(nested_mapping::NestedMapping, element_idx_domain::Int)
    return nested_mapping.dmapping(element_idx_domain)
end

"""
    get_element_volume_scale(nested_mapping::NestedMapping, element_idx_domain::Int)

Get the ratio of the volume scales of the domain and range geometries for a given element.

# Arguments
- `nested_mapping::NestedMapping`: The nested mapping.
- `element_idx_domain::Int`: The element index in the domain geometry.

# Returns
- `volume_scale::Float64`: The ratio of the volume scales.
"""
function get_element_volume_scale(nested_mapping::NestedMapping, element_idx_domain::Int)
    return prod(get_element_length_scales(nested_mapping, element_idx_domain))
end

"""
    evaluate(
        nested_mapping::NestedMapping,
        element_idx_domain::Int,
        xi_domain::NTuple{N, Vector{Float64}}
    )

Evaluate the mapping from the domain to the range geometry.

# Arguments
- `nested_mapping::NestedMapping`: The nested mapping.
- `element_idx_domain::Int`: The element index in the domain geometry.
- `xi_domain::NTuple{N, Vector{Float64}}`: The canonical points in the domain geometry.

# Returns
- `element_idx_range::Int`: The element index in the range geometry.
- `xi_range::NTuple{N, Vector{Float64}}`: The canonical points in the range geometry.
"""
function evaluate(
    nested_mapping::NestedMapping,
    element_idx_domain::Int,
    xi_domain::NTuple{N, Vector{Float64}}
)
    element_idx_range, xi_range = nested_mapping.mapping(element_idx_domain, xi_domain)
    return element_idx_range, xi_range
end

"""
    NestedGeometry{manifold_dim, G, Map}

A nested geometry is a geometry that is obtained by a refinement of another geometry. The
geometry `base_geometry` is the range geometry of the nested mapping `nested_map`.

# Fields
- `base_geometry::G`: The base geometry.
- `nested_map::Map`: The nested mapping.
"""
struct NestedGeometry{manifold_dim, G, Map} <: AbstractGeometry{manifold_dim}
    base_geometry::G
    nested_map::Map

    function NestedGeometry(
        base_geometry::G, mapping::Map
    ) where {manifold_dim, G <: AbstractGeometry{manifold_dim}, Map <: NestedMapping}
        if get_num_elements(base_geometry) != mapping.num_elements_range
            throw(ArgumentError("Number of elements in geometry and mapping do not match."))
        end

        return new{manifold_dim, G, Map}(base_geometry, mapping)
    end
end

"""
    get_num_elements(geometry::NestedGeometry)

Get the number of elements in the geometry.

# Arguments
- `geometry::NestedGeometry`: The nested geometry.

# Returns
- `num_elements::Int`: The number of elements in the geometry.
"""
function get_num_elements(geometry::NestedGeometry)
    return get_num_elements(geometry.nested_map)
end

"""
    get_image_dim(geometry::NestedGeometry)

Get the dimension of the image of the geometry.

# Arguments
- `geometry::NestedGeometry`: The nested geometry.

# Returns
- `image_dim::Int`: The dimension of the image.
"""
function get_image_dim(geometry::NestedGeometry)
    return get_image_dim(geometry.base_geometry)
end

"""
    get_element_lengths(geometry::NestedGeometry, element_id::Int)

Get the lengths of the elements in the geometry.

# Arguments
- `geometry::NestedGeometry`: The nested geometry.
- `element_id::Int`: The element index.

# Returns
- `lengths::Vector{Float64}`: The lengths of the element.
"""
function get_element_lengths(geometry::NestedGeometry, element_id::Int)
    lengths = get_element_lengths(geometry.base_geometry, element_id)
    scales = get_element_length_scales(geometry.nested_map, element_id)
    return lengths .* scales
end

"""
    get_element_volume(geometry::NestedGeometry, element_id::Int)

Get the volume of the element in the geometry.

# Arguments
- `geometry::NestedGeometry`: The nested geometry.
- `element_id::Int`: The element index.

# Returns
- `volume::Float64`: The volume of the element.
"""
function get_element_volume(geometry::NestedGeometry, element_id::Int)
    volume = get_element_volume(geometry.base_geometry, element_id)
    scale = get_element_volume_scale(geometry.nested_map, element_id)
    return volume * scale
end

"""
    evaluate(
        geometry::NestedGeometry,
        element_idx_domain::Int,
        xi_domain::NTuple{manifold_dim, Vector{Float64}}
    )

Evaluate the geometry.

# Arguments
- `geometry::NestedGeometry`: The nested geometry.
- `element_idx_domain::Int`: The element index in the domain geometry.
- `xi_domain::NTuple{manifold_dim, Vector{Float64}}`: The canonical points in the domain geometry.

# Returns
- `x::Vector{Float64}`: The points in the image of the geometry.
"""
function evaluate(
    geometry::NestedGeometry,
    element_idx_domain::Int,
    xi_domain::NTuple{manifold_dim, Vector{Float64}}
) where {manifold_dim}
    # map the canonical points to the base geometry
    element_idx_range, xi_range = evaluate(
        geometry.nested_map, element_idx_domain, xi_domain
    )
    # evaluate the base geometry
    x = evaluate(geometry.base_geometry, element_idx_range, xi_range)

    return x
end

"""
    jacobian(
        geometry::NestedGeometry,
        element_idx_domain::Int,
        xi_domain::NTuple{manifold_dim, Vector{Float64}}
    )

Evaluate the jacobian of the geometry.

# Arguments
- `geometry::NestedGeometry`: The nested geometry.
- `element_idx_domain::Int`: The element index in the domain geometry.
- `xi_domain::NTuple{manifold_dim, Vector{Float64}}`: The canonical points in the domain geometry.

# Returns
- `J::Array{Float64, 3}`: The jacobian of the geometry.
"""
function jacobian(
    geometry::NestedGeometry,
    element_idx_domain::Int,
    xi_domain::NTuple{manifold_dim, Vector{Float64}}
) where {manifold_dim}
    # evaluate the mapping jacobian
    length_scales = get_element_length_scales(geometry.nested_map, element_idx_domain)
    J_map = zeros(Float64, manifold_dim, manifold_dim)
    for i in 1:manifold_dim
        J_map[i, i] = length_scales[i]
    end

    # map the canonical points to the base geometry
    element_idx_range, xi_range = evaluate(
        geometry.nested_map, element_idx_domain, xi_domain
    )

    # evaluate the base geometry
    J_base = jacobian(geometry.base_geometry, element_idx_range, xi_range)
    # scale the base geometry jacobian
    for i in axes(J_base, 1)
        J_base[i, :, :] .= J_base[i, :, :] * J_map
    end

    return J_base
end

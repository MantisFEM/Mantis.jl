################################################################################
# Some standard geometries
################################################################################

"""
    create_cartesian_box(starting_points::NTuple{manifold_dim, Float64}, box_sizes::NTuple{manifold_dim, Float64}, num_elements::NTuple{manifold_dim, Int}) where {manifold_dim}

Create a Cartesian box geometry with `manifold_dim` dimensions, starting at `starting_points` and with `box_sizes` and `num_elements` defining the size of the box.

# Arguments

  - `starting_points::NTuple{manifold_dim, Float64}`: The starting points of the box.
  - `box_sizes::NTuple{manifold_dim, Float64}`: The size of the box.
  - `num_elements::NTuple{manifold_dim, Int}`: The number of elements in each dimension.

# Output

  - `geometry::CartesianGeometry{manifold_dim}`: The Cartesian box geometry.
"""
function create_cartesian_box(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
) where {manifold_dim}
    breakpoints = map(
        LinRange, starting_points, starting_points .+ box_sizes, num_elements .+ 1
    )
    return CartesianGeometry(map(collect, breakpoints))
end

"""
    create_curvilinear_square(num_el::NTuple{2,Int}, crazy_c::Float64 = 0.2)

Create a curvilinear square geometry with `num_el` elements in each direction and a `crazy_c` parameter.

# Arguments

  - `num_el::NTuple{2,Int}`: The number of elements in each direction.
  - `crazy_c::Float64 = 0.2`: The `crazy_c` parameter.

# Output

  - `geometry::MappedGeometry{2}`: The curvilinear square geometry.
"""
function create_curvilinear_square(
    starting_points::NTuple{2, Float64},
    box_sizes::NTuple{2, Float64},
    num_elements::NTuple{2, Int};
    crazy_c::Float64=0.1,
)
    # build underlying Cartesian geometry
    unit_square = create_cartesian_box(starting_points, box_sizes, num_elements)

    # build curved mapping
    function mapping(x::AbstractVector)
        x1_new =
            (2.0 / (box_sizes[1])) * x[1] - 2.0 * starting_points[1] / (box_sizes[1]) - 1.0
        x2_new =
            (2.0 / (box_sizes[2])) * x[2] - 2.0 * starting_points[2] / (box_sizes[2]) - 1.0
        return [
            x[1] + ((box_sizes[1]) / 2.0) * crazy_c * sinpi(x1_new) * sinpi(x2_new),
            x[2] + ((box_sizes[2]) / 2.0) * crazy_c * sinpi(x1_new) * sinpi(x2_new),
        ]
    end
    function dmapping(x::AbstractVector)
        x1_new =
            (2.0 / (box_sizes[1])) * x[1] - 2.0 * starting_points[1] / (box_sizes[1]) - 1.0
        x2_new =
            (2.0 / (box_sizes[2])) * x[2] - 2.0 * starting_points[2] / (box_sizes[2]) - 1.0
        return [
            1.0+pi * crazy_c * cospi(x1_new) * sinpi(x2_new) ((box_sizes[1])/(box_sizes[2]))*pi*crazy_c*sinpi(x1_new)*cospi(x2_new)
            ((box_sizes[2])/(box_sizes[1]))*pi*crazy_c*cospi(x1_new)*sinpi(x2_new) 1.0+pi * crazy_c * sinpi(x1_new) * cospi(x2_new)
        ]
    end
    dimension = (2, 2)
    curved_mapping = Mapping(dimension, mapping, dmapping)

    return MappedGeometry(unit_square, curved_mapping)
end

################################################################################
# Nested mappings for hierarchical meshes
################################################################################

function create_hierarchical_mesh_nestedness_map(
    hierarchical_space::FunctionSpaces.HierarchicalFiniteElementSpace{manifold_dim, S, T},
    exclude_elements::Vector{Int}
) where {
    manifold_dim,
    S<:FunctionSpaces.AbstractFESpace{manifold_dim, 1},
    T<:FunctionSpaces.AbstractTwoScaleOperator
}
    # mesh for range space
    range_space = FunctionSpaces.get_space(hierarchical_space, 1)
    # number of elements in range (i.e., the coarsest level mesh)
    num_elements_range = FunctionSpaces.get_num_elements(range_space)
    # get element vertices for the range
    range_element_vertices = [
        FunctionSpaces.get_element_vertices(range_space, i) for i in 1:num_elements_range
    ]

    # number of elements in domain (i.e., the active hierarchical mesh)
    num_elements_domain_all = FunctionSpaces.get_num_elements(hierarchical_space)
    num_elements_domain = num_elements_domain_all - length(exclude_elements)
    # get domain to range element index map
    domain_to_range_map = zeros(Int64, num_elements_domain)
    # get element vertices for the domain
    domain_element_vertices = Vector{NTuple{manifold_dim, Vector{Float64}}}(
        undef, num_elements_domain
    )
    count = 0
    for i in 1:num_elements_domain_all
        element_level, element_level_id =
        FunctionSpaces.convert_to_element_level_and_level_id(
            hierarchical_space, i
        )
        if ~(i in exclude_elements)
            domain_element_vertices[count+1] = FunctionSpaces.get_element_vertices(
                    FunctionSpaces.get_space(hierarchical_space, element_level),
                    element_level_id
            )
            domain_to_range_map[count+1] = FunctionSpaces.get_element_ancestor(
                hierarchical_space.two_scale_operators,
                i,
                element_level,
                element_level-1
            )
            count += 1
        end
    end

    # create dmapping: length scale ratios of child and ancestor
    function dmapping(element_idx_domain)
        range_el_verts = range_element_vertices[domain_to_range_map[element_idx_domain]]
        domain_el_verts = domain_element_vertices[element_idx_domain]
        return [domain_el_verts[k][2] - domain_el_verts[k][1] for k in 1:manifold_dim] ./
                [range_el_verts[k][2] - range_el_verts[k][1] for k in 1:manifold_dim]
    end

    # create mapping: affine map from domain to range
    function mapping(element_idx_domain, xi_domain)
        element_idx_range = domain_to_range_map[element_idx_domain]
        range_el_verts = range_element_vertices[element_idx_range]
        domain_el_verts = domain_element_vertices[element_idx_domain]
        length_scales = dmapping(element_idx_domain)
        xi_range = Vector{Vector{Float64}}(undef, manifold_dim)
        for k in 1:manifold_dim
            xi_range[k] = xi_domain[k] .* length_scales[k] .+
                (domain_el_verts[k][1] .- range_el_verts[k][1]) ./
                (range_el_verts[k][2] .- range_el_verts[k][1])
        end

        return element_idx_range, tuple(xi_range...)
    end

    return NestedMapping(num_elements_domain, num_elements_range, mapping, dmapping)
 end

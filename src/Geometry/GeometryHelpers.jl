################################################################################
# Some standard geometries
################################################################################

"""
    create_cartesian_box(
        starting_points::NTuple{manifold_dim, Float64},
        box_sizes::NTuple{manifold_dim, Float64},
        num_elements::NTuple{manifold_dim, Int},
    ) where {manifold_dim}

Create a Cartesian box geometry with `manifold_dim` dimensions, starting at
`starting_points` and with `box_sizes` and `num_elements` defining the size of the box.

# Arguments
  - `starting_points::NTuple{manifold_dim, Float64}`: The starting points of the box.
  - `box_sizes::NTuple{manifold_dim, Float64}`: The size of the box.
  - `num_elements::NTuple{manifold_dim, Int}`: The number of elements in each dimension.

# Output
  - `::CartesianGeometry{manifold_dim}`: The Cartesian box geometry.
"""
function create_cartesian_box(
    starting_points::NTuple{manifold_dim, Float64},
    box_sizes::NTuple{manifold_dim, Float64},
    num_elements::NTuple{manifold_dim, Int},
) where {manifold_dim}
    breakpoints = map(
        LinRange, starting_points, starting_points .+ box_sizes, num_elements .+ 1
    )
    return CartesianGeometry(breakpoints)
end

function create_curvilinear_mapping(
    starting_points::NTuple{2, Float64}, box_sizes::NTuple{2, Float64}, c::Float64=0.1
)
    # build curved mapping
    function mapping(x::AbstractVector)
        x1_new =
            (2.0 / (box_sizes[1])) * x[1] - 2.0 * starting_points[1] / (box_sizes[1]) - 1.0
        x2_new =
            (2.0 / (box_sizes[2])) * x[2] - 2.0 * starting_points[2] / (box_sizes[2]) - 1.0
        return [
            x[1] + ((box_sizes[1]) / 2.0) * c * sinpi(x1_new) * sinpi(x2_new),
            x[2] + ((box_sizes[2]) / 2.0) * c * sinpi(x1_new) * sinpi(x2_new),
        ]
    end

    function dmapping(x::AbstractVector)
        x1_new =
            (2.0 / (box_sizes[1])) * x[1] - 2.0 * starting_points[1] / (box_sizes[1]) - 1.0
        x2_new =
            (2.0 / (box_sizes[2])) * x[2] - 2.0 * starting_points[2] / (box_sizes[2]) - 1.0
        return SMatrix{2, 2}( # Note: SMatrix creates the matrix per column.
            1.0 + pi * c * cospi(x1_new) * sinpi(x2_new),
            (box_sizes[2] / box_sizes[1]) * pi * c * cospi(x1_new) * sinpi(x2_new),
            (box_sizes[1] / box_sizes[2]) * pi * c * sinpi(x1_new) * cospi(x2_new),
            1.0 + pi * c * sinpi(x1_new) * cospi(x2_new),
        )
    end

    function ddmapping(x::AbstractVector)
        x1_new =
            (2.0 / (box_sizes[1])) * x[1] - 2.0 * starting_points[1] / (box_sizes[1]) - 1.0
        x2_new =
            (2.0 / (box_sizes[2])) * x[2] - 2.0 * starting_points[2] / (box_sizes[2]) - 1.0
        return (
            [
                -(2.0 / box_sizes[1]) * pi^2 * c * sinpi(x1_new) * sinpi(x2_new) (2 / box_sizes[2]) * pi^2 * c * cospi(
                    x1_new
                ) * cospi(x2_new)
                (2.0 / box_sizes[2]) * pi^2 * c * cospi(x1_new) * cospi(x2_new) -(2 * box_sizes[1] / (box_sizes[2]^2)) * pi^2 * c * sinpi(x1_new) * sinpi(x2_new)
            ],
            [
                -(2 * box_sizes[2] / (box_sizes[1]^2)) * pi^2 * c * sinpi(x1_new) * sinpi(
                    x2_new
                ) (2.0 / box_sizes[1]) * pi^2 * c * cospi(x1_new) * cospi(x2_new)
                (2.0 / box_sizes[1]) * pi^2 * c * cospi(x1_new) * cospi(x2_new) -(2.0 / box_sizes[2]) * pi^2 * c * sinpi(x1_new) * sinpi(x2_new)
            ],
        )
    end
    dimension = (2, 2)
    curved_mapping = Mapping(dimension, mapping, dmapping, ddmapping)

    return curved_mapping
end

function create_curvilinear_mapping(
    starting_points::NTuple{3, Float64}, box_sizes::NTuple{3, Float64}, c::Float64=0.1
)
    # build curved mapping
    function mapping(x::AbstractVector)
        x1_new =
            (2.0 / (box_sizes[1])) * x[1] - 2.0 * starting_points[1] / (box_sizes[1]) - 1.0
        x2_new =
            (2.0 / (box_sizes[2])) * x[2] - 2.0 * starting_points[2] / (box_sizes[2]) - 1.0
        return [
            x[1] + ((box_sizes[1]) / 2.0) * c * sinpi(x1_new) * sinpi(x2_new),
            x[2] + ((box_sizes[2]) / 2.0) * c * sinpi(x1_new) * sinpi(x2_new),
            x[3],
        ]
    end
    function dmapping(x::AbstractVector)
        x1_new =
            (2.0 / (box_sizes[1])) * x[1] - 2.0 * starting_points[1] / (box_sizes[1]) - 1.0
        x2_new =
            (2.0 / (box_sizes[2])) * x[2] - 2.0 * starting_points[2] / (box_sizes[2]) - 1.0
        return [
            [
                1.0 + pi * c * cospi(x1_new) * sinpi(x2_new)
                (box_sizes[1] / box_sizes[2]) * pi * c * sinpi(x1_new) * cospi(x2_new)
                0.0
            ]
            [
                (box_sizes[2] / box_sizes[1]) * pi * c * cospi(x1_new) * sinpi(x2_new)
                1.0 + pi * c * sinpi(x1_new) * cospi(x2_new)
                0.0
            ]
            [
                0.0
                0.0
                1.0
            ]
        ]
    end
    dimension = (3, 3)
    curved_mapping = Mapping(dimension, mapping, dmapping)

    return curved_mapping
end

"""
    create_curvilinear_square(
        starting_points::NTuple{2, Float64},
        box_sizes::NTuple{2, Float64},
        num_elements::NTuple{2, Int};
        c::Float64=0.1,
    )

Create a single-patch curvilinear square geometry with `num_elements` elements in each
direction and a `c` parameter to change the deformation of the mapping. Not that the mapping
becomes singular with `c` = 0.3.

# Arguments
  - `num_elements::NTuple{2,Int}`: The number of elements in each direction.
  - `c::Float64 = 0.1`: The `c` parameter.

# Output
  - `geometry::MappedGeometry{2, 2, 1}`: The curvilinear square geometry.
"""
function create_curvilinear_square(
    starting_points::NTuple{2, Float64},
    box_sizes::NTuple{2, Float64},
    num_elements::NTuple{2, Int};
    c::Float64=0.1,
)
    # build underlying Cartesian geometry
    unit_square = create_cartesian_box(starting_points, box_sizes, num_elements)

    # build curved mapping
    curved_mapping = create_curvilinear_mapping(starting_points, box_sizes, c)

    return MappedGeometry(unit_square, curved_mapping)
end

############################################################################################
#                                         Mappings                                         #
############################################################################################

function affine_map(xi, A, b)
    return xi * A + b
end

function affine_map(
    xi::Points.AbstractPoints{manifold_dim}, A::Matrix, b::Vector
) where {manifold_dim}
    num_points = Points.get_num_points(xi)
    mapped_dim = size(A, 1)
    mapped_xi = ntuple(mapped_dim) do _
        return zeros(Float64, num_points)
    end

    for (point_id, point) in enumerate(xi)
        for j in axes(A, 2)
            for i in axes(A, 1)
                mapped_xi[i][point_id] += affine_map(point[j], A[i, j], 0.0)
            end
        end

        for i in axes(b, 1)
            mapped_xi[i][point_id] += b[i]
        end
    end

    return mapped_xi
end

############################################################################################
#                                     Immersed Methods                                     #
############################################################################################

function create_immersed_geometry(
    background_geometry::AbstractGeometry, active_elements::Vector{Int}
)
    active_info = Hierarchical.ActiveInfo([active_elements, Int[]])
    scaling = Hierarchical.Scaling(background_geometry)

    return HierarchicalGeometry(active_info, [scaling])
end

function create_immersed_geometry(
    background_geometry::AbstractGeometry,
    level_set::Function;
    inside_tol=1e-14,
    num_facet_points=5,
)
    active_elements, _ = active_element_identification(
        background_geometry,
        level_set;
        inside_tol=inside_tol,
        num_facet_points=num_facet_points,
    )

    return create_immersed_geometry(background_geometry, active_elements)
end

function level_set_identification(
    background_geometry::AbstractGeometry{manifold_dim},
    level_set::Function;
    inside_tol=1e-14,
    num_facet_points=5,
) where {manifold_dim}
    active_elements = Int[]
    cut_elements = Bool[]
    # We can re-use the same vectors to check the if vertices and edges are inside the
    # domain.
    vertex_checks = zeros(Bool, 2^manifold_dim)
    facet_points = ntuple(_ -> zeros(num_facet_points), manifold_dim)
    for element_id in 1:get_num_elements(background_geometry)
        vertices = get_element_vertices(background_geometry, element_id)
        for (count, vertex) in enumerate(Iterators.product(vertices...))
            vertex_checks[count] = level_set(vertex) < inside_tol
        end

        # If any vertex of the element is inside the domain, then that element is
        # necessarily active.
        # Else, we still need to check whether any of the facets are cut, to avoid
        # incorrectly marking elements as inactive when a curve misses any of the vertices.
        # This is checked only up to a certain resolution, dictated by `num_facet_points`.
        if any(vertex_checks)
            push!(active_elements, element_id)
            push!(cut_elements, !all(vertex_checks))
        else
            cut = false
            for dim in 1:manifold_dim
                # We first generate the facet points points covering the two facets
                # orthogonal to `dim`, vertices[d] for d == dim, and a range of
                # `num_facet_points` otherwise.
                for d in 1:manifold_dim
                    if d == dim
                        facet_points[d][1] = vertices[d][1]
                        facet_points[d][2] = vertices[d][2]
                    else
                        facet_points[d] .= LinRange(
                            vertices[d][1], vertices[d][2], num_facet_points
                        )
                    end
                end
                # Then we check if any point of the facet lies inside the curve.
                for facet_point in Iterators.product(
                    (
                        d == dim ? view(facet_points[d], 1:2) : facet_points[d] for
                        d in 1:manifold_dim
                    )...,
                )
                    if level_set(facet_point) < inside_tol
                        push!(active_elements, element_id)
                        push!(cut_elements, true)
                        cut = true
                        break
                    end
                end

                cut && break
            end
        end
    end

    return active_elements, cut_elements
end

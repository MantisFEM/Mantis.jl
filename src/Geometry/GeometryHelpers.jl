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
    dimension = (2, 2)
    curved_mapping = Mapping(dimension, mapping, dmapping)

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
            1.0+pi * crazy_c * cospi(x1_new) * sinpi(x2_new) (box_sizes[1]/box_sizes[2])*pi*crazy_c*sinpi(x1_new)*cospi(x2_new)
            (box_sizes[2]/box_sizes[1])*pi*crazy_c*cospi(x1_new)*sinpi(x2_new) 1.0+pi * crazy_c * sinpi(x1_new) * cospi(x2_new)
        ]
    end
    function ddmapping(x::AbstractVector)
        x1_new =
            (2.0 / (box_sizes[1])) * x[1] - 2.0 * starting_points[1] / (box_sizes[1]) - 1.0
        x2_new =
            (2.0 / (box_sizes[2])) * x[2] - 2.0 * starting_points[2] / (box_sizes[2]) - 1.0
        return (
            [
                -(2.0 / box_sizes[1])*pi^2*crazy_c*sinpi(x1_new)*sinpi(x2_new) (2/box_sizes[2])*pi^2*crazy_c*cospi(x1_new)*cospi(x2_new)
                (2.0/box_sizes[2])*pi^2*crazy_c*cospi(x1_new)*cospi(x2_new) -(2 * box_sizes[1] / (box_sizes[2]^2))*pi^2*crazy_c*sinpi(x1_new)*sinpi(x2_new)
            ],
            [
                -(2 * box_sizes[2] / (box_sizes[1]^2))*pi^2*crazy_c*sinpi(x1_new)*sinpi(x2_new) (2.0/box_sizes[1])*pi^2*crazy_c*cospi(x1_new)*cospi(x2_new)
                (2.0/box_sizes[1])*pi^2*crazy_c*cospi(x1_new)*cospi(x2_new) -(2.0 / box_sizes[2])*pi^2*crazy_c*sinpi(x1_new)*sinpi(x2_new)
            ],
        )
    end
    dimension = (2, 2)
    curved_mapping = Mapping(dimension, mapping, dmapping, ddmapping)

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

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
                -(2.0 / box_sizes[1])*pi^2*c*sinpi(x1_new)*sinpi(x2_new) (2/box_sizes[2])*pi^2*c*cospi(x1_new)*cospi(x2_new)
                (2.0/box_sizes[2])*pi^2*c*cospi(x1_new)*cospi(x2_new) -(2 * box_sizes[1] / (box_sizes[2]^2))*pi^2*c*sinpi(x1_new)*sinpi(x2_new)
            ],
            [
                -(2 * box_sizes[2] / (box_sizes[1]^2))*pi^2*c*sinpi(x1_new)*sinpi(x2_new) (2.0/box_sizes[1])*pi^2*c*cospi(x1_new)*cospi(x2_new)
                (2.0/box_sizes[1])*pi^2*c*cospi(x1_new)*cospi(x2_new) -(2.0 / box_sizes[2])*pi^2*c*sinpi(x1_new)*sinpi(x2_new)
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

# The geometries below are all built on cylindrical coordinates. They are explained in the
# Geometry example, which builds each of them step by step.

function _check_radii(radii::NTuple{2, Float64})
    if !(0.0 < radii[1] < radii[2])
        throw(
            ArgumentError(
                LazyString(
                    "The radii must satisfy 0 < radii[1] < radii[2], but they are ",
                    radii,
                    ".",
                ),
            ),
        )
    end

    return nothing
end

function _check_positive(value::Float64, name::String)
    if !(value > 0.0)
        throw(
            ArgumentError(
                LazyString("The ", name, " must be positive, but it is ", value, ".")
            ),
        )
    end

    return nothing
end

"""
    create_archimedean_spiral(
        num_elements::Int;
        inner_radius::Real=0.5,
        radial_growth::Real=1 / (4π),
        angle::Real=4π,
    )

Create a single-patch geometry of an Archimedean spiral in the plane, a curve given by
```math
\\Phi(\\theta) = r(\\theta)\\,(\\cos\\theta, \\sin\\theta)\\,, \\qquad
r(\\theta) = r_0 + a\\,\\theta\\,, \\qquad \\theta \\in [0, \\theta_{\\max}]\\,,
```
with ``r_0`` = `inner_radius`, ``a`` = `radial_growth` and ``\\theta_{\\max}`` = `angle`.
Successive turns are a distance ``2\\pi a`` apart. The default values give two turns, with
the radius growing from 1/2 to 3/2. The first and second derivatives of the mapping are
included.

# Arguments
- `num_elements::Int`: The number of elements along the spiral, uniform in ``\\theta``.
- `inner_radius::Real=0.5`: The radius ``r_0`` at ``\\theta = 0``. Must be positive.
- `radial_growth::Real=1 / (4π)`: The growth ``a`` of the radius per radian. Must be
    positive.
- `angle::Real=4π`: The angle ``\\theta_{\\max}`` swept by the spiral. Must be positive.

# Returns
- `::MappedGeometry{1, 2, 1}`: The spiral.

# Exceptions
- ArgumentError: If `inner_radius`, `radial_growth` or `angle` is not positive.
"""
function create_archimedean_spiral(
    num_elements::Int; inner_radius::Real=0.5, radial_growth::Real=1 / (4π), angle::Real=4π
)
    r₀, a, θ_max = Float64(inner_radius), Float64(radial_growth), Float64(angle)
    _check_positive(r₀, "inner radius")
    _check_positive(a, "radial growth")
    _check_positive(θ_max, "angle")

    function mapping(x::AbstractVector)
        θ = x[1]
        r = r₀ + a * θ
        return SVector(r * cos(θ), r * sin(θ))
    end
    function dmapping(x::AbstractVector)
        θ = x[1]
        r = r₀ + a * θ
        return SMatrix{2, 1}(a * cos(θ) - r * sin(θ), a * sin(θ) + r * cos(θ))
    end
    function ddmapping(x::AbstractVector)
        θ = x[1]
        r = r₀ + a * θ
        return (
            SMatrix{1, 1}(-2a * sin(θ) - r * cos(θ)),
            SMatrix{1, 1}(2a * cos(θ) - r * sin(θ)),
        )
    end

    parametric_geometry = create_cartesian_box((0.0,), (θ_max,), (num_elements,))

    return MappedGeometry(
        parametric_geometry, Mapping((1, 2), mapping, dmapping, ddmapping)
    )
end

"""
    create_annulus_sector(
        num_elements::NTuple{2, Int};
        radii::NTuple{2, Real}=(1.0, 2.0),
        angle::Real=π,
    )

Create a single-patch geometry of a sector of an annulus in the plane, given by the
polar-coordinate map
```math
\\Phi(r, \\theta) = (r\\cos\\theta, r\\sin\\theta)\\,, \\qquad
(r, \\theta) \\in [r_1, r_2] \\times [0, \\theta_{\\max}]\\,,
```
with ``(r_1, r_2)`` = `radii` and ``\\theta_{\\max}`` = `angle`. The default values give the
upper half of the annulus between the circles of radii 1 and 2. The first and second
derivatives of the mapping are included.

# Arguments
- `num_elements::NTuple{2, Int}`: The number of elements in ``r`` and in ``\\theta``.
- `radii::NTuple{2, Real}=(1.0, 2.0)`: The inner and outer radii, with
    ``0 < r_1 < r_2``.
- `angle::Real=π`: The angle ``\\theta_{\\max}`` of the sector, with
    ``0 < \\theta_{\\max} \\leq 2\\pi``.

# Returns
- `::MappedGeometry{2, 2, 1}`: The annulus sector.

# Exceptions
- ArgumentError: If the radii are not ordered and positive, or if the angle is not in
    ``(0, 2\\pi]``, in which case the mapping would not be injective.
"""
function create_annulus_sector(
    num_elements::NTuple{2, Int}; radii::NTuple{2, Real}=(1.0, 2.0), angle::Real=π
)
    radii, θ_max = Float64.(radii), Float64(angle)
    _check_radii(radii)
    _check_positive(θ_max, "angle")
    if θ_max > 2π
        throw(
            ArgumentError(
                LazyString(
                    "The angle of an annulus sector must be at most 2π, otherwise the ",
                    "mapping is not injective, but it is ",
                    θ_max,
                    ".",
                ),
            ),
        )
    end

    parametric_geometry = create_cartesian_box(
        (radii[1], 0.0), (radii[2] - radii[1], θ_max), num_elements
    )

    return MappedGeometry(
        parametric_geometry, _create_cylindrical_mapping(Val(2), Val(2), 0.0)
    )
end

"""
    create_helicoid(
        num_elements::NTuple{2, Int};
        radii::NTuple{2, Real}=(1.0, 2.0),
        angle::Real=2π,
        rise_per_turn::Real=2.0,
    )

Create a single-patch geometry of a helicoid, the surface of a spiral ramp, given by
```math
\\Phi(r, \\theta) = (r\\cos\\theta, r\\sin\\theta, c\\,\\theta)\\,, \\qquad
(r, \\theta) \\in [r_1, r_2] \\times [0, \\theta_{\\max}]\\,,
```
with ``(r_1, r_2)`` = `radii`, ``\\theta_{\\max}`` = `angle` and ``2\\pi c`` =
`rise_per_turn`. The default values give one full turn that rises by 2. The first and
second derivatives of the mapping are included.

# Arguments
- `num_elements::NTuple{2, Int}`: The number of elements in ``r`` and in ``\\theta``.
- `radii::NTuple{2, Real}=(1.0, 2.0)`: The inner and outer radii, with
    ``0 < r_1 < r_2``.
- `angle::Real=2π`: The angle ``\\theta_{\\max}`` swept by the helicoid. Must be positive.
- `rise_per_turn::Real=2.0`: The height gained per turn. Must be positive if the angle
    exceeds ``2\\pi``.

# Returns
- `::MappedGeometry{2, 3, 1}`: The helicoid.

# Exceptions
- ArgumentError: If the radii are not ordered and positive, if the angle is not positive,
    or if the angle exceeds ``2\\pi`` without a positive rise, in which case the mapping
    would not be injective.
"""
function create_helicoid(
    num_elements::NTuple{2, Int};
    radii::NTuple{2, Real}=(1.0, 2.0),
    angle::Real=2π,
    rise_per_turn::Real=2.0,
)
    radii, θ_max, rise = Float64.(radii), Float64(angle), Float64(rise_per_turn)
    _check_radii(radii)
    _check_positive(θ_max, "angle")
    if θ_max > 2π && !(rise > 0.0)
        throw(
            ArgumentError(
                LazyString(
                    "A helicoid with an angle larger than 2π needs a positive rise per ",
                    "turn, otherwise the mapping is not injective, but the rise is ",
                    rise,
                    ".",
                ),
            ),
        )
    end

    parametric_geometry = create_cartesian_box(
        (radii[1], 0.0), (radii[2] - radii[1], θ_max), num_elements
    )

    return MappedGeometry(
        parametric_geometry, _create_cylindrical_mapping(Val(2), Val(3), rise / (2π))
    )
end

"""
    create_helical_duct(
        num_elements::NTuple{3, Int};
        radii::NTuple{2, Real}=(1.0, 2.0),
        angle::Real=3π,
        thickness::Real=1.0,
        rise_per_turn::Real=2.0,
    )

Create a single-patch geometry of a helical duct, a helicoid (see
[`create_helicoid`](@ref)) thickened in the vertical direction:
```math
\\Phi(r, \\theta, \\zeta) = (r\\cos\\theta, r\\sin\\theta, \\zeta + c\\,\\theta)\\,, \\qquad
(r, \\theta, \\zeta) \\in [r_1, r_2] \\times [0, \\theta_{\\max}] \\times [0, t]\\,,
```
with ``(r_1, r_2)`` = `radii`, ``\\theta_{\\max}`` = `angle`, ``t`` = `thickness` and
``2\\pi c`` = `rise_per_turn`. The default values give one and a half turns of a duct with
a square cross-section of side 1, rising by 2 per turn. The first and second derivatives of
the mapping are included.

# Arguments
- `num_elements::NTuple{3, Int}`: The number of elements in ``r``, ``\\theta`` and
    ``\\zeta``.
- `radii::NTuple{2, Real}=(1.0, 2.0)`: The inner and outer radii, with
    ``0 < r_1 < r_2``.
- `angle::Real=3π`: The angle ``\\theta_{\\max}`` swept by the duct. Must be positive.
- `thickness::Real=1.0`: The height ``t`` of the cross-section. Must be positive and, if
    the angle exceeds ``2\\pi``, smaller than `rise_per_turn`.
- `rise_per_turn::Real=2.0`: The height gained per turn.

# Returns
- `::MappedGeometry{3, 3, 1}`: The helical duct.

# Exceptions
- ArgumentError: If the radii are not ordered and positive, if the angle or the thickness
    is not positive, or if the angle exceeds ``2\\pi`` and the thickness is not smaller
    than the rise per turn, in which case consecutive turns overlap and the mapping is not
    injective.
"""
function create_helical_duct(
    num_elements::NTuple{3, Int};
    radii::NTuple{2, Real}=(1.0, 2.0),
    angle::Real=3π,
    thickness::Real=1.0,
    rise_per_turn::Real=2.0,
)
    radii, θ_max = Float64.(radii), Float64(angle)
    t, rise = Float64(thickness), Float64(rise_per_turn)
    _check_radii(radii)
    _check_positive(θ_max, "angle")
    _check_positive(t, "thickness")
    if θ_max > 2π && !(t < rise)
        throw(
            ArgumentError(
                LazyString(
                    "A helical duct with an angle larger than 2π needs a thickness smaller ",
                    "than the rise per turn, otherwise consecutive turns overlap, but the ",
                    "thickness is ",
                    t,
                    " and the rise per turn is ",
                    rise,
                    ".",
                ),
            ),
        )
    end

    parametric_geometry = create_cartesian_box(
        (radii[1], 0.0, 0.0), (radii[2] - radii[1], θ_max, t), num_elements
    )

    return MappedGeometry(
        parametric_geometry, _create_cylindrical_mapping(Val(3), Val(3), rise / (2π))
    )
end

# The cylindrical-coordinate mapping (r, θ[, ζ]) ↦ (r cos θ, r sin θ[, ζ + c θ]) and its
# derivatives. With image_dim = 2 it is the polar-coordinate map; with image_dim = 3 and
# manifold_dim = 2 it is a helicoid (ζ = 0); with image_dim = manifold_dim = 3 it is a
# helical duct.
function _create_cylindrical_mapping(
    ::Val{manifold_dim}, ::Val{image_dim}, c::Float64
) where {manifold_dim, image_dim}
    function mapping(x::AbstractVector)
        r, θ = x[1], x[2]
        if image_dim == 2
            return SVector(r * cos(θ), r * sin(θ))
        end
        ζ = manifold_dim == 3 ? x[3] : 0.0
        return SVector(r * cos(θ), r * sin(θ), ζ + c * θ)
    end
    function dmapping(x::AbstractVector)
        r, θ = x[1], x[2]
        J = zeros(MMatrix{image_dim, manifold_dim, Float64})
        J[1, 1], J[1, 2] = cos(θ), -r * sin(θ)
        J[2, 1], J[2, 2] = sin(θ), r * cos(θ)
        if image_dim == 3
            J[3, 2] = c
            if manifold_dim == 3
                J[3, 3] = 1.0
            end
        end
        return SMatrix(J)
    end
    function ddmapping(x::AbstractVector)
        r, θ = x[1], x[2]
        H_1 = zeros(MMatrix{manifold_dim, manifold_dim, Float64})
        H_2 = zeros(MMatrix{manifold_dim, manifold_dim, Float64})
        H_1[1, 2], H_1[2, 1], H_1[2, 2] = -sin(θ), -sin(θ), -r * cos(θ)
        H_2[1, 2], H_2[2, 1], H_2[2, 2] = cos(θ), cos(θ), -r * sin(θ)
        if image_dim == 2
            return (SMatrix(H_1), SMatrix(H_2))
        end
        return (
            SMatrix(H_1), SMatrix(H_2), zeros(SMatrix{manifold_dim, manifold_dim, Float64})
        )
    end

    return Mapping((manifold_dim, image_dim), mapping, dmapping, ddmapping)
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

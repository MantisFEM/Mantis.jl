module MetricTests

using Mantis
using LinearAlgebra
using StaticArrays

include("GeometryTestsHelpers.jl")

using Test

# CartesianGeometry (1, 1) homogeneous grid ------------------------------------------------
dim = 1
nx = 5
breakpoints_cart_1_1 = (LinRange(0.0, 1.0, nx + 1),)
cartesian_geometry_cart_1_1 = Geometry.CartesianGeometry(breakpoints_cart_1_1)

# Expected Jacobian per element (the same for all elements)
dx_cart_1_1 = [1.0 / nx]
g_ref_cart_1_1 = [dx_cart_1_1[1]^2]
inv_g_ref_cart_1_1 = [dx_cart_1_1[1]^(-2)]
det_g_ref_cart_1_1 = sqrt(prod(dx_cart_1_1 .^ 2))

# Points where to evaluate the metric
nx_evaluate = 3
xi_1_cart_1_1 = Points.CartesianPoints((LinRange(0.0, 1.0, nx_evaluate + 1),))
n_evaluation_points = nx_evaluate

# Evaluate the metric, its inverse and its determinant
for element_idx in 1:Geometry.get_num_elements(cartesian_geometry_cart_1_1)
    inv_g, g, sqrt_g = Geometry.inv_metric(
        cartesian_geometry_cart_1_1, element_idx, xi_1_cart_1_1
    )
    g_test = true
    inv_g_test = true
    for point in eachindex(g)
        if !isapprox(g[point][1], g_ref_cart_1_1[1]; rtol=rtol)
            g_test = false
        end
        if !isapprox(inv_g[point][1], inv_g_ref_cart_1_1[1]; rtol=rtol)
            inv_g_test = false
        end
    end
    @test g_test
    @test inv_g_test

    @test all(isapprox.(sqrt_g, det_g_ref_cart_1_1; rtol=rtol))
end
# ------------------------------------------------------------------------------------------

# CartesianGeometry (2, 2) homogeneous grid ------------------------------------------------
dim = 2
nx = 4
ny = 5
breakpoints_cart_2_2 = (
    collect(LinRange(0.0, 1.0, nx + 1)), collect(LinRange(0.0, 2.0, ny + 1))
)
cartesian_geometry_cart_2_2 = Geometry.CartesianGeometry(breakpoints_cart_2_2)

# Expected Jacobian per element (the same for all elements)
dx_cart_2_2 = [1.0 / nx, 2.0 / ny]
g_ref_cart_2_2 = [dx_cart_2_2[1]^2 0.0; 0.0 dx_cart_2_2[2]^2]
inv_g_ref_cart_2_2 = [dx_cart_2_2[1]^(-2) 0.0; 0.0 dx_cart_2_2[2]^(-2)]
det_g_ref_cart_2_2 = prod(dx_cart_2_2)

# Points where to evaluate the metric
nx_evaluate = 3
ny_evaluate = 7
xi_cart_2_2 = Points.CartesianPoints((
    LinRange(0.0, 1.0, nx_evaluate + 1), LinRange(0.0, 1.0, ny_evaluate + 1)
))
n_evaluation_points = nx_evaluate * ny_evaluate

# Evaluate the metric, its inverse and its determinant
for element_idx in 1:Geometry.get_num_elements(cartesian_geometry_cart_2_2)
    inv_g, g, sqrt_g = Geometry.inv_metric(
        cartesian_geometry_cart_2_2, element_idx, xi_cart_2_2
    )
    g_test = true
    inv_g_test = true
    for point in eachindex(g)
        if !all(isapprox.(g[point], g_ref_cart_2_2; rtol=rtol))
            g_test = false
        end
        if !all(isapprox.(inv_g[point], inv_g_ref_cart_2_2; rtol=rtol))
            inv_g_test = false
        end
    end
    @test g_test
    @test inv_g_test

    @test all(isapprox.(sqrt_g, det_g_ref_cart_2_2; rtol=rtol))
end
# ------------------------------------------------------------------------------------------

# CartesianGeometry (2, 2) inhomogeneous grid ----------------------------------------------
dim = 2
breakpoints_cart_2_2_inh = ([0.0, 0.25, 1.0], [0.0, 0.5, 0.9, 1.0])
cartesian_geometry_cart_2_2_inh = Geometry.CartesianGeometry(breakpoints_cart_2_2_inh)

# Expected metric terms per element (allocation)
dx_cart_2_2_inh_all = [
    0.25 0.75 0.25 0.75 0.25 0.75
    0.5 0.5 0.4 0.4 0.1 0.1
]  # the dxs for each element are over the columns
dx_cart_2_2_inh = [dx_cart_2_2_inh_all[:, i] for i in axes(dx_cart_2_2_inh_all, 2)]
g_ref_cart_2_2_inh = [
    [dx_cart_2_2_inh[i][1]^2 0.0; 0.0 dx_cart_2_2_inh[i][2]^2] for
    i in eachindex(dx_cart_2_2_inh)
]
inv_g_ref_cart_2_2_inh = [
    [1.0/dx_cart_2_2_inh[i][1]^2 0.0; 0.0 1.0/dx_cart_2_2_inh[i][2]^2] for
    i in eachindex(dx_cart_2_2_inh)
]
det_g_ref_cart_2_2_inh = [prod(dx_cart_2_2_inh[i]) for i in eachindex(dx_cart_2_2_inh)]

# Evaluate the metric, its inverse and its determinant
for element_idx in 1:Geometry.get_num_elements(cartesian_geometry_cart_2_2_inh)
    inv_g, g, sqrt_g = Geometry.inv_metric(
        cartesian_geometry_cart_2_2_inh, element_idx, xi_cart_2_2
    )
    g_test = true
    inv_g_test = true
    for point in eachindex(g)
        if !all(isapprox.(g[point], g_ref_cart_2_2_inh[element_idx]; rtol=rtol))
            g_test = false
        end
        if !all(isapprox.(inv_g[point], inv_g_ref_cart_2_2_inh[element_idx]; rtol=rtol))
            inv_g_test = false
        end
    end
    @test g_test
    @test inv_g_test

    @test all(isapprox.(sqrt_g, det_g_ref_cart_2_2_inh[element_idx]; rtol=rtol))
end
# ------------------------------------------------------------------------------------------

# Surface embedded in 3D. So image_dim =/= manifold_dim
geo(x) = [x[1], x[2], x[1] * x[2]]
dgeo(x) = [[1.0 0.0]; [0.0 1.0]; [x[2] x[1]]]
mapping2to3 = Mantis.Geometry.Mapping((2, 3), geo, dgeo)
geom_cart = Geometry.CartesianGeometry((0.0:(1.0 / 4):1.0, 0.0:(1.0 / 4):1.0))
geometry2to3 = Mantis.Geometry.MappedGeometry(geom_cart, mapping2to3)
for (k, IJ) in enumerate(CartesianIndices((4, 4)))
    inv_g, g, det_g = Geometry.inv_metric(
        geometry2to3, k, Points.CartesianPoints(([0.0, 1.0], [0.0, 1.0]))
    )

    i, j = Tuple(IJ)
    xans = [
        x_i for _ in (1, 2) for x_i in LinRange((i - 1) * 1.0 / 16.0, i * 1.0 / 16.0, 2)
    ]
    yans = [
        y_i for y_i in LinRange((j - 1) * 1.0 / 16.0, j * 1.0 / 16.0, 2) for _ in (1, 2)
    ]
    invgtest = true
    gtest = true
    for p in eachindex(g)
        if !all(
            isapprox.(
                inv_g[p],
                1.0 /
                ((0.0625 + yans[p]^2) * (0.0625 + xans[p]^2) - (xans[p] * yans[p])^2) *
                [0.0625+xans[p]^2 -xans[p]*yans[p]; -xans[p]*yans[p] 0.0625+yans[p]^2],
                rtol=1e-14,
            ),
        )
            invgtest = false
        end
        if !all(
            isapprox.(
                g[p][:, :],
                [0.0625+yans[p]^2 xans[p]*yans[p]; xans[p]*yans[p] 0.0625+xans[p]^2],
                rtol=1e-14,
            ),
        )
            gtest = false
        end
    end
    @test invgtest
    @test gtest
    @test all(
        isapprox.(
            det_g,
            sqrt.([
                (0.0625 + yans[p]^2) * (0.0625 + xans[p]^2) - (xans[p] * yans[p])^2 for
                p in eachindex(det_g)
            ]),
            rtol=1e-14,
        ),
    )
end

# Same geometry as before, but using only 1 element. This way, all the expressions are just
# those coming from the mapping.
ddgeo(x) = (
    [
        [0.0 0.0]
        [0.0 0.0]
    ],
    [
        [0.0 0.0]
        [0.0 0.0]
    ],
    [
        [0.0 1.0]
        [1.0 0.0]
    ],
)
mapping2to3_ext = Mantis.Geometry.Mapping((2, 3), geo, dgeo, ddgeo)
geom_cart_ext = Geometry.CartesianGeometry((0.0:1.0:1.0, 0.0:1.0:1.0))
geometry2to3_ext = Mantis.Geometry.MappedGeometry(geom_cart_ext, mapping2to3_ext)

Jans(u, v) = [1.0 0.0; 0.0 1.0; v u]
gans(u, v) = [1.0+v^2 u*v; u*v 1.0+u^2]
sqrtgans(u, v) = sqrt(1.0 + u^2 + v^2)
ginvans(u, v) = (1.0 / (1.0 + u^2 + v^2)) .* [1.0+u^2 -u*v; -u*v 1.0+v^2]
Hans(u, v) = ([0.0 0.0; 0.0 0.0], [0.0 0.0; 0.0 0.0], [0.0 1.0; 1.0 0.0])
dgduans(u, v) = [0.0 v; v 2*u]
dgdvans(u, v) = [2*v u; u 0.0]
dginvgduans(u, v) =
    (1.0 / (1.0 + u^2 + v^2)^2) .*
    [2.0*u*v^2 -v + u^2 * v-v^3; -v + u^2 * v-v^3 -2 * u-2.0 * u * v^2]
dginvgdvans(u, v) =
    (1.0 / (1.0 + u^2 + v^2)^2) .*
    [-2 * v-2.0 * u^2 * v -u + u * v^2-u^3; -u + u * v^2-u^3 2.0*u^2*v]

xi = Points.CartesianPoints((LinRange(0.0, 1.0, 8), LinRange(0.0, 1.0, 12)))
J, inv_g, g, sqrt_g, dgdxs, dinv_g_dxs, dsqrt_g_dxs, Hs = Geometry.metric_derivatives(
    geometry2to3_ext, 1, xi
)
for p in eachindex(xi)
    @test all(isapprox.(J[p], Jans(xi[p]...), rtol=1e-14))
    @test all(isapprox.(g[p], gans(xi[p]...), rtol=1e-14))
    @test all(isapprox.(sqrt_g[p], sqrtgans(xi[p]...), rtol=1e-14))
    @test all(isapprox.(inv_g[p], ginvans(xi[p]...), rtol=1e-14))
    @test all(isapprox.(Hs[p][1], Hans(xi[p]...)[1], rtol=1e-14))
    @test all(isapprox.(Hs[p][2], Hans(xi[p]...)[2], rtol=1e-14))
    @test all(isapprox.(Hs[p][3], Hans(xi[p]...)[3], rtol=1e-14))
    @test all(isapprox.(dgdxs[1][p], dgduans(xi[p]...), rtol=1e-14))
    @test all(isapprox.(dgdxs[2][p], dgdvans(xi[p]...), rtol=1e-14))
    @test all(isapprox.(dinv_g_dxs[1][p], dginvgduans(xi[p]...), rtol=1e-12))
    @test all(isapprox.(dinv_g_dxs[2][p], dginvgdvans(xi[p]...), rtol=1e-12))
end

# Same geometry as before, but using multiple elements.
mapping2to3_ext2 = Mantis.Geometry.Mapping((2, 3), geo, dgeo, ddgeo)
geom_cart_ext2 = Geometry.CartesianGeometry((0.0:(1.0 / 3):1.0, 0.0:(1.0 / 4):1.0))
geometry2to3_ext2 = Mantis.Geometry.MappedGeometry(geom_cart_ext2, mapping2to3_ext2)

Jans(u, v) = [1.0/3 0.0; 0.0 1.0/4; v/3 u/4]
gans(u, v) = [(1.0 + v^2)/9 (u * v)/12; (u * v)/12 (1.0 + u^2)/16]
sqrtgans(u, v) = sqrt(1.0 + u^2 + v^2) * 1 / 12.0
ginvans(u, v) =
    (144.0 / (1.0 + u^2 + v^2)) .* [(1.0 + u^2)/16 (-u * v)/12; (-u * v)/12 (1.0 + v^2)/9]
Hans(u, v) = ([0.0 0.0; 0.0 0.0], [0.0 0.0; 0.0 0.0], [0.0 1.0/12; 1.0/12 0.0])
dgduans(u, v) = [0.0 v/36; v/36 (2 * u)/48]
dgdvans(u, v) = [(2 * v)/36 u/48; u/48 0.0]
dginvgduans(u, v) =
    (144.0 / (1.0 + u^2 + v^2)^2) .* [
        (2.0 * u * v^2) / 16/3 (-v + u^2 * v - v^3) / 12/3
        (-v + u^2 * v - v^3) / 12/3 (-2 * u - 2.0 * u * v^2) / 9/3
    ]
dginvgdvans(u, v) =
    (144.0 / (1.0 + u^2 + v^2)^2) .* [
        (-2 * v - 2.0 * u^2 * v) / 16/4 (-u + u * v^2 - u^3) / 12/4
        (-u + u * v^2 - u^3) / 12/4 (2.0 * u^2 * v) / 9/4
    ]

xi = Points.CartesianPoints((LinRange(0.0, 1.0, 8), LinRange(0.0, 1.0, 12)))
for (k, IJ) in enumerate(CartesianIndices((3, 4)))
    J, inv_g, g, sqrt_g, dgdxs, dinv_g_dxs, dsqrt_g_dxs, Hs = Geometry.metric_derivatives(
        geometry2to3_ext2, k, xi
    )
    uv = Geometry.evaluate(geom_cart_ext2, k, xi)
    for p in eachindex(xi)
        u, v = uv[p, :]
        @test all(isapprox.(J[p], Jans(u, v), rtol=1e-12))
        @test all(isapprox.(g[p], gans(u, v), rtol=1e-14))
        @test all(isapprox.(sqrt_g[p], sqrtgans(u, v), rtol=1e-14))
        @test all(isapprox.(inv_g[p], ginvans(u, v), rtol=1e-14))
        @test all(isapprox.(Hs[p][1], Hans(u, v)[1], rtol=1e-14))
        @test all(isapprox.(Hs[p][2], Hans(u, v)[2], rtol=1e-14))
        @test all(isapprox.(Hs[p][3], Hans(u, v)[3], rtol=1e-14))
        @test all(isapprox.(dgdxs[1][p], dgduans(u, v), rtol=1e-14))
        @test all(isapprox.(dgdxs[2][p], dgdvans(u, v), rtol=1e-14))
        @test all(isapprox.(dinv_g_dxs[1][p], dginvgduans(u, v), rtol=1e-12))
        @test all(isapprox.(dinv_g_dxs[2][p], dginvgdvans(u, v), rtol=1e-12))
    end
end

# Testing geometries defined in GeometryHelpers
# Cartesian Box
cart_box = Geometry.create_cartesian_box((0.0, 0.0), (1.0, 1.0), (7, 8))
Jans_cart_box(u, v) = [1.0/7.0 0.0; 0.0 1.0/8.0]
gans_cart_box(u, v) = [1.0/49.0 0.0; 0.0 1.0/64.0]
sqrtgans_cart_box(u, v) = sqrt((1.0 / 49.0) * (1.0 / 64.0))
ginvans_cart_box(u, v) = [49.0 0.0; 0.0 64.0]
Hans_cart_box(u, v) = ([0.0 0.0; 0.0 0.0], [0.0 0.0; 0.0 0.0])
dgduans_cart_box(u, v) = [0.0 0.0; 0.0 0.0]
dgdvans_cart_box(u, v) = [0.0 0.0; 0.0 0.0]
dginvgduans_cart_box(u, v) = [0.0 0.0; 0.0 0.0]
dginvgdvans_cart_box(u, v) = [0.0 0.0; 0.0 0.0]

xi = Points.CartesianPoints((LinRange(0.0, 1.0, 8), LinRange(0.0, 1.0, 12)))
J, inv_g, g, sqrt_g, dgdxs, dinv_g_dxs, dsqrt_g_dxs, Hs = Geometry.metric_derivatives(
    cart_box, 1, xi
)
for p in eachindex(xi)
    @test all(isapprox.(J[p], Jans_cart_box(xi[p]...), rtol=1e-14))
    @test all(isapprox.(g[p], gans_cart_box(xi[p]...), rtol=1e-14))
    @test all(isapprox.(sqrt_g[p], sqrtgans_cart_box(xi[p]...), rtol=1e-14))
    @test all(isapprox.(inv_g[p], ginvans_cart_box(xi[p]...), rtol=1e-14))
    @test all(isapprox.(Hs[p][1], Hans_cart_box(xi[p]...)[1], rtol=1e-14))
    @test all(isapprox.(Hs[p][2], Hans_cart_box(xi[p]...)[2], rtol=1e-14))
    @test all(isapprox.(dgdxs[1][p], dgduans_cart_box(xi[p]...), rtol=1e-14))
    @test all(isapprox.(dgdxs[2][p], dgdvans_cart_box(xi[p]...), rtol=1e-14))
    @test all(isapprox.(dinv_g_dxs[1][p], dginvgduans_cart_box(xi[p]...), rtol=1e-12))
    @test all(isapprox.(dinv_g_dxs[2][p], dginvgdvans_cart_box(xi[p]...), rtol=1e-12))
end

# Curvilinear geometry. Note that we cannot test the metric derivatives here, as the second
# derivative of the mapping is not given in the helper.
const c = 0.1
const l = 0.0
const r = 1.0
const b = 0.0
const t = 1.0
const hx = 1.0 / 3
const hy = 1.0 / 4
square_of_curv = Geometry.create_cartesian_box((l, b), (r - l, t - b), (3, 4))
geometrycurv = Geometry.create_curvilinear_square((l, b), (r - l, t - b), (3, 4); c=c)

Jans_curv(u, v) = [
    (1.0 + pi * c * cospi(u) * sinpi(v))*hx (r - l)/(t - b)*hy*pi*c*sinpi(u)*cospi(v)
    (t - b)/(r - l)*hx*pi*c*cospi(u)*sinpi(v) (1.0 + pi * c * sinpi(u) * cospi(v))*hy
]
gans_curv(u, v) = [
    (Jans_curv(u, v)[1, 1])^2+(Jans_curv(u, v)[2, 1])^2 Jans_curv(u, v)[1, 1] * Jans_curv(u, v)[1, 2]+Jans_curv(u, v)[2, 1] * Jans_curv(u, v)[2, 2]
    Jans_curv(u, v)[1, 1] * Jans_curv(u, v)[1, 2]+Jans_curv(u, v)[2, 1] * Jans_curv(u, v)[2, 2] (Jans_curv(u, v)[1, 2])^2+(Jans_curv(u, v)[2, 2])^2
]
sqrtgans_curv(u, v) = sqrt(det(gans_curv(u, v)))
invgans_curv(u, v) = inv(gans_curv(u, v))

xi = Points.CartesianPoints((LinRange(0.0, 1.0, 8), LinRange(0.0, 1.0, 12)))
for (k, IJ) in enumerate(CartesianIndices((3, 4)))
    J = Geometry.jacobian(geometrycurv, k, xi)
    inv_g, g, sqrt_g = Geometry.inv_metric(geometrycurv, k, xi)
    uv = Geometry.evaluate(square_of_curv, k, xi)
    for p in eachindex(xi)
        u, v = uv[p, :]
        u = (2.0 / (r - l)) * uv[p, 1] - 2.0 * l / (r - l) - 1.0
        v = (2.0 / (t - b)) * uv[p, 2] - 2.0 * b / (t - b) - 1.0
        @test all(isapprox.(J[p], Jans_curv(u, v), rtol=1e-12))
        @test all(isapprox.(g[p], gans_curv(u, v), rtol=1e-12))
        @test all(isapprox.(sqrt_g[p], sqrtgans_curv(u, v), rtol=1e-12))
        @test all(isapprox.(inv_g[p], invgans_curv(u, v), rtol=1e-12))
    end
end

# Mapping applied to a mapping (non-zero base Hessian). The second mapping, in this case, is
# the inverse of the first, so that the composistion in total should be the identity.
geo_exp1(x) = SVector{2}(exp(x[1]) + exp(x[2]), x[2])
dgeo_exp1(x) = SMatrix{2, 2}(exp(x[1]), 0.0, exp(x[2]), 1.0)
ddgeo_exp1(x) = (SMatrix{2, 2}(exp(x[1]), 0.0, 0.0, exp(x[2])), zeros(SMatrix{2, 2}))
mapping_exp1 = Mantis.Geometry.Mapping((2, 2), geo_exp1, dgeo_exp1, ddgeo_exp1)
geom_cart_exp = Geometry.CartesianGeometry((0.0:1.0:1.0, 0.0:1.0:1.0))
geometry_exp1 = Mantis.Geometry.MappedGeometry(geom_cart_exp, mapping_exp1)

geo_exp2(x) = SVector{2}(log(x[1] - exp(x[2])), x[2])
dgeo_exp2(x) =
    SMatrix{2, 2}(1.0 / (x[1] - exp(x[2])), 0.0, -exp(x[2]) / (x[1] - exp(x[2])), 1.0)
ddgeo_exp2(x) = (
    SMatrix{2, 2}(
        -1.0 / (x[1] - exp(x[2]))^2,
        exp(x[2]) / (x[1] - exp(x[2]))^2,
        exp(x[2]) / (x[1] - exp(x[2]))^2,
        -x[1] * exp(x[2]) / (x[1] - exp(x[2]))^2,
    ),
    zeros(SMatrix{2, 2}),
)
mapping_exp2 = Mantis.Geometry.Mapping((2, 2), geo_exp2, dgeo_exp2, ddgeo_exp2)
geometry_exp12 = Mantis.Geometry.MappedGeometry(geometry_exp1, mapping_exp2)

Jans_exp12(u, v) = [1.0/1.0 0.0; 0.0 1.0/1.0]
gans_exp12(u, v) = [1.0/1.0 0.0; 0.0 1.0/1.0]
sqrtgans_exp12(u, v) = sqrt((1.0 / 1.0) * (1.0 / 1.0))
ginvans_exp12(u, v) = [1.0 0.0; 0.0 1.0]
Hans_exp12(u, v) = ([0.0 0.0; 0.0 0.0], [0.0 0.0; 0.0 0.0])
dgduans_exp12(u, v) = [0.0 0.0; 0.0 0.0]
dgdvans_exp12(u, v) = [0.0 0.0; 0.0 0.0]
dginvgduans_exp12(u, v) = [0.0 0.0; 0.0 0.0]
dginvgdvans_exp12(u, v) = [0.0 0.0; 0.0 0.0]

xi_exp12 = Points.CartesianPoints((LinRange(0.0, 1.0, 8), LinRange(0.0, 1.0, 12)))
for (k, IJ) in enumerate(CartesianIndices((1, 1)))
    J, inv_g, g, sqrt_g, dgdxs, dinv_g_dxs, dsqrt_g_dxs, Hs = Geometry.metric_derivatives(
        geometry_exp12, k, xi_exp12
    )
    uv = Geometry.evaluate(geometry_exp12, k, xi_exp12)
    for p in eachindex(xi_exp12)
        u, v = uv[p, :]
        @test all(isapprox.(J[p], Jans_exp12(u, v), rtol=1e-14, atol=1e-14))
        @test all(isapprox.(g[p], gans_exp12(u, v), rtol=1e-14, atol=1e-14))
        @test all(isapprox.(sqrt_g[p], sqrtgans_exp12(u, v), rtol=1e-14, atol=1e-14))
        @test all(isapprox.(inv_g[p], ginvans_exp12(u, v), rtol=1e-14, atol=1e-14))
        @test all(isapprox.(Hs[p][1], Hans_exp12(u, v)[1], rtol=1e-14, atol=1e-14))
        @test all(isapprox.(Hs[p][2], Hans_exp12(u, v)[2], rtol=1e-14, atol=1e-14))
        @test all(isapprox.(dgdxs[1][p], dgduans_exp12(u, v), rtol=1e-14, atol=1e-14))
        @test all(isapprox.(dgdxs[2][p], dgdvans_exp12(u, v), rtol=1e-14, atol=1e-14))
        @test all(
            isapprox.(dinv_g_dxs[1][p], dginvgduans_exp12(u, v), rtol=1e-14, atol=1e-14)
        )
        @test all(
            isapprox.(dinv_g_dxs[2][p], dginvgdvans_exp12(u, v), rtol=1e-14, atol=1e-14)
        )
    end
end

end

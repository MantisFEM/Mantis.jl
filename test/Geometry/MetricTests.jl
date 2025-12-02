module MetricTests

using Mantis

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

# Surface embedded in 3D
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

xi = Points.CartesianPoints(([0.0, 1.0], [0.0, 1.0]))
J, inv_g, g, sqrt_g, dgdxs, dinv_g_dxs, dsqrt_g_dxs, Hs = Geometry.metric_derivatives(
    geometry2to3_ext, 1, xi
)
for p in eachindex(xi)
    @test all(isapprox.(J[p], Jans(xi[p]...), rtol=1e-14))
end

end

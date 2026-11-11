module GeometryHelpersTests

using Mantis

using LinearAlgebra
using Test

# Points of the canonical element at which the derivatives are checked.
function canonical_points(manifold_dim)
    return Points.TensorProductPoints(ntuple(_ -> [0.0, 0.3, 1.0], manifold_dim))
end

# Compares the Jacobian and the Hessian of `geometry` with the chain rule applied to the
# analytical derivatives `dΦ` and `ddΦ` of its mapping (functions of the parametric point).
function check_derivatives(geometry, element_id, dΦ, ddΦ)
    manifold_dim = Geometry.get_manifold_dim(geometry)
    ξ = canonical_points(manifold_dim)
    parametric_geometry = Geometry.get_base_geometry(geometry)
    x̂ = Geometry.evaluate(parametric_geometry, element_id, ξ)
    Dφ = Diagonal(collect(Geometry.get_element_lengths(parametric_geometry, element_id)))
    J = Geometry.jacobian(geometry, element_id, ξ)
    H = Geometry.hessian(geometry, element_id, ξ)
    for i in eachindex(J, H)
        @test J[i] ≈ dΦ(x̂[i, :]) * Dφ
        for k in eachindex(H[i])
            @test H[i][k] ≈ Dφ * ddΦ(x̂[i, :])[k] * Dφ
        end
    end
end

# Analytical derivatives of the Archimedean spiral (θ ↦ r(θ)(cos θ, sin θ)).
function d_spiral(x, r₀, a)
    θ, r = x[1], r₀ + a * x[1]
    return reshape([a * cos(θ) - r * sin(θ), a * sin(θ) + r * cos(θ)], 2, 1)
end
function dd_spiral(x, r₀, a)
    θ, r = x[1], r₀ + a * x[1]
    return ([-2a * sin(θ) - r * cos(θ);;], [2a * cos(θ) - r * sin(θ);;])
end

# Analytical derivatives of the cylindrical mapping (r, θ[, ζ]) ↦ (r cos θ, r sin θ[, ζ + cθ]),
# with manifold dimension n and image dimension m.
function d_cylindrical(x, c, n, m)
    r, θ = x[1], x[2]
    J = zeros(m, n)
    J[1, 1], J[1, 2] = cos(θ), -r * sin(θ)
    J[2, 1], J[2, 2] = sin(θ), r * cos(θ)
    if m == 3
        J[3, 2] = c
        n == 3 && (J[3, 3] = 1.0)
    end
    return J
end
function dd_cylindrical(x, n, m)
    r, θ = x[1], x[2]
    H = ntuple(_ -> zeros(n, n), m)
    H[1][1, 2], H[1][2, 1], H[1][2, 2] = -sin(θ), -sin(θ), -r * cos(θ)
    H[2][1, 2], H[2][2, 1], H[2][2, 2] = cos(θ), cos(θ), -r * sin(θ)
    return H
end

# The measure of the whole geometry, computed with a Gauss-Legendre rule.
function compute_measure(geometry, num_points)
    manifold_dim = Geometry.get_manifold_dim(geometry)
    quadrature_rule = Quadrature.tensor_product_rule(
        ntuple(_ -> num_points, manifold_dim), Quadrature.gauss_legendre
    )
    nodes = Quadrature.get_nodes(quadrature_rule)
    weights = Quadrature.get_weights(quadrature_rule)
    measure = 0.0
    for element_id in 1:Geometry.get_num_elements(geometry)
        _, sqrt_g = Geometry.metric(geometry, element_id, nodes)
        measure += sum(weights .* sqrt_g)
    end

    return measure
end

first_point(geometry) = Geometry.evaluate(
    geometry,
    1,
    Points.TensorProductPoints(ntuple(_ -> [0.0], Geometry.get_manifold_dim(geometry))),
)[
    1, :,
]
last_point(geometry) = Geometry.evaluate(
    geometry,
    Geometry.get_num_elements(geometry),
    Points.TensorProductPoints(ntuple(_ -> [1.0], Geometry.get_manifold_dim(geometry))),
)[
    1, :,
]

@testset "Archimedean spiral" begin
    r₀, a, θ_max = 0.5, 1 / (4π), 4π
    spiral = Geometry.create_archimedean_spiral(16)
    @test Geometry.get_manifold_dim(spiral) == 1
    @test Geometry.get_image_dim(spiral) == 2
    @test Geometry.get_num_elements(spiral) == 16
    @test first_point(spiral) ≈ [r₀, 0.0]
    @test isapprox(last_point(spiral), [r₀ + a * θ_max, 0.0]; atol=1e-14)
    for element_id in (1, 7, 16)
        check_derivatives(
            spiral, element_id, x -> d_spiral(x, r₀, a), x -> dd_spiral(x, r₀, a)
        )
    end
    u₀, u₁ = r₀ / a, r₀ / a + θ_max
    F(u) = u * sqrt(1 + u^2) + asinh(u)
    @test compute_measure(spiral, 6) ≈ a / 2 * (F(u₁) - F(u₀))

    other = Geometry.create_archimedean_spiral(
        3; inner_radius=1, radial_growth=0.1, angle=π
    )
    @test isapprox(last_point(other), [-(1 + 0.1π), 0.0]; atol=1e-14)

    @test_throws ArgumentError Geometry.create_archimedean_spiral(4; inner_radius=0.0)
    @test_throws ArgumentError Geometry.create_archimedean_spiral(4; radial_growth=-1.0)
    @test_throws ArgumentError Geometry.create_archimedean_spiral(4; angle=0.0)
end

@testset "Annulus sector" begin
    sector = Geometry.create_annulus_sector((4, 12))
    @test Geometry.get_manifold_dim(sector) == 2
    @test Geometry.get_image_dim(sector) == 2
    @test Geometry.get_num_elements(sector) == 48
    @test first_point(sector) ≈ [1.0, 0.0]
    @test isapprox(last_point(sector), [-2.0, 0.0]; atol=1e-14)
    for element_id in (1, 20, 48)
        check_derivatives(
            sector,
            element_id,
            x -> d_cylindrical(x, 0.0, 2, 2),
            x -> dd_cylindrical(x, 2, 2),
        )
        J = Geometry.jacobian(sector, element_id, canonical_points(2))
        @test all(det.(J) .> 0.0)
    end
    @test compute_measure(sector, 2) ≈ 3π / 2

    quarter = Geometry.create_annulus_sector((2, 3); radii=(0.5, 1.5), angle=π / 2)
    @test compute_measure(quarter, 2) ≈ π / 4 * (1.5^2 - 0.5^2)

    @test_throws ArgumentError Geometry.create_annulus_sector((2, 2); radii=(2.0, 1.0))
    @test_throws ArgumentError Geometry.create_annulus_sector((2, 2); radii=(0.0, 1.0))
    @test_throws ArgumentError Geometry.create_annulus_sector((2, 2); angle=3π)
end

@testset "Helicoid" begin
    c = 1 / π
    helicoid = Geometry.create_helicoid((4, 24))
    @test Geometry.get_manifold_dim(helicoid) == 2
    @test Geometry.get_image_dim(helicoid) == 3
    @test Geometry.get_num_elements(helicoid) == 96
    @test first_point(helicoid) ≈ [1.0, 0.0, 0.0]
    @test isapprox(last_point(helicoid), [2.0, 0.0, 2.0]; atol=1e-14)
    for element_id in (1, 50, 96)
        check_derivatives(
            helicoid,
            element_id,
            x -> d_cylindrical(x, c, 2, 3),
            x -> dd_cylindrical(x, 2, 3),
        )
    end
    F(r) = r / 2 * sqrt(r^2 + c^2) + c^2 / 2 * log(r + sqrt(r^2 + c^2))
    @test compute_measure(helicoid, 6) ≈ 2π * (F(2.0) - F(1.0))

    @test_throws ArgumentError Geometry.create_helicoid((2, 2); radii=(1.0, 1.0))
    @test_throws ArgumentError Geometry.create_helicoid((2, 2); angle=-1.0)
    @test_throws ArgumentError Geometry.create_helicoid((2, 2); angle=3π, rise_per_turn=0.0)
    # Without rise, one turn is still injective (an annulus sector lying flat).
    @test Geometry.create_helicoid((2, 2); rise_per_turn=0.0) isa Geometry.MappedGeometry
end

@testset "Helical duct" begin
    duct = Geometry.create_helical_duct((2, 36, 2))
    @test Geometry.get_manifold_dim(duct) == 3
    @test Geometry.get_image_dim(duct) == 3
    @test Geometry.get_num_elements(duct) == 144
    @test first_point(duct) ≈ [1.0, 0.0, 0.0]
    @test isapprox(last_point(duct), [-2.0, 0.0, 4.0]; atol=1e-14)
    for element_id in (1, 77, 144)
        check_derivatives(
            duct,
            element_id,
            x -> d_cylindrical(x, 1 / π, 3, 3),
            x -> dd_cylindrical(x, 3, 3),
        )
        J = Geometry.jacobian(duct, element_id, canonical_points(3))
        @test all(det.(J) .> 0.0)
    end
    @test compute_measure(duct, 2) ≈ 9π / 2

    @test_throws ArgumentError Geometry.create_helical_duct((2, 2, 2); thickness=0.0)
    @test_throws ArgumentError Geometry.create_helical_duct((2, 2, 2); thickness=2.0)
    # Thick ducts are fine as long as they do not complete a turn.
    @test Geometry.create_helical_duct((2, 2, 2); thickness=3.0, angle=π) isa
        Geometry.MappedGeometry
end

end

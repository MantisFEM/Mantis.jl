module NormsTests

using Mantis
using Test

include("../TestHelpers.jl")

import Mantis.Analysis: L2_norm, L2_norm_per_element, Linf_norm, Linf_norm_per_element

# Geometry: [0, 2] x [0, 1] with 4 x 3 elements, so the domain has area 2.
breakpoints = (range(0.0, 2.0; length=5), range(0.0, 1.0; length=4))
geom = Geometry.CartesianGeometry(breakpoints)
num_elements = Geometry.get_num_elements(geom)
domain_area = 2.0
sqrt_domain_area = sqrt(domain_area)
element_area = domain_area / num_elements
num_eval_points = 5
xi = Points.TensorProductPoints((
    LinRange(0.0, 1.0, num_eval_points), LinRange(0.0, 1.0, num_eval_points)
))
xi101 = Points.TensorProductPoints((LinRange(0.0, 1.0, 101), LinRange(0.0, 1.0, 101)))
dΩ = Quadrature.StandardQuadrature(
    Quadrature.tensor_product_rule((4, 4), Quadrature.gauss_legendre), num_elements
)

constant_form(c) = Forms.AnalyticalFormField(0, x -> [fill(c, size(x, 1))], geom, "f")
# f(x, y) = x
linear_form = Forms.AnalyticalFormField(0, x -> [x[:, 1]], geom, "f")
# f(x, y) = sin(2πx) sin(πy) - changes sign and is not polynomial.
oscillating_form = Forms.AnalyticalFormField(
    0, x -> [sinpi.(2.0 .* x[:, 1]) .* sinpi.(x[:, 2])], geom, "f"
)

approx_wtol(a, b) = isapprox(a, b; rtol=1e-14)

@testset "L2 norm" begin
    @testset "Values" begin
        # ‖c‖₂ = |c| √|Ω|
        expected = Dict(c => abs(c) * sqrt_domain_area for c in (0.0, 1.0, -1.0, 2.5, -3.0))
        @test @expected expected c -> L2_norm(constant_form(c), dΩ) approx_wtol

        # ∫₀² ∫₀¹ x² dy dx = 8/3
        @test isapprox(L2_norm(linear_form, dΩ), sqrt(8 / 3); rtol=1e-14)

        # ∫₀² sin²(2πx) dx ⋅ ∫₀¹ sin²(πy) dy = 1 ⋅ 1/2
        @test isapprox(L2_norm(oscillating_form, dΩ), sqrt(1 / 2); rtol=1e-3)
    end

    @testset "Properties" begin
        # Positivity (also for the zero function).
        @test L2_norm(oscillating_form, dΩ) > 0.0
        @test L2_norm(constant_form(0.0), dΩ) == 0.0
        @test L2_norm(constant_form(-2.0), dΩ) > 0.0

        # Absolute homogeneity: ‖αf‖ = |α| ‖f‖.
        expected = Dict(
            a => abs(a) * L2_norm(linear_form, dΩ) for a in (-4.0, -0.5, 2.0, 10.0)
        )
        @test @expected expected a -> L2_norm(a * linear_form, dΩ) approx_wtol

        # Triangle inequality: ‖f + g‖ ≤ ‖f‖ + ‖g‖.
        @test L2_norm(oscillating_form + linear_form, dΩ) <=
            L2_norm(oscillating_form, dΩ) + L2_norm(linear_form, dΩ)
    end
end

@testset "L2 norm per element" begin
    @testset "Constant functions" begin
        # ‖c‖₂,K = |c| √|K|
        expected = Dict(
            c => fill(abs(c) * sqrt(element_area), num_elements) for
            c in (0.0, 1.0, -2.0, 3.5)
        )
        @test @expected expected c -> L2_norm_per_element(constant_form(c), dΩ) approx_wtol
    end

    @testset "Properties" begin
        for f in (constant_form(-1.5), linear_form, oscillating_form)
            per_element = L2_norm_per_element(f, dΩ)
            @test length(per_element) == num_elements
            # Positivity.
            @test all(per_element .>= 0.0)

            # The squared per-element norms add up to the squared global norm.
            @test sqrt(sum(per_element .^ 2)) ≈ L2_norm(f, dΩ)
        end
    end
end

@testset "Linf norm" begin
    @testset "Values" begin
        expected = Dict(c => abs(c) for c in (0.0, 1.0, -1.0, 2.5, -3.0))
        @test @expected expected c -> Linf_norm(constant_form(c), xi)
        @test @expected expected c -> Linf_norm(constant_form(c), dΩ)

        # max |x| over [0, 2] is attained at the boundary, which is an evaluation point.
        @test isapprox(Linf_norm(linear_form, xi), 2.0, rtol=1e-14)
        # However, if we're use the points from the Gauss rule, the boundary maximum is not
        # found. It should find the largest Gauss node (mapped to the last element) instead.
        nodes = Quadrature.get_nodes(Quadrature.get_canonical_quadrature_rule(dΩ))
        @test isapprox(
            Linf_norm(linear_form, dΩ), 1.5 + maximum(nodes)[1] * 0.5, rtol=1e-14
        )

        # sin(2πx) sin(πy) has maximum 1 at (1/4, 1/2), which is not exactly an evaluation
        # point, so the computed norm slightly underestimates it.
        value = Linf_norm(oscillating_form, xi101)
        @test value <= 1.0
        @test isapprox(value, 1.0; rtol=1e-12)
    end

    @testset "Properties" begin
        f = oscillating_form
        # Positivity (also for the zero function).
        @test Linf_norm(f, xi) > 0.0
        @test Linf_norm(constant_form(0.0), xi) == 0.0
        @test Linf_norm(constant_form(-2.0), xi) > 0.0
        @test Linf_norm(f, dΩ) > 0.0
        @test Linf_norm(constant_form(0.0), dΩ) == 0.0
        @test Linf_norm(constant_form(-2.0), dΩ) > 0.0

        # Absolute homogeneity.
        g = linear_form
        expected = Dict(a => abs(a) * Linf_norm(g, xi) for a in (-4.0, -0.5, 2.0))
        @test @expected expected a -> Linf_norm(a * g, xi) approx_wtol
        expected = Dict(a => abs(a) * Linf_norm(g, dΩ) for a in (-4.0, -0.5, 2.0))
        @test @expected expected a -> Linf_norm(a * g, dΩ) approx_wtol

        # Triangle inequality.
        @test Linf_norm(f + g, xi) <= Linf_norm(f, xi) + Linf_norm(g, xi)
        @test Linf_norm(f + g, xi) <= Linf_norm(f, dΩ) + Linf_norm(g, dΩ)

        # The L² norm is bounded by the L∞ norm times √|Ω|.
        @test L2_norm(f, dΩ) <= sqrt_domain_area * Linf_norm(f, xi101)
    end
end

@testset "Linf norm per element" begin
    @testset "Values" begin
        expected = Dict(c => fill(abs(c), num_elements) for c in (0.0, 1.0, -2.0, 3.5))
        @test @expected expected c -> Linf_norm_per_element(constant_form(c), xi) approx_wtol
        @test @expected expected c -> Linf_norm_per_element(constant_form(c), dΩ) approx_wtol

        # On each element the maximum of |x| is the right element boundary.
        per_element = Linf_norm_per_element(linear_form, xi)
        x_right = breakpoints[1][2:end]

        # Elements are ordered with the first dimension running fastest.
        expected = repeat(x_right, length(breakpoints[2]) - 1)
        @test all(isapprox.(per_element, expected; rtol=1e-14))
    end

    @testset "Properties" begin
        for f in (constant_form(-1.5), linear_form, oscillating_form)
            per_element = Linf_norm_per_element(f, xi)
            @test length(per_element) == num_elements
            # Positivity.
            @test all(per_element .>= 0.0)
            # The global norm is the maximum of the per-element norms.
            @test isapprox(maximum(per_element), Linf_norm(f, xi); rtol=1e-14)

            # The same tests using the nodes from the quadrature rule.
            per_element_dΩ = Linf_norm_per_element(f, dΩ)
            @test length(per_element_dΩ) == num_elements
            @test all(per_element_dΩ .>= 0.0)
            @test isapprox(maximum(per_element_dΩ), Linf_norm(f, dΩ); rtol=1e-14)
        end
    end
end

end

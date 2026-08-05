module GeneralizedSpacesTests

using Mantis

using Test

rtol = 1e-9
atol = 1e-11
degrees_to_test = 3:7

# Test the GeneralizedTrigonometric space on [0, L] ---------------------------
Wt = pi / 2
for L in [1.0, 0.5]
    for p in degrees_to_test
        q = max(2, ceil(Int, (p + 1) / 2))
        quad_rule = Quadrature.gauss_legendre(q)
        x = Quadrature.get_nodes(quad_rule)
        w = Quadrature.get_weights(quad_rule)

        sum_all = zeros(Points.get_num_points(x))
        sum_all2 = zeros(Points.get_num_points(x))

        b = FunctionSpaces.GeneralizedTrigonometric(p, Wt, L)
        b_eval = FunctionSpaces.evaluate(b, x, 1)

        # Positivity of the polynomials
        @test minimum(b_eval[1][1]) >= 0.0

        # Partition of unity
        @test all(isapprox.(sum(b_eval[1][1]; dims=2), 1.0, atol=atol, rtol=rtol))

        # Zero sum of derivatives
        @test all(isapprox.(abs.(sum(b_eval[2][1]; dims=2)), 0.0, atol=atol, rtol=rtol))

        # interpolate a function and check derivatives
        # f = cos(Wt x) + sin(Wt x)
        # interpolate via collocation at following points
        q = p + 1
        x = collect(LinRange(0.0, 1.0, q))
        if p > 2
            # trigonometric target
            f_eval = cos.(Wt * x * L) + sin.(Wt * x * L)
            df_dx_eval = -Wt * sin.(Wt * x * L) + Wt * cos.(Wt * x * L)
            d2f_dx2_eval = -Wt * Wt * cos.(Wt * x * L) - Wt * Wt * sin.(Wt * x * L)

            # interpolate via collocation
            b_eval = FunctionSpaces.evaluate(b, Points.TensorProductPoints((x,)), 2)
            coeff_b = b_eval[1][1] \ f_eval

            # Check that the values match f ...
            @test all(isapprox.(b_eval[1][1] * coeff_b, f_eval, atol=atol, rtol=rtol))
            # ... the first order derivative matches df/dx ...
            @test all(
                isapprox.(b_eval[2][1] * coeff_b / L, df_dx_eval, atol=atol, rtol=rtol)
            )
            # ... and the second order derivative matches d2f/dx2.
            @test all(
                isapprox.(b_eval[3][1] * coeff_b / L^2, d2f_dx2_eval, atol=atol, rtol=rtol)
            )
        end
    end
end

# Test the GeneralizedExponential space on [0, L] ---------------------------
Wt = 10.0
for L in [1.0, 0.5]
    for p in degrees_to_test
        q = max(2, ceil(Int, (p + 1) / 2))
        quad_rule = Quadrature.gauss_legendre(q)
        x = Quadrature.get_nodes(quad_rule)
        w = Quadrature.get_weights(quad_rule)

        sum_all = zeros(Points.get_num_points(x))
        sum_all2 = zeros(Points.get_num_points(x))

        b = FunctionSpaces.GeneralizedExponential(p, Wt, L)
        b_eval = FunctionSpaces.evaluate(b, x, 1)

        # Positivity of the polynomials
        @test minimum(b_eval[1][1]) >= 0.0

        # Partition of unity
        @test all(isapprox.(sum(b_eval[1][1]; dims=2), 1.0, atol=atol, rtol=rtol))

        # Zero sum of derivatives
        @test all(isapprox.(abs.(sum(b_eval[2][1]; dims=2)), 0.0, atol=atol, rtol=rtol))

        # interpolate a function and check derivatives
        # f = cos(Wt x) + sin(Wt x)
        # interpolate via collocation at following points
        q = p + 1
        x = collect(LinRange(0.0, 1.0, q))
        if p > 2
            # exponential target
            f_eval = exp.(Wt * x * L) + 2 .* exp.(-Wt * x * L)
            df_dx_eval = Wt .* exp.(Wt * x * L) - Wt .* 2 .* exp.(-Wt * x * L)
            d2f_dx2_eval = Wt .* Wt .* exp.(Wt * x * L) + Wt .* Wt .* 2 .* exp.(-Wt * x * L)

            # interpolate via collocation
            b_eval = FunctionSpaces.evaluate(b, Points.TensorProductPoints((x,)), 2)
            coeff_b = b_eval[1][1] \ f_eval

            # Check that the values match f ...
            @test all(isapprox.(b_eval[1][1] * coeff_b, f_eval, atol=atol, rtol=rtol))
            # ... the first order derivative matches df/dx ...
            @test all(
                isapprox.(b_eval[2][1] * coeff_b / L, df_dx_eval, atol=atol, rtol=rtol)
            )
            # ... and the second order derivative matches d2f/dx2.
            @test all(
                isapprox.(b_eval[3][1] * coeff_b / L^2, d2f_dx2_eval, atol=atol, rtol=rtol)
            )
        end
    end
end

end

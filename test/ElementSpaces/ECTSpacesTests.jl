import Mantis

using Test

degrees_to_test = 3:7
Wt = pi/2

for p in degrees_to_test
    q = max(2, ceil(Int, (p+1)/2))
    x, w = Mantis.Quadrature.gauss_legendre(q)
    
    sum_all = zeros(size(x))
    sum_all2 = zeros(size(x))

    b = Mantis.ElementSpaces.GeneralizedTrigonometric(p, Wt)
    b_eval = Mantis.ElementSpaces.evaluate(b, x, 1)

    # Positivity of the polynomials
    @test minimum(b_eval[:,:,1]) >= 0.0

    # Partition of unity
    @test all(isapprox.(sum(b_eval[:,:,1], dims=2), 1.0))

    # Zero sum of derivatives
    @test all(isapprox.(abs.(sum(b_eval[:,:,2], dims=2)), 0.0, atol=1e-12))

    # interpolate a function and check derivatives
    # f = cos(Wt x) + sin(Wt x)
    # interpolate via collocation at following points
    q = p+1
    x = collect(LinRange(0.0,1.0,q))#Mantis.Quadrature.gauss_legendre(q)
    if p > 2
        # trigonometric component
        f_eval = cos.(Wt * x) + sin.(Wt * x)
        df_dx_eval = -Wt * sin.(Wt * x) + Wt * cos.(Wt * x)
        d2f_dx2_eval = -Wt * Wt * cos.(Wt * x) - Wt * Wt * sin.(Wt * x)
        
        # interpolate via collocation
        b_eval = Mantis.ElementSpaces.evaluate(b, x, 2)
        coeff_b = b_eval[:,:,1] \ f_eval

        # Check that the values match f ...
        @test isapprox(maximum(abs.(b_eval[:,:,1] * coeff_b .- f_eval)), 0.0, atol = 1e-14)
        # ... the first order derivative matches df/dx ...
        @test isapprox(maximum(abs.(b_eval[:,:,2] * coeff_b .- df_dx_eval)), 0.0, atol = 2e-12)
        # ... and the second order derivative matches d2f/dx2.
        @test isapprox(maximum(abs.(b_eval[:,:,3] * coeff_b .- d2f_dx2_eval)), 0.0, atol = 1e-10)
    end
end
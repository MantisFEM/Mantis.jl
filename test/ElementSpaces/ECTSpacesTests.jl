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
    # f = alpha cos(Wt x) + beta sin(Wt x) + x^(p-2) + x^(p-3)
    # interpolate via collocation at following points
    q = p+1
    x, w = Mantis.Quadrature.gauss_legendre(q)
    if p > 2
        # polynomial component
        f_poly = (x::Float64,p::Int64,m::Int64) -> (p - m >= 0 ? 1.0 : 0.0) * prod(LinRange(p:-1:(p-m+1))) * (p-m > 0 ? x^(p-m) : 1.0) + (p - 1 - m >= 0 ? 1.0 : 0.0)* prod(LinRange((p-1):-1:(p-m))) * (p-1-m > 0 ? x^(p-1-m) : 1.0)
        # trigonometric component
        alpha = rand()
        beta = rand()
        f_trig_eval = alpha * cos.(Wt * x) + beta * sin.(Wt * x)
        df_dx_trig_eval = -Wt * alpha * sin.(Wt * x) + Wt * beta * cos.(Wt * x)
        d2f_dx2_trig_eval = -Wt * Wt * alpha * cos.(Wt * x) - Wt * Wt * beta * sin.(Wt * x)
        
        # full function
        f_eval = f_poly.(x,p-2,0) + f_trig_eval
        df_dx_eval = f_poly.(x,p-2,1) + df_dx_trig_eval
        d2f_dx2_eval = f_poly.(x,p-2,2) + d2f_dx2_trig_eval

        # interpolate via collocation
        b_eval = Mantis.ElementSpaces.evaluate(b, x, 2)
        coeff_b = b_eval[:,:,1] \ f_eval

        # Check that the values match f ...
        @test isapprox(maximum(abs.(b_eval[:,:,1] * coeff_b .- f_eval)), 0.0, atol = 1e-15)
        # ... the first order derivative matches df/dx ...
        @test isapprox(maximum(abs.(b_eval[:,:,2] * coeff_b .- df_dx_eval)), 0.0, atol = 1e-13)
        # ... and the second order derivative matches d2f/dx2.
        @test isapprox(maximum(abs.(b_eval[:,:,3] * coeff_b .- d2f_dx2_eval)), 0.0, atol = 1e-11)
    end
end
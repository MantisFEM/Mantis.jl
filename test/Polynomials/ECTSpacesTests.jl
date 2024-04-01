import Mantis

using Test

degrees_to_test = 3:7
Wt = 1.5

for p in degrees_to_test
    q = max(2, ceil(Int, (p+1)/2))
    x, w = Mantis.Quadrature.gauss_legendre(q)
    
    sum_all = zeros(size(x))
    sum_all2 = zeros(size(x))

    b = Mantis.Polynomials.GeneralizedTrigonometric(p, Wt)
    b_eval = Mantis.Polynomials.evaluate(b, x, 1)

    # Positivity of the polynomials
    @test minimum(b_eval[:,:,1]) >= 0.0

    # Partition of unity
    @test all(isapprox.(sum(b_eval[:,:,1], dims=2), 1.0))

    # Zero sum of derivatives
    @test all(isapprox.(abs.(sum(b_eval[:,:,2], dims=2)), 0.0, atol=1e-12))
end
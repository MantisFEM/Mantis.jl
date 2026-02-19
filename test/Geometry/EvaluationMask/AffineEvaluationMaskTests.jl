module AffineEvaluationMaskTests

using Mantis

using Test

################################################################################
# UNIVARIATE TESTS
################################################################################

# initialize a trivial evaluation mask
manifold_dim = 1
num_elements_base = 2

E₀ = Geometry.trivial_evaluation_mask(manifold_dim, num_elements_base)
@test Geometry.get_num_elements(E₀) == num_elements_base

npts = 4
xi = Points.PointSet((range(0.0, 1.0, npts),))
for element_idx in 1:Geometry.get_num_elements(E₀)
    element_idx_base, xi_base, scale = Geometry.transform_evaluation_points(
        E₀, element_idx, xi
    )
    @test element_idx_base == element_idx
    @test isapprox(
        collect(Points.get_constituent_points(xi_base)[1]),
        collect(Points.get_constituent_points(xi)[1]),
        atol=1e-12,
    )
    @test isapprox(scale[1], 1.0, atol=1e-12)
end

# perform a unifom subdivision of the evaluation mask
nsubd = 2
E₁ = Geometry.subdivide_evaluation_mask(E₀, nsubd)
@test Geometry.get_num_elements(E₁) == 2 * num_elements_base

npts = 4
xi = Points.PointSet((range(0.0, 1.0, npts),))
for element_idx in 1:Geometry.get_num_elements(E₀)
    for subd in 1:nsubd
        element_idx_base, xi_base, scale = Geometry.transform_evaluation_points(
            E₁, nsubd * (element_idx - 1) + subd, xi
        )
        @test element_idx_base == element_idx
        @test isapprox(
            collect(Points.get_constituent_points(xi_base)[1]),
            (collect(Points.get_constituent_points(xi)[1]) .+ (subd - 1)) ./ nsubd,
            atol=1e-12,
        )
        @test isapprox(scale[1], 1 / nsubd, atol=1e-12)
    end
end

# perform a non-unifom subdivision of the evaluation mask
nsubd = [2, 3]
E₁ = Geometry.subdivide_evaluation_mask(E₀, nsubd)
@test Geometry.get_num_elements(E₁) == sum(nsubd)

npts = 4
xi = Points.PointSet((range(0.0, 1.0, npts),))
for element_idx in 1:Geometry.get_num_elements(E₀)
    for subd in 1:nsubd[element_idx]
        element_idx_base, xi_base, scale = Geometry.transform_evaluation_points(
            E₁, sum(nsubd[1:(element_idx - 1)]) * (element_idx - 1) + subd, xi
        )
        @test element_idx_base == element_idx
        @test isapprox(
            collect(Points.get_constituent_points(xi_base)[1]),
            (collect(Points.get_constituent_points(xi)[1]) .+ (subd - 1)) ./
            nsubd[element_idx],
            atol=1e-12,
        )
        @test isapprox(scale[1], 1 / nsubd[element_idx], atol=1e-12)
    end
end

################################################################################
# BIVARIATE TESTS
################################################################################

# initialize a trivial evaluation mask
manifold_dim = 2
num_elements_base = 2

E₀ = Geometry.trivial_evaluation_mask(manifold_dim, num_elements_base)
@test Geometry.get_num_elements(E₀) == num_elements_base

npts = (2, 2)
xi = Points.PointSet((range(0.0, 1.0, npts[1]), range(0.0, 1.0, npts[2])))
for element_idx in 1:Geometry.get_num_elements(E₀)
    element_idx_base, xi_base, scale = Geometry.transform_evaluation_points(
        E₀, element_idx, xi
    )
    @test element_idx_base == element_idx
    @test isapprox(
        collect(Points.get_constituent_points(xi_base)[1]),
        collect(Points.get_constituent_points(xi)[1]),
        atol=1e-12,
    )
    @test isapprox(
        collect(Points.get_constituent_points(xi_base)[2]),
        collect(Points.get_constituent_points(xi)[2]),
        atol=1e-12,
    )
    @test isapprox(scale[1], 1.0, atol=1e-12)
    @test isapprox(scale[2], 1.0, atol=1e-12)
end

# perform a non-unifom subdivision of the evaluation mask
nsubd = [2, 3]
cumsubd = cumsum([0; nsubd .^ manifold_dim])
E₁ = Geometry.subdivide_evaluation_mask(E₀, nsubd)
@test Geometry.get_num_elements(E₁) == cumsubd[end]

npts = (2, 2)
xi = Points.PointSet((range(0.0, 1.0, npts[1]), range(0.0, 1.0, npts[2])))
for element_idx in 1:Geometry.get_num_elements(E₀)
    for subd_idx in Iterators.product(1:nsubd[element_idx], 1:nsubd[element_idx])
        element_idx_base, xi_base, scale = Geometry.transform_evaluation_points(
            E₁,
            cumsubd[element_idx] + subd_idx[1] + (subd_idx[2] - 1) * nsubd[element_idx],
            xi,
        )
        @test element_idx_base == element_idx
        @test isapprox(
            collect(Points.get_constituent_points(xi_base)[1]),
            (collect(Points.get_constituent_points(xi)[1]) .+ (subd_idx[1] - 1)) ./
            nsubd[element_idx],
            atol=1e-12,
        )
        @test isapprox(
            collect(Points.get_constituent_points(xi_base)[2]),
            (collect(Points.get_constituent_points(xi)[2]) .+ (subd_idx[2] - 1)) ./
            nsubd[element_idx],
            atol=1e-12,
        )
        @test isapprox(scale[1], 1 / nsubd[element_idx], atol=1e-12)
        @test isapprox(scale[2], 1 / nsubd[element_idx], atol=1e-12)
    end
end

end

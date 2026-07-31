module UnstructuredGeometryTests

using Mantis

using Test

# Refer to the following file for method and variable definitions.
include("GeometryTestsHelpers.jl")

# Constructor, property, and getters and setters tests -------------------------------------
function basic_tests(geometry, answers)
    @test Geometry.get_num_patches(geometry) == answers[1]
    @test Geometry.get_num_elements(geometry) == answers[2]
    @test Geometry.get_manifold_dim(geometry) == answers[3]
    @test Geometry.get_image_dim(geometry) == answers[4]
    @test all(Geometry.get_num_elements_per_patch(geometry) .== answers[5])
    @test Geometry.get_num_elements(geometry, 1) == answers[6]
    @test all(isapprox.(Geometry.get_element_lengths(geometry, 1), answers[7], rtol=1e-14))

    return nothing
end

# Reduction test, single-patch, single element, 1D.
geometry1 = Geometry.UnstructuredGeometry((Geometry.CartesianGeometry(([-1, 1],)),))
answers_1 = (1, 1, 1, 1, (1,), 1, (2.0,))
basic_tests(geometry1, answers_1)

# LinRange input. Single-patch, 2D.
cg1 = Geometry.CartesianGeometry((LinRange(0.5, 2.5, 5), LinRange(-0.75, 0.75, 3)))
cg2 = Geometry.CartesianGeometry((LinRange(2.5, 5.0, 6), LinRange(-0.75, 0.75, 5)))
geometry_2 = Geometry.UnstructuredGeometry((cg1, cg2))
answers_2 = (2, 28, 2, 2, (8, 20), 8, (0.5, 0.75))
basic_tests(geometry_2, answers_2)

xi = Points.CartesianPoints(([0.0, 1.0], [0.0, 1.0]))
for i in 1:Geometry.get_num_elements(geometry_2)
    evals = Geometry.evaluate(geometry_2, i, xi)
    jac = Geometry.jacobian(geometry_2, i, xi)
    if i <= 8
        evals_ans = Geometry.evaluate(cg1, i, xi)
        jac_ans = Geometry.jacobian(cg1, i, xi)
        for p in axes(evals, 1)
            @test all(isapprox.(evals[p, :, :], evals_ans[p, :, :], rtol=1e-14))
        end
        for p in axes(jac, 1)
            @test all(isapprox.(jac[p], jac_ans[p], rtol=1e-14))
        end
    else
        evals_ans = Geometry.evaluate(cg2, i-8, xi)
        jac_ans = Geometry.jacobian(cg2, i-8, xi)
        for p in axes(evals, 1)
            @test all(isapprox.(evals[p, :, :], evals_ans[p, :, :], rtol=1e-14))
        end
        for p in axes(jac, 1)
            @test all(isapprox.(jac[p], jac_ans[p], rtol=1e-14))
        end
    end
end

end

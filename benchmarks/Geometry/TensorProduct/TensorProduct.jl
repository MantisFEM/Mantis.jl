module EvaluateTensorProductGeometryBenchmarks

using Mantis

# Refer to the following file for method and variable definitions
include(joinpath(pwd(), "BenchmarkHelpers.jl"))
include(joinpath(mantis_dir, "examples", "HelperFunctions.jl"))

############################################################################################
#                                      Problem setup                                       #
############################################################################################

# These benchmarks are evaluations on a given element, so the total number of elements in
# the geometry is not relevant. Hence, we picked 10 elements per dimension.

function geo_and_eval_points(::Val{1})
    geo = Geometry.CartesianGeometry(
        ntuple(1) do m
            return LinRange(0.0, 1.0, 11)
        end,
    )
    eval_points = Points.CartesianPoints(
        ntuple(1) do m
            return LinRange(0.0, 1.0, 10)
        end,
    )
    return Geometry.TensorProductGeometry((geo,)), eval_points
end

function geo_and_eval_points(::Val{2})
    geo1 = Geometry.CartesianGeometry(
        ntuple(1) do m
            return LinRange(0.0, 1.0, 11)
        end,
    )
    geo2 = Geometry.CartesianGeometry(
        ntuple(1) do m
            return LinRange(0.0, 1.0, 11)
        end,
    )
    eval_points = Points.CartesianPoints(
        ntuple(2) do m
            return LinRange(0.0, 1.0, 10)
        end,
    )
    return Geometry.TensorProductGeometry((geo1, geo2)), eval_points
end

function geo_and_eval_points(::Val{3})
    geo1 = Geometry.CartesianGeometry(
        ntuple(1) do m
            return LinRange(0.0, 1.0, 11)
        end,
    )
    geo2 = Geometry.CartesianGeometry(
        ntuple(2) do m
            return LinRange(0.0, 1.0, 11)
        end,
    )
    eval_points = Points.CartesianPoints(
        ntuple(3) do m
            return LinRange(0.0, 1.0, 10)
        end,
    )
    return Geometry.TensorProductGeometry((geo1, geo2)), eval_points
end

function run_problem(::Val{manifold_dim}) where {manifold_dim}
    geo, eval_points = geo_and_eval_points(Val(manifold_dim))

    return @benchmarkable Geometry.evaluate($geo, 5, $eval_points) samples = 500 evals = 1 seconds =
        Inf
end

function run_problem_jacobian(::Val{manifold_dim}) where {manifold_dim}
    geo, eval_points = geo_and_eval_points(Val(manifold_dim))

    return @benchmarkable Geometry.jacobian($geo, 5, $eval_points) samples = 500 evals = 1 seconds =
        Inf
end

function run_problem_inv_metric(::Val{manifold_dim}) where {manifold_dim}
    geo, eval_points = geo_and_eval_points(Val(manifold_dim))

    return @benchmarkable Geometry.inv_metric($geo, 5, $eval_points) samples = 500 evals = 1 seconds =
        Inf
end

############################################################################################
#                                       Run problems                                       #
############################################################################################

group = BenchmarkGroup()

manifold_dims = (1, 2, 3)
for manifold_dim in manifold_dims
    benchmarkable = run_problem(Val(manifold_dim))
    group["evaluate-$(manifold_dim)D"] = benchmarkable

    benchmarkable_jac = run_problem_jacobian(Val(manifold_dim))
    group["jacobian-$(manifold_dim)D"] = benchmarkable_jac

    benchmarkable_inv = run_problem_inv_metric(Val(manifold_dim))
    group["inv-metric-$(manifold_dim)D"] = benchmarkable_inv
end

end

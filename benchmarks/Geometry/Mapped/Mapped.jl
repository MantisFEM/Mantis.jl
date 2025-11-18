module EvaluateMappedGeometryBenchmarks

using Mantis

# Refer to the following file for method and variable definitions
include(joinpath(pwd(), "BenchmarkHelpers.jl"))
include(joinpath(mantis_dir, "examples", "HelperFunctions.jl"))

############################################################################################
#                                      Problem setup                                       #
############################################################################################

# These benchmarks are evaluations on a given element, so the total number of elements in
# the geometry is not relevant. Hence, we picked 10 elements per dimension.

function create_1D_geometry(r=2.5)
    # build underlying Cartesian geometry
    unit_line = Geometry.create_cartesian_box((-1.0,), (1.0,), (10,))

    # build curved mapping
    function mapping(x::AbstractVector)
        return [r * cospi(x[1]), r * sinpi(x[1])]
    end
    function dmapping(x::AbstractVector)
        return [
            -r * pi * sin(x[1])
            r * pi * cos(x[1])
        ]
    end
    dimension = (1, 2)
    curved_mapping = Geometry.Mapping(dimension, mapping, dmapping)

    return Geometry.MappedGeometry(unit_line, curved_mapping)
end

function create_3D_geometry()
    # build underlying Cartesian geometry
    unit_box = Geometry.create_cartesian_box((0.5, 0.5, 0.5), (1.5, 1.5, 1.5), (10, 10, 10))

    # build curved mapping
    function mapping(x::AbstractVector)
        return [x[1]^3 + x[2]^2 + x[3], x[1] + x[2]^3 + x[3]^2, x[1] + x[2]^2 + x[3]^3]
    end
    function dmapping(x::AbstractVector)
        return [
            3*x[1]^2 2*x[2] 1.0
            1.0 3*x[2]^2 2*x[3]
            1.0 2*x[2] 3*x[3]^2
        ]
    end
    dimension = (3, 3)
    curved_mapping = Geometry.Mapping(dimension, mapping, dmapping)

    return Geometry.MappedGeometry(unit_box, curved_mapping)
end

function geo_and_eval_points(::Val{1})
    geo = create_1D_geometry()
    eval_points = Points.CartesianPoints(
        ntuple(1) do m
            return LinRange(0.0, 1.0, 10)
        end,
    )
    return geo, eval_points
end

function geo_and_eval_points(::Val{2})
    geo = Geometry.create_curvilinear_square((0.0, 0.0), (2.0, 2.0), (10, 10); crazy_c=0.2)
    eval_points = Points.CartesianPoints(
        ntuple(2) do m
            return LinRange(0.0, 1.0, 10)
        end,
    )
    return geo, eval_points
end

function geo_and_eval_points(::Val{3})
    geo = create_3D_geometry()
    eval_points = Points.CartesianPoints(
        ntuple(3) do m
            return LinRange(0.0, 1.0, 10)
        end,
    )
    return geo, eval_points
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

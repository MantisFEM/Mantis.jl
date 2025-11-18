module EvaluateAlgebraicBenchmarks

using Mantis

# Refer to the following file for method and variable definitions
include(joinpath(pwd(), "BenchmarkHelpers.jl"))
include(joinpath(mantis_dir, "examples", "HelperFunctions.jl"))

############################################################################################
#                                      Problem setup                                       #
############################################################################################

# These benchmarks are evaluations on a given element, so the total number of elements in
# the geometry is not relevant. Hence, we picked 10 elements per dimension.
const p = 3
const r = 0

function run_problem_unary(manifold_dim, op, args...)
    starting_points = ntuple(i -> 0.0, manifold_dim)
    box_sizes = ntuple(i -> 1.0, manifold_dim)
    num_elements = ntuple(i -> 10, manifold_dim)
    degrees = ntuple(i -> p, manifold_dim)
    regularities = ntuple(i -> r, manifold_dim)
    R = Forms.create_tensor_product_bspline_de_rham_complex(
        starting_points, box_sizes, num_elements, degrees, regularities
    )
    canonical_qrule = Quadrature.tensor_product_rule(
        degrees .+ 1, Quadrature.gauss_legendre
    )
    dΩ = Quadrature.StandardQuadrature(
        canonical_qrule, Geometry.get_num_elements(Forms.get_geometry(R[end]))
    )
    integral = Forms.Integral(R[end], dΩ)
    operator = op(args..., integral)

    return @benchmarkable Forms.evaluate($operator, 5) samples = 500 evals = 1 seconds = Inf
end

function run_problem_binary(manifold_dim, op)
    starting_points = ntuple(i -> 0.0, manifold_dim)
    box_sizes = ntuple(i -> 1.0, manifold_dim)
    num_elements = ntuple(i -> 10, manifold_dim)
    degrees = ntuple(i -> p, manifold_dim)
    regularities = ntuple(i -> r, manifold_dim)
    R = Forms.create_tensor_product_bspline_de_rham_complex(
        starting_points, box_sizes, num_elements, degrees, regularities
    )
    canonical_qrule = Quadrature.tensor_product_rule(
        degrees .+ 1, Quadrature.gauss_legendre
    )
    dΩ = Quadrature.StandardQuadrature(
        canonical_qrule, Geometry.get_num_elements(Forms.get_geometry(R[end]))
    )
    integral = Forms.Integral(R[end], dΩ)
    operator = op(integral, integral)

    return @benchmarkable Forms.evaluate($operator, 5) samples = 500 evals = 1 seconds = Inf
end

############################################################################################
#                                       Run problems                                       #
############################################################################################

group = BenchmarkGroup()

for manifold_dim in (1, 2, 3)
    group["unary-operator-evaluate-minus-form-$(manifold_dim)D"] = run_problem_unary(
        manifold_dim, -
    )

    group["unary-operator-evaluate-times-form-$(manifold_dim)D"] = run_problem_unary(
        manifold_dim, *, 5.0
    )
end

for manifold_dim in (1, 2, 3)
    group["binary-operator-evaluate-minus-form-$(manifold_dim)D"] = run_problem_binary(
        manifold_dim, -
    )

    group["binary-operator-evaluate-plus-form-$(manifold_dim)D"] = run_problem_binary(
        manifold_dim, +
    )
end

end

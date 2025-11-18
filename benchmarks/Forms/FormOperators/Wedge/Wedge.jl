module EvaluateWedgeBenchmarks

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

function run_problem(manifold_dim, form_rank, form_rank2)
    starting_points = ntuple(i -> 0.0, manifold_dim)
    box_sizes = ntuple(i -> 1.0, manifold_dim)
    num_elements = ntuple(i -> 10, manifold_dim)
    degrees = ntuple(i -> p, manifold_dim)
    regularities = ntuple(i -> r, manifold_dim)
    R = Forms.create_tensor_product_bspline_de_rham_complex(
        starting_points, box_sizes, num_elements, degrees, regularities
    )
    field = Forms.FormField(
        R[form_rank2 + 1], rand(Forms.get_num_basis(R[form_rank2 + 1])), "test"
    )
    wedge = Forms.Wedge(R[form_rank + 1], field)
    eval_points = Points.CartesianPoints(
        ntuple(manifold_dim) do m
            return LinRange(0.0, 1.0, 10)
        end,
    )

    return @benchmarkable Forms.evaluate($wedge, 5, $eval_points) samples = 500 evals = 1 seconds =
        Inf
end

############################################################################################
#                                       Run problems                                       #
############################################################################################

group = BenchmarkGroup()

for manifold_dim in (1, 2, 3)
    for r in 0:manifold_dim
        for r2 in 0:(manifold_dim - r)
            benchmarkable = run_problem(manifold_dim, r, r2)
            group["evaluate-$(r)-form-$(r2)-form-$(manifold_dim)D"] = benchmarkable
        end
    end
end

end

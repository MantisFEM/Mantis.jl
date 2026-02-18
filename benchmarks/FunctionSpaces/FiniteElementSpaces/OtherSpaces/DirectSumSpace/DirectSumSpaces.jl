module EvaluateDirectSumSpaceBenchmarks

using Mantis

# Refer to the following file for method and variable definitions
include(joinpath(pwd(), "BenchmarkHelpers.jl"))
include(joinpath(mantis_dir, "examples", "HelperFunctions.jl"))

############################################################################################
#                                      Problem setup                                       #
############################################################################################

function run_problem(manifold_dim::Int, num_components::Int, p::Int, k::Int)
    breakpoints = collect(LinRange(0.0, 1.0, 5))
    patch = Geometry.CartesianGeometry(breakpoints)
    regularity = fill(k, 5)
    regularity[1] = regularity[end] = -1
    Bpk = FunctionSpaces.BSplineSpace(patch, FunctionSpaces.Bernstein(p), regularity)
    TPspaces = ntuple(manifold_dim) do i
        return Bpk
    end
    TP = FunctionSpaces.TensorProductSpace(TPspaces)

    comp_spaces = ntuple(num_components) do i
        return TP
    end
    DS = FunctionSpaces.DirectSumSpace(comp_spaces)

    eval_points = Points.CartesianPoints(
        ntuple(dim -> LinRange(0.0, 1.0, 25), manifold_dim)
    )
    dim = FunctionSpaces.get_num_basis(DS)

    return dim,
    @benchmarkable FunctionSpaces.evaluate($DS, 1, $eval_points) samples = 100 evals = 1 seconds =
        Inf
end

############################################################################################
#                                       Run problems                                        #
############################################################################################

group = BenchmarkGroup()

num_components = [1, 2, 3]
manifold_dims = [2, 3]
for manifold_dim in manifold_dims
    sub_group = BenchmarkGroup()

    for num_comp in num_components
        dim, benchmarkable = run_problem(manifold_dim, num_comp, 3, 2)
        name = "num_components=$(num_comp)-num_basis=$(dim)-p=$(3)-k=$(2)"
        sub_group[name] = benchmarkable
    end

    group["$(manifold_dim)D"] = sub_group
end

end

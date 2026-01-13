module EvaluateTensorProductSpacesBenchmarks

using Mantis

# Refer to the following file for method and variable definitions
include(joinpath(pwd(), "BenchmarkHelpers.jl"))
include(joinpath(mantis_dir, "examples", "HelperFunctions.jl"))

############################################################################################
#                                      Problem setup                                       #
############################################################################################

function run_problem(
    manifold_dim::Int, section_space::Type{T}, p::Int, k::Int
) where {T <: FunctionSpaces.AbstractCanonicalSpace}
    breakpoints = collect(LinRange(0.0, 1.0, 5))
    patch = Geometry.CartesianGeometry(breakpoints)
    regularity = fill(k, 5)
    regularity[1] = regularity[end] = -1
    Bpk = FunctionSpaces.BSplineSpace(patch, section_space(p), regularity)
    TPspaces = ntuple(manifold_dim) do i
        return Bpk
    end
    TP = FunctionSpaces.TensorProductSpace(TPspaces)
    eval_points = Points.CartesianPoints(
        ntuple(dim -> LinRange(0.0, 1.0, 25), manifold_dim)
    )
    dim = FunctionSpaces.get_num_basis(TP)

    return dim,
    @benchmarkable FunctionSpaces.evaluate($TP, 1, $eval_points) samples = 500 evals = 1 seconds =
        Inf
end

############################################################################################
#                                       Run problems                                        #
############################################################################################

group = BenchmarkGroup()
section_spaces = (
    FunctionSpaces.Bernstein,
    FunctionSpaces.LobattoLegendre,
    FunctionSpaces.EdgeLobattoLegendre,
)

manifold_dims = [2, 3]
ps = [3, 5]
k = -1
for manifold_dim in manifold_dims
    foreach(section_spaces) do section_space
        sub_group = BenchmarkGroup()
        for p in ps
            dim, benchmarkable = run_problem(manifold_dim, section_space, p, k)
            name = "num_basis=$(dim)-p=$(p)-k=$(k)"
            sub_group[name] = benchmarkable
        end

        group["$(manifold_dim)D-$section_space"] = sub_group
    end
end

end

module EvaluateUnivariateSplinesBenchmarks

using Mantis

# Refer to the following file for method and variable definitions
include(joinpath(pwd(), "BenchmarkHelpers.jl"))
include(joinpath(mantis_dir, "examples", "HelperFunctions.jl"))

############################################################################################
#                                      Problem setup                                       #
############################################################################################

function run_problem(
    section_space::Type{T}, num_els::Int, p::Int, k::Int
) where {T <: FunctionSpaces.AbstractCanonicalSpace}
    breakpoints = collect(LinRange(0.0, 1.0, num_els + 1))
    patch = Geometry.CartesianGeometry(breakpoints)
    regularity = fill(k, num_els + 1)
    regularity[1] = regularity[end] = -1
    Bpk = FunctionSpaces.BSplineSpace(patch, section_space(p), regularity)
    eval_points = Points.CartesianPoints((LinRange(0.0, 1.0, 25),))

    dim = FunctionSpaces.get_num_basis(Bpk)

    return dim,
    @benchmarkable FunctionSpaces.evaluate($Bpk, 1, $eval_points) samples = 500 evals = 1 seconds =
        Inf
end

############################################################################################
#                                       Run problems                                       #
############################################################################################

group = BenchmarkGroup()

section_spaces = (
    FunctionSpaces.Bernstein,
    FunctionSpaces.LobattoLegendre,
    FunctionSpaces.EdgeLobattoLegendre,
)
ps = [1, 2, 3, 5, 8]
k = -1
foreach(section_spaces) do section_space
    sub_group = BenchmarkGroup()
    for p in ps
        dim, benchmarkable = run_problem(section_space, 4, p, k)
        name = "num_basis=$(dim)-p=$(p)"
        sub_group[name] = benchmarkable
    end

    group["$section_space-k=$(k)"] = sub_group
end
section_spaces2 = (FunctionSpaces.Bernstein, FunctionSpaces.LobattoLegendre)
k2 = 0
foreach(section_spaces2) do section_space
    sub_group = BenchmarkGroup()
    for p2 in ps
        dim, benchmarkable = run_problem(section_space, 4, p2, k2)
        name = "num_basis=$(dim)-p=$(p2)"
        sub_group[name] = benchmarkable
    end

    group["$section_space-k=$(k2)"] = sub_group
end

section_spaces3 = (FunctionSpaces.Bernstein,)
foreach(section_spaces3) do section_space
    sub_group = BenchmarkGroup()
    for p3 in ps
        dim, benchmarkable = run_problem(section_space, 4, p3, p3 - 1)
        name = "num_basis=$(dim)-p=$(p3)"
        sub_group[name] = benchmarkable
    end

    group["$section_space-k=p-1"] = sub_group
end

end

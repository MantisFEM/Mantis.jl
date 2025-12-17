module EvaluateClenshawCurtisQuadratureBenchmarks

using Mantis

# Refer to the following file for method and variable definitions
include(joinpath(pwd(), "BenchmarkHelpers.jl"))
include(joinpath(mantis_dir, "examples", "HelperFunctions.jl"))

############################################################################################
#                                      Problem setup                                       #
############################################################################################
function run_problem(p)
    return @benchmarkable Quadrature.clenshaw_curtis($p) samples = 5000 evals = 1 seconds =
        Inf
end

############################################################################################
#                                       Run problems                                       #
############################################################################################

group = BenchmarkGroup()

degrees = (3, 4, 12, 15)
for p in degrees
    benchmarkable = run_problem(p+1)
    group["clenshaw-curtis-degree-$(p)"] = benchmarkable
end

end

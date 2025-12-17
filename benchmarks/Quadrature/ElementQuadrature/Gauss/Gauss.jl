module EvaluateGaussQuadratureBenchmarks

using Mantis

# Refer to the following file for method and variable definitions
include(joinpath(pwd(), "BenchmarkHelpers.jl"))
include(joinpath(mantis_dir, "examples", "HelperFunctions.jl"))

############################################################################################
#                                      Problem setup                                       #
############################################################################################
function run_problem_lobatto(N)
    return @benchmarkable Quadrature.gauss_lobatto($N) samples = 5000 evals = 1 seconds =
        Inf
end

function run_problem_legendre(N)
    return @benchmarkable Quadrature.gauss_legendre($N) samples = 5000 evals = 1 seconds =
        Inf
end

############################################################################################
#                                       Run problems                                       #
############################################################################################

group = BenchmarkGroup()

degrees = (3, 4, 12, 15)
for p in degrees
    benchmarkable = run_problem_lobatto(p+1)
    group["lobatto-degree-$(p)"] = benchmarkable

    benchmarkable_legendre = run_problem_legendre(p+1)
    group["legendre-degree-$(p)"] = benchmarkable_legendre
end

end

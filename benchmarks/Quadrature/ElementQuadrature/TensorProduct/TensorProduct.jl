module EvaluateTensorProductQuadratureBenchmarks

using Mantis

# Refer to the following file for method and variable definitions
include(joinpath(pwd(), "BenchmarkHelpers.jl"))
include(joinpath(mantis_dir, "examples", "HelperFunctions.jl"))

############################################################################################
#                                      Problem setup                                       #
############################################################################################
function run_problem(N, f)
    return @benchmarkable Quadrature.tensor_product_rule($N, $f) samples = 5000 evals = 1 seconds =
        Inf
end

############################################################################################
#                                       Run problems                                       #
############################################################################################

group = BenchmarkGroup()

rules = (Quadrature.gauss_lobatto, Quadrature.gauss_legendre, Quadrature.clenshaw_curtis)
degrees = (3, 4)
foreach(rules) do qrule
    for p in degrees
        benchmarkable = run_problem((p+1,), qrule)
        group["tp-$(qrule)-1D-degree-$(p)"] = benchmarkable
    end
    for p in degrees
        benchmarkable = run_problem((p+1, p+1), qrule)
        group["tp-$(qrule)-2D-degree-$(p)"] = benchmarkable
    end
    for p in degrees
        benchmarkable = run_problem((p+1, p+1, p+1), qrule)
        group["tp-$(qrule)-3D-degree-$(p)"] = benchmarkable
    end
end

end

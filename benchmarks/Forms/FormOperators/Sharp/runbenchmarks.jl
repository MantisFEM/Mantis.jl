module SharpBenchmarks

include(joinpath(pwd(), "BenchmarkHelpers.jl"))

verbose = true
show = true
save = true

group = BenchmarkGroup()

# Module-level benchmark groups
if verbose
    println("Generating benchmarkable objects...")
end

mod = include("Sharp.jl")
sub_group = getproperty(mod, :group)
group["Sharp"] = sub_group

if verbose
    println("Done!")
end

if show || save
    for (file_name, benchmark) in group["Sharp"]
        file_location = relpath(dirname(@__FILE__), pwd())
        run_benchmark(benchmark, file_location, file_name; show=show, save=save, rtol=rtol)
    end
end

end

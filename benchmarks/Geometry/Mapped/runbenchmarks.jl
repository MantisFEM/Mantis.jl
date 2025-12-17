module MappedGeometryBenchmarks

include(joinpath(pwd(), "BenchmarkHelpers.jl"))

verbose = true
show = true
save = true

group = BenchmarkGroup()

# Module-level benchmark groups
if verbose
    println("Generating benchmarkable objects...")
end

mod = include("Mapped.jl")
sub_group = getproperty(mod, :group)
group["Mapped"] = sub_group

if verbose
    println("Done!")
end

if show || save
    for (file_name, benchmark) in group["Mapped"]
        file_location = relpath(dirname(@__FILE__), pwd())
        run_benchmark(benchmark, file_location, file_name; show=show, save=save, rtol=rtol)
    end
end

end

module UnivariateSplinesBenchmarks

include(joinpath(pwd(), "BenchmarkHelpers.jl"))

verbose = true
show = true
save = true

group = BenchmarkGroup()

# Module-level benchmark groups
if verbose
    println("Generating benchmarkable objects...")
end

mod = include("UnivariateSplines.jl")
sub_group = getproperty(mod, :group)
group["UnivariateSplines"] = sub_group

if verbose
    println("Done!")
end

if show || save
    for (section_space, section_space_group) in group["UnivariateSplines"]
        for (name, benchmark) in section_space_group
            file_name = "$(section_space)-$(name)"
            file_location = relpath(dirname(@__FILE__), pwd())
            run_benchmark(
                benchmark, file_location, file_name; show=show, save=save, rtol=rtol
            )
        end
    end
end

end

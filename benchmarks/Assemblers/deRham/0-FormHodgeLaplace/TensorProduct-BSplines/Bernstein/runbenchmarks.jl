module TensorProductBSplinesBenchmarks

include(joinpath(pwd(), "BenchmarkHelpers.jl"))

verbose = true
show = true
save = true

group = BenchmarkGroup()

# Module-level benchmark groups
if verbose
    println("Generating benchmarkable objects...")
end

mod = include("Bernstein.jl")
sub_group = getproperty(mod, :group)
group["Bernstein"] = sub_group

if verbose
    println("Done!")
end

if show || save
    for (manifold_dim, manifold_group) in group["Bernstein"]
        for (name, benchmark) in manifold_group
            file_name = "$(manifold_dim)-$(name)"
            file_location = relpath(dirname(@__FILE__), pwd())
            run_benchmark(
                benchmark, file_location, file_name; show=show, save=save, rtol=rtol
            )
        end
    end
end

end

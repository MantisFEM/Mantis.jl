module CachingTests

import Pkg
using Test

Pkg.activate(@__DIR__)
Pkg.instantiate()

@testset verbose = true "API" begin
    include("API/runtests.jl")
end

@testset verbose = true "Geometry" begin
    include("Geometry/runtests.jl")
end

end

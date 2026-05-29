module BernsteinPolynomialsTests

using Mantis
using Mantis: FunctionSpaces as FS
using Test
using JET
using BenchmarkTools

const VERBOSE = false

const p = 3
const bern = FS.Bernstein(p)
const points = Points.PointSet(LinRange(0, 1, 5))

############################################################################################
#                                        Buffer API                                        #
############################################################################################

@testset verbose = true "Buffer" begin
    buff = FS.preallocate(bern, points)
    @test isa(buff, FS.CanonicalBuffer{eltype(points)})
    cache = Caching.Cache(bern, points)
    obj, buff = Caching.extract(cache)
    @test obj === bern
    @test isa(buff, FS.CanonicalBuffer{eltype(points)})
    Caching.setfilled!(buff, true)
    Caching.clear!(buff)
    @test buff() == zeros(length(points), p + 1, 1)
    @test Caching.isfilled(buff) == false
end

############################################################################################
#                                       Evaluations                                        #
############################################################################################

@testset verbose = true "Truth Comparison" begin
    cache = Caching.Cache(bern, points)
    buff = Caching.get_buffer(cache)
    max_der_order = 3
    og_eval = FS.evaluate(bern, points, max_der_order)
    for der_order in 0:max_der_order
        cache(1, points, der_order)
        @test cache()[:, :, 1] == og_eval[der_order+1][1]
    end
end

@testset verbose = true "Performance Comparison" begin
    cache = Caching.Cache(bern, points)
    og_bench = @benchmark FS.evaluate($bern, $points, 0)
    ca_bench = @benchmark Caching.update!($cache, 1, $points, 0)
    @test iszero(ca_bench.allocs)
    @test mean(og_bench).time > mean(ca_bench).time
    if VERBOSE
        println("Original benchmark")
        display(og_bench)
        println("")
        println("Cached benchmark")
        display(ca_bench)
    end
end

@testset verbose = true "JET Type Stability" begin
    @test_call Caching.Cache(bern, points)
    cache = Caching.Cache(bern, points)
    @test_call cache(1, points, 2)
    @test_call cache(1, points, 2)
end

end

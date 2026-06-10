module BSplinesTests

using Mantis
import Mantis: FunctionSpaces as FS
using Test
using JET
using BenchmarkTools

const VERBOSE = false

const points = Points.PointSet((LinRange(0, 1, 7),))
const starting_point = (0.3,)
const box_size = (0.75,)
const num_elements = (5,)
const p = (3,)
const k = (2,)
const bsp = FS.create_bspline_space(starting_point, box_size, num_elements, p, k)

############################################################################################
#                                        Buffer API                                        #
############################################################################################

@testset verbose = true "Buffer" begin
    buff = FS.preallocate(bsp, points)
    @test isa(buff, FS.FEBuffer{eltype(points)})
    cache = Caching.Cache(bsp, points)
    obj, buff = Caching.extract(cache)
    @test obj === bsp
    @test isa(buff, FS.FEBuffer{eltype(points)})
    Caching.setfilled!(buff, true)
    Caching.clear!(buff)
    @test buff() == zeros(length(points), p[1] + 1, 1)
    @test Caching.isfilled(buff) == false
end

############################################################################################
#                                       Evaluations                                        #
############################################################################################

@testset verbose = true "Truth Comparison" begin
    cache = Caching.Cache(bsp, points)
    can_cache = FS.get_canonical_cache(Caching.get_buffer(cache))
    der_keys = (0, 1, 2)
    truth_comparison = true
    for der_key in der_keys
        VERBOSE && println("Running truth comparison for p=$(p) and der_key=$(der_key).")
        for element in 1:FS.get_num_elements(bsp)
            og_eval, og_indices = FS.evaluate(bsp, element, points, der_key)
            fill!(cache, element, points, der_key)
            if !(og_eval[der_key+1][1][1] == cache()[:, :, 1])
                truth_comparison = false
            end
        end
        #=
        We need to explicitly declare the canonical cache not to be filled because we change
        the derivative key. Note that this is not needed while looping over elements, since
        the derivative key is the same.
        =#
        Caching.setfilled!(can_cache, false)
    end

    @test truth_comparison
end

@testset verbose = true "Performance Comparison" begin
    der_key = 0
    cache = Caching.Cache(bsp, points)
    og_bench = @benchmark FS.evaluate($bsp, 2, $points, $der_key)
    ca_bench = @benchmark fill!($cache, 2, $points, $der_key)
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
    cache = Caching.Cache(bsp, points)
    der_key = 2
    @test_call Caching.Cache(bsp, points)
    @test_call fill!(cache, 2, points, der_key)
end

end

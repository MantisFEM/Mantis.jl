module CartesianGeometryTests

using Mantis
using Test
using JET
using BenchmarkTools

VERBOSE = false

############################################################################################
#                                            1D                                            #
############################################################################################

@testset verbose = true "1D" begin
    VERBOSE && println("Running 1D tests...")
    geo = Geometry.create_cartesian_box((1.0,), (2.0,), (4,))
    points = Points.PointSet((LinRange(0, 1, 5),))
    cache = Caching.Cache(geo, points)
    obj, buff = Caching.extract(cache)
    @test obj === geo
    @test isa(buff, Geometry.GeometryBuffer)

    @testset verbose = true "Truth-comparison" begin
        for element in 1:Geometry.get_num_elements(geo)
            og_eval = Geometry.evaluate(geo, element, points)
            cache(element, points)
            @test og_eval == cache()
        end

        Caching.update!(cache, 3, points)
        og_eval =  Geometry.evaluate(geo, 3, points)
        @test cache() == og_eval
        Caching.update!(cache, 4, points)
        @test cache() == og_eval
    end

    og_bench = @benchmark Geometry.evaluate($geo, 2, $points)
    ca_bench = @benchmark $cache(2, $points)
    @testset verbose = true "Performance-comparison" begin
        @test iszero(ca_bench.allocs)
        @test mean(og_bench).time > mean(ca_bench).time
    end

    @testset verbose = true "JET Type Stability" begin
        @test_call Caching.Cache(geo, points)
        @test_call cache(2, points)
    end

    if VERBOSE
        println("Original benchmark")
        display(og_bench)
        println("")
        println("Cached benchmark")
        display(ca_bench)
    end
end

@testset verbose = true "2D" begin
    VERBOSE && println("Running 2D tests...")
    geo = Geometry.create_cartesian_box((1.0, 0.5), (2.0, 1.0), (4, 3))
    point_set = Points.PointSet((LinRange(0, 1, 7), LinRange(0.5, 0.6, 7)))
    point_cart = Points.CartesianPoints((LinRange(0, 1, 3), LinRange(0.5, 0.7, 2)))
    cache_set = Caching.Cache(geo, point_set)
    cache_cart = Caching.Cache(geo, point_cart)
    obj_cart, buff_cart = Caching.extract(cache_cart)
    @test obj_cart === geo
    @test isa(buff_cart, Geometry.GeometryBuffer)
    obj_set, buff_set = Caching.extract(cache_set)
    @test obj_set === geo
    @test isa(buff_set, Geometry.GeometryBuffer)

    @testset verbose = true "Truth-comparison" begin
        for element in 1:Geometry.get_num_elements(geo)
            og_eval_set = Geometry.evaluate(geo, element, point_set)
            og_eval_cart = Geometry.evaluate(geo, element, point_cart)
            cache_set(element, point_set)
            cache_cart(element, point_cart)
            @test og_eval_set == cache_set()
            @test og_eval_cart == cache_cart()
        end
    end

    og_bench_set = @benchmark Geometry.evaluate($geo, 2, $point_set)
    ca_bench_set = @benchmark $cache_set(2, $point_set)
    og_bench_cart = @benchmark Geometry.evaluate($geo, 2, $point_cart)
    ca_bench_cart = @benchmark $cache_cart(2, $point_cart)
    @testset verbose = true "Performance-comparison" begin
        @test iszero(ca_bench_set.allocs)
        @test iszero(ca_bench_cart.allocs)
        @test mean(og_bench_set).time > mean(ca_bench_set).time
        @test mean(og_bench_cart).time > mean(ca_bench_cart).time
    end

    @testset verbose = true "JET Type Stability" begin
        @test_call Caching.Cache(geo, point_set)
        @test_call Caching.Cache(geo, point_cart)
        @test_call cache_set(2, point_set)
        @test_call cache_cart(2, point_cart)
    end

    if VERBOSE
        println("Cartesian Points")
        println("Original benchmark")
        display(og_bench_cart)
        println("")
        println("Cached benchmark")
        display(ca_bench_cart)

        println("Point Set")
        println("Original benchmark")
        display(og_bench_set)
        println("")
        println("Cached benchmark")
        display(ca_bench_set)
    end
end

end

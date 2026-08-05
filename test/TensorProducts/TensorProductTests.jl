module TensorProductTests

using Test
using Mantis.TensorProducts

struct NewObject
    n::Int
end

s = NewObject(0)

@test_throws MethodError TensorProducts.get_num_objects(s)

TensorProducts.get_num_objects(s::NewObject) = s.n

@testset "TensorProduct constructor" begin
    tp = TensorProduct((NewObject(2), NewObject(3), NewObject(4)))

    @test TensorProducts.get_num_factors(tp) == 3
    @test TensorProducts.get_factor_num_objects(tp) == (2, 3, 4)
    @test TensorProducts.get_num_objects(tp) == 24

    @test TensorProducts.get_factors(tp) == (NewObject(2), NewObject(3), NewObject(4))

    tp_symb = NewObject(2) ⊗ NewObject(3) ⊗ NewObject(4)
    @test tp_symb == tp
end

@testset "TensorProduct indexing" begin
    tp = TensorProduct((NewObject(2), NewObject(3)))

    @test TensorProducts.get_factor_ids(tp, 1) == (1, 1)
    @test TensorProducts.get_factor_ids(tp, 2) == (2, 1)
    @test TensorProducts.get_factor_ids(tp, 3) == (1, 2)
    @test TensorProducts.get_factor_ids(tp, 4) == (2, 2)
    @test TensorProducts.get_factor_ids(tp, 5) == (1, 3)
    @test TensorProducts.get_factor_ids(tp, 6) == (2, 3)

    @test TensorProducts.get_lin_ids(tp)[1, 1] == 1
    @test TensorProducts.get_lin_ids(tp)[2, 1] == 2
    @test TensorProducts.get_lin_ids(tp)[1, 2] == 3
    @test TensorProducts.get_lin_ids(tp)[2, 2] == 4
    @test TensorProducts.get_lin_ids(tp)[1, 3] == 5
    @test TensorProducts.get_lin_ids(tp)[2, 3] == 6
end

@testset "TensorProduct mapping" begin
    tp = TensorProduct((NewObject(2), NewObject(3)))

    f(s::NewObject) = NewObject(2s.n)

    tp2 = map(f, tp)

    @test TensorProducts.get_factor_num_objects(tp2) == (4, 6)
    @test TensorProducts.get_num_objects(tp2) == 24

    factors = TensorProducts.mapfactors(f, tp)
    @test factors == (NewObject(4), NewObject(6))

    fi(s::NewObject, i::Int) = NewObject(i * s.n)
    factors = TensorProducts.mapfactors(fi, tp, 2) # i1= 2, i2 = 1
    @test factors == (NewObject(4), NewObject(3))
end

@testset "TensorProduct singleton" begin
    tp = TensorProduct((NewObject(5),))

    @test TensorProducts.get_num_factors(tp) == 1
    @test TensorProducts.get_num_objects(tp) == 5

    @test TensorProducts.get_factor_ids(tp, 1) == (1,)
    @test TensorProducts.get_factor_ids(tp, 5) == (5,)
end

@testset "TensorProduct empty dimension" begin
    tp = TensorProduct((NewObject(2), NewObject(0)))

    @test TensorProducts.get_num_objects(tp) == 0
    @test TensorProducts.get_factor_num_objects(tp) == (2, 0)
end

@testset "Nested TensorProducts" begin
    tp = NewObject(2) ⊗ NewObject(2)
    # Left-composition
    tpl = tp ⊗ NewObject(2)
    @test TensorProducts.get_num_objects(tpl) == 8
    @test TensorProducts.get_factor_num_objects(tpl) == (2, 2, 2)
    # Right-composition
    tpr = NewObject(2) ⊗ tp
    @test TensorProducts.get_num_objects(tpr) == 8
    @test TensorProducts.get_factor_num_objects(tpr) == (2, 2, 2)
    # Both
    tpb = tp ⊗ tp
    @test TensorProducts.get_num_objects(tpb) == 16
    @test TensorProducts.get_factor_num_objects(tpb) == (2, 2, 2, 2)
    ## Does not store unnecessary TensorProducts
    @test TensorProducts.get_factors(tpb) ==
        (NewObject(2), NewObject(2), NewObject(2), NewObject(2))
    # Singletons
    tp4 = NewObject(2) ⊗ NewObject(3) ⊗ NewObject(1) ⊗ NewObject(2)
    @test TensorProducts.get_num_objects(tp4) == 12
    @test TensorProducts.get_factor_num_objects(tp4) == (2, 3, 1, 2)
    ## Does not create unnecessary TensorProducts
    @test TensorProducts.get_factors(tp4) ==
        (NewObject(2), NewObject(3), NewObject(1), NewObject(2))
end

end

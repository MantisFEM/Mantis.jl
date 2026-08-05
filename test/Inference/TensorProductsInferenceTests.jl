module TensorProductsInferenceTests

using Mantis.TensorProducts
using Test
using JET

struct NewObject
    n::Int
end

TensorProducts.get_num_objects(s::NewObject) = s.n


@test_opt TensorProducts.TensorProduct((NewObject(2), NewObject(3))) # tuple
@test_opt TensorProducts.TensorProduct(NewObject(2), NewObject(3)) # splat

tp = NewObject(2) ⊗ NewObject(3)

@test_opt TensorProducts.get_factors(tp)
@test_opt TensorProducts.get_cart_ids(tp)
@test_opt TensorProducts.get_lin_ids(tp)
@test_opt TensorProducts.get_num_factors(tp)
@test_opt TensorProducts.get_factor_num_objects(tp)
@test_opt TensorProducts.get_num_objects(tp)
@test_opt TensorProducts.get_factor_ids(tp, 1)
@test_opt TensorProducts.mapfactors(identity, tp)
@test_opt map(identity, tp)

end

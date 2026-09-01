"""
    module TensorProducts

Utilities for constructing and working with tensor-products of sets.

A `TensorProduct` represents the tensor-product of multiple, lower-dimensional factor sets,
providing efficient conversion between linear indices and Cartesian indices of the product
space. This module provides the shared infrastructure for this pattern: a generic interface
for creating and indexing tensor-products of arbitrary sets.
"""
module TensorProducts

export TensorProduct
export ⊗

"""
    TensorProduct{I,CI,LI}

Represents the tensor (Cartesian) product of a tuple of factor sets.

A `TensorProduct` stores the factor sets together with the corresponding `CartesianIndices`
and `LinearIndices`, allowing efficient conversion between linear and multi-dimensional
indexing.

# Fields
- `factors::I`: Tuple of factor sets.
- `cart_ids::CI`: Cartesian indices of the tensor product.
- `lin_ids::LI`: Linear index mapping for the Cartesian indices.
"""
struct TensorProduct{I, CI, LI}
    factors::I
    cart_ids::CI
    lin_ids::LI

    function TensorProduct(factors::I) where {I <: Tuple}
        factor_num_objects = map(get_num_objects, factors)
        cart_ids = CartesianIndices(factor_num_objects)
        lin_ids = LinearIndices(cart_ids)

        return new{I, typeof(cart_ids), typeof(lin_ids)}(factors, cart_ids, lin_ids)
    end
end

TensorProduct(factors...) = TensorProduct(factors)
TensorProduct(tp::TensorProduct, b) = TensorProduct(get_factors(tp)..., b)
TensorProduct(a, tp::TensorProduct) = TensorProduct(a, get_factors(tp)...)

function TensorProduct(tp1::TensorProduct, tp2::TensorProduct)
    return TensorProduct(get_factors(tp1)..., get_factors(tp2)...)
end

"""
Alias for [`TensorProduct`](@ref).
"""
const ⊗ = TensorProduct

"""
    get_num_objects(set::Any)

Return the number of objects contained in `set`.

This function must be implemented for any type that is used as an factor to a
`TensorProduct`.

Defaults are defined in later modules for `Geometry.AbstractGeometry` and
`FunctionSpaces.AbstractFESpace`, namely `Geometry.get_num_elements` and
`FunctionSpaces.get_num_basis`.

# Examples
```jldoctest
julia> using Mantis

julia> TensorProducts.get_num_objects(["a", "b", "c"]) == 3
true

julia> TensorProducts.get_num_objects(Geometry.create_cartesian_box((0.0, 0.0), (1.0, 1.0), (2, 2))) == 4
true

julia> TensorProducts.get_num_objects(word::String) = length(word); nothing

julia> TensorProducts.get_num_objects("Pequod") == 6
true
```
"""
get_num_objects(set::Any) = throw(MethodError(get_num_objects, (set,)))

get_num_objects(set::Union{Tuple, AbstractArray}) = length(set)

get_num_objects(tp::TensorProduct) = length(get_cart_ids(tp))

"""
    get_factors(tp::TensorProduct)

Return the tuple of factor sets defining the tensor product.
"""
get_factors(tp::TensorProduct) = tp.factors

"""
    get_cart_ids(tp::TensorProduct)

Return the `CartesianIndices` associated with the tensor product.
"""
get_cart_ids(tp::TensorProduct) = tp.cart_ids

"""
    get_lin_ids(tp::TensorProduct)

Return the `LinearIndices` associated with the tensor product.
"""
get_lin_ids(tp::TensorProduct) = tp.lin_ids

"""
    get_num_factors(tp::TensorProduct)

Return the number of factor sets in the tensor product.
"""
get_num_factors(tp::TensorProduct) = length(get_factors(tp))

"""
    get_factor_num_objects(tp::TensorProduct)

Return a tuple containing the number of objects in each factor set.
"""
get_factor_num_objects(tp::TensorProduct) = Tuple(last(get_cart_ids(tp)))

"""
    get_factor_ids(tp::TensorProduct, id)

Return the tuple of indices corresponding to the linear index `id`. The tuple can be used to
index each factor of `tp`.
"""
function get_factor_ids(tp::TensorProduct, id)
    return Tuple(get_cart_ids(tp)[id])
end

"""
    mapfactors(f, tp::TensorProduct)

Apply `f` to each factor set and return the resulting tuple.
"""
mapfactors(f, tp::TensorProduct) = map(f, get_factors(tp))

"""
    mapfactors(f, tp::TensorProduct, id)

Apply `f` to each factor set and factor id, converted from the global `id`, and return the
resulting tuple.
"""
mapfactors(f, tp::TensorProduct, id) = map(f, get_factors(tp), get_factor_ids(tp, id))

"""
    map(f, tp::TensorProduct, args...)

Apply `f` to each factor set and construct a new `TensorProduct` from the resulting sets.
"""
Base.map(f, tp::TensorProduct, args...) = TensorProduct(mapfactors(f, tp, args...))

end

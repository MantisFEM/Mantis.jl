
############################################################################################
#                                    AbstractRefinement                                    #
############################################################################################

abstract type AbstractRefinement{O, M} end

abstract type RefinementMethod end

struct RefinementExplicit{F <: Function} <: RefinementMethod
    method::F
end

function (refinement::RefinementExplicit)(parent)
    return refinement.method(parent)
end

struct Refinement{O, M <: RefinementMethod} <: AbstractRefinement{O, M}
    parent::O
    method::M
end

function Refinement(parent, method::Function)
    return Refinement(parent, RefinementExplicit(method))
end

get_parent(refinement::AbstractRefinement) = refinement.parent
get_method(refinement::AbstractRefinement) = refinement.method

function (refinement::AbstractRefinement)()
    method = get_method(refinement)
    parent = get_parent(refinement)

    return method(parent)
end

function Refinement(
    parent::TensorProducts.TensorProduct, methods::NTuple{num_methods, Function}
) where {num_methods}
    parent_factors = TensorProducts.get_factors(parent)
    if length(parent_factors) != num_methods
        throw(
            ArgumentError(
                LazyString(
                    "Number of factors does not match number of methods. ",
                    "Got ",
                    length(parent_factors),
                    " parent factors, and ",
                    num_methods,
                    " methods.",
                ),
            ),
        )
    end

    child_factors = ntuple(i -> methods[i](parent_factors[i]), num_methods)
    refinement(_) = TensorProducts.TensorProduct(child_factors)

    return Refinement(parent, RefinementExplicit(refinement))
end

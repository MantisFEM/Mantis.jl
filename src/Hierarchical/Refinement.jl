
############################################################################################
#                                    AbstractRefinement                                    #
############################################################################################

# NOTE: The choice to implement lazy instantitation of child sets is to leverage the fact
# that in adaptive methods when or _if_ a refinement will happen is uncertain.
# By making the children creation lazy we can create a refinement possibly without ever
# instantiating the child set.

"""
    AbstractRefinement{O, M}

Supertype for all refinements. Concrete subtypes _must_ hold a parent set and a refinement
method that produces a child set from the parent set.

See also [`RefinementMethod`](@ref).

!!!note
    Concrete subtypes can be called with no arguments to produce the child set in a lazy
    manner.

# Type Parameters
- `O`: Type of the parent set. 
- `M`: Type of the refinement method.
"""
abstract type AbstractRefinement{O, M} end

"""
    (refinement::AbstractRefinement)()

Return the child set produced from applying the refinement method to the parent set.
"""
function (refinement::AbstractRefinement)()
    method = get_method(refinement)
    parent = get_parent(refinement)

    return method(parent)
end

############################################################################################
#                                     RefinementMethod                                     #
############################################################################################

"""
    RefinementMethod

Supertype for all refinement methods. Concrete subtypes store any methods or state necessary
to produce child sets from parent sets.
"""
abstract type RefinementMethod end

"""
    RefinementExplicit{F <: Function} <: RefinementMethod

A refinement method in closed form, depending only on its underlying expression.

!!! note
    This struct can be called on a parent set.

# Fields
- `method::F`: The method defining the refinement.
"""
struct RefinementExplicit{F <: Function} <: RefinementMethod
    method::F
end

"""
    (refinement::RefinementExplicit)(parent)

Return the child set produced by refining `parent`.
"""
function (refinement::RefinementExplicit)(parent)
    return refinement.method(parent)
end

############################################################################################
#                                        Refinement                                        #
############################################################################################

"""
    Refinement{O, M <: RefinementMethod} <: AbstractRefinement{O, M}

General concrete refinement holding a parent set and a refinement method.

# Fields
- `parent::O`: The parent set.
- `method::M`: The refinement method to apply to `parent`.

# Examples
```jldoctest
julia> using Mantis;

julia> using Mantis.Hierarchical;

julia> refinement = Refinement("John", RefinementExplicit(father -> father * "'s son"));

julia> refinement() == "John's son"
true
```

# Outer Constructors
- [`Refinement(parent, method::Function)`](@ref)
"""
struct Refinement{O, M <: RefinementMethod} <: AbstractRefinement{O, M}
    parent::O
    method::M
end

"""
    Refinement(parent, method::Function)

Constructor that wraps `method` in a `RefinementExplicit`.
"""
function Refinement(parent, method::Function)
    return Refinement(parent, RefinementExplicit(method))
end

"""
    Refinement(
        parent::TensorProduct, methods::NTuple{num_methods, Function}
    ) where {num_methods}

Constructor for `parent` sets with a tensor-product structure.

# Arguments
- `parent::TensorProduct`: The parent set.
- `methods::NTuple{num_methods, Function}`: The refinement methods to apply to each factor
    parent set of `parent`.
"""
function Refinement(
    parent::TensorProduct, methods::NTuple{num_methods, Function}
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
    refinement(_) = TensorProduct(child_factors)

    return Refinement(parent, RefinementExplicit(refinement))
end

"""
    get_parent(refinement::AbstractRefinement)

Return the parent of `refinement`.
"""
get_parent(refinement::AbstractRefinement) = refinement.parent

"""
    get_method(refinement::AbstractRefinement)

Return the refinement method of `refinement`.
"""
get_method(refinement::AbstractRefinement) = refinement.method

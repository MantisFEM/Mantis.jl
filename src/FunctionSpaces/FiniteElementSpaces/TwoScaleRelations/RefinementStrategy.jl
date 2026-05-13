abstract type AbstractGlobalRefinementStrategy{manifold_int} end

"""
    struct GlobalRefinement{manifold_dim, ref_order} <: AbstractGlobalRefinementStrategy{manifold_dim}

Refinement strategy for refinement of a tensor product finite element space.

# Type Parameters
- `manifold_dim::Int`: Dimension of the space.
- `ref_order::Symbol`: Order of the refinement strategy (e.g., `:h_first`, `:p_first`)

# Arguments
- `num_subdivisions::NTuple{manifold_dim, Int}`: Number of subdivisions per element for each finite element space.
- `degree_delta::NTuple{manifold_dim, Int}`: Change in degree of the basis functions for each finite element space.
- `regularity::NTuple{num_spaces, Int}`: Smoothness of the refined basis functions for each finite element space.

# Returns
- `GlobalRefinement{manifold_dim, ref_order}`: 

"""
struct GlobalRefinement{manifold_dim, ref_order} <: AbstractGlobalRefinementStrategy{manifold_dim}
    num_subdivisions::NTuple{manifold_dim, Int}
    degree_delta::NTuple{manifold_dim, Int}
    regularity::NTuple{manifold_dim, Int}

    function GlobalRefinement(
        num_subdivisions::NTuple{manifold_dim, Int},
        degree_delta::NTuple{manifold_dim, Int},
        regularity::NTuple{manifold_dim, Int},
        ref_order::Symbol
    ) where {manifold_dim}
        return new{manifold_dim, ref_order}(num_subdivisions, degree_delta, regularity)
    end
end

get_num_subdivisions(strategy::GlobalRefinement) = strategy.num_subdivisions
get_degree_delta(strategy::GlobalRefinement) = strategy.degree_delta
get_regularity(strategy::GlobalRefinement) = strategy.regularity

get_num_subdivisions(strategy::GlobalRefinement, dirs::AbstractVector{Int}) = strategy.num_subdivisions[dirs]
get_degree_delta(strategy::GlobalRefinement, dirs::AbstractVector{Int}) = strategy.degree_delta[dirs]
get_regularity(strategy::GlobalRefinement, dirs::AbstractVector{Int}) = strategy.regularity[dirs]

function get_refinement_strategy(
    strategy::GlobalRefinement{manifold_dim, ref_order}, dirs::AbstractVector{Int}
) where {manifold_dim, ref_order <: Symbol}
    return GlobalRefinement(
        get_num_subdivisions(strategy, dirs),
        get_degree_delta(strategy, dirs),
        get_regularity(strategy, dirs),
        ref_order
    )
end
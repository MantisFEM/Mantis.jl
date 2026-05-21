module GeneralHelpers

export integer_sums, get_derivative_idx, num_der_indices, export_path

import Combinatorics

####################################################
# Integer sums and derivatives helper functions
####################################################

#=
    The caching we chose here is a compromise between a few things:
    1. Using Memoize.jl: This does not really handle type stability very well. Most
        dictionaries are Dict{Any, Any}.
    2. Previously we were storing the cached helpers in this file as Vector{Int}, because
        then everything would be type-stable. The problem of course is that we didn't
        really use all the information we had (the manifold_dim sized tuples).
    3. Using @generated to create caches at compile time, which are then passed by
        reference, seems to be the best approach. However, as mentioned in Julia's
        documentation, it is not guaranteed that the cache is not erased after compilation,
        so that is the downside.
=#

"""
    cache_dict(::Type{K}, ::Type{V}, id=Val{1}()) where {K, V}

Return a `Dict{K,V}` dictionary stored at compile time.
An optional `id` can be given to force the compilation of a new cache. This is useful to
avoid clashes of dictionaries using the same types.

!!! warning
    As per Julia's documentation, it is not guaranteed that the dictionary is not wiped
    during runtime. Read
    [this](https://docs.julialang.org/en/v1/manual/metaprogramming/#Generated-functions) for
    more information.

# Examples
```jldoctest
julia> using Mantis

julia> dict = Mantis.GeneralHelpers.cache_dict(Int, String)
Dict{Int64, String}()

julia> dict[42] = "Answer"
"Answer"

julia> Mantis.GeneralHelpers.cache_dict(Int, String)
Dict{Int64, String} with 1 entry:
  42 => "Answer"

julia> not_new_dict = Mantis.GeneralHelpers.cache_dict(Int, String)
Dict{Int64, String} with 1 entry:
  42 => "Answer"

julia> new_dict = Mantis.GeneralHelpers.cache_dict(Int, String, Val(2))
Dict{Int64, String}()

```
"""
@generated function cache_dict(::Type{K}, ::Type{V}, id=Val{1}()) where {K, V}
    cache = Dict{K, V}()

    return :($cache)
end

"""
    get_from_cache(::Type{K}, ::Type{V}, key, f, id=Val{1}()) where {K, V}

Call `cache_dict` on the types `K` and `V` with `id` to retrieve a cached dictionary.
Then, call `get!` on the dictionary with `key` and `f(key)`.
  
An optional `id` can be given choose the used cache. This is useful to avoid clashes of
dictionaries using the same types.

See also [`cache_dict`](@ref) and `get!`.
"""
function get_from_cache(::Type{K}, ::Type{V}, key, f, id=Val{1}()) where {K, V}
    dict = cache_dict(K, V, id)
    get!(dict, key) do
        f(key)
    end
end

"""
    get_derivative_idx(der_key::NTuple{manifold_dim, Int}, id=Val{1}()) where {manifold_dim}

Convert the given derivative key to a linear index corresponding to its storage location.

If `local_basis` corresponds to basis evaluations for some `manifold_dim`-variate function
space, then its `k`-th derivatives will all be stored in the location `local_basis[k+1]`.
Moreover, the `k`-th derivative corresponding to the key `[i₁,i₂,...,iₙ]` in the location
`local_basis[k+1][m]` where:
- `m = 1` when `iⱼ = 0` for all `j`, i.e., for basis function values;
- `m = 1+r` when `iⱼ = 0` for all `j` except for `j = r` and `iⱼ = 1`, i.e.,
    for the first derivative w.r.t. the `j`-th canonical coordinate;
- in all other cases (i.e., when `k>1`),  the value of `m` is equal to `l`
    if `[i₁,i₂,...,iₙ]` is the `l`-th key returned by the function
    `integer_sums(k, Val(manifold_dim))`.

As an example, consider the first derivative with respect to x₁ (∂/∂x₁)
in 2D, which has key [1, 0]. In 3D, this same derivative has key
[1, 0, 0]. In 3D, the derivative ∂³/∂x₁²∂x₂ thus has key [2, 1, 0].

An optional `id` can be given choose the used cache. This is useful to avoid clashes of
dictionaries using the same types. See also [`cache_dict`](@ref).

# Arguments
- `der_key::NTuple{manifold_dim, Int}`: A key for the desired derivative order.

# Returns
- `::Int`: The linear index corresponding to the derivative's storage location in basis
    evaluations.
"""
function get_derivative_idx(
    der_key::NTuple{manifold_dim, Int}, id=Val{1}()
) where {manifold_dim}
    return get_from_cache(NTuple{manifold_dim, Int}, Int, der_key, _get_derivative_idx, id)
end

function _get_derivative_idx(der_key::NTuple{manifold_dim, Int}) where {manifold_dim}
    if any(<(0), der_key)
        throw(ArgumentError("Derivative key $der_key is not valid!"))
    end

    s = sum(der_key)
    if s == 0
        return 1
    elseif s == 1
        return findfirst(==(1), der_key)::Int
    end

    all_keys = integer_sums(s, Val(manifold_dim))
    derivative_idx = findfirst(==(der_key), all_keys)
    if isnothing(derivative_idx)
        throw(ArgumentError("Derivative key $der_key is not valid!"))
    end

    return derivative_idx
end

"""
    integer_sums(sum_indices::Int, num_indices::Val{N}, id=Val{1}()) where {N}

Generates all possible combinations of non-negative integers that sum up to a given value,
where each combination has a specified number of elements.

An optional `id` can be given choose the used cache. This is useful to avoid clashes of
dictionaries using the same types. See also [`cache_dict`](@ref).

# Arguments
- `sum_indices::Int`: The target sum of the integers in each combination.
- `num_indices::Val{N}`: The number of integers in each combination, given by `N`.

# Returns
- `::Vector{NTuple{N, Int}}`: Each inner vector represents a combination of integers that
    sum up to `sum_indices`. If no valid combinations exist, the vectors are empty.
"""
function integer_sums(sum_indices::Int, num_indices::Val{N}, id=Val{1}()) where {N}
    return get_from_cache(
        Int, Vector{NTuple{N, Int}}, sum_indices, s -> _integer_sums(s, num_indices), id
    )
end

function _integer_sums(sum_indices::Int, ::Val{1})
    if sum_indices < 0
        return Tuple{Int}[]
    end

    return [(sum_indices,)]
end

function _integer_sums(sum_indices::Int, ::Val{N}) where {N}
    N > 0 || throw(ArgumentError("Number of indices must be greater than 0. Got $(N)."))
    solutions = Vector{NTuple{N, Int}}(undef, 0)
    for combo in Combinatorics.combinations(0:(sum_indices + N - 2), N - 1)
        s = zeros(Int, N)
        s[1] = combo[1]
        for i in 2:(N - 1)
            s[i] = combo[i] - combo[i - 1] - 1
        end

        s[end] = sum_indices + N - 2 - combo[N - 1]
        push!(solutions, NTuple{N, Int}(s))
    end

    return solutions
end

"""
    integer_sums(init_sum::Int, final_sum::Int, num_indices::Val, id=Val{1}())

Generates all possible combinations of non-negative integers that sum up to `init_sum` until
`final_sum`, where each combination has `num_indices` length.

An optional `id` can be given choose the used cache. This is useful to avoid clashes of
dictionaries using the same types. See also [`cache_dict`](@ref).

# Examples
```jldoctest
julia> using Mantis;

julia> Mantis.GeneralHelpers.integer_sums(0, 2, Val(2))
6-element Vector{Tuple{Int64, Int64}}:
 (0, 0)
 (0, 1)
 (1, 0)
 (0, 2)
 (1, 1)
 (2, 0)

```
"""
function integer_sums(init_sum::Int, final_sum::Int, num_indices::Val, id=Val{1}())
    return mapreduce(s -> integer_sums(s, num_indices, id), vcat, init_sum:final_sum)
end

"""
    num_der_indices(n, d)

Return the number of distinct partial derivatives of order `d` in an `n`-variate space,
assuming equality of mixed partial derivatives.

# Examples
```jldoctest
using Mantis

julia> Mantis.GeneralHelpers.num_der_indices(1, 2)
1

julia> Mantis.GeneralHelpers.num_der_indices(3, 2)
6

```
"""
num_der_indices(n, d) = binomial(n + d - 1, n - 1)

####################################################
# Export path helper function
####################################################
"""
    export_path(output_directory_tree::Vector, filename)

Create a directory (if needed) and return the path to the output file.

# Arguments
- `output_directory_tree::Vector`: A vector representing the directory tree.
- `filename`: The name of the output file.

# Example
```@example
output_file = export_path(["examples", "data", "output"], "output.vtu")
```
"""
function export_path(output_directory_tree::Vector, filename)
    output_directory = joinpath(output_directory_tree...)
    output_file = joinpath(output_directory, filename)

    if !isdir(output_directory)
        println("Creating new directory $output_directory ...")
        mkpath(output_directory)
    end

    return output_file
end

end

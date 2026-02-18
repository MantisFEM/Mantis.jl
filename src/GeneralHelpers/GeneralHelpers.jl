module GeneralHelpers

export integer_sums, get_derivative_idx, export_path

import Combinatorics

####################################################
# Integer sums and derivatives helper functions
####################################################

function _integer_sums_no_cache(sum_indices::Int, num_indices::Int)
    solutions = Vector{Vector{Int}}(undef, 0)
    if num_indices == 1
        push!(solutions, [sum_indices])
    elseif num_indices > 1
        for combo in
            Combinatorics.combinations(0:(sum_indices + num_indices - 2), num_indices - 1)
            s = zeros(Int, num_indices)
            s[1] = combo[1]
            for i in 2:(num_indices - 1)
                s[i] = combo[i] - combo[i - 1] - 1
            end
            s[end] = sum_indices + num_indices - 2 - combo[num_indices - 1]
            push!(solutions, s)
        end
    else
        throw(ArgumentError("Number of indices must be greater than 0."))
    end

    return solutions
end

"""
    integer_sums(sum_indices::Int, num_indices::Int)

Generates all possible combinations of non-negative integers that sum up
to a given value, where each combination has a specified number of elements.

# Arguments
- `sum_indices::Int`: The target sum of the integers in each combination.
- `num_indices::Int`: The number of integers in each combination.

# Returns
- `::Vector{Vector{Int}}`: Each inner vector represents a combination of integers that sum
    up to `sum_indices`. If no valid combinations exist, the vectors are empty.
"""
const integer_sums = let
    cache = Dict{Tuple{Int, Int}, Vector{Vector{Int}}}()

    function (sum_indices::Int, num_indices::Int)
        get!(cache, (sum_indices, num_indices)) do
            _integer_sums_no_cache(sum_indices, num_indices)
        end
    end
end

function _get_derivative_idx_no_cache(der_key::Vector{Int})
    if any(der_key .< 0)
        throw(ArgumentError("Derivative key $der_key is not valid!"))
    end

    if sum(der_key) == 0
        # Request 0th order derivatives, i.e., evaluation of the function
        derivative_idx = 1

    elseif sum(der_key) == 1
        # Request first derivatives

        # Trivial indexing: the linear index associated to the input key der_key is just the
        # index of the value with the 1 (the first derivative we wish)
        derivative_idx = findfirst(x -> x == 1, der_key)

    else
        # Request derivatives with order higher than 1

        # Generate all valid derivative keys
        all_keys = integer_sums(sum(der_key), length(der_key))

        # Find the linear index associated to the input key der_key
        derivative_idx = findfirst(x -> x == der_key, all_keys)

        if isnothing(derivative_idx)
            throw(ArgumentError("Derivative key $der_key is not valid!"))
        end
    end

    return derivative_idx
end

"""
    get_derivative_idx(der_key::Vector{Int})

Convert the given derivative key to a linear index corresponding to its storage location.

If `local_basis` corresponds to basis evaluations for some `manifold_dim`-variate function
space, then its `k`-th derivatives will all be stored in the location `local_basis[k+1]`.
Moreover, the `k`-th derivative corresponding to the key `[i₁,i₂,...,iₙ]` in the
location `local_basis[k+1][m]` where:
- `m = 1` when `iⱼ = 0` for all `j`, i.e., for basis function values;
- `m = 1+r` when `iⱼ = 0` for all `j` except for `j = r` and `iⱼ = 1`, i.e.,
    for the first derivative w.r.t. the `j`-th canonical coordinate;
- in all other cases (i.e., when `k>1`),  the value of `m` is equal to `l`
    if `[i₁,i₂,...,iₙ]` is the `l`-th key returned by the function
    `integer_sums(k, manifold_dim)`.

As an example, consider the first derivative with respect to x₁ (∂/∂x₁)
in 2D, which has key [1, 0]. In 3D, this same derivative has key
[1, 0, 0]. In 3D, the derivative ∂³/∂x₁²∂x₂ thus has key [2, 1, 0].

# Arguments
- `der_key::Vector{Int}`: A key for the desired derivative order.

# Returns
- `::Int`: The linear index corresponding to the derivative's storage location in basis
    evaluations.
"""
const get_derivative_idx = let
    cache = Dict{Vector{Int}, Int}()

    function (der_key::Vector{Int})
        get!(cache, der_key) do
            _get_derivative_idx_no_cache(der_key)
        end
    end
end

####################################################
# Export path helper function
####################################################
"""
    export_path(output_directory_tree::Vector{String}, filename::String)

Create a directory (if needed) and return the path to the output file.

# Arguments
- `output_directory_tree::Vector{String}`: A vector of strings representing the directory tree.
- `filename::String`: The name of the output file.

# Example
```julia
output_file = export_path(["examples", "data", "output"], "output.vtk") # "examples/data/output/output.vtk"
```
"""
function export_path(output_directory_tree::Vector{String}, filename::String)
    Mantis_folder = dirname(dirname(pathof(parentmodule(GeneralHelpers))))
    output_directory = joinpath(Mantis_folder, output_directory_tree...)
    output_file = joinpath(output_directory, filename)

    if !isdir(output_directory)
        println("Creating new directory $output_directory ...")
        mkpath(output_directory)
    end

    return output_file
end

end

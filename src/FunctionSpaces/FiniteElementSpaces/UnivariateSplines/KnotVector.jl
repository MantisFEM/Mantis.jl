"""
    KnotVector

1-dimensional knot vector.

# Fields
- `geometry::G`: A 1D geometry.
- `polynomial_degree::Int`: Polynomial degree.
- `multiplicity::TM`: Number of repetitions of each knot.
"""
struct KnotVector{G, TM}
    geometry::G
    polynomial_degree::Int
    multiplicity::TM

    function KnotVector(
        geometry::G, polynomial_degree::Int, multiplicity::TM
    ) where {
        image_dim,
        G <: Geometry.CartesianGeometry{1, image_dim, 1},
        TM <: AbstractVector{Int},
    }
        num_elements = Geometry.get_num_elements(geometry)
        if num_elements + 1 != length(multiplicity)
            throw(
                ArgumentError(
                    LazyString(
                        "The number of breakpoints and multiplicities must be the same, ",
                        "but there are ",
                        num_elements + 1,
                        " breakpoints and ",
                        length(multiplicity),
                        " multiplicities.",
                    ),
                ),
            )
        end

        # Ensure all multiplicities are greater than 0
        if any(multiplicity .<= 0)
            error_index = findfirst(multiplicity .<= 0)
            throw(
                ArgumentError(
                    LazyString(
                        "All multiplicities must be greater than or equal to 1. However ",
                        "at index ",
                        error_index,
                        " the multiplicity is ",
                        multiplicity[error_index],
                        ".",
                    ),
                ),
            )
        end

        return new{G, TM}(geometry, polynomial_degree, multiplicity)
    end
end

function KnotVector(
    geometry::G, polynomial_degree::Int, multiplicity::Int
) where {G <: Geometry.AbstractGeometry{1}}
    return KnotVector(geometry, polynomial_degree, [multiplicity])
end

get_geometry(knot_vector::KnotVector) = knot_vector.geometry

get_knot_vector_types(::KnotVector{G, TM}) where {G, TM} = G, TM

get_num_elements(knot_vector::KnotVector) =
    Geometry.get_num_elements(get_geometry(knot_vector))

"""
    create_knot_vector(
        breakpoints::AbstractVector{NT},
        p::Int,
        breakpoint_condition::AbstractVector{Int},
        condition_type::String,
    ) where {NT <: Number}

Creates a uniform knot vector corresponding to B-splines basis functions of polynomial
degree `p` and continuity `regularity[i]` on `breakpoints[i]`.

# Arguments
- `breakpoints::AbstractVector{NT}`: Location of each breakpoint.
- `p::Int`: Degree of the polynomial (``p ≥ 0``).
- `breakpoint_condition::AbstractVector{Int}`: Either the regularity or multiplicity of each
    breakpoint.
- `condition_type::String`: Determines whether `breakpoint_condition` provides the
    regularity or multiplicity.

# Returns
- `::KnotVector`: Knot vector.
"""
function create_knot_vector(
    geometry::Geometry.AbstractGeometry{1},
    p::Int,
    breakpoint_condition::AbstractVector{Int},
    condition_type::String,
)
    if condition_type == "regularity"
        multiplicity = similar(breakpoint_condition)

        # Calculate multiplicity based on regularity
        for i in eachindex(breakpoint_condition)
            multiplicity[i] = p - breakpoint_condition[i]
        end

        return KnotVector(geometry, p, multiplicity)

    elseif condition_type == "multiplicity"
        return KnotVector(geometry, p, breakpoint_condition)

    else
        throw(
            ArgumentError(
                LazyString(
                    "Allowed breakpoint conditions are `regularity` or `multiplicity`, but",
                    " the given condition is ",
                    condition_type,
                    ".",
                ),
            ),
        )
    end
end

"""
    get_multiplicity(knot_vector::KnotVector)

Returns the multiplicity of each knot in `knot_vector`.

# Arguments
- `knot_vector::KnotVector`.

# Returns
- `<:AbstractVector{Int}`: Multiplicity of each knot.
"""
function get_multiplicity(knot_vector::KnotVector)
    return knot_vector.multiplicity
end

get_polynomial_degree(knot_vector::KnotVector) = knot_vector.polynomial_degree

function get_breakpoints(knot_vector::KnotVector)
    return Geometry.get_breakpoints_per_dim(get_geometry(knot_vector), 1, 1)
end

function get_element_measure(knot_vector::KnotVector, element_id::Int)
    return Geometry.get_element_measure(get_geometry(knot_vector), element_id)
end

"""
    get_knot_vector_length(knot_vector::KnotVector)

Determines the length of `knot_vector` by summing the multiplicities of each knot vector.

# Arguments
- `knot_vector::KnotVector`: Knot vector for length calculation.

# Returns
- `::Int`: Length of the knot vector.
"""
function get_knot_vector_length(knot_vector::KnotVector)
    return sum(knot_vector.multiplicity)
end

"""
    convert_knot_to_breakpoint_idx(knot_vector::KnotVector, knot_index::Int)

Retrieves the breakpoint index corresponding to `knot_vector` at `knot_index`, i.e., the
index of the vector where every `breakpoint[i]` appears `knot_vector.multiplicity[i]`-times.

# Arguments
- `knot_vector::KnotVector`: Knot vector for length calculation.
- `knot_index::Int`: Index in the knot vector.

# Returns
- `::Int`: Index of breakpoint corresponding to `knot_index`.
"""
function convert_knot_to_breakpoint_idx(knot_vector::KnotVector, knot_index::Int)
    return findfirst(idx -> idx >= knot_index, cumsum(knot_vector.multiplicity))
end

"""
    get_first_knot_index(knot_vector::KnotVector, breakpoint_index::Int)

Get the index of the first knot corresponding to a given breakpoint.

# Arguments
- `knot_vector::KnotVector`: The knot vector.
- `breakpoint_index::Int`: Index of the breakpoint.

# Returns
- `::Int`: Index of the first knot for the given breakpoint.
"""
function get_first_knot_index(knot_vector::KnotVector, breakpoint_index::Int)
    return cumsum(knot_vector.multiplicity)[breakpoint_index] -
           knot_vector.multiplicity[breakpoint_index] + 1
end

"""
    get_last_knot_index(knot_vector::KnotVector, breakpoint_index::Int)

Get the index of the last knot corresponding to a given breakpoint.

# Arguments
- `knot_vector::KnotVector`: The knot vector.
- `breakpoint_index::Int`: Index of the breakpoint.

# Returns
- `::Int`: Index of the last knot for the given breakpoint.
"""
function get_last_knot_index(knot_vector::KnotVector, breakpoint_index::Int)
    return cumsum(knot_vector.multiplicity)[breakpoint_index]
end

"""
    get_knot_value(knot_vector::KnotVector, knot_index::Int)

Retrieves the breakpoint corresponding to `knot_vector` at `knot_index`, i.e., the index
of the vector where every `breakpoint[i]` appears `knot_vector.multiplicity[i]`-times.

# Arguments
- `knot_vector::KnotVector`: Knot vector for length calculation.
- `knot_index::Int`: Index in the knot vector.

# Returns
- `::Number`: Breakpoint corresponding to `knot_index`.
"""
function get_knot_value(knot_vector::KnotVector, knot_index::Int)
    idx = convert_knot_to_breakpoint_idx(knot_vector, knot_index)
    return get_breakpoints(knot_vector)[idx]
end

"""
    get_knot_multiplicity(knot_vector::KnotVector, knot_index::Int)

Retrieves the multiplicity of the breakpoint corresponding to `knot_vector` at
`knot_index`, i.e., the index of the vector where every `breakpoint[i]` appears
`knot_vector.multiplicity[i]`-times.

# Arguments
- `knot_vector::KnotVector`: Knot vector for length calculation.
- `knot_index::Int`: Index in the knot vector.

# Returns
- `::Int`: Multiplicity of the breakpoint corresponding to `knot_index`.
"""
function get_knot_multiplicity(knot_vector::KnotVector, knot_index::Int)
    index = convert_knot_to_breakpoint_idx(knot_vector, knot_index)
    return knot_vector.multiplicity[index]
end

"""
    get_local_knot_vector(knot_vector::KnotVector, basis_id::Int)

Returns the local knot vector necessary to characterize the B-spline identified by
`basis_id`.

# Arguments
- `knot_vector::KnotVector`: The knot vector of the full B-spline basis.
- `basis_id::Int`: The id of the B-spline.

# Returns
- `::KnotVector`: The knot vector of the B-spline identified by `basis_id`.
"""
function get_local_knot_vector(knot_vector::KnotVector, basis_id::Int)
    deg = get_polynomial_degree(knot_vector)
    knot_cum_sum = cumsum(get_multiplicity(knot_vector))
    first_breakpoint_idx = convert_knot_to_breakpoint_idx(knot_vector, basis_id)
    last_breakpoint_idx = convert_knot_to_breakpoint_idx(knot_vector, basis_id + deg + 1)
    first_knot_mult = knot_cum_sum[first_breakpoint_idx] - basis_id + 1
    last_knot_mult = basis_id + deg + 1 - knot_cum_sum[last_breakpoint_idx - 1]
    geometry = Geometry.CartesianGeometry(
        get_breakpoints(knot_vector)[first_breakpoint_idx:last_breakpoint_idx]
    )
    multiplicity = vcat(
        first_knot_mult,
        get_multiplicity(knot_vector)[(first_breakpoint_idx + 1):(last_breakpoint_idx - 1)],
        last_knot_mult,
    )

    return KnotVector(geometry, deg, multiplicity)
end

"""
    get_greville_points(knot_vector::KnotVector)

Compute the Greville points for the given knot vector.

# Arguments
- `knot_vector::KnotVector`: The knot vector.

# Returns
- `::Tuple{Vector{Float64}}`: Vector of Greville points.
"""
function get_greville_points(knot_vector::KnotVector)
    p = get_polynomial_degree(knot_vector)
    num_knots = sum(get_multiplicity(knot_vector))
    num_basis = num_knots - (p + 1)
    greville_points = zeros(num_basis)
    if p == 0
        for i in eachindex(greville_points)
            greville_points[i] =
                (get_knot_value(knot_vector, i) + get_knot_value(knot_vector, i + 1)) / 2
        end
    else
        for i in eachindex(greville_points)
            greville_points[i] =
                sum(get_knot_value(knot_vector, ii) for ii in (i + 1):(i + p)) / p
        end
    end

    return (greville_points,)
end

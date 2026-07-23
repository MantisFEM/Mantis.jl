
"""
    extract_bspline_to_section_space(
        knot_vector::KnotVector, canonical_space::AbstractCanonicalSpace
    )

Compute the extraction coefficients of B-Spline basis functions in terms of canonical basis
functions; see [Borden2011](@cite) [Toshniwal2020](@cite) [Hiemstra2020](@cite).

# Arguments

  - `knot_vector::KnotVector`: The knot vector defining the B-Spline basis.
  - `polynomials::AbstractCanonicalSpace`: The canonical space to extract to.

# Returns

  - `::ExtractionOperator{Indices{1, TE, TI, TJ}}`: See [`ExtractionOperator`](@ref)
    for the details.

# Note

The extraction coefficients `E[el]` for element `el` contain the coefficients of the linear
combination of reference Canonical polynomials determining the basis functions on that
element.
"""
function extract_bspline_to_section_space(
    knot_vector::KnotVector, canonical_space::AbstractCanonicalSpace
)
    return error(
        "The extraction to $(typeof(canonical_space)) spaces has not been implemented."
    )
end

function extract_bspline_to_section_space(
    knot_vector::KnotVector, canonical_space::AbstractLagrangePolynomials
)
    nel = get_num_elements(knot_vector)
    p = get_polynomial_degree(knot_vector)
    multiplicity = get_multiplicity(knot_vector)

    # Ensure that regularities at all breakpoints are either 0 or -1
    for el in 1:(nel - 1)
        # Get multiplicity of the knot at the end of the current element
        mult = multiplicity[el + 1]
        if mult < p
            throw(
                ArgumentError(
                    "The Lagrange extraction is only implemented for regularities 0 and -1."
                ),
            )
        end
    end

    # Initialize extraction matrices for each element
    E = [(LinearAlgebra.I,) for _ in 1:nel]

    # Compute indices of supported basis functions on each element
    basis_indices = Vector{Indices{1, UnitRange{Int}, UnitRange{Int}}}(undef, nel)
    basis_indices[1] = Indices(1:(p + 1), (1:(p + 1),))
    for el in 2:nel
        basis_indices[el] = Indices(
            get_basis_indices(basis_indices[el - 1]) .+ multiplicity[el], (1:(p + 1),)
        )
    end

    return ExtractionOperator(
        E, basis_indices, nel, get_basis_indices(basis_indices[end])[end]
    )
end

function extract_bspline_to_section_space(
    knot_vector::KnotVector, canonical_space::AbstractEdgePolynomials
)
    nel = get_num_elements(knot_vector)
    p = get_polynomial_degree(knot_vector)
    multiplicity = get_multiplicity(knot_vector)

    # Ensure that regularities at all breakpoints are either 0 or -1
    for el in 1:(nel - 1)
        # Get multiplicity of the knot at the end of the current element
        mult = multiplicity[el + 1]
        if mult < p + 1
            throw(
                ArgumentError(
                    "The Edge-polynomial extraction is only implemented for regularity -1."
                ),
            )
        end
    end

    # Initialize extraction matrices for each element
    E = [(LinearAlgebra.I,) for _ in 1:nel]

    # Compute indices of supported basis functions on each element
    basis_indices = Vector{Indices{1, UnitRange{Int}, UnitRange{Int}}}(undef, nel)
    basis_indices[1] = Indices(1:(p + 1), (1:(p + 1),))
    for el in 2:nel
        basis_indices[el] = Indices(
            get_basis_indices(basis_indices[el - 1]) .+ multiplicity[el], (1:(p + 1),)
        )
    end

    return ExtractionOperator(
        E, basis_indices, nel, get_basis_indices(basis_indices[end])[end]
    )
end

function extract_bspline_to_section_space(
    knot_vector::KnotVector, canonical_space::Bernstein
)
    nel = get_num_elements(knot_vector)
    p = get_polynomial_degree(knot_vector)
    breakpoints = get_breakpoints(knot_vector)
    num_knots = sum(get_multiplicity(knot_vector))

    # Initialize extraction matrices for each element
    E = [(Matrix{Float64}(LinearAlgebra.I, p + 1, p + 1),) for _ in 1:nel]

    # Array to store knot insertion coefficients
    alphas = zeros(max(p - 1, 0))

    # indices used for computing alphas
    k_id_start = p + 1
    k_id_end = k_id_start + 1
    el_id = 1

    # Iterate over all knot indices
    while k_id_end < num_knots
        b_id_start = convert_knot_to_breakpoint_idx(knot_vector, k_id_start)
        b_id_end = convert_knot_to_breakpoint_idx(knot_vector, k_id_end)

        # Count forwrd multiplicity of knot at k_id_end
        t = k_id_end
        while k_id_end < num_knots &&
              convert_knot_to_breakpoint_idx(knot_vector, k_id_end + 1) == b_id_end
            k_id_end += 1
        end
        mult = k_id_end - t + 1
        b_id_end = convert_knot_to_breakpoint_idx(knot_vector, k_id_end)

        # If multiplicity is less than polynomial degree, perform knot insertion
        if mult < p
            r = p - mult
            # Calculate numerator for alpha coefficients
            numer = breakpoints[b_id_end] - breakpoints[b_id_start]
            # Compute alpha coefficients
            for j in p:-1:(mult + 1)
                b_id_j = convert_knot_to_breakpoint_idx(knot_vector, k_id_start + j)
                alphas[j - mult] = numer / (breakpoints[b_id_j] - breakpoints[b_id_start])
            end

            # Update extraction coefficients
            for j in 1:r
                s = mult + j
                for k in (p + 1):-1:(s + 1)
                    alpha = alphas[k - s]
                    E[el_id][1][k, :] .=
                        (@view E[el_id][1][k, :]) .* alpha .+
                        (@view E[el_id][1][k - 1, :]) .* (1.0 - alpha)
                end

                # Save coefficients for the next element
                save = r - j + 1
                if k_id_end < num_knots && el_id < nel
                    E[el_id + 1][1][save, save:(save + j)] .= (@view E[el_id][1][
                        p + 1, (p - j + 1):(p + 1)
                    ])
                end
            end
        end
        el_id += 1
        if k_id_end < num_knots
            k_id_start = k_id_end
            k_id_end += 1
        end
    end

    # Compute indices of supported basis functions on each element
    basis_indices = Vector{Indices{1, UnitRange{Int}, UnitRange{Int}}}(undef, nel)
    basis_indices[1] = Indices(1:(p + 1), (1:(p + 1),))
    for el in 2:nel
        basis_indices[el] = Indices(
            get_basis_indices(basis_indices[el - 1]) .+ get_multiplicity(knot_vector)[el],
            (1:(p + 1),),
        )
    end

    return ExtractionOperator(
        E, basis_indices, nel, get_basis_indices(basis_indices[end])[end]
    )
end

function extract_bspline_to_section_space(
    knot_vector::KnotVector, canonical_space::AbstractECTSpaces
)
    nel = get_num_elements(knot_vector)
    p = get_polynomial_degree(knot_vector)
    multiplicity = get_multiplicity(knot_vector)

    # Construct cumulative sum of all element-wise canonical space dimensions
    canonical_dims = cumsum([0, repeat([p + 1], nel, 1)...])

    # Initialize global extraction matrix as an identity matrix
    H = SparseArrays.sparse(
        1:canonical_dims[nel + 1],
        1:canonical_dims[nel + 1],
        ones(Float64, canonical_dims[nel + 1]),
        canonical_dims[nel + 1],
        canonical_dims[nel + 1],
    )

    # Loop over all internal breakpoints and update extraction by imposing smoothness
    for el in 1:(nel - 1)
        # Get regularity at this breakpoint
        mult = multiplicity[el + 1]
        r = p - mult

        # Construct smoothness constraint matrix contributions from the left and right of
        # the breakpoint
        KL = _evaluate_all_at_point(canonical_space, 1.0, r)
        SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, KL)
        KR = _evaluate_all_at_point(canonical_space, 0.0, r)
        SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, KR)
        # element sizes where constraints are evaluated
        h_L = get_element_measure(knot_vector, el)
        h_R = get_element_measure(knot_vector, el + 1)
        # scale the constraints by the element sizes and findnz values
        scaling_L = [h_L^(-j) for j in 0:r]
        scaling_R = [h_R^(-j) for j in 0:r]
        KL = SparseArrays.findnz(SparseArrays.sparse(KL * LinearAlgebra.diagm(scaling_L)))
        KR = SparseArrays.findnz(SparseArrays.sparse(KR * LinearAlgebra.diagm(scaling_R)))
        # join the constraints together
        rows = [KL[1]; KR[1] .+ (canonical_dims[el + 1] - canonical_dims[el])]
        cols = [KL[2]; KR[2]]
        vals = [-KL[3]; KR[3]]
        K = SparseArrays.sparse(
            rows, cols, vals, (canonical_dims[el + 2] - canonical_dims[el]), r + 1
        )

        # Update local extraction matrix by building double-diagonal nullspace
        L = H[:, (canonical_dims[el] + 1):canonical_dims[el + 2]] * K
        for j in 0:r
            Hbar = build_sparse_nullspace(L[:, j + 1])
            H = Hbar * H
            L = Hbar * L
        end
    end

    # Remove small values obtained as a result of round-off errors
    SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, H)

    # Compute basis indices of supported basis functions on each element.
    basis_indices = Vector{Indices{1, UnitRange{Int}, UnitRange{Int}}}(undef, nel)
    basis_indices[1] = Indices(1:(p + 1), (1:(p + 1),))
    for el in 2:nel
        basis_indices[el] = Indices(
            get_basis_indices(basis_indices[el - 1]) .+ multiplicity[el], (1:(p + 1),)
        )
    end

    # Convert global extraction matrix to element local extractions
    # (the matrix is transposed here so that [canonical_space] * [extraction] = [B-splines])
    extraction_coefficients = Vector{Matrix{Float64}}(undef, nel)
    for el in 1:nel
        cols_el = (canonical_dims[el] + 1):canonical_dims[el + 1]
        # Matrix of coefficients
        extraction_coefficients[el] = Matrix(
            H[get_basis_indices(basis_indices[el]), cols_el]
        )'
    end

    Etup = [(extraction_coefficients[el],) for el in eachindex(extraction_coefficients)]

    return ExtractionOperator(Etup, basis_indices, nel, size(H, 1))
end

"""
    extract_gtbspline_to_bspline(
        spline_spaces::NTuple{m, F}, regularity::Vector{Int}
    ) where {m, F <: Union{BSplineSpace, RationalFESpace}}

Compute the extraction coefficients of GTB-Spline basis functions in terms of (rational)
B-spline basis functions.

# Arguments

  - `spline_spaces::NTuple{m,F}`: Collection of (rational) B-spline spaces.
  - `regularity::Vector{Int}`: Smoothness to be imposed at patch interfaces.

# Returns

  - `ExtractionOperator{Indices{1, TE, TI, TJ}}`: The extraction operator containing the
    coefficients. See [`ExtractionOperator`](@ref) for the details.
"""
function extract_gtbspline_to_bspline(
    spline_spaces::NTuple{m, F}, regularity::Vector{Int}
) where {m, F <: Union{BSplineSpace, RationalFESpace}}
    # Construct cumulative sum of all B-spline dimensions
    spl_dims = zeros(Int, m + 1)
    for i in 2:(m + 1)
        spl_dims[i] = spl_dims[i - 1] + get_num_basis(spline_spaces[i - 1])
    end

    # Number of elements for all spline_spaces
    bspl_nels = [get_num_elements(spline_spaces[i]) for i in 1:m]
    nel = sum(bspl_nels)

    # Initialize global extraction matrix
    H = SparseArrays.sparse(
        1:spl_dims[m + 1],
        1:spl_dims[m + 1],
        ones(Float64, spl_dims[m + 1]),
        spl_dims[m + 1],
        spl_dims[m + 1],
    )

    # Loop over all internal patch interfaces and update extraction by imposing smoothness
    for i in 1:(m - 1)
        # Regularity at this interface
        r = regularity[i]

        # Smoothness constraint matrix
        KL = _evaluate_all_at_point(spline_spaces[i], bspl_nels[i], 1.0, r)
        SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, KL)
        KR = _evaluate_all_at_point(spline_spaces[i + 1], 1, 0.0, r)
        SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, KR)
        # element sizes where constraints are evaluated
        h_L = get_element_measure(spline_spaces[i], bspl_nels[i])
        h_R = get_element_measure(spline_spaces[i + 1], 1)
        # scale the constraints by the element sizes and findnz values
        scaling_L = [h_L^(-j) for j in 0:r]
        scaling_R = [h_R^(-j) for j in 0:r]
        KL = SparseArrays.findnz(SparseArrays.sparse(KL * LinearAlgebra.diagm(scaling_L)))
        KR = SparseArrays.findnz(SparseArrays.sparse(KR * LinearAlgebra.diagm(scaling_R)))
        # join the constraints together
        rows = [KL[1]; KR[1] .+ (spl_dims[i + 1] - spl_dims[i])]
        cols = [KL[2]; KR[2]]
        vals = [-KL[3]; KR[3]]
        K = SparseArrays.sparse(rows, cols, vals, (spl_dims[i + 2] - spl_dims[i]), r + 1)

        # Update local extraction matrix by building double-diagonal nullspace of constraints
        L = H[:, (spl_dims[i] + 1):spl_dims[i + 2]] * K
        for j in 0:r
            Hbar = build_sparse_nullspace(L[:, j + 1])
            H = Hbar * H
            L = Hbar * L
        end
    end

    # Impose periodicity if desired for i = m
    if regularity[m] > -1
        r = regularity[m]
        if size(H, 1) >= 2 * (r + 1)
            Hper = circshift(H, r + 1)

            # smoothness constraints
            KL = _evaluate_all_at_point(spline_spaces[m], bspl_nels[m], 1.0, r)
            SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, KL)
            KR = _evaluate_all_at_point(spline_spaces[1], 1, 0.0, r)
            SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, KR)
            # element sizes where constraints are evaluated
            h_L = get_element_measure(spline_spaces[m], bspl_nels[m])
            h_R = get_element_measure(spline_spaces[1], 1)
            # scale the constraints by the element sizes and findnz values
            scaling_L = [h_L^(-j) for j in 0:r]
            scaling_R = [h_R^(-j) for j in 0:r]
            KL = SparseArrays.findnz(
                SparseArrays.sparse(KL * LinearAlgebra.diagm(scaling_L))
            )
            KR = SparseArrays.findnz(
                SparseArrays.sparse(KR * LinearAlgebra.diagm(scaling_R))
            )

            rows = [KL[1]; KR[1] .+ (spl_dims[m + 1] - spl_dims[m])]
            cols = [KL[2]; KR[2]]
            vals = [-KL[3]; KR[3]]
            K = SparseArrays.sparse(
                rows,
                cols,
                vals,
                (spl_dims[m + 1] - spl_dims[m] + spl_dims[2] - spl_dims[1]),
                r + 1,
            )
            Lper =
                Hper[
                    :, [(spl_dims[m] + 1):spl_dims[m + 1]; (spl_dims[1] + 1):spl_dims[2]]
                ] * K
            for j in 0:r
                Hbar = build_sparse_nullspace(Lper[:, j + 1])
                Hper = Hbar * Hper
                Lper = Hbar * Lper
            end
            H = Hper
        end
    end

    # Remove small values obtained as a result of round-off errors.
    SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, H)

    # Convert global extraction matrix to element local extractions.
    # The matrix is transposed so that [spline_spaces] * [extraction] = [GTB-splines].
    extraction_coefficients = Vector{Tuple{Matrix{Float64}}}(undef, nel)
    basis_indices = Vector{Indices{1, Vector{Int}, UnitRange{Int}}}(undef, nel)
    count = 0
    for i in 1:m
        for j in 1:bspl_nels[i]
            cols_ij = get_basis_indices(spline_spaces[i], j)
            eij = SparseArrays.findnz(H[:, cols_ij .+ spl_dims[i]])
            # Unique indices for non-zero rows and columns
            unique_eij = unique(eij[1])
            basis_indices[count + 1] = Indices(unique_eij, (1:length(unique_eij),))
            # Matrix of coefficients
            extraction_coefficients[count + 1] = (
                Matrix(
                    H[get_basis_indices(basis_indices[count + 1]), cols_ij .+ spl_dims[i]]
                )',
            )
            count += 1
        end
    end

    return ExtractionOperator(extraction_coefficients, basis_indices, nel, size(H, 1))
end

"""
    build_sparse_nullspace(constraint::SparseArrays.SparseVector{Float64})

Build the sparsest possible nullspace of a constraint vector with no zero entries.

# Arguments

  - `constraint::SparseArrays.SparseVector{Float64}`: The constraint vector.

# Returns

  - `::SparseMatrixCSC{Float64}`: The sparse nullspace matrix.
"""
function build_sparse_nullspace(constraint::SparseArrays.SparseVector{Float64})
    q = length(constraint)
    nz_flag = .!isapprox.(constraint, 0.0; atol=1e-14)
    i1 = findfirst(nz_flag)
    i2 = findlast(nz_flag)
    dd = zeros(Float64, q - 1, 2)
    dd[1:i1, 1] .= 1.0
    for j in i1:(i2 - 2)
        dd[j, 2] = -constraint[j] / constraint[j + 1] * dd[j, 1]
        dd[j + 1, 1] = 1 - dd[j, 2]
    end
    dd[(i2 - 1):(q - 1), 2] .= 1.0

    return SparseArrays.spdiagm(q - 1, q, 0 => dd[:, 1], 1 => dd[:, 2])
end

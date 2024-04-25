"""
    create_identity(nel::Int, d::Int)::Array{Float64, 3}

Creates a `(d, d, ni)` array where each slice along the last dimension is a `(d, d)` identity matrix.

# Arguments 
- `ni::Int`: number of identity matrices in the array.
- `d::Int`: size of the identity matrices in the array.

# Returns
- `Id::Array{Float64, 3}`: array of identity matrices with size (d, d, ni).
"""
function create_identity(ni::Int, d::Int)
    Id = Vector{Array{Float64, 2}}(undef, ni)
    for i in 1:ni
        Id[i] = zeros(d,d)
        for j in 1:d
            Id[i][j,j] = 1.0
        end
    end

    return Id
end

"""
extract_bspline_to_bernstein(knot_vector::KnotVector, p::Int)

Computes the extraction coefficients of the B-Spline basis functions defined 
by `knot_vector`.

`E[:, j, el]` contains the coefficients of the linear combination of reference bernstein polynomials determining the `j`-th basis function on element `el`.

# Arguments
- `knot_vector::KnotVector`: knot vector defining the B-Spline basis.

# Returns
- `E::Array{Float64, 3}`: a `(knot_vector.polynomial_degree+1, knot_vector.polynomial_degree+1, nel)` matrix with the extraction coefficients of the b-splines basis functions on every element.
"""
function extract_bspline_to_bernstein(knot_vector::KnotVector)
    # number of elements
    nel = size(knot_vector.patch_1d)

    # first extraction matrix
    E = create_identity(nel, knot_vector.polynomial_degree+1)
    # this is where knot-insertion coefficients are saved
    alphas = zeros(max(knot_vector.polynomial_degree - 1, 0))

    for el in 1:1:nel
        mult = knot_vector.multiplicity[el+1]

        if mult < knot_vector.polynomial_degree
            numer = knot_vector.patch_1d.breakpoints[el+1] - knot_vector.patch_1d.breakpoints[el]
            r = knot_vector.polynomial_degree - mult

            for j in knot_vector.polynomial_degree-1:-1:mult
                idx = el+1+floor(Int, j/mult)

                if idx > nel+1
                    alphas[j-mult+1] = numer / (knot_vector.patch_1d.breakpoints[end] - knot_vector.patch_1d.breakpoints[el])
                    continue
                end

                alphas[j-mult+1] = numer / (knot_vector.patch_1d.breakpoints[idx] - knot_vector.patch_1d.breakpoints[el])
            end

            for j in 1:r
                s = mult + j - 1
                for k in knot_vector.polynomial_degree:-1:s+1
                    alpha = alphas[k-s]
                    E[el][k+1, :] .=  (@view E[el][k+1, :]).* alpha  .+  (@view E[el][k, :]) .* (1.0-alpha)
                end

                save = r - j + 1
                if el < nel
                    E[el+1][save, save:save+j] .= (@view E[el][knot_vector.polynomial_degree+1, knot_vector.polynomial_degree-j+1:knot_vector.polynomial_degree+1])
                end
            end
        end
    end

    # get indices of supported basis functions on each element
    basis_indices = Vector{Vector{Int}}(undef, nel)
    basis_indices[1] = 1:knot_vector.polynomial_degree+1
    for el = 2:nel
        basis_indices[el] = basis_indices[el-1] .+ knot_vector.multiplicity[el]
    end

    return ExtractionOperator(E, basis_indices, nel, basis_indices[nel][knot_vector.polynomial_degree+1])
end

"""
extract_bspline_to_bspline(bsplines::NTuple{m,BSplineSpace}, regularity::Vector{Int}) where {m}

Computes the extraction coefficients of GTB-Spline basis functions in terms of constitutent B-spline basis functions.

# Arguments
- `bsplines::NTuple{m,BSplineSpace}`: collection of (generalized) bspline spaces
- `regularity::Vector{Int}`: smoothness to be imposed at patch interfaces

# Returns
- `::ExtractionOperator`

"""
function extract_gtbspline_to_bspline(bsplines::NTuple{m,BSplineSpace}, regularity::Vector{Int}) where {m}
    # construct cumulative sum of all bspline dims
    bspl_dims = zeros(Int, m+1)
    for i = 2:m+1
        bspl_dims[i] = bspl_dims[i-1] + get_dim(bsplines[i-1])
    end

    # number of elements for all bsplines
    bspl_nels = [get_num_elements(bsplines[i]) for i = 1:m]
    nel = sum(bspl_nels)

    # initialize global extraction matrix
    H = SparseArrays.sparse(1:bspl_dims[m+1], 1:bspl_dims[m+1], ones(Float64,bspl_dims[m+1]), bspl_dims[m+1], bspl_dims[m+1])
    # loop over all internal patch interfaces and update extraction by imposing smoothness
    for i = 1:m-1
        # regularity at this interface
        r = regularity[i]
        # smoothness constraint matrix
        KL = SparseArrays.findnz(evaluate_all_at_point(bsplines[i], bspl_nels[i], 1.0, r))
        KR = SparseArrays.findnz(evaluate_all_at_point(bsplines[i+1], 1, 0.0, r))
        rows = [KL[1]; KR[1] .+ (bspl_dims[i+1] - bspl_dims[i])]
        cols = [KL[2]; KR[2]]
        vals = [-KL[3]; KR[3]]
        K = SparseArrays.sparse(rows,cols,vals,(bspl_dims[i+2] - bspl_dims[i]),r+1)
        # update local extraction matrix by building double-diagonal nullspace of constraints
        L = H[:, bspl_dims[i]+1:bspl_dims[i+2]] * K
        for j = 0:r
            Hbar = build_sparse_nullspace(L[:, j+1])
            H = Hbar * H
            L = Hbar * L
        end
    end

    # impose periodicity if desired for i = m
    if regularity[m] > -1
        r = regularity[m]
        if size(H, 1) >= 2*(r+1)
            Hper = circshift(H, r+1)
            KL = SparseArrays.findnz(evaluate_all_at_point(bsplines[m], bspl_nels[m], 1.0, r))
            KR = SparseArrays.findnz(evaluate_all_at_point(bsplines[1], 1, 0.0, r))
            rows = [KL[1]; KR[1] .+ (bspl_dims[m+1] - bspl_dims[m])]
            cols = [KL[2]; KR[2]]
            vals = [-KL[3]; KR[3]]
            K = SparseArrays.sparse(rows,cols,vals,(bspl_dims[m+1] - bspl_dims[m] + bspl_dims[2] - bspl_dims[1]),r+1)
            Lper = Hper[:, [bspl_dims[m]+1:bspl_dims[m+1]; bspl_dims[1]+1:bspl_dims[2]]] * K
            for j = 0:r
                Hbar = build_sparse_nullspace(Lper[:, j+1])
                Hper = Hbar * Hper
                Lper = Hbar * Lper
            end
            H = Hper
        end
    end

    # remove small values obtained as a result of round-off errors
    SparseArrays.fkeep!(H, (i,j,x) -> abs(x) > 1e-14)

    # convert global extraction matrix to element local extractions
    # (here, the matrix is transposed so that [Bsplines] * [extraction] = [GTB-splines])
    extraction_coefficients = Vector{Array{Float64}}(undef, nel)
    basis_indices = Vector{Vector{Int}}(undef, nel)
    count = 0
    for i = 1:m
        for j = 1:bspl_nels[i]
            _, cols_ij = get_extraction(bsplines[i], j)
            cols_ij .+= bspl_dims[i]
            eij = SparseArrays.findnz(H[:,cols_ij])
            # unique indices for non-zero rows and columns
            basis_indices[count+1] = unique(eij[1])
            # matrix of coefficients
            extraction_coefficients[count+1] = Array(H[basis_indices[count+1], cols_ij])'
            count += 1
        end
    end

    return ExtractionOperator(extraction_coefficients, basis_indices, nel, size(H,1))
end

"""
build_sparse_nullspace(constraint::Vector{Float64})

Builds sparsest possible nullspace of a constraint vector with no zero entries.

# Arguments

# Returns
"""
function build_sparse_nullspace(constraint::SparseArrays.SparseVector{Float64})
    q = length(constraint)
    nz_flag = .!isapprox.(constraint, 0.0, atol=1e-13)
    i1 = findfirst(nz_flag)
    i2 = findlast(nz_flag)
    dd = zeros(Float64, q-1, 2)
    dd[1:i1, 1] .= 1.0
    for j = i1:i2-2
        dd[j, 2] = -constraint[j] / constraint[j+1] * dd[j, 1]
        dd[j+1, 1] = 1 - dd[j, 2]
    end
    dd[i2-1:q-1, 2] .= 1.0

    return SparseArrays.spdiagm(q-1, q, 0 => dd[:,1], 1 => dd[:,2])
end

"""
extract_gtbspline_to_canonical(bsplines::NTuple{m,CanonicalFiniteElementSpace{1}}, regularity::Vector{Int}) where {m}

Computes the extraction coefficients of GTB-Spline basis functions in terms of constitutent canonical basis functions.

# Arguments
- `canonical_spaces::NTuple{m,CanonicalFiniteElementSpace{1}}`: collection of canonical spaces treated as finite element spaces
- `regularity::Vector{Int}`: smoothness to be imposed at patch interfaces

# Returns
- `::ExtractionOperator`

"""
function extract_gtbspline_to_canonical(canonical_spaces::NTuple{m,CanonicalFiniteElementSpace}, regularity::Vector{Int}) where {m}
    # construct cumulative sum of all bspline dims
    canonical_dims = zeros(Int, m+1)
    for i = 2:m+1
        canonical_dims[i] = canonical_dims[i-1] + get_dim(canonical_spaces[i-1])
    end

    # initialize global extraction matrix
    H = SparseArrays.sparse(1:canonical_dims[m+1], 1:canonical_dims[m+1], ones(Float64,canonical_dims[m+1]), canonical_dims[m+1], canonical_dims[m+1])
    # loop over all internal patch interfaces and update extraction by imposing smoothness
    for i = 1:m-1
        # regularity at this interface
        r = regularity[i]
        # smoothness constraint matrix
        KL = SparseArrays.findnz(evaluate_all_at_point(canonical_spaces[i], 1.0, r))
        KR = SparseArrays.findnz(evaluate_all_at_point(canonical_spaces[i+1], 0.0, r))
        rows = [KL[1]; KR[1] .+ (canonical_dims[i+1] - canonical_dims[i])]
        cols = [KL[2]; KR[2]]
        vals = [-KL[3]; KR[3]]
        K = SparseArrays.sparse(rows,cols,vals,(canonical_dims[i+2] - canonical_dims[i]),r+1)
        # update local extraction matrix by building double-diagonal nullspace of constraints
        L = H[:, canonical_dims[i]+1:canonical_dims[i+2]] * K
        for j = 0:r
            Hbar = build_sparse_nullspace(L[:, j+1])
            H = Hbar * H
            L = Hbar * L
        end
    end

    # impose periodicity if desired for i = m
    if regularity[m] > -1
        r = regularity[m]
        if size(H, 1) >= 2*(r+1)
            Hper = circshift(H, r+1)
            KL = SparseArrays.findnz(evaluate_all_at_point(canonical_spaces[m], 1.0, r))
            KR = SparseArrays.findnz(evaluate_all_at_point(canonical_spaces[1], 0.0, r))
            rows = [KL[1]; KR[1] .+ (canonical_dims[m+1] - canonical_dims[m])]
            cols = [KL[2]; KR[2]]
            vals = [-KL[3]; KR[3]]
            K = SparseArrays.sparse(rows,cols,vals,(canonical_dims[m+1] - canonical_dims[m] + canonical_dims[2] - canonical_dims[1]),r+1)
            Lper = Hper[:, [canonical_dims[m]+1:canonical_dims[m+1]; canonical_dims[1]+1:canonical_dims[2]]] * K
            for j = 0:r
                Hbar = build_sparse_nullspace(Lper[:, j+1])
                Hper = Hbar * Hper
                Lper = Hbar * Lper
            end
            H = Hper
        end
    end

    # remove small values obtained as a result of round-off errors
    SparseArrays.fkeep!(H, (i,j,x) -> abs(x) > 1e-14)

    # convert global extraction matrix to element local extractions
    # (here, the matrix is transposed so that [canonical_spaces] * [extraction] = [GTB-splines])
    extraction_coefficients = Vector{Array{Float64}}(undef, m)
    basis_indices = Vector{Vector{Int}}(undef, m)
    for i = 1:m
        cols_i = collect(1:get_dim(canonical_spaces[i]))
        cols_i .+= canonical_dims[i]
        ei = SparseArrays.findnz(H[:,cols_i])
        # unique indices for non-zero rows and columns
        basis_indices[i] = unique(ei[1])
        # matrix of coefficients
        extraction_coefficients[i] = Array(H[basis_indices[i], cols_i])'
    end

    return ExtractionOperator(extraction_coefficients, basis_indices, m, size(H,1))
end
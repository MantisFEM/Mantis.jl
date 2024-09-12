"""
    create_identity(ni::Int, d::Int) -> Vector{Matrix{Float64}}

Create a vector of `ni` identity matrices, each of size `d × d`.

# Arguments
- `ni::Int`: Number of identity matrices to create.
- `d::Int`: Size of each identity matrix.

# Returns
- `Vector{Matrix{Float64}}`: A vector containing `ni` identity matrices, each of size `d × d`.

# Examples
julia> create_identity(2, 3)
2-element Vector{Matrix{Float64}}:
 [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
 [1.0 0.0 0.0; 0.0 1.0 0.0; 0.0 0.0 1.0]
"""
function create_identity(ni::Int, d::Int)
    # Initialize a vector to store the identity matrices
    Id = Vector{Matrix{Float64}}(undef, ni)
    
    # Create 'ni' identity matrices
    for i in 1:ni
        # Initialize a d×d zero matrix
        Id[i] = zeros(d, d)
        
        # Set diagonal elements to 1.0 to create an identity matrix
        for j in 1:d
            Id[i][j,j] = 1.0
        end
    end

    return Id
end

"""
    extract_bspline_to_bernstein(knot_vector::KnotVector) -> ExtractionOperator

Compute the extraction coefficients of B-Spline basis functions defined by the given knot vector.

This function implements the Bézier extraction process, converting B-Spline basis functions
to Bernstein polynomials on each element of the knot vector.

# Arguments
- `knot_vector::KnotVector`: The knot vector defining the B-Spline basis.

# Returns
- `ExtractionOperator`: A struct containing:
  - `E::Vector{Matrix{Float64}}`: Extraction coefficients for each element.
  - `basis_indices::Vector{Vector{Int}}`: Indices of supported basis functions on each element.
  - `nel::Int`: Number of elements.
  - `num_basis_functions::Int`: Total number of basis functions.

# Note
The extraction coefficients `E[el]` for element `el` contain the coefficients of the linear 
combination of reference Bernstein polynomials determining the basis functions on that element.

# References
- Borden, M. J., Scott, M. A., Evans, J. A., & Hughes, T. J. R. (2011). 
  Isogeometric finite element data structures based on Bézier extraction of NURBS. 
  International Journal for Numerical Methods in Engineering, 87(1-5), 15-47.
"""
function extract_bspline_to_bernstein(knot_vector::KnotVector)
    # Number of elements in the knot vector
    nel = size(knot_vector.patch_1d)

    # Initialize extraction matrices for each element
    E = create_identity(nel, knot_vector.polynomial_degree + 1)
    
    # Array to store knot insertion coefficients
    alphas = zeros(max(knot_vector.polynomial_degree - 1, 0))

    # Iterate over all elements
    for el in 1:nel
        # Get multiplicity of the knot at the end of the current element
        mult = knot_vector.multiplicity[el+1]

        # If multiplicity is less than polynomial degree, perform knot insertion
        if mult < knot_vector.polynomial_degree
            # Calculate numerator for alpha coefficients
            numer = knot_vector.patch_1d.breakpoints[el+1] - knot_vector.patch_1d.breakpoints[el]
            r = knot_vector.polynomial_degree - mult

            # Compute alpha coefficients
            for j in knot_vector.polynomial_degree-1:-1:mult
                idx = el+1+floor(Int, j/mult)

                if idx > nel+1
                    alphas[j-mult+1] = numer / (knot_vector.patch_1d.breakpoints[end] - knot_vector.patch_1d.breakpoints[el])
                else
                    alphas[j-mult+1] = numer / (knot_vector.patch_1d.breakpoints[idx] - knot_vector.patch_1d.breakpoints[el])
                end
            end

            # Update extraction coefficients
            for j in 1:r
                s = mult + j - 1
                for k in knot_vector.polynomial_degree:-1:s+1
                    alpha = alphas[k-s]
                    E[el][k+1, :] .=  (@view E[el][k+1, :]) .* alpha  .+  (@view E[el][k, :]) .* (1.0-alpha)
                end

                # Save coefficients for the next element
                save = r - j + 1
                if el < nel
                    E[el+1][save, save:save+j] .= (@view E[el][knot_vector.polynomial_degree+1, knot_vector.polynomial_degree-j+1:knot_vector.polynomial_degree+1])
                end
            end
        end
    end

    # Compute indices of supported basis functions on each element
    basis_indices = Vector{Vector{Int}}(undef, nel)
    basis_indices[1] = 1:knot_vector.polynomial_degree+1
    for el = 2:nel
        basis_indices[el] = basis_indices[el-1] .+ knot_vector.multiplicity[el]
    end

    # Create and return the ExtractionOperator struct
    return ExtractionOperator(E, basis_indices, nel, basis_indices[nel][knot_vector.polynomial_degree+1])
end

"""
    extract_gtbspline_to_nurbs(nurbs::NTuple{m,F}, regularity::Vector{Int}) where {m, F <: Union{BSplineSpace, RationalFiniteElementSpace}}

Compute the extraction coefficients of GTB-Spline basis functions in terms of NURBS basis functions.

# Arguments
- `nurbs::NTuple{m,F}`: Collection of NURBS spaces.
- `regularity::Vector{Int}`: Smoothness to be imposed at patch interfaces.

# Returns
- `ExtractionOperator`: The extraction operator containing the coefficients.
"""
function extract_gtbspline_to_nurbs(nurbs::NTuple{m,F}, regularity::Vector{Int}) where {m, F <: Union{BSplineSpace, RationalFiniteElementSpace}}
    # Construct cumulative sum of all B-spline dimensions
    bspl_dims = zeros(Int, m+1)
    for i in 2:m+1
        bspl_dims[i] = bspl_dims[i-1] + get_num_basis(nurbs[i-1])
    end

    # Number of elements for all NURBS
    bspl_nels = [get_num_elements(nurbs[i]) for i = 1:m]
    nel = sum(bspl_nels)

    # Initialize global extraction matrix
    H = SparseArrays.sparse(1:bspl_dims[m+1], 1:bspl_dims[m+1], ones(Float64, bspl_dims[m+1]), bspl_dims[m+1], bspl_dims[m+1])
    
    # Loop over all internal patch interfaces and update extraction by imposing smoothness
    for i = 1:m-1
        # Regularity at this interface
        r = regularity[i]
        
        # Smoothness constraint matrix
        KL = SparseArrays.findnz(_evaluate_all_at_point(nurbs[i], bspl_nels[i], 1.0, r))
        KR = SparseArrays.findnz(_evaluate_all_at_point(nurbs[i+1], 1, 0.0, r))
        rows = [KL[1]; KR[1] .+ (bspl_dims[i+1] - bspl_dims[i])]
        cols = [KL[2]; KR[2]]
        vals = [-KL[3]; KR[3]]
        K = SparseArrays.sparse(rows, cols, vals, (bspl_dims[i+2] - bspl_dims[i]), r+1)
        
        # Update local extraction matrix by building double-diagonal nullspace of constraints
        L = H[:, bspl_dims[i]+1:bspl_dims[i+2]] * K
        for j = 0:r
            Hbar = build_sparse_nullspace(L[:, j+1])
            H = Hbar * H
            L = Hbar * L
        end
    end

    # Impose periodicity if desired for i = m
    if regularity[m] > -1
        r = regularity[m]
        if size(H, 1) >= 2*(r+1)
            Hper = circshift(H, r+1)
            KL = SparseArrays.findnz(_evaluate_all_at_point(nurbs[m], bspl_nels[m], 1.0, r))
            KR = SparseArrays.findnz(_evaluate_all_at_point(nurbs[1], 1, 0.0, r))
            rows = [KL[1]; KR[1] .+ (bspl_dims[m+1] - bspl_dims[m])]
            cols = [KL[2]; KR[2]]
            vals = [-KL[3]; KR[3]]
            K = SparseArrays.sparse(rows, cols, vals, (bspl_dims[m+1] - bspl_dims[m] + bspl_dims[2] - bspl_dims[1]), r+1)
            Lper = Hper[:, [bspl_dims[m]+1:bspl_dims[m+1]; bspl_dims[1]+1:bspl_dims[2]]] * K
            for j = 0:r
                Hbar = build_sparse_nullspace(Lper[:, j+1])
                Hper = Hbar * Hper
                Lper = Hbar * Lper
            end
            H = Hper
        end
    end

    # Remove small values obtained as a result of round-off errors
    SparseArrays.fkeep!((i, j, x) -> abs(x) > 1e-14, H)

    # Convert global extraction matrix to element local extractions
    # (here, the matrix is transposed so that [nurbs] * [extraction] = [GTB-splines])
    extraction_coefficients = Vector{Array{Float64}}(undef, nel)
    basis_indices = Vector{Vector{Int}}(undef, nel)
    count = 0
    for i = 1:m
        for j = 1:bspl_nels[i]
            _, cols_ij = get_extraction(nurbs[i], j)
            cols_ij .+= bspl_dims[i]
            eij = SparseArrays.findnz(H[:, cols_ij])
            # Unique indices for non-zero rows and columns
            basis_indices[count+1] = unique(eij[1])
            # Matrix of coefficients
            extraction_coefficients[count+1] = Array(H[basis_indices[count+1], cols_ij])'
            count += 1
        end
    end

    return ExtractionOperator(extraction_coefficients, basis_indices, nel, size(H, 1))
end

"""
    build_sparse_nullspace(constraint::SparseArrays.SparseVector{Float64}) -> SparseMatrixCSC{Float64}

Build the sparsest possible nullspace of a constraint vector with no zero entries.

# Arguments
- `constraint::SparseArrays.SparseVector{Float64}`: The constraint vector.

# Returns
- `SparseMatrixCSC{Float64}`: The sparse nullspace matrix.
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
    extract_gtbspline_to_canonical(canonical_spaces::NTuple{m,CanonicalFiniteElementSpace}, regularity::Vector{Int}) where {m}

Compute the extraction coefficients of GTB-Spline basis functions in terms of constituent canonical basis functions.

# Arguments
- `canonical_spaces::NTuple{m,CanonicalFiniteElementSpace}`: Collection of canonical spaces treated as finite element spaces.
- `regularity::Vector{Int}`: Smoothness to be imposed at patch interfaces.

# Returns
- `ExtractionOperator`: An object containing extraction coefficients, basis indices, number of elements, and total number of basis functions.
"""
function extract_gtbspline_to_canonical(canonical_spaces::NTuple{m,CanonicalFiniteElementSpace}, regularity::Vector{Int}) where {m}
    # Construct cumulative sum of all canonical space dimensions
    canonical_dims = zeros(Int, m+1)
    for i = 2:m+1
        canonical_dims[i] = canonical_dims[i-1] + get_num_basis(canonical_spaces[i-1])
    end

    # Initialize global extraction matrix as identity
    H = SparseArrays.sparse(1:canonical_dims[m+1], 1:canonical_dims[m+1], ones(Float64,canonical_dims[m+1]), canonical_dims[m+1], canonical_dims[m+1])
    
    # Loop over all internal patch interfaces and update extraction by imposing smoothness
    for i = 1:m-1
        # Get regularity at this interface
        r = regularity[i]
        
        # Construct smoothness constraint matrix
        KL = SparseArrays.findnz(_evaluate_all_at_point(canonical_spaces[i], 1, 1.0, r))
        KR = SparseArrays.findnz(_evaluate_all_at_point(canonical_spaces[i+1], 1, 0.0, r))
        rows = [KL[1]; KR[1] .+ (canonical_dims[i+1] - canonical_dims[i])]
        cols = [KL[2]; KR[2]]
        vals = [-KL[3]; KR[3]]
        K = SparseArrays.sparse(rows,cols,vals,(canonical_dims[i+2] - canonical_dims[i]),r+1)
        
        # Update local extraction matrix by building double-diagonal nullspace of constraints
        L = H[:, canonical_dims[i]+1:canonical_dims[i+2]] * K
        for j = 0:r
            Hbar = build_sparse_nullspace(L[:, j+1])
            H = Hbar * H
            L = Hbar * L
        end
    end

    # Impose periodicity if desired for i = m
    if regularity[m] > -1
        r = regularity[m]
        if size(H, 1) >= 2*(r+1)
            Hper = circshift(H, r+1)
            KL = SparseArrays.findnz(_evaluate_all_at_point(canonical_spaces[m], 1, 1.0, r))
            KR = SparseArrays.findnz(_evaluate_all_at_point(canonical_spaces[1], 1, 0.0, r))
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

    # Remove small values obtained as a result of round-off errors
    SparseArrays.fkeep!((i,j,x) -> abs(x) > 1e-14, H)

    # Convert global extraction matrix to element local extractions
    # (here, the matrix is transposed so that [canonical_spaces] * [extraction] = [GTB-splines])
    extraction_coefficients = Vector{Array{Float64}}(undef, m)
    basis_indices = Vector{Vector{Int}}(undef, m)
    for i = 1:m
        cols_i = collect(1:get_num_basis(canonical_spaces[i]))
        cols_i .+= canonical_dims[i]
        ei = SparseArrays.findnz(H[:,cols_i])
        # Unique indices for non-zero rows and columns
        basis_indices[i] = unique(ei[1])
        # Matrix of coefficients
        extraction_coefficients[i] = Array(H[basis_indices[i], cols_i])'
    end

    # Create and return the ExtractionOperator
    return ExtractionOperator(extraction_coefficients, basis_indices, m, size(H,1))
end
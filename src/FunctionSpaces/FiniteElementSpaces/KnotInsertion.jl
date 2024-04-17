"""
Functions and algorithms used for knot insertion.

"""

# TwoScaleOperator structure

"""
    struct TwoScaleOperator

Two-scale operator for a change of basis between two finite element spaces.

# Fields
- `coarse_space::AbstractFiniteElementSpace`: Coarse finite element space.
- `fine_space::AbstractFiniteElementSpace`: Fine finite element space.
- `global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int64}`: Change of basis matrix.
- `coarse_to_fine::Vector{Vector{Int}}`: Relation between coarser and finer basis functions.
"""
struct TwoScaleOperator
    coarse_space::AbstractFiniteElementSpace
    fine_space::AbstractFiniteElementSpace
    global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int64}
    coarse_to_fine::Vector{Vector{Int}}

    function TwoScaleOperator(coarse_space::S, fine_space::T, global_subdiv_matrix::SparseArrays.SparseMatrixCSC{Float64, Int64}) where {S<:AbstractFiniteElementSpace, T<:AbstractFiniteElementSpace}
        ncoarse = size(global_subdiv_matrix)[2]
        coarse_to_fine = Vector{Vector{Int}}(undef, ncoarse)

        for i in 1:1:ncoarse
            coarse_to_fine[i] = global_subdiv_matrix.rowval[SparseArrays.nzrange(global_subdiv_matrix, i)]
        end

        new(coarse_space, fine_space, global_subdiv_matrix, coarse_to_fine)
    end
end

# Helper functions for knot insertion

"""
    subdivide_breakpoints(coarse_breakpoints::Vector{Float64}, nsubdivisions::Int)

Subdivides `coarse_breakpoints` by uniformly subdiving each element 'nsubdivisions' times.

# Arguments
- `coarse_breakpoints::Vector{Float64}`: Coarse set of breakpoints.
- `nsubdivisions::Int`: Number of times each element is subdivided.
# Returns 
- `fine_breakpoints::Vector{Float64}`: Fine set of breakpoints.
"""
function subdivide_breakpoints(coarse_breakpoints::Vector{Float64}, nsubdivisions::Int)
    num_points = length(coarse_breakpoints) + (length(coarse_breakpoints) - 1)* (nsubdivisions - 1)

    fine_breakpoints = Vector{Float64}(undef, num_points)

    step_size = 1 / nsubdivisions

    for i in 1:length(coarse_breakpoints)-1, j in 0:nsubdivisions-1
        index = (i-1) * nsubdivisions + j + 1
        fine_breakpoints[index] = coarse_breakpoints[i] + j * step_size * (coarse_breakpoints[i+1] - coarse_breakpoints[i])
    end

    fine_breakpoints[end] = coarse_breakpoints[end]

    return fine_breakpoints
end

"""
    subdivide_breakpoints(coarse_breakpoints::NTuple{n, Vector{Float64}}, nsubdivisions::NTuple{n, Int}) where {n}

Subdivides `coarse_breakpoints` by uniformly subdiving each element 'nsubdivisions' times,
across each dimension.

# Arguments
- `coarse_breakpoints::NTuple{n, Vector{Float64}}`: Coarse set of breakpoints.
- `nsubdivisions::NTuple{n, Int}`: Number of times each element is subdivided.
# Returns 
- `::NTuple{n, Vector{Float64}}`: Fine set of breakpoints.
"""
function subdivide_breakpoints(coarse_breakpoints::NTuple{n, Vector{Float64}}, nsubdivisions::NTuple{n, Int}) where {n}
    return ntuple(d -> subdivide_breakpoints(coarse_breakpoints[d], nsubdivisions[d]), n)
end

"""
    subdivide_patch(coarse_patch::Mesh.Patch1D, nsubdivisions::Int)

Subdivides `coarse_patch` by uniformly subdiving each element 'nsubdivisions' times.

# Arguments
- `coarse_patch::Mesh.Patch1D`: Coarse patch.
- `nsubdivisions::Int`: Number of times each element is subdivided.
# Returns 
- `fine_patch::Mesh.Patch1D`: Fine patch.
"""
function subdivide_patch(coarse_patch::Mesh.Patch1D, nsubdivisions::Int)
    fine_breakpoints = subdivide_breakpoints(Mesh.get_breakpoints(coarse_patch), nsubdivisions)
    return Mesh.Patch1D(fine_breakpoints)
end

"""
    subdivide_patch(coarse_patch::Mesh.Patch, nsubdivisions::NTuple{n, Int}) where {n}

Subdivides `coarse_patch` by uniformly subdiving each element 'nsubdivisions' times,
across each dimension.

# Arguments
- `coarse_patch::Mesh.Patch`: Coarse patch.
- `nsubdivisions::NTuple{n, Int}`: Number of times each element is subdivided.
# Returns 
- `::Mesh.Patch`: Fine patch.
"""
function subdivide_patch(coarse_patch::Mesh.Patch, nsubdivisions::NTuple{n, Int}) where {n}
    fine_breakpoints = subdivide_breakpoints(Mesh.get_breakpoints(coarse_patch), nsubdivisions)
    return Mesh.Patch(fine_breakpoints)
end

"""
    subdivide_multiplicity_vector(coarse_multiplicity::Vector{Int}, nsubdivisions::Int, fine_multiplicity::Int)

Subdivides `coarse_multiplicity` by uniformly subdiving each element 'nsubdivisions' times.
The coarse multiplicities are preserved in the `fine_multiplicity_vector`, and newly inserted ones 
are given multiplicity `fine_multiplicity`.

# Arguments
- `coarse_multiplicity::Vector{Int}`: Coarse multiplicity vector.
- `nsubdivisions::Int`: Number of times each element is subdivided.
- `fine_multiplicity::Int`: Multiplicity of each new knot.
# Returns 
- `fine_multiplicity_vector::Vector{Int}`: Fine multiplicity vector.
"""
function subdivide_multiplicity_vector(coarse_multiplicity::Vector{Int}, nsubdivisions::Int, fine_multiplicity::Int)
    mult_length = 1 + length(coarse_multiplicity)*(nsubdivisions) - nsubdivisions 
    
    fine_multiplicity_vector = fill(fine_multiplicity, mult_length)
    
    coarse_idx = 1
    for k in eachindex(fine_multiplicity_vector)
        if (k-1)%nsubdivisions + 1 == 1 
            if fine_multiplicity_vector[k] < coarse_multiplicity[coarse_idx]
                fine_multiplicity_vector[k] = coarse_multiplicity[coarse_idx]
            end
            coarse_idx += 1
        end
    end

    return fine_multiplicity_vector
end

"""
    subdivide_multiplicity_vector(coarse_multiplicity::Vector{Int}, nsubdivisions::Int)

Subdivides `coarse_multiplicity` by uniformly subdiving each element 'nsubdivisions' times.
The coarse multiplicities are preserved in the final multiplicity vector, and newly inserted ones are given multiplicity 1.

# Arguments
- `coarse_multiplicity::Vector{Int}`: Coarse multiplicity vector.
- `nsubdivisions::Int`: Number of times each element is subdivided.
# Returns 
- `::Vector{Int}`: Fine multiplicity vector.
"""
function subdivide_multiplicity_vector(coarse_multiplicity::Vector{Int}, nsubdivisions::Int)
    return subdivide_multiplicity_vector(coarse_multiplicity, nsubdivisions, 1)
end

"""
    subdivide_multiplicity_vector(coarse_multiplicity::NTuple{n, Vector{Int}}, nsubdivisions::NTuple{n, Int}) where {n}

Subdivides `coarse_multiplicity` by uniformly subdiving each element 'nsubdivisions' times,
across each dimension.
The coarse multiplicities are preserved in the final multiplicity vector, and newly inserted ones are given multiplicity 1.

# Arguments
- `coarse_multiplicity::NTuple{n, Vector{Int}}`: Coarse multiplicity vector.
- `nsubdivisions::NTuple{n, Int}`: Number of times each element is subdivided.
# Returns 
- `::NTuple{n, Vector{Int}}`: Fine multiplicity vectors.
"""
function subdivide_multiplicity_vector(coarse_multiplicity::NTuple{n, Vector{Int}}, nsubdivisions::NTuple{n, Int}) where {n}
    return ntuple(d -> subdivide_multiplicity_vector(coarse_multiplicity[d], nsubdivisions[d]), n)
end

"""
    subdivide_multiplicity_vector(coarse_multiplicity::NTuple{n, Vector{Int}}, nsubdivisions::NTuple{n, Int}, fine_multiplicity::NTuple{n, Int}) where {n}

Subdivides `coarse_multiplicity` by uniformly subdiving each element 'nsubdivisions' times,
across each dimension.
The coarse multiplicities are preserved in the final multiplicity vector, and newly inserted ones are given multiplicity `fine_multiplicity`.

# Arguments
- `coarse_multiplicity::NTuple{n, Vector{Int}}`: Coarse multiplicity vector.
- `nsubdivisions::NTuple{n, Int}`: Number of times each element is subdivided.
- `fine_multiplicity::NTuple{n, Vector{Int}}`: Multiplicity of each new knot.
# Returns 
- `::NTuple{n, Vector{Int}}`: Fine multiplicity vectors.
"""
function subdivide_multiplicity_vector(coarse_multiplicity::NTuple{n, Vector{Int}}, nsubdivisions::NTuple{n, Int}, fine_multiplicity::NTuple{n, Int}) where {n}
    return ntuple(d -> subdivide_multiplicity_vector(coarse_multiplicity[d], nsubdivisions[d], fine_multiplicity[d]), n)
end

"""
    subdivide_knot_vector(coarse_knot_vector::KnotVector, nsubdivisions::Int)

Subdivides `coarse_knot_vector` by uniformly subdiving each element 'nsubdivisions' times.
The coarse multiplicities are preserved in the `fine_multiplicity`, and newly inserted ones are given multiplicity 1.

# Arguments
- `coarse_knot_vector::KnotVector`: Coarse knot vector.
- `nsubdivisions::Int`: Number of times each element is subdivided.
# Returns 
- `::KnotVector`: Fine knot vector.
"""
function subdivide_knot_vector(coarse_knot_vector::KnotVector, nsubdivisions::Int)
    fine_patch = subdivide_patch(coarse_knot_vector.patch_1d, nsubdivisions)
    fine_multiplicity = subdivide_multiplicity_vector(coarse_knot_vector.multiplicity, nsubdivisions)

    return KnotVector(fine_patch, coarse_knot_vector.polynomial_degree, fine_multiplicity)
end

"""
    subdivide_knot_vector(coarse_knot_vector::KnotVector, nsubdivisions::Int, fine_multiplicity::Int)

Subdivides `coarse_knot_vector` by uniformly subdiving each element 'nsubdivisions' times.
The coarse multiplicities are preserved in the `fine_multiplicity_vector`, and newly inserted ones are given multiplicity `fine_multiplicity`.

# Arguments
- `coarse_knot_vector::KnotVector`: Coarse knot vector.
- `nsubdivisions::Int`: Number of times each element is subdivided.
- `fine_multiplicity::Int`: Multiplicity of each new knot.
# Returns 
- `::KnotVector`: Fine knot vector.
"""
function subdivide_knot_vector(coarse_knot_vector::KnotVector, nsubdivisions::Int, fine_multiplicity::Int)
    fine_patch = subdivide_patch(coarse_knot_vector.patch_1d, nsubdivisions)
    fine_multiplicity_vector = subdivide_multiplicity_vector(coarse_knot_vector.multiplicity, nsubdivisions, fine_multiplicity)

    return KnotVector(fine_patch, coarse_knot_vector.polynomial_degree, fine_multiplicity_vector)
end

"""
    subdivide_bspline(coarse_bspline::BSplineSpace, nsubdivisions::Int, fine_multiplicity::Int)

Subdivides `coarse_bspline` by uniformly subdiving each element 'nsubdivisions' times. The coarse multiplicities
are preserved in the final multiplicity vector, and newly inserted ones are given multiplicity `fine_multiplicity`.

# Arguments
- `coarse_bspline::BSplineSpace`: Coarse B-spline.
- `nsubdivisions::Int`: Number of times each element is subdivided.
- `fine_multiplicity::Int`: Multiplicity of each new knot.
# Returns 
- `::TwoScaleOperator`: Two-scale operator for a change of B-spline basis.
"""
function subdivide_bspline(coarse_bspline::BSplineSpace, nsubdivisions::Int, fine_multiplicity::Int)
    fine_knot_vector = subdivide_knot_vector(coarse_bspline.knot_vector, nsubdivisions, fine_multiplicity)
    p = coarse_bspline.knot_vector.polynomial_degree
    fine_bspline = BSplineSpace(fine_knot_vector.patch_1d, p, p .- fine_knot_vector.multiplicity)

    return build_two_scale_operator(coarse_bspline, fine_bspline)
end

"""
    subdivide_bspline(coarse_bspline::BSplineSpace, nsubdivisions::Int)

Subdivides `coarse_bspline` by uniformly subdiving each element 'nsubdivisions' times. The coarse multiplicities
are preserved in the `fine_multiplicity`, and newly inserted ones are given multiplicity 1.

# Arguments
- `coarse_bspline::BSplineSpace`: Coarse B-spline.
- `nsubdivisions::Int`: Number of times each element is subdivided.
# Returns 
- `::TwoScaleOperator`: Two-scale operator for a change of B-spline basis.
"""
function subdivide_bspline(coarse_bspline::BSplineSpace, nsubdivisions::Int)
    return subdivide_bspline(coarse_bspline, nsubdivisions, 1)
end

"""
    subdivide_bspline(coarse_bspline::NTuple{n, BSplineSpace}, nsubdivisions::NTuple{n, Int}) where {n}

Subdivides `coarse_bspline` by uniformly subdiving each element 'nsubdivisions' times,
across each dimentions. 
The coarse multiplicities are preserved in the `fine_multiplicity`, and newly inserted ones 
are given multiplicity 1.

# Arguments
- `coarse_bspline::NTuple{n, BSplineSpace}`: Coarse B-spline.
- `nsubdivisions::NTuple{n, Int}`: Number of times each element is subdivided.
# Returns 
- `::NTuple{n, BSplineSpace}`: Fine B-spline.
"""
function subdivide_bspline(coarse_bspline::NTuple{n, BSplineSpace}, nsubdivisions::NTuple{n, Int}) where {n}
    return ntuple(d -> subdivide_bspline(coarse_bspline[d], nsubdivisions[d]), n)
end

"""
    subdivide_bspline(coarse_bspline::NTuple{n, BSplineSpace}, nsubdivisions::NTuple{n, Int}, fine_multiplicity::NTuple{n, Vector{Int}}) where {n}

Subdivides `coarse_bspline` by uniformly subdiving each element 'nsubdivisions' times,
across each dimentions. 
The coarse multiplicities are preserved in the final multiplicity vectors, and newly inserted ones 
are given multiplicity `fine_multiplicity`.

# Arguments
- `coarse_bspline::NTuple{n, BSplineSpace}`: Coarse B-spline.
- `nsubdivisions::NTuple{n, Int}`: Number of times each element is subdivided.
- `fine_multiplicity::NTuple{n, Vector{Int}}`: Multiplicity of each new knot.
# Returns 
- `::NTuple{n, BSplineSpace}`: Fine B-spline.
"""
function subdivide_bspline(coarse_bspline::NTuple{n, BSplineSpace}, nsubdivisions::NTuple{n, Int}, fine_multiplicity::NTuple{n, Vector{Int}}) where {n}
    return ntuple(d -> subdivide_bspline(coarse_bspline[d], nsubdivisions[d], fine_multiplicity[d]), n)
end

# Oslo knot insertion algorithms

"""
    single_knot_insertion_oslo(coarse_knot_vector::KnotVector, fine_knot_vector::KnotVector, cf::Int, rf::Int)

Algorithm for the coefficients of a change of B-spline representation for a single knot insertion.
The coarse knot vector is `coarse_knot_vector` and the inserted knot is given by `fine_knot_vector`.

For more information, see [A note on the Oslo Algorithm](https://collections.lib.utah.edu/dl_files/66/d4/66d493df0f5c97cce67e0bc1294363d64dde7f06.pdf).

# Arguments
- `coarse_knot_vector::KnotVector`: Coarse knot vector.
- `fine_knot_vector::KnotVector`: Fine knot vector, with the extra knot.
- `cf::Int`: Index of the coarse knot vector.
- `rf::Int`: Index of the fine knot vector such that `get_knot_breakpoint(coarse_knot_vector,cf) <= get_knot_breakpoint(fine_knot_vector,rf) < get_knot_breakpoint(coarse_knot_vector,cf+1)`.
# Returns 
- `b::Vector{Float64}`: Coefficients for the change of basis.
"""
function single_knot_insertion_oslo(coarse_knot_vector::KnotVector, fine_knot_vector::KnotVector, cf::Int, rf::Int)
    b = [1.0]

    for k in 1:coarse_knot_vector.polynomial_degree
        t1 = get_knot_breakpoint.((coarse_knot_vector,), cf+1-k:cf)
        t2 = get_knot_breakpoint.((coarse_knot_vector,), cf+1:cf+k)
        x = get_knot_breakpoint(fine_knot_vector, rf+k)

        w =  (x .- t1) ./ (t2 .- t1)

        b = push!((1 .- w) .* b, 0) .+ pushfirst!(w .* b, 0)
    end


    return b
end

"""
    build_two_scale_operator(coarse_knot_vector::KnotVector, fine_knot_vector::KnotVector)

Algorithm for the coefficients of a change of B-spline representation for knot insertion 
of multiple knots, recursively using `single_knot_insertion_oslo()`.
The coarse knot vector is `coarse_knot_vector` and the inserted knots are given by `fine_knot_vector`.

For more information, see [Paper](https://doi.org/10.1016/j.cma.2017.08.017).

# Arguments
- `coarse_knot_vector::KnotVector`: Coarse knot vector.
- `fine_knot_vector::KnotVector`: Fine knot vector, with the extra knots.
# Returns 
- `(::FiniteElementSpaces.TwoScaleOperator, fine_bspline::BSplineSpace`: Tuple with a two_scale_operator
and finer B-spline space.
"""
function build_two_scale_operator(coarse_knot_vector::KnotVector, fine_knot_vector::KnotVector)
    m = get_knot_length(fine_knot_vector)
    nel = size(fine_knot_vector.patch_1d)
    p = coarse_knot_vector.polynomial_degree
    nfine = m-p-1

    gm_values = zeros(Float64, nfine*(p+1))
    gm_rows = zeros(Int, nfine*(p+1))
    gm_columns = similar(gm_rows)
    sparse_idx = zeros(Int, p+1)

    cf = p + 1
    rf = 1
    e = 1

    local_subdiv_matrix = create_identity(nel, p+1)

    offs = 0

    while rf <= nfine
        mult = get_knot_multiplicity(fine_knot_vector, rf)

        lastcf = cf
        while get_knot_breakpoint(coarse_knot_vector, cf+1) <= get_knot_breakpoint(fine_knot_vector, rf)
            cf += 1
        end

        if e > 1 
            offs = cf - lastcf
            local_subdiv_matrix[e][1:p+1-offs, 1:p+1-mult] .= local_subdiv_matrix[e-1][1+offs:p+1, 1+mult:p+1] 
        end

        for t in p+2-mult:p+1
            sparse_idx .= (rf-1)*(p+1)+1:rf*(p+1)
            gm_columns[sparse_idx] .= cf-p:cf
            gm_rows[sparse_idx] .= rf

            local_subdiv_matrix[e][:, t] = single_knot_insertion_oslo(coarse_knot_vector, fine_knot_vector, cf, rf)
            gm_values[sparse_idx] .= local_subdiv_matrix[e][:, t]

            rf += 1
        end
        
        e += 1
    end

    global_extraction_matrix = SparseArrays.sparse(gm_rows, gm_columns, gm_values, rf-1, cf)
    
    return global_extraction_matrix
end

"""
    build_two_scale_operator(coarse_bspline::BSplineSpace, fine_bspline::BSplineSpace)

Algorithm for the coefficients of a change of B-spline representation for knot insertion 
of multiple knots, recursively using `single_knot_insertion_oslo()`.
The coarse knot vector is `coarse_bspline.knot_vector` and the inserted knots are given by `fine_bspline.knot_vector`.

For more information, see [Paper](https://doi.org/10.1016/j.cma.2017.08.017).

# Arguments
- `coarse_bspline::BSplineSpace`: Coarse B-spline.
- `fine_bspline::BSplineSpace`: Fine B-spline, with extra knots.
# Returns 
- `(::FiniteElementSpaces.TwoScaleOperator, fine_bspline::BSplineSpace`: Tuple with a two_scale_operator
and finer B-spline space.
"""
function build_two_scale_operator(coarse_bspline::BSplineSpace, fine_bspline::BSplineSpace)
    gm = build_two_scale_operator(coarse_bspline.knot_vector, fine_bspline.knot_vector)
    
    return TwoScaleOperator(coarse_bspline, fine_bspline, gm), fine_bspline
end

"""
    build_two_scale_operator(coarse_bspline::BSplineSpace, nsubdivisions::Int, fine_multiplicity::Int)

Algorithm for the coefficients of a change of B-spline representation for knot insertion 
of multiple knots, recursively using `single_knot_insertion_oslo()`.
The coarse knot vector is `coarse_bspline.knot_vector` and the inserted knots are given by `nsubdivisions`, meaning `nsubdivisions-1` uniformly spaced knots are inserted, between coarse breakpoints, with multiplicity 1.

For more information, see [Paper](https://doi.org/10.1016/j.cma.2017.08.017).

# Arguments
- `coarse_bspline::BSplineSpace`: Coarse B-spline.
- `nsubdivisions::Int`: Number of times each element is subdivided.
- `fine_multiplicity::Int`: Multiplicity of each new knot in refined knot vector.
# Returns 
- `(::FiniteElementSpaces.TwoScaleOperator, fine_bspline::BSplineSpace`: Tuple with a two_scale_operator
and finer B-spline space.
"""
function build_two_scale_operator(coarse_bspline::BSplineSpace, nsubdivisions::Int, fine_multiplicity::Int)
    nsubdivisions > 0 || throw(ArgumentError("Number of subdivions must be greater than 0. 
    nsubdivisions=$nsubdivisions was given."))

    fine_bspline = subdivide_bspline(coarse_bspline, nsubdivisions, fine_multiplicity)

    return build_two_scale_operator(coarse_bspline, fine_bspline)
end

"""
    build_two_scale_operator(coarse_bspline::BSplineSpace, nsubdivisions::Int)

Algorithm for the coefficients of a change of B-spline representation for knot insertion 
of multiple knots, recursively using `single_knot_insertion_oslo()`.
The coarse knot vector is `coarse_bspline.knot_vector` and the inserted knots are given by `nsubdivisions`, meaning `nsubdivisions-1` uniformly spaced knots are inserted, between coarse breakpoints, with multiplicity 1.

For more information, see [Paper](https://doi.org/10.1016/j.cma.2017.08.017).

# Arguments
- `coarse_bspline::BSplineSpace`: Coarse B-spline.
- `nsubdivisions::Int`: Number of times each element is subdivided.
# Returns 
- `(::FiniteElementSpaces.TwoScaleOperator, fine_bspline::BSplineSpace`: Tuple with a two_scale_operator
and finer B-spline space.
"""
function build_two_scale_operator(coarse_bspline::BSplineSpace, nsubdivisions::Int)
    return build_two_scale_operator(coarse_bspline, nsubdivisions, 1)
end

"""
    build_two_scale_operator(coarse_bspline::NTuple{n, BSplineSpace}, nsubdivisions::NTuple{n, Int}) where {n}

Algorithm for the coefficients of a change of B-spline representation for knot insertion 
of multiple knots, recursively using `single_knot_insertion_oslo()`.
The coarse knot vector is `coarse_bspline.knot_vector` and the inserted knots are given by `nsubdivisions`, meaning `nsubdivisions-1` uniformly spaced knots are inserted, between coarse breakpoints, with multiplicity 1,
across each dimension.

For more information, see [Paper](https://doi.org/10.1016/j.cma.2017.08.017).

# Arguments
- `coarse_bspline::NTuple{n, BSplineSpace}`: Coarse B-spline.
- `nsubdivisions::NTuple{n, Int}`: Number of times each element is subdivided.
# Returns 
- `::NTuple{n, Vector{Array{Float64}}}`: Two-scale relation operators and finer B-spline spaces.
"""
function build_two_scale_operator(coarse_bspline::NTuple{n, BSplineSpace}, nsubdivisions::NTuple{n, Int}) where {n}
    return ntuple(d -> build_two_scale_operator(coarse_bspline.knot_vector[d], nsubdivisions[d]), n)
end

"""
    build_two_scale_operator(coarse_bspline::NTuple{n, BSplineSpace}) where {n}

Algorithm for the coefficients of a change of B-spline representation for knot insertion 
of multiple knots, recursively using `single_knot_insertion_oslo()`.
The coarse knot vector is `coarse_bspline.knot_vector` and the inserted knots are given by bisection,
with multiplicity 1, across each dimension.

For more information, see [Paper](https://doi.org/10.1016/j.cma.2017.08.017).

# Arguments
- `coarse_bspline::NTuple{n, BSplineSpace}`: Coarse B-spline.
# Returns 
- `::NTuple{n, Vector{Array{Float64}}}`: Two-scale relation operators and finer B-spline spaces.
"""
function build_two_scale_operator(coarse_bspline::NTuple{n, BSplineSpace}) where {n}
    return build_two_scale_operator(coarse_bspline, ntuple(d -> 2, n))
end

"""
    build_two_scale_operator(coarse_bspline::NTuple{n, BSplineSpace}, fine_bspline::NTuple{n, BSplineSpace}) where {n}

Algorithm for the coefficients of a change of B-spline representation for knot insertion 
of multiple knots, recursively using `single_knot_insertion_oslo()`.
The coarse knot vector is `coarse_bspline.knot_vector` and the inserted knots are given by `fine_bspline.knot_vector`,
across each dimension.

For more information, see [Paper](https://doi.org/10.1016/j.cma.2017.08.017).

# Arguments
- `coarse_bspline::NTuple{n, BSplineSpace}`: Coarse B-splines.
- `fine_bspline::NTuple{n, BSplineSpace}`: Fine B-splines, with extra knots.
# Returns 
- `::NTuple{n, Vector{Array{Float64}}}`: Two-scale relation operators and finer B-spline spaces.
"""
function build_two_scale_operator(coarse_bspline::NTuple{n, BSplineSpace}, fine_bspline::NTuple{n, BSplineSpace}) where {n}
    return ntuple(d -> build_two_scale_operator(coarse_bspline.knot_vector[d], fine_bspline.knot_vector[d]), n)
end

# Getters for change of basis

"""
    subdivide_coeffs(coarse_basis_coeffs::Vector{Float64}, two_scale_operator::TwoScaleOperator)

Returns the spline coefficients in a refined basis from coefficients in a coarser B-spline basis.

# Arguments
- `coarse_basis_coeffs::Vector{Float64}`: Coefficients in the coarser basis.
- `two_scale_operator::TwoScaleOperator`: Two-scale operator the change of basis matrix.
# Returns
-`::Vector{Float64}`: Coefficients in the finer basis.
"""
function subdivide_coeffs(coarse_basis_coeffs::Vector{Float64}, two_scale_operator::TwoScaleOperator)
    return two_scale_operator.global_subdiv_matrix * coarse_basis_coeffs
end

"""
    get_local_subdiv_matrix(coarse_el_id::Int, fine_el_id::Int, two_scale_operator::TwoScaleOperator)

Returns the local subdivision matrix necessary to represent functions from a coarser space on
`coarse_el_id` in terms of finer functions on element `fine_el_id`.

# Arguments 
- `coarse_el_id::Int`: Id of the coarse element.
- `fine_el_id::Int`: Id of the fine element.
- `two_scale_operator::TwoScaleOperator`: Two-scale operator relating two B-spline spaces.
# Returns
- `::@views Array{Float64, 2}`: Local change of basis extraction matrix.
"""
function get_local_subdiv_matrix(coarse_el_id::Int, fine_el_id::Int, two_scale_operator::TwoScaleOperator)
    _, rows = get_extraction(two_scale_operator.fine_space, fine_el_id)
    _, columns = get_extraction(two_scale_operator.coarse_space, coarse_el_id)

    @views two_scale_operator.global_subdiv_matrix[rows, columns]
end
"""
Functions and algorithms used for knot insertion.

"""

# Helper functions to subdivide different objects

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
    subdivide_multiplicity(coarse_multiplicity::Vector{Int}, nsubdivisions::Int)

Subdivides `coarse_multiplicity` by uniformly subdiving each element 'nsubdivisions' times.
The coarse multiplicities are preserved in the `fine_multiplicity`, and newly inserted ones are given multiplicity 1.

# Arguments
- `coarse_multiplicity::Vector{Int}`: Coarse multiplicity vector.
- `nsubdivisions::Int`: Number of times each element is subdivided.
# Returns 
- `fine_multiplicity::Vector{Int}`: Fine multiplicity.
"""
function subdivide_multiplicity(coarse_multiplicity::Vector{Int}, nsubdivisions::Int)
    mult_length = 1 + length(coarse_multiplicity)*(nsubdivisions) - nsubdivisions 
    
    fine_multiplicity = fill(1, mult_length)
    
    coarse_idx = 1
    for k in eachindex(fine_multiplicity)
        if (k-1)%nsubdivisions + 1 == 1
            fine_multiplicity[k] = coarse_multiplicity[coarse_idx]
            coarse_idx += 1
        end
    end

    return fine_multiplicity
end

"""
    subdivide_bspline(coarse_bspline::FiniteElementSpaces.BSplineSpace, nsubdivisions::Int)

Subdivides `coarse_bspline` by uniformly subdiving each element 'nsubdivisions' times. The coarse multiplicities
are preserved in the `fine_multiplicity`, and newly inserted ones are given multiplicity 1.

# Arguments
- `coarse_bspline::FiniteElementSpaces.BSplineSpace`: Coarse B-spline.
- `nsubdivisions::Int`: Number of times each element is subdivided.
# Returns 
- `fine_bspline::FiniteElementSpaces.BSplineSpace`: Fine B-spline.
"""
function subdivide_bspline(coarse_bspline::FiniteElementSpaces.BSplineSpace, nsubdivisions::Int)
    fine_patch = subdivide_patch(FiniteElementSpaces.get_patch(coarse_bspline), nsubdivisions)
    fine_multiplicity = subdivide_multiplicity(FiniteElementSpaces.get_multiplicity(coarse_bspline), nsubdivisions)
    fine_regularity = FiniteElementSpaces.get_polynomial_degree(coarse_bspline) .- fine_multiplicity

    return FiniteElementSpaces.BSplineSpace(fine_patch, coarse_bspline.knot_vector.polynomial_degree, fine_regularity)
end

# Oslo knot insertion algorithms

"""
    single_knot_insertion_oslo(coarse_knot_vector::FiniteElementSpaces.KnotVector, fine_knot_vector::FiniteElementSpaces.KnotVector, cf::Int, rf::Int)

Algorithm for the coefficients of a change of B-spline representation for a single knot insertion.
The coarse knot vector is `coarse_knot_vector` and the inserted knot is given by `fine_knot_vector`.

For more information, see [A note on the Oslo Algorithm](https://collections.lib.utah.edu/dl_files/66/d4/66d493df0f5c97cce67e0bc1294363d64dde7f06.pdf).

# Arguments
- `coarse_knot_vector::FiniteElementSpaces.KnotVector`: Coarse knot vector.
- `fine_knot_vector::FiniteElementSpaces.KnotVector`: Fine knot vector, with the extra knot.
- `cf::Int`: Index of the coarse knot vector.
- `rf::Int`: Index of the fine knot vector such that `get_knot_breakpoint(coarse_knot_vector,cf) <= get_knot_breakpoint(fine_knot_vector,rf) < get_knot_breakpoint(coarse_knot_vector,cf+1)`.
# Returns 
- `b::Vector{Float64}`: Coefficients for the change of basis.
"""
function single_knot_insertion_oslo(coarse_knot_vector::FiniteElementSpaces.KnotVector, fine_knot_vector::FiniteElementSpaces.KnotVector, cf::Int, rf::Int)
    b = [1.0]

    for k in 1:coarse_knot_vector.polynomial_degree
        t1 = FiniteElementSpaces.get_knot_breakpoint.((coarse_knot_vector,), cf+1-k:cf)
        t2 = FiniteElementSpaces.get_knot_breakpoint.((coarse_knot_vector,), cf+1:cf+k)
        x = FiniteElementSpaces.get_knot_breakpoint(fine_knot_vector, rf+k)

        w =  (x .- t1) ./ (t2 .- t1)

        b = push!((1 .- w) .* b, 0) .+ pushfirst!(w .* b, 0)
    end


    return b
end

"""
    element_knot_insertion_operators(coarse_knot_vector::FiniteElementSpaces.KnotVector, fine_knot_vector::FiniteElementSpaces.KnotVector)

Algorithm for the coefficients of a change of B-spline representation for knot insertion 
of multiple knots, recursively using `single_knot_insertion_oslo()`.
The coarse knot vector is `coarse_knot_vector` and the inserted knots are given by `fine_knot_vector`.

For more information, see [Paper](https://doi.org/10.1016/j.cma.2017.08.017).

# Arguments
- `coarse_knot_vector::FiniteElementSpaces.KnotVector`: Coarse knot vector.
- `fine_knot_vector::FiniteElementSpaces.KnotVector`: Fine knot vector, with the extra knots.
# Returns 
- `R::Vector{Array{Float64}}`: Coefficients for the change of basis.
"""
function element_knot_insertion_operators(coarse_knot_vector::FiniteElementSpaces.KnotVector, fine_knot_vector::FiniteElementSpaces.KnotVector)
    m = FiniteElementSpaces.get_knot_length(fine_knot_vector)
    nel = size(fine_knot_vector.patch_1d)
    p = coarse_knot_vector.polynomial_degree

    cf = p + 1
    rf = 1
    e = 1

    R = FiniteElementSpaces.create_identity(nel, p+1)

    while rf <= m - p - 1
        mult = FiniteElementSpaces.get_knot_multiplicity(fine_knot_vector, rf)

        lastcf = cf
        while FiniteElementSpaces.get_knot_breakpoint(coarse_knot_vector, cf+1) <= FiniteElementSpaces.get_knot_breakpoint(fine_knot_vector, rf)
            cf += 1
        end

        if e > 1 
            offs = cf - lastcf
            R[e][1:p+1-offs, 1:p+1-mult] .= R[e-1][1+offs:p+1, 1+mult:p+1] 
        end
        for t in p+2-mult:p+1
            R[e][:, t] = single_knot_insertion_oslo(coarse_knot_vector, fine_knot_vector, cf, rf)
            rf += 1
        end
        e += 1
    end

    return R
end

"""
    element_knot_insertion_operators(coarse_bspline::FiniteElementSpaces.BSplineSpace, nsubdivisions::Int)

Algorithm for the coefficients of a change of B-spline representation for knot insertion 
of multiple knots, recursively using `single_knot_insertion_oslo()`.
The coarse knot vector is `coarse_bspline.knot_vector` and the inserted knots are given by `nsubdivisions`, meaning `nsubdivisions-1` uniformly spaced knots are inserted, between coarse breakpoints, with multiplicity 1.

For more information, see [Paper](https://doi.org/10.1016/j.cma.2017.08.017).

# Arguments
- `coarse_bspline::FiniteElementSpaces.BSplineSpace`: Coarse B-spline.
- `nsubdivisions::Int`: Number of times each element is subdivided.
# Returns 
- `R::Vector{Array{Float64}}`: Coefficients for the change of basis.
"""
function element_knot_insertion_operators(coarse_bspline::FiniteElementSpaces.BSplineSpace, nsubdivisions::Int)
    return element_knot_insertion_operators(coarse_bspline.knot_vector, nsubdivisions)
end

"""
    element_knot_insertion_operators(coarse_bspline::FiniteElementSpaces.BSplineSpace, fine_bspline::FiniteElementSpaces.BSplineSpace)

Algorithm for the coefficients of a change of B-spline representation for knot insertion 
of multiple knots, recursively using `single_knot_insertion_oslo()`.
The coarse knot vector is `coarse_bspline.knot_vector` and the inserted knots are given by `fine_bspline.knot_vector`.

For more information, see [Paper](https://doi.org/10.1016/j.cma.2017.08.017).

# Arguments
- `coarse_bspline::FiniteElementSpaces.BSplineSpace`: Coarse B-spline.
- `fine_bspline::FiniteElementSpaces.BSplineSpace`: Fine B-spline, with extra knots.
# Returns 
- `R::Vector{Array{Float64}}`: Coefficients for the change of basis.
"""
function element_knot_insertion_operators(coarse_bspline::FiniteElementSpaces.BSplineSpace, fine_bspline::FiniteElementSpaces.BSplineSpace)
    return element_knot_insertion_operators(coarse_bspline.knot_vector, fine_bspline.knot_vector)
end

"""
    element_knot_insertion_operators(coarse_knot_vector::FiniteElementSpaces.KnotVector, nsubdivisions::Int, fine_multiplicity::Vector{Int})

Algorithm for the coefficients of a change of B-spline representation for knot insertion 
of multiple knots, recursively using `single_knot_insertion_oslo()`.
The coarse knot vector is `coarse_knot_vector` and the inserted knots are given by `nsubdivisions`, meaning `nsubdivisions-1` uniformly spaced knots are inserted, between coarse breakpoints, with multiplicity given by `fine_multiplicity`.

For more information, see [Paper](https://doi.org/10.1016/j.cma.2017.08.017).

# Arguments
- `coarse_knot_vector::FiniteElementSpaces.KnotVector`: Coarse knot vector.
- `nsubdivisions::Int`: Number of times each element is subdivided.
 `fine_multiplicity::Vector{Int}`: Multiplicity of each knot in refined knot vector.
# Returns 
- `R::Vector{Array{Float64}}`: Coefficients for the change of basis.
"""
function element_knot_insertion_operators(coarse_knot_vector::FiniteElementSpaces.KnotVector, nsubdivisions::Int, fine_multiplicity::Vector{Int})
    fine_patch = subdivide_patch(coarse_knot_vector.patch_1d, nsubdivisions)
    fine_knot_vector = FiniteElementSpaces.KnotVector(fine_patch, coarse_knot_vector.polynomial_degree, fine_multiplicity)
    
    return element_knot_insertion_operators(coarse_knot_vector, fine_knot_vector)
end

"""
    element_knot_insertion_operators(coarse_knot_vector::FiniteElementSpaces.KnotVector, nsubdivisions::Int)

Algorithm for the coefficients of a change of B-spline representation for knot insertion 
of multiple knots, recursively using `single_knot_insertion_oslo()`.
The coarse knot vector is `coarse_knot_vector` and the inserted knots are given by `nsubdivisions`, meaning `nsubdivisions-1` uniformly spaced knots are inserted, between coarse breakpoints, with multiplicity 1.

For more information, see [Paper](https://doi.org/10.1016/j.cma.2017.08.017).

# Arguments
- `coarse_knot_vector::FiniteElementSpaces.KnotVector`: Coarse knot vector.
- `nsubdivisions::Int`: Number of times each element is subdivided.
# Returns 
- `R::Vector{Array{Float64}}`: Coefficients for the change of basis.
"""
function element_knot_insertion_operators(coarse_knot_vector::FiniteElementSpaces.KnotVector, nsubdivisions::Int)
    nsubdivisions > 0 || throw(ArgumentError("Number of subdivions must be greater than 0. 
    nsubdivisions=$nsubdivisions was given."))

    fine_multiplicity = subdivide_multiplicity(coarse_knot_vector.multiplicity, nsubdivisions)

    return element_knot_insertion_operators(coarse_knot_vector, nsubdivisions, fine_multiplicity)
end

"""
    element_knot_insertion_operators(coarse_knot_vector::FiniteElementSpaces.KnotVector)

Algorithm for the coefficients of a change of B-spline representation for knot insertion 
of multiple knots, recursively using `single_knot_insertion_oslo()`.
The coarse knot vector is `coarse_knot_vector` and the finer knot is obtained by bisection.

For more information, see [Paper](https://doi.org/10.1016/j.cma.2017.08.017).

# Arguments
- `coarse_knot_vector::FiniteElementSpaces.KnotVector`: Coarse knot vector.
# Returns 
- `R::Vector{Array{Float64}}`: Coefficients for the change of basis.
"""
function element_knot_insertion_operators(coarse_knot_vector::FiniteElementSpaces.KnotVector)
    return element_knot_insertion_operators(coarse_knot_vector, 2)
end

# Evaluations with refinement operator

"""
    evaluate(coarse_bspline::FiniteElementSpaces.BSplineSpace, fine_bspline::FiniteElementSpaces.BSplineSpace, fine_element_id::Int, xi::Vector{Float64}, nderivatives::Int, coefficients::Vector{Float64}, refinement_operator::Vector{Array{Float64}}, nsubdivisions::Int)

Evaluates a spline on the element specified by `fine_element_id` and points `xi` and all derivatives up to nderivatives, form a 
`coarse_bspline` basis with given `coefficients`, with respect to the basis `fine_bspline`.

# Arguments
- `coarse_bspline::BSplineSpace`: A coarse univariate B-Spline function space.
- `fine_bspline::BSplineSpace`: A fine univariate B-Spline function space.
- `fine_element_id::Int`: The id of the element.
- `xi::Vector{Float64}`: The points where the global basis is evaluated.
- `nderivatives::Int`: The order upto which derivatives need to be computed.
- `coefficients::Vector{Float64}`: Coefficients of the spline with basis `coarse_bspline`.
- `refinement_operator::Vector{Array{Float64}}`: Operator used for basis transformation.
- `nsubdivisions::Int`: Number of times each element is subdivided.
# Returns
- `::Array{Float64}`: Spline evaluation (size = n_eval_points x nderivatives+1).
"""
function evaluate(coarse_bspline::FiniteElementSpaces.BSplineSpace, fine_bspline::FiniteElementSpaces.BSplineSpace, fine_element_id::Int, xi::Vector{Float64}, nderivatives::Int, coefficients::Vector{Float64}, refinement_operator::Vector{Array{Float64}}, nsubdivisions::Int)
    coarse_element_id = get_coarser_element(fine_element_id, nsubdivisions)

    _, coarse_basis_indices = FiniteElementSpaces.get_extraction(coarse_bspline, coarse_element_id)
    fine_local_basis, _ = FiniteElementSpaces.evaluate(fine_bspline, fine_element_id, xi, nderivatives)
    evaluation = zeros(Float64, (size(fine_local_basis)[1], nderivatives+1) )
    
    for r = 0:nderivatives
        evaluation[:,r+1] .= @views sum((fine_local_basis[:,:,r+1] * refinement_operator[fine_element_id]') .* coefficients[coarse_basis_indices]', dims=2)
    end

    return evaluation
end
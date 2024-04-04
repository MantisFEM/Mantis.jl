"""
Functions and algorithms used for knot insertion.

"""

function subdivide_breakpoints(original_breakpoints::Vector{Float64}, n_subdivisions::Int)
    num_points = length(original_breakpoints) + (length(original_breakpoints) - 1)* (n_subdivisions - 1)

    subdivided_breakpoints = Vector{Float64}(undef, num_points)

    step_size = 1 / n_subdivisions

    for i in 1:length(original_breakpoints)-1, j in 0:n_subdivisions-1
        index = (i-1) * n_subdivisions + j + 1
        subdivided_breakpoints[index] = original_breakpoints[i] + j * step_size * (original_breakpoints[i+1] - original_breakpoints[i])
    end

    subdivided_breakpoints[end] = original_breakpoints[end]

    return subdivided_breakpoints
end

function subdivide_patch(original_patch::Mesh.Patch1D, n_subdivisions::Int)
    subdivided_breakpoints = subdivide_breakpoints(Mesh.get_breakpoints(original_patch), n_subdivisions)
    return Mesh.Patch1D(subdivided_breakpoints)
end

function subdivide_multiplicity(original_multiplicity::Vector{Int}, nsubdivisions::Int)
    mult_length = 1 + length(original_multiplicity)*(nsubdivisions) - nsubdivisions 
    
    fine_multiplicity = fill(1, mult_length)
    
    coarse_idx = 1
    for k in eachindex(fine_multiplicity)
        if (k-1)%nsubdivisions + 1 == 1
            fine_multiplicity[k] = original_multiplicity[coarse_idx]
            coarse_idx += 1
        end
    end

    return fine_multiplicity
end

function subdivide_bspline(original_bspline::FiniteElementSpaces.BSplineSpace, nsubdivisions::Int)
    fine_patch = subdivide_patch(original_bspline.knot_vector.patch_1d, nsubdivisions)
    fine_multiplicity = subdivide_multiplicity(original_bspline.knot_vector.multiplicity, nsubdivisions)
    fine_regularity = original_bspline.knot_vector.polynomial_degree .- fine_multiplicity

    return FiniteElementSpaces.BSplineSpace(fine_patch, original_bspline.knot_vector.polynomial_degree, fine_regularity)
end

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

function element_knot_insertion_operators(coarse_bspline::FiniteElementSpaces.BSplineSpace, n_subdivisions::Int)
    return element_knot_insertion_operators(coarse_bspline.knot_vector, n_subdivisions)
end

function element_knot_insertion_operators(coarse_bspline::FiniteElementSpaces.BSplineSpace, fine_bspline::FiniteElementSpaces.BSplineSpace)
    return element_knot_insertion_operators(coarse_bspline.knot_vector, fine_bspline.knot_vector)
end

function element_knot_insertion_operators(coarse_knot_vector::FiniteElementSpaces.KnotVector, n_subdivisions::Int, fine_multiplicity::Vector{Int})
    fine_patch = subdivide_patch(coarse_knot_vector.patch_1d, n_subdivisions)
    fine_knot_vector = FiniteElementSpaces.KnotVector(fine_patch, coarse_knot_vector.polynomial_degree, fine_multiplicity)
    
    return element_knot_insertion_operators(coarse_knot_vector, fine_knot_vector)
end

function element_knot_insertion_operators(coarse_knot_vector::FiniteElementSpaces.KnotVector, n_subdivisions::Int)
    n_subdivisions > 0 || throw(ArgumentError("Number of subdivions must be greater than 0. 
    n_subdivisions=$n_subdivisions was given."))

    fine_multiplicity = subdivide_multiplicity(coarse_knot_vector.multiplicity, n_subdivisions)

    return element_knot_insertion_operators(coarse_knot_vector, n_subdivisions, fine_multiplicity)
end

function element_knot_insertion_operators(coarse_knot_vector::FiniteElementSpaces.KnotVector)
    return element_knot_insertion_operators(coarse_knot_vector, 2)
end

function evaluate(coarse_bspline::FiniteElementSpaces.BSplineSpace, fine_bspline::FiniteElementSpaces.BSplineSpace, fine_element_id::Int, xi::Vector{Float64}, nderivatives::Int, coefficients::Vector{Float64}, refinement_operator::Vector{Array{Float64}}, nsubdivisions::Int)
    coarse_element_id = get_coarser_element(fine_element_id, nsubdivisions)

    _, coarse_basis_indices = FiniteElementSpaces.get_extraction(coarse_bspline, coarse_element_id)
    fine_local_basis, _ = FiniteElementSpaces.evaluate(fine_bspline, fine_element_id, xi, nderivatives)
    evaluation = zeros(Float64, (size(fine_local_basis)[1],nderivatives+1) )
    
    for r = 0:nderivatives
        evaluation[:,r+1] .= @views sum((fine_local_basis[:,:,r+1] * refinement_operator[fine_element_id]') .* coefficients[coarse_basis_indices]', dims=2)
    end

    return evaluation
end
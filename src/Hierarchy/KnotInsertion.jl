"""
Functions and algorithms used for knot insertion.

"""

function single_knot_insertion_oslo(coarse_knot_vector::FunctionSpaces.KnotVector, fine_knot_vector::FunctionSpaces.KnotVector, cf::Int, rf::Int)
    b = [1.0]

    for k in 1:coarse_knot_vector.polynomial_degree
        t1 = FunctionSpaces.get_knot_breakpoint.((coarse_knot_vector,), cf+1-k:cf)
        t2 = FunctionSpaces.get_knot_breakpoint.((coarse_knot_vector,), cf+1:cf+k)
        x = FunctionSpaces.get_knot_breakpoint(fine_knot_vector, rf+k)

        w =  (x .- t1) ./ (t2 .- t1)

        b = push!((1 .- w) .* b, 0) .+ pushfirst!(w .* b, 0)
    end


    return b
end

function element_knot_insertion_operators(coarse_knot_vector::FunctionSpaces.KnotVector, fine_knot_vector::FunctionSpaces.KnotVector)
    m = FunctionSpaces.get_knot_length(fine_knot_vector)
    nel = size(fine_knot_vector.patch_1d) + 1
    p = coarse_knot_vector.polynomial_degree

    cf = p + 1
    rf = 1
    e = 1

    R = FunctionSpaces.create_identity(nel-1, p+1)

    while rf <= m - p - 1
        mult = FunctionSpaces.get_knot_multiplicity(fine_knot_vector, rf)

        lastcf = cf
        while FunctionSpaces.get_knot_breakpoint(coarse_knot_vector, cf+1) <= FunctionSpaces.get_knot_breakpoint(fine_knot_vector, rf)
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

function element_knot_insertion_operators(polynomial_degree::Int, n_subdivisions::Int)
    coarse_patch = Mesh.Patch1D([0.0, 1.0])
    fine_patch = Mesh.Patch1D(collect(range(0.0, 1.0, n_subdivisions+1)))
    
    coarse_multiplicity = fill(polynomial_degree+1, 2)
    fine_multiplicity = fill(polynomial_degree, n_subdivisions+1)
    fine_multiplicity[1] = polynomial_degree + 1
    fine_multiplicity[end] = polynomial_degree + 1

    coarse_knot_vector = FunctionSpaces.KnotVector(coarse_patch, polynomial_degree, coarse_multiplicity)
    fine_knot_vector = FunctionSpaces.KnotVector(fine_patch, polynomial_degree, fine_multiplicity)

    return element_knot_insertion_operators(coarse_knot_vector, fine_knot_vector)
end
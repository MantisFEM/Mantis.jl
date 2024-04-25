import Mantis

using Test

const Patch1D = Mantis.Mesh.Patch1D
const KnotVector = Mantis.FunctionSpaces.KnotVector
const BSplineSpace = Mantis.FunctionSpaces.BSplineSpace

# Piece-wise degree of the basis functions on which the tests are performed.
degrees_for_test = 0:20
const subdivisions_to_test = 2:8

# Tests for validity of knot insertion algorithm
nq = 30
coeff_factor = 5

const fine_x = collect(range(0, 1, nq + 1))

for p in degrees_for_test
    nel = p+2
    coarse_regularity = fill(p-1, nel+1)
    coarse_regularity[1] = coarse_regularity[end] = -1
    coarse_bspline = BSplineSpace(Patch1D(collect(range(0,1, nel+1))), p, coarse_regularity)

    coarse_coeffs = (rand(Mantis.FunctionSpaces.get_dim(coarse_bspline)) .* 2 .- 1 ) .* coeff_factor
    
    for nsubdivision in subdivisions_to_test
        coarse_x = collect(range(0, 1, nq * nsubdivision + 1))
        
        two_scale_operator, fine_bspline = Mantis.FunctionSpaces.subdivide_bspline(coarse_bspline, nsubdivision)

        fine_coeffs = Mantis.FunctionSpaces.subdivide_coeffs(coarse_coeffs, two_scale_operator)
        
        for fine_el in 1:size(fine_bspline.knot_vector.patch_1d)
            coarse_el = Mantis.FunctionSpaces.get_coarser_element(fine_el, nsubdivision)
            coarse_spline_eval = Mantis.FunctionSpaces.evaluate(coarse_bspline, coarse_el, coarse_x, 0, coarse_coeffs)
            coarse_idx = (fine_el - 1)%nsubdivision + 1

            fine_spline_eval = Mantis.FunctionSpaces.evaluate(fine_bspline, fine_el, fine_x, 0, fine_coeffs)
            
            @test all(isapprox.(fine_spline_eval .- coarse_spline_eval[1+(coarse_idx-1)*nq:1+coarse_idx*nq], 0.0, atol=1e-13))
        end
    end
end

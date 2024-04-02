import Mantis

using Test

const Patch1D = Mantis.Mesh.Patch1D
const KnotVector = Mantis.FunctionSpaces.KnotVector
const BSplineSpace = Mantis.FunctionSpaces.BSplineSpace

# Piece-wise degree of the basis functions on which the tests are performed.
const degrees_to_test = 0:25
const subdivisions_to_test = 2:15

# Tests for validity of knot insertion algorithm

const test_patch = Patch1D([0.0, 1.0])
const regularity = [-1, -1]
const fine_x = collect(range(0,1, 10 + 1))

for p in degrees_to_test
    b_spline = BSplineSpace(test_patch, p, regularity)
    
    for subdivision in subdivisions_to_test
        refinement_operator = Mantis.Hierarchy.element_knot_insertion_operators(p, subdivision)

        
        coarse_x = collect(range(0,1, 10*subdivision + 1))
        coarse_spline_eval = Mantis.FunctionSpaces.evaluate(b_spline, 1, coarse_x, 0)

        
        for s in 1:subdivision
            fine_spline_eval = Mantis.FunctionSpaces.evaluate(b_spline, 1, fine_x, 0, refinement_operator, s)
            @test all(isapprox.(fine_spline_eval[1][:,:,1] .- coarse_spline_eval[1][1+(s-1)*10:1+s*10,:,1], 0.0, atol=1e-14))
        end
        
    end
    
end

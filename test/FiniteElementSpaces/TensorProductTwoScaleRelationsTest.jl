import Mantis

using Test

# Test parameters
breakpoints1 = [0.0, 0.5, 1.0]
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
breakpoints2 = [0.0, 0.5, 0.6, 1.0]
patch2 = Mantis.Mesh.Patch1D(breakpoints2)
breakpoints3 = [0.0, 1.0]
patch3 = Mantis.Mesh.Patch1D(breakpoints3)

deg1 = 2
deg2 = 3
deg3 = 1

B1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1, deg1-1, -1])
B2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1, min(deg2-1,1),  deg2-1, -1])
B3 = Mantis.FunctionSpaces.BSplineSpace(patch3, deg3, [-1, -1])

nsub1 = 3
nsub2 = 2
nsub3 = 4

ts1, BF1 = Mantis.FunctionSpaces.subdivide_bspline(B1, nsub1)
ts2, BF2 = Mantis.FunctionSpaces.subdivide_bspline(B2, nsub2)
ts3, BF3 = Mantis.FunctionSpaces.subdivide_bspline(B3, nsub3)

ttsl = Mantis.FunctionSpaces.TensorProductTwoScaleOperator(ts1, ts2)
ttsr = Mantis.FunctionSpaces.TensorProductTwoScaleOperator(ts2, ts3)

tts1 = Mantis.FunctionSpaces.TensorProductTwoScaleOperator(ttsl, ts3)
tts2 = Mantis.FunctionSpaces.TensorProductTwoScaleOperator(ts1, ttsr)

# Tests for TensorTwoScaleOperator

@test all(isapprox.(tts1.global_subdiv_matrix .- tts2.global_subdiv_matrix, 0.0, atol=1e-14)) # Check if global subdivision matrices are equal
@test Mantis.FunctionSpaces.get_num_elements(tts1.coarse_space) == Mantis.FunctionSpaces.get_num_elements(tts2.coarse_space) # Check total number of coarse elements
@test Mantis.FunctionSpaces.get_num_elements(tts1.fine_space) == Mantis.FunctionSpaces.get_num_elements(tts2.fine_space) # Check total number of fine elements

for el ∈ 1:1:Mantis.FunctionSpaces.get_num_elements(tts1.fine_space) # Check if local subdivision matrices are the same.
    coarse_el = Mantis.FunctionSpaces.get_coarser_element(tts1, el)
    all(isapprox.(Mantis.FunctionSpaces.get_local_subdiv_matrix(tts1, coarse_el, el) .- Mantis.FunctionSpaces.get_local_subdiv_matrix(tts2, coarse_el, el), 0.0, atol=1e-14))

    @test Mantis.FunctionSpaces.get_coarser_element(tts1, el) ==  Mantis.FunctionSpaces.get_coarser_element(tts2, el) # Check if coarse elements are the same.
end

for el ∈ 1:1:Mantis.FunctionSpaces.get_num_elements(tts1.coarse_space) # Check if finer elements are the same.
    @test Mantis.FunctionSpaces.get_finer_elements(tts1, el) == Mantis.FunctionSpaces.get_finer_elements(tts2, el) 
end

@test Mantis.FunctionSpaces.get_num_basis(tts1.coarse_space) == Mantis.FunctionSpaces.get_num_basis(tts2.coarse_space) # Check coarse dimensions
@test Mantis.FunctionSpaces.get_num_basis(tts1.fine_space) == Mantis.FunctionSpaces.get_num_basis(tts2.fine_space) # Check fine dimensions

for basis_id ∈ 1:1:Mantis.FunctionSpaces.get_num_basis(tts1.coarse_space) # Check if finer basis are the same
    @test Mantis.FunctionSpaces.get_finer_basis_id(tts1, basis_id) == Mantis.FunctionSpaces.get_finer_basis_id(tts2, basis_id)
end

for basis_id ∈ 1:1:Mantis.FunctionSpaces.get_num_basis(tts1.fine_space) # Check if coarser basis are the same
    @test Mantis.FunctionSpaces.get_coarser_basis_id(tts1, basis_id) == Mantis.FunctionSpaces.get_coarser_basis_id(tts2, basis_id)
end

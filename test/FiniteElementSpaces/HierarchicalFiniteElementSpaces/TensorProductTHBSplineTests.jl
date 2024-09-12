import Mantis

using Test
using Random

# Tests for a tensor product HierarchicalSplineSpace
ne1 = 5
ne2 = 5
breakpoints1 = collect(range(0,1,ne1+1))
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
breakpoints2 = collect(range(0,1,ne2+1))
patch2 = Mantis.Mesh.Patch1D(breakpoints2)
deg1 = 2
deg2 = 2
nsubs = (2, 2) 
nlevels = 3 # should be 2 or higher, otherwise the hierarchical space is not actually hierarchical
Random.seed!(9)

CB1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1; fill(deg1-1, ne1-1); -1])
CB2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1; fill(deg2-1, ne2-1); -1])
CTP = Mantis.FunctionSpaces.TensorProductSpace(CB1, CB2)


TTS, FTP = Mantis.FunctionSpaces.subdivide_bspline(CTP, nsubs)

spaces = [CTP, FTP]
operators = [TTS]

for level ∈ 3:nlevels
    new_operator, new_space = Mantis.FunctionSpaces.subdivide_bspline(spaces[level-1], nsubs)
    push!(spaces, new_space)
    push!(operators, new_operator)
end

marked_elements_per_level = [Int[], Mantis.FunctionSpaces.get_finer_elements(operators[1], [7,8,9,12,13,14,17,18,19]), Mantis.FunctionSpaces.get_finer_elements(operators[2], [23, 24, 25, 33, 34, 35, 43, 44, 45])] 
hspace = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, operators, marked_elements_per_level, true)

qrule = Mantis.Quadrature.tensor_product_rule((deg1+1, deg2+1), Mantis.Quadrature.gauss_legendre)
xi = Mantis.Quadrature.get_quadrature_nodes(qrule)

# Tests for coefficients and evaluation
for el in 1:1:Mantis.FunctionSpaces.get_num_elements(hspace)
    # check extraction coefficients
    ex_coeffs, _ = Mantis.FunctionSpaces.get_extraction(hspace, el)
    @test all(ex_coeffs .>= 0.0) # Test for non-negativity

    # check Hierarchical B-spline evaluation
    h_eval, _ = Mantis.FunctionSpaces.evaluate(hspace, el, xi, 0)
    # Positivity of the basis
    @test minimum(h_eval[1][1]) >= 0.0
    # Partition of unity
    @test all(isapprox.(sum(h_eval[1][1], dims=2), 1.0, atol=1e-14))
end

# Geometry visualization

function get_thb_geometry(hspace::Mantis.FunctionSpaces.HierarchicalFiniteElementSpace{n, S, T}) where {n, S<:Mantis.FunctionSpaces.AbstractFiniteElementSpace{n}, T<:Mantis.FunctionSpaces.AbstractTwoScaleOperator}
    L = Mantis.FunctionSpaces.get_num_levels(hspace)
    
    coefficients = Matrix{Float64}(undef, (Mantis.FunctionSpaces.get_num_basis(hspace), 2))

    id_sum = 1
    for level ∈ 1:1:L
        max_ind_basis = Mantis.FunctionSpaces._get_num_basis_per_space(hspace.spaces[level])
        x_greville_points = Mantis.FunctionSpaces.get_greville_points(hspace.spaces[level].function_space_1.knot_vector)
        y_greville_points = Mantis.FunctionSpaces.get_greville_points(hspace.spaces[level].function_space_2.knot_vector)
        grevile_mesh(x_id,y_id) = x_greville_points[x_id]*y_greville_points[y_id]
        
        _, level_active_basis = Mantis.FunctionSpaces.get_level_active(hspace.active_basis, level)

        for (y_count, y_id) ∈ enumerate(y_greville_points)
            for (x_count, x_id) ∈ enumerate(x_greville_points)
                if Mantis.FunctionSpaces.ordered_to_linear_index((x_count, y_count), max_ind_basis) ∈ level_active_basis
                    coefficients[id_sum, :] .= [x_id, y_id]
                    id_sum += 1
                end
            end
        end
    end

    return Mantis.Geometry.FEMGeometry(hspace, coefficients)
end

hspace_geo = get_thb_geometry(hspace)

# Generate the Plot
Mantis_folder =  dirname(dirname(pathof(Mantis)))
data_folder = joinpath(Mantis_folder, "test", "data")
output_data_folder = joinpath(data_folder, "output", "Geometry")

output_filename = "thb-partition-of-unity-test.vtu"
output_file = joinpath(output_data_folder, output_filename)
Mantis.Plot.plot(hspace_geo; vtk_filename = output_file[1:end-4], n_subcells = 1, degree = 4, ascii = false, compress = false)

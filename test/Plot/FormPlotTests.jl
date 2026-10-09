module FormPlotTests

using Mantis

import ReadVTK
using Test

# Setup the form spaces
# Space information
starting_points = (0.0, 0.0); box_sizes = (1.0, 1.0)
num_elements = (10, 10)
degrees = (2, 2); regularities = (1, 1)
# Then the form space
zero_form_space, one_form_space, top_form_space = Mantis.Forms.create_tensor_product_bspline_de_rham_complex(
    starting_points, box_sizes, num_elements,
    degrees, regularities
)

# Generate the form expressions
α⁰ = Forms.FormField(zero_form_space, rand(Forms.get_num_basis(zero_form_space)), "α")
ξ¹ = Forms.FormField(one_form_space, rand(Forms.get_num_basis(one_form_space)), "ξ")
β² = Forms.FormField(top_form_space, rand(Forms.get_num_basis(top_form_space)), "β")

num_basis = Mantis.Forms.get_num_basis(zero_form_space)

# Compute base directories for data input and output
output_directory_tree = [dirname(dirname(pathof(Mantis))), "test","data","output","Plot"]

out_deg = maximum([1, maximum(degrees)])

zero_form_filename = "zero-form-field-test.vtu"
zero_form_file = Mantis.GeneralHelpers.export_path(output_directory_tree, zero_form_filename)

one_form_filename = "one-form-field-test.vtu"
one_form_file = Mantis.GeneralHelpers.export_path(output_directory_tree, one_form_filename)

two_form_filename = "two-form-field-test.vtu"
two_form_file = Mantis.GeneralHelpers.export_path(output_directory_tree, two_form_filename)

@test_nowarn Mantis.Plot.plot(α⁰; vtk_filename = zero_form_file, n_subcells = 1, degree = out_deg, ascii = false, compress = false)
@test_nowarn Mantis.Plot.plot(ξ¹; vtk_filename = one_form_file, n_subcells = 1, degree = out_deg, ascii = false, compress = false)
@test_nowarn Mantis.Plot.plot(β²; vtk_filename = two_form_file, n_subcells = 1, degree = out_deg, ascii = false, compress = false)

# 2-forms in 3D are exported as the vector proxy of their Hodge star --------------------------
function read_point_data(file, label)
    vtk = ReadVTK.VTKFile(ReadVTK.get_example_file(file))
    return ReadVTK.get_data(ReadVTK.get_point_data(vtk)[label])
end

function export_point_data(form, filename)
    file = Mantis.GeneralHelpers.export_path(output_directory_tree, filename)
    Mantis.Plot.plot(form; vtk_filename=file, n_subcells=1, degree=2, ascii=false, compress=false)
    return read_point_data(file * ".vtu", Forms.get_label(form))
end

starting_points_3d = (0.0, 0.0, 0.0)
box_sizes_3d = (1.0, 1.0, 1.0)
num_elements_3d = (2, 3, 2)
degrees_3d = (2, 2, 2)
regularities_3d = (1, 1, 1)

# On the unit cube, the 2-form with all coefficients equal to one is
# dx²∧dx³ + dx³∧dx¹ + dx¹∧dx², by the partition of unity of the B-splines, so its vector
# proxy is (1, 1, 1) everywhere.
cartesian_complex_3d = Forms.create_tensor_product_bspline_de_rham_complex(
    starting_points_3d, box_sizes_3d, num_elements_3d, degrees_3d, regularities_3d
)
cartesian_two_form_space = cartesian_complex_3d[3]
ω² = Forms.FormField(
    cartesian_two_form_space, ones(Forms.get_num_basis(cartesian_two_form_space)), "ω"
)
ω²_data = export_point_data(ω², "two-form-field-3d-test")
@test size(ω²_data, 1) == 3
@test all(isapprox.(ω²_data, 1.0, atol=1e-12))

# On a curvilinear geometry, the exported 2-form is the export of its Hodge star.
curvilinear_mapping_3d = Geometry.create_curvilinear_mapping(
    starting_points_3d, box_sizes_3d, 0.2
)
curvilinear_complex_3d = Forms.create_tensor_product_bspline_de_rham_complex(
    starting_points_3d,
    box_sizes_3d,
    num_elements_3d,
    map(FunctionSpaces.Bernstein, degrees_3d),
    regularities_3d,
    curvilinear_mapping_3d,
)
curvilinear_two_form_space = curvilinear_complex_3d[3]
η² = Forms.FormField(
    curvilinear_two_form_space, rand(Forms.get_num_basis(curvilinear_two_form_space)), "η"
)
η²_data = export_point_data(η², "two-form-field-3d-curvilinear-test")
star_η²_data = export_point_data(★(η²), "star-two-form-field-3d-curvilinear-test")
@test any(abs.(η²_data) .> 1e-8)
@test η²_data ≈ star_η²_data

end

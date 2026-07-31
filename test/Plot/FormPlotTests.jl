module FormPlotTests

using Mantis

using Test

# Setup the form spaces
# Space information
starting_points = (0.0, 0.0);
box_sizes = (1.0, 1.0)
num_elements = (10, 10)
degrees = (2, 2);
regularities = (1, 1)
# Then the form space
zero_form_space, one_form_space, top_form_space = Mantis.Forms.create_tensor_product_bspline_de_rham_complex(
    starting_points, box_sizes, num_elements, degrees, regularities
)

# Generate the form expressions
α⁰ = Forms.FormField(zero_form_space, rand(Forms.get_num_basis(zero_form_space)), "α")
ξ¹ = Forms.FormField(one_form_space, rand(Forms.get_num_basis(one_form_space)), "ξ")
β² = Forms.FormField(top_form_space, rand(Forms.get_num_basis(top_form_space)), "β")

num_basis = Mantis.Forms.get_num_basis(zero_form_space)

# Compute base directories for data input and output
output_directory_tree = [dirname(dirname(pathof(Mantis))), "test", "data", "output", "Plot"]

out_deg = maximum([1, maximum(degrees)])

zero_form_filename = "zero-form-field-test.vtu"
zero_form_file = Mantis.GeneralHelpers.export_path(
    output_directory_tree, zero_form_filename
)

one_form_filename = "one-form-field-test.vtu"
one_form_file = Mantis.GeneralHelpers.export_path(output_directory_tree, one_form_filename)

two_form_filename = "two-form-field-test.vtu"
two_form_file = Mantis.GeneralHelpers.export_path(output_directory_tree, two_form_filename)

@test_nowarn Mantis.Plot.plot(
    α⁰;
    vtk_filename=zero_form_file,
    n_subcells=1,
    degree=out_deg,
    ascii=false,
    compress=false,
)
@test_nowarn Mantis.Plot.plot(
    ξ¹;
    vtk_filename=one_form_file,
    n_subcells=1,
    degree=out_deg,
    ascii=false,
    compress=false,
)
@test_nowarn Mantis.Plot.plot(
    β²;
    vtk_filename=two_form_file,
    n_subcells=1,
    degree=out_deg,
    ascii=false,
    compress=false,
)

end

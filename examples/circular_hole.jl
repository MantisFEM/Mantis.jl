import Mantis

import ReadVTK
using Printf
using Test
using LinearAlgebra

# parameterized closed curve
function implicit_curve(x::AbstractArray)
    return (x[:, 1] .- 1.5).^2 + (x[:, 2] .- 1.5).^2 .- 0.5 # (x[:, 1].^2 + x[:, 2].^2 .- 1).^3 - x[:, 1].^2 .* x[:, 2].^3
end
function on_the_curve(x::AbstractArray, tol)
    dist = implicit_curve(x)
    return any(abs.(dist) .<= tol)
end
function inside_the_box(x::AbstractArray)
    dist = implicit_curve(x)
    return any(dist .<= 0)
end

# degree
degree = (2, 2)
# number of levels
nlevels = 5
# number of subdivisions
nsub = (2, 2)
# origin
origin = (-2.0, -2.0)
# extent
extent = (4.0, 4.0)
# number of elements
num_elements = (ne1, ne2)

# create vectors of univariate space and two-scale hierarchy
B_uni = Vector{NTuple{2, Mantis.FunctionSpaces.BSplineSpace}}(undef, nlevels)
TS_uni = Vector{NTuple{2, Mantis.FunctionSpaces.AbstractTwoScaleOperator}}(undef, nlevels-1)
B_uni[1] = Mantis.FunctionSpaces.create_dim_wise_bspline_spaces(
    origin, extent, num_elements, degree, degree .- 1, (1, 1), (1, 1)
)
for l in 2:nlevels
    Ts1, Bf1 = Mantis.FunctionSpaces.build_two_scale_operator(B_uni[l-1][1], nsub[1])
    Ts2, Bf2 = Mantis.FunctionSpaces.build_two_scale_operator(B_uni[l-1][2], nsub[2])

    B_uni[l] = (Bf1, Bf2)
    TS_uni[l-1] = (Ts1, Ts2)
end

# create tensor-product space and two-scale hierarchy
B_tp = [Mantis.FunctionSpaces.TensorProductSpace(B_uni[l]) for l in 1:nlevels]
TS_tp = [
    Mantis.FunctionSpaces.TensorProductTwoScaleOperator(B_tp[l], B_tp[l+1], TS_uni[l])
    for l in 1:nlevels-1
]

# domain hierarchy
domains = Mantis.FunctionSpaces.HierarchicalActiveInfo(
    [
        collect(1:Mantis.FunctionSpaces.get_num_elements(B_tp[1])),
        [Int[] for l in 2:nlevels]...
    ]
)

# find elements to refine
inside_elements = [
    falses(Mantis.FunctionSpaces.get_num_elements(B_tp[l])) for l in 1:nlevels-1
]
tol = 0.3
for l = 1:nlevels-1
    # loop over all level-l elements and check which ones are inside the box
    for el in 1:Mantis.FunctionSpaces.get_num_elements(B_tp[l])
        # get element vertices
        element_vertices = Mantis.FunctionSpaces.get_element_vertices(
            B_tp[l], el
        )
        element_vertices = [
            [element_vertices[1][i] element_vertices[2][j]] for i in 1:2 for j in 1:2
        ]
        element_vertices = vcat(element_vertices...)

        if on_the_curve(element_vertices, tol / (2^(l-1)))
            inside_elements[l][el] = true
        end
    end
end

# find elements marked for refinement
marked_elements = [
    findall(inside_elements[l]) for l in 1:nlevels-1
]

# find corresponding elements that define the domain hierarchy
domain_elements = [
    vcat(Mantis.FunctionSpaces.get_element_children.(Ref(TS_tp[l]), marked_elements[l])...)
    for l in 1:nlevels-1
]

# update the domain hierarchy
domains = Mantis.FunctionSpaces.HierarchicalActiveInfo(
    [
        collect(1:Mantis.FunctionSpaces.get_num_elements(B_tp[1])),
        domain_elements...
    ]
)

# build the hierarchical space
HB_space = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(B_tp, TS_tp, domains)

# find which elements should be included in the output
exclude_elements = falses(Mantis.FunctionSpaces.get_num_elements(HB_space))
xi_corner = ([0.01, 0.99], [0.01, 0.99])
for el in 1:Mantis.FunctionSpaces.get_num_elements(HB_space)
    element_level, element_index = Mantis.FunctionSpaces.convert_to_element_level_and_level_id(HB_space, el)
    element_vertices = Mantis.FunctionSpaces.get_element_vertices(
        B_tp[element_level], element_index
    )
    element_vertices = [
            [element_vertices[1][i] element_vertices[2][j]] for i in 1:2 for j in 1:2
        ]
    element_vertices = vcat(element_vertices...)

    if inside_the_box(element_vertices)
        exclude_elements[el] = true
    end
end

# base cartesian mesh
breakpoints1 = collect(range(origin[1],origin[1]+extent[1],ne1+1))
patch1 = Mantis.Mesh.Patch1D(breakpoints1)
breakpoints2 = collect(range(origin[2],origin[2]+extent[2],ne2+1))
patch2 = Mantis.Mesh.Patch1D(breakpoints2)
# base cartesian geometry
cart_geometry = Mantis.Geometry.CartesianGeometry((breakpoints1, breakpoints2))

# reframe base geometry as a hierarchical geometry
hmapping = Mantis.Geometry.create_hierarchical_mesh_nestedness_map(HB_space, findall(exclude_elements))
hgeometry_cart = Mantis.Geometry.NestedGeometry(cart_geometry, hmapping)

# output file
file_name = "circular_hole.vtu"
output_directory_tree = ["examples", "data", "output"]
output_file_path = Mantis.Plot.export_path(output_directory_tree, file_name)
Mantis.Plot.plot(
    hgeometry_cart;
    vtk_filename=output_file_path[1:(end - 4)],
    n_subcells=1,
    degree=2,
    ascii=false,
    compress=false
)

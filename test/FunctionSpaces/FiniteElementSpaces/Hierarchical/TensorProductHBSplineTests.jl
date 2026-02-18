module TensorProductHBSplineTests

import Mantis

using Test

# Tests for a tensor product HierarchicalSplineSpace

ne1 = 5
ne2 = 5
breakpoints1 = collect(range(0,1,ne1+1))
patch1 = Mantis.Geometry.CartesianGeometry(breakpoints1)
breakpoints2 = collect(range(0,1,ne2+1))
patch2 = Mantis.Geometry.CartesianGeometry(breakpoints2)

deg1 = 2
deg2 = 2

CB1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1; fill(deg1-1, ne1-1); -1])
CB2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1; fill(deg2-1, ne2-1); -1])

nsub1 = 2
nsub2 = 2

TS1,FB1 = Mantis.FunctionSpaces.build_two_scale_operator(CB1, nsub1)
TS2, FB2 = Mantis.FunctionSpaces.build_two_scale_operator(CB2, nsub2)

CTP = Mantis.FunctionSpaces.TensorProductSpace((CB1, CB2))
FTP = Mantis.FunctionSpaces.TensorProductSpace((FB1, FB2))
spaces = [CTP, FTP]

CTP_num_els = Mantis.FunctionSpaces.get_num_elements(CTP)

CTS = Mantis.FunctionSpaces.TensorProductTwoScaleOperator(CTP, FTP, (TS1,TS2))

coarse_elements_to_refine = [3,4,5,8,9,10]
refined_elements = vcat(Mantis.FunctionSpaces.get_element_children.(Ref(CTS), coarse_elements_to_refine)...)

refined_domains = Mantis.FunctionSpaces.HierarchicalActiveInfo([collect(1:CTP_num_els),refined_elements])

###
hier_space = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(spaces, [CTS], refined_domains, (nsub1, nsub2))

qrule = Mantis.Quadrature.tensor_product_rule((deg1+1, deg2+1), Mantis.Quadrature.gauss_legendre)
xi = Mantis.Quadrature.get_nodes(qrule)

# Tests for coefficients and evaluation
for element_id in 1:1:Mantis.FunctionSpaces.get_num_elements(hier_space)

    # check extraction coefficients
    ex_coeffs, _ = Mantis.FunctionSpaces.get_extraction(hier_space, element_id)
    @test all(ex_coeffs .>= 0.0) # Test for non-negativity

    # check Hierarchical B-spline evaluation
    h_eval, _ = Mantis.FunctionSpaces.evaluate(hier_space, element_id, xi, 0)
    # Positivity of the basis
    @test minimum(h_eval[1][1][1]) >= 0.0
end


ne1 = 10
ne2 = 10
breakpoints1 = collect(range(0,1,ne1+1))
patch1 = Mantis.Geometry.CartesianGeometry(breakpoints1)
breakpoints2 = collect(range(0,1,ne2+1))
patch2 = Mantis.Geometry.CartesianGeometry(breakpoints2)

deg1 = 3
deg2 = 3

CB1 = Mantis.FunctionSpaces.BSplineSpace(patch1, deg1, [-1; fill(deg1-1, ne1-1); -1])
CB2 = Mantis.FunctionSpaces.BSplineSpace(patch2, deg2, [-1; fill(deg2-1, ne2-1); -1])

nsub1 = 2
nsub2 = 2

TS1,FB1 = Mantis.FunctionSpaces.build_two_scale_operator(CB1, nsub1)
TS2, FB2 = Mantis.FunctionSpaces.build_two_scale_operator(CB2, nsub2)

CTP = Mantis.FunctionSpaces.TensorProductSpace((CB1, CB2))
FTP = Mantis.FunctionSpaces.TensorProductSpace((FB1, FB2))
spaces = [CTP, FTP]

CTP_num_els = Mantis.FunctionSpaces.get_num_elements(CTP)

CTS = Mantis.FunctionSpaces.TensorProductTwoScaleOperator(CTP, FTP, (TS1,TS2))

basis1_support = Mantis.FunctionSpaces.get_support(CTP, 57)
basis2_support = Mantis.FunctionSpaces.get_support(CTP, 98)
coarse_elements_to_refine = vcat(basis1_support, basis2_support)
refined_elements = vcat(Mantis.FunctionSpaces.get_element_children.(Ref(CTS), coarse_elements_to_refine)...)

refined_domains = Mantis.FunctionSpaces.HierarchicalActiveInfo([collect(1:CTP_num_els),refined_elements])

non_simplified_hier_space = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(
    spaces, [CTS], refined_domains, (nsub1, nsub2), false
)
simplified_hier_space = Mantis.FunctionSpaces.HierarchicalFiniteElementSpace(
    spaces, [CTS], refined_domains, (nsub1, nsub2), false, true
)

for level in 1:2
    @test length(Mantis.FunctionSpaces.get_level_basis_ids(simplified_hier_space, level)) <=
        length(Mantis.FunctionSpaces.get_level_basis_ids(non_simplified_hier_space, level))
    if level == 2
        @test Mantis.FunctionSpaces.get_level_basis_ids(simplified_hier_space, level) !=
            Mantis.FunctionSpaces.get_level_basis_ids(non_simplified_hier_space, level)
    end
end

end

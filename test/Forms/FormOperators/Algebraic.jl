module AlgebraicTests

using Mantis
using Test

starting_point_2d = (0.0, 0.0)
box_size_2d = (1.0, 1.0)
num_elements_2d = (1, 1)
num_elements = prod(num_elements_2d)
degree_2d = (2, 3)
section_spacesW = map(FunctionSpaces.Bernstein, degree_2d)
section_spacesX = map(FunctionSpaces.Bernstein, degree_2d[end:-1:1])
regularity_2d = (1, 1)
# Create a global quadrature rule for testing
nqrule = max(degree_2d...) + 1
qrule = Quadrature.tensor_product_rule((nqrule, nqrule), Quadrature.gauss_legendre)
dΩ = Quadrature.StandardQuadrature(qrule, prod(num_elements_2d))
xi = Points.CartesianPoints(([0.0, 0.5, 1.0], [0.1, 0.2, 0.86]))
# Create the geometry
G = Geometry.create_cartesian_box(starting_point_2d, box_size_2d, num_elements_2d)
# Create the B-spline de Rham complex
W0, W1, W2 = Forms.create_tensor_product_bspline_de_rham_complex(
    starting_point_2d, box_size_2d, num_elements_2d, section_spacesW, regularity_2d, G
)
X0, X1, X2 = Forms.create_tensor_product_bspline_de_rham_complex(
    starting_point_2d, box_size_2d, num_elements_2d, section_spacesX, regularity_2d, G
)
# Create FormFields for testing
w0_c = 1.0
w1_c = 1.0
w2_c = 1.0
x0_c = 2.0
x1_c = 3.0
x2_c = 4.0
w0 = Forms.FormField(W0, "w⁰")
w0.coefficients .= w0_c
w1 = Forms.FormField(W1, "w¹")
w1.coefficients .= w1_c
w2 = Forms.FormField(W2, "w²")
w2.coefficients .= w2_c
x0 = Forms.FormField(X0, "x⁰")
x0.coefficients .= x0_c
x1 = Forms.FormField(X1, "x¹")
x1.coefficients .= x1_c
x2 = Forms.FormField(X2, "x²")
x2.coefficients .= x2_c
@testset "UnaryOperatorTransformation" begin
    int0 = ∫(w0 ∧ ★(x0), dΩ)
    int1 = ∫(w1 ∧ ★(x1), dΩ)
    int2 = ∫(w2 ∧ ★(x2), dΩ)
    int0_val = w0_c * x0_c
    int1_val = 2.0 * w1_c * x1_c
    int2_val = w2_c * x2_c
    # Coefficient
    c = 1.337
    @test isapprox(Forms.evaluate(c * int0, 1)[1][1], c * int0_val)
    @test isapprox(Forms.evaluate(c * int1, 1)[1][1], c * int1_val)
    @test isapprox(Forms.evaluate(c * int2, 1)[1][1], c * int2_val)
    # Additive inverse
    @test isapprox(Forms.evaluate(-int0, 1)[1][1], -int0_val)
    @test isapprox(Forms.evaluate(-int1, 1)[1][1], -int1_val)
    @test isapprox(Forms.evaluate(-int2, 1)[1][1], -int2_val)
end

@testset "BinaryOperatorTransformation" begin
    int0 = ∫(w0 ∧ ★(x0), dΩ)
    int1 = ∫(w1 ∧ ★(x1), dΩ)
    int2 = ∫(w2 ∧ ★(x2), dΩ)
    int0_val = w0_c * x0_c
    int1_val = 2.0 * w1_c * x1_c
    int2_val = w2_c * x2_c
    # Addition
    @test isapprox(Forms.evaluate(int0 + int1, 1)[1][1], int0_val + int1_val)
    @test isapprox(Forms.evaluate(int1 + int2, 1)[1][1], int1_val + int2_val)
    @test isapprox(Forms.evaluate(int0 + int2, 1)[1][1], int0_val + int2_val)
    # Subtraction
    @test isapprox(Forms.evaluate(int0 - int1, 1)[1][1], int0_val - int1_val)
    @test isapprox(Forms.evaluate(int1 - int2, 1)[1][1], int1_val - int2_val)
    @test isapprox(Forms.evaluate(int2 - int0, 1)[1][1], int2_val - int0_val)
    # Scalar multiplication and addition
    c1 = 2.5
    c2 = -0.7
    @test isapprox(
        Forms.evaluate(c1 * int0 + c2 * int1, 1)[1][1], c1 * int0_val + c2 * int1_val
    )
    @test isapprox(
        Forms.evaluate(c1 * int2 - c2 * int1, 1)[1][1], c1 * int2_val - c2 * int1_val
    )
    # Nested binary operations
    @test isapprox(
        Forms.evaluate((int0 + int1) - int2, 1)[1][1], (int0_val + int1_val) - int2_val
    )
    @test isapprox(
        Forms.evaluate(int0 + (int1 - int2), 1)[1][1], int0_val + (int1_val - int2_val)
    )
end

@testset "UnaryFormTransformation" begin
    w0_val = Forms.evaluate(w0, 1, xi)[1][1]
    w1_val = Forms.evaluate(w1, 1, xi)[1]
    w2_val = Forms.evaluate(w2, 1, xi)[1][1]
    x0_val = Forms.evaluate(x0, 1, xi)[1][1]
    x1_val = Forms.evaluate(x1, 1, xi)[1]
    x2_val = Forms.evaluate(x2, 1, xi)[1][1]
    # Coefficient
    c = 6.66
    @test isapprox(Forms.evaluate(c * w0, 1, xi)[1][1], c * w0_val)
    @test isapprox(Forms.evaluate(c * w1, 1, xi)[1][1], c * w1_val[1])
    @test isapprox(Forms.evaluate(c * w1, 1, xi)[1][2], c * w1_val[2])
    @test isapprox(Forms.evaluate(c * w2, 1, xi)[1][1], c * w2_val)
    @test isapprox(Forms.evaluate(c * x0, 1, xi)[1][1], c * x0_val)
    @test isapprox(Forms.evaluate(c * x1, 1, xi)[1][1], c * x1_val[1])
    @test isapprox(Forms.evaluate(c * x1, 1, xi)[1][2], c * x1_val[2])
    @test isapprox(Forms.evaluate(c * x2, 1, xi)[1][1], c * x2_val)
    # Additive inverse
    @test isapprox(Forms.evaluate(-w0, 1, xi)[1][1], -w0_val)
    @test isapprox(Forms.evaluate(-w1, 1, xi)[1][1], -w1_val[1])
    @test isapprox(Forms.evaluate(-w1, 1, xi)[1][2], -w1_val[2])
    @test isapprox(Forms.evaluate(-w2, 1, xi)[1][1], -w2_val)
    @test isapprox(Forms.evaluate(-x0, 1, xi)[1][1], -x0_val)
    @test isapprox(Forms.evaluate(-x1, 1, xi)[1][1], -x1_val[1])
    @test isapprox(Forms.evaluate(-x1, 1, xi)[1][2], -x1_val[2])
    @test isapprox(Forms.evaluate(-x2, 1, xi)[1][1], -x2_val)
end

@testset "BinaryFormTransformation" begin
    # Addition
    w0_plus_x0 = w0 + x0
    w1_plus_x1 = w1 + x1
    w2_plus_x2 = w2 + x2
    w0_val = Forms.evaluate(w0, 1, xi)[1][1]
    x0_val = Forms.evaluate(x0, 1, xi)[1][1]
    w1_val = Forms.evaluate(w1, 1, xi)[1]
    x1_val = Forms.evaluate(x1, 1, xi)[1]
    w2_val = Forms.evaluate(w2, 1, xi)[1][1]
    x2_val = Forms.evaluate(x2, 1, xi)[1][1]
    @test isapprox(Forms.evaluate(w0_plus_x0, 1, xi)[1][1], w0_val + x0_val)
    @test isapprox(Forms.evaluate(w1_plus_x1, 1, xi)[1][1], w1_val[1] + x1_val[1])
    @test isapprox(Forms.evaluate(w1_plus_x1, 1, xi)[1][2], w1_val[2] + x1_val[2])
    @test isapprox(Forms.evaluate(w2_plus_x2, 1, xi)[1][1], w2_val + x2_val)
    # Subtraction
    w0_minus_x0 = w0 - x0
    w1_minus_x1 = w1 - x1
    w2_minus_x2 = w2 - x2
    @test isapprox(Forms.evaluate(w0_minus_x0, 1, xi)[1][1], w0_val - x0_val)
    @test isapprox(Forms.evaluate(w1_minus_x1, 1, xi)[1][1], w1_val[1] - x1_val[1])
    @test isapprox(Forms.evaluate(w1_minus_x1, 1, xi)[1][2], w1_val[2] - x1_val[2])
    @test isapprox(Forms.evaluate(w2_minus_x2, 1, xi)[1][1], w2_val - x2_val)
    # Coefficient and addition
    c1 = 2.5
    c2 = -0.7
    expr1 = c1 * w0 + c2 * x0
    expr2 = c1 * w1 - c2 * x1
    expr3 = c1 * w2 + c2 * x2
    @test isapprox(Forms.evaluate(expr1, 1, xi)[1][1], c1 * w0_val + c2 * x0_val)
    @test isapprox(Forms.evaluate(expr2, 1, xi)[1][1], c1 * w1_val[1] - c2 * x1_val[1])
    @test isapprox(Forms.evaluate(expr2, 1, xi)[1][2], c1 * w1_val[2] - c2 * x1_val[2])
    @test isapprox(Forms.evaluate(expr3, 1, xi)[1][1], c1 * w2_val + c2 * x2_val)
    # Nested binary operations
    nested1 = (w0 + x0) - w0
    nested2 = w1 + (x1 - w1)
    @test isapprox(Forms.evaluate(nested1, 1, xi)[1][1], (w0_val + x0_val) - w0_val)
    @test isapprox(
        Forms.evaluate(nested2, 1, xi)[1][1], w1_val[1] + (x1_val[1] - w1_val[1])
    )
    @test isapprox(
        Forms.evaluate(nested2, 1, xi)[1][2], w1_val[2] + (x1_val[2] - w1_val[2])
    )
end

end

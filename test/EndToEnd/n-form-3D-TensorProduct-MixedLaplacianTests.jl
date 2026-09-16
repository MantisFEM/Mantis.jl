module NForm3DTensorProductMixedLaplacianTests

using Mantis

using Test

# Exact solution
const w = 2.0 * pi
function my_sol(xx::Matrix{Float64})
    xs = xx[:, 1]
    ys = xx[:, 2]
    zs = xx[:, 3]
    return [[sin(w*x) * sin(w*y) * sin(w*z) for (x, y, z) in zip(xs, ys, zs)]]
end

function flux_my_sol(xx::Matrix{Float64})
    xs = xx[:, 1]
    ys = xx[:, 2]
    zs = xx[:, 3]
    return [
        [-w * cos(w*x) * sin(w*y) * sin(w*z) for (x, y, z) in zip(xs, ys, zs)],
        [-w * sin(w*x) * cos(w*y) * sin(w*z) for (x, y, z) in zip(xs, ys, zs)],
        [-w * sin(w*x) * sin(w*y) * cos(w*z) for (x, y, z) in zip(xs, ys, zs)],
    ]
end

function forcing_function(xx::Matrix{Float64})
    xs = xx[:, 1]
    ys = xx[:, 2]
    zs = xx[:, 3]
    return [[3.0 * w^2 * sin(w*x) * sin(w*y) * sin(w*z) for (x, y, z) in zip(xs, ys, zs)]]
end

errors = Float64[]
errors_u = Float64[]
canonical_qrule = Quadrature.tensor_product_rule((4, 4, 4), Quadrature.gauss_legendre)
for num_elements in ((2, 2, 2), (4, 4, 4), (8, 8, 8), (16, 16, 16))
    dΩ = Quadrature.StandardQuadrature(canonical_qrule, prod(num_elements))

    X0, X1, X2, X3 = Forms.create_tensor_product_bspline_de_rham_complex(
        (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), num_elements, (2, 2, 2), (1, 1, 1)
    )

    geometry = Forms.get_geometry(X0)
    # exact solution for the problem
    ϕₑ = Forms.AnalyticalFormField(3, my_sol, geometry, "ϕ")
    uₑ = Forms.AnalyticalFormField(2, flux_my_sol, geometry, "δϕ")
    f = Forms.AnalyticalFormField(3, forcing_function, geometry, "f")

    # solve the problem
    uₕ, ϕₕ = Assemblers.solve_volume_form_hodge_laplacian(X2, X3, f, dΩ)

    # compute error
    error = Analysis.L2_norm(ϕₕ - ϕₑ, dΩ)
    δerror = Analysis.L2_norm(uₕ - uₑ, dΩ)
    append!(errors, error)
    append!(errors_u, δerror)
end

hs = [1.0/i for i in (2, 4, 8, 16)]
rates = [
    log(errors[i]/errors[i + 1])/(log(hs[i]/hs[i + 1])) for
    i in eachindex(errors)[1:(end - 1)]
]
rates_u = [
    log(errors_u[i]/errors_u[i + 1])/(log(hs[i]/hs[i + 1])) for
    i in eachindex(errors_u)[1:(end - 1)]
]

# The rate is accurately computed to 1 decimal place.
@test isapprox(rates[end], 2, rtol=1e-1)
@test isapprox(rates_u[end], 2, rtol=1e-1)

end

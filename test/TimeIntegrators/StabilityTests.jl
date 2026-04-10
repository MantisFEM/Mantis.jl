module TimeIntegrationStabilityTests

import SparseArrays
import LinearAlgebra
using Mantis
using Test

# We can test (some) numerical schemes by performing a numerical Von Neumann analysis over
# one time step and then check if the computed eigenmode is the same as predicted by theory.
# The theoretical stability functions are exact, so we can match the value to machine
# precision.
# Because the eigenmodes are expressed as complex numbers, this test also ensures that we
# can use more general number types than just Float64.

# We use the advection equation in 1D with a simple finite difference spatial
# discretisation to verify stability in the case where we solve for multiple variables.

# Consider the advection equation on a 1D domain (x_L, x_R) as
# dc/dt = Ac
# where c is the vector of unknowns, and A is the (N-1, N-1) discretisation matrix.

const dx = 0.01
const L = 1.0
const velocity = 0.1
const dt = dx / velocity  # gives CFL = 1
const grid = 0.0:dx:L
const A =
    (velocity / dx) *
    SparseArrays.spdiagm(0 => ones(length(grid)), -1 => -1 .* ones(length(grid)-1))
# Set periodic boundary conditions.
A[1, end] = -(velocity / dx)

function discretised_advection_equation(output, c, t)
    LinearAlgebra.mul!(output, -A, c)
    return nothing
end

c_0 = sinpi.(2.0 .* grid)

linear_advection_ode = TimeIntegrators.define_explicit_ode(discretised_advection_equation)

# Amplification factor analysis ------------------------------------------------------------
const k = 2 * pi  # wavenumber
const ck_0 = exp.(im .* k .* grid)

# From the spatial discretisation:
const lambda_k = - velocity * (1 - exp(-im * k * dx)) / dx

const z_k = dt * lambda_k

# Known stability functions:
# Note that these functions hold for rk methods of the same order if they have the same
# number of stages. This means that, for example, HEUN2 and RALSTON2 have the same
# stability function because they are both 2-stage RK methods of order 2.
function exact_stability(z, order)
    if order == 1
        return 1 + z
    elseif order == 2
        return 1 + z + 1/2 * z^2
    elseif order == 3
        return 1 + z + 1/2 * z^2 + 1/6 * z^3
    elseif order == 4
        return 1 + z + 1/2 * z^2 + 1/6 * z^3 + 1/24 * z^4
    end
end

const integrators = (
    TimeIntegrators.FORWARD_EULER,
    TimeIntegrators.EXPLICIT_MIDPOINT,
    TimeIntegrators.HEUN2,
    TimeIntegrators.RALSTON2,
    TimeIntegrators.HEUN3,
    TimeIntegrators.RK3,
    TimeIntegrators.RALSTON3,
    TimeIntegrators.VDHW3,
    TimeIntegrators.SSPRK3,
    TimeIntegrators.RK4,
    TimeIntegrators.RK4_3_8,
    TimeIntegrators.RALSTON4,
)

@testset "Single-Step Multi-Stage Explicit Integrators" verbose = true begin
    foreach(integrators) do scheme
        ck_n = TimeIntegrators.initialize_scheme(ck_0, scheme)
        TimeIntegrators.time_integrate!(ck_n, linear_advection_ode, 0.0, dt)

        # Pick just one factor to test (away from the boundary condition).
        amplification_factor_scheme = (TimeIntegrators.get_solution(ck_n) ./ ck_0)[14]
        @test isapprox(
            exact_stability(z_k, TimeIntegrators.get_order(scheme)),
            amplification_factor_scheme,
            rtol=1e-15,
        )
    end
end

# Known stability functions for diagonally implicit schemes. Here, the stability functions
# are usually not the same per order, so we specify them per integrator.
const gamma4 = (1 + TimeIntegrators._α_DIRK4) / 2
const diagonally_implicit_integrators = (
    (TimeIntegrators.BACKWARD_EULER, z -> 1 / (1-z)),
    (TimeIntegrators.RADAU_IA_1, z -> 1 / (1-z)),
    (TimeIntegrators.IMPLICIT_MIDPOINT, z -> (1 + 1/2 * z) / (1 - 1/2*z)),
    (
        TimeIntegrators.DIRK2,
        z ->
            (
                1 +
                (1 - 2*TimeIntegrators._α_DIRK2)z +
                (TimeIntegrators._α_DIRK2^2 - 2*TimeIntegrators._α_DIRK2 + 1/2) * z^2
            ) / (1 - TimeIntegrators._α_DIRK2*z)^2,
    ),
    (
        TimeIntegrators.DIRK3,
        z ->
            (
                1 +
                (1 - 2*(1/2 + sqrt(3)/6))z +
                ((1/2 + sqrt(3)/6)^2 - 2*(1/2 + sqrt(3)/6) + 1/2) * z^2
            ) / (1 - (1/2 + sqrt(3)/6)*z)^2,
    ),
    (
        TimeIntegrators.DIRK4,
        z ->
            (
                1 +
                (1 - 3*gamma4)z +
                (3*gamma4^2 - 3*gamma4 + 1/2) * z^2 +
                (-gamma4^3 + 3*gamma4^2 - 3/2*gamma4 + 1/6) * z^3
            ) / (1 - gamma4*z)^3,
    ),
)

function implicitSolve(output, x, h, t)
    LinearAlgebra.ldiv!(output, LinearAlgebra.lu(LinearAlgebra.I + h * A), x)
    return nothing
end
dimplicit_linear_advection_ode = TimeIntegrators.define_diagonally_implicit_ode(
    implicitSolve
)

@testset "Single-Step Multi-Stage Diagonally Implicit Integrators" verbose = true begin
    foreach(diagonally_implicit_integrators) do (scheme, exact_stability_function)
        ck_n = TimeIntegrators.initialize_scheme(ck_0, scheme)
        TimeIntegrators.time_integrate!(ck_n, dimplicit_linear_advection_ode, 0.0, dt)

        # Pick just one factor to test (away from the boundary condition).
        amplification_factor_scheme = (TimeIntegrators.get_solution(ck_n) ./ ck_0)[84]
        @test isapprox(
            exact_stability_function(z_k), amplification_factor_scheme, rtol=1e-15
        )
    end
end

const implicit_integrators = (
    TimeIntegrators.RADAU_IA_3,
    TimeIntegrators.GAUSS_LEGENDRE_4,
    TimeIntegrators.GAUSS_LEGENDRE_6,
)
function rk_stability(A, b, z)
    A = Matrix(A)
    b = vec(Matrix(b))
    s = size(A, 1)
    e = ones(ComplexF64, s)
    return 1 + z * LinearAlgebra.dot(b, (LinearAlgebra.I - z*A) \ e)
end
function large_solve(output, x, h, t; num_stages)
    # Julia's kron is column major, so we have to reverse the arguments.
    output .= (LinearAlgebra.I - kron(h, -A)) \ x
    return nothing
end
function large_eval(output, c, t; num_stages)
    bigA = SparseArrays.blockdiag([-A for i in 1:num_stages]...)
    LinearAlgebra.mul!(output, bigA, c)
    return nothing
end
implicit_linear_advection_ode = TimeIntegrators.define_implicit_ode(large_solve, large_eval)

@testset "Single-Step Multi-Stage Implicit Integrators" verbose = true begin
    foreach(implicit_integrators) do scheme
        ck_n = TimeIntegrators.initialize_scheme(ck_0, scheme)
        TimeIntegrators.time_integrate!(
            ck_n,
            implicit_linear_advection_ode,
            0.0,
            dt;
            num_stages=TimeIntegrators.get_num_stages(scheme)
        )

        # Pick just one factor to test (away from the boundary condition).
        amplification_factor_scheme = (TimeIntegrators.get_solution(ck_n) ./ ck_0)[84]
        @test isapprox(
            rk_stability(scheme.A, scheme.B, z_k), amplification_factor_scheme, rtol=1e-15
        )
    end
end



function glm_amplification(scheme, z)

    A = Matrix(scheme.A)
    B = Matrix(scheme.B)
    U = Matrix(scheme.U)
    V = Matrix(scheme.V)

    s = size(A,1)

    M = V + z * B * ((LinearAlgebra.I - z*A) \ U)

    λ = LinearAlgebra.eigvals(M)

    return λ[argmin(abs.(λ .- 1))]
end

function create_history(scheme, z)
    ξ = glm_amplification(scheme, z)

    history = zeros(
        ComplexF64,
        length(grid),
        length(scheme.time_levels.step_values) + length(scheme.time_levels.step_derivatives_implicit) + length(scheme.time_levels.step_derivatives_explicit)
    )

    history[:,1] .= ck_0
    idx = 1

    for lag in scheme.time_levels.step_values
        history[:,idx] .= ξ^(-lag) .* ck_0
        idx += 1
    end

    for lag in scheme.time_levels.step_derivatives_implicit
        history[:,idx] .= z_k * ξ^(-lag) .* ck_0
        idx += 1
    end

    for lag in scheme.time_levels.step_derivatives_explicit
        history[:,idx] .= z_k * ξ^(-lag) .* ck_0
        idx += 1
    end

    return ξ, history
end

multistep_integrators_explicit = (
    TimeIntegrators.AB1,
    TimeIntegrators.AB2,
    TimeIntegrators.AB3,
    TimeIntegrators.AB4,
)

@testset "Multi-Step Single-Stage Explicit Integrators" verbose=true begin
    foreach(multistep_integrators_explicit) do scheme
        ξ, history = create_history(scheme, z_k)

        yn = TimeIntegrators.TimeIntegrationSolution(history, scheme, nothing, 0)

        TimeIntegrators.time_integrate!(
            yn,
            linear_advection_ode,
            0.0,
            dt,
        )

        amplification = TimeIntegrators.get_solution(yn) ./ history

        @test maximum(abs.(amplification[84,:] .- ξ)) < 1e-14
    end
end


multistep_integrators_diag_impl = (
    TimeIntegrators.AM0,
    TimeIntegrators.AM1,
    TimeIntegrators.AM2,
    TimeIntegrators.AM3,
    TimeIntegrators.AM4,
    TimeIntegrators.BDF1,
    TimeIntegrators.BDF2,
    TimeIntegrators.BDF3,
    TimeIntegrators.BDF4,
)

@testset "Multi-Step Single-Stage (Diagonally) Implicit Integrators" verbose=true begin
    foreach(multistep_integrators_diag_impl) do scheme
        ξ, history = create_history(scheme, z_k)

        yn = TimeIntegrators.TimeIntegrationSolution(history, scheme, nothing, 0)

        TimeIntegrators.time_integrate!(
            yn,
            implicit_linear_advection_ode,
            0.0,
            dt;
            num_stages=TimeIntegrators.get_num_stages(scheme)
        )

        amplification = TimeIntegrators.get_solution(yn) ./ history

        @test maximum(abs.(amplification[84,:] .- ξ)) < 2e-14
    end
end

# IMEX -------------------------------------------------------------------------------------
function imex_solve(output, x, h, t; num_stages)
    output .= (LinearAlgebra.I - (h* -0.8*A)) \ x
    return nothing
end
function imex_eval(output, c, t; num_stages)
    LinearAlgebra.mul!(output, -0.8*A, c)
    return nothing
end
function imex_expl(output, c, t; num_stages)
    LinearAlgebra.mul!(output, -0.2*A, c)
    return nothing
end
imex_linear_advection_ode = TimeIntegrators.define_imex_ode(
    imex_expl, imex_solve, imex_eval
)

function glm_amplification_imex(scheme, ze, zi)

    AE = Matrix(scheme.A_EX)
    AI = Matrix(scheme.A_IM)

    BE = Matrix(scheme.B_EX)
    BI = Matrix(scheme.B_IM)

    U = Matrix(scheme.U)
    V = Matrix(scheme.V)

    M =V + (ze*BE + zi*BI) * ((LinearAlgebra.I - ze*AE - zi*AI) \ U)

    λ = LinearAlgebra.eigvals(M)

    return λ[argmin(abs.(λ .- 1))]
end

function create_history_imex(scheme, ze, zi)
    ξ = glm_amplification_imex(scheme, ze, zi)

    history = zeros(
        ComplexF64,
        length(grid),
        length(scheme.time_levels.step_values) + length(scheme.time_levels.step_derivatives_implicit) + length(scheme.time_levels.step_derivatives_explicit)
    )

    history[:,1] .= ck_0
    idx = 1

    for lag in scheme.time_levels.step_values
        history[:,idx] .= ξ^(-lag) .* ck_0
        idx += 1
    end

    for lag in scheme.time_levels.step_derivatives_implicit
        history[:,idx] .= zi * ξ^(-lag) .* ck_0
        idx += 1
    end

    for lag in scheme.time_levels.step_derivatives_explicit
        history[:,idx] .= ze * ξ^(-lag) .* ck_0
        idx += 1
    end

    return ξ, history
end

const one_step_imex_integrators = (
    # Single-step, multi-stage
    TimeIntegrators.BACKWARD_FORWARD_EULER,
    TimeIntegrators.MIDPOINT_IMEX,
    TimeIntegrators.RK3_IMEX,
)
@testset "Single-Step Multi-Stage IMEX Integrators" verbose=true begin
    foreach(one_step_imex_integrators) do scheme
        ξ, history = create_history_imex(scheme, 0.2*z_k, 0.8*z_k)

        yn = TimeIntegrators.TimeIntegrationSolution(history, scheme, nothing, 0)

        TimeIntegrators.time_integrate!(
            yn,
            imex_linear_advection_ode,
            0.0,
            dt;
            num_stages=TimeIntegrators.get_num_stages(scheme)
        )

        amplification = TimeIntegrators.get_solution(yn) ./ history

        @test maximum(abs.(amplification[84,:] .- ξ)) < 1e-14
    end
end

const multi_step_imex_integrators = (
    # Multi-step, single-stage
    TimeIntegrators.CNAB2,
    TimeIntegrators.SSSS2,
)

@testset "Multi-Step Single-Stage IMEX Integrators" verbose=true begin
    foreach(multi_step_imex_integrators) do scheme
        ξ, history = create_history_imex(scheme, 0.2*z_k, 0.8*z_k)

        yn = TimeIntegrators.TimeIntegrationSolution(history, scheme, nothing, 0)

        TimeIntegrators.time_integrate!(
            yn,
            imex_linear_advection_ode,
            0.0,
            dt;
            num_stages=TimeIntegrators.get_num_stages(scheme)
        )

        amplification = TimeIntegrators.get_solution(yn) ./ history

        @test maximum(abs.(amplification[84,:] .- ξ)) < 1e-14
    end
end

end

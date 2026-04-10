module TimeIntegrationTimeConvergenceTests

using Mantis
using Test
import LinearAlgebra
import StaticArrays

# We can check the correctness of the implemented schemes by computing their rate of
# convergence and checking if this matches the expected rate. Here, we use a simple ODE
# with one unknown and a time-dependent forcing.
#
# Consider the ODE:
# dy/dt = lambda y - lambda sin(t) + cos(t),  y(t=0) = 1.0
# which has exact solution y(t) = exp(lambda t) + sin(t).

const lambda = -4
const t_final = 1

y_0 = 1.0
function exact_sol(t)
    return exp(lambda * t) + sin(t)
end

# Fully explicit ---------------------------------------------------------------------------
function test_ode_explicit_func!(output, yn, t)
    for n in eachindex(output)
        output[n] = lambda * yn[n] - lambda * sin(t) + cos(t)
    end
    return nothing
end
test_ode_explicit = TimeIntegrators.define_explicit_ode(test_ode_explicit_func!)

function dimplicitSolve(output, x, h, t)
    rhs_mod = -lambda * h * sin(t) + h * cos(t)
    output .= (LinearAlgebra.I - h * lambda) \ (x .+ rhs_mod)
    return nothing
end
test_ode_dimplicit = TimeIntegrators.define_diagonally_implicit_ode(dimplicitSolve)

function dimplicitEvaluate!(output, yn, t)
    for n in eachindex(output)
        output[n] = lambda * yn[n] - lambda * sin(t) + cos(t)
    end
    return nothing
end
test_ode_dimplicit2 = TimeIntegrators.define_diagonally_implicit_ode(
    dimplicitSolve, dimplicitEvaluate!
)

function implicitSolve(output, x, h, t)
    rhs_mod = -lambda * h * sin.(t) + h * cos.(t)
    output .= (LinearAlgebra.I - h .* lambda) \ (x .+ rhs_mod)
    return nothing
end
function implicitEvaluate!(output, yn, t)
    output .= lambda * yn - lambda * sin.(t) + cos.(t)
    return nothing
end
test_ode_implicit = TimeIntegrators.define_implicit_ode(implicitSolve, implicitEvaluate!)


function explicit_imex_function!(output, yn, t)
    for n in axes(output, 1)
        output[n] = - lambda * sin(t) + cos(t)
    end
    return nothing
end
function implicit_imex_function!(output, yn, t)
    for n in axes(output, 1)
        output[n] = lambda * yn[n]
    end
    return nothing
end
test_ode_imex = TimeIntegrators.define_imex_ode(
    explicit_imex_function!,
    (output, x, h, t) -> LinearAlgebra.ldiv!(
        output, LinearAlgebra.lu(LinearAlgebra.I - h * lambda), x
    ),
    implicit_imex_function!,
)

const explicit_integrators = (
    # Single-step, multi-stage
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
    foreach(explicit_integrators) do scheme
        errors = zeros(8)
        dts = zeros(length(errors))
        dt = 0.2
        for i in eachindex(errors)
            y_n = TimeIntegrators.initialize_scheme([y_0], scheme)
            dt = dt / 2
            dts[i] = dt
            for t in 0.0:dt:(t_final - dt)
                TimeIntegrators.time_integrate!(y_n, test_ode_explicit, t, dt)
            end

            errors[i] = abs(exact_sol(t_final) - TimeIntegrators.get_solution(y_n)[1])
        end
        rates = [
            log(errors[i]/errors[i + 1])/(log(dts[i]/dts[i + 1])) for
            i in eachindex(errors)[1:(end - 1)]
        ]

        # The rate is computed to 2 decimal places.
        @test isapprox(rates[end], TimeIntegrators.get_order(scheme), rtol=2e-2)
    end
end

const explicit_multi_step_integrators = (
    # Multi-step, single-stage
    (TimeIntegrators.AB1, nothing),
    (TimeIntegrators.AB2, TimeIntegrators.HEUN2),
    (TimeIntegrators.AB3, TimeIntegrators.HEUN3),
    (TimeIntegrators.AB4, TimeIntegrators.RK4),
)
@testset "Multi-Step Single-Stage Explicit Integrators" verbose = true begin
    foreach(explicit_multi_step_integrators) do (scheme, startup_scheme)
        errors = zeros(8)
        dts = zeros(length(errors))
        dt = 0.2
        for i in eachindex(errors)
            if !isnothing(startup_scheme)
                y_n = TimeIntegrators.initialize_scheme([y_0], scheme, startup_scheme)
            else
                y_n = TimeIntegrators.initialize_scheme([y_0], scheme)
            end
            dt = dt / 2
            dts[i] = dt
            for t in 0.0:dt:(t_final - dt)
                TimeIntegrators.time_integrate!(y_n, test_ode_explicit, t, dt)
            end

            errors[i] = abs(exact_sol(t_final) - TimeIntegrators.get_solution(y_n)[1])
        end
        rates = [
            log(errors[i]/errors[i + 1])/(log(dts[i]/dts[i + 1])) for
            i in eachindex(errors)[1:(end - 1)]
        ]

        # The rate is computed to 2 decimal places.
        @test isapprox(rates[end], TimeIntegrators.get_order(scheme), rtol=1e-2)
    end
end

# Fully implicit ---------------------------------------------------------------------------
const implicit_integrators = (
    # Single-step, multi-stage
    TimeIntegrators.BACKWARD_EULER,
    TimeIntegrators.RADAU_IA_1,
    TimeIntegrators.IMPLICIT_MIDPOINT,
    TimeIntegrators.DIRK2,
    TimeIntegrators.RADAU_IA_3,
    TimeIntegrators.DIRK3,
    TimeIntegrators.DIRK4,
    TimeIntegrators.GAUSS_LEGENDRE_4,
    TimeIntegrators.GAUSS_LEGENDRE_6,
)

@testset "Single-Step Multi-Stage Implicit Integrators" verbose = true begin
    foreach(implicit_integrators) do scheme
        ode = scheme isa TimeIntegrators.DiagonallyImplicit ? test_ode_dimplicit : test_ode_implicit
        errors = zeros(8)
        dts = zeros(length(errors))
        dt = 0.2
        for i in eachindex(errors)
            y_n = TimeIntegrators.initialize_scheme([y_0], scheme)
            dt = dt / 2
            dts[i] = dt
            for t in 0.0:dt:(t_final - dt)
                TimeIntegrators.time_integrate!(y_n, ode, t, dt)
            end

            errors[i] = abs(exact_sol(t_final) - TimeIntegrators.get_solution(y_n)[1])
        end
        rates = [
            log(errors[i]/errors[i + 1])/(log(dts[i]/dts[i + 1])) for
            i in eachindex(errors)[1:(end - 1)]
        ]

        # The rate is computed to 2 decimal places.
        if TimeIntegrators.get_order(scheme) > 4
            # For high-order methods, we reach machine precision so the rate bottoms out.
            # We can pick an earlier rate to check correctness.
            test_rate = rates[3]
        else
            test_rate = rates[end]
        end

        @test isapprox(test_rate, TimeIntegrators.get_order(scheme), rtol=1e-1)
    end
end

# The BDF schemes also test the initialisation of schemes that only require previous
# solutions, but not previous stage derivatives. The AM schemes require implicit stage
# derivatives, but no additional previous solutions. The AB schemes require explicit stage
# derivatives, but no additional previous solutions.
const implicit_multi_step_integrators = (
    # Multi-step, single-stage
    (TimeIntegrators.AM0, nothing),
    (TimeIntegrators.AM1, TimeIntegrators.BACKWARD_EULER),
    (TimeIntegrators.AM2, TimeIntegrators.DIRK2),
    (TimeIntegrators.AM3, TimeIntegrators.DIRK3),
    (TimeIntegrators.AM4, TimeIntegrators.DIRK4),
    (TimeIntegrators.BDF1, nothing),
    (TimeIntegrators.BDF2, TimeIntegrators.BACKWARD_EULER),
    (TimeIntegrators.BDF3, TimeIntegrators.DIRK2),
    (TimeIntegrators.BDF4, TimeIntegrators.DIRK3),
)
@testset "Multi-Step Single-Stage Implicit Integrators" verbose = true begin
    foreach(implicit_multi_step_integrators) do (scheme, startup_scheme)
        errors = zeros(8)
        dts = zeros(length(errors))
        dt = 0.2
        for i in eachindex(errors)
            dt = dt / 2
            if !isnothing(startup_scheme)
                y_n = TimeIntegrators.initialize_scheme([y_0], scheme, startup_scheme)
            else
                y_n = TimeIntegrators.initialize_scheme([y_0], scheme)
            end

            dts[i] = dt
            for t in 0.0:dt:(t_final - dt)
                TimeIntegrators.time_integrate!(y_n, test_ode_dimplicit2, t, dt)
            end

            errors[i] = abs(exact_sol(t_final) - TimeIntegrators.get_solution(y_n)[1])
        end
        rates = [
            log(errors[i]/errors[i + 1])/(log(dts[i]/dts[i + 1])) for
            i in eachindex(errors)[1:(end - 1)]
        ]

        # The rate is computed to 2 decimal places.
        if TimeIntegrators.get_order(scheme) > 4
            # For high-order methods, we reach machine precision so the rate bottoms out.
            # We can pick an earlier rate to check correctness.
            test_rate = rates[4]
        else
            test_rate = rates[end]
        end
        @test isapprox(test_rate, TimeIntegrators.get_order(scheme), rtol=2e-1)
    end
end

# IMEX -------------------------------------------------------------------------------------
const one_step_imex_integrators = (
    # Single-step, multi-stage
    TimeIntegrators.BACKWARD_FORWARD_EULER,
    TimeIntegrators.MIDPOINT_IMEX,
    TimeIntegrators.RK3_IMEX,
)

@testset "Single-Step Multi-Stage IMEX Integrators" verbose = true begin
    foreach(one_step_imex_integrators) do scheme
        errors = zeros(8)
        dts = zeros(length(errors))
        dt = 0.2
        for i in eachindex(errors)
            y_n = TimeIntegrators.initialize_scheme([y_0], scheme)
            dt = dt / 2
            dts[i] = dt
            for t in 0.0:dt:(t_final - dt)
                TimeIntegrators.time_integrate!(y_n, test_ode_imex, t, dt)
            end

            errors[i] = abs(exact_sol(t_final) - TimeIntegrators.get_solution(y_n)[1])
        end

        rates = [
            log(errors[i]/errors[i + 1])/(log(dts[i]/dts[i + 1])) for
            i in eachindex(errors)[1:(end - 1)]
        ]

        # The rate is computed to 2 decimal places.
        @test isapprox(rates[end], TimeIntegrators.get_order(scheme), rtol=1e-2)
    end
end

const multi_step_imex_integrators = (
    # Multi-step, single-stage
    (TimeIntegrators.CNAB2, TimeIntegrators.MIDPOINT_IMEX),
    (TimeIntegrators.SSSS2, TimeIntegrators.MIDPOINT_IMEX),
)

@testset "Multi-Step Single-Stage IMEX Integrators" verbose = true begin
    foreach(multi_step_imex_integrators) do (scheme, startup_scheme)
        errors = zeros(8)
        dts = zeros(length(errors))
        dt = 0.2
        for i in eachindex(errors)
            if !isnothing(startup_scheme)
                y_n = TimeIntegrators.initialize_scheme([y_0], scheme, startup_scheme)
            else
                y_n = TimeIntegrators.initialize_scheme([y_0], scheme)
            end

            dt = dt / 2
            dts[i] = dt
            for t in 0.0:dt:(t_final - dt)
                TimeIntegrators.time_integrate!(y_n, test_ode_imex, t, dt)
            end

            errors[i] = abs(exact_sol(t_final) - TimeIntegrators.get_solution(y_n)[1])
        end

        rates = [
            log(errors[i]/errors[i + 1])/(log(dts[i]/dts[i + 1])) for
            i in eachindex(errors)[1:(end - 1)]
        ]

        # The rate is computed to 2 decimal places.
        @test isapprox(rates[end], TimeIntegrators.get_order(scheme), rtol=1e-2)
    end
end

# Combined multi-step multi-stage ----------------------------------------------------------

# Almost Runge-Kutta methods----------------------------------------------------------------
# Rattenbury, N., 2005. Almost Runge–Kutta methods for stiff  and non-stiff problems.
# Thesis (PhD). The University of Auckland.
# This scheme requires a specialised initialisation, since it needs an estimate of the
# second derivative which is not accounted for in the available initialisations. As a
# result, this scheme is not part of Mantis.
const ARK3 = TimeIntegrators.Explicit(
    StaticArrays.SMatrix{3, 3}(0.0, 1/2, 0.0, 0.0, 0.0, 3/4, 0.0, 0.0, 0.0), # A
    StaticArrays.SMatrix{3, 3}(0.0, 0.0, 3.0, 3/4, 0.0, -3.0, 0.0, 1.0, 2.0), # B
    StaticArrays.SMatrix{3, 3}(1.0, 1.0, 1.0, 1/3, 1/6, 1/4, 1/18, 1/18, 0.0), # U
    StaticArrays.SMatrix{3, 3}(1.0, 0.0, 0.0, 1/4, 0, -2.0, 0.0, 0.0, 0.0), # V
    StaticArrays.SVector(1 / 3, 2 / 3, 1.0),
    TimeIntegrators.TimeLevels(
        [0], # y
        Int[], # Δt G
        [0, 1],  # Δt F and Δt^2 F'
    ),
    3,
)
const explicit_multi_multi_integrators = (ARK3,)
@testset "Multi-Step Multi-Stage Explicit Integrators" verbose = true begin
    foreach(explicit_multi_multi_integrators) do scheme
        errors = zeros(8)
        dts = zeros(length(errors))
        dt = 0.2
        for i in eachindex(errors)
            dt = dt / 2
            yn = zeros(Float64, 1, 3)
            yn[:, 1] .= [y_0]
            @. yn[:, 2] .= lambda * [y_0] * dt - lambda * sin(0.0) * dt + cos(0.0) * dt
            @. yn[:, 3] .= lambda^2 * [y_0] * dt^2 - lambda^2 * cos(0.0) * dt^2 - sin(0.0) * dt^2
            y_n = TimeIntegrators.TimeIntegrationSolution(yn, scheme, nothing, 0)

            dts[i] = dt
            for t in 0.0:dt:(t_final - dt)
                TimeIntegrators.time_integrate!(y_n, test_ode_explicit, t, dt)
            end

            errors[i] = abs(exact_sol(t_final) - TimeIntegrators.get_solution(y_n)[1])
        end
        rates = [
            log(errors[i]/errors[i + 1])/(log(dts[i]/dts[i + 1])) for
            i in eachindex(errors)[1:(end - 1)]
        ]

        # The rate is computed to 2 decimal places.
        @test isapprox(rates[end], TimeIntegrators.get_order(scheme), rtol=1e-2)
    end
end

end

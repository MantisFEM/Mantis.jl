module TimeIntegratorsInferenceTests

import Pkg

using Mantis
using Test
using JET
using LinearAlgebra
using StaticArrays

# Test problem, to have some parameters and equations
const lambda = -4
const t_final = 1

y_0 = 1.0
function exact_sol(t)
    return exp(lambda * t)
end

function test_ode_explicit_func(output, yn, t)
    for n in eachindex(output)
        output[n] = lambda * yn[n]
    end
    return nothing
end

function implicitSolve(output, x, h, t)
    output .= (LinearAlgebra.I - h * lambda) \ x
    return nothing
end
function implicitEvaluate(output, yn, t)
    for n in eachindex(output)
        output[n] = lambda * yn[n]
    end
    return nothing
end

function test_ode_explicit_imex_func(output, yn, t)
    for n in eachindex(output)
        output[n] = 0.5 * lambda * yn[n]
    end
    return nothing
end

# Test functions defining the odes.
@test_opt TimeIntegrators.define_explicit_ode(test_ode_explicit_func)
@test_opt TimeIntegrators.define_diagonally_implicit_ode(implicitSolve)
@test_opt TimeIntegrators.define_diagonally_implicit_ode(implicitSolve, implicitEvaluate)
@test_opt TimeIntegrators.define_implicit_ode(implicitSolve, implicitEvaluate)
@test_opt TimeIntegrators.define_imex_ode(
    test_ode_explicit_imex_func,
    (output, x, h, t) -> LinearAlgebra.ldiv!(
        output, LinearAlgebra.lu(LinearAlgebra.I - 0.5 * h * lambda), x
    ),
)
@test_opt TimeIntegrators.define_imex_ode(
    test_ode_explicit_imex_func,
    (output, x, h, t) -> LinearAlgebra.ldiv!(
        output, LinearAlgebra.lu(LinearAlgebra.I - 0.5 * h * lambda), x
    ),
    test_ode_explicit_imex_func,  # Implicit evaluate is the same as the explicit one here.
)

test_ode_explicit = TimeIntegrators.define_explicit_ode(test_ode_explicit_func)
test_ode_dimplicit = TimeIntegrators.define_diagonally_implicit_ode(
    implicitSolve, implicitEvaluate
)
test_ode_implicit = TimeIntegrators.define_implicit_ode(implicitSolve, implicitEvaluate)
test_ode_imex = TimeIntegrators.define_imex_ode(
    test_ode_explicit_imex_func,  # Explcit evaluation
    (output, x, h, t) -> LinearAlgebra.ldiv!(
        output, LinearAlgebra.lu(LinearAlgebra.I - 0.5 * h * lambda), x
    ),  # Implicit solver
    test_ode_explicit_imex_func,
)

const schemes = (
    # Single-step, multi-stage, Explicit
    (TimeIntegrators.FORWARD_EULER, nothing),
    (TimeIntegrators.EXPLICIT_MIDPOINT, nothing),
    (TimeIntegrators.HEUN2, nothing),
    (TimeIntegrators.RALSTON2, nothing),
    (TimeIntegrators.HEUN3, nothing),
    (TimeIntegrators.RK3, nothing),
    (TimeIntegrators.RALSTON3, nothing),
    (TimeIntegrators.VDHW3, nothing),
    (TimeIntegrators.SSPRK3, nothing),
    (TimeIntegrators.RK4, nothing),
    (TimeIntegrators.RK4_3_8, nothing),
    (TimeIntegrators.RALSTON4, nothing),
    # Multi-step, single-stage, Explicit
    (TimeIntegrators.AB1, nothing),
    (TimeIntegrators.AB2, TimeIntegrators.HEUN2),
    (TimeIntegrators.AB3, TimeIntegrators.HEUN3),
    (TimeIntegrators.AB4, TimeIntegrators.RK4),
    # Single-step, multi-stage, Implicit
    (TimeIntegrators.BACKWARD_EULER, nothing),
    (TimeIntegrators.RADAU_IA_1, nothing),
    (TimeIntegrators.IMPLICIT_MIDPOINT, nothing),
    (TimeIntegrators.CRANK_NICOLSON, nothing),
    (TimeIntegrators.DIRK2, nothing),
    (TimeIntegrators.RADAU_IA_3, nothing),
    (TimeIntegrators.DIRK3, nothing),
    (TimeIntegrators.ESDIRK32, nothing),
    (TimeIntegrators.DIRK4, nothing),
    (TimeIntegrators.GAUSS_LEGENDRE_4, nothing),
    (TimeIntegrators.GAUSS_LEGENDRE_6, nothing),
    # Multi-step, single-stage, Implicit
    (TimeIntegrators.AM0, nothing),
    (TimeIntegrators.AM1, TimeIntegrators.BACKWARD_EULER),
    (TimeIntegrators.AM2, TimeIntegrators.DIRK2),
    (TimeIntegrators.AM3, TimeIntegrators.DIRK3),
    (TimeIntegrators.AM4, TimeIntegrators.DIRK4),
    (TimeIntegrators.BDF1, nothing),
    (TimeIntegrators.BDF2, TimeIntegrators.BACKWARD_EULER),
    (TimeIntegrators.BDF3, TimeIntegrators.DIRK2),
    (TimeIntegrators.BDF4, TimeIntegrators.DIRK3),
    # Single-step, multi-stage, IMEX
    (TimeIntegrators.BACKWARD_FORWARD_EULER, nothing),
    (TimeIntegrators.MIDPOINT_IMEX, nothing),
    (TimeIntegrators.RK3_IMEX, nothing),
    (TimeIntegrators.IMEX331, nothing),
    # Multi-step, single-stage, IMEX
    (TimeIntegrators.CNAB2, TimeIntegrators.MIDPOINT_IMEX),
    (TimeIntegrators.SSSS2, TimeIntegrators.MIDPOINT_IMEX),
)

const dt = 0.2
const t = 0.0

# Test the convenience function
@test_opt TimeIntegrators.butcher_tableau_to_glm( # Explicit RK4
    SMatrix{4, 4}(
        0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0
    ),
    SVector(1 / 6, 1 / 3, 1 / 3, 1 / 6),
    SVector(0.0, 0.5, 0.5, 1.0),
    4,
)
@test_opt TimeIntegrators.butcher_tableau_to_glm( # DiagonallyImplicit DIRK3
    SMatrix{2, 2}(1/2 + sqrt(3)/6, -sqrt(3)/3, 0.0, 1/2+sqrt(3)/6),
    SVector(1 / 2, 1 / 2),
    SVector(1 / 2 + sqrt(3) / 6, 1 / 2 - sqrt(3) / 6),
    3,
)
@test_opt TimeIntegrators.butcher_tableau_to_glm( # Implicit GAUSS_LEGENDRE_4
    SMatrix{2, 2}(1/4, 1/4+sqrt(3)/6, 1/4-sqrt(3)/6, 1/4),
    SVector(1 / 2, 1 / 2),
    SVector(1 / 2 - sqrt(3) / 6, 1 / 2 + sqrt(3) / 6),
    4,
)

foreach(schemes) do (scheme, startup_scheme)
    if isnothing(startup_scheme)
        @test_opt TimeIntegrators.initialise_scheme([y_0], scheme)
        y_n = TimeIntegrators.initialise_scheme([y_0], scheme)
    else
        @test_opt TimeIntegrators.initialise_scheme([y_0], scheme, startup_scheme)
        y_n = TimeIntegrators.initialise_scheme([y_0], scheme, startup_scheme)
    end

    if isa(scheme, TimeIntegrators.Explicit)
        @test_opt TimeIntegrators.time_integrate!(y_n, test_ode_explicit, t, dt)
        TimeIntegrators.time_integrate!(y_n, test_ode_explicit, t, dt)
    elseif isa(scheme, TimeIntegrators.DiagonallyImplicit)
        @test_opt TimeIntegrators.time_integrate!(y_n, test_ode_dimplicit, t, dt)
        TimeIntegrators.time_integrate!(y_n, test_ode_dimplicit, t, dt)
    elseif isa(scheme, TimeIntegrators.Implicit)
        @test_opt TimeIntegrators.time_integrate!(y_n, test_ode_implicit, t, dt)
        TimeIntegrators.time_integrate!(y_n, test_ode_implicit, t, dt)
    elseif isa(scheme, TimeIntegrators.IMEX)
        @test_opt TimeIntegrators.time_integrate!(y_n, test_ode_imex, t, dt)
        TimeIntegrators.time_integrate!(y_n, test_ode_imex, t, dt)
    else
        @warn "Unknown TimeIntegrator type: $(typeof(scheme))"
    end

    # Test the functions applied to a TimeIntegrationSolution
    @test_opt eltype(y_n)
    @test_opt TimeIntegrators.get_num_variables(y_n)
    @test_opt TimeIntegrators.get_solution(y_n)
    @test_opt TimeIntegrators.get_scheme(y_n)
    @test_opt TimeIntegrators.get_startup_scheme(y_n)
    @test_opt TimeIntegrators.get_remaining_startup_steps(y_n)
    @test_opt TimeIntegrators.get_F_allocated(y_n)
    @test_opt TimeIntegrators.get_G_allocated(y_n)

    # Test the functions applied to a scheme (other than integration and startup).
    @test_opt TimeIntegrators.get_num_stages(scheme)
    @test_opt TimeIntegrators.get_num_steps(scheme)
    @test_opt TimeIntegrators.get_order(scheme)
end

end

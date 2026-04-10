module TimeIntegrationConvergenceTests

using Mantis
using Test

# Explicit integrator
@test TimeIntegrators.get_num_steps(TimeIntegrators.RK4) == 1
@test TimeIntegrators.get_num_stages(TimeIntegrators.RK4) == 4
@test typeof(TimeIntegrators.RK4) == TimeIntegrators.Explicit{4, 1, Float64, 16, 4, 1}
@test TimeIntegrators.get_order(TimeIntegrators.RK4) == 4

# Diagonally implicit integrator
@test TimeIntegrators.get_num_steps(TimeIntegrators.DIRK2) == 1
@test TimeIntegrators.get_num_stages(TimeIntegrators.DIRK2) == 2
@test typeof(TimeIntegrators.DIRK2) == TimeIntegrators.DiagonallyImplicit{
    2, 1, Float64, 4, 2, 1
}
@test TimeIntegrators.get_order(TimeIntegrators.DIRK2) == 2

# Implicit integrator
@test TimeIntegrators.get_num_steps(TimeIntegrators.GAUSS_LEGENDRE_6) == 1
@test TimeIntegrators.get_num_stages(TimeIntegrators.GAUSS_LEGENDRE_6) == 3
@test typeof(TimeIntegrators.GAUSS_LEGENDRE_6) == TimeIntegrators.Implicit{
    3, 1, Float64, 9, 3, 1
}
@test TimeIntegrators.get_order(TimeIntegrators.GAUSS_LEGENDRE_6) == 6

# IMEX integrator
@test TimeIntegrators.get_num_steps(TimeIntegrators.CNAB2) == 4
@test TimeIntegrators.get_num_stages(TimeIntegrators.CNAB2) == 1
@test typeof(TimeIntegrators.CNAB2) == TimeIntegrators.IMEX{1, 4, Float64, 1, 4, 16}
@test TimeIntegrators.get_order(TimeIntegrators.CNAB2) == 2

# Incorrect matrices
@test_throws ArgumentError TimeIntegrators.Explicit(
    TimeIntegrators.GAUSS_LEGENDRE_6.A,
    TimeIntegrators.GAUSS_LEGENDRE_6.B,
    TimeIntegrators.GAUSS_LEGENDRE_6.U,
    TimeIntegrators.GAUSS_LEGENDRE_6.V,
    TimeIntegrators.GAUSS_LEGENDRE_6.C,
    TimeIntegrators.GAUSS_LEGENDRE_6.time_levels,
    TimeIntegrators.GAUSS_LEGENDRE_6.order,
)

@test_throws ArgumentError TimeIntegrators.DiagonallyImplicit(
    TimeIntegrators.GAUSS_LEGENDRE_6.A,
    TimeIntegrators.GAUSS_LEGENDRE_6.B,
    TimeIntegrators.GAUSS_LEGENDRE_6.U,
    TimeIntegrators.GAUSS_LEGENDRE_6.V,
    TimeIntegrators.GAUSS_LEGENDRE_6.C,
    TimeIntegrators.GAUSS_LEGENDRE_6.time_levels,
    TimeIntegrators.GAUSS_LEGENDRE_6.order,
)

@test_throws ArgumentError TimeIntegrators.Explicit(
    TimeIntegrators.DIRK2.A,
    TimeIntegrators.DIRK2.B,
    TimeIntegrators.DIRK2.U,
    TimeIntegrators.DIRK2.V,
    TimeIntegrators.DIRK2.C,
    TimeIntegrators.DIRK2.time_levels,
    TimeIntegrators.DIRK2.order,
)

@test_throws ArgumentError TimeIntegrators.IMEX(
    TimeIntegrators.GAUSS_LEGENDRE_6.A,
    TimeIntegrators.GAUSS_LEGENDRE_6.A,
    TimeIntegrators.GAUSS_LEGENDRE_6.B,
    TimeIntegrators.GAUSS_LEGENDRE_6.B,
    TimeIntegrators.GAUSS_LEGENDRE_6.U,
    TimeIntegrators.GAUSS_LEGENDRE_6.V,
    TimeIntegrators.GAUSS_LEGENDRE_6.C,
    TimeIntegrators.GAUSS_LEGENDRE_6.C,
    TimeIntegrators.GAUSS_LEGENDRE_6.time_levels,
    TimeIntegrators.GAUSS_LEGENDRE_6.order,
)

# TimeIntegrationOperators should have at least one Function.
@test_throws ArgumentError TimeIntegrators.TimeIntegrationOperators(
    nothing, nothing, nothing
)
@test_throws ArgumentError TimeIntegrators.TimeIntegrationOperators(
    Union{}(), Union{}(), Union{}()
)



# Setup some ODEs. Then check if we throw an error if the wrong ODE/scheme pairing is used.
const lambda = -4
const t_final = 1

y_0 = 1.0
function exact_sol(t)
    return exp(lambda * t)
end

function test_ode_explicit_func!(output, yn, t)
    for n in eachindex(output)
        output[n] = lambda * yn[n]
    end
    return nothing
end
test_ode_explicit = TimeIntegrators.define_explicit_ode(test_ode_explicit_func!)

function implicitSolve(output, x, h, t)
    output .= (LinearAlgebra.I - h * lambda) \ x
    return nothing
end
test_ode_dimplicit = TimeIntegrators.define_diagonally_implicit_ode(implicitSolve)

function implicitEvaluate!(output, yn, t)
    for n in eachindex(output)
        output[n] = lambda * yn[n]
    end
    return nothing
end
test_ode_implicit = TimeIntegrators.define_implicit_ode(implicitSolve, implicitEvaluate!)

function explicit_imex_function!(output, yn, t)
    for n in axes(output, 1)
        output[n] = 0.5 * lambda * yn[n]
    end
    return nothing
end
function implicit_imex_function!(output, yn, t)
    for n in axes(output, 1)
        output[n] = 0.5 * lambda * yn[n]
    end
    return nothing
end
test_ode_imex = TimeIntegrators.define_imex_ode(
    explicit_imex_function!,
    (output, x, h, t) -> LinearAlgebra.ldiv!(
        output, LinearAlgebra.lu(LinearAlgebra.I - 0.5 * h * lambda), x
    ),
    implicit_imex_function!,
)

y_ne = TimeIntegrators.initialize_scheme([y_0], TimeIntegrators.RK4)
@test_throws ArgumentError TimeIntegrators.time_integrate!(
    y_ne, test_ode_dimplicit, 0.0, 0.1
)
@test_throws ArgumentError TimeIntegrators.time_integrate!(
    y_ne, test_ode_implicit, 0.0, 0.1
)

y_ndi = TimeIntegrators.initialize_scheme([y_0], TimeIntegrators.DIRK2)
@test_throws ArgumentError TimeIntegrators.time_integrate!(
    y_ndi, test_ode_explicit, 0.0, 0.1
)

y_ni = TimeIntegrators.initialize_scheme([y_0], TimeIntegrators.GAUSS_LEGENDRE_6)
@test_throws ArgumentError TimeIntegrators.time_integrate!(
    y_ni, test_ode_explicit, 0.0, 0.1
)

y_nimex = TimeIntegrators.initialize_scheme([y_0], TimeIntegrators.MIDPOINT_IMEX)
@test_throws ArgumentError TimeIntegrators.time_integrate!(
    y_nimex, test_ode_explicit, 0.0, 0.1
)
@test_throws ArgumentError TimeIntegrators.time_integrate!(
    y_nimex, test_ode_dimplicit, 0.0, 0.1
)
@test_throws ArgumentError TimeIntegrators.time_integrate!(
    y_nimex, test_ode_implicit, 0.0, 0.1
)

end

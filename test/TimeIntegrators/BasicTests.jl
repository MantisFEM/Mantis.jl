module TimeIntegrationBasicTests

using Mantis
using Test
using StaticArrays

# Explicit integrator
@test TimeIntegrators.get_num_steps(TimeIntegrators.RK4) == 1
@test TimeIntegrators.get_num_stages(TimeIntegrators.RK4) == 4
@test typeof(TimeIntegrators.RK4) == TimeIntegrators.Explicit{4, 1, Float64, 16, 4, 1}
@test TimeIntegrators.get_order(TimeIntegrators.RK4) == 4

# Diagonally implicit integrators
@test TimeIntegrators.get_num_steps(TimeIntegrators.DIRK2) == 1
@test TimeIntegrators.get_num_stages(TimeIntegrators.DIRK2) == 2
@test typeof(TimeIntegrators.DIRK2) ==
    TimeIntegrators.DiagonallyImplicit{2, 1, Float64, 4, 2, 1}
@test TimeIntegrators.get_order(TimeIntegrators.DIRK2) == 2

@test TimeIntegrators.get_num_steps(TimeIntegrators.CRANK_NICOLSON) == 1
@test TimeIntegrators.get_num_stages(TimeIntegrators.CRANK_NICOLSON) == 2
@test typeof(TimeIntegrators.CRANK_NICOLSON) ==
    TimeIntegrators.DiagonallyImplicit{2, 1, Float64, 4, 2, 1}
@test TimeIntegrators.get_order(TimeIntegrators.CRANK_NICOLSON) == 2

# Implicit integrator
@test TimeIntegrators.get_num_steps(TimeIntegrators.GAUSS_LEGENDRE_6) == 1
@test TimeIntegrators.get_num_stages(TimeIntegrators.GAUSS_LEGENDRE_6) == 3
@test typeof(TimeIntegrators.GAUSS_LEGENDRE_6) ==
    TimeIntegrators.Implicit{3, 1, Float64, 9, 3, 1}
@test TimeIntegrators.get_order(TimeIntegrators.GAUSS_LEGENDRE_6) == 6

# IMEX integrator
@test TimeIntegrators.get_num_steps(TimeIntegrators.CNAB2) == 4
@test TimeIntegrators.get_num_stages(TimeIntegrators.CNAB2) == 1
@test typeof(TimeIntegrators.CNAB2) == TimeIntegrators.IMEX{1, 4, Float64, 1, 4, 16}
@test TimeIntegrators.get_order(TimeIntegrators.CNAB2) == 2

# Correct matrices
function check_equal(scheme1, scheme2)
    if !all(isequal.(scheme1.A, scheme2.A))
        return false
    elseif !all(isequal.(scheme1.B, scheme2.B))
        return false
    elseif !all(isequal.(scheme1.U, scheme2.U))
        return false
    elseif !all(isequal.(scheme1.V, scheme2.V))
        return false
    elseif !all(isequal.(scheme1.C, scheme2.C))
        return false
    elseif !(
        all(isequal.(scheme1.time_levels.step_values, scheme2.time_levels.step_values)) &&
        all(
            isequal.(
                scheme1.time_levels.step_derivatives_implicit,
                scheme2.time_levels.step_derivatives_implicit,
            ),
        ) &&
        all(
            isequal.(
                scheme1.time_levels.step_derivatives_explicit,
                scheme2.time_levels.step_derivatives_explicit,
            ),
        )
    )
        return false
    elseif scheme1.order != scheme2.order
        return false
    end

    return true
end
function check_equal_imex(scheme1, scheme2)
    if !all(isequal.(scheme1.A_IM, scheme2.A_IM))
        return false
    elseif !all(isequal.(scheme1.A_EX, scheme2.A_EX))
        return false
    elseif !all(isequal.(scheme1.B_IM, scheme2.B_IM))
        return false
    elseif !all(isequal.(scheme1.B_EX, scheme2.B_EX))
        return false
    elseif !all(isequal.(scheme1.U, scheme2.U))
        return false
    elseif !all(isequal.(scheme1.V, scheme2.V))
        return false
    elseif !all(isequal.(scheme1.C_IM, scheme2.C_IM))
        return false
    elseif !all(isequal.(scheme1.C_EX, scheme2.C_EX))
        return false
    elseif !(
        all(isequal.(scheme1.time_levels.step_values, scheme2.time_levels.step_values)) &&
        all(
            isequal.(
                scheme1.time_levels.step_derivatives_implicit,
                scheme2.time_levels.step_derivatives_implicit,
            ),
        ) &&
        all(
            isequal.(
                scheme1.time_levels.step_derivatives_explicit,
                scheme2.time_levels.step_derivatives_explicit,
            ),
        )
    )
        return false
    elseif scheme1.order != scheme2.order
        return false
    end

    return true
end

@test check_equal(
    TimeIntegrators.butcher_tableau_to_glm(
        SMatrix{3, 3}(0.0, 1/2, -1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
        SVector(1 / 6, 2 / 3, 1 / 6),
        SVector(0.0, 1 / 2, 1.0),
        3,
    ),
    TimeIntegrators.RK3,
)
@test check_equal(
    TimeIntegrators.Explicit(
        SMatrix{3, 3}(0.0, 1/2, -1.0, 0.0, 0.0, 2.0, 0.0, 0.0, 0.0),
        SMatrix{1, 3}(1 / 6, 2 / 3, 1 / 6),
        ones(SMatrix{3, 1}),
        ones(SMatrix{1, 1}),
        SVector(0.0, 1 / 2, 1.0),
        TimeIntegrators.TimeLevels([0], Int[], Int[]),
        3,
    ),
    TimeIntegrators.RK3,
)

@test check_equal(
    TimeIntegrators.butcher_tableau_to_glm(
        SMatrix{2, 2}(1/2 + 1/(2*sqrt(3)), -1/sqrt(3), 0.0, 1/2+1/(2*sqrt(3))),
        SVector(1 / 2, 1 / 2),
        SVector(1 / 2 + 1/(2*sqrt(3)), 1 / 2 - 1/(2*sqrt(3))),
        3,
    ),
    TimeIntegrators.DIRK3,
)
@test check_equal(
    TimeIntegrators.DiagonallyImplicit(
        SMatrix{2, 2}(1/2 + 1/(2*sqrt(3)), -1/sqrt(3), 0.0, 1/2+1/(2*sqrt(3))),
        SMatrix{1, 2}(1 / 2, 1 / 2),
        ones(SMatrix{2, 1}),
        ones(SMatrix{1, 1}),
        SVector(1 / 2 + 1/(2*sqrt(3)), 1 / 2 - 1/(2*sqrt(3))),
        TimeIntegrators.TimeLevels([0], Int[], Int[]),
        3,
    ),
    TimeIntegrators.DIRK3,
)

@test check_equal(
    TimeIntegrators.butcher_tableau_to_glm(
        SMatrix{2, 2}(1/4, 1/4, -1/4, 5/12), SVector(1 / 4, 3 / 4), SVector(0, 2 / 3), 3
    ),
    TimeIntegrators.RADAU_IA_3,
)
@test check_equal(
    TimeIntegrators.Implicit(
        SMatrix{2, 2}(1/4, 1/4, -1/4, 5/12),
        SMatrix{1, 2}(1 / 4, 3 / 4),
        ones(SMatrix{2, 1}),
        ones(SMatrix{1, 1}),
        SVector(0, 2 / 3),
        TimeIntegrators.TimeLevels([0], Int[], Int[]),
        3,
    ),
    TimeIntegrators.RADAU_IA_3,
)

@test check_equal_imex(
    TimeIntegrators.IMEX(
        SMatrix{2, 2}(0.0, 0.0, 0.0, 1/2),
        SMatrix{2, 2}(0.0, 1/2, 0.0, 0.0),
        SMatrix{1, 2}(0.0, 1.0),
        SMatrix{1, 2}(0.0, 1.0),
        SMatrix{2, 1}(1.0, 1.0),
        SMatrix{1, 1}(1.0),
        SVector(0.0, 1/2),
        SVector(0.0, 1/2),
        TimeIntegrators.TimeLevels([0], Int[], Int[]),
        2,
    ),
    TimeIntegrators.MIDPOINT_IMEX,
)

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

@test_throws ArgumentError TimeIntegrators.Explicit(
    TimeIntegrators.DIRK2.A,
    TimeIntegrators.DIRK2.B,
    TimeIntegrators.DIRK2.U,
    TimeIntegrators.DIRK2.V,
    TimeIntegrators.DIRK2.C,
    TimeIntegrators.DIRK2.time_levels,
    TimeIntegrators.DIRK2.order,
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

# Check the length getters for TimeLevels.
tl1 = TimeIntegrators.TimeLevels([0, 1, 2], [0], [0, 1])
@test TimeIntegrators.get_num_step_values(tl1) == 3
@test TimeIntegrators.get_num_implicit_derivatives(tl1) == 1
@test TimeIntegrators.get_num_explicit_derivatives(tl1) == 2

tl2 = TimeIntegrators.TimeLevels(Int[], Int[], Int[])
@test TimeIntegrators.get_num_step_values(tl2) == 0
@test TimeIntegrators.get_num_implicit_derivatives(tl2) == 0
@test TimeIntegrators.get_num_explicit_derivatives(tl2) == 0

tl3 = TimeIntegrators.TimeLevels([6, -8], [2, 4], [3])
@test TimeIntegrators.get_num_step_values(tl3) == 2
@test TimeIntegrators.get_num_implicit_derivatives(tl3) == 2
@test TimeIntegrators.get_num_explicit_derivatives(tl3) == 1

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

test_ode_ee = TimeIntegrators.TimeIntegrationOperators(
    test_ode_explicit_func!, nothing, implicitEvaluate!
)

@test_throws ArgumentError TimeIntegrators.TimeIntegrationOperators(
    nothing, nothing, implicitEvaluate!
)

y_ne = TimeIntegrators.initialise_scheme([y_0], TimeIntegrators.RK4)
@test_throws ArgumentError TimeIntegrators.time_integrate!(
    y_ne, test_ode_dimplicit, 0.0, 0.1
)
@test_throws ArgumentError TimeIntegrators.time_integrate!(
    y_ne, test_ode_implicit, 0.0, 0.1
)

y_ndi = TimeIntegrators.initialise_scheme([y_0], TimeIntegrators.DIRK2)
@test_throws ArgumentError TimeIntegrators.time_integrate!(
    y_ndi, test_ode_explicit, 0.0, 0.1
)

y_ni = TimeIntegrators.initialise_scheme([y_0], TimeIntegrators.GAUSS_LEGENDRE_6)
@test_throws ArgumentError TimeIntegrators.time_integrate!(
    y_ni, test_ode_explicit, 0.0, 0.1
)
@test_throws ArgumentError TimeIntegrators.time_integrate!(
    y_ni, test_ode_dimplicit, 0.0, 0.1
)
@test_throws ArgumentError TimeIntegrators.time_integrate!(y_ni, test_ode_ee, 0.0, 0.1)

y_nimex = TimeIntegrators.initialise_scheme([y_0], TimeIntegrators.MIDPOINT_IMEX)
@test_throws ArgumentError TimeIntegrators.time_integrate!(
    y_nimex, test_ode_explicit, 0.0, 0.1
)
@test_throws ArgumentError TimeIntegrators.time_integrate!(
    y_nimex, test_ode_dimplicit, 0.0, 0.1
)
@test_throws ArgumentError TimeIntegrators.time_integrate!(
    y_nimex, test_ode_implicit, 0.0, 0.1
)
@test_throws ArgumentError TimeIntegrators.time_integrate!(y_nimex, test_ode_ee, 0.0, 0.1)

# Provide a startup-scheme when not needed, and vice versa
@test_throws ArgumentError TimeIntegrators.initialise_scheme([y_0], TimeIntegrators.BDF2)
@test_throws ArgumentError TimeIntegrators.initialise_scheme(
    [y_0], TimeIntegrators.DIRK3, TimeIntegrators.DIRK2
)

# Get the startup scheme
y_nbdf3 = TimeIntegrators.initialise_scheme(
    [y_0], TimeIntegrators.BDF3, TimeIntegrators.DIRK3
)
@test TimeIntegrators.get_startup_scheme(y_nbdf3) == TimeIntegrators.DIRK3

# Copying integrate should do the same as the non-copying one.
function check_equal_sol(sol1, sol2)
    if !(sol1.N == sol2.N)
        return false
    elseif !all(isequal.(sol1.solution, sol2.solution))
        return false
    elseif !check_equal(sol1.scheme, sol2.scheme)
        return false
    elseif !(isnothing(sol1.startup_scheme) && isnothing(sol1.startup_scheme))
        return false
    elseif !(sol1.remaining_startup_steps == sol2.remaining_startup_steps)
        return false
    elseif !all(isequal.(sol1.solution_allocated, sol2.solution_allocated))
        return false
    elseif !all(isequal.(sol1.F_allocated, sol2.F_allocated))
        return false
    elseif !all(isequal.(sol1.G_allocated, sol2.G_allocated))
        return false
    elseif !(isnothing(sol1.startup_solution) && isnothing(sol1.startup_solution))
        return false
    elseif !all(isequal.(sol1.stage_values, sol2.stage_values))
        return false
    elseif !all(isequal.(sol1.temp_var, sol2.temp_var))
        return false
    end

    return true
end

y_ne = TimeIntegrators.initialise_scheme([y_0], TimeIntegrators.RK4)
y_ne2 = TimeIntegrators.time_integrate(y_ne, test_ode_explicit, 0.0, 0.1)
TimeIntegrators.time_integrate!(y_ne, test_ode_explicit, 0.0, 0.1)
@test check_equal_sol(y_ne2, y_ne)

end

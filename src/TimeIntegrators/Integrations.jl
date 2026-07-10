"""
    time_integrate(
        y_n::TimeIntegrationSolution,
        ode::TimeIntegrationOperators,
        t::Float64,
        dt::Float64;
        kwargs...,
    )

Perform a single time integration step using the given time integration scheme and ODE
system operators. Creates a copy of the solution when called.

# See also
[`time_integrate!`](@ref)

# Arguments
- `y_n::TimeIntegrationSolution`: The current solution vector.
- `ode::TimeIntegrationOperators`: The ODE system operators.
- `t::Float64`: The current time.
- `dt::Float64`: The time step.
- `kwargs...`: Additional arguments passed to the ODE system.

# Returns
- `TimeIntegrationSolution`: The updated solution vector after one time step.
"""
function time_integrate(
    y_n::TimeIntegrationSolution,
    ode::TimeIntegrationOperators,
    t::Float64,
    dt::Float64;
    kwargs...,
)
    y_n = deepcopy(y_n)
    time_integrate!(y_n, ode, t, dt; kwargs...)
    return y_n
end

"""
    time_integrate!(
        y_n::TimeIntegrationSolution,
        ode::TimeIntegrationOperators,
        t::Float64,
        dt::Float64;
        kwargs...,
    )

Perform a single, in-place time integration step using the given time integration scheme
and ODE system operators.

# See also
[`time_integrate`](@ref)

# Arguments
- `y_n::TimeIntegrationSolution`: The current solution vector.
- `ode::TimeIntegrationOperators`: The ODE system operators.
- `t::Float64`: The current time.
- `dt::Float64`: The time step.
- `kwargs...`: Additional arguments passed to the ODE system.

# Returns (in-place)
- `TimeIntegrationSolution`: The updated solution vector after one time step.
"""
function time_integrate!(
    y_n::TimeIntegrationSolution{T, S},
    ode::TimeIntegrationOperators,
    t::Float64,
    dt::Float64;
    kwargs...,
) where {T, S <: AbstractTimeIntegrator}
    remaining_startup_steps = get_remaining_startup_steps(y_n)
    if remaining_startup_steps > 0
        scheme = get_scheme(y_n)
        tl = get_time_levels(scheme)
        num_steps = get_num_step_values(tl)
        num_G = get_num_implicit_derivatives(tl)
        num_F = get_num_explicit_derivatives(tl)

        startup_scheme = get_startup_scheme(y_n)
        y_n_startup = get_startup_solution(y_n)

        # We do have to advance the solution using the startup scheme, to ensure that the
        # solution remains at the expected time level, even if some of the startup steps
        # are only needed to initialise the step derivatives.
        ynm1 = get_solution(y_n_startup)
        _time_integrate!(y_n_startup, startup_scheme, ode, t, dt; kwargs...)

        # Then we initialise the required solutions and stage derivatives.
        #
        # Note that the layout of y_n.solution should be
        # [
        # y_n,
        # y_{n-1},
        # ...,
        # y_{n-num_steps+1}
        # \Delta t G_n,
        # \Delta t G_{n-1},
        # ...,
        # \Delta t G_{n-num_G+1},
        # \Delta t F_n,
        # \Delta t F_{n-1},
        # ...,
        # \Delta t F_{n-num_F+1},
        # ]
        #
        # The solution vector is filled from the back (remaining_startup_steps is
        # decreasing) to comply with this layout.
        if num_steps > 0 && num_steps >= remaining_startup_steps
            index_steps = remaining_startup_steps
            y_n.solution[:, index_steps] = get_solution(y_n_startup)
        end
        if num_G > 0 && num_G >= remaining_startup_steps
            index_implicit = num_steps + remaining_startup_steps
            ode.implicitEvaluate!(
                view(y_n.solution, :, index_implicit),
                get_solution(y_n_startup),
                t+dt;
                kwargs...,
            )
            y_n.solution[:, index_implicit] .*= dt
        end
        if num_F > 0 && num_F >= remaining_startup_steps
            index_explicit = num_steps + num_G + remaining_startup_steps
            ode.explicitEvaluate!(view(y_n.solution, :, index_explicit), ynm1, t; kwargs...)
            y_n.solution[:, index_explicit] .*= dt
        end

        t += dt
        y_n.remaining_startup_steps = y_n.remaining_startup_steps - 1

        return y_n
    else
        _time_integrate!(y_n, get_scheme(y_n), ode, t, dt; kwargs...)

        return y_n
    end
end

function time_integrate!(
    y_n::TimeIntegrationSolution{T, Nothing}, # No startup scheme
    ode::TimeIntegrationOperators,
    t::Float64,
    dt::Float64;
    kwargs...,
) where {T}
    return _time_integrate!(y_n, get_scheme(y_n), ode, t, dt; kwargs...)
end

############################################################################################
##                              Type-specific implementations                             ##
############################################################################################

# First, we create the integrate methods that error when the TimeIntegrationOperators do
# not have the right functions initialised. This allows us to throw more precise errors.
function _time_integrate!(
    y_n::TimeIntegrationSolution,
    scheme::Explicit{num_stages, num_steps},
    ode::TimeIntegrationOperators{Nothing, IS, IE},
    t::Float64,
    dt::Float64;
    kwargs...,
) where {num_steps, num_stages, IS, IE}
    return throw(
        ArgumentError(
            LazyString(
                "You are trying to use an Explicit scheme but have provided an ",
                "ode (in a TimeIntegrationOperators struct) which does not have an ",
                "explicitEvaluate! function.",
            ),
        ),
    )
end

function _time_integrate!(
    y_n::TimeIntegrationSolution,
    scheme::DiagonallyImplicit{num_stages, num_steps},
    ode::TimeIntegrationOperators{EE, Nothing, IE},
    t::Float64,
    dt::Float64;
    kwargs...,
) where {num_steps, num_stages, EE, IE}
    return throw(
        ArgumentError(
            LazyString(
                "You are trying to use a DiagonallyImplicit scheme but have provided an ",
                "ode (in a TimeIntegrationOperators struct) which does not have an ",
                "implicitSolve! function.",
            ),
        ),
    )
end

function _time_integrate!(
    y_n::TimeIntegrationSolution,
    scheme::Implicit{num_stages, num_steps},
    ode::TimeIntegrationOperators{EE, Nothing, IE},
    t::Float64,
    dt::Float64;
    kwargs...,
) where {num_steps, num_stages, EE, IE}
    return throw(
        ArgumentError(
            LazyString(
                "You are trying to use an Implicit scheme but have provided an ",
                "ode (in a TimeIntegrationOperators struct) which does not have an ",
                "implicitSolve! function.",
            ),
        ),
    )
end

function _time_integrate!(
    y_n::TimeIntegrationSolution,
    scheme::Implicit{num_stages, num_steps},
    ode::TimeIntegrationOperators{EE, IS, Nothing},
    t::Float64,
    dt::Float64;
    kwargs...,
) where {num_steps, num_stages, EE, IS}
    return throw(
        ArgumentError(
            LazyString(
                "You are trying to use an Implicit scheme but have provided an ",
                "ode (in a TimeIntegrationOperators struct) which does not have an ",
                "implicitEvaluate! function.",
            ),
        ),
    )
end

function _time_integrate!(
    y_n::TimeIntegrationSolution,
    scheme::Implicit{num_stages, num_steps},
    ode::TimeIntegrationOperators{EE, Nothing, Nothing},
    t::Float64,
    dt::Float64;
    kwargs...,
) where {num_steps, num_stages, EE}
    return throw(
        ArgumentError(
            LazyString(
                "You are trying to use an Implicit scheme but have provided an ",
                "ode (in a TimeIntegrationOperators struct) which does not have an ",
                "implicitSolve! nor an implicitEvaluate! function.",
            ),
        ),
    )
end

function _time_integrate!(
    y_n::TimeIntegrationSolution,
    scheme::IMEX{num_stages, num_steps},
    ode::TimeIntegrationOperators{Nothing, IS, IE},
    t::Float64,
    dt::Float64;
    kwargs...,
) where {num_steps, num_stages, IS, IE}
    return throw(
        ArgumentError(
            LazyString(
                "You are trying to use an IMEX scheme but have provided an ",
                "ode (in a TimeIntegrationOperators struct) which does not have an ",
                "explicitEvaluate! function.",
            ),
        ),
    )
end

function _time_integrate!(
    y_n::TimeIntegrationSolution,
    scheme::IMEX{num_stages, num_steps},
    ode::TimeIntegrationOperators{EE, Nothing, IE},
    t::Float64,
    dt::Float64;
    kwargs...,
) where {num_steps, num_stages, EE, IE}
    return throw(
        ArgumentError(
            LazyString(
                "You are trying to use an IMEX scheme but have provided an ",
                "ode (in a TimeIntegrationOperators struct) which does not have an ",
                "implicitSolve! function.",
            ),
        ),
    )
end

# Now we can define the integrators themselves, without having to worry about the ode
# having the required functions.

"""
    _time_integrate!(
        y_n::TimeIntegrationSolution,
        scheme::Explicit{num_stages, num_steps},
        ode::TimeIntegrationOperators,
        t::Float64,
        dt::Float64;
        kwargs...,
    ) where {num_steps, num_stages}

Perform a single time integration step using the given Explicit time integration scheme and
ODE system operators. Implements algorithm 1 of [Vos2011](@cite) specifically for when the
integrator is explicit.

# Arguments
- `y_n::TimeIntegrationSolution`: The current solution vector.
- `scheme::Explicit{num_stages, num_steps}`: The Explicit time integration scheme.
- `ode::TimeIntegrationOperators`: The ODE system operators.
- `t::Float64`: The current time.
- `dt::Float64`: The time step.
- `kwargs...`: Additional arguments passed to the ODE system.

# Returns (in-place)
- `TimeIntegrationSolution{num_steps}`: The updated solution vector after one time step.
"""
function _time_integrate!(
    y_n::TimeIntegrationSolution,
    scheme::Explicit{num_stages, num_steps},
    ode::TimeIntegrationOperators,
    t::Float64,
    dt::Float64;
    kwargs...,
) where {num_steps, num_stages}
    T = eltype(y_n)
    N = get_num_variables(y_n)
    y_nm1 = get_solution(y_n)

    # Get pre-allocated scratch spaces.
    solution_allocated = get_solution_allocated(y_n)
    solution_allocated .= zero(T)
    stage_values = get_stage_allocated(y_n)
    F_allocated = get_F_allocated(y_n)
    F_allocated .= zero(T)

    @inbounds for i in 1:num_stages
        stage_values .= zero(T)
        # Calculate the stage value Yi
        for k in 1:(i - 1)
            for n in 1:N
                stage_values[n] += F_allocated[n, k] * scheme.A[i, k] * dt
            end
        end
        for j in 1:num_steps
            for n in 1:N
                stage_values[n] += scheme.U[i, j] * y_nm1[n, j]
            end
        end

        # Calculate the stage derivative Fᵢ
        ode.explicitEvaluate!(
            view(F_allocated, :, i), stage_values, t + scheme.C[i] * dt; kwargs...
        )
    end

    # Optimisation when both the last stages are the same as the first step, so we can skip
    # the calculation of the final solution
    start_index = 1
    if scheme.U[end, :] == scheme.V[1, :] && scheme.A[end, :] == scheme.B[1, :]
        for n in 1:N
            solution_allocated[n, 1] = stage_values[n]
        end
        start_index = 2
    end

    @inbounds for i in start_index:num_steps
        for k in 1:num_stages
            for n in 1:N
                solution_allocated[n, i] += F_allocated[n, k] * scheme.B[i, k] * dt
            end
        end
        for k in 1:num_steps
            for n in 1:N
                solution_allocated[n, i] += y_nm1[n, k] * scheme.V[i, k]
            end
        end
    end

    # swich the pointers of y_n.solution and y_n.solution_allocated
    y_n.solution, y_n.solution_allocated = y_n.solution_allocated, y_n.solution

    return y_n
end

"""
    _time_integrate!(
        y_n::TimeIntegrationSolution,
        scheme::DiagonallyImplicit{num_stages, num_steps},
        ode::TimeIntegrationOperators,
        t::Float64,
        dt::Float64;
        kwargs...,
    ) where {num_steps, num_stages}

Perform a single time integration step using the given DiagonallyImplicit time integration
scheme and ODE system operators. Implements algorithm 1 of [Vos2011](@cite) specifically
for when the integrator in diagonally implicit.

# Arguments
- `y_n::TimeIntegrationSolution{num_steps}`: The current solution vector.
- `scheme::DiagonallyImplicit{num_stages,num_steps}`: The DiagonallyImplicit time integration scheme.
- `ode::TimeIntegrationOperators`: The ODE system operators.
- `t::Float64`: The current time.
- `dt::Float64`: The time step.
- `kwargs...`: Additional arguments passed to the ODE system.

# Returns (in-place)
- `TimeIntegrationSolution{num_steps}`: The updated solution vector after one time step.
"""
function _time_integrate!(
    y_n::TimeIntegrationSolution,
    scheme::DiagonallyImplicit{num_stages, num_steps},
    ode::TimeIntegrationOperators,
    t::Float64,
    dt::Float64;
    kwargs...,
) where {num_steps, num_stages}
    T = eltype(y_n)
    N = get_num_variables(y_n)
    y_nm1 = get_solution(y_n)

    # Get pre-allocated scratch spaces.
    solution_allocated = get_solution_allocated(y_n)
    solution_allocated .= zero(T)
    temp_var = get_temp_var(y_n)
    G = get_G_allocated(y_n)
    G .= zero(T)

    @inbounds for i in 1:num_stages
        # calculate the temporary value xᵢ
        temp_var .= zero(T)
        for k in 1:(i - 1)
            for n in 1:N
                temp_var[n] += G[n, k] * scheme.A[i, k] * dt
            end
        end
        for j in 1:num_steps
            for n in 1:N
                temp_var[n] += scheme.U[i, j] * y_nm1[n, j]
            end
        end

        # Calculate the stage value Yᵢ
        # solve (Yᵢ - aᵢᵢᴵᴹ * h * g(Yᵢ,t)) = xᵢ
        ode.implicitSolve!(
            y_n.stage_values, temp_var, scheme.A[i, i] * dt, t + scheme.C[i] * dt; kwargs...
        )
        # Calculate the stage derivative Gᵢ
        let a_ii = scheme.A[i, i]
            if !iszero(a_ii)
                for n in 1:N
                    G[n, i] = (y_n.stage_values[n] - temp_var[n]) / (a_ii * dt)
                end
            else
                ode.implicitEvaluate!(
                    view(G, :, i), y_n.stage_values, t + scheme.C[i] * dt; kwargs...
                )
            end
        end
    end

    # Optimisation when both the last stages are the same as the first step, so we can skip
    # the calculation of the final solution
    start_index = 1
    if scheme.U[end, :] == scheme.V[1, :] && scheme.A[end, :] == scheme.B[1, :]
        for n in 1:N
            solution_allocated[n, 1] = y_n.stage_values[n]
        end
        start_index = 2
    end

    @inbounds for i in start_index:num_steps
        for k in 1:num_stages
            for n in 1:N
                solution_allocated[n, i] += G[n, k] * scheme.B[i, k] * dt
            end
        end
        for k in 1:num_steps
            for n in 1:N
                solution_allocated[n, i] += y_nm1[n, k] * scheme.V[i, k]
            end
        end
    end

    y_n.solution, y_n.solution_allocated = y_n.solution_allocated, y_n.solution

    return y_n
end

"""
    _time_integrate!(
        y_n::TimeIntegrationSolution,
        scheme::Implicit{num_stages, num_steps},
        ode::TimeIntegrationOperators,
        t::Float64,
        dt::Float64;
        kwargs...,
    ) where {num_steps, num_stages}

Perform a single time integration step using the given Implicit time integration scheme and
ODE system operators.

# Arguments
- `y_n::TimeIntegrationSolution{num_steps}`: The current solution vector.
- `scheme::Implicit{num_stages,num_steps}`: The Implicit time integration scheme.
- `ode::TimeIntegrationOperators`: The ODE system operators.
- `t::Float64`: The current time.
- `dt::Float64`: The time step.
- `kwargs...`: Additional arguments passed to the ODE system.

# Returns (in-place)
- `TimeIntegrationSolution{num_steps}`: The updated solution vector after one time step.
"""
function _time_integrate!(
    y_n::TimeIntegrationSolution,
    scheme::Implicit{num_stages, num_steps},
    ode::TimeIntegrationOperators,
    t::Float64,
    dt::Float64;
    kwargs...,
) where {num_steps, num_stages}
    T = eltype(y_n)
    N = get_num_variables(y_n)
    y_nm1 = get_solution(y_n)

    # Get pre-allocated scratch spaces.
    solution_allocated = get_solution_allocated(y_n)
    solution_allocated .= zero(T)
    xi = vec(reduce(vcat, y_nm1 for i in 1:num_stages))
    Y = reduce(vcat, y_n.stage_values for i in 1:num_stages)

    # Setup a time variable to know at what time instant each stage is computed.
    tis = SVector((t + scheme.C[i] * dt for i in 1:num_stages)...)

    ode.implicitSolve!(Y, xi, scheme.A * dt, tis; kwargs...)

    allG = Vector{T}(undef, N*num_stages)
    ode.implicitEvaluate!(allG, Y, tis; kwargs...)
    @inbounds for i in 1:num_steps
        for k in 1:num_stages
            for n in 1:N
                solution_allocated[n, i] += allG[(k - 1) * N + n] * scheme.B[i, k] * dt
            end
        end
        for k in 1:num_steps
            for n in 1:N
                solution_allocated[n, i] += y_nm1[n, k] * scheme.V[i, k]
            end
        end
    end

    y_n.solution, y_n.solution_allocated = y_n.solution_allocated, y_n.solution

    return y_n
end

"""
    _time_integrate!(
        y_n::TimeIntegrationSolution,
        scheme::IMEX{num_stages, num_steps},
        ode::TimeIntegrationOperators,
        t::Float64,
        dt::Float64;
        kwargs...,
    ) where {num_steps, num_stages}

Perform a single time integration step using the given IMEX time integration scheme and ODE
system operators. Implements algorithm 1 of [Vos2011](@cite) in full.

# Arguments
- `y_n::TimeIntegrationSolution{num_steps}`: The current solution vector.
- `scheme::IMEX{num_stages,num_steps}`: The IMEX time integration scheme.
- `ode::TimeIntegrationOperators`: The ODE system operators.
- `t::Float64`: The current time.
- `dt::Float64`: The time step.
- `kwargs...`: Additional arguments passed to the ODE system.

# Returns (in-place)
- `TimeIntegrationSolution{num_steps}`: The updated solution vector after one time step.
"""
function _time_integrate!(
    y_n::TimeIntegrationSolution,
    scheme::IMEX{num_stages, num_steps},
    ode::TimeIntegrationOperators,
    t::Float64,
    dt::Float64;
    kwargs...,
) where {num_steps, num_stages}
    T = eltype(y_n)
    N = get_num_variables(y_n)
    y_nm1 = get_solution(y_n)

    # Get pre-allocated scratch spaces.
    solution_allocated = get_solution_allocated(y_n)
    solution_allocated .= zero(T)
    stage_values = get_stage_allocated(y_n)
    F_allocated = get_F_allocated(y_n)
    F_allocated .= zero(T)
    temp_var = get_temp_var(y_n)
    G = get_G_allocated(y_n)
    G .= zero(T)

    @inbounds for i in 1:num_stages
        # calculate the temporary value xᵢ
        temp_var .= zero(T)
        for k in 1:(i - 1)
            for n in 1:N
                temp_var[n] += G[n, k] * scheme.A_IM[i, k] * dt
                temp_var[n] += F_allocated[n, k] * scheme.A_EX[i, k] * dt
            end
        end
        for j in 1:num_steps
            for n in 1:N
                temp_var[n] += scheme.U[i, j] * y_nm1[n, j]
            end
        end

        # Calculate the stage value Yᵢ
        # solve (Yᵢ - aᵢᵢᴵᴹ * h * g(Yᵢ,t)) = xᵢ
        ode.implicitSolve!(
            stage_values,
            temp_var,
            scheme.A_IM[i, i] * dt,
            t + scheme.C_IM[i] * dt;
            kwargs...,
        )

        # Calculate the stage derivative Fᵢ
        ode.explicitEvaluate!(
            view(F_allocated, :, i), stage_values, t + scheme.C_EX[i] * dt; kwargs...
        )

        # Calculate the stage derivative Gᵢ
        let a_ii = scheme.A_IM[i, i]
            if !iszero(a_ii)
                for n in 1:N
                    G[n, i] = (stage_values[n] - temp_var[n]) / (a_ii * dt)
                end
            else
                ode.implicitEvaluate!(
                    view(G, :, i), stage_values, t + scheme.C_IM[i] * dt; kwargs...
                )
            end
        end
    end

    start_index = 1
    if scheme.U[end, :] == scheme.V[1, :] &&
        scheme.A_IM[end, :] == scheme.B_IM[1, :] &&
        scheme.A_EX[end, :] == scheme.B_EX[1, :]
        solution_allocated[:, 1] .= stage_values
        start_index = 2
    end

    @inbounds for i in start_index:num_steps
        for k in 1:num_stages
            for n in 1:N
                solution_allocated[n, i] += G[n, k] * scheme.B_IM[i, k] * dt
                solution_allocated[n, i] += F_allocated[n, k] * scheme.B_EX[i, k] * dt
            end
        end
        for k in 1:num_steps
            for n in 1:N
                solution_allocated[n, i] += y_nm1[n, k] * scheme.V[i, k]
            end
        end
    end

    y_n.solution, y_n.solution_allocated = y_n.solution_allocated, y_n.solution

    return y_n
end

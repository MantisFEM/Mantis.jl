# # Three-body problem

# ## [Background knowledge](@id ThreeBodyBackground)
# The [three-body problem](https://en.wikipedia.org/wiki/Three-body_problem) is a classical
# physics problem where three bodies are orbiting each other and are thus interacting based
# on Newton's law of universal gravitation. From a computational point of view, the goal is
# to compute the trajecteries of all three bodies given the initial position and velocity
# of each body.
#
# The problem is an application of the Newton's second law and Newton's law of universal
# gravitation, that is, we can define the following system of equations:
# ```math
# \begin{cases}
#   \frac{\partial^2 x_1}{\partial t^2} = -Gm_2 \frac{\mathbf{x}_1 - \mathbf{x}_2}{||\mathbf{x}_1 - \mathbf{x}_2||^3} -Gm_3 \frac{\mathbf{x}_1 - \mathbf{x}_3}{||\mathbf{x}_1 - \mathbf{x}_3||^3}\;, \\
#   \frac{\partial^2 x_2}{\partial t^2} = -Gm_3 \frac{\mathbf{x}_2 - \mathbf{x}_3}{||\mathbf{x}_1 - \mathbf{x}_3||^3} -Gm_1 \frac{\mathbf{x}_2 - \mathbf{x}_1}{||\mathbf{x}_2 - \mathbf{x}_1||^3}\;, \\
#   \frac{\partial^2 x_3}{\partial t^2} = -Gm_1 \frac{\mathbf{x}_3 - \mathbf{x}_1}{||\mathbf{x}_1 - \mathbf{x}_1||^3} -Gm_2 \frac{\mathbf{x}_3 - \mathbf{x}_2}{||\mathbf{x}_3 - \mathbf{x}_2||^3}\;,
# \end{cases}
# ```
# where ``\mathbf{x}_i`` is the position of the ``i``-th body, and G is the gravitational
# constant (which will be set to ``1`` below).
#
# Instead of using one second order ODE for each body, we can reformulate the problem in
# terms of two ODEs for each body as
# ```math
# \begin{cases}
#   \frac{\partial v_i}{\partial t} = -G \sum_{j=1, j \neq i}^3 \frac{\mathbf{x}_i - \mathbf{x}_j}{||\mathbf{x}_i - \mathbf{x}_j||^3}\; \\
#   \frac{\partial x_i}{\partial t} = v_i\;.
# \end{cases}
# ```
# This system of two ODEs will be used in the implementation below. In this example, we
# will consider a 2D version of this problem.

# ## [Implementation](@id ThreeBodyImplementation)
# In this section, we will implement the above three-body problem using the
# `TimeIntegrators` module within `Mantis`.
#
# ### [Packages](@id ThreeBodyPackages)
# In this example, we will use `Mantis` for the integration, `GLMakie` to create an
# animation, and `Printf` to easily create a nicely formatted title for the animation. So,
# we are `using` these three packages.

using Mantis
using GLMakie
using Printf

# ### [Test case](@id ThreeBodyTestCase)
# In this example, we will consider the 'figure 8' periodic solution as described in
# [Chenciner2000](@cite). This means that the three bodies each have the same mass (we use
# the case where the mass is 1), and we define the initial positions and velocities as
# shown in the code. This solution is part of the Broucke–Hénon–Hadjidemetriou family.
m1 = 1.0
m2 = 1.0
m3 = 1.0

## x₀ = [x₁, y₁, x₂, y₂, x₃, y₃] and similarly for v₀.
x₀ = [0.97000436, -0.24308753, -0.97000436, 0.24308753, 0.0, 0.0]
v₀ = [-0.4662036850, -0.4323657300, -0.4662036850, -0.4323657300, 0.93240737, 0.86473146]

# ### [Setting up the ODE](@id ThreeBodyODESetup)
# The ODEs as described in the [background section](@ref ThreeBodyBackground) need to be
# implemented. We can do this by creating the `get_forces` function, which will compute the
# gravitation force. We can then define the forcing functions for the position and velocity
# as `f_position` and `f_velocity`. The latter computes and returns the force, while the
# former returns the velocity. Note that both forcing functions take in a keyword argument,
# as the forces do not depend on ``t`` nor on the variable in the ODE itself.
function get_forces!(force, x)
    ## x = [x1, y1, x2, y2, x3, y3]
    force .= zero(eltype(force))
    for i in 1:2:5
        for j in 1:2:5
            if i != j
                r = sqrt((x[i] - x[j])^2 + (x[i + 1] - x[j + 1])^2)
                force[i] += (x[j] - x[i]) / r^3 ## x component
                force[i + 1] += (x[j + 1] - x[i + 1]) / r^3 ## y component
            end
        end
    end
    return force
end

function f_velocity(output, vel, t; pos)
    return get_forces!(output, pos)
end
function f_position(output, pos, t; vel)
    for i in eachindex(output, vel)
        output[i] = vel[i]
    end
    return output
end

# With these forcing functions, we can define the ODEs by simply calling the
# [`Mantis.TimeIntegrators.define_explicit_ode`](@ref) function.
ode_velocity = TimeIntegrators.define_explicit_ode(f_velocity)
ode_position = TimeIntegrators.define_explicit_ode(f_position)

# ### [Time integration](@id ThreeBodyTimeIntegration)
# We can now setup the time integrator of our choosing. Since we are evolving two ODEs at
# the same time, and we would like to ensure that we preserve the Hamiltonian of the
# system (see [the Wikipedia page](https://en.wikipedia.org/wiki/Three-body_problem) for
# the Hamiltonian), we will solve the two ODEs in sequence. Using the forward Euler method
# for each ODE while using the result from the first ODE, we will end up with the so-called
# symplectic Euler method.
#
# To make this work in `Mantis`, we setup the two solutions `x_n` and `v_n` using the
# previously defined initial conditions and using the forward Euler method in each.
const x_n_E = TimeIntegrators.initialize_scheme(x₀, TimeIntegrators.FORWARD_EULER)
const v_n_E = TimeIntegrators.initialize_scheme(v₀, TimeIntegrators.FORWARD_EULER)

function integrate_sym_Euler!(x_n, v_n, t)
    ## Advance both ODEs. Note that the velocity must be updated first, to ensure that the
    ## position update can use the new velocity. This is what makes this the symplectic
    ## Euler scheme.
    TimeIntegrators.time_integrate!(
        v_n, ode_velocity, t, dt; pos=TimeIntegrators.get_solution(x_n)
    )
    TimeIntegrators.time_integrate!(
        x_n, ode_position, t, dt; vel=TimeIntegrators.get_solution(v_n)
    )

    return nothing
end

# Alternatively, we can also use a Störmer-Verlet type scheme, by combining the same
# integrators in a different way.
const x_n_S = TimeIntegrators.initialize_scheme(x₀, TimeIntegrators.FORWARD_EULER)
const v_n_S = TimeIntegrators.initialize_scheme(v₀, TimeIntegrators.FORWARD_EULER)

function integrate_Stormer_Verlet!(x_n, v_n, t)
    ## Advance both ODEs. By first updating the velocity with a half time-step, followed by
    ## a full position updated, followed by another half time-step velocity update, we
    ## obtain the Störmer-Verlet (or kick-drift-kick) algorithm.
    TimeIntegrators.time_integrate!(
        v_n, ode_velocity, t, 0.5*dt; pos=TimeIntegrators.get_solution(x_n)
    )
    TimeIntegrators.time_integrate!(
        x_n, ode_position, t, dt; vel=TimeIntegrators.get_solution(v_n)
    )
    TimeIntegrators.time_integrate!(
        v_n, ode_velocity, t+0.5*dt, 0.5*dt; pos=TimeIntegrators.get_solution(x_n)
    )

    return nothing
end

# For comparison, we also show what happens when choosing an integrator that does not
# conserve the Hamiltonian. Note that the scheme used for both ODEs is a higher-order
# integrator. However, since the coupling between the ODEs is poor, this still does not
# provide accurate results.
const x_n_R = TimeIntegrators.initialize_scheme(x₀, TimeIntegrators.RK4)
const v_n_R = TimeIntegrators.initialize_scheme(v₀, TimeIntegrators.RK4)

function integrate_RK4!(x_n, v_n, t)
    ## Advance both ODEs. Here, we use the current velocity and position in both cases.
    v_current = TimeIntegrators.get_solution(v_n)
    TimeIntegrators.time_integrate!(
        v_n, ode_velocity, t, dt; pos=TimeIntegrators.get_solution(x_n)
    )
    TimeIntegrators.time_integrate!(x_n, ode_position, t, dt; vel=v_current)

    return nothing
end

# Then we can set the desired start time t0, the time step size dt and the final time T.
const dt = 0.05
const T = 10.0
const t0 = 0.0

# The remaining code is just for the visualisation. We want to create an animation where we
# see the masses move in their orbit with a tail. We set the tail length to be ``30``,
# meaning that we see at most ``30`` points. We will also make them increasingly less
# visible, which is why we define the alphas. The animation is then created using the
# `record` function (from `GLMakie`) and `Makie`'s `Observables`. See the `Makie`
# documentation for more details on creating animations.
const tail_length = 30
alphas = LinRange(0.0, 1.0, tail_length)
time = Observable(t0)

## Histories for the symplectic Euler integrator.
tail_x_E = [copy(TimeIntegrators.get_solution(x_n_E)[1:2:end]) for _ in 1:tail_length]
tail_y_E = [copy(TimeIntegrators.get_solution(x_n_E)[2:2:end]) for _ in 1:tail_length]
hamiltonian_history_vec_E = Tuple{Float64, Float64}[]
colors_E = [(:black, i) for i in alphas]

## Histories for the Stormer-Verlet integrator.
tail_x_S = [copy(TimeIntegrators.get_solution(x_n_S)[1:2:end]) for _ in 1:tail_length]
tail_y_S = [copy(TimeIntegrators.get_solution(x_n_S)[2:2:end]) for _ in 1:tail_length]
hamiltonian_history_vec_S = Tuple{Float64, Float64}[]
colors_S = [(:red, i) for i in alphas]

## Histories for the RK4 integrator.
tail_x_R = [copy(TimeIntegrators.get_solution(x_n_R)[1:2:end]) for _ in 1:tail_length]
tail_y_R = [copy(TimeIntegrators.get_solution(x_n_R)[2:2:end]) for _ in 1:tail_length]
hamiltonian_history_vec_R = Tuple{Float64, Float64}[]
colors_R = [(:green, i) for i in alphas]

## This part is doing the actual computation at each timestep, everything else is just for
## the animation.
integrator = lift(time) do t
    integrate_sym_Euler!(x_n_E, v_n_E, t)
    integrate_Stormer_Verlet!(x_n_S, v_n_S, t)
    integrate_RK4!(x_n_R, v_n_R, t)
    return t
end

function update_tails!(x_n, tail_x, tail_y)
    x = TimeIntegrators.get_solution(x_n)

    push!(tail_x, x[1:2:end])
    push!(tail_y, x[2:2:end])
    if length(tail_x) > tail_length
        popfirst!(tail_x)
        popfirst!(tail_y)
    end

    return nothing
end

update_all_tails = lift(integrator) do update
    ## Now we update the trails for the visualisation. If there are more entries than the
    ## desired `trail_length`, we remove the oldest position.
    update_tails!(x_n_E, tail_x_E, tail_y_E)
    update_tails!(x_n_S, tail_x_S, tail_y_S)
    update_tails!(x_n_R, tail_x_R, tail_y_R)

    return nothing
end

tail1_E = lift(update_all_tails) do update
    return Point2f.(getindex.(tail_x_E, 1), getindex.(tail_y_E, 1))
end
tail2_E = lift(update_all_tails) do update
    return Point2f.(getindex.(tail_x_E, 2), getindex.(tail_y_E, 2))
end
tail3_E = lift(update_all_tails) do update
    return Point2f.(getindex.(tail_x_E, 3), getindex.(tail_y_E, 3))
end

tail1_S = lift(update_all_tails) do update
    return Point2f.(getindex.(tail_x_S, 1), getindex.(tail_y_S, 1))
end
tail2_S = lift(update_all_tails) do update
    return Point2f.(getindex.(tail_x_S, 2), getindex.(tail_y_S, 2))
end
tail3_S = lift(update_all_tails) do update
    return Point2f.(getindex.(tail_x_S, 3), getindex.(tail_y_S, 3))
end

tail1_R = lift(update_all_tails) do update
    return Point2f.(getindex.(tail_x_R, 1), getindex.(tail_y_R, 1))
end
tail2_R = lift(update_all_tails) do update
    return Point2f.(getindex.(tail_x_R, 2), getindex.(tail_y_R, 2))
end
tail3_R = lift(update_all_tails) do update
    return Point2f.(getindex.(tail_x_R, 3), getindex.(tail_y_R, 3))
end

function compute_hamiltonian(x_n, v_n)
    x = TimeIntegrators.get_solution(x_n)
    v = TimeIntegrators.get_solution(v_n)
    K = 0.0
    U = 0.0
    for i in 1:2:5
        for j in 1:2:5
            if i < j
                r = sqrt((x[i] - x[j])^2 + (x[i + 1] - x[j + 1])^2)
                U += 1.0 / r
            end
        end
        K += 0.5 * (v[i]^2 + v[i+1]^2)
    end
    return K - U
end

hamiltonian_E = lift(integrator) do t
    return t, compute_hamiltonian(x_n_E, v_n_E)
end
hamiltonian_history_E = lift(hamiltonian_E) do (t, H)
    H = push!(hamiltonian_history_vec_E, (t, H))
    return Point2f.(hamiltonian_history_vec_E)
end

hamiltonian_S = lift(integrator) do t
    return t, compute_hamiltonian(x_n_S, v_n_S)
end
hamiltonian_history_S = lift(hamiltonian_S) do (t, H)
    H = push!(hamiltonian_history_vec_S, (t, H))
    return Point2f.(hamiltonian_history_vec_S)
end

hamiltonian_R = lift(integrator) do t
    return t, compute_hamiltonian(x_n_R, v_n_R)
end
hamiltonian_history_R = lift(hamiltonian_R) do (t, H)
    H = push!(hamiltonian_history_vec_R, (t, H))
    return Point2f.(hamiltonian_history_vec_R)
end

fig = Figure()
ax_sol = Axis(
    fig[1, 1];
    title=@lift("t = $(@sprintf("%0.2f", round($time, digits = 2)))"),
    limits=(-1.2, 1.2, -1.2, 1.2),
)
scatter!(ax_sol, tail1_E; color=colors_E)
scatter!(ax_sol, tail2_E; color=colors_E)
scatter!(ax_sol, tail3_E; color=colors_E)

scatter!(ax_sol, tail1_S; color=colors_S)
scatter!(ax_sol, tail2_S; color=colors_S)
scatter!(ax_sol, tail3_S; color=colors_S)

scatter!(ax_sol, tail1_R; color=colors_R)
scatter!(ax_sol, tail2_R; color=colors_R)
scatter!(ax_sol, tail3_R; color=colors_R)

ax_ham = Axis(
    fig[1, 2];
    title=@lift("t = $(@sprintf("%0.2f", round($time, digits = 2)))"),
    limits=(t0, T, -2, -0.5),
)

scatter!(ax_ham, hamiltonian_E; color=:black)
lines!(ax_ham, hamiltonian_history_E; color=:black)

scatter!(ax_ham, hamiltonian_S; color=:red)
lines!(ax_ham, hamiltonian_history_S; color=:red)

scatter!(ax_ham, hamiltonian_R; color=:green)
lines!(ax_ham, hamiltonian_history_R; color=:green)

record(
    fig, "three_body_problem.mp4", LinRange(t0, T, round(Int, T / dt)); framerate=30
) do t
    time[] = t
end

# ```@raw html
# <video autoplay loop muted playsinline controls src="./three_body_problem.mp4" />
# ```

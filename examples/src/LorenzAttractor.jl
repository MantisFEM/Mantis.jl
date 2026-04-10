# # Lorenz Attractor

# ## [Background knowledge](@id LorenzBackground)
# The [Lorenz system](https://en.wikipedia.org/wiki/Lorenz_system) is well-known system of
# three non-linear ODEs. It is defined by the following system of equations
# ```math
# \begin{cases}
#   \frac{\partial x}{\partial t} = \sigma (y - x) \;, \\
#   \frac{\partial y}{\partial t} = x (\rho - z) \;, \\
#   \frac{\partial z}{\partial t} = x y - \beta z\;,
# \end{cases}
# ```
# where ``x``, ``y``, and ``z`` are the coordinates in 3D space, ``t`` is the time, and
# ``\sigma``, ``\rho``, and ``\beta`` are constants.
#
# In this example, we will solve the above system using an implicit time integrator and the
# Newton-Rhapson method for the non-linearities.

# ## [Implementation](@id ThreeBodyImplementation)
# In this section, we will implement the above Lorenz system using the `TimeIntegrators`
# module within `Mantis`.
#
# ### [Packages](@id ThreeBodyPackages)
# In this example, we will use `Mantis` for the integration, `GLMakie` to create an
# animation, `Printf` to easily create a nicely formatted title for the animation,
# `StaticArrays` for efficiently storing small arrays, and LinearAlgebra to make use of the
# identity matrix. So, we are `using` these five packages.

using Mantis
using GLMakie
using Printf
using StaticArrays
using LinearAlgebra

# ### [Test case](@id LorenzTestCase)
# In this example, we will consider the 'butterfly'-shaped solutions with the parameters as
# shown in the code.

const σ = 10
const ρ = 28
const β = 8/3

# ### [Setting up the ODE](@id LorenzODESetup)
# The ODEs as described in the [background section](@ref LorenzBackground) need to be
# implemented. In this case, the ODEs themselves are relatively simple to implement. Next
# to the ODEs, we also create the Jacobian of the system, which will be used in the
# Newton-Rhapson method.

function g(sol)
    x = sol[1]
    y = sol[2]
    z = sol[3]
    return [σ * (y - x), x * (ρ - z) - y, x * y - β * z]
end

function J(sol)
    x, y, z = sol

    ## Note that SMatrix creates a matrix per column.
    return SMatrix{3,3}(-σ, ρ-z, y, σ, -1.0, x, 0.0, -x, -β)
end

function newton_rhapson(
    output, x, h, g, jacobian, t; eps=1e-14, iter=10,
)
    for i in eachindex(output, x)
        output[i] = x[i]
    end

    for i in 1:iter
        residual = output - x - h * g(output)
        if all(r -> abs(r) < eps, residual)
            return output
        end
        diff = -(I - h * jacobian(output)) \ residual
        output .+= diff
    end

    return output
end

# With these functions, we can now define the implicit ode in `Mantis`.

const ode = TimeIntegrators.define_implicit_ode(
    (output, x, h, t) -> newton_rhapson(output, x, h, g, J, t), g
)

# ### [Time integration](@id LorenzTimeIntegration)
# We can now setup the time integrator of our choosing. We are solving an implicit system,
# so we should choose a diagonally (or fully) implicit integrator.

const method = TimeIntegrators.DIRK3

y₀ = [0.1, 0.0, 0.0]
const y_n = TimeIntegrators.initialize_scheme(y₀, method)
const dt = 1e-2
const n_steps = 10000

# The remaining code is almost all for the visualisation.

steps = Observable(1)
position_vec = Point3f[]
colour_vec = Int[]
colours = lift(steps) do step
    push!(colour_vec, step)

    return colour_vec
end
integrator = lift(steps) do step
    ## Here we advance our solution.
    TimeIntegrators.time_integrate!(
        y_n, ode, (step-1)*dt, dt
    )
    push!(position_vec, Point3f(vec(TimeIntegrators.get_solution(y_n))))

    return position_vec[1:step]
end


fig, ax, line = lines(
    integrator,
    color = colours,
    colormap = Reverse(:copper),
    transparency = true,
    axis = (;
        type = Axis3,
        protrusions = (0, 0, 0, 0),
        title=@lift("t = $(@sprintf("%0.2f", round(($steps-1)*dt, digits = 2)))"),
        viewmode = :fit,
        limits = (-30, 30, -30, 30, 0, 50),
        azimuth = 1.7pi,
    ),
)
line.colorrange = (0, n_steps)

record(
    fig, "lorenz_attractor.mp4", 1:n_steps; framerate=144
) do step
    steps[] = step
end

# ```@raw html
# <video autoplay loop muted playsinline controls src="./lorenz_attractor.mp4" />
# ```

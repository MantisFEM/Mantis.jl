# # Getting Started with Time Integrators
# In this example, we go through the basic setup needed when using the `TimeIntegrators`
# module in `Mantis`. We will use a very simple ODE and show how to setup different time
# integrators for this problem.

# ## [Background knowledge](@id GSTIBackground)
# We will consider the following ODE:
# ```math
#   \frac{dy}{dt} = \lambda y, \quad y(t=0) = 1.0\;,
# ```
# which has exact solution ``y(t) = e^{\lambda t}``.
# We will set ``\lambda`` as

const lambda = -4

# ## [Packages](@id GSTIPackages)
# As the goal of this example is to setup basic time integration problems in `Mantis`, we
# will only use `Mantis` for now.

using Mantis

# ## [Explicit Integrators](@id GSTIExplicit)
# In this section, we will treat the [previously introduced ODE](@ref GSTIBackground) with
# an [`TimeIntegrators.Explicit`](@ref) integrator. We will go through the setup
# step-by-step.

# ### [Step 1: Defining an explicit ODE](@id GSTIExplicit_1)
# For an explicit integrator, we need to provide a function that provides the explicit part
# of the ODE. In the terminology of the [TimeIntegrators](@ref)-module, this is the
# function ``F``. In our case, we simply have ``F(y, t) = \lambda y``.
#
# This is implemented as follows.
#
# !!! note "Make sure to overwrite the first argument (here called `output`)."
#     As explained in the [TimeIntegrators.`TimeIntegrationOperators`](@ref) docstring,
#     the evaluate function for an explicit ODE should have three input arguments, and the
#     first one must be overwritten. Not overwriting the first argument will lead to
#     incorrect results.

function ode_explicit_function!(output, yn, t)
    for n in eachindex(output)
        output[n] = lambda * yn[n]
    end
    return nothing
end

# We can use this newly defined function and pass it to the
# [`TimeIntegrators.define_explicit_ode`](@ref)-function, which will create the required
# [`TimeIntegrators.TimeIntegrationOperators`](@ref) object.

ode_explicit = TimeIntegrators.define_explicit_ode(ode_explicit_function!)

# ### [Step 2: Picking and initialising a scheme](@id GSTIExplicit_2)
# Now we can pick an [`TimeIntegrators.Explicit`](@ref) integrator and initialise it. We
# will pick the provided [`TimeIntegrators.RK4`](@ref) method. Since this is a multi-stage
# but single-step method, the initialisation procedure for this integrator is simple.

const method_ex = TimeIntegrators.RK4

const y₀_ex = [1.0]
const y_n_ex = TimeIntegrators.initialise_scheme(y₀_ex, method_ex)

# ### [Step 3: Integrating the ODE](@id GSTIExplicit_3)
# Now we can integrate our ODE. Say we want to integrate for 100 time steps with a time
# step size of 0.1, then our code would look like this.

const n_steps_ex = 100
const dt_ex = 0.1
t_ex = 0.0
for step in 1:100
    global t_ex
    ## Call this function to advance the solution. Note that the input time is the current
    ## time.
    TimeIntegrators.time_integrate!(y_n_ex, ode_explicit, t_ex, dt_ex)
    t_ex += dt_ex
end

# ### [Step 2 take 2: Picking and initialising a scheme](@id GSTIExplicit_2v2)
# If we instead want to use a multi-step method, we will need to be a little more careful
# when initialising the scheme. Say we pick the provided [`TimeIntegrators.AB2`](@ref)
# method. We will also have to pick a startup scheme. In this case, we pick
# [`TimeIntegrators.HEUN2`](@ref). Note that a startup scheme cannot be a multi-step scheme
# itself. Also, make sure that the startup scheme is accurate enough to initialise your
# multi-step scheme, or you may lose accuracy.

const method_ex2 = TimeIntegrators.AB2
const startup_method_ex2 = TimeIntegrators.HEUN2

const y₀_ex2 = [1.0]
const y_n_ex2 = TimeIntegrators.initialise_scheme(y₀_ex2, method_ex2, startup_method_ex2)

# ## [Diagonally Implicit Integrators](@id GSTIDiagImplicit)
# In this section, we will treat the [previously introduced ODE](@ref GSTIBackground) with
# an [`TimeIntegrators.DiagonallyImplicit`](@ref) integrator. We will go through the setup
# step-by-step.

# ### [Step 1: Defining an diagonally Implicit ODE](@id GSTIDiagImplicit_1)
# For a diagonally implicit integrator, we need to provide a solver that solves the
# implicit part of the ODE. That is, in the terminology of the
# [TimeIntegrators](@ref)-module, we need to solve
# ``\mathbf{Y} - h \mathbf{g}(\mathbf{Y}) = \mathbf{x}``. If
# ``\mathbf{g}(\mathbf{Y})`` is linear, as it is in this example, we can use a direct
# solver. This is implemented as follows. Note that, for efficiency reasons, you may want
# to use solver that directly overwrite the output argument. For simplicity, that is not
# done here.
#
# !!! note "Make sure to overwrite the first argument (here called `output`)."
#     As explained in the [`TimeIntegrators.TimeIntegrationOperators`](@ref) docstring, the
#     solver function for a diagonally implicit ODE should have four input arguments, and
#     the first one must be overwritten. Not overwriting the first argument will lead to
#     incorrect results.
#
# !!! note "An implicit evaluate function may be needed."
#     As explained in the [`TimeIntegrators.define_diagonally_implicit_ode`](@ref)
#     docstring, if you use a multi-step scheme or a scheme with zeros on the diagonal, you
#     will also have to provide an implicit evaluate function.

import LinearAlgebra
function implicit_solve!(output, x, h, t)
    output .= (LinearAlgebra.I - h * lambda) \ x
    return nothing
end

# We can use this newly defined function and pass it to the
# [`TimeIntegrators.define_diagonally_implicit_ode`](@ref)-function, which will create the
# required [`TimeIntegrators.TimeIntegrationOperators`](@ref) object.

ode_diag_implicit = TimeIntegrators.define_diagonally_implicit_ode(implicit_solve!)

# ### [Step 2: Picking and initialising a scheme](@id GSTIDiagImplicit_2)
# Now we can pick a [`TimeIntegrators.DiagonallyImplicit`](@ref) integrator and initialise
# it. We will pick the provided [`TimeIntegrators.DIRK3`](@ref) method. Since this is a
# multi-stage but single-step method, the initialisation procedure for this integrator is
# simple.

const method_di = TimeIntegrators.DIRK3

const y₀_di = [1.0]
const y_n_di = TimeIntegrators.initialise_scheme(y₀_di, method_di)

# ### [Step 3: Integrating the ODE](@id GSTIDiagImplicit_3)
# Now we can integrate our ODE. Say we want to integrate for 100 time steps with a time
# step size of 0.1, then our code would look like this. Note that this has the same
# structure as the [explicit case](@ref GSTIExplicit_3).

const n_steps_di = 100
const dt_di = 0.1
t_di = 0.0
for step in 1:100
    global t_di
    ## Call this function to advance the solution. Note that the input time is the current
    ## time.
    TimeIntegrators.time_integrate!(y_n_di, ode_diag_implicit, t_di, dt_di)
    t_di += dt_di
end

# ## [Fully Implicit Integrators](@id GSTIImplicit)
# In this section, we will treat the [previously introduced ODE](@ref GSTIBackground) with
# an [`TimeIntegrators.Implicit`](@ref) integrator. We will go through the setup
# step-by-step.

# ### [Step 1: Defining an Implicit ODE](@id GSTIImplicit_1)
# For an implicit integrator, we need to provide both a solver that solves the implicit
# part of the ODE as well as an evaluate function for this implicit part. In the
# terminology of the  [TimeIntegrators](@ref)-module, we need to solve
# ``\mathbf{Y} - h \mathbf{g}(\mathbf{Y}) = \mathbf{x}``. If
# ``\mathbf{g}(\mathbf{Y})`` is linear, as it is in this example, we can use a direct
# solver. Note that, for efficiency reasons, you may want
# to use a solver that directly overwrites the output argument. For simplicity, that is not
# done here. Additionally, we need to define the function ``G(y, t) = \lambda y``. This is
# implemented as follows.
#
# !!! note "Make sure to overwrite the first argument (here called `output`)."
#     As explained in the [`TimeIntegrators.TimeIntegrationOperators`](@ref) docstring, the
#     solver function for an implicit ODE should have four input arguments, and the first
#     one must be overwritten. Not overwriting the first argument will not lead to correct
#     results. The evaluate function should have three argument, and the first one must
#     also be overwritten.

function implicit_solve!(output, x, h, t)
    output .= (LinearAlgebra.I - h * lambda) \ x
    return nothing
end

function implicit_evaluate!(output, y, t)
    for n in eachindex(output, y)
        output[n] = lambda * y[n]
    end
    return nothing
end

# We can use this newly defined function and pass it to the
# [`TimeIntegrators.define_implicit_ode`](@ref)-function, which will create the required
# [`TimeIntegrators.TimeIntegrationOperators`](@ref) object.

ode_implicit = TimeIntegrators.define_implicit_ode(implicit_solve!, implicit_evaluate!)

# ### [Step 2: Picking and initialising a scheme](@id GSTIImplicit_2)
# Now we can pick an [`TimeIntegrators.Implicit`](@ref) integrator and initialise it. We
# will pick the provided [`TimeIntegrators.GAUSS_LEGENDRE_4`](@ref) method. Since this is a
# multi-stage but single-step method, the initialisation procedure for this integrator is
# simple.

const method_impl = TimeIntegrators.GAUSS_LEGENDRE_4

const y₀_impl = [1.0]
const y_n_impl = TimeIntegrators.initialise_scheme(y₀_impl, method_impl)

# ### [Step 3: Integrating the ODE](@id GSTIImplicit_3)
# Now we can integrate our ODE. Say we want to integrate for 100 time steps with a time
# step size of 0.1, then our code would look like this. Note that this again has the same
# structure as in the previous cases.

const n_steps_impl = 100
const dt_impl = 0.1
t_impl = 0.0
for step in 1:100
    global t_impl
    ## Call this function to advance the solution. Note that the input time is the current
    ## time.
    TimeIntegrators.time_integrate!(y_n_impl, ode_implicit, t_impl, dt_impl)
    t_impl += dt_impl
end

# ## [IMEX Integrators](@id GSTIIMEX)
# In this section, we will treat the [previously introduced ODE](@ref GSTIBackground) with
# an [`TimeIntegrators.IMEX`](@ref) integrator. We will go through the setup step-by-step.

# ### [Step 1: Defining an IMEX ODE](@id GSTIIMEX_1)
# IMEX integrators can treat one part of the ODE with an implicit step, while treating the
# rest with an explicit step. The stiffness of the terms in your ODE often influences this
# split. In this simple example, we introduce the following split:
# ```math
#     \frac{dy}{dt} = 0.5 \lambda y + 0.5 \lambda y\;.
# ```
# That is, the implicit and explicit parts are the same.
#
# For an IMEX integrator, we need to provide a function for evaluating the explicit part,
# here ``F(y, t) = 0.5 \lambda y``, and a solver for the implicit part, just like in the
# [diagonally implicit case](@ref GSTIDiagImplicit_1). This is implemented as follows.
#
# !!! note "Make sure to overwrite the first argument (here called `output`)."
#     As explained in the [`TimeIntegrators.TimeIntegrationOperators`](@ref) docstring,
#     the evaluate function for an explicit ODE should have three input arguments, and the
#     first one must be overwritten. Not overwriting the first argument will not lead to
#     correct results.
#
# !!! note "An implicit evaluate function may be needed."
#     As explained in the [`TimeIntegrators.define_imex_ode`](@ref) docstring, if you use a
#     multi-step scheme or a scheme with zeros on the diagonal (as done here), you will
#     also have to provide an implicit evaluate function.

function explicit_imex_function!(output, yn, t)
    for n in axes(output, 1)
        output[n] = 0.5 * lambda * yn[n]
    end
    return nothing
end

function implicit_imex_solver!(output, x, h, t)
    LinearAlgebra.ldiv!(output, LinearAlgebra.lu(LinearAlgebra.I - 0.5 * h * lambda), x)
    return nothing
end

function implicit_imex_function!(output, yn, t)
    for n in axes(output, 1)
        output[n] = 0.5 * lambda * yn[n]
    end
    return nothing
end

# We can use this newly defined function and pass it to the
# [`TimeIntegrators.define_imex_ode`](@ref)-function, which will create the required
# [`TimeIntegrators.TimeIntegrationOperators`](@ref) object.

ode_imex = TimeIntegrators.define_imex_ode(
    explicit_imex_function!, implicit_imex_solver!, implicit_imex_function!
)

# ### [Step 2: Picking and initialising a scheme](@id GSTIIMEX_2)
# Now we can pick an [`TimeIntegrators.IMEX`](@ref) integrator and initialise it. We will
# pick the provided [`TimeIntegrators.RK3_IMEX`](@ref) method. Since this is a multi-stage
# but single-step method, the initialisation procedure for this integrator is simple.

const method_imex = TimeIntegrators.RK3_IMEX

const y₀_imex = [1.0]
const y_n_imex = TimeIntegrators.initialise_scheme(y₀_imex, method_imex)

# ### [Step 3: Integrating the ODE](@id GSTIIMEX_3)
# Now we can integrate our ODE. Say we want to integrate for 100 time steps with a time
# step size of 0.1, then our code would look like this. Note that this again has the same
# structure as in the previous cases.

const n_steps_imex = 100
const dt_imex = 0.1
t_imex = 0.0
for step in 1:100
    global t_imex
    ## Call this function to advance the solution. Note that the input time is the current
    ## time.
    TimeIntegrators.time_integrate!(y_n_imex, ode_imex, t_imex, dt_imex)
    t_imex += dt_imex
end

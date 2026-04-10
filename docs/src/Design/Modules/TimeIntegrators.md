```@meta
CurrentModule = Mantis.TimeIntegrators
```
# TimeIntegrators

The time integration module implemented in `Mantis` is based on the framework developed by [Vos2011](@cite).
This framework allows for an easy implementation of a variety of explicit, implicit, and implicit-explicit (IMEX) time stepping schemes, and is based on the concept of general linear methods (GLMs). 
See, for example, [Butcher2006](@cite), for more details.
These methods are applicable to both ODEs and PDEs, so that both are available in `Mantis`. 

## GLMs: Notation and Theory
General linear methods can be characterised as follows (see [Butcher2006](@cite), [Vos2011](@cite)). 
Consider the initial value problem defined as the ODE
```math
\frac{d\mathbf{y}}{dt} = \mathbf{f}(\mathbf{y}), \quad \mathbf{y}(t_0) = \mathbf{y}_0\;,
```
where ``\mathbf{f}: \mathbb{R}^N \to \mathbb{R}^N``. 
The ``n``-th (time) step of the GLM comprised of ``r`` (integrator) steps and ``s`` stages is then formulated as
```math
\begin{align}
    \mathbf{Y}_i &= \Delta t \sum_{j=1}^{s} a_{ij} \mathbf{F}_j + \sum_{j=1}^{r} u_{ij} \mathbf{y}_j^{n-1}, \quad i = 1, \dots, s\;, \\
    \mathbf{y}_i^n &= \Delta t \sum_{j=1}^{s} b_{ij} \mathbf{F}_j + \sum_{j=1}^{r} v_{ij} \mathbf{y}_j^{n-1}, \quad i = 1, \dots, r\;,
\end{align}
```
where ``\mathbf{Y}_i`` are called the stage values and ``\mathbf{F}_i`` are called the stage derivatives.
These two quantities are related by the differential equation
```math
\mathbf{F}_i = \mathbf{f}(\mathbf{Y}_i)\;.
```
The above formulation can be cast into the following matrix form
```math
\begin{bmatrix}
\mathbf{Y} \\
\mathbf{y}^n
\end{bmatrix} = 
\begin{bmatrix}
A \otimes I & U \otimes I \\
B \otimes I & V \otimes I
\end{bmatrix}
\begin{bmatrix}
\Delta t \mathbf{Y} \\
\mathbf{y}^{n-1}
\end{bmatrix} \;,
```
which is often simplified (with some abuse of notation) to
```math
\begin{bmatrix}
\mathbf{Y} \\
\mathbf{y}^n
\end{bmatrix} = 
\begin{bmatrix}
A & U \\
B & V 
\end{bmatrix}
\begin{bmatrix}
\Delta t \mathbf{Y} \\
\mathbf{y}^{n-1}
\end{bmatrix} \;.
```
Either way, the vectors ``\mathbf{y}`` (in/output approximations), ``\mathbf{Y}`` (stage values), and ``\mathbf{F}`` (stage derivatives) are defined as
```math
\mathbf{y}^{n-1} = \begin{bmatrix}
y_1^{n-1} \\
y_2^{n-1} \\
\vdots \\
y_r^{n-1}
\end{bmatrix}\;, \quad 
\mathbf{y}^n = \begin{bmatrix}
y_1^{n} \\
y_2^{n} \\
\vdots \\
y_r^{n}
\end{bmatrix}\;, \quad
\mathbf{Y} = \begin{bmatrix}
Y_1 \\
Y_2 \\
\vdots \\
Y_s
\end{bmatrix}\;, \quad
\mathbf{F} = \begin{bmatrix}
F_1 \\
F_2 \\
\vdots \\
F_s
\end{bmatrix}\;.
```
It is important to note that the in/output vectors can contain more than just the solution. 
The exact content depends on the specific method, but often includes previously computed 
stage derivatives. This is particularly important when creating and/or initialising new 
methods.

### [GLMs: Characterising a GLM](@id TIGLMCharacter)
Any time integrator that fits in the above framework can thus be characterised by the four matrices ``A``, ``B``, ``U``, ``V``, and the layout of the in- and output vectors ``\mathbf{y}``. 
In addition, evey time integrator will also need a vector ``C``, which keeps track of the time at which the stages are evaluated.
In `Mantis`, the ``A``, ``B``, ``U``, ``V`` are stored in the time integrator structs (see [this section below](@ref TIwhatareTIs)).
The matrices have the following sizes:

| Matrix        | Size          |
| :-----------: | :-----------: |
| ``A``         | `num_stages` x `num_stages` |
| ``B``         | `num_steps` x `num_stages`  |
| ``U``         | `num_stages` x `num_steps`  |
| ``V``         | `num_steps` x `num_steps`   |
| ``C``         | `num_stages` |


### GLMs: Extension to IMEX Schemes
The framework introduced by [Vos2011](@cite) extends the GLM idea to IMEX integrators.
The ODE from the previous section is now split into
```math
\frac{d\mathbf{y}}{dt} = \mathbf{f}(\mathbf{y}) + \mathbf{g}(\mathbf{y}), \quad \mathbf{y}(t_0) = \mathbf{y}_0\;,
```
where ``\mathbf{f}: \mathbb{R}^N \to \mathbb{R}^N`` and ``\mathbf{g}: \mathbb{R}^N \to \mathbb{R}^N``. 
The ``\mathbf{f}``-part represent the part of the ODE that is treated explicitly, while the ``\mathbf{g}``-part is treated implicitly.
The ``n``-th (time) step of the IMEX-GLM comprised of ``r`` (integrator) steps and ``s`` stages is then formulated as
```math
\begin{align}
    \mathbf{Y}_i &= \Delta t \sum_{j=1}^{s} a^{IM}_{ij} \mathbf{G}_j + \Delta t \sum_{j=1}^{s} a^{EX}_{ij} \mathbf{F}_j + \sum_{j=1}^{r} u_{ij} \mathbf{y}_j^{n-1}, \quad i = 1, \dots, s\;, \\
    \mathbf{y}_i^n &= \Delta t \sum_{j=1}^{s} b^{IM}_{ij} \mathbf{G}_j + \Delta t \sum_{j=1}^{s} b^{EX}_{ij} \mathbf{F}_j + \sum_{j=1}^{r} v_{ij} \mathbf{y}_j^{n-1}, \quad i = 1, \dots, r\;,
\end{align}
```
where ``\mathbf{Y}_i`` are called the stage values and ``\mathbf{F}_i`` and ``\mathbf{G}_i`` are called the (explicit and implicit) stage derivatives.
These quantities are related by the differential equation
```math
\mathbf{F}_i = \mathbf{f}(\mathbf{Y}_i), \quad \mathbf{G}_i = \mathbf{g}(\mathbf{Y}_i)\;.
```
The matrix form is obtained in the same way as in the previous section.

IMEX GLMs are characterised in the same way as described in [GLMs: Characterising a GLM](@ref TIGLMCharacter). 
The only difference is that an IMEX GLM will have two ``A`` and ``B`` matrices, and two ``C`` vectors : one for the explicit part and one for the implicit part.


## [What are time integrators in `Mantis`?](@id TIwhatareTIs)
The top-level type within the `TimeIntegrators` module is the `AbstractTimeIntegrator{num_stages, num_steps}` type. 
```@docs
AbstractTimeIntegrator
```

Note that you can always obtain the number of stages and steps using the following functions.
```@docs
get_num_stages
get_num_steps
```

The `AbstractTimeIntegrator{num_stages, num_steps}` type has four concrete subtypes, each 
representing a specific class of time integrators.
```@docs
Explicit
DiagonallyImplicit
Implicit
IMEX
```

The ``A``-matrix (see above) in the GLM framework dictates whether a scheme is implicit or not.
When initialising one of the above structs, this is checked using the following function.
You can also use this to check what to expect.
```@docs
check_implicit
```

Since all schemes store the order of the scheme, you can always retrieve that information using the following getter.
```@docs
get_order
```

### [The solution objects](@id TIsolutions)
The time integrators themselves do not store the solution. 
This is instead handled by the `TimeIntegrationSolution`-object. 
```@docs
TimeIntegrationSolution
```

The `TimeIntegrationSolution` stores the information about the state of the time integration problem.
You can inspect such objects through the following getters.
```@docs
get_num_variables
get_solution
get_scheme
get_startup_scheme
get_remaining_startup_steps
```

::: details Internals: solution objects

We explain some internals related to the solution objects. 
However, this is considered an implementational detail.

Next to the information just mentioned, a `TimeIntegrationSolution` also stored the pre-allocated arrays, which can be obtained with the following getters.

```@docs
get_solution_allocated
get_F_allocated
get_G_allocated
get_stage_allocated
get_temp_var
```

> [!CAUTION]
> Do not manually modify the pre-allocated arrays.
> 
> Modifying the pre-allocated arrays may lead to incorrect results or unexpected behaviour.

The above getters are internally used to access the pre-allocated arrays and to overwrite their values.
You should not need these functions unless you are extending the integrate functionality to new types.

:::

## [Adding problem-specific information](@id TIprobleminfo)
To use the `TimeIntegrators`-module, you have to specify which problem you want to solve. 
Information about the problem is collected in `TimeIntegrationOperators`. 
See [GLMs: Notation and Theory](@ref) and [GLMs: Extension to IMEX Schemes](@ref) for the notation.
```@docs
TimeIntegrationOperators
```

There are various ways in which you can construct the above object, using the following functions.

::: tip The input functions are generic

This means that within these functions, you can choose how to treat your problem. For example, you can solve a non-linear system via the Newton-Rhapson method, as done in the [Lorenz Attractor](@ref)

:::

```@docs
define_explicit_ode
define_diagonally_implicit_ode
define_implicit_ode
define_imex_ode
```

## [Time Integration](@id TIintegration)
Now that the scheme, solution object, and the problem are all defined, we can perform the actual time integration.

### [Time Integrate](@id TIintegrate)
The integration happens by calling one of the following two methods.
```@docs
time_integrate
time_integrate!
```

::: details Internals: time integration

We explain how the time integration is performed. 
However, this is considered an implementational detail.

The integration functions from [Time Integrate](@ref TIintegrate) end up calling the following internal integrator.
This integrator function is specialised for different integrators and encodes how the time stepping is actually performed.
```@docs
_time_integrate!
```

:::

### [Initialisation](@id TIsolutions)
All integrators must be initialised. 
For multi-stage schemes, this is often just a matter of adding the initial condition. 
For multi-step schemes, this requires a startup scheme and more computation.
To handle these different initialisation requirements, `Mantis` has a `TimeLevels` struct, as introduced in [Vos2011](@cite), to keep track of what needs to be initialised. 
Every time integrator has a `TimeLevels` struct to define what information is needed from previous steps.
```@docs
TimeLevels
```

The actual initialisation step(s) can be performed by calling the following function.
```@docs
initialize_scheme
```

::: tip Other initialisation procedures

`Mantis` does not provide an exhaustive set of initialisation procedures. 
Some GLM schemes may require a different quantity to be initialised than what the `initialize_scheme`-method provides.
If this is the case, you can always perform the initialisation manually.
See [Adding your own scheme](@ref TimeIntegratorsAddYourOwn) for the details.

:::

## [Pre-implemented schemes](@id TIschemes)
`Mantis` provides a few pre-implemented schemes for convience. 

::: details Pre-implemented explicit integrators in the Runge-Kutta family.
```@docs
FORWARD_EULER
EXPLICIT_MIDPOINT
HEUN2
RALSTON2
HEUN3
RK3
RALSTON3
VDHW3
SSPRK3
RK4
RK4_3_8
RALSTON4
```
:::

::: details Pre-implemented (diagonally) implicit integrators in the Runge-Kutta family.
```@docs
BACKWARD_EULER
RADAU_IA_1
IMPLICIT_MIDPOINT
DIRK2
DIRK3
RADAU_IA_3
DIRK4
GAUSS_LEGENDRE_4
GAUSS_LEGENDRE_6
```
:::

::: details Pre-implemented explicit multi-step integrators.
```@docs
AB1
AB2
AB3
AB4
```
:::

::: details Pre-implemented (diagonally) implicit multi-step integrators.
```@docs
AM0
AM1
AM2
AM3
AM4
BDF1
BDF2
BDF3
BDF4
```
:::

::: details Pre-implemented IMEX integrators.
```@docs
BACKWARD_FORWARD_EULER
MIDPOINT_IMEX
RK3_IMEX
CNAB2
SSSS2
```
:::

You can, of course, always initialise a new scheme yourself (see the concrete types in [this section](@ref TIwhatareTIs) or [Adding your own scheme](@ref TimeIntegratorsAddYourOwn)).

Next to the pre-implemented schemes, `Mantis` also provides the following convenience function to take a Butcher-Tableau and turn it into a GLM-based time integrator.
```@docs
butcher_tableau_to_glm
```

## [Adding your own scheme](@id TimeIntegratorsAddYourOwn)
As an example of how to add your own time integrator, we look at how to implement an Almost Runge Kutta (ARK) scheme.
We use a specific scheme introduced in [Rattenbury2005](@cite).

This scheme requires a specialised initialisation, since it needs an estimate of the second derivative which is not accounted for in the available initialisations. 
As a result, this scheme is not part of Mantis.

::: details Example: solving a simple ODE with an ARK scheme.

Consider the ODE:
```math
\frac{dy}{dt} = \lambda y,\quad  y(t=0) = 1.0
```
which has exact solution 
```math
y(t) = \exp(\lambda t)\;.
```
We can encode this in code as
```@example arkexample
using Mantis
import StaticArrays

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
```
:::

The ARK3 scheme that we use here has a known GLM-representation (see [Rattenbury2005](@cite), page 50). 
This method, however, requires its own initialisation, see [Rattenbury2005](@cite), pages 37-38. 
This initialisation, can be easily implemented as shown below. 
Note that the `TimeLevels` struct has more entries for the explicit forcing step. 
No other changes are required. 
```@example arkexample
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

dt = 0.1 
yn = zeros(Float64, 1, 3)
yn[:, 1] .= [y_0]
yn[:, 2] .= lambda .* [y_0] .* dt
yn[:, 3] .= lambda^2 .* [y_0] .* dt^2
y_n = TimeIntegrators.TimeIntegrationSolution(yn, ARK3, nothing, 0)

for t in 0.0:dt:(t_final - dt)
    TimeIntegrators.time_integrate!(y_n, test_ode_explicit, t, dt)
end
```

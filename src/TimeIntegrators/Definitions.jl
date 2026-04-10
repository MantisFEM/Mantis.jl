"""
    AbstractTimeIntegrator{num_stages, num_steps}

Supertype for all time integrators.

# Type parameters
- `num_stages`: The number of stages for a multi-step scheme (such as the Runge-Kutta
    family). Since every scheme is at least a single-stage scheme, `num_stages` >= 1.
- `num_steps`: The number of steps for a multi-step scheme (such as the Adams-Bashforth
    family). Since every scheme is at least a single-step scheme, `num_steps` >= 1.
"""
abstract type AbstractTimeIntegrator{num_stages, num_steps} end

"""
    get_num_stages(
        ::AbstractTimeIntegrator{num_stages, num_steps}
    ) where {num_stages, num_steps}

Return the number of stages of the given [`AbstractTimeIntegrator`](@ref).
"""
function get_num_stages(
    ::AbstractTimeIntegrator{num_stages, num_steps}
) where {num_stages, num_steps}
    return num_stages
end

"""
    get_num_steps(
        ::AbstractTimeIntegrator{num_stages, num_steps}
    ) where {num_stages, num_steps}

Return the number of steps of the given [`AbstractTimeIntegrator`](@ref).
"""
function get_num_steps(
    ::AbstractTimeIntegrator{num_stages, num_steps}
) where {num_stages, num_steps}
    return num_steps
end

############################################################################################
##                              Problem-Specific Information                              ##
############################################################################################

"""
    TimeIntegrationOperators{EF, IF, IE}

Defines the ODE-specific operators used in the time integration.

Makes a distiction between the explicit and implicit operators. `EF`, `IF`, and `IE` are
the types the explicit and implicit functions, which are `Nothing` if not defined. Note
that at least `EF` or `IF` and `IE` must be a function.

# Constructors
- `define_explicit_ode(explicit_evaluate::Function)`: For fully explicit ODEs.
- `define_diagonally_implicit_ode(implicit_solve::Function)`: For diagonally implicit ODEs.
- `define_implicit_ode(implicit_solve::Function, implicit_evaluate::Function)`: For fully
    implicit ODEs.
- `define_imex_ode(explicit_evaluate::Function, implicit_solve::Function)`: For IMEX ODEs.
- `TimeIntegrationOperators(
        explicit_evaluate::Union{Nothing, Function},
        implicit_solve::Union{Nothing, Function},
        implicit_evaluate::Union{Nothing, Function},
    )`: Generic constructor.

# Fields
- `explicitEvaluate!::EF`: A function that evaluates the explicit part of the ODE, that is,
    the function that evaluates ``F = f(y, t)``. See the manual section on
    [TimeIntegrators](@ref) for the terminology. **This function must have the following
    inputs: (output, yn, t). It must also overwrite the output argument.** The output
    argument will be a vector-like object of length N (the number of variables), as will
    yn. The argument t will be a number indicating the current time.
- `implicitSolve!::IF`: A function that solves the implicit part of the ODE, that is, the
    function that solves the equation
    ``\\mathbf{Y} - h \\mathbf{g}(\\mathbf{Y}) = \\mathbf{x}``, with
    ``h = a^{IM}_{ii} \\Delta t``. See the manual section on [TimeIntegrators](@ref) for
    the terminology. In case g is a linear operator, a direct solution method can be
    through the inverse operator ``(I - h \\mathbf{g})\\mathbf{Y} = \\mathbf{x}``, where
    ``I`` is the identity function. **This function must have the following inputs:
    (output, x, h, t). It must also overwrite the output argument.** The output argument
    will be a matrix-like object of size (N, num_stages), as will yn. The arguments h will
    be either a number (for DiagonallyImplicit integrators) or a SMatrix (for Implicit
    integrators) and t will be a number (for DiagonallyImplicit integrators) or an SVector
    (for Implicit integrators) indicating the current time(s).
- `implicitEvaluate!::IE`: A function that evaluates the implicit part of the ODE, that is,
    the function that evaluates ``G = g(y, t)``. See the manual section on
    [TimeIntegrators](@ref) for the terminology. **This function must have the following
    inputs: (output, yn, t). It must also overwrite the output argument.** The output
    argument will be a vector-like object of length N (the number of variables), as will
    yn, and t will be a number indicating the current time.

# Type parameters
- `EF`: `typeof(explicitEvaluate!)` if initialised, `Nothing` otherwise.
- `IF`: `typeof(implicitSolve!)` if initialised, `Nothing` otherwise.
- `IE`: `typeof(implicitEvaluate!)` if initialised, `Nothing` otherwise.
"""
struct TimeIntegrationOperators{EF, IF, IE}
    explicitEvaluate!::EF
    implicitSolve!::IF
    implicitEvaluate!::IE

    function TimeIntegrationOperators(
        explicit_evaluate::Union{Nothing, Function},
        implicit_solve::Union{Nothing, Function},
        implicit_evaluate::Union{Nothing, Function},
    )
        if !(explicit_evaluate isa Function || implicit_solve isa Function || implicit_evaluate isa Function)
            throw(ArgumentError("At least one of the inputs must be a Function."))
        end

        new{typeof(explicit_evaluate), typeof(implicit_solve), typeof(implicit_evaluate)}(
            explicit_evaluate, implicit_solve, implicit_evaluate
        )
    end
end

"""
    define_explicit_ode(explicit_evaluate::Function)

Creates a [`TimeIntegrationOperators`](@ref) object for an explicit ODE.
"""
function define_explicit_ode(explicit_evaluate::Function)
    return TimeIntegrationOperators(explicit_evaluate, nothing, nothing)
end
"""
    define_diagonally_implicit_ode(
        implicit_solve::Function, implicit_evaluate::Union{Nothing, Function}=nothing
    )

Creates a [`TimeIntegrationOperators`](@ref) object for a diagonally implicit ODE. The
`implicit_evaluate` argument is only needed when using a multi-step scheme.
"""
function define_diagonally_implicit_ode(
    implicit_solve::Function, implicit_evaluate::Union{Nothing, Function}=nothing
)
    return TimeIntegrationOperators(nothing, implicit_solve, implicit_evaluate)
end
"""
    define_implicit_ode(implicit_solve::Function, implicit_evaluate::Function)

Creates a [`TimeIntegrationOperators`](@ref) object for an implicit ODE.
"""
function define_implicit_ode(implicit_solve::Function, implicit_evaluate::Function)
    return TimeIntegrationOperators(nothing, implicit_solve, implicit_evaluate)
end
"""
    define_imex_ode(
        explicit_evaluate::Function,
        implicit_solve::Function,
        implicit_evaluate::Union{Nothing, Function}=nothing
    )

Creates a [`TimeIntegrationOperators`](@ref) object for an IMEX ODE. The
`implicit_evaluate` argument is only needed when using a multi-step scheme.
"""
function define_imex_ode(
    explicit_evaluate::Function,
    implicit_solve::Function,
    implicit_evaluate::Union{Nothing, Function}=nothing
)
    return TimeIntegrationOperators(explicit_evaluate, implicit_solve, implicit_evaluate)
end

############################################################################################
##                                       TimeLevels                                       ##
############################################################################################

"""
    TimeLevels

The length of the fields of the `TimeLevels` object descibe the structure of the input and
output vectors for an integrator. The initialisation procedure uses this information to
determine what needs to be initialised. Note that the vectors may be empty.

# Fields
- `step_values::Vector{Int}`: The length of this vector determines how many previous
    solutions. For multi-stage methods, this is often just `[0]`. For multi-step methods,
    this may include a longer history.
- `step_derivatives_implicit::Vector{Int}`: Required implicit step derivatives from
    previous steps. Note that this is ``\\Delta t G``. This is, for example, used in the
    Adams-Moulton schemes.
- `step_derivatives_explicit::Vector{Int}`: Required explicit step derivatives from
    previous steps. Note that this is ``\\Delta t F``. This is, for example, used in the
    Adams-Bashford schemes.
"""
struct TimeLevels
    step_values::Vector{Int}
    step_derivatives_implicit::Vector{Int}
    step_derivatives_explicit::Vector{Int}
end

############################################################################################
##                                      Integrators                                       ##
############################################################################################

"""
    check_implicit(A::AbstractMatrix)

Check if the GLM matrix `A` belongs to an implicit or diagonally implicit scheme. Returns
two booleans; the first indicates if the scheme is fully implicit, the second if the scheme
is diagonally implicit.
"""
function check_implicit(A::AbstractMatrix)
    is_implicit = false
    is_diagonally_implicit = false
    for j in axes(A, 2)
        for i in axes(A, 1)
            if i == j && A[i, j] != zero(eltype(A))
                is_diagonally_implicit = true
            elseif j > i && A[i, j] != zero(eltype(A))
                is_implicit = true
            end
        end
    end

    return is_implicit, is_diagonally_implicit
end

"""
    Explicit{num_stages, num_steps, NT, AA, AE, EE} <:
        AbstractTimeIntegrator{num_stages, num_steps}

Explicit time integration scheme.

!!! note "Explicit time integrators are explicit in the ODE sense"
    Following [Vos2011](@cite), the explicit time integrators in this framework are
    considered explicit integrators when applied to ODEs. When applied to PDEs using a
    Galerkin method, one still has to solve a linear system. This can be referred to as an
    indirect explicit method in this case.

# Fields
- `A`, `B`, `U`, `V`: See the [GLM characterisation](@ref TIGLMCharacter) for more details,
    including the matrix sizes. All matrices are of type `SMatrix` with the appropriate
    size and `NT` as eltype.
- `C::SVector{num_stages, NT}`: Time Vector C of length `num_stages`. Indicates at what
    time each stage is evaluated.
- `time_levels::TimeLevels`: Required information from previous steps. See
    [`TimeLevels`](@ref) for the details.
- `order::Int`: Order of the scheme.

# Type parameters
- `num_stages`, `num_steps`: See [AbstractTimeIntegrator](@ref) for the details.
- `NT`: Element type of the `A`, `B`, `U`, `V` matrices.
- `AA`, `AE`, `EE`: Number of entries in `A`, (`B` and `U`), and `V`, respectively.
"""
struct Explicit{num_stages, num_steps, NT, AA, AE, EE} <:
       AbstractTimeIntegrator{num_stages, num_steps}
    A::SMatrix{num_stages, num_stages, NT, AA}
    B::SMatrix{num_steps, num_stages, NT, AE}
    U::SMatrix{num_stages, num_steps, NT, AE}
    V::SMatrix{num_steps, num_steps, NT, EE}
    C::SVector{num_stages, NT}
    time_levels::TimeLevels
    order::Int

    function Explicit(
        A::SMatrix{num_stages, num_stages, NT, AA},
        B::SMatrix{num_steps, num_stages, NT, AE},
        U::SMatrix{num_stages, num_steps, NT, AE},
        V::SMatrix{num_steps, num_steps, NT, EE},
        C::SVector{num_stages, NT},
        time_levels::TimeLevels,
        order::Int,
    ) where {num_stages, num_steps, NT, AA, AE, EE}
        is_implicit, is_diagonally_implicit = check_implicit(A)
        if is_implicit || is_diagonally_implicit
            throw(
                ArgumentError(
                    LazyString(
                        "Tried to build an Explicit integrator with an A matrix that ",
                        "belongs to an Implicit or DiagonallyImplicit scheme",
                    ),
                ),
            )
        end

        return new{num_stages, num_steps, NT, AA, AE, EE}(A, B, U, V, C, time_levels, order)
    end
end

"""
    DiagonallyImplicit{num_stages, num_steps, NT, AA, AE, EE} <:
        AbstractTimeIntegrator{num_stages, num_steps}

DiagonallyImplicit time integration scheme.

# Fields
- `A`, `B`, `U`, `V`: See the [GLM characterisation](@ref TIGLMCharacter) for more details,
    including the matrix sizes. All matrices are of type `SMatrix` with the appropriate
    size and `NT` as eltype.
- `C::SVector{num_stages, NT}`: Time Vector C of length `num_stages`. Indicates at what
    time each stage is evaluated.
- `time_levels::TimeLevels`: Required information from previous steps. See
    [`TimeLevels`](@ref) for the details.
- `order::Int`: Order of the scheme.

# Type parameters
- `num_stages`, `num_steps`: See [AbstractTimeIntegrator](@ref) for the details.
- `NT`: Element type of the `A`, `B`, `U`, `V` matrices.
- `AA`, `AE`, `EE`: Number of entries in `A`, (`B` and `U`), and `V`, respectively.
"""
struct DiagonallyImplicit{num_stages, num_steps, NT, AA, AE, EE} <:
       AbstractTimeIntegrator{num_stages, num_steps}
    A::SMatrix{num_stages, num_stages, NT, AA}
    B::SMatrix{num_steps, num_stages, NT, AE}
    U::SMatrix{num_stages, num_steps, NT, AE}
    V::SMatrix{num_steps, num_steps, NT, EE}
    C::SVector{num_stages, NT}
    time_levels::TimeLevels
    order::Int

    function DiagonallyImplicit(
        A::SMatrix{num_stages, num_stages, NT, AA},
        B::SMatrix{num_steps, num_stages, NT, AE},
        U::SMatrix{num_stages, num_steps, NT, AE},
        V::SMatrix{num_steps, num_steps, NT, EE},
        C::SVector{num_stages, NT},
        time_levels::TimeLevels,
        order::Int,
    ) where {num_stages, num_steps, NT, AA, AE, EE}
        is_implicit, is_diagonally_implicit = check_implicit(A)
        if is_implicit
            throw(
                ArgumentError(
                    LazyString(
                        "Tried to build a DiagonallyImplicit integrator with an A matrix ",
                        "that belongs to an Implicit scheme",
                    ),
                ),
            )
        end

        return new{num_stages, num_steps, NT, AA, AE, EE}(A, B, U, V, C, time_levels, order)
    end
end

"""
    Implicit{num_stages, num_steps, NT, AA, AE, EE} <:
        AbstractTimeIntegrator{num_stages, num_steps}

Implicit time integration scheme

# Fields
- `A`, `B`, `U`, `V`: See the [GLM characterisation](@ref TIGLMCharacter) for more details,
    including the matrix sizes. All matrices are of type `SMatrix` with the appropriate
    size and `NT` as eltype.
- `C::SVector{num_stages, NT}`: Time Vector C of length `num_stages`. Indicates at what
    time each stage is evaluated.
- `time_levels::TimeLevels`: Required information from previous steps. See
    [`TimeLevels`](@ref) for the details.
- `order::Int`: Order of the scheme.

# Type parameters
- `num_stages`, `num_steps`: See [AbstractTimeIntegrator](@ref) for the details.
- `NT`: Element type of the `A`, `B`, `U`, `V` matrices.
- `AA`, `AE`, `EE`: Number of entries in `A`, (`B` and `U`), and `V`, respectively.
"""
struct Implicit{num_stages, num_steps, NT, AA, AE, EE} <:
       AbstractTimeIntegrator{num_stages, num_steps}
    A::SMatrix{num_stages, num_stages, NT, AA}
    B::SMatrix{num_steps, num_stages, NT, AE}
    U::SMatrix{num_stages, num_steps, NT, AE}
    V::SMatrix{num_steps, num_steps, NT, EE}
    C::SVector{num_stages, NT}
    time_levels::TimeLevels
    order::Int

    function Implicit(
        A::SMatrix{num_stages, num_stages, NT, AA},
        B::SMatrix{num_steps, num_stages, NT, AE},
        U::SMatrix{num_stages, num_steps, NT, AE},
        V::SMatrix{num_steps, num_steps, NT, EE},
        C::SVector{num_stages, NT},
        time_levels::TimeLevels,
        order::Int,
    ) where {num_stages, num_steps, NT, AA, AE, EE}
        return new{num_stages, num_steps, NT, AA, AE, EE}(A, B, U, V, C, time_levels, order)
    end
end

"""
    IMEX{num_stages, num_steps, NT, AA, AE, EE} <:
        AbstractTimeIntegrator{num_stages, num_steps}

Implicit-Explicit (IMEX) time integration scheme. Currently only supportes implicit parts
which are diagonally implicit.

# Fields
- `A_IM`, `A_EX`, `B_IM`, `B_EX`, `U`, `V`: See the
    [GLM characterisation](@ref TIGLMCharacter) for more details, including the matrix
    sizes. All matrices are of type `SMatrix` with the appropriate size and `NT` as eltype.
- `C_IM::SVector{num_stages, NT}`: Time Vector C of length `num_stages`. Indicates at
    what time each implicit stage is evaluated.
- `C_EX::SVector{num_stages, NT}`: Time Vector C of length `num_stages`. Indicates at
    what time each explicit stage is evaluated.
- `time_levels::TimeLevels`: Required information from previous steps. See
    [`TimeLevels`](@ref) for the details.
- `order::Int`: order of the scheme

# Type parameters
- `num_stages`, `num_steps`: See [AbstractTimeIntegrator](@ref) for the details.
- `NT`: Element type of the `A`, `B`, `U`, `V` matrices.
- `AA`, `AE`, `EE`: Number of entries in (`A_IM` and `A_EX`), (`B_IM`, `B_EX` and `U`),
    and `V`, respectively.
"""
struct IMEX{num_stages, num_steps, NT, AA, AE, EE} <:
       AbstractTimeIntegrator{num_stages, num_steps}
    A_IM::SMatrix{num_stages, num_stages, NT, AA}
    A_EX::SMatrix{num_stages, num_stages, NT, AA}
    B_IM::SMatrix{num_steps, num_stages, NT, AE}
    B_EX::SMatrix{num_steps, num_stages, NT, AE}
    U::SMatrix{num_stages, num_steps, NT, AE}
    V::SMatrix{num_steps, num_steps, NT, EE}
    C_IM::SVector{num_stages, NT}
    C_EX::SVector{num_stages, NT}
    time_levels::TimeLevels
    order::Int

    function IMEX(
        A_IM::SMatrix{num_stages, num_stages, NT, AA},
        A_EX::SMatrix{num_stages, num_stages, NT, AA},
        B_IM::SMatrix{num_steps, num_stages, NT, AE},
        B_EX::SMatrix{num_steps, num_stages, NT, AE},
        U::SMatrix{num_stages, num_steps, NT, AE},
        V::SMatrix{num_steps, num_steps, NT, EE},
        C_IM::SVector{num_stages, NT},
        C_EX::SVector{num_stages, NT},
        time_levels::TimeLevels,
        order::Int,
    ) where {num_stages, num_steps, NT, AA, AE, EE}
        is_implicit, is_diagonally_implicit = check_implicit(A_IM)
        if is_implicit
            throw(
                ArgumentError(
                    LazyString(
                        "Tried to build an IMEX integrator with an A matrix ",
                        "that is fully implicit. This is currenlty not supported.",
                    ),
                ),
            )
        end

        return new{num_stages, num_steps, NT, AA, AE, EE}(
            A_IM, A_EX, B_IM, B_EX, U, V, C_IM, C_EX, time_levels, order
        )
    end
end

"""
    get_order(scheme::AbstractTimeIntegrator)

Return the order of the time integration scheme
"""
function get_order(scheme::AbstractTimeIntegrator)
    return scheme.order
end

############################################################################################
##                                       Solutions                                        ##
############################################################################################

"""
    TimeIntegrationSolution{T, S, NT, ST}

Solution and current state of the time integration problem.

!!! note "`TimeIntegrationSolution` does not store problem-specific information."
    While a `TimeIntegrationSolution` stores most information, it does not store problem-
    specific information. See [`TimeIntegrationOperators`](@ref) for the problem-specific
    information.


# Constructors
- `TimeIntegrationSolution(
        solution::Matrix{NT},
        scheme::AbstractTimeIntegrator{num_stages, num_steps},
        startup_scheme::Union{Nothing, AbstractTimeIntegrator},
        remaining_startup_steps::Int,
        startup_solution::ST=nothing,
    ) where {NT, num_stages, num_steps, ST}`: General constructor. Note that the eltype of
        the solution matrix will dictate the number type used in the
        `TimeIntegrationSolution`.

# Fields
- `N::Int`: Number varables in the system.
- `solution::Matrix{NT}`: Of size (`N`, num_steps).
- `scheme::T`: The time integration scheme.
- `startup_scheme::S`: The startup scheme, if desired. Will be `nothing` if there is no
    startup scheme.
- `remaining_startup_steps::Int`: remaining startup steps
- `solution_alocated::Matrix{NT}`: Pre-allocated memory for calculations.
- `F_alocated::Matrix{NT}`: Pre-allocated memory for calculations.
- `G_alocated::Matrix{NT}`: Pre-allocated memory for calculations.
- `startup_solution::ST`: The `TimeIntegrationSolution` object used for the startup
    procedure. This will contain information specific to the startup scheme and its current
    state. Will be `nothing` if there is no startup scheme.
- `stage_values::Vector{NT}`: Pre-allocated memory for calculations.
- `temp_var::Vector{NT}`: Pre-allocated memory for calculations.

# Type parameters
- `T`: Type of the scheme.
- `S`: Type of the startup scheme. `Nothing` if no startup scheme is provided.
- `NT`: eltype of the solution and pre-allocated arrays.
- `ST`: Type of the startup solution object, `Nothing` if no startup solution is provided.
"""
mutable struct TimeIntegrationSolution{T, S, NT, ST}
    N::Int
    solution::Matrix{NT}
    scheme::T
    startup_scheme::S
    remaining_startup_steps::Int
    solution_allocated::Matrix{NT}
    F_allocated::Matrix{NT}
    G_allocated::Matrix{NT}
    startup_solution::ST
    stage_values::Vector{NT}
    temp_var::Vector{NT}

    function TimeIntegrationSolution(
        solution::Matrix{NT},
        scheme::AbstractTimeIntegrator{num_stages, num_steps},
        startup_scheme::Union{Nothing, AbstractTimeIntegrator},
        remaining_startup_steps::Int,
        startup_solution::ST=nothing,
    ) where {NT, num_stages, num_steps, ST}
        return new{typeof(scheme), typeof(startup_scheme), NT, ST}(
            size(solution, 1),
            solution,
            scheme,
            startup_scheme,
            remaining_startup_steps,
            similar(solution),
            zeros(NT, size(solution, 1), num_stages),
            zeros(NT, size(solution, 1), num_stages),
            startup_solution,
            zeros(NT, size(solution, 1)),
            zeros(NT, size(solution, 1)),
        )
    end
end

Base.eltype(::Type{TimeIntegrationSolution{T, S, NT, ST}}) where {T, S, NT, ST} = NT

"""
    get_num_variables(sol::TimeIntegrationSolution)

Return the number of variables in the system.
"""
function get_num_variables(sol::TimeIntegrationSolution)
    return sol.N
end

"""
    get_solution(sol::TimeIntegrationSolution{T, S, NT}) where {T, S, NT}

Return the current solution values. This is a Matrix{NT} of size (`N`, num_steps), where
`N` is the number of variables in the system.
"""
function get_solution(sol::TimeIntegrationSolution{T, S, NT}) where {T, S, NT}
    return sol.solution
end

"""
    get_scheme(sol::TimeIntegrationSolution)

Return the time integration scheme used to obtain the solution.
"""
function get_scheme(sol::TimeIntegrationSolution)
    return sol.scheme
end

"""
    get_startup_scheme(sol::TimeIntegrationSolution)

Return the startup scheme used to initialise the solution. Will be nothing if no startup
scheme is present.
"""
function get_startup_scheme(sol::TimeIntegrationSolution)
    return sol.startup_scheme
end

"""
    get_remaining_startup_steps(sol::TimeIntegrationSolution)

Return the number of startup steps that still have to be performed. These are the steps
that will be computed using the startup scheme instead of the main scheme.
"""
function get_remaining_startup_steps(sol::TimeIntegrationSolution)
    return sol.remaining_startup_steps
end

"""
    get_solution_allocated(sol::TimeIntegrationSolution)

Return the pre-allocated matrix for the solution. To obtain the solution itself, use
[`get_solution`](@ref).
"""
function get_solution_allocated(sol::TimeIntegrationSolution)
    return sol.solution_allocated
end

"""
    get_F_allocated(sol::TimeIntegrationSolution)

Return the pre-allocated matrix for the explicit stage derivatives.
"""
function get_F_allocated(sol::TimeIntegrationSolution)
    return sol.F_allocated
end

"""
    get_G_allocated(sol::TimeIntegrationSolution)

Return the pre-allocated matrix for the implicit stage derivatives
"""
function get_G_allocated(sol::TimeIntegrationSolution)
    return sol.G_allocated
end

"""
    get_stage_allocated(sol::TimeIntegrationSolution)

Return the pre-allocated vector for the stage values.
"""
function get_stage_allocated(sol::TimeIntegrationSolution)
    return sol.stage_values
end

"""
    get_temp_var(sol::TimeIntegrationSolution)

Return the pre-allocated vector for the temporary variable for the implicit schemes.
"""
function get_temp_var(sol::TimeIntegrationSolution)
    return sol.temp_var
end

# return a scalar max of all elements in the TimeLevels struct
Base.maximum(tl::TimeLevels) = maximum([
    maximum(tl.step_values; init=0),
    maximum(tl.step_derivatives_implicit; init=0),
    maximum(tl.step_derivatives_explicit; init=0),
])

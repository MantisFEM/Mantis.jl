# [Why Julia?](@id WhyJulia)

In the section on [why we are developing `Mantis`](@ref WhyMantis), we briefly mentioned choosing Julia as programming language. Here, we detail this choice.

When we started the `Mantis`-project, we were not a priori tied to a specific programming language.
Already at the start, the expected users were researchers and students like ourselves.
We assumed that most researchers and students only had some experience in programming for scientific computing purposes.
Most likely in languages like Python or MATLAB.
This seemed to favour interpreted languages over compiled ones, to reduce the burden on the developers, the potential contributors, and the user.

Since the goal was to build an open source project, we also wanted to choose an open-source language.
So far, a language like Python would fit all our boxes.
However, languages like Python often start to suffer performance penalties when trying somewhat larger simulations.
One way around that is writing performance critical code in a lower level language.
This, however, leads to a two-language problem, which is something that we wanted to avoid.

Then we learned about the Julia programming language.
It was designed to fill precisely this gap [Bezanson2017, Bezanson2018](@cite).
Next to the promise of performance, Julia also introduced dynamic multiple dispatch.
This ensures that Julia code is dynamic, easily extensible, and specialisable.
That is, you can easily add new code, without performance penalty, and on the fly.

The same idea of being able to writing general code and have it specialise to a specific situation is also present when using differential forms.
[Differential forms](@ref TheoryForms) allow us to work with operators like the gradient, curl, and divergence in arbitrary dimensions and on general manifolds. 
Combining differential forms with Julia thus allows us to write finite element simulations directly using differential forms that are applicable to various geometries and use cases without per case modifications.
It also allowed our code to be open, accessible, and extensible.

This is why we think that the Julia programming language is a great choice for `Mantis`.

# [Why `Mantis`?](@id WhyMantis)

`Mantis` aims to provide a research toolbox for structure-preserving discretisations with a focus on differential forms.

[Differential forms](@ref TheoryForms) provide a generalised formalism for working with operators like the gradient, curl, and divergence in arbitrary dimensions and on general manifolds. 
From a [finite element method](@ref "The Finite Element Method (FEM)") perspective, this allows one to derive structure-preserving discretisations, study their properties, and do all of this in a general setting.

Unfortunately, while differential forms provide a powerful theoretical framework, their formalisms are often absent in FEM codes.
This is precisely the gap that `Mantis` aims to fill.

Working with differential forms and structure-preserving finite elements is currently mostly done by researchers and students. 
As a result, this is also the target audience for `Mantis`.


## Room for experimentation

In our opinion, a code for researchers should be easy to adapt and extend.
After all, researchers often want to experiment with things that are not (yet) standard.
Using existing codes, especially closed-source or multi-language ones, often creates a barier in this regard.
This is an important part of why `Mantis` is an open-source project and why we wrote it in Julia (more on why we choose Julia [here](@ref WhyJulia)).

Additionally, you can easily extend Julia code, and thus `Mantis`, to adapt it to your need.
And, if you think that such an extensions may be useful to others, feel free to [contribute](@ref "Contribution Guide").

We want you to be able to experiment with different choices and setups. 
So, whenever a choice is needed, we try to provide you with the options to choose for yourself.
However, if you do want to use what is common, we try to make this easy through defaults and/or helper functions.


## 'Blackboxability' & Abstraction Layers

`Mantis` is designed with research in mind. 
Researchers often focus on one part of a problem at a time. 
For example, when researching a new finite element space, you focus on precisely this aspect.
How do you construct this space, and what are its properties?
Nevertheless, once you have some prototype for the space, you will likely want to test it in a finite element simulation.
This requires all other aspects of a finite element code, but is something you probably prefer not to have to reimplement.
`Mantis` allows you to write your new space as a [`FunctionSpaces.AbstractFESpace`](@ref), allowing you to focus on implementing the new space.
Every other module and function will understand how to deal with your new space out-of-the-box.
This 'blackboxability' applies to each module of `Mantis` and is there for precisely this use case.

Additionally, abstract types and functions defined for them, allow us to create multiple abstraction layers.
This allows us to create different entry points into `Mantis`.
Whether you want to quickly set up some well-known spaces and solve a simple problem, or you want to have fine-grained control over one or more steps, the abstraction layers can help do (some) of the heavy lifting.
Working with functions on abstract types and relying on Julia's multiple dispatch to select the most appropriate algorithm is also ideal for differential forms. 
With forms, we often work with generalisations of the special cases seen in standard calculus.
Such abstraction layers allow us, and you, to turn these concepts into code.

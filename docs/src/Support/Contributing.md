# Contribution Guide

Thanks for your interest in contributing to `Mantis`! We appreciate you taking the time
to read this guide so that your efforts can be implemented into `Mantis` as easily as
possible. All contributions are welcome; so, if you are in doubt that your contribution is
too small or trivial, opt for creating a pull-request.

To make sure your contribution has the highest chance of being merged, please **read this
guide completely and carefully**.

## First steps

The first thing you should do is double-check that whatever you are proposing does not
already exist and is not already an issue or pull-request.
If an issue does exist but no associated pull-request to address it, then, when you get to
the point of creating your pull-request, please link back to the relevant issue.
If a pull-request also exists, you can just add your contribution directly to it.

Before you create a pull-request with your proposed changes, please ensure that what you
wrote is indeed a *contribution*. Here is what we mean by that:

> [!NOTE] Contribution
>
> A change to the source-code of `Mantis` that either:
>   1. Adds a new feature or extends the generality of existing features.
>   2. Fixes a bug.
>   3. Adds tests for existing features previously uncovered, or fixes current tests.
>   4. Increases performance, as in faster computation times, reduced memory usage, or
>      better type-stability.
>   5. Refactors existing code to improve readability or clarity.
>   6. Extends the documentation, doc-strings, or corrects outdated information, or typos.

If what you have in mind does not fall into one of these categories, then it is likely one
of the following:

1. A feature request, which you can read more about [here](./FeatureRequest.md).
1. A bug report, which you can read more about [here](./SubmitBugReport.md).
3. You need help getting `Mantis` up-and-running, which you can read more about
   [here](./GettingHelp.md).

## Creating a pull-request

If you made it here that means that what you wrote is a contribution.
Some of the guidelines we have below are general for all categories of a contribution, while
others are specific for each case.

### General guidelines

1. The code in `Mantis` follows, for the most part, the [Blue style
   guide](https://github.com/JuliaDiff/BlueStyle) for Julia, with minor modifications.
   Please try as best as possible to adhere to the conventions of this guide.
   We recommend you set up your language-server protocol (lsp) to automatically use the
   `.JuliaFormatter.toml` file in the `Mantis` repository and format the code accordingly,
   for example whenever you save a file.
   This will help you be consistent.
2. Write informative commit messages. These don't need to explain every line of code, that's
   what diffs are for, but they should explain the overall change implemented in the commit.
   (For example, `feat: add new cool geometry` or `perf: reduce evaluate allocations`).
   Have a look at [Conventional Commits](https://www.conventionalcommits.org/en/v1.0.0/) for
   more information. This is what we follow.
3. Make commits with a “single” purpose. This ties in nicely with the previous point. If you
   have a single commit that adds a new feature, tests and documentation for it, and fix a
   typo somewhere, what identifier do you give it? `feat:`, `test:`, `doc:`? Who knows!
4. When creating the pull-request, give a brief explanation what your intent is with the
   contribution. This should be only one or two sentences; the commit messages should handle
   the rest! Also, please add all relevant labels to make it immediately obvious what the
   intent of the pull-request is.
5. One of the rules in our repository is that we require commits to be signed. If you are
   already doing that, great. If not, you can read about why it is important, and how to set
   it up [here](https://git-scm.com/book/en/v2/Git-Tools-Signing-Your-Work).

### Specific guidelines

The following guidelines are specific to each type of contribution, so feel free to go
straight to the part that concerns your contribution.

1. [New feature](@ref)
2. [Bug fix](@ref)
3. [Performance](@ref)
4. [Refactor](@ref)
5. [Tests](@ref)
6. [Documentation](@ref)

#### New feature

1. Part of the commits in this case will be identified by `feat:`. It's okay to make a
   single `feat: add foo feature` commit adding everything that is relevant to your new
   feature, but consider breaking it apart into smaller `feat: add foo feature 1`, `feat:
   add foo feature 2`, ..., commits if you think it will help comprehension.
2. Properly document your feature! Take the time to write informative descriptions of new
   structures, methods, or whatever new things you add. This will make the whole review
   process easier for everyone. Note that you don't need to write separate `doc:` commits
   for your new feature, as the documentation we mention here is not an extension to the
   current one, but an integral part of understanding your contribution; it should be
   included in the `feat:` commits, so that the reviewer can immediately see what the intent
   of the new feature is.
3. Properly test your feature! If you are adding a new feature, then you should also add
   tests that validate it. These should be small unit tests that confirm your assumptions on
   what the code you wrote does. They will also help other people have confidence that your
   code serves its intended purpose, including the reviewers. Unlike in the previous point,
   tests should be written in separate `test:` commits, as these are not essential to
   understand your feature. Instead, they serve the separate goal of validating it.

#### Bug fix

1. Hopefully your contribution here will be small, so it should in principle fit under a
   single `fix:` commit. If by any chance you happen to fix several unrelated bugs please
   separate them into different commits: `fix: add missing variable 1`, `fix: correct
   parameter type 2`, for example.
2. If you are correcting a bug, that means the tests were incomplete and did not cover the
   case where that bug occurs. As such, consider accompanying your `fix:` commit with a
   `test:` commit that would have prevented that bug from going unnoticed.

#### Performance

1. Commits for this contribution should be identified with `perf:`. Since it is expected
   that the contribution provides some improvement, be that in terms of speed, memory, or
   type-stability, please include some sort of comparison that quantifies and showcases
   exactly what the contribution is. For example
   ```julia
   julia> @allocations old_foo()
   42
   julia> @allocations new_foo() # Reduces the number of allocations of `old_foo`.
   0
   ```

#### Refactor

1. The purpose of this contribution is to improve the readability or clarity of some part of
   the code. As such, please include a one-sentence description of what you think is
   improved by your change. Commits of this type should be labelled with `refactor:`.

#### Tests

1. As you probably guessed, commits in this contribution should be given the identifier
   `test:`. If you are contributing multiple tests, with distinct purposes, please try to
   break the commits apart: `test: add test for foo1`, `test: extend tests for foo2`.
2. Make your tests as atomic as possible. Rather than testing directly the output of a
   complicated method, whose arguments are themselves non-trivial, it's preferable to test
   each constituent method individually, desirably with simple arguments.
3. If the previous point seems impossible to achieve, then it's likely that the code you are
   trying to test is too convoluted and should be distilled into several simpler methods.
   Maybe a [Refactor](@ref) contribution is also due.

#### Documentation

1. For this contribution, the commit label should be `doc:`. As with other types of commits,
   it's good practice to keep commits separate if you are adding documentation to different
   things.
2. The main purpose of documentation is to improve the reader's understanding of something.
   Write in whatever way you feel that best achieves this purpose. However, it's generally
   advisable to be concise and stick to simple language.
3. If possible, please include an example of what you are documenting. This is often very
   helpful for the reader to fully grasp what you are explaining. For instance, consider the
   difference between
   ```julia
   """
      aggregate(a, b) 

   The aggregate of two yonder numerals, which bear the appelations `a` and `b`.
   """
   ```
   and
   ````julia
   """
      aggregate(a, b) 

   The aggregate of two yonder numerals, which bear the appelations `a` and `b`.

   # Example
   ```julia-repl
   julia> aggregate(1, 2)
   3
   ```
   """
   ````

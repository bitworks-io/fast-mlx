---
name: codebase-exploration
description: Structured read-only exploration of a specific question within a bounded scope. Use before implementing to locate code — files, signatures, call sites, and dependencies — without proposing or applying changes.
---

# Codebase Exploration

Targeted, read-only exploration to answer one specific question within a bounded scope. Use before implementing, to locate code without guessing. Do not propose or apply any changes.

## Inputs

- **Question** — the specific thing to find.
- **Scope paths** — the only paths to look within.

## Procedure

Explore the codebase to answer the question. For each match, report:

- File path and line number (`file:line`).
- Function, class, or type signature where applicable.
- Known call sites within the scope paths.
- Direct dependencies imported or referenced at that location.

If uncertain whether the right location was found, say so explicitly and name the next file or symbol that would resolve the uncertainty.

## Constraints

- Do not propose changes, refactors, or improvements.
- Do not read or report on files outside the scope paths unless required to resolve a direct dependency of the target.
- Do not speculate beyond what the code text supports.

## Output

A structured list of `file:line` references with signatures, call sites, and dependencies; uncertainty notes where applicable.

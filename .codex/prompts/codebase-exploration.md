# Codebase Exploration Prompt Packet

Purpose: targeted read-only exploration of a specific question within a bounded scope.
When to use: before implementing; hand to a scout subagent to locate code without guessing.

```
Explore the codebase to answer a specific question. Do NOT propose or apply any changes.

What to find:
{target}

Scope — look only within these paths:
{scope_paths}

For each match report:
- File path and line number (file:line format).
- Function/class/type signature where applicable.
- Known call sites within the scope paths.
- Direct dependencies imported or referenced at that location.

If you are uncertain whether you have found the right location, say so explicitly and
name the next file or symbol that would resolve the uncertainty.

Do NOT:
- Propose changes, refactors, or improvements.
- Read or report on files outside {scope_paths} unless required to resolve a
  direct dependency of the target.
- Speculate beyond what the code text supports.
```

Keep output to: a structured list of file:line references with signatures, call sites, and dependencies; uncertainty notes where applicable.

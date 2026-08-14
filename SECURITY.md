# Security policy

fast-mlx is experimental inference software that can load large local model artifacts and expose an
HTTP service. Treat model directories, evidence outputs, API keys, and benchmark hosts as sensitive.

## Reporting a vulnerability

Use GitHub's private vulnerability reporting flow (`Security` → `Advisories` →
`Report a vulnerability`) after the public repository launches. Do not include exploit details,
credentials, model paths, host identities, or private evidence in a public issue.

If private vulnerability reporting is temporarily unavailable, open a minimal public issue asking
the maintainers to establish a private channel; do not disclose technical details there.

## Current security boundaries

- The default server bind is loopback.
- A non-loopback bind requires a non-empty `FASTMLX_API_KEY`.
- Loaded-model serving requires explicit memory and cache limits.
- Evidence and source-admission controls are not substitutes for operating-system containment.
- Diagnostic and comparison evidence never grants runtime or publication authority by itself.

Only released versions and the current default branch receive security fixes. Because the project is
pre-release, interfaces and support windows may change with explicit release notes.

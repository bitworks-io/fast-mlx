---
paths:
  - "**/auth/**"
  - "**/migrations/**"
  - "**/payments/**"
  - "**/billing/**"
  - "**/admin/**"
  - "**/*webhook*"
---

# Security Rules

These apply when touching risk surfaces. The universal "never commit secrets" and
"run a secret scanner before publishing" rules live in `AGENTS.md` (always loaded).

- For changes touching auth, data migration, payments, or admin surfaces: state the threat model, blast radius, and rollback plan before implementing.
- Treat third-party MCP servers and shell hooks as privileged code; review them before enabling, just as you would a production dependency.
- Apply least-privilege by default — request read-only access where write access is not required.

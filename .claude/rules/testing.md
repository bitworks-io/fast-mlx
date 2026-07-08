---
paths:
  - "**/*test*"
  - "**/*spec*"
  - "tests/**"
---

# Testing Rules

- Write or observe a failing test first for any behavioral change; confirm it fails for the right reason before fixing.
- Run the narrowest meaningful test scope first (single file or function), then broaden to the full suite.
- Map each test to a named acceptance criterion from the user story; untethered tests are noise.
- Report what ran, what passed, what was skipped, and why — do not silently omit failures.
- When coverage tooling exists, include the coverage command and its result in the verification evidence.

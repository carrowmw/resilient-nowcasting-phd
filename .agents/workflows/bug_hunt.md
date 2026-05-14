---
description: Bug Hunt & Resolve
---

**Trigger:** When a bug, tensor mismatch, or NaN error is reported.

**Steps:**

1. **Scout:** The `@interrogator` searches the codebase and log files (`grep`, reading specific files) to locate the source of the error.
2. **Analyze:** The `@architect` analyzes the findings and proposes a fix, ensuring it adheres to the "Offensive Programming" (fail fast/hard) principles in the NSTAR docs.
3. **Desk Check:** If the bug is math/tensor related, the `@architect` performs a `Desk Check` skill to prove the fix works.
4. **Patch:** The `@implementer` applies the fix.
5. **Verify:** The `@implementer` runs the relevant tests via Docker.
6. **Log:** The `@implementer` logs the bug and the fix in `EXPERIMENTS.md`.

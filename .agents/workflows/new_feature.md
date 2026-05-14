---
description: Develop New Feature
---

**Trigger:** When requested to build a new feature or model component.

**Steps:**

1. **Branching:** The `@implementer` asks the user to confirm they are on a new Git branch (e.g., `git checkout -b feature/new-loss-function`).
2. **Investigation:** The `@interrogator` explores the codebase to find where the feature should be implemented and gathers the current file context.
3. **Design:** The `@architect` reviews the context against the `NSTAR` documents and writes a step-by-step implementation plan in `plan.md`.
4. **Execution:** Upon user approval of the plan, the `@implementer` writes the code.
5. **Testing:** The `@implementer` runs tests via Docker (`docker compose exec...`).
6. **Logging:** The `@implementer` adds an entry to `EXPERIMENTS.md` detailing the feature, date, and outcome.
7. **Commit:** The `@implementer` stages and commits the code.

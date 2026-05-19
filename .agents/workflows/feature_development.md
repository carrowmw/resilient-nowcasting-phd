# Feature Development Pipeline

**Trigger:** When the user suggests a feature to work on, a bug to fix, or some refactoring.

**Steps:**

1. **Interrogation:** `@interrogator` investigates the codebase relevant to the user's request. It outputs all findings, file paths, and context to `.agents/plans/context_for_architect.md` (CLEAN overwrite). It does NOT create a plan.
2. **Architecture:** `@architect` reads `context_for_architect.md` and the `NSTAR_*.md` files. It outputs a high-level design plan to `.agents/plans/architect_plan.md` (CLEAN overwrite).
3. **Implementation Planning:** `@implementer` reads `architect_plan.md` and outputs a step-by-step execution plan (files to edit, code to write) to `.agents/plans/implementation_plan.md` (CLEAN overwrite).
4. **Approval:** Wait for the USER to explicitly approve `implementation_plan.md`.
5. **Execution:** `@implementer` writes the code, runs necessary tests via Docker, updates `EXPERIMENTS.md`, and commits the changes.

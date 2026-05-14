# Skill: Execute Commands in Docker

**Context:** The workspace is strictly containerized. You cannot run `python`, `pytest`, or `npm` directly on the host machine.

**Action:** Whenever you need to execute a command, you must route it through the appropriate Docker container using `docker compose exec`.

**Syntax:**

- For Python/ML tasks: `docker compose exec [service_name] python [script_name.py]`
- For Frontend tasks: `docker compose exec [frontend_service_name] npm run [script]`

**Example:**
Instead of running `pytest tests/test_model.py`, you MUST run `docker compose exec [service_name] pytest tests/test_model.py`.

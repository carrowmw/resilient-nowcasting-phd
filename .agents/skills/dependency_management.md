# Skill: Sync or Sink Dependency Management

**Context:** We follow a strict "Zero Host Download" and "Sync or Sink" policy. Virtual environments (`venv`) on the host are strictly forbidden.

**Action:** 1. To install a Python package: `docker compose exec [service_name] pip install [package_name]` 2. **Crucial Step (Sync or Sink):** Immediately after installing a Python package, you MUST append the package name to `requirements.txt` in the root directory to ensure future container builds do not break. 3. To install a Node package: `docker compose exec [frontend_service] npm install [package_name]` (this will automatically update `package.json`).

# Agentic Workflow Summary

## Phase 1: Infrastructure & Security

The foundation of this workflow is an isolated, disposable environment where your code is safe, but the agent's memory is intentionally volatile.

- The Unified Dockerfile: Your image is built on a lightweight Python base. It installs system dependencies, Node.js (for Gemini CLI), and native curl binaries (for Claude Code), alongside your Python ML requirements.

- Volume Mounts (The Portal): You map your local Mac directory to /workspace inside the container. This ensures your code, data, and models are permanently saved on your host machine, while the agent edits them live from inside the container.

- Security (.env & .gitignore): API keys are never baked into the Docker image or stored globally in ~/.zshrc. They are strictly kept in a local .env file, which is added to .gitignore to prevent accidental GitHub leaks.

- The --rm Mandate: You boot the environment using docker compose run --rm agent-workspace bash. The --rm flag is critical: it ensures that the moment you exit the shell, the container and the agent's internal chat history are completely destroyed, giving you a pristine state for your next session.

## Phase 2: The Agent's Operating System (System Prompt)

The agent is governed by a strict system prompt (saved as GEMINI.md or CLAUDE.md in your workspace) that establishes its rules of engagement.

- Plan First: The agent is forbidden from writing complex execution code immediately. It must first write a step-by-step architectural plan to plan.md and wait for human approval.

- Safety First (Git): The agent must execute git commit to create local rollback points before refactoring working code or beginning new major tasks.

- The Experiment Ledger: The agent is mandated to maintain an EXPERIMENTS.md file, logging the date, active branch, prompt used, and the mathematical outcome of every model training run.

## Phase 3: The "Offensive ML" Coding Paradigm

Machine learning prototyping requires "Offensive Programming." The agent is instructed to write code that fails fast and hard rather than silently coercing bad data.

- Ruthless Assertions: The agent must explicitly assert tensor shapes, dataframe columns, and exact data types at the top of every major function.

- Let it Burn: Generic except: blocks are banned. If a matrix dimension mismatches, the script must crash immediately and print the full traceback for diagnosis.

- NaN Tripwires: The agent must place explicit checks after transformations to crash the script if unexpected NaNs or infinite values are introduced.

- No Silent Drops: Functions like .dropna() are banned unless explicitly approved in the data cleaning plan.

## Phase 4: The Execution Loop (Human-in-the-Loop Orchestration)

You act as the Senior Engineer; the AI acts as the Junior Developer. You do not ask the agent to "build a pipeline." You micro-batch its tasks.

- The Blind EDA: You begin by asking the agent to analyze the raw data without telling it what the data is or what your goal is. This forces the agent to read the literal matrix (identifying true anomalies like time-series gaps) rather than hallucinating a perfect dataset.

- The Common Preprocessing Pipeline: Based on the EDA, the agent builds a single, hardened data pipeline (handling resampling, imputation, and spatial joins) that outputs clean matrices (e.g., .npy files). This is committed to the main Git branch.

- The Crash-and-Fix Loop: When the agent writes a script, you run it. When it crashes (due to your offensive programming assertions), you do not fix it yourself. You paste the exact traceback back to the agent and instruct it to resolve the anomaly.

## Phase 5: State Management (Git Branching & Memory Wipes)

To prevent the agent's context window from degrading into a polluted mess of broken PyTorch tensors and overlapping logic, you strictly sandbox your experiments.

- Model Sandboxing: Every new model architecture (e.g., XGBoost, Random Forest, Neural Networks) gets its own Git branch branched off main.

- The Context Wipe: Before starting a new branch or moving to a radically new step in the pipeline, you explicitly clear the agent's memory (using /clear or restarting the CLI).

- Re-Grounding: Upon starting a fresh branch or cleared session, you point the agent to the current working state of the files and the EXPERIMENTS.md ledger, ensuring it builds exclusively on clean, working logic.

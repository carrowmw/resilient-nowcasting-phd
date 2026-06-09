---
description: Analyzes the codebase and generates a Mermaid.js architecture diagram and connection manifest.
---

1. Analyze the core system architecture. Limit scope to system-level components, primary classes, and major API boundaries. Ignore helper functions and configuration files.
   - **Data Flow Directive:** When mapping the data pipeline, map the logical flow of data (Ingestion -> Feature Engineering -> Preprocessing -> Model Integration) rather than just file-to-file dependencies.
   - Ensure you capture distinct preprocessing steps (e.g., Scaling vs. Masking) and highlight variables that bypass certain steps (e.g., raw target counts must bypass standard scalers). Include critical architectural notes as detached nodes (e.g., preventing spatial leakage).
   - **Model Internal Flow Directive:** For the ResilientIoTPipeline model, map the detailed internal tensor flows and logical sub-components (e.g., how DynamicGraphLearner computes the sparse adjacency matrix, how MaskedSpatiotemporalEncoder applies Spatial GATv2Conv followed by Temporal Transformers, and how DecoupledDecoders branch into Imputation vs. ZINB Forecasting) rather than just listing the class files.
2. Check if an existing `.mmd` diagram file exists in the root directory. If so, read it to establish the baseline architecture.
3. Dynamically invoke subagents to analyze the primary application directories in parallel to summarize their current state and dependencies.
4. Compare the current codebase state against the baseline (if it exists) to determine what connections have changed.
5. Output the result strictly in the following JSON schema:
   {
     "diagram_code": "The raw Mermaid.js graph TD code",
     "human_instructions": "A bulleted changelog of what nodes/edges were added, removed, or modified."
   }

# Post-processing Hook
> Pipe output to: `python3 .agents/scripts/process_diagram.py`
#!/usr/bin/env python3
import os
import sys
import json
from datetime import datetime

def main():
    # Read the JSON payload from standard input (piped from Antigravity)
    try:
        payload = json.load(sys.stdin)
    except json.JSONDecodeError as e:
        print(f"Error: Invalid JSON input received. {e}", file=sys.stderr)
        sys.exit(1)

    diagram_code = payload.get("diagram_code")
    human_instructions = payload.get("human_instructions")

    if not diagram_code or not human_instructions:
        print("Error: Missing 'diagram_code' or 'human_instructions' in payload.", file=sys.stderr)
        sys.exit(1)

    # 1. Update the current Mermaid diagram file
    diagram_dir = "docs/diagrams" if os.path.exists("docs/diagrams") else "docs/architecture"
    os.makedirs(diagram_dir, exist_ok=True)
    diagram_filename = os.path.join(diagram_dir, "architecture_current.mmd")
    with open(diagram_filename, "w", encoding="utf-8") as f:
        f.write(diagram_code)
    print(f"Successfully updated {diagram_filename}")

    # 2. Append to the Architecture Changelog
    changelog_filename = os.path.join(diagram_dir, "ARCHITECTURE_CHANGELOG.md")

    # Create file with a header if it doesn't exist
    if not os.path.exists(changelog_filename):
        with open(changelog_filename, "w", encoding="utf-8") as f:
            f.write("# Architecture Evolution Changelog\n\n")

    current_time = datetime.now().strftime("%Y-%m-%d %H:%M")

    changelog_entry = f"## Update: {current_time}\n\n{human_instructions}\n\n---\n\n"

    # Prepend or append? Usually, appending or writing to top is preferred.
    # Let's prepend right under the main header for easy reading.
    with open(changelog_filename, "r+", encoding="utf-8") as f:
        content = f.read()
        # Find the main header to place the new content right below it
        header_marker = "# Architecture Evolution Changelog\n\n"
        if content.startswith(header_marker):
            body = content[len(header_marker):]
            f.seek(len(header_marker))
            f.write(changelog_entry + body)
        else:
            # Fallback to append if header was modified
            f.seek(0, 2)
            f.write(changelog_entry)

    print(f"Successfully appended entry to {changelog_filename}")

if __name__ == "__main__":
    main()
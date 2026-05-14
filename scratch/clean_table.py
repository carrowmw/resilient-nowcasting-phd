import re

def clean_experiments():
    with open('EXPERIMENTS.md', 'r') as f:
        lines = f.readlines()

    # Header and alignment rows
    header = "| Date       | Branch | Core Prompt / Goal | Outcome |\n"
    alignment = "| :--------- | :----- | :----------------- | :------ |\n"
    
    # Extract data rows
    data_rows = []
    
    # Skip the title and existing header/alignment
    for line in lines[4:]:
        line = line.strip()
        if not line:
            continue
        
        # Split by | and filter out empty strings
        # But be careful with pipes inside the content (though markdown tables use them as separators)
        # Markdown tables require escaping pipes if they are content.
        # Let's see how they are separated.
        
        # If a line has multiple | 2026- it might be merged
        # Use regex to find start of rows: | 2026-
        sub_rows = re.split(r'(?=\| 202\d-\d{2}-\d{2})', line)
        for row in sub_rows:
            row = row.strip()
            if not row or not row.startswith('|'):
                continue
            
            # Count pipes. A valid row should have at least 5 pipes for 4 columns.
            # | Date | Branch | Goal | Outcome |
            # However, some outcomes might have pipes in them (bad practice but possible).
            # The user's table seems to have 4 columns.
            
            parts = row.split('|')
            # parts[0] is empty (before first pipe)
            # parts[1] is Date
            # parts[2] is Branch
            # parts[3] is Goal
            # parts[4:] is Outcome (merged back if there were pipes in outcome)
            
            if len(parts) >= 5:
                date = parts[1].strip()
                branch = parts[2].strip()
                goal = parts[3].strip()
                outcome = "|".join(parts[4:]).strip()
                # Remove trailing | if it exists
                if outcome.endswith('|'):
                    outcome = outcome[:-1].strip()
                
                # Check if date is valid format
                if re.match(r'202\d-\d{2}-\d{2}', date):
                    data_rows.append((date, branch, goal, outcome))

    # Sort data rows by date
    data_rows.sort(key=lambda x: x[0])

    # Construct final content
    new_content = [
        "# Machine Learning Experiment Ledger\n",
        "\n",
        header,
        alignment
    ]
    
    for row in data_rows:
        new_content.append(f"| {row[0]} | {row[1]} | {row[2]} | {row[3]} |\n")

    with open('EXPERIMENTS.md', 'w') as f:
        f.writelines(new_content)

if __name__ == "__main__":
    clean_experiments()

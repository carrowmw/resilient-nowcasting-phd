import re

def clean_outcome(text):
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove sequences of pipes that are clearly garbage (more than 2)
    text = re.sub(r'\|{2,}', '|', text)
    return text.strip()

def clean_experiments():
    with open('EXPERIMENTS.md', 'r') as f:
        content = f.read()

    # Find all potential rows starting with | 202x-xx-xx
    # We use a non-greedy match for the content until the next row or end of string
    pattern = r'\| (202\d-\d{2}-\d{2}) \| (.*?) \| (.*?) \| (.*?)(?=\n\| 202\d|$)'
    matches = re.findall(pattern, content, re.DOTALL)
    
    data_rows = []
    for date, branch, goal, outcome in matches:
        branch = branch.strip()
        goal = goal.strip()
        # Outcome might contain internal pipes if it was merged or corrupted
        # We'll clean it up
        outcome = clean_outcome(outcome)
        # If outcome ends with a pipe (from the table structure), remove it
        if outcome.endswith('|'):
            outcome = outcome[:-1].strip()
        
        data_rows.append((date, branch, goal, outcome))

    # Sort data rows by date (ascending)
    data_rows.sort(key=lambda x: x[0])

    # Construct final content
    header = "| Date       | Branch | Core Prompt / Goal | Outcome |\n"
    alignment = "| :--------- | :----- | :----------------- | :------ |\n"
    
    new_content = [
        "# Machine Learning Experiment Ledger\n",
        "\n",
        header,
        alignment
    ]
    
    for row in data_rows:
        # Final safety check on each field to remove any remaining garbage pipes
        date = row[0].strip()
        branch = row[1].strip('| ').strip()
        goal = row[2].strip('| ').strip()
        outcome = row[3].strip('| ').strip()
        
        new_content.append(f"| {date} | {branch} | {goal} | {outcome} |\n")

    with open('EXPERIMENTS.md', 'w') as f:
        f.writelines(new_content)

if __name__ == "__main__":
    clean_experiments()

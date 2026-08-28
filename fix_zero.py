import os

filepath = '/Users/administrator/Code/ml-agent-workspace/docs/mdpi_sensors/mdpi_sensors.tex'
with open(filepath, 'r') as f:
    content = f.read()

# Replacement 1 (Line 353)
target1 = r"Practitioners should therefore choose an interpolation strategy according to the error metric that matters for their application and, critically, avoid night-zero filling for longer-horizon forecasts."
rep1 = r"Practitioners should therefore choose an interpolation strategy according to the error metric that matters for their application. Critically, the failure of night-time zero-filling highlights a hazard specific to event-triggered sensors: when missing data arises from hardware configuration (e.g., deactivating during idle periods) rather than confirmed zero events, naive zero-filling injects spurious patterns that degrade longer-horizon forecasts."

# Replacement 2 (Line 462)
target2 = r"The practical recommendation is to prefer simple, faithful interpolation (e.g.\ linear) where the target metric rewards it, and to avoid zero-imputation of overnight gaps entirely."
rep2 = r"The practical recommendation for event-triggered networks is to prefer simple, faithful interpolation (e.g.\ linear) where the target metric rewards it, and to avoid zero-imputing gaps that actually represent sensor inactivity rather than confirmed zero traffic."

# Replacement 3 (Line 474)
target3 = r"even though naive zero-filling cannot."
rep3 = r"whereas naively zero-filling periods of sensor inactivity proves actively harmful."


changes_made = 0
if target1 in content:
    content = content.replace(target1, rep1)
    changes_made += 1
else:
    print("target1 not found")

if target2 in content:
    content = content.replace(target2, rep2)
    changes_made += 1
else:
    print("target2 not found")

if target3 in content:
    content = content.replace(target3, rep3)
    changes_made += 1
else:
    print("target3 not found")

if changes_made == 3:
    print("Success")
    with open(filepath, 'w') as f:
        f.write(content)
else:
    print(f"Failed. Only {changes_made} changes made.")

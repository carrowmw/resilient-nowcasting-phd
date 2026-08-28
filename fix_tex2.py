import os

filepath = '/Users/administrator/Code/ml-agent-workspace/docs/mdpi_sensors/mdpi_sensors.tex'
with open(filepath, 'r') as f:
    content = f.read()

target = r"By holding these previously tuned architectures constant in this study, we explicitly test the models' ability to generalise to the much broader, noisier distribution of data quality found across the entire network."
replacement = target + r" While this approach risks favouring high-quality sensors, it serves as a rigorous stress test: it establishes an `upper bound' of model capability on ideal data, allowing us to isolate exactly how and why performance degrades when this ideally tuned model is exposed to the severe sequence fragmentation typical of real-world deployments."

if target in content:
    content = content.replace(target, replacement)
    print("Success")
    with open(filepath, 'w') as f:
        f.write(content)
else:
    print("Target not found")

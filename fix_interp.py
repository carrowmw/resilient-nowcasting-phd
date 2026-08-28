import os

filepath = '/Users/administrator/Code/ml-agent-workspace/docs/mdpi_sensors/mdpi_sensors.tex'
with open(filepath, 'r') as f:
    content = f.read()

target = r"The practical recommendation is to prefer simple, faithful interpolation (e.g.\ linear) where the target metric rewards it, and to avoid zero-imputation of overnight gaps entirely."

replacement = target + r" Ultimately, these findings underscore the value of developing predictive models that do not strictly require continuous inputs. While principled interpolation can yield improvements at short horizons, any form of data imputation risks introducing artefacts---such as the spurious overnight patterns observed here---that actively mislead models over longer windows. Forecasting architectures capable of natively handling irregular sampling or explicit missingness (such as time-aware attention mechanisms or specialized RNNs) would avoid the need to hallucinate data entirely, preserving the fidelity of the raw pedestrian signal while naturally bridging temporal gaps."

if target in content:
    content = content.replace(target, replacement)
    print("Success")
    with open(filepath, 'w') as f:
        f.write(content)
else:
    print("Target not found")

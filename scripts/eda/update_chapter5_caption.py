#!/usr/bin/env python3
import sys

path = '/Users/administrator/Code/latex-documents/thesis/chapters/chapter5_experiment1_single_node.tex'
with open(path, 'r', encoding='utf-8') as f:
    lines = f.readlines()

new_caption = (
    '    \\caption[Fixed-size data loading window and direct point forecast]{'
    'Example fixed-size window used in data loading ($W=24$, $h=24$). '
    'The primary sensor reading (blue solid line with markers) is only observed within the historical input window ($n \\in [0, 23]$, 6~hours). '
    'A deterministic temporal covariate (green dashed line, e.g.\\ harmonic calendar encoding) is known in advance across both the input window and the forecast lead time. '
    'The model performs direct point forecasting, predicting a single target point at lead time $t+h$ (red star, $n=47$), '
    'while leaving the intermediate horizon gap ($h=24$ steps) unobserved and unpredicted.}\n'
)

for i, line in enumerate(lines):
    if '\\caption' in line and 'dataloader_example_training_sequence' in lines[i+1]:
        lines[i] = new_caption
        print(f"Replaced line {i+1}")
        break

with open(path, 'w', encoding='utf-8') as f:
    f.writelines(lines)

print("Finished updating caption.")

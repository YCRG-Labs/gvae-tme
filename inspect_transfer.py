#!/usr/bin/env python3
"""Quick inspection of transfer_joint_results.json structure and key metrics."""
import json
import sys
from pathlib import Path

path = Path('outputs/melanoma_to_nsclc_ici/transfer_joint/transfer_joint_results.json')
if len(sys.argv) > 1:
    path = Path(sys.argv[1])

with open(path) as f:
    r = json.load(f)

print(f'File: {path}')
print(f'Top-level keys: {list(r.keys())}')
print()

for k, v in r.items():
    if k == 'shared_genes':
        print(f'  {k}: [{len(v)} genes]')
    elif isinstance(v, dict):
        print(f'  {k}: {list(v.keys())}')
    elif isinstance(v, list):
        print(f'  {k}: [{len(v)} items]')
    else:
        print(f'  {k}: {v}')

print()
print('=== Key metrics ===')
for section in ['source_downstream', 'transfer_evaluation', 'target_evaluation',
                'clustering', 'rare_cells', 'marker_concordance']:
    if section in r:
        print(f'\n{section}:')
        v = r[section]
        if isinstance(v, dict):
            for kk, vv in v.items():
                if isinstance(vv, (int, float, str, bool)):
                    print(f'  {kk}: {vv}')
                elif isinstance(vv, dict):
                    print(f'  {kk}: {{...{len(vv)} keys}}')
                elif isinstance(vv, list):
                    print(f'  {kk}: [{len(vv)} items]')

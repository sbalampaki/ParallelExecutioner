#!/usr/bin/env python3
"""Build the variant table from skewcast sweep JSONs."""
import glob
import json
import os

ORDER = ['Wfull', 'W6', 'W12', 'W24', 'W48',
         'shape_W24_trim0', 'shape_W24_trim1', 'shape_W24_trim5',
         'shape_W24_trim10',
         'sizematched_W6', 'sizematched_W12', 'sizematched_W24',
         'sizematched_W48', 'sizematched_Wfull']

PS = [16, 32, 64, 80]

rows = {}
for f in glob.glob('sweep_*.json'):
    d = json.load(open(f))
    v = os.path.basename(f)[len('sweep_'):-len('.json')]
    rows[v] = d

if not rows:
    raise SystemExit('no sweep_*.json found in this directory')

cols = [f'C({p})' for p in PS]
hdr = (f"{'variant':<20}{'n':>8}{'N':>14}{'CV':>7}{'Gini':>7}"
       f"{'p*':>7}{'p_atom':>9}" + ''.join(f'{c:>9}' for c in cols))
print(hdr)
print('-' * len(hdr))

ordered = [v for v in ORDER if v in rows] + \
          [v for v in sorted(rows) if v not in ORDER]

for v in ordered:
    d = rows[v]
    c = [s['ceiling'] * 100 if s['ceiling'] is not None else float('nan')
         for s in d['sweep']]
    ps = d['p_star'] if d['p_star'] else 0
    print(f"{v:<20}{d['n_units']:>8,}{d['n_records']:>14,}"
          f"{d['cv']:>7.3f}{d['gini']:>7.3f}{ps:>7,}{d['p_atom']:>9,.0f}"
          + ''.join(f'{x:>8.2f}%' for x in c))

print('\n% --- LaTeX rows ---')
for v in ordered:
    d = rows[v]
    c = [s['ceiling'] * 100 if s['ceiling'] is not None else 0
         for s in d['sweep']]
    name = v.replace('_', r'\_')
    print(f"\\texttt{{{name}}} & {d['n_units']:,} & {d['n_records']:,} & "
          f"{d['cv']:.3f} & {d['gini']:.3f} & {d['p_star']} & "
          f"{d['p_atom']:,.0f} & "
          + ' & '.join(f'{x:.2f}' for x in c) + r' \\')

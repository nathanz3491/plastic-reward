"""Stage 3 test: do evolved taste rules hold up in worlds they never saw?

Every candidate meets the same fresh lifetimes in every world, so all
comparisons are paired. The candidates:
  external      no taste
  hand_set      the stage-1 once-bitten rule
  <name>        each champion given as name=path/to/champions.json
                (the 'evolved' entry; 'random_search' too if present)

The worlds:
  - two flips (A -> B -> A) at poison delays 0, 4, 15, 20, 30 (4, 20 and 30
    are in no training mix);
  - one flip and three flips, at delays 15 and 20;
  - different brain sizes (GRU 64 and 256, two flips, delay 15).

  python run_stage3_test.py stage2=runs/stage2/champions.json \
      mix0=runs/mix0/champions.json --out runs/stage3_test
"""

import argparse
import dataclasses
import json
import math
import pathlib
import time

import jax
import numpy as np

from plastic_reward import ppo
from plastic_reward import reward as R

HAND_SET = dict(w0=0.2, alpha=0.5, rho=0.01, pos=1.0, neg=5.0, decay=0.5, lam=3.0)
ONE = ((0.3, 0.4),)
TWO = ((0.25, 0.35), (0.6, 0.7))
THREE = ((0.2, 0.28), (0.45, 0.53), (0.7, 0.78))
GENE_KEYS = ('w0', 'alpha', 'rho', 'pos', 'neg', 'decay', 'lam')


def worlds(base):
  w = {}
  for d in (0, 4, 15, 20, 30):
    w[f'2 flips, delay {d}'] = dataclasses.replace(base, flips=TWO, delays=(d,))
  for name, flips in (('1 flip', ONE), ('3 flips', THREE)):
    for d in (15, 20):
      w[f'{name}, delay {d}'] = dataclasses.replace(base, flips=flips, delays=(d,))
  for h in (64, 256):
    w[f'GRU {h}, 2 flips, delay 15'] = dataclasses.replace(
        base, flips=TWO, delays=(15,), hidden=h)
  return w


def candidates(specs, n):
  genes = dict(external=R.make_gene(n, lam=0.0),
               hand_set=R.make_gene(n, **HAND_SET))
  for spec in specs:
    name, path = spec.split('=', 1)
    champ = json.loads(pathlib.Path(path).read_text())
    genes[name] = R.make_gene(n, **{k: champ['evolved'][k] for k in GENE_KEYS})
    if 'random_search' in champ:
      genes[f'{name}_random'] = R.make_gene(
          n, **{k: champ['random_search'][k] for k in GENE_KEYS})
  return genes


def mse(x):
  return [float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))]


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('champions', nargs='*', help='name=path/to/champions.json')
  ap.add_argument('--lifetimes', type=int, default=16)
  ap.add_argument('--updates', type=int, default=300)
  ap.add_argument('--envs', type=int, default=64)
  ap.add_argument('--seed', type=int, default=31_337)
  ap.add_argument('--types', type=int, default=3, help='object types in the world')
  ap.add_argument('--per-type', type=int, default=3, help='objects of each type')
  ap.add_argument('--buttons', type=int, default=0, help='extra button actions')
  ap.add_argument('--events', default='objects', choices=('objects', 'buttons'),
                  help='what taste is about')
  ap.add_argument('--out', default='runs/stage3_test')
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  keys = jax.random.split(jax.random.PRNGKey(args.seed), args.lifetimes)
  base = ppo.Config(n_envs=args.envs, n_updates=args.updates,
                    n_types=args.types, per_type=args.per_type,
                    n_buttons=args.buttons, events=args.events)
  genes = candidates(args.champions, base.n_events)

  table = {}
  for wname, cfg in worlds(base).items():
    t0 = time.time()
    whole = {}
    for gname, gene in genes.items():
      fn = jax.jit(jax.vmap(lambda k, g=gene, c=cfg: ppo.train_lifetime(c, g, k)[0]))
      whole[gname] = np.asarray(fn(keys)['lifespan']).mean(1)
    row = {}
    for gname, x in whole.items():
      # per_life keeps every lifetime, so any two candidates can be paired
      row[gname] = dict(whole=mse(x), per_life=x.tolist())
      if gname != 'external':
        d = x - whole['external']
        row[gname]['vs_external'] = mse(d) + [int((d > 0).sum())]
    table[wname] = row
    print(wname, f'{(time.time() - t0) / 60:.1f} min', json.dumps(
        {g: round(r['vs_external'][0], 1) for g, r in row.items()
         if 'vs_external' in r}), flush=True)
    (out / 'test.json').write_text(json.dumps(dict(
        note='whole = mean lifespan over a lifetime, [mean, SE]; vs_external '
             'adds lifetimes won out of --lifetimes',
        candidates=list(genes), table=table), indent=1))
  plot(table, list(genes), out / 'stage3.png')
  print('wrote', out)


def plot(table, names, path):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  names = [n for n in names if n != 'external']
  wn = list(table)
  m = np.array([[table[w][n]['vs_external'][0] for w in wn] for n in names])
  t = np.array([[table[w][n]['vs_external'][0] / max(table[w][n]['vs_external'][1], 1e-6)
                 for w in wn] for n in names])
  fig, ax = plt.subplots(figsize=(1.1 * len(wn) + 3, 0.55 * len(names) + 2))
  lim = np.abs(m).max()
  im = ax.imshow(m, cmap='RdBu', vmin=-lim, vmax=lim, aspect='auto')
  for i in range(len(names)):
    for j in range(len(wn)):
      ax.text(j, i, f'{m[i, j]:+.0f}' + ('*' if abs(t[i, j]) > 2 else ''),
              ha='center', va='center', fontsize=8)
  ax.set_xticks(range(len(wn)), wn, rotation=40, ha='right', fontsize=8)
  ax.set_yticks(range(len(names)), names, fontsize=8)
  ax.set_title('lifespan gain over no taste (paired; * = beyond 2 SE)')
  fig.colorbar(im, ax=ax, shrink=0.8)
  fig.tight_layout()
  fig.savefig(path, dpi=130)


if __name__ == '__main__':
  main()

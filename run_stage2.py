"""Stage 2: does evolution, selecting on survival alone, find a good taste rule?

1. Evolution: a population of random rules, each scored on the mean lifespan
   over a few lifetimes in the delayed-poison world. The best are kept and
   mutated, generation after generation.
2. Control: random search over the same rule family with the same number of
   evaluations. Does selection beat just trying random rules?
3. Champions: the top few of each search are re-measured on fresh lifetimes
   and the best is kept (a single score can be a lucky draw).
4. Held-out test: the champions against external-only and the hand-set
   once-bitten rule, on fresh lifetimes. The worlds are the training one
   (poison delay 15) plus delays never seen in evolution (0, 8, 25).

Writes <out>/history.json, champions.json, test.json and stage2.png.

  python run_stage2.py --out runs/stage2
"""

import argparse
import dataclasses
import json
import math
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from plastic_reward import evolve as EV
from plastic_reward import ppo
from plastic_reward import reward as R

HAND_SET = dict(w0=0.2, alpha=0.5, rho=0.01, pos=1.0, neg=5.0, decay=0.5, lam=3.0)
TEST_DELAYS = (15, 0, 8, 25)
WINDOW = 30


def search(evaluate, rng, args, evolve):
  """Evolution (evolve=True) or random search, same evaluation budget."""
  genomes = EV.random_genomes(rng, args.pop)
  history, pool = [], []
  for gen in range(args.gens):
    t0 = time.time()
    if not evolve and gen > 0:
      genomes = EV.random_genomes(rng, args.pop)
    key = jax.random.PRNGKey(args.seed * 100_000 + gen)   # same lifetimes for all
    fit = evaluate(genomes, key)
    best = int(np.argmax(fit))
    pool += [(float(f), g) for f, g in zip(fit, genomes)]
    desc = [EV.describe(g) for g in genomes]
    history.append(dict(
        gen=gen, best=float(fit[best]), mean=float(fit.mean()),
        best_gene=desc[best],
        pop_median={k: float(np.median([d[k] for d in desc]))
                    for k in ('neg_over_pos', 'decay', 'lam', 'alpha')},
        minutes=round((time.time() - t0) / 60, 2)))
    print(('evo' if evolve else 'rnd'), json.dumps(history[-1]), flush=True)
    if evolve:
      genomes = EV.next_generation(rng, genomes, fit, args.elites,
                                   args.immigrants, args.sigma)
  return history, pool


def pick_champion(evaluate, candidates, args):
  """Re-measure candidates on fresh lifetimes; the best mean wins."""
  key = jax.random.PRNGKey(args.seed * 100_000 + 77_777)
  fit = evaluate(np.stack(candidates), key)
  return candidates[int(np.argmax(fit))], fit.tolist()


def test(cfg, genes, n, seed):
  """Per-lifetime lifespans for each named Gene on n fresh lifetimes."""
  keys = jax.random.split(jax.random.PRNGKey(seed), n)
  out = {}
  for name, gene in genes.items():
    fn = jax.jit(jax.vmap(lambda k, g=gene: ppo.train_lifetime(cfg, g, k)[0]))
    logs = jax.tree.map(np.asarray, fn(keys))
    L, flip = logs['lifespan'], logs['flipped'].argmax(1)
    out[name] = dict(
        whole=L.mean(1),
        post=np.array([L[i, f:].mean() for i, f in enumerate(flip)]),
        last=L[:, -WINDOW:].mean(1))
  return out


def mse(x):
  return [float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))]


def plot(evo_h, rnd_h, table, path):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
  g = [h['gen'] for h in evo_h]
  ax[0].plot(g, [h['best'] for h in evo_h], color='#1baf7a', label='evolution: best')
  ax[0].plot(g, [h['mean'] for h in evo_h], color='#1baf7a', ls=':',
             label='evolution: population mean')
  ax[0].plot(g, np.maximum.accumulate([h['best'] for h in rnd_h]),
             color='#52514e', label='random search: best so far')
  ax[0].set(title='fitness (mean lifespan over a lifetime)', xlabel='generation')
  ax[0].legend(frameon=False, fontsize=8)
  for k, c in (('neg_over_pos', '#b5651d'), ('decay', '#2a78d6'), ('lam', '#8a5cd6')):
    ax[1].plot(g, [h['pop_median'][k] for h in evo_h], color=c, label=k)
  ax[1].set(title='population median of the rule', xlabel='generation',
            yscale='log')
  ax[1].legend(frameon=False, fontsize=8)
  names = list(next(iter(table.values())).keys())
  delays = list(table.keys())
  width = 0.8 / len(names)
  colors = dict(external='#52514e', hand_set='#e8a33d', evolved='#1baf7a',
                random_search='#9ab8e0')
  for i, n in enumerate(names):
    m = [table[d][n]['whole'][0] for d in delays]
    e = [table[d][n]['whole'][1] for d in delays]
    ax[2].bar(np.arange(len(delays)) + i * width, m, width, yerr=e,
              color=colors.get(n), label=n)
  ax[2].set_xticks(np.arange(len(delays)) + 0.4 - width / 2,
                   [f'delay {d}' + (' (trained)' if d == 15 else '') for d in delays])
  ax[2].set(title='held-out lifetimes: mean lifespan', ylabel='steps')
  ax[2].legend(frameon=False, fontsize=8)
  fig.tight_layout()
  fig.savefig(path, dpi=130)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--pop', type=int, default=16)
  ap.add_argument('--gens', type=int, default=25)
  ap.add_argument('--lifetimes', type=int, default=3, help='per genome per generation')
  ap.add_argument('--elites', type=int, default=4)
  ap.add_argument('--immigrants', type=int, default=2)
  ap.add_argument('--sigma', type=float, default=0.3)
  ap.add_argument('--delay', type=int, default=15)
  ap.add_argument('--updates', type=int, default=300)
  ap.add_argument('--envs', type=int, default=64)
  ap.add_argument('--chunk', type=int, default=8, help='genomes per GPU batch')
  ap.add_argument('--finalists', type=int, default=4)
  ap.add_argument('--test-lifetimes', type=int, default=16)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--out', default='runs/stage2')
  args = ap.parse_args()
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  cfg = ppo.Config(n_envs=args.envs, n_updates=args.updates, delay=args.delay,
                   flip_lo=0.3, flip_hi=0.4)
  evaluate = EV.make_evaluator(cfg, args.lifetimes, args.chunk)

  evo_h, evo_pool = search(evaluate, np.random.default_rng(args.seed), args, True)
  rnd_h, rnd_pool = search(evaluate, np.random.default_rng(args.seed + 1), args, False)
  (out / 'history.json').write_text(json.dumps(dict(
      evolution=evo_h, random_search=rnd_h, args=vars(args)), indent=1))

  def top(pool):
    return [g for _, g in sorted(pool, key=lambda p: -p[0])[:args.finalists]]
  pick_eval = EV.make_evaluator(cfg, 8, args.chunk)
  evo_champ, evo_fit = pick_champion(pick_eval, top(evo_pool), args)
  rnd_champ, rnd_fit = pick_champion(pick_eval, top(rnd_pool), args)
  champions = dict(evolved=EV.describe(evo_champ), evolved_finalists=evo_fit,
                   random_search=EV.describe(rnd_champ),
                   random_finalists=rnd_fit, hand_set=HAND_SET)
  (out / 'champions.json').write_text(json.dumps(champions, indent=1))
  print('champions', json.dumps(champions), flush=True)

  genes = dict(external=R.make_gene(3, lam=0.0),
               hand_set=R.make_gene(3, **HAND_SET),
               evolved=EV.decode(evo_champ), random_search=EV.decode(rnd_champ))
  table = {}
  for d in TEST_DELAYS:
    res = test(dataclasses.replace(cfg, delay=d), genes, args.test_lifetimes,
               seed=args.seed + 9_999)
    table[d] = {}
    for name, r in res.items():
      row = {k: mse(v) for k, v in r.items()}
      if name != 'external':
        diff = r['whole'] - res['external']['whole']
        row['whole_vs_external'] = mse(diff) + [int((diff > 0).sum())]
      table[d][name] = row
    print('test delay', d, json.dumps(table[d]), flush=True)
  (out / 'test.json').write_text(json.dumps(dict(
      note='[mean, SE over lifetimes]; *_vs_external adds lifetimes won',
      table=table), indent=1))
  plot(evo_h, rnd_h, table, out / 'stage2.png')
  print('wrote', out)


if __name__ == '__main__':
  main()

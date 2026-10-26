"""Stage 1: does the intrinsic reward follow the evidence, and does it help?

Arms run over the same lifetimes (same seeds, so the same world rules and
flip times), so every comparison is paired lifetime by lifetime:
  external     external reward only
  fixed3       plus a fixed intrinsic reward for touching anything (w0 0.2,
               lam 3) -- tests whether the intrinsic channel matters at all
  plastic1/3/10  the same start, revised from evidence by a hand-set rule,
               at intrinsic weight lam 1, 3, 10

Writes <out>/logs.npz, <out>/summary.json and <out>/stage1.png.

  python run_stage1.py --lifetimes 16 --updates 300 --delay 15 --out runs/delayed
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

PLASTIC = dict(w0=0.2, alpha=0.5, rho=0.01, decay=0.9)
# "Once bitten": bad evidence counts 5x and old evidence fades fast, the way one
# bad meal turns an animal off a food while liking it takes many good ones.
AVERSE = dict(w0=0.2, alpha=0.5, rho=0.01, pos=1.0, neg=5.0, decay=0.5)
ARMS = {
    'external': dict(lam=0.0, gene=dict()),
    'fixed1': dict(lam=1.0, gene=dict(w0=0.2)),
    'fixed3': dict(lam=3.0, gene=dict(w0=0.2)),
    'plastic1': dict(lam=1.0, gene=PLASTIC),
    'plastic3': dict(lam=3.0, gene=PLASTIC),
    'plastic10': dict(lam=10.0, gene=PLASTIC),
    'averse1': dict(lam=1.0, gene=AVERSE),
    'averse3': dict(lam=3.0, gene=AVERSE),
}
DEFAULT_ARMS = 'external,fixed3,plastic1,plastic3,plastic10'
WINDOW = 30           # batches before the flip, and at the end of life
BEFORE = 60           # aligned plots show this many batches before the flip


def run_arm(cfg, arm, keys):
  gene = R.make_gene(cfg.n_events, **arm['gene'])
  c = dataclasses.replace(cfg, lam=arm['lam'])
  fn = jax.jit(jax.vmap(lambda k: ppo.train_lifetime(c, gene, k)))
  logs, life = fn(keys)
  return jax.tree.map(np.asarray, logs), jax.tree.map(np.asarray, life)


def per_life(logs):
  """Per lifetime: lifespan before the flip, after it, and at the end."""
  L, flip = logs['lifespan'], logs['flipped'].argmax(1)
  pre = np.array([L[i, max(0, f - WINDOW):f].mean() for i, f in enumerate(flip)])
  post = np.array([L[i, f:].mean() for i, f in enumerate(flip)])
  last = L[:, -WINDOW:].mean(1)
  return dict(pre=pre, post=post, last=last, flip=flip)


def mse(x):
  return float(x.mean()), float(x.std(ddof=1) / math.sqrt(len(x)))


def aligned(x, flip, after):
  """(lifetimes, batches) -> (BEFORE + after,) mean over lifetimes, flip at 0."""
  out = np.full((len(x), BEFORE + after), np.nan)
  for i, f in enumerate(flip):
    seg = x[i, max(0, f - BEFORE):f + after]
    start = BEFORE - (f - max(0, f - BEFORE))
    out[i, start:start + len(seg)] = seg
  return np.nanmean(out, 0)


def plot(results, cfg, path, title):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  palette = dict(external='#52514e', fixed1='#9ab8e0', fixed3='#2a78d6',
                 plastic1='#f2c27a', plastic3='#e8a33d', plastic10='#b5651d',
                 averse1='#8fd19e', averse3='#1baf7a')
  after = int(cfg.n_updates * (1 - cfg.flips[0][1]))
  x = np.arange(-BEFORE, after)
  fig, ax = plt.subplots(1, 3, figsize=(16, 4.2))
  for a in ax:
    a.axvline(0, color='#999999', lw=0.8, ls='--')
  for name, (logs, _) in results.items():
    flip = logs['flipped'].argmax(1)
    y = aligned(logs['lifespan'], flip, after)
    ax[0].plot(x, np.convolve(y, np.ones(5) / 5, 'same'), color=palette[name],
               label=name)
    t = logs['touch_role']
    share = t[..., 0] / np.maximum(t.sum(-1), 1e-9)
    ax[2].plot(x, aligned(share, flip, after), color=palette[name], label=name)
    if name.startswith(('plastic', 'averse')):
      w = logs['w_role']
      ax[1].plot(x, aligned(w[..., 0], flip, after), color=palette[name],
                 label=f'{name} food')
      ax[1].plot(x, aligned(w[..., 1], flip, after), color=palette[name],
                 ls='--', label=f'{name} poison')
  ax[0].set(title='lifespan (steps per episode)', xlabel='batches since the flip')
  ax[1].set(title='taste w by current role (solid food, dashed poison)',
            xlabel='batches since the flip')
  ax[1].axhline(0, color='#cccccc', lw=0.8)
  ax[2].set(title='share of touches that hit FOOD', xlabel='batches since the flip')
  for a in ax:
    a.legend(frameon=False, fontsize=7)
  fig.suptitle(title)
  fig.tight_layout()
  fig.savefig(path, dpi=130)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--lifetimes', type=int, default=16)
  ap.add_argument('--updates', type=int, default=300)
  ap.add_argument('--envs', type=int, default=64)
  ap.add_argument('--delay', type=int, default=0,
                  help='steps until poison damage lands')
  ap.add_argument('--flip-lo', type=float, default=0.3)
  ap.add_argument('--flip-hi', type=float, default=0.4)
  ap.add_argument('--arms', default=DEFAULT_ARMS)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--out', default='runs/stage1')
  args = ap.parse_args()
  cfg = ppo.Config(n_envs=args.envs, n_updates=args.updates,
                   delays=(args.delay,), flips=((args.flip_lo, args.flip_hi),))
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  keys = jax.random.split(jax.random.PRNGKey(args.seed), args.lifetimes)

  results, stats, summary = {}, {}, {}
  for name in args.arms.split(','):
    t0 = time.time()
    logs, life = run_arm(cfg, ARMS[name], keys)
    results[name] = (logs, life)
    stats[name] = per_life(logs)
    summary[name] = {k: mse(stats[name][k]) for k in ('pre', 'post', 'last')}
    summary[name]['minutes'] = round((time.time() - t0) / 60, 1)
    print(name, json.dumps(summary[name]), flush=True)

  if 'external' in stats:
    base = stats['external']
    for name, s in stats.items():
      if name == 'external':
        continue
      for k in ('post', 'last'):
        d = s[k] - base[k]
        summary[name][f'{k}_vs_external'] = mse(d) + (int((d > 0).sum()),)
      print(name, 'vs external: post', summary[name]['post_vs_external'],
            'last', summary[name]['last_vs_external'], flush=True)

  np.savez_compressed(out / 'logs.npz', **{
      f'{arm}/{k}': v for arm, (logs, _) in results.items()
      for k, v in logs.items()})
  (out / 'summary.json').write_text(json.dumps(dict(
      config=dataclasses.asdict(cfg), arms={a: ARMS[a] for a in results},
      lifetimes=args.lifetimes,
      note='values are [mean, SE over lifetimes]; *_vs_external adds the '
           'number of lifetimes where the arm beat external',
      summary=summary), indent=1))
  plot(results, cfg, out / 'stage1.png',
       f'poison delay {args.delay} steps, {args.lifetimes} lifetimes')
  print('wrote', out)


if __name__ == '__main__':
  main()

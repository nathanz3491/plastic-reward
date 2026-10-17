"""Stage 1: does the intrinsic reward follow the evidence, and does it help?

Three arms over the same lifetimes (same seeds, so the same world rules):
  external   external reward only (lam 0)
  fixed      plus a fixed intrinsic reward for touching anything (w0 0.2)
  plastic    the same start, revised from evidence by a hand-set rule

Writes <out>/logs.npz, <out>/summary.json and <out>/stage1.png.

  python run_stage1.py --lifetimes 8 --out runs/stage1
"""

import argparse
import dataclasses
import json
import pathlib
import time

import jax
import jax.numpy as jnp
import numpy as np

from plastic_reward import ppo
from plastic_reward import reward as R

ARMS = {
    'external': dict(lam=0.0, gene=dict()),
    'fixed': dict(lam=1.0, gene=dict(w0=0.2)),
    'plastic': dict(lam=1.0, gene=dict(w0=0.2, alpha=0.5, rho=0.01, decay=0.9)),
}
WINDOW = 30           # batches before / after the flip compared


def run_arm(cfg, arm, keys):
  gene = R.make_gene(3, **arm['gene'])
  c = dataclasses.replace(cfg, lam=arm['lam'])
  fn = jax.jit(jax.vmap(lambda k: ppo.train_lifetime(c, gene, k)))
  logs, life = fn(keys)
  return jax.tree.map(np.asarray, logs), jax.tree.map(np.asarray, life)


def flip_metrics(lifespan, flipped):
  """Per lifetime: lifespan just before vs just after the flip, and recovery."""
  out = []
  for r, f in zip(lifespan, flipped):
    i = int(np.argmax(f))                      # first batch after the flip
    pre = r[max(0, i - WINDOW):i].mean()
    post = r[i:i + WINDOW].mean()
    after = r[i:]
    # batches until a 5-batch average is back at 95% of the pre-flip level
    smooth = np.convolve(after, np.ones(5) / 5, 'valid')
    back = np.flatnonzero(smooth >= 0.95 * pre)
    out.append(dict(pre=pre, post=post, recovery=int(back[0]) if len(back)
                    else len(after)))
  return out


def plot(results, cfg, path):
  import matplotlib
  matplotlib.use('Agg')
  import matplotlib.pyplot as plt
  colors = dict(external='#52514e', fixed='#2a78d6', plastic='#e8a33d')
  roles = ('food', 'poison', 'neutral')
  fig, ax = plt.subplots(1, 3, figsize=(15, 4))
  lo, hi = cfg.flip_lo * cfg.n_updates, cfg.flip_hi * cfg.n_updates
  for a in ax:
    a.axvspan(lo, hi, color='#eeeeea', label='flip falls here')
  for name, (logs, _) in results.items():
    m = logs['lifespan'].mean(0)
    ax[0].plot(np.convolve(m, np.ones(5) / 5, 'valid'), color=colors[name],
               label=name)
  ax[0].set(title='lifespan (steps per episode; external return)',
            xlabel='batch')
  ax[0].legend(frameon=False)
  w = results['plastic'][0]['w_role'].mean(0)
  for i, (role, ls) in enumerate(zip(roles, ('-', '--', ':'))):
    ax[1].plot(w[:, i], color=colors['plastic'], ls=ls, label=role)
  ax[1].axhline(0, color='#cccccc', lw=0.8)
  ax[1].set(title='plastic arm: intrinsic weight by current role',
            xlabel='batch')
  ax[1].legend(frameon=False)
  for name, (logs, _) in results.items():
    t = logs['touch_role'].mean(0)
    ax[2].plot(t[:, 0] / np.maximum(t.sum(-1), 1e-9), color=colors[name],
               label=f'{name}')
  ax[2].set(title='share of touches that hit FOOD (by current role)',
            xlabel='batch')
  ax[2].legend(frameon=False)
  fig.tight_layout()
  fig.savefig(path, dpi=130)


def main():
  ap = argparse.ArgumentParser()
  ap.add_argument('--lifetimes', type=int, default=8)
  ap.add_argument('--updates', type=int, default=300)
  ap.add_argument('--envs', type=int, default=128)
  ap.add_argument('--seed', type=int, default=0)
  ap.add_argument('--out', default='runs/stage1')
  args = ap.parse_args()
  cfg = ppo.Config(n_envs=args.envs, n_updates=args.updates)
  out = pathlib.Path(args.out)
  out.mkdir(parents=True, exist_ok=True)
  keys = jax.random.split(jax.random.PRNGKey(args.seed), args.lifetimes)

  results, summary = {}, {}
  for name, arm in ARMS.items():
    t0 = time.time()
    logs, life = run_arm(cfg, arm, keys)
    results[name] = (logs, life)
    fm = flip_metrics(logs['lifespan'], logs['flipped'])
    summary[name] = dict(
        mean_lifespan=float(logs['lifespan'].mean()),
        pre_flip=float(np.mean([m['pre'] for m in fm])),
        post_flip=float(np.mean([m['post'] for m in fm])),
        recovery_batches=float(np.mean([m['recovery'] for m in fm])),
        final_lifespan=float(logs['lifespan'][:, -WINDOW:].mean()),
        minutes=round((time.time() - t0) / 60, 1))
    print(name, json.dumps(summary[name]), flush=True)

  np.savez_compressed(out / 'logs.npz', **{
      f'{arm}/{k}': v for arm, (logs, _) in results.items()
      for k, v in logs.items()})
  (out / 'summary.json').write_text(json.dumps(dict(
      config=dataclasses.asdict(cfg), arms=ARMS, lifetimes=args.lifetimes,
      summary=summary), indent=1))
  plot(results, cfg, out / 'stage1.png')
  print('wrote', out)


if __name__ == '__main__':
  main()

"""Stage 2: evolve the taste rule, selecting on survival alone.

A genome is 7 numbers in an unbounded space, decoded into a Gene:
  w0     taste at birth                    tanh        -> (-1, 1)
  alpha  how fast evidence moves taste     exp         -> > 0
  rho    pull back toward w0               sigmoid*0.2 -> (0, 0.2)
  pos    weight of good evidence           exp
  neg    weight of bad evidence            exp
  decay  how much old evidence is kept     sigmoid     -> (0, 1)
  lam    how loud taste is                 exp
Mutation is Gaussian in that space, so positive quantities change by ratios.

Fitness is the mean lifespan over a whole lifetime (external performance,
area under the curve), never the agent's own taste reward. Every genome in a
generation meets the same lifetimes (same worlds, same flip times), drawn
fresh each generation so that no genome is rewarded for one lucky world, and
elites are re-measured every generation rather than keeping an old score.
"""

import numpy as np
import jax
import jax.numpy as jnp

from plastic_reward import ppo
from plastic_reward import reward as R

NAMES = ('w0', 'alpha', 'rho', 'pos', 'neg', 'decay', 'lam')


def decode(theta, n_events=3):
  """(..., 7) genome -> Gene (batched over leading axes); one taste per event."""
  t = jnp.asarray(theta, jnp.float32)
  w0 = jnp.tanh(t[..., 0])
  return R.Gene(
      w0=jnp.broadcast_to(w0[..., None], t.shape[:-1] + (n_events,)),
      alpha=jnp.exp(t[..., 1]),
      rho=0.2 * jax.nn.sigmoid(t[..., 2]),
      pos=jnp.exp(t[..., 3]),
      neg=jnp.exp(t[..., 4]),
      decay=jax.nn.sigmoid(t[..., 5]),
      lam=jnp.exp(t[..., 6]))


def describe(theta):
  g = decode(theta)
  return dict(w0=float(g.w0[..., 0]), alpha=float(g.alpha), rho=float(g.rho),
              pos=float(g.pos), neg=float(g.neg), decay=float(g.decay),
              lam=float(g.lam), neg_over_pos=float(g.neg / g.pos))


def random_genomes(rng, n):
  """Broad, neutral prior: no built-in asymmetry between good and bad."""
  return np.stack([
      rng.uniform(-0.5, 0.5, n),                 # w0 ~ (-0.46, 0.46)
      rng.uniform(np.log(0.05), np.log(2.0), n),  # alpha
      rng.uniform(-4.0, 0.0, n),                 # rho ~ (0.004, 0.1)
      rng.normal(0.0, 0.7, n),                   # pos ~ exp(N(0, .7))
      rng.normal(0.0, 0.7, n),                   # neg, same law as pos
      rng.uniform(-1.0, 3.0, n),                 # decay ~ (0.27, 0.95)
      rng.uniform(np.log(0.3), np.log(10.0), n),  # lam
  ], -1).astype(np.float32)


def mutate(rng, theta, sigma):
  return (theta + rng.normal(0.0, sigma, theta.shape)).astype(np.float32)


def make_evaluator(cfg, lifetimes, chunk):
  """genomes (P, 7), key -> fitness (P,), mean lifespan over each lifetime."""
  def one(theta, key):
    keys = jax.random.split(key, lifetimes)
    gene = decode(theta, cfg.n_events)
    logs = jax.vmap(lambda k: ppo.train_lifetime(cfg, gene, k)[0])(keys)
    return logs['lifespan'].mean()
  run = jax.jit(jax.vmap(one, in_axes=(0, None)))

  def evaluate(genomes, key):
    out = [np.asarray(run(jnp.asarray(genomes[i:i + chunk]), key))
           for i in range(0, len(genomes), chunk)]
    return np.concatenate(out)
  return evaluate


def next_generation(rng, genomes, fitness, elites, immigrants, sigma):
  """Keep the elites, fill with mutated top-half parents and fresh randoms."""
  order = np.argsort(-fitness)
  pop = len(genomes)
  keep = genomes[order[:elites]]
  parents = genomes[order[:max(pop // 2, 1)]]
  n_kids = pop - elites - immigrants
  kids = mutate(rng, parents[rng.integers(0, len(parents), n_kids)], sigma)
  return np.concatenate([keep, kids, random_genomes(rng, immigrants)])

"""Plastic intrinsic reward: what is worth touching, revised from experience.

Two things are kept apart:
  w     the CURRENT preference -- how much touching each object type pays as
        intrinsic reward (or, for actions, how much the policy leans towards
        each one; see `shown`). Changes during a lifetime.
  Gene  the RULE that forms and revises w -- fixed for a lifetime, the thing
        evolution will search over (stage 2). Hand-set in stage 1.

The memory is bounded: one slot per event type holding a forgetting count and
the mean evidence. Evidence is associative, not causal: after an event, was
the external return over the next `horizon` steps above the batch's average?
Only windows that close inside the batch are used, so no outcome is read
before it happened.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp


class Gene(NamedTuple):
  w0: jax.Array      # (E,) preference at birth, and where w relaxes back to
  alpha: jax.Array   # () how fast evidence moves w
  rho: jax.Array     # () how fast w relaxes back to w0
  pos: jax.Array     # () sensitivity to good evidence
  neg: jax.Array     # () sensitivity to bad evidence
  decay: jax.Array   # () per batch, how much old evidence is kept (0..1)
  lam: jax.Array     # () how loud taste is next to survival (x Config.lam)


class RewardState(NamedTuple):
  w: jax.Array       # (E,) current intrinsic reward per event type
  n: jax.Array       # (E,) forgetting count of observed events
  eff: jax.Array     # (E,) mean evidence per event type


def make_gene(n_events, w0=0.0, alpha=0.0, rho=0.0, pos=1.0, neg=1.0,
              decay=1.0, lam=1.0):
  f = lambda v: jnp.asarray(v, jnp.float32)
  return Gene(jnp.broadcast_to(f(w0), (n_events,)), f(alpha), f(rho), f(pos),
              f(neg), f(decay), f(lam))


def init(gene):
  z = jnp.zeros_like(gene.w0)
  return RewardState(gene.w0, z, z)


def intrinsic(w, events, clip=1.0):
  """(..., E) event indicators -> (...) intrinsic reward, clipped."""
  return jnp.clip(events @ w, -clip, clip)


def evidence(reward, events, done, horizon):
  """Per event type: summed evidence and count over one batch.

  ``reward`` (T, N) external reward, ``events`` (T, N, E), ``done`` (T, N).
  For each step t whose window [t, t + horizon) fits in the batch, G_t is the
  external return over that window, cut at the end of the episode. Evidence
  for an event at t is G_t minus the mean G over the steps with NO event.
  (Against the mean over all steps, poison deaths dragged the baseline down
  and an object that does nothing looked good.)
  """
  T = reward.shape[0]
  L = T - horizon + 1
  ret = jnp.zeros((L,) + reward.shape[1:], jnp.float32)
  alive = jnp.ones_like(ret)
  for i in range(horizon):
    ret = ret + alive * reward[i:i + L]
    alive = alive * (1.0 - done[i:i + L].astype(jnp.float32))
  ev = events[:L].astype(jnp.float32)
  quiet = 1.0 - ev.sum(-1).clip(0.0, 1.0)
  base = jnp.where(quiet.sum() > 0,
                   (ret * quiet).sum() / jnp.maximum(quiet.sum(), 1.0),
                   ret.mean())
  eff = ret - base
  return (ev * eff[..., None]).sum((0, 1)), ev.sum((0, 1))


def advantage_evidence(adv, events):
  """Per event type: summed advantage and count over one batch.

  ``adv`` (T, N) is the external-reward advantage of each step, in steps of
  survival: what followed, minus what the critic expected from that state. The
  state is its own baseline here. Against the quiet-step mean, a button
  pressed mostly in safe moments looked good, even one that does nothing.
  """
  ev = events.astype(jnp.float32)
  return (ev * adv[..., None]).sum((0, 1)), ev.sum((0, 1))


def shown(rs, optimism, n0=20.0):
  """The taste the policy sees: w where evidence is fresh, `optimism` where not.

  Confidence fades with the memory count, so an action that is avoided long
  enough drifts back to `optimism` and gets tried again. Without this, a
  disliked action is never retried and its taste never revises.
  """
  conf = rs.n / (rs.n + n0)
  return conf * rs.w + (1.0 - conf) * optimism


def update(gene, rs, ev_sum, ev_count, horizon, n0=20.0, wmax=1.0):
  """Fold one batch of evidence into memory, then move w.

  Memory: forget old counts by `decay`, then a count-weighted mean of old and
  new evidence. Confidence n / (n + n0) keeps a few events from moving w much.
  """
  n_old = gene.decay * rs.n
  n = n_old + ev_count
  eff = jnp.where(n > 0, (n_old * rs.eff + ev_sum) / jnp.maximum(n, 1e-6),
                  rs.eff)
  q = (n / (n + n0)) * eff / horizon                      # roughly in [-1, 1]
  dw = (gene.alpha * (gene.pos * jax.nn.relu(q) - gene.neg * jax.nn.relu(-q))
        - gene.rho * (rs.w - gene.w0))
  return RewardState(jnp.clip(rs.w + dw, -wmax, wmax), n, eff)

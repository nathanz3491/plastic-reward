import jax
import numpy as np

from plastic_reward import ppo
from plastic_reward import reward as R

CFG = ppo.Config(n_envs=8, n_steps=16, n_updates=3, minibatches=2, hidden=16,
                 horizon=8)


def test_a_short_lifetime_runs_and_logs_every_batch():
  gene = R.make_gene(3, w0=0.1, alpha=0.5, decay=0.9)
  logs, life = jax.jit(ppo.train_lifetime, static_argnums=0)(
      CFG, gene, jax.random.PRNGKey(0))
  assert logs['reward'].shape == (3,) and logs['w'].shape == (3, 3)
  for v in logs.values():
    assert np.isfinite(np.asarray(v, np.float32)).all()
  assert sorted(np.asarray(life.roles).tolist()) == [0, 1, 2]


def test_two_flips_and_shuffled_evidence_run():
  import dataclasses
  cfg = dataclasses.replace(CFG, n_updates=4, flips=((0.2, 0.4), (0.6, 0.8)),
                            delays=(0, 3), shuffle_evidence=True)
  gene = R.make_gene(3, alpha=0.5)
  logs, life = jax.jit(ppo.train_lifetime, static_argnums=0)(
      cfg, gene, jax.random.PRNGKey(2))
  assert np.asarray(logs['flips_passed']).max() <= 2
  assert int(life.delay) in (0, 3)


def test_taste_over_buttons_runs():
  import dataclasses
  cfg = dataclasses.replace(CFG, n_buttons=4, events='buttons', delays=(3,))
  gene = R.make_gene(cfg.n_events, alpha=0.5)
  logs, _ = jax.jit(ppo.train_lifetime, static_argnums=0)(
      cfg, gene, jax.random.PRNGKey(3))
  assert logs['w'].shape == (3, 4) and logs['press_role'].shape == (3, 3)
  assert np.isfinite(np.asarray(logs['lifespan'])).all()


def test_lifetimes_vmap():
  gene = R.make_gene(3)
  run = jax.jit(jax.vmap(lambda k: ppo.train_lifetime(CFG, gene, k)[0]))
  logs = run(jax.random.split(jax.random.PRNGKey(1), 2))
  assert logs['reward'].shape == (2, 3)


def test_taste_as_prior_with_advantage_evidence_runs():
  import dataclasses
  cfg = dataclasses.replace(CFG, n_buttons=4, events='buttons', delays=(3,),
                            evidence='advantage', taste_as='prior',
                            optimism=0.2)
  gene = R.make_gene(cfg.n_events, alpha=0.5, lam=2.0)
  logs, _ = jax.jit(ppo.train_lifetime, static_argnums=0)(
      cfg, gene, jax.random.PRNGKey(4))
  assert logs['shown_role'].shape == (3, 3)
  for v in logs.values():
    assert np.isfinite(np.asarray(v, np.float32)).all()


def test_a_taste_prior_needs_buttons():
  import dataclasses
  import pytest
  with pytest.raises(AssertionError):
    _ = dataclasses.replace(CFG, taste_as='prior').n_events

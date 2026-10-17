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


def test_lifetimes_vmap():
  gene = R.make_gene(3)
  run = jax.jit(jax.vmap(lambda k: ppo.train_lifetime(CFG, gene, k)[0]))
  logs = run(jax.random.split(jax.random.PRNGKey(1), 2))
  assert logs['reward'].shape == (2, 3)

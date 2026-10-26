import jax
import numpy as np

from plastic_reward import evolve as EV
from plastic_reward import ppo


def test_decode_keeps_every_gene_in_its_range():
  rng = np.random.default_rng(0)
  theta = rng.normal(0, 5, (64, 7)).astype(np.float32)
  g = EV.decode(theta)
  assert g.w0.shape == (64, 3)
  assert float(abs(g.w0).max()) <= 1.0              # tanh saturates in float32
  for x in (g.alpha, g.pos, g.neg, g.lam):
    assert float(x.min()) > 0.0
  assert 0.0 <= float(g.rho.min()) and float(g.rho.max()) <= 0.2
  assert 0.0 <= float(g.decay.min()) and float(g.decay.max()) <= 1.0


def test_prior_builds_in_no_asymmetry_between_good_and_bad():
  t = EV.random_genomes(np.random.default_rng(1), 20000)
  ratio = np.log(EV.decode(t).neg / EV.decode(t).pos)
  assert abs(float(ratio.mean())) < 0.05          # neg / pos centred on 1


def test_next_generation_keeps_elites_and_the_population_size():
  rng = np.random.default_rng(2)
  pop = EV.random_genomes(rng, 10)
  fit = np.arange(10, dtype=np.float32)            # genome 9 is best
  nxt = EV.next_generation(rng, pop, fit, elites=3, immigrants=2, sigma=0.3)
  assert nxt.shape == pop.shape
  np.testing.assert_array_equal(nxt[:3], pop[[9, 8, 7]])


def test_describe_reports_the_asymmetry():
  theta = np.zeros(7, np.float32)
  theta[4] = np.log(5.0)                           # neg = 5, pos = 1
  assert abs(EV.describe(theta)['neg_over_pos'] - 5.0) < 1e-4


def test_evaluator_scores_a_small_population():
  cfg = ppo.Config(n_envs=8, n_steps=16, n_updates=3, minibatches=2,
                   hidden=16, horizon=8, delays=(0, 4), flips=((0.2, 0.4), (0.6, 0.8)))
  evaluate = EV.make_evaluator(cfg, lifetimes=2, chunk=2)
  fit = evaluate(EV.random_genomes(np.random.default_rng(3), 3),
                 jax.random.PRNGKey(0))
  assert fit.shape == (3,) and np.isfinite(fit).all() and (fit > 0).all()

import jax.numpy as jnp
import numpy as np

from plastic_reward import reward as R

H = 4


def _batch(T=12, N=2):
  return (jnp.ones((T, N)), jnp.zeros((T, N, 3)), jnp.zeros((T, N)))


def test_evidence_sign_follows_what_came_after():
  rew, ev, done = _batch()
  ev = ev.at[0, 0, 0].set(1.0)                 # type 0 at t=0, lane 0: lives on
  ev = ev.at[0, 1, 1].set(1.0)                 # type 1 at t=0, lane 1: dies next
  done = done.at[1, 1].set(1.0)
  rew = rew.at[1, 1].set(0.0)
  s, n = R.evidence(rew, ev, done, H)
  assert n.tolist() == [1.0, 1.0, 0.0]
  assert float(s[0]) > 0 > float(s[1])


def test_an_event_with_no_effect_gets_no_evidence_despite_deaths():
  # Lane 0: a poison touch, then death. Lane 1: a neutral touch, life goes on.
  # The neutral touch's outcome equals that of quiet steps, so evidence 0 --
  # against the all-steps mean the death would have made it look good.
  rew, ev, done = _batch(T=40, N=2)
  ev = ev.at[0, 0, 1].set(1.0)
  done = done.at[2, 0].set(1.0)
  rew = rew.at[2, 0].set(0.0)
  ev = ev.at[5, 1, 2].set(1.0)
  s, n = R.evidence(rew, ev, done, H)
  assert float(n[2]) == 1.0
  assert abs(float(s[2])) < 0.3 and float(s[1]) < -1.0


def test_evidence_never_reads_past_the_batch():
  rew, ev, done = _batch(T=12)
  ev = ev.at[12 - H + 1, 0, 0].set(1.0)        # its window would end past T
  _, n = R.evidence(rew, ev, done, H)
  assert float(n[0]) == 0.0


def test_good_evidence_raises_w_bad_lowers_it():
  g = R.make_gene(3, w0=0.0, alpha=1.0, decay=1.0)
  rs = R.update(g, R.init(g), jnp.array([40.0, -40.0, 0.0]),
                jnp.array([20.0, 20.0, 0.0]), H)
  assert float(rs.w[0]) > 0 > float(rs.w[1]) and float(rs.w[2]) == 0.0


def test_fixed_rule_keeps_the_birth_preference():
  g = R.make_gene(3, w0=0.2, alpha=0.0)
  rs = R.init(g)
  for _ in range(5):
    rs = R.update(g, rs, jnp.array([40.0, -40.0, 5.0]), jnp.full(3, 20.0), H)
  np.testing.assert_allclose(np.asarray(rs.w), 0.2)


def test_forgetting_lets_reversed_evidence_reverse_w():
  g = R.make_gene(3, alpha=0.5, decay=0.5)
  rs = R.init(g)
  for _ in range(20):                          # type 0 good
    rs = R.update(g, rs, jnp.array([40.0, 0, 0]), jnp.array([20.0, 0, 0]), H)
  assert float(rs.w[0]) > 0.5
  for _ in range(30):                          # the world flips: type 0 bad
    rs = R.update(g, rs, jnp.array([-40.0, 0, 0]), jnp.array([20.0, 0, 0]), H)
  assert float(rs.w[0]) < -0.5


def test_without_forgetting_old_evidence_slows_the_reversal():
  fast = R.make_gene(3, alpha=0.5, decay=0.5)
  slow = R.make_gene(3, alpha=0.5, decay=1.0)
  out = []
  for g in (fast, slow):
    rs = R.init(g)
    for _ in range(20):
      rs = R.update(g, rs, jnp.array([40.0, 0, 0]), jnp.array([20.0, 0, 0]), H)
    for _ in range(10):
      rs = R.update(g, rs, jnp.array([-40.0, 0, 0]), jnp.array([20.0, 0, 0]), H)
    out.append(float(rs.w[0]))
  assert out[0] < out[1]


def test_little_evidence_moves_w_little():
  g = R.make_gene(3, alpha=1.0)
  few = R.update(g, R.init(g), jnp.array([2.0, 0, 0]), jnp.array([1.0, 0, 0]), H)
  many = R.update(g, R.init(g), jnp.array([200.0, 0, 0]),
                  jnp.array([100.0, 0, 0]), H)
  assert 0 < float(few.w[0]) < float(many.w[0])


def test_relaxation_pulls_w_back_to_birth():
  g = R.make_gene(3, w0=0.0, alpha=0.0, rho=0.5)
  rs = R.RewardState(jnp.array([1.0, -1.0, 0.0]), jnp.zeros(3), jnp.zeros(3))
  rs = R.update(g, rs, jnp.zeros(3), jnp.zeros(3), H)
  np.testing.assert_allclose(np.asarray(rs.w), [0.5, -0.5, 0.0])


def test_intrinsic_is_clipped():
  assert float(R.intrinsic(jnp.array([5.0, 0, 0]), jnp.array([1.0, 0, 0]))) == 1.0

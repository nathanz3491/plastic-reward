import jax
import jax.numpy as jnp
import numpy as np

from plastic_reward import env as E


def _life(roles=(0, 1, 2), flip=1000):
  return E.Lifetime(jnp.array(roles, jnp.int32), jnp.int32(flip))


def _state(pos, obj_cells, hunger=10, hp=E.HP_MAX, t_ep=0, t_life=0):
  return E.State(jnp.array(pos, jnp.int32), jnp.array(obj_cells, jnp.int32),
                 jnp.int32(hunger), jnp.int32(hp), jnp.int32(t_ep),
                 jnp.int32(t_life), jnp.zeros((E.QLEN,), jnp.int32))


# Types are laid out A A A B B B C C C; put one of each type right of (4, 4).
def _objs(first):
  far = [[0, 0], [0, 8], [8, 0], [8, 8], [0, 4], [8, 4], [4, 0], [4, 8], [2, 2]]
  far[first] = [4, 5]
  return far


def test_observation_shape_and_range():
  s = E.reset(jax.random.PRNGKey(0))
  o = E.observe(s)
  assert o.shape == (E.OBS_DIM,)
  assert float(o.min()) >= 0.0 and float(o.max()) <= 1.0


def test_window_marks_outside_cells_in_a_corner():
  s = _state([0, 0], _objs(0))
  o = np.asarray(E.observe(s))[:-2].reshape(E.VIEW, E.VIEW, 4)
  assert o[0, 0, 0] == 1.0 and o[2, 2, 0] == 0.0     # outside vs the agent's cell


def test_roles_swap_food_and_poison_at_the_flip():
  life = _life((2, 0, 1), flip=50)                   # A neutral, B food, C poison
  assert E.roles_at(life, 49).tolist() == [2, 0, 1]
  assert E.roles_at(life, 50).tolist() == [2, 1, 0]


def test_food_feeds_poison_hurts_neutral_does_nothing():
  life = _life((0, 1, 2))                            # A food, B poison, C neutral
  key = jax.random.PRNGKey(1)
  for first, dh, dhp, role in ((0, E.FOOD_GAIN - 1, 0, 0),
                               (3, -1, -E.POISON_DMG, 1),
                               (6, -1, 0, 2)):
    s = _state([4, 4], _objs(first), hunger=5)       # room below the cap
    _, s2, r, done, dead, touched, role1h = E.step(key, s, 4, life)   # right
    assert int(s2.hunger) - 5 == dh and int(s2.hp) - E.HP_MAX == dhp
    assert touched.tolist() == np.eye(3)[first // 3].tolist()
    assert role1h.tolist() == np.eye(3)[role].tolist()
    assert float(r) == 1.0 and not bool(done)
    # The consumed object moved, and not under the agent.
    assert not bool(jnp.all(s2.obj_pos[first] == jnp.array([4, 5])))


def test_delayed_poison_lands_exactly_delay_steps_later():
  life = _life((0, 1, 2))                            # B is poison
  delay = 5
  s = _state([4, 4], _objs(3))
  _, s, *_ = E.step(jax.random.PRNGKey(5), s, 4, life, delay=delay)
  hps = [int(s.hp)]
  for k in range(delay):
    _, s, *_ = E.step(jax.random.PRNGKey(6 + k), s, 0, life, delay=delay)
    hps.append(int(s.hp))
  # Unhurt at eating and for delay-1 more steps; the damage lands on step delay.
  assert hps[:delay] == [E.HP_MAX] * delay
  assert hps[delay] == E.HP_MAX - E.POISON_DMG


def test_flip_changes_what_the_same_type_does():
  life = _life((0, 1, 2), flip=5)
  s = _state([4, 4], _objs(0), t_life=5)             # after the flip A poisons
  _, s2, *_ = E.step(jax.random.PRNGKey(2), s, 4, life)
  assert int(s2.hp) == E.HP_MAX - E.POISON_DMG


def test_starving_ends_the_episode_and_the_lifetime_continues():
  life = _life()
  s = _state([4, 4], _objs(8), hunger=1, t_life=77)
  obs, s2, r, done, dead, *_ = E.step(jax.random.PRNGKey(3), s, 0, life)
  assert bool(done) and bool(dead) and float(r) == 0.0
  assert int(s2.t_ep) == 0 and int(s2.hunger) == E.H_MAX     # fresh episode
  assert int(s2.t_life) == 78                                # same lifetime


def test_timeout_is_done_but_not_terminal():
  s = _state([4, 4], _objs(8), t_ep=E.MAX_LEN - 1)
  _, _, r, done, dead, *_ = E.step(jax.random.PRNGKey(4), s, 0, _life())
  assert bool(done) and not bool(dead) and float(r) == 1.0

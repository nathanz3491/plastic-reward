import jax
import jax.numpy as jnp
import numpy as np

from plastic_reward import env as E


def _life(roles=(0, 1, 2), flips=(1000,), delay=0, buttons=()):
  f = list(flips) + [E.NEVER] * (E.MAX_FLIPS - len(flips))
  return E.Lifetime(jnp.array(roles, jnp.int32), jnp.array(f, jnp.int32),
                    jnp.int32(delay), jnp.array(buttons, jnp.int32))


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
  assert o.shape == (E.DEFAULT.obs_dim,)
  assert float(o.min()) >= 0.0 and float(o.max()) <= 1.0


def test_window_marks_outside_cells_in_a_corner():
  s = _state([0, 0], _objs(0))
  o = np.asarray(E.observe(s))[:-2].reshape(E.VIEW, E.VIEW, 4)
  assert o[0, 0, 0] == 1.0 and o[2, 2, 0] == 0.0     # outside vs the agent's cell


def test_a_six_type_world_has_two_of_each_role_and_sees_every_type():
  w = E.World(n_types=6, per_type=2)
  life = E.sample_lifetime(jax.random.PRNGKey(3), ((10, 20),), (0,), w)
  assert sorted(np.asarray(life.roles).tolist()) == [0, 0, 1, 1, 2, 2]
  s = E.reset(jax.random.PRNGKey(4), world=w)
  assert s.obj_pos.shape == (12, 2)
  assert E.observe(s, w).shape == (w.obs_dim,) == (7 * 25 + 2,)
  _, _, _, _, _, touched, role, _ = E.step(jax.random.PRNGKey(5), s, 0, life, w)
  assert touched.shape == (6,) and role.shape == (E.N_ROLES,)


BW = E.World(n_buttons=4)                            # good, harm, null, null


def _press(s, button, life, k=0):
  return E.step(jax.random.PRNGKey(20 + k), s, E.N_MOVES + button, life, BW)


def test_good_button_feeds_only_next_to_an_object():
  life = _life(buttons=(0, 1, 2, 2))                 # button 0 is GOOD
  near = _state([4, 4], _objs(8), hunger=5)          # an object at (4, 5)
  _, s2, *_, info = _press(near, 0, life)
  assert bool(info['worked']) and int(s2.hunger) == 5 - 1 + E.FOOD_GAIN
  assert int(s2.pos[0]) == 4 and int(s2.pos[1]) == 4  # pressing does not move
  far = _state([2, 2], _objs(8), hunger=5)           # (2, 2) shares no edge
  _, s3, *_, info = _press(far, 0, life)
  assert not bool(info['worked']) and int(s3.hunger) == 4


def test_harm_button_hurts_after_the_delay_and_null_does_nothing():
  life = _life(buttons=(0, 1, 2, 2), delay=3)
  s = _state([2, 2], _objs(8))
  _, s, *_, info = _press(s, 1, life)
  assert info['press_role'].tolist() == [0.0, 1.0, 0.0]
  for k in range(3):
    _, s, *_ = _press(s, 2, life, k + 1)             # null presses meanwhile
  assert int(s.hp) == E.HP_MAX - E.POISON_DMG
  assert info['pressed'].tolist() == [0.0, 1.0, 0.0, 0.0]


def test_good_and_harm_buttons_swap_at_a_flip():
  life = _life(buttons=(0, 1, 2, 2), flips=(5,))
  assert E.button_roles_at(life, 4).tolist() == [0, 1, 2, 2]
  assert E.button_roles_at(life, 5).tolist() == [1, 0, 2, 2]


def test_roles_swap_food_and_poison_at_the_flip():
  life = _life((2, 0, 1), flips=(50,))               # A neutral, B food, C poison
  assert E.roles_at(life, 49).tolist() == [2, 0, 1]
  assert E.roles_at(life, 50).tolist() == [2, 1, 0]


def test_a_second_flip_swaps_back():
  life = _life((0, 1, 2), flips=(10, 20))            # A -> B -> A
  assert E.roles_at(life, 5).tolist() == [0, 1, 2]
  assert E.roles_at(life, 15).tolist() == [1, 0, 2]
  assert E.roles_at(life, 25).tolist() == [0, 1, 2]
  assert int(E.flips_passed(life, 25)) == 2


def test_sampled_lifetimes_respect_windows_and_delays():
  keys = jax.random.split(jax.random.PRNGKey(9), 200)
  lives = jax.vmap(lambda k: E.sample_lifetime(
      k, ((100, 200), (300, 400)), (0, 8, 15)))(keys)
  f = np.asarray(lives.flips)
  assert ((f[:, 0] >= 100) & (f[:, 0] < 200)).all()
  assert ((f[:, 1] >= 300) & (f[:, 1] < 400)).all()
  assert (f[:, 2] == E.NEVER).all()
  assert set(np.asarray(lives.delay).tolist()) == {0, 8, 15}


def test_food_feeds_poison_hurts_neutral_does_nothing():
  life = _life((0, 1, 2))                            # A food, B poison, C neutral
  key = jax.random.PRNGKey(1)
  for first, dh, dhp, role in ((0, E.FOOD_GAIN - 1, 0, 0),
                               (3, -1, -E.POISON_DMG, 1),
                               (6, -1, 0, 2)):
    s = _state([4, 4], _objs(first), hunger=5)       # room below the cap
    _, s2, r, done, dead, touched, role1h, _ = E.step(key, s, 4, life)  # right
    assert int(s2.hunger) - 5 == dh and int(s2.hp) - E.HP_MAX == dhp
    assert touched.tolist() == np.eye(3)[first // 3].tolist()
    assert role1h.tolist() == np.eye(3)[role].tolist()
    assert float(r) == 1.0 and not bool(done)
    # The consumed object moved, and not under the agent.
    assert not bool(jnp.all(s2.obj_pos[first] == jnp.array([4, 5])))


def test_delayed_poison_lands_exactly_delay_steps_later():
  delay = 5
  life = _life((0, 1, 2), delay=delay)               # B is poison
  s = _state([4, 4], _objs(3))
  _, s, *_ = E.step(jax.random.PRNGKey(5), s, 4, life)
  hps = [int(s.hp)]
  for k in range(delay):
    _, s, *_ = E.step(jax.random.PRNGKey(6 + k), s, 0, life)
    hps.append(int(s.hp))
  # Unhurt at eating and for delay-1 more steps; the damage lands on step delay.
  assert hps[:delay] == [E.HP_MAX] * delay
  assert hps[delay] == E.HP_MAX - E.POISON_DMG


def test_flip_changes_what_the_same_type_does():
  life = _life((0, 1, 2), flips=(5,))
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

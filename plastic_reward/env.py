"""Foraging with a flip: the stage-1 test bed for plastic intrinsic reward.

A 9x9 grid, the agent sees the 5x5 window around it plus its hunger and health.
Three object types A, B, C lie around; stepping onto one consumes it and it
reappears elsewhere. What a type does is a property of the LIFETIME, not of
the agent: one type feeds, one poisons, one does nothing, assigned at random
per lifetime -- the agent can only learn which by touching. At a hidden
lifetime step the feeding and poisoning types swap, so a judgement that was
right becomes wrong without any signal but its consequences.

The external reward is survival only: +1 for every step the agent lives.
Nothing in the observation names a role. Episodes end on death (hunger or
health at 0, terminal) or after MAX_LEN steps (truncation); the lifetime --
roles, flip time, lifetime step counter -- carries over episode resets.

Pure JAX and functional, so thousands of copies run under vmap.
"""

from typing import NamedTuple

import jax
import jax.numpy as jnp

GRID = 9
VIEW = 5
N_TYPES = 3
PER_TYPE = 3
N_OBJ = N_TYPES * PER_TYPE
FOOD, POISON, NEUTRAL = 0, 1, 2
H_MAX = 20            # hunger starts here and falls by 1 per step
FOOD_GAIN = 12
HP_MAX = 4
POISON_DMG = 2        # two poisons kill
MAX_LEN = 200
N_ACTIONS = 5         # stay, up, down, left, right
MOVES = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], jnp.int32)
OBS_DIM = 4 * VIEW * VIEW + 2
QLEN = 32             # longest supported poison delay
OBJ_TYPE = jnp.repeat(jnp.arange(N_TYPES, dtype=jnp.int32), PER_TYPE)


class Lifetime(NamedTuple):
  roles: jax.Array      # (N_TYPES,) role of each type before the flip
  flip_step: jax.Array  # () lifetime step from which FOOD and POISON swap


class State(NamedTuple):
  pos: jax.Array        # (2,) agent cell
  obj_pos: jax.Array    # (N_OBJ, 2)
  hunger: jax.Array     # ()
  hp: jax.Array         # ()
  t_ep: jax.Array       # () steps in this episode
  t_life: jax.Array     # () steps in this lifetime
  queue: jax.Array      # (QLEN,) poison damage still to land, by steps ahead


def sample_lifetime(key, flip_lo, flip_hi):
  """Random roles and a flip step drawn uniformly from [flip_lo, flip_hi)."""
  k1, k2 = jax.random.split(key)
  roles = jax.random.permutation(k1, jnp.arange(N_TYPES, dtype=jnp.int32))
  flip = jax.random.randint(k2, (), flip_lo, flip_hi)
  return Lifetime(roles, flip)


def roles_at(life, t_life):
  """Each type's role at lifetime step t: FOOD and POISON swap at the flip."""
  r = life.roles
  swapped = jnp.where(r == FOOD, POISON, jnp.where(r == POISON, FOOD, r))
  return jnp.where(t_life >= life.flip_step, swapped, r)


def _cells(idx):
  return jnp.stack([idx // GRID, idx % GRID], -1).astype(jnp.int32)


def reset(key, t_life=0):
  idx = jax.random.permutation(key, GRID * GRID)[:N_OBJ + 1]
  cells = _cells(idx)
  return State(cells[0], cells[1:], jnp.int32(H_MAX), jnp.int32(HP_MAX),
               jnp.int32(0), jnp.asarray(t_life, jnp.int32),
               jnp.zeros((QLEN,), jnp.int32))


def observe(state):
  """(OBS_DIM,): 5x5 window one-hot over {outside, A, B, C}, hunger, health."""
  board = jnp.full((GRID, GRID), -1, jnp.int32)
  board = board.at[state.obj_pos[:, 0], state.obj_pos[:, 1]].set(OBJ_TYPE)
  r = VIEW // 2
  board = jnp.pad(board, r, constant_values=-2)
  win = jax.lax.dynamic_slice(board, (state.pos[0], state.pos[1]), (VIEW, VIEW))
  chans = jnp.concatenate([
      (win == -2)[..., None],
      win[..., None] == jnp.arange(N_TYPES)], -1).astype(jnp.float32)
  return jnp.concatenate([
      chans.reshape(-1),
      jnp.array([state.hunger / H_MAX, state.hp / HP_MAX], jnp.float32)])


def step(key, state, action, life, delay=0):
  """One step; auto-resets the episode on death or timeout.

  ``delay`` (static, 0..QLEN): poison damage lands that many steps after the
  poison is eaten, the way food poisoning arrives hours after the meal. With a
  delay, a touch's consequence is no longer immediate, so a fast internal
  judgement of what is worth touching has something to add.

  Returns obs, state, reward, done, terminal, touched (N_TYPES,) one-hot of the
  type stepped on, and role (N_TYPES,) one-hot of that type's role at the time
  -- the role is for logging only and must never reach the agent.
  """
  assert 0 <= delay <= QLEN, delay
  k_spawn, k_reset = jax.random.split(key)
  pos = jnp.clip(state.pos + MOVES[action], 0, GRID - 1)
  hit = jnp.all(state.obj_pos == pos, -1)                      # (N_OBJ,)
  touched_any = hit.any()
  j = jnp.argmax(hit)
  ttype = OBJ_TYPE[j]
  role = roles_at(life, state.t_life)[ttype]
  is_food = touched_any & (role == FOOD)
  is_poison = touched_any & (role == POISON)
  hunger = jnp.minimum(state.hunger - 1 + jnp.where(is_food, FOOD_GAIN, 0), H_MAX)
  dmg = jnp.where(is_poison, POISON_DMG, 0)
  queue = jnp.concatenate([state.queue[1:], jnp.zeros((1,), jnp.int32)])
  if delay:
    landed = state.queue[0]
    queue = queue.at[delay - 1].add(dmg)
  else:
    landed = state.queue[0] + dmg
  hp = state.hp - landed

  # The consumed object reappears on a random free cell (not under the agent).
  occ = jnp.zeros((GRID, GRID), bool).at[
      state.obj_pos[:, 0], state.obj_pos[:, 1]].set(True)
  occ = occ.at[pos[0], pos[1]].set(True)
  free = (~occ).reshape(-1).astype(jnp.float32)
  new_cell = _cells(jax.random.choice(k_spawn, GRID * GRID, p=free / free.sum()))
  obj_pos = jnp.where((hit & touched_any)[:, None], new_cell, state.obj_pos)

  dead = (hunger <= 0) | (hp <= 0)
  t_ep = state.t_ep + 1
  done = dead | (t_ep >= MAX_LEN)
  reward = jnp.where(dead, 0.0, 1.0)
  nxt = State(pos, obj_pos, hunger, hp, t_ep, state.t_life + 1, queue)
  fresh = reset(k_reset, state.t_life + 1)
  nxt = jax.tree.map(lambda a, b: jnp.where(done, a, b), fresh, nxt)
  onehot = lambda i: jax.nn.one_hot(i, N_TYPES) * touched_any
  return (observe(nxt), nxt, reward, done, dead, onehot(ttype), onehot(role))

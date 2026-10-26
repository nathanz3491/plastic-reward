"""Foraging with flips: the test bed for plastic intrinsic reward.

A 9x9 grid, the agent sees the 5x5 window around it plus its hunger and health.
Three object types A, B, C lie around; stepping onto one consumes it and it
reappears elsewhere. What a type does is a property of the LIFETIME, not of
the agent: one type feeds, one poisons, one does nothing, assigned at random
per lifetime -- the agent can only learn which by touching. At hidden
lifetime steps the feeding and poisoning types swap (once, or back and forth
A -> B -> A), so a judgement that was right becomes wrong without any signal
but its consequences. How late poison damage lands is also a property of the
lifetime.

The external reward is survival only: +1 for every step the agent lives.
Nothing in the observation names a role. Episodes end on death (hunger or
health at 0, terminal) or after MAX_LEN steps (truncation); the lifetime --
roles, flip times, poison delay, lifetime step counter -- carries over
episode resets.

Pure JAX and functional, so thousands of copies run under vmap.
"""

import dataclasses
from typing import NamedTuple

import jax
import jax.numpy as jnp

GRID = 9
VIEW = 5
FOOD, POISON, NEUTRAL = 0, 1, 2
N_ROLES = 3
H_MAX = 20            # hunger starts here and falls by 1 per step
FOOD_GAIN = 12
HP_MAX = 4
POISON_DMG = 2        # two poisons kill
MAX_LEN = 200
N_MOVES = 5           # stay, up, down, left, right; buttons come after
MOVES = jnp.array([[0, 0], [-1, 0], [1, 0], [0, -1], [0, 1]], jnp.int32)
QLEN = 32             # longest supported poison delay
MAX_FLIPS = 6
NEVER = 2 ** 30       # an unused flip slot


@dataclasses.dataclass(frozen=True)
class World:
  """How many object types there are and how many of each lie around.

  Types are given roles round-robin -- 3 types: one food, one poison, one
  neutral; 6 types: two of each -- and the roles are shuffled per lifetime.
  More types make blanket caution costly: an agent wary of everything misses
  most of the food, so learning WHICH types are good has to pay.
  """
  n_types: int = 3
  per_type: int = 3
  # Buttons: extra actions with per-lifetime roles, mirroring Craftax's
  # special actions. GOOD acts like eating, but only works when an object is
  # next to the agent (otherwise it does nothing -- valid only in context);
  # HARM deals poison damage (with the lifetime's delay); the rest are NULL
  # and do nothing. GOOD and HARM swap at each flip.
  n_buttons: int = 0

  @property
  def n_obj(self):
    return self.n_types * self.per_type

  @property
  def obj_type(self):
    return jnp.repeat(jnp.arange(self.n_types, dtype=jnp.int32), self.per_type)

  @property
  def obs_dim(self):
    return (1 + self.n_types) * VIEW * VIEW + 2

  @property
  def base_roles(self):
    return jnp.arange(self.n_types, dtype=jnp.int32) % N_ROLES

  @property
  def n_actions(self):
    return N_MOVES + self.n_buttons

  @property
  def base_button_roles(self):
    roles = [FOOD, POISON] + [NEUTRAL] * (self.n_buttons - 2)
    return jnp.asarray(roles[:self.n_buttons], jnp.int32)


DEFAULT = World()


class Lifetime(NamedTuple):
  roles: jax.Array      # (n_types,) role of each type before any flip
  flips: jax.Array      # (MAX_FLIPS,) sorted lifetime steps of each swap
  delay: jax.Array      # () steps until poison damage lands (0..QLEN)
  buttons: jax.Array    # (n_buttons,) role of each button before any flip


class State(NamedTuple):
  pos: jax.Array        # (2,) agent cell
  obj_pos: jax.Array    # (n_obj, 2)
  hunger: jax.Array     # ()
  hp: jax.Array         # ()
  t_ep: jax.Array       # () steps in this episode
  t_life: jax.Array     # () steps in this lifetime
  queue: jax.Array      # (QLEN,) poison damage still to land, by steps ahead


def sample_lifetime(key, windows, delays=(0,), world=DEFAULT):
  """Random roles, one flip per window, and a poison delay.

  ``windows`` is a static tuple of (lo, hi) lifetime-step ranges, one per
  flip, in order; each flip is drawn uniformly from its range. ``delays`` is a
  static tuple the lifetime's poison delay is drawn from.
  """
  assert 1 <= len(windows) <= MAX_FLIPS, windows
  k1, k2, k3, k4 = jax.random.split(key, 4)
  roles = jax.random.permutation(k1, world.base_roles)
  lo = jnp.array([w[0] for w in windows], jnp.int32)
  hi = jnp.array([w[1] for w in windows], jnp.int32)
  flips = jax.random.randint(k2, lo.shape, lo, hi)
  flips = jnp.concatenate([flips, jnp.full((MAX_FLIPS - len(windows),), NEVER,
                                           jnp.int32)])
  delay = jnp.asarray(delays, jnp.int32)[
      jax.random.randint(k3, (), 0, len(delays))]
  assert max(delays) <= QLEN, delays
  buttons = jax.random.permutation(k4, world.base_button_roles)
  return Lifetime(roles, flips, delay, buttons)


def flips_passed(life, t_life):
  return (t_life >= life.flips).sum()


def _swap(r, flipped):
  swapped = jnp.where(r == FOOD, POISON, jnp.where(r == POISON, FOOD, r))
  return jnp.where(flipped, swapped, r)


def roles_at(life, t_life):
  """Each type's role at lifetime step t: FOOD and POISON swap at each flip."""
  return _swap(life.roles, flips_passed(life, t_life) % 2 == 1)


def button_roles_at(life, t_life):
  """Each button's role at lifetime step t: GOOD and HARM swap at each flip."""
  return _swap(life.buttons, flips_passed(life, t_life) % 2 == 1)


def _cells(idx):
  return jnp.stack([idx // GRID, idx % GRID], -1).astype(jnp.int32)


def reset(key, t_life=0, world=DEFAULT):
  idx = jax.random.permutation(key, GRID * GRID)[:world.n_obj + 1]
  cells = _cells(idx)
  return State(cells[0], cells[1:], jnp.int32(H_MAX), jnp.int32(HP_MAX),
               jnp.int32(0), jnp.asarray(t_life, jnp.int32),
               jnp.zeros((QLEN,), jnp.int32))


def observe(state, world=DEFAULT):
  """(obs_dim,): 5x5 window one-hot over {outside, types...}, hunger, health."""
  board = jnp.full((GRID, GRID), -1, jnp.int32)
  board = board.at[state.obj_pos[:, 0], state.obj_pos[:, 1]].set(world.obj_type)
  r = VIEW // 2
  board = jnp.pad(board, r, constant_values=-2)
  win = jax.lax.dynamic_slice(board, (state.pos[0], state.pos[1]), (VIEW, VIEW))
  chans = jnp.concatenate([
      (win == -2)[..., None],
      win[..., None] == jnp.arange(world.n_types)], -1).astype(jnp.float32)
  return jnp.concatenate([
      chans.reshape(-1),
      jnp.array([state.hunger / H_MAX, state.hp / HP_MAX], jnp.float32)])


def step(key, state, action, life, world=DEFAULT):
  """One step; auto-resets the episode on death or timeout.

  Poison damage lands ``life.delay`` steps after the poison is eaten (0 =
  at once), the way food poisoning arrives hours after the meal. With a
  delay, a touch's consequence is no longer immediate, so a fast internal
  judgement of what is worth touching has something to add.

  Actions 0..4 move; actions from N_MOVES on press a button (World docs).

  Returns obs, state, reward, done, terminal, touched (n_types,) one-hot of the
  type stepped on, role (N_ROLES,) one-hot of that type's role at the time,
  and info: pressed (n_buttons,) one-hot of the button pressed, press_role
  (N_ROLES,) its role, worked (did a GOOD press find an object to use). Roles
  are for logging only and must never reach the agent.
  """
  k_spawn, k_reset = jax.random.split(key)
  action = jnp.asarray(action, jnp.int32)
  is_move = action < N_MOVES
  pos = jnp.clip(state.pos + MOVES[jnp.where(is_move, action, 0)], 0, GRID - 1)
  hit = jnp.all(state.obj_pos == pos, -1)                      # (n_obj,)
  touched_any = hit.any()
  j = jnp.argmax(hit)
  ttype = world.obj_type[j]
  role = roles_at(life, state.t_life)[ttype]
  is_food = touched_any & (role == FOOD)
  is_poison = touched_any & (role == POISON)

  pressed = ~is_move
  if world.n_buttons:
    b = jnp.clip(action - N_MOVES, 0, world.n_buttons - 1)
    brole = button_roles_at(life, state.t_life)[b]
  else:
    b, brole = jnp.int32(0), jnp.int32(NEUTRAL)
  good = pressed & (brole == FOOD)
  harm = pressed & (brole == POISON)
  # GOOD needs context: an object in one of the four neighbouring cells,
  # which it consumes. Pressed anywhere else it does nothing.
  nbr = state.pos[None] + MOVES[1:]                              # (4, 2)
  adj = jnp.any(jnp.all(state.obj_pos[:, None, :] == nbr[None], -1), -1)
  worked = good & adj.any()
  ja = jnp.argmax(adj)

  gain = jnp.where(is_food | worked, FOOD_GAIN, 0)
  hunger = jnp.minimum(state.hunger - 1 + gain, H_MAX)
  dmg = jnp.where(is_poison | harm, POISON_DMG, 0)
  now = life.delay == 0
  landed = state.queue[0] + jnp.where(now, dmg, 0)
  queue = jnp.concatenate([state.queue[1:], jnp.zeros((1,), jnp.int32)])
  queue = queue.at[jnp.maximum(life.delay - 1, 0)].add(jnp.where(now, 0, dmg))
  hp = state.hp - landed

  # A consumed object reappears on a random free cell (not under the agent).
  consumed = (hit & touched_any) | ((jnp.arange(world.n_obj) == ja) & worked)
  occ = jnp.zeros((GRID, GRID), bool).at[
      state.obj_pos[:, 0], state.obj_pos[:, 1]].set(True)
  occ = occ.at[pos[0], pos[1]].set(True)
  free = (~occ).reshape(-1).astype(jnp.float32)
  new_cell = _cells(jax.random.choice(k_spawn, GRID * GRID, p=free / free.sum()))
  obj_pos = jnp.where(consumed[:, None], new_cell, state.obj_pos)

  dead = (hunger <= 0) | (hp <= 0)
  t_ep = state.t_ep + 1
  done = dead | (t_ep >= MAX_LEN)
  reward = jnp.where(dead, 0.0, 1.0)
  nxt = State(pos, obj_pos, hunger, hp, t_ep, state.t_life + 1, queue)
  fresh = reset(k_reset, state.t_life + 1, world)
  nxt = jax.tree.map(lambda a, b_: jnp.where(done, a, b_), fresh, nxt)
  info = dict(pressed=jax.nn.one_hot(b, max(world.n_buttons, 1))[:world.n_buttons]
              * pressed,
              press_role=jax.nn.one_hot(brole, N_ROLES) * pressed,
              worked=worked)
  return (observe(nxt, world), nxt, reward, done, dead,
          jax.nn.one_hot(ttype, world.n_types) * touched_any,
          jax.nn.one_hot(role, N_ROLES) * touched_any, info)

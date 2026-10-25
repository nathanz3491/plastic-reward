"""One lifetime: PPO with a GRU learns from external + plastic intrinsic reward.

Per interaction batch, in this order (docs/plan.md, section 7):
  1. w is fixed for the batch; collect fresh experience with it;
  2. update the policy on reward = external + lam * intrinsic(w);
  3. fold the batch into the bounded memory;
  4. revise w from the evidence (the Gene's rule).
PPO's epochs reuse one batch; they are not new experience and never revise w.

All parallel environments of a lifetime share its world rules (roles, flip
time): they are one individual's many simultaneous episodes, so the evidence
it gathers is about one world. Pure JAX: a whole lifetime is one jit, and
lifetimes vmap.
"""

import dataclasses
import functools

import jax
import jax.numpy as jnp
import optax

from plastic_reward import env as E
from plastic_reward import reward as R


@dataclasses.dataclass(frozen=True)
class Config:
  n_envs: int = 128
  n_steps: int = 64           # one interaction batch per lane
  n_updates: int = 300        # batches per lifetime
  epochs: int = 4
  minibatches: int = 4
  lr: float = 3e-4
  gamma: float = 0.99
  gae_lambda: float = 0.95
  clip: float = 0.2
  ent: float = 0.01
  vf: float = 0.5
  hidden: int = 128
  horizon: int = 32           # evidence window, steps
  lam: float = 1.0            # weight of the intrinsic reward
  # Learning reward is scaled for PPO only: at +1 per step, returns of 20-100
  # made the value loss swamp the policy loss under gradient clipping.
  reward_scale: float = 0.1
  flip_lo: float = 0.4        # the flip falls in [lo, hi) of the lifetime
  flip_hi: float = 0.6
  delay: int = 0              # steps until poison damage lands (env.step)

  @property
  def life_steps(self):
    return self.n_steps * self.n_updates


# --- network -------------------------------------------------------------------
def init_params(key, hidden):
  ks = jax.random.split(key, 5)
  orth = lambda k, shape, s: jax.nn.initializers.orthogonal(s)(k, shape)
  z = lambda n: jnp.zeros((n,), jnp.float32)
  return dict(
      w1=orth(ks[0], (E.OBS_DIM, hidden), jnp.sqrt(2.0)), b1=z(hidden),
      wzr=orth(ks[1], (2 * hidden, 2 * hidden), 1.0), bzr=z(2 * hidden),
      wh=orth(ks[2], (2 * hidden, hidden), 1.0), bh=z(hidden),
      wa=orth(ks[3], (hidden, E.N_ACTIONS), 0.01), ba=z(E.N_ACTIONS),
      wc=orth(ks[4], (hidden, 1), 1.0), bc=z(1))


def net(p, h, obs, done_prev):
  """GRU step; the hidden state is zeroed where a new episode began."""
  h = h * (1.0 - done_prev)[..., None]
  x = jax.nn.relu(obs @ p['w1'] + p['b1'])
  zr = jax.nn.sigmoid(jnp.concatenate([x, h], -1) @ p['wzr'] + p['bzr'])
  zg, rg = jnp.split(zr, 2, -1)
  cand = jnp.tanh(jnp.concatenate([x, rg * h], -1) @ p['wh'] + p['bh'])
  h = (1.0 - zg) * h + zg * cand
  return h, h @ p['wa'] + p['ba'], (h @ p['wc'] + p['bc'])[..., 0]


def unroll(p, h0, obs, done_prev):
  """(T, N, ...) sequences from hidden h0 -> logits (T, N, A), values (T, N)."""
  def body(h, xs):
    h, logits, value = net(p, h, *xs)
    return h, (logits, value)
  _, (logits, values) = jax.lax.scan(body, h0, (obs, done_prev))
  return logits, values


# --- one lifetime --------------------------------------------------------------
def gae(cfg, rew, val, done, last_val):
  def body(carry, xs):
    adv_next, v_next = carry
    r, v, d = xs
    nd = 1.0 - d
    delta = r + cfg.gamma * v_next * nd - v
    adv = delta + cfg.gamma * cfg.gae_lambda * nd * adv_next
    return (adv, v), adv
  _, adv = jax.lax.scan(body, (jnp.zeros_like(last_val), last_val),
                        (rew, val, done), reverse=True)
  return adv, adv + val


def train_lifetime(cfg, gene, key):
  """Returns per-batch logs (n_updates, ...) and the lifetime's world rules."""
  k_life, k_net, k_env, k_run = jax.random.split(key, 4)
  life = E.sample_lifetime(k_life, int(cfg.flip_lo * cfg.life_steps),
                           int(cfg.flip_hi * cfg.life_steps))
  params = init_params(k_net, cfg.hidden)
  opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg.lr, eps=1e-5))
  states = jax.vmap(E.reset)(jax.random.split(k_env, cfg.n_envs))
  carry = dict(
      params=params, opt=opt.init(params), states=states,
      obs=jax.vmap(E.observe)(states),
      h=jnp.zeros((cfg.n_envs, cfg.hidden), jnp.float32),
      done=jnp.zeros((cfg.n_envs,), jnp.float32),
      rs=R.init(gene), key=k_run)
  env_step = jax.vmap(functools.partial(E.step, delay=cfg.delay),
                      in_axes=(0, 0, 0, None))

  def rollout(c):
    def body(c, _):
      key, k_act, k_env = jax.random.split(c['key'], 3)
      h, logits, value = net(c['params'], c['h'], c['obs'], c['done'])
      action = jax.random.categorical(k_act, logits)
      logp = jax.nn.log_softmax(logits)[jnp.arange(cfg.n_envs), action]
      obs, states, rew, done, dead, touched, role = env_step(
          jax.random.split(k_env, cfg.n_envs), c['states'], action, life)
      out = dict(obs=c['obs'], done_prev=c['done'], action=action, logp=logp,
                 value=value, reward=rew, done=done.astype(jnp.float32),
                 dead=dead, touched=touched, role=role)
      c = {**c, 'key': key, 'h': h, 'obs': obs, 'states': states,
           'done': done.astype(jnp.float32)}
      return c, out
    return jax.lax.scan(body, c, None, cfg.n_steps)

  def ppo(params, opt_state, h0, batch, adv, ret, key):
    def loss_fn(p, mb):
      logits, values = unroll(p, mb['h0'], mb['obs'], mb['done_prev'])
      logp_all = jax.nn.log_softmax(logits)
      logp = jnp.take_along_axis(logp_all, mb['action'][..., None], -1)[..., 0]
      ratio = jnp.exp(logp - mb['logp'])
      a = (mb['adv'] - mb['adv'].mean()) / (mb['adv'].std() + 1e-8)
      pg = -jnp.minimum(ratio * a,
                        jnp.clip(ratio, 1 - cfg.clip, 1 + cfg.clip) * a).mean()
      vl = 0.5 * ((values - mb['ret']) ** 2).mean()
      ent = -(jnp.exp(logp_all) * logp_all).sum(-1).mean()
      return pg + cfg.vf * vl - cfg.ent * ent

    data = dict(obs=batch['obs'], done_prev=batch['done_prev'],
                action=batch['action'], logp=batch['logp'], adv=adv, ret=ret)
    per = cfg.n_envs // cfg.minibatches

    def epoch(c, k):
      params, opt_state = c
      perm = jax.random.permutation(k, cfg.n_envs)
      def mb_step(c, idx):
        params, opt_state = c
        mb = {k_: v[:, idx] for k_, v in data.items()}
        mb['h0'] = h0[idx]
        grads = jax.grad(loss_fn)(params, mb)
        upd, opt_state = opt.update(grads, opt_state, params)
        return (optax.apply_updates(params, upd), opt_state), None
      (params, opt_state), _ = jax.lax.scan(
          mb_step, (params, opt_state), perm.reshape(cfg.minibatches, per))
      return (params, opt_state), None

    (params, opt_state), _ = jax.lax.scan(
        epoch, (params, opt_state), jax.random.split(key, cfg.epochs))
    return params, opt_state

  def one_batch(c, _):
    h0 = c['h']
    c, b = rollout(c)
    _, _, last_val = net(c['params'], c['h'], c['obs'], c['done'])
    r_int = R.intrinsic(c['rs'].w, b['touched'])
    rew = cfg.reward_scale * (b['reward'] + cfg.lam * r_int)
    adv, ret = gae(cfg, rew, b['value'], b['done'], last_val)
    key, k_ppo = jax.random.split(c['key'])
    params, opt_state = ppo(c['params'], c['opt'], h0, b, adv, ret, k_ppo)
    ev_sum, ev_n = R.evidence(b['reward'], b['touched'], b['done'], cfg.horizon)
    rs = R.update(gene, c['rs'], ev_sum, ev_n, cfg.horizon)
    t_life = c['states'].t_life[0]
    roles = E.roles_at(life, t_life)
    log = dict(
        reward=b['reward'].mean(),                  # external, per step
        deaths=b['dead'].sum() / cfg.n_envs,        # per lane per batch
        # steps per ended episode: the lifespan, the readable form of survival
        lifespan=cfg.n_steps * cfg.n_envs / jnp.maximum(b['done'].sum(), 1.0),
        touch_role=b['role'].mean((0, 1)),          # food, poison, neutral
        w=rs.w, eff=rs.eff,
        w_role=jnp.zeros_like(rs.w).at[roles].set(rs.w),
        eff_role=jnp.zeros_like(rs.eff).at[roles].set(rs.eff),
        flipped=t_life >= life.flip_step)
    c = {**c, 'params': params, 'opt': opt_state, 'rs': rs, 'key': key}
    return c, log

  _, logs = jax.lax.scan(one_batch, carry, None, cfg.n_updates)
  return logs, life

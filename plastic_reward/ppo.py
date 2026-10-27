"""One lifetime: PPO with a GRU learns from external + plastic intrinsic reward.

Per interaction batch, in this order (docs/plan.md, section 7):
  1. w is fixed for the batch; collect fresh experience with it;
  2. update the policy on reward = external + lam * intrinsic(w) (or, with
     Config.taste_as='prior', on external reward with w as a button logit bias);
  3. fold the batch into the bounded memory;
  4. revise w from the evidence (the Gene's rule).
PPO's epochs reuse one batch; they are not new experience and never revise w.

All parallel environments of a lifetime share its world rules (roles, flip
time): they are one individual's many simultaneous episodes, so the evidence
it gathers is about one world. Pure JAX: a whole lifetime is one jit, and
lifetimes vmap.
"""

import dataclasses

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
  # One (lo, hi) window per flip, as fractions of the lifetime; two windows
  # make an A -> B -> A lifetime.
  flips: tuple = ((0.4, 0.6),)
  delays: tuple = (0,)        # each lifetime's poison delay is drawn from these
  # Control: give each event type the evidence of a randomly chosen type, so
  # taste still moves as much but no longer about the right object.
  shuffle_evidence: bool = False
  n_types: int = 3            # object types (= taste event types)
  per_type: int = 3           # objects of each type in the grid
  n_buttons: int = 0          # extra actions with lifetime roles (env.World)
  # What taste is about: 'objects' (touching each object type) or 'buttons'
  # (pressing each button) -- the latter is the action-level taste a
  # knowledge-free action mask would need.
  events: str = 'objects'
  # How taste is judged: 'window' (return over the next `horizon` steps
  # against quiet steps) or 'advantage' (external-reward GAE advantage: what
  # followed minus what the critic expected from that state).
  evidence: str = 'window'
  # How taste acts: 'reward' (intrinsic reward when the event happens) or
  # 'prior' (buttons only: a bias added to the button logits, which cannot be
  # farmed the way a reward on a free action can).
  taste_as: str = 'reward'
  # With taste_as='prior', what the policy sees where evidence has faded
  # (reward.shown): 0 leans neither way, > 0 retries avoided actions.
  optimism: float = 0.0

  @property
  def life_steps(self):
    return self.n_steps * self.n_updates

  @property
  def world(self):
    return E.World(self.n_types, self.per_type, self.n_buttons)

  @property
  def n_events(self):
    assert self.events in ('objects', 'buttons'), self.events
    assert self.evidence in ('window', 'advantage'), self.evidence
    assert self.taste_as in ('reward', 'prior'), self.taste_as
    assert self.taste_as == 'reward' or self.events == 'buttons', \
        'a taste prior is over actions, so it needs events=buttons'
    return self.n_types if self.events == 'objects' else self.n_buttons


# --- network -------------------------------------------------------------------
def init_params(key, hidden, obs_dim, n_actions):
  ks = jax.random.split(key, 5)
  orth = lambda k, shape, s: jax.nn.initializers.orthogonal(s)(k, shape)
  z = lambda n: jnp.zeros((n,), jnp.float32)
  return dict(
      w1=orth(ks[0], (obs_dim, hidden), jnp.sqrt(2.0)), b1=z(hidden),
      wzr=orth(ks[1], (2 * hidden, 2 * hidden), 1.0), bzr=z(2 * hidden),
      wh=orth(ks[2], (2 * hidden, hidden), 1.0), bh=z(hidden),
      wa=orth(ks[3], (hidden, n_actions), 0.01), ba=z(n_actions),
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
  life = E.sample_lifetime(
      k_life, tuple((int(lo * cfg.life_steps), int(hi * cfg.life_steps))
                    for lo, hi in cfg.flips), cfg.delays, cfg.world)
  params = init_params(k_net, cfg.hidden, cfg.world.obs_dim, cfg.world.n_actions)
  opt = optax.chain(optax.clip_by_global_norm(0.5), optax.adam(cfg.lr, eps=1e-5))
  states = jax.vmap(lambda k: E.reset(k, world=cfg.world))(
      jax.random.split(k_env, cfg.n_envs))
  carry = dict(
      params=params, opt=opt.init(params), states=states,
      obs=jax.vmap(lambda s: E.observe(s, cfg.world))(states),
      h=jnp.zeros((cfg.n_envs, cfg.hidden), jnp.float32),
      done=jnp.zeros((cfg.n_envs,), jnp.float32),
      rs=R.init(gene), key=k_run)
  env_step = jax.vmap(lambda k, s, a, l: E.step(k, s, a, l, cfg.world),
                      in_axes=(0, 0, 0, None))

  def action_bias(rs):
    """(n_actions,) taste as a logit prior on the buttons; zero otherwise."""
    if cfg.taste_as == 'reward':
      return jnp.zeros((cfg.world.n_actions,), jnp.float32)
    lean = cfg.lam * gene.lam * R.shown(rs, cfg.optimism)
    return jnp.concatenate([jnp.zeros((E.N_MOVES,), jnp.float32), lean])

  def rollout(c, bias):
    def body(c, _):
      key, k_act, k_env = jax.random.split(c['key'], 3)
      h, logits, value = net(c['params'], c['h'], c['obs'], c['done'])
      logits = logits + bias
      action = jax.random.categorical(k_act, logits)
      logp = jax.nn.log_softmax(logits)[jnp.arange(cfg.n_envs), action]
      obs, states, rew, done, dead, touched, role, info = env_step(
          jax.random.split(k_env, cfg.n_envs), c['states'], action, life)
      out = dict(obs=c['obs'], done_prev=c['done'], action=action, logp=logp,
                 value=value, reward=rew, done=done.astype(jnp.float32),
                 dead=dead, touched=touched, role=role,
                 events=touched if cfg.events == 'objects' else info['pressed'],
                 press_role=info['press_role'], worked=info['worked'])
      c = {**c, 'key': key, 'h': h, 'obs': obs, 'states': states,
           'done': done.astype(jnp.float32)}
      return c, out
    return jax.lax.scan(body, c, None, cfg.n_steps)

  def ppo(params, opt_state, h0, batch, adv, ret, key, bias):
    def loss_fn(p, mb):
      logits, values = unroll(p, mb['h0'], mb['obs'], mb['done_prev'])
      logits = logits + bias       # the policy acted with this prior
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
    bias = action_bias(c['rs'])
    c, b = rollout(c, bias)
    _, _, last_val = net(c['params'], c['h'], c['obs'], c['done'])
    rew = cfg.reward_scale * b['reward']
    if cfg.taste_as == 'reward':
      r_int = R.intrinsic(c['rs'].w, b['events'])
      rew = rew + cfg.reward_scale * cfg.lam * gene.lam * r_int
    adv, ret = gae(cfg, rew, b['value'], b['done'], last_val)
    key, k_ppo, k_shuf = jax.random.split(c['key'], 3)
    params, opt_state = ppo(c['params'], c['opt'], h0, b, adv, ret, k_ppo, bias)
    if cfg.evidence == 'window':
      ev_sum, ev_n = R.evidence(b['reward'], b['events'], b['done'], cfg.horizon)
    else:
      # External-only advantage, in steps of survival. With taste as a prior
      # this is `adv` itself; with taste as reward the critic also predicts
      # taste, so the baseline is only approximate.
      adv_ext = adv if cfg.taste_as == 'prior' else gae(
          cfg, cfg.reward_scale * b['reward'], b['value'], b['done'], last_val)[0]
      ev_sum, ev_n = R.advantage_evidence(adv_ext / cfg.reward_scale, b['events'])
    if cfg.shuffle_evidence:
      src = jax.random.randint(k_shuf, ev_sum.shape, 0, ev_sum.shape[0])
      ev_sum, ev_n = ev_sum[src], ev_n[src]
    rs = R.update(gene, c['rs'], ev_sum, ev_n, cfg.horizon)
    t_life = c['states'].t_life[0]
    roles_now = (E.roles_at if cfg.events == 'objects' else E.button_roles_at)
    by_role = jax.nn.one_hot(roles_now(life, t_life), E.N_ROLES)
    log = dict(
        reward=b['reward'].mean(),                  # external, per step
        deaths=b['dead'].sum() / cfg.n_envs,        # per lane per batch
        # steps per ended episode: the lifespan, the readable form of survival
        lifespan=cfg.n_steps * cfg.n_envs / jnp.maximum(b['done'].sum(), 1.0),
        touch_role=b['role'].mean((0, 1)),          # food, poison, neutral
        press_role=b['press_role'].mean((0, 1)),    # good, harm, null buttons
        worked=b['worked'].mean(),                  # good presses that fed
        w=rs.w, eff=rs.eff,
        # mean taste and evidence over the types holding each role right now
        w_role=by_role.T @ rs.w / jnp.maximum(by_role.sum(0), 1.0),
        eff_role=by_role.T @ rs.eff / jnp.maximum(by_role.sum(0), 1.0),
        shown_role=by_role.T @ R.shown(rs, cfg.optimism)
        / jnp.maximum(by_role.sum(0), 1.0),
        flips_passed=E.flips_passed(life, t_life),
        flipped=t_life >= life.flips[0],            # the first flip is behind
        delay=life.delay)
    c = {**c, 'params': params, 'opt': opt_state, 'rs': rs, 'key': key}
    return c, log

  _, logs = jax.lax.scan(one_batch, carry, None, cfg.n_updates)
  return logs, life

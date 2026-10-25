# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
- **"Once bitten" taste rule** (`averse1` / `averse3` arms): bad evidence
  counts 5× and memory fades fast (decay 0.5).
  - With poison 15 steps late, lifespan at end of life is 87 against 63 for
    external only (+23.6 ± 2.7, better in 15 of 16 lifetimes).
  - The symmetric rule was worse than no taste (−6 to −23).

  Details are in `docs/results-stage1.md`.
- **Delayed poison** (`Config.delay` / `env.step(delay=...)`): damage lands N
  steps after the poison is eaten. A fast internal judgement only has
  something to add when the real consequence is late, as with food poisoning.
- Runner arms at intrinsic weight 1, 3 and 10, plus a fixed-taste control at
  3. Each arm reports lifespan before the flip, after it and at the end of
  life, as mean ± SE over lifetimes, with a paired difference against
  external-only. Plots are aligned on each lifetime's flip.

### Changed
- **Evidence is now measured against steps where nothing was touched.**
  Against the all-steps mean, poison deaths pulled the baseline down and the
  do-nothing object drifted to a positive taste (+0.6).
- Runner defaults: 16 lifetimes of 300 batches, with the flip at 30-40% of
  life so recovery has room to show.

- Stage 1 prototype (no evolution yet), pure JAX:
  - **Environment** (`plastic_reward/env.py`): foraging with a flip. Three
    object types, whose roles (feed, poison, nothing) are re-randomised every
    lifetime; feed and poison swap at a hidden lifetime step. The external
    reward is survival.
  - **Plastic intrinsic reward** (`plastic_reward/reward.py`): a bounded
    memory holding per-event evidence (return over a window after the event,
    against the batch baseline) and the gene-controlled rule that revises the
    intrinsic weights w.
  - **Agent** (`plastic_reward/ppo.py`): PPO with a GRU, where one jit runs a
    whole lifetime and lifetimes vmap. The learning reward is scaled by 0.1,
    since unscaled returns let the value loss swamp the policy loss.
  - **Runner** (`run_stage1.py`): compares the external-only, fixed-intrinsic
    and plastic arms over the same lifetimes, writing logs, a summary and a
    plot.
  - Tests for the environment, the reward rule, and a PPO smoke run.
- `docs/plan.md`: the working plan, summarising the source proposal.

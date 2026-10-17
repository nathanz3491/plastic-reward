# Changelog

All notable changes are recorded here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/).

## [Unreleased]

### Added
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

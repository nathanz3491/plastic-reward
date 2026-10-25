# HANDOFF (plastic-reward)

## 2026-10-18 run

**Yesterday:** project did not exist; created 2026-10-17 22:19-22:36 with no commits.

**Today:** first commit of the stage-1 prototype (env, reward rule, PPO+GRU, runner, tests, plan, README, CHANGELOG). Stage-1 result (6 lifetimes x 200 updates x 64 envs, CPU, ~20 min/arm): post-flip lifespan ~19 for external, fixed and plastic alike; recovery 102.5 batches for all; final lifespan plastic 43.0 vs 37.0 external/fixed. Arms not separable at this scale. `runs/` is git-ignored (outputs stay local).

**Tomorrow:** longer or GPU run with more lifetimes to separate arms; then stage 2 (evolve the genes: w0, alpha, rho, decay, ...) per `docs/plan.md`.

## 2026-10-26 run

**Yesterday:** HEAD `fea80de` (2026-10-25 13:45, on `feat/delayed-poison-once-bitten`): delayed poison, quiet-step evidence baseline, "once bitten" taste rule (87 vs 63 end-of-life lifespan, 15/16 paired lifetimes), `docs/results-stage1.md`. HANDOFF last written 2026-10-18.

**Today (work found uncommitted from 2026-10-25 evening, committed now):**
- **Stage 2 evolution** (`plastic_reward/evolve.py`, `run_stage2.py`, `tests/test_evolve.py`): 7-number genome decoded to a Gene; population 16 x 25 generations x 3 lifetimes; fitness = mean lifespan; random-search control on the same budget; held-out test on 16 fresh lifetimes at delays 15/25/8/0. `Gene.lam` added so evolution can tune taste loudness (`reward.py`, `ppo.py`).
- **Result** (`docs/results-stage2.md`, seed 0, 75 min on RTX 4090): evolution beats random search; it rediscovers the lopsided rule unprompted (bad/good weight 1.4 -> ~5-8, memory kept 0.88 -> 0.26). Champion +17.1 lifespan vs external-only at delay 15 and +20.2 at unseen delay 25 (16/16 lifetimes both); +8.4 at delay 8; **-2.9 at delay 0** (instant poison).
- README gets a Stage 2 section; CHANGELOG updated. 25 tests pass.

**Tomorrow:**
- Independent evolution seeds, to show the result is reliable (currently one run).
- Stage 3: evolve across a mix of delays; A -> B -> A flips; time-schedule-only taste control; hand frozen rule to new brains.

# HANDOFF (plastic-reward)

## 2026-10-18 run

**Yesterday:** project did not exist; created 2026-10-17 22:19-22:36 with no commits.

**Today:** first commit of the stage-1 prototype (env, reward rule, PPO+GRU, runner, tests, plan, README, CHANGELOG). Stage-1 result (6 lifetimes x 200 updates x 64 envs, CPU, ~20 min/arm): post-flip lifespan ~19 for external, fixed and plastic alike; recovery 102.5 batches for all; final lifespan plastic 43.0 vs 37.0 external/fixed. Arms not separable at this scale. `runs/` is git-ignored (outputs stay local).

**Tomorrow:** longer or GPU run with more lifetimes to separate arms; then stage 2 (evolve the genes: w0, alpha, rho, decay, ...) per `docs/plan.md`.

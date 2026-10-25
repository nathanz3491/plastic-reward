# plastic-reward

Can an agent learn to revise **what it finds worth pursuing**, and can
evolution find a good rule for doing so?

Animals are not born with a fixed reward. Sugar tastes good when you are
hungry; one bad meal makes a food unpleasant for a long time. Evolution fixes
the rule by which preferences form and change, and the lifetime fills them
in. This project builds that split into a reinforcement-learning agent:

- the **policy** learns by RL (PPO with a GRU);
- an **intrinsic reward** w says what is worth touching, and is revised after
  every batch from evidence held in a small bounded memory;
- the **genes** are the rule that revises w (birth preference, update speed,
  relaxation, sensitivity to good and bad evidence, forgetting). They are
  hand-set in stage 1 and evolved in stage 2, with fitness judged on external
  performance only.

Plan and stages: [`docs/plan.md`](docs/plan.md).

## Stage 1: foraging with a flip (current)

The world is a 9×9 grid with three object types:
- one feeds, one poisons, one does nothing, assigned at random per lifetime;
- the agent only learns which by touching;
- at a hidden time, feed and poison **swap**;
- the external reward is survival: +1 per step alive.

The question: does the plastic intrinsic reward track the evidence, reverse
after the flip, and so speed the policy's recovery? It is compared against
external reward only, and against a fixed intrinsic reward.

## Run

```bash
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements.txt   # Linux/macOS: .venv/bin/python
.venv/Scripts/python run_stage1.py --delay 15 --out runs/delay15
```

This runs 16 lifetimes per arm: external-only, fixed taste, and plastic taste
at strength 1, 3 and 10 (`--arms` picks a subset). Options:
- `--delay N` makes poison damage land N steps after eating.
- `--flip-lo` / `--flip-hi` set when the flip falls, as fractions of life.

It writes `summary.json` (mean ± SE per arm, with a paired difference against
external-only), `logs.npz` and `stage1.png`. It runs on CPU, but a GPU is much
faster.

## Stage 2: evolving the rule

```bash
.venv/Scripts/python run_stage2.py --out runs/stage2
```

The run has four steps:
1. **Evolution.** A population of 16 random taste rules, each scored on
   survival over 3 lifetimes in the delayed-poison world, for 25 generations.
2. **Random search** over the same rule family, on the same budget.
3. **Champions.** The finalists of each search are re-measured on fresh
   lifetimes.
4. **Held-out test.** The champions go up against external-only and the
   hand-set rule on 16 fresh lifetimes, at the trained delay and at unseen
   ones.

It takes about 75 minutes on an RTX 4090 and writes `history.json`,
`champions.json`, `test.json` and `stage2.png`.

## Test

```bash
.venv/Scripts/python -m pytest -q
```

The tests cover the environment's rules and the flip, the reward rule (the
sign of the evidence, forgetting, confidence, relaxation), and a PPO smoke
run that also checks that lifetimes vmap.

## Layout

| path | what |
|---|---|
| `plastic_reward/env.py` | the foraging-with-flip environment (pure JAX) |
| `plastic_reward/reward.py` | bounded memory, evidence, the gene-controlled update rule |
| `plastic_reward/ppo.py` | PPO + GRU; one jit per lifetime |
| `run_stage1.py` | the three-arm stage-1 comparison |
| `requirements.txt` / `requirements.lock` | pinned and fully frozen dependencies |

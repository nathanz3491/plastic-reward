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
.venv/Scripts/python run_stage1.py --lifetimes 6 --updates 200 --envs 64 --out runs/stage1
```

This writes `runs/stage1/summary.json`, `logs.npz` and `stage1.png`. On CPU
it takes about 40 minutes; the run uses JAX and goes faster on a GPU.

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

# Plan: evolving a plastic intrinsic reward with bounded memory

Source: the Chinese proposal 进化算法.md (研究建议：进化具有长期记忆的可塑内部奖励机制),
shared 2026-10-11. This is the working summary the code follows.

## Question

Can a population evolve a rule for revising intrinsic reward, such that an
agent with a small bounded memory keeps correcting what it finds worth
pursuing, and so adapts faster when the world changes?

The contribution is **not** "evolving a reward function": genetic programming
of intrinsic rewards, learning rewards across lifetimes (Zheng et al., ICML
2020), meta-learned curiosity (Alet et al., ICLR 2020) and continuously
evolving rewards (JMLR 2025) all exist. The claim to test is narrower:
bounded memory, reward plasticity driven by experience, and continued
adaptation to changes never seen during evolution.

## The agent

| part | role | changes |
|---|---|---|
| policy θ | chooses actions | every batch, by RL (PPO + GRU) |
| preference w | intrinsic reward per event type | after every batch, from evidence |
| memory M | bounded evidence per event type | during the lifetime |
| genes g | the rule that forms and revises w | between generations only |

- Learning reward = external + λ · clip(w · x). Only the policy learns from
  it.
- **Fitness is external performance only.** An agent judged on its own
  intrinsic reward could simply inflate it.
- Offspring inherit g alone: the policy, memory and w restart at birth.
- Genes (stage 1, hand-set): w0, α (evidence speed), ρ (relaxation to w0),
  sensitivity to positive and negative evidence, memory decay.

## Three timescales

1. **Batch:** collect new experience with w fixed, update the policy, update
   memory, revise w. PPO's epochs over one batch are not new experience and
   never revise w.
2. **Lifetime:** many batches; the world may change partway, with no reset.
3. **Generation:** select on external performance, copy, mutate. Each gene
   is tested over several lifetimes.

## Stages

1. **Inspectable prototype, no evolution** (this repo now). A toy foraging
   world with a hidden flip. Check that w follows the evidence and reverses
   after the flip, and that this speeds the policy's recovery against
   external-only and fixed-reward controls.
2. **Evolution** of the ~6 gene values: population 8-16, 2-3 lifetimes per
   gene, elites plus fresh random genes, no crossover. Controls: fixed
   reward, a reward that changes on a time schedule, random search over the
   same rule family.
3. **Continual adaptation.** Unknown mechanisms, A→B→A changes at random
   times, unseen combinations of changes, frozen genes at test time,
   no-memory and frozen-update ablations, transfer of the evolved rule to a
   fresh policy. Then Craftax, where potions re-randomise every episode.
4. **Extensions, only if 1-3 work:** evolving the memory rules, new reward
   terms, inheriting experience, multi-agent worlds.

## Metrics

- External performance over the whole lifetime, and its area under the
  curve.
- Speed of recovery after a change.
- Performance on returning to the old environment.
- Adaptation to unseen mechanisms.
- Memory and compute cost.

Total search cost (population × generations × lifetimes × steps) is reported
alongside the cost of deploying one agent. An evolved champion is never
compared against a baseline trained once.

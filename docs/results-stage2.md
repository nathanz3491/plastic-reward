# Stage 2 results (2026-10-25)

Setup:
- Population 16, 25 generations, 3 lifetimes per genome per generation.
- Trained in the delayed-poison world (delay 15); fitness is mean lifespan
  over a lifetime.
- The random-search control had the same budget (400 evaluations).
- The finalists of each search were re-measured on 8 fresh lifetimes.
- The held-out test used 16 fresh lifetimes, the same for every rule
  (paired).

The run takes 75 minutes on an RTX 4090 (`run_stage2.py`).

## Evolution beats random search

| | best rule per generation | population mean |
|---|---|---|
| evolution | 51 → 62-69 | 44 → 55-58 |
| random search | best ever 58.5 | about 45 throughout |

## Evolution rediscovered the lopsided rule, unprompted

The starting population had no built-in preference: weights for good and bad
evidence were drawn from the same distribution. Population medians,
generation 0 → 24:

- **bad / good evidence weight:** 1.4 → about 5-8. Bad news came to count
  several times more than good.
- **memory kept per batch:** 0.88 → 0.26 (fast forgetting).
- **taste loudness:** 2.3 → about 5.

That is the "once bitten" rule found by hand in stage 1, now found by
selection on survival alone.

The champion:
- **Born disliking every object** (birth taste −0.94): wary of the unknown.
  This resembles the food neophobia of rats, which approach new foods
  cautiously.
- **Bad news counts 2.7×,** and it learns fast (α 2.2).
- **Keeps only 14% of old evidence per batch** (decay 0.135).
- **Loud taste** (λ 5.3).

## Held-out lifetimes

Mean lifespan over a whole lifetime; paired difference against external-only,
with the number of lifetimes won out of 16:

| world | external | hand-set (stage 1) | random-search champion | **evolved champion** |
|---|---|---|---|---|
| delay 15 (trained) | 45.6 | 52.8 (+7.2 ± 0.8, 16/16) | 55.6 (+10.0 ± 0.8, 16/16) | **62.7 (+17.1 ± 0.7, 16/16)** |
| delay 25 (unseen) | 45.7 | 51.6 (+5.9 ± 0.6, 16/16) | 58.0 (+12.3 ± 0.6, 16/16) | **66.0 (+20.2 ± 1.1, 16/16)** |
| delay 8 (unseen) | 50.2 | 56.2 (+6.1 ± 1.0, 14/16) | 56.8 (+6.6 ± 1.2, 14/16) | **58.6 (+8.4 ± 1.2, 15/16)** |
| delay 0 (unseen) | 58.3 | 62.2 (+3.9 ± 2.0, 12/16) | 62.2 (+3.9 ± 2.1, 10/16) | 55.4 (−2.9 ± 1.6, 6/16) |

At the end of life with delay 15, the evolved champion lives 120 steps,
against 67 for external-only and 90 for the hand-set rule.

## What this says

- **Selection on survival alone finds a much better taste rule than hand
  design, and than random search.** It also rediscovers the biological
  asymmetry and fast forgetting.
- **It transfers to longer and shorter delays it never saw** (8, 25).
- **It fails when consequences are instant** (delay 0). There its wariness
  of everything (birth taste −0.94) and loud taste cost a little, where taste
  has nothing to add. The rule specialised to a world where consequences come
  late, the way an animal's instincts fit its own niche.

## Caveats and next steps

- One evolutionary run (seed 0). Independent runs are needed to show the
  outcome is reliable.
- Stage 3 tests:
  - evolving across a mix of delays, to see whether a rule that is good
    everywhere emerges;
  - flipping back and forth (A → B → A);
  - a taste that changes on a time schedule only (the "is it really using
    evidence" control);
  - handing the frozen rule to new brains.

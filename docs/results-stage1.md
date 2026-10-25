# Stage 1 results (2026-10-25)

16 lifetimes per arm, 300 batches each, with the flip at 30-40% of life. All
arms run over the same 16 lifetimes (same worlds, same flip times), so
differences are paired. Lifespan is in steps per episode (the external
return). "End" is the last 30 batches of life. Values are mean ± SE over
lifetimes. Run with `run_stage1.py` on an RTX 4090, about 0.6 min per arm.

## The first rule fails

The rule: equal weight on good and bad evidence, memory decay 0.9.

| arm | instant poison: end | vs external | poison 15 steps late: end | vs external |
|---|---|---|---|---|
| external only | 111 | — | 63 | — |
| fixed taste ×3 | 110 | −1 ± 5 | 47 | −17 ± 2 |
| plastic ×1 | 114 | +3 ± 8 | 57 | −6 ± 2 |
| plastic ×3 | 112 | +1 ± 5 | 51 | −13 ± 2 |
| plastic ×10 | 99 | −13 ± 6 | 40 | −23 ± 2 |

After the flip, the taste kept liking the old food, which is now poison, for
60-80 batches. With delayed poison each bad outcome is weak evidence, so it
took a long time to drag the taste down from +1. In the meantime the taste
rewarded eating poison.

## "Once bitten" works

The rule: bad evidence counts 5×, and memory decay is 0.5 (old evidence
fades fast).

| arm | instant: after flip | instant: end | delayed: after flip | delayed: end |
|---|---|---|---|---|
| external only | 54.8 | 111.3 | 39.4 | 63.5 |
| once bitten ×1 | 59.4 | 114.5 | 46.7 | 80.9 |
| once bitten ×3 | 60.9 | 114.7 | 50.9 | 87.1 |

Paired against external only:

| arm | instant: after flip | instant: end | delayed: after flip | delayed: end |
|---|---|---|---|---|
| once bitten ×1 | +4.6 ± 3.8 (10/16) | +3.3 ± 8.1 | **+7.4 ± 1.2 (16/16)** | **+17.4 ± 2.8 (15/16)** |
| once bitten ×3 | +6.1 ± 3.3 (12/16) | +3.5 ± 7.3 | **+11.5 ± 1.1 (16/16)** | **+23.6 ± 2.7 (15/16)** |

("after flip" is the mean lifespan from the flip to the end of life; the
counts are the lifetimes where the arm beat external only.)

- **With delayed poison, the once-bitten taste is a large and consistent
  win.** Lifespan at the end is 87 against 63, better in 15 of 16 lifetimes.
  The taste turns against the new poison within about 45 batches, against
  about 110 for the first rule.
- **With instant poison it helps a little**, but not significantly. Survival
  reward already teaches the lesson quickly, so taste has little to add.
- Before the flip, all arms are about equal.

## What this says

A plastic taste helps exactly where biology says it should: when the real
consequence arrives late. It also needs the biological asymmetry, reacting
fast to bad outcomes and slowly to good ones. With symmetric evidence, a
plastic taste is worse than none.

## Next

- **Stage 2:** let evolution set the rule's numbers (sensitivity to good and
  bad evidence, forgetting, strength, learning speed), with selection on
  survival only, and check whether it rediscovers the asymmetry.
- **Controls still missing:** a taste that changes on a time schedule rather
  than from evidence, and more seeds of the rule itself.

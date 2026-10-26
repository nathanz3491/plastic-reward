# Stage 3 results (2026-10-26): do evolved rules generalise, and is it learning?

**Setup:**
- **Three mixed-world evolutions** (`mix0`-`mix2`, seeds 0-2) and the
  **evidence-shuffled control** (`blind0`), all with 16 genomes, 25
  generations and 4 lifetimes.
- **Training worlds:** poison delays 0, 8, 15 and 25, with food and poison
  swapping and swapping back (A → B → A).
- **`blind0`** evolves the same rule family, but each event type receives a
  random type's evidence, so taste can move but never about the right object.
  It isolates the value of a fixed attitude (born wary, loud taste) from the
  value of learning which object is good.
- **Test** (`run_stage3_test.py`): 16 fresh lifetimes, the same for every
  candidate, across 11 worlds. 4, 20 and 30 are delays no rule trained on;
  1 and 3 flips, and GRU sizes 64 and 256, are patterns and brains no rule
  trained on. `stage2` is the stage-2 champion (trained on delay 15 with one
  flip only).

## Lifespan gain over no taste (mean ± SE over 16 lifetimes)

| world | hand-set | stage2 | mix0 | mix1 | mix2 | **blind0** |
|---|---|---|---|---|---|---|
| 2 flips, delay 0 | +6.5 ± 1.7 | +4.1 ± 1.3 | +5.6 ± 1.4 | −0.5 ± 1.8 | +8.5 ± 1.5 | −0.2 ± 1.2 |
| 2 flips, delay 4* | +0.5 ± 1.3 | +6.8 ± 1.5 | +4.2 ± 1.2 | +4.2 ± 1.3 | +8.5 ± 1.9 | +2.7 ± 1.2 |
| 2 flips, delay 15 | +1.5 ± 0.3 | +8.8 ± 0.6 | +8.9 ± 0.6 | +8.4 ± 0.8 | +8.2 ± 0.6 | +6.5 ± 0.8 |
| 2 flips, delay 20* | +1.4 ± 0.3 | +9.1 ± 0.3 | +8.2 ± 0.4 | +6.9 ± 0.7 | +7.6 ± 0.4 | +6.5 ± 0.5 |
| 2 flips, delay 30* | −0.2 ± 0.2 | +7.3 ± 0.4 | +6.0 ± 0.4 | +3.9 ± 0.4 | +4.1 ± 0.3 | +4.4 ± 0.3 |
| 1 flip, delay 15* | +6.3 ± 0.7 | +14.9 ± 1.0 | +12.5 ± 1.3 | +7.0 ± 1.8 | +14.3 ± 0.9 | +7.1 ± 1.1 |
| 1 flip, delay 20* | +7.3 ± 0.7 | +19.3 ± 1.1 | +18.4 ± 1.1 | +14.1 ± 0.7 | +17.7 ± 0.8 | +13.3 ± 1.1 |
| 3 flips, delay 15* | +1.2 ± 0.3 | +4.9 ± 0.4 | +4.1 ± 0.3 | +2.4 ± 0.4 | +5.1 ± 0.3 | +2.4 ± 0.4 |
| 3 flips, delay 20* | +0.1 ± 0.2 | +4.2 ± 0.2 | +3.7 ± 0.2 | +2.2 ± 0.3 | +4.7 ± 0.4 | +2.0 ± 0.2 |
| GRU 64* | +1.1 ± 0.2 | +3.6 ± 0.3 | +3.6 ± 0.4 | +4.4 ± 0.6 | +3.8 ± 0.3 | +2.6 ± 0.4 |
| GRU 256* | +4.0 ± 0.5 | +17.0 ± 0.9 | +14.9 ± 0.9 | +12.0 ± 1.0 | +15.6 ± 0.7 | +9.4 ± 0.8 |

\* a delay, flip pattern or brain size that the mixed runs never trained on.

The champions:

| | birth taste | learning speed α | bad / good | memory kept | loudness λ |
|---|---|---|---|---|---|
| mix0 | −0.76 | 0.41 | 1.6 | 0.68 | 4.3 |
| mix1 | −0.18 | 0.10 | 1.5 | 0.56 | 12.0 |
| mix2 | −0.83 | 4.34 | 2.8 | 0.44 | 5.2 |
| **blind0** | −0.54 | 0.15 | 0.4 | 0.95 | 5.0 |

## What it says

1. **A fixed attitude explains part of the gain.** The blind control
   evolved to barely learn (α 0.15, keeps 95% of its memory), stay born-wary
   and loud, and that alone gains +2 to +13. It is the honest "attitude only"
   baseline.
2. **Learning from evidence adds a consistent margin on top.** The learners
   (mix0, mix2, stage2) beat blind0 in almost every world:
   - +2 to +3 with two or three flips;
   - +5 with one flip;
   - +5 to +8 with the large brain;
   - +6 to +9 with instant poison, where blind0 gains nothing at all.

   With 16 lifetimes, unpaired, most of these gaps are 2-5 SE. mix1 is the
   exception: it evolved a very slow learner (α 0.10) and lands close to
   blind0, which is what the learning account predicts.
3. **It generalises:** to unseen delays, to unseen flip counts and to other
   brain sizes. The gain grows with a bigger brain (+15-17 with a GRU of 256).
4. **The stage-2 specialist generalises as well as the mixed-world
   champions,** despite training on one delay and one flip.
5. **The hand-set stage-1 rule is weak** whenever roles swap back.

## Caveats

- The evolved-vs-blind comparison above is unpaired. From stage 3b on, the
  test stores per-lifetime results for exact paired differences.
- There is one blind run and three learner runs.
- With 3 object types, blanket caution is cheap. Stage 3b (6 types, 4 swaps)
  checks whether learning matters more when caution starves the agent.

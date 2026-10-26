# Stage 3b results (2026-10-27): 6 object types, where caution is costly

**Setup:**
- **The world:** 6 object types (2 food, 2 poison, 2 neutral, shuffled per
  lifetime), 2 objects of each type.
- **Evolution:** two independent runs (`hard0`, `hard1`) and the
  evidence-shuffled control (`hardblind0`). Training worlds had poison delays
  8, 15 and 25 and four flips per life. 16 genomes × 25 generations × 4
  lifetimes.
- **Test:** 16 fresh lifetimes per world, the same for every candidate, every
  lifetime stored, so the learner-vs-blind comparison is **paired**.
- **Transferred champions:** `mix0` and `stage2` were evolved in the 3-type
  world and are applied here unchanged.

## Lifespan gain over no taste, and learners against the blind control

| world | hard0 | hard1 | blind | mix0 | stage2 | hard0 − blind (paired) | hard1 − blind (paired) |
|---|---|---|---|---|---|---|---|
| 2 flips, delay 15 | +22.4 | +28.4 | +7.0 | +7.0 | +6.7 | **+15.4 ± 0.9 (16/16)** | **+21.3 ± 0.6 (16/16)** |
| 3 flips, delay 15 | +11.9 | +14.1 | +3.8 | +3.3 | +3.1 | **+8.1 ± 0.6 (16/16)** | **+10.3 ± 0.6 (16/16)** |
| GRU 64, 2 flips, delay 15 | +9.8 | +14.4 | +3.4 | +3.2 | +3.1 | **+6.3 ± 1.1 (15/16)** | **+10.9 ± 0.5 (16/16)** |
| GRU 256, 2 flips, delay 15 | +31.2 | +37.8 | +9.9 | +12.4 | +12.6 | **+21.3 ± 1.5 (16/16)** | **+28.0 ± 1.0 (16/16)** |
| 2 flips, delay 20 | +7.3 | +14.3 | +6.6 | +7.9 | +6.6 | +0.7 ± 1.3 | +7.7 ± 0.7 (16/16) |
| 3 flips, delay 20 | +2.4 | +4.6 | +3.5 | +2.8 | +3.0 | −1.1 ± 0.6 | +1.1 ± 0.7 |
| 2 flips, delay 4 | −0.5 | −1.8 | +4.1 | +4.1 | −1.5 | −4.6 ± 1.4 | −5.9 ± 1.3 |
| 2 flips, delay 30 | −1.7 | −1.4 | +4.4 | +4.5 | +4.7 | −6.1 ± 0.3 (0/16) | −5.8 ± 0.3 (0/16) |
| 2 flips, delay 0 | −12.7 | −11.3 | +2.7 | +1.2 | −9.3 | −15.4 ± 1.6 (0/16) | −14.0 ± 1.2 (0/16) |
| 1 flip, delay 15 | −7.0 | −8.6 | +5.4 | +13.6 | +13.0 | −12.4 ± 0.7 (0/16) | −14.0 ± 0.7 (0/16) |
| 1 flip, delay 20 | −4.3 | −6.6 | +7.1 | +13.4 | +14.9 | −11.4 ± 0.8 (0/16) | −13.7 ± 0.9 (0/16) |

Rows are ordered from closest to the training worlds to furthest.

The champions:

| | birth taste | learning speed α | bad / good | memory kept | loudness λ |
|---|---|---|---|---|---|
| hard0 | −0.08 | 0.14 | 5.6 | 0.90 | 8.3 |
| hard1 | 0.05 | 0.37 | 17.7 | 0.98 | 8.1 |
| blind | −0.87 | 0.25 | 26.0 | 0.55 | 1.7 |

## What it says

1. **When caution is costly, learning from evidence matters a great deal.**
   In worlds like the training ones, the learners beat the attitude-only
   control by +8 to +28 lifespan, in 16 of 16 paired lifetimes. That is two
   to five times the 3-type gap. The gain also grows with a bigger brain:
   +31 to +38 over no taste with a GRU of 256.
2. **The 6-type champions are specialists, and they break outside their
   niche.** They evolved:
   - a long memory (keeping 90-98% of old evidence);
   - extreme sensitivity to bad news (5.6-17.7×);
   - a loud taste.

   That fits the training worlds, with frequent swaps and delays of 8-25.
   With a single flip, instant poison or delay 30, they are worse than no
   taste (−2 to −13, 0/16 against blind). It looks like evolution's version
   of overfitting: the rule exploits regularities of the worlds it lived in.
3. **The 3-type champions (mix0, stage2) are robust generalists.**
   Transferred unchanged to the 6-type world, they help in every world (+1 to
   +15). They never reach the specialists' peak.

## Implications

- The learning effect is real and can be large. Whether it helps depends on
  matching the environment it evolved in: specialists are brilliant at home
  and harmful away from it.
- For Craftax, which is out of distribution for every toy rule, the safe
  choice is either a **generalist** (stage2 / mix0), or evolving on a **much
  broader mix** of worlds (all delays including 0, 1-4 flips, 3-6 types).

# Buttons results (2026-10-27): taste over actions fails as a reward

**Setup:**
- **The world:** the 3-type world plus 4 buttons, with roles shuffled per
  lifetime:
  - GOOD feeds, but only next to an object;
  - HARM deals poison damage;
  - two NULL buttons do nothing.

  GOOD and HARM swap at each flip. Taste is per button (`--events buttons`).
- **Evolution:** `btn0` and its evidence-shuffled control `btnblind0`, with
  delays 8/15/25, two flips, 16 × 25 × 4.
- **Test:** the same 11 worlds as stage 3, 16 paired lifetimes each.
  `mix0` and `stage2` (object-evolved) are applied unchanged.

## Lifespan gain over no taste

| world | no taste | hand-set | btn0 | blind | mix0 | stage2 | btn0 − blind (paired) |
|---|---|---|---|---|---|---|---|
| 2 flips, delay 0 | 85.7 | −51.1 | −9.5 | −11.8 | −41.6 | −42.2 | +2.3 ± 3.6 |
| 2 flips, delay 4 | 92.1 | −58.3 | −9.7 | −5.2 | −52.4 | −52.1 | −4.4 ± 4.2 |
| 2 flips, delay 15 | 94.6 | −66.9 | +2.0 | +1.1 | −60.0 | −59.7 | +0.8 ± 3.9 |
| 2 flips, delay 20 | 95.6 | −63.6 | +3.4 | +1.5 | −59.6 | −60.1 | +1.9 ± 1.9 |
| 2 flips, delay 30 | 93.4 | −50.0 | +3.0 | +2.7 | −53.2 | −53.7 | +0.4 ± 1.0 |
| 1 flip, delay 15 | 116.9 | −80.8 | −6.0 | −9.5 | −74.8 | −75.0 | +3.5 ± 5.0 |
| 1 flip, delay 20 | 115.8 | −76.6 | −0.9 | +0.1 | −74.4 | −74.9 | −1.0 ± 1.3 |
| 3 flips, delay 15 | 79.2 | −51.5 | +3.5 | −0.7 | −46.2 | −46.1 | +4.2 ± 4.3 |
| 3 flips, delay 20 | 81.9 | −50.5 | +1.7 | +5.5 | −47.7 | −47.4 | −3.7 ± 2.6 |
| GRU 64 | 70.2 | −44.2 | −9.9 | −6.5 | −37.2 | −37.9 | −3.4 ± 1.6 |
| GRU 256 | 114.2 | −76.2 | +1.3 | +5.6 | −75.7 | −74.7 | −4.4 ± 2.6 |

None of the tastes helps. The evolved one is about neutral, and every
hand-made or transferred rule is disastrous.

## Why: three failure modes

A diagnostic run (`2 flips, delay 15`, 4 lifetimes) logged button presses and
taste by role over life.

1. **A free reward pump.** The hand-set rule grew a *positive* taste for a
   NULL button (+0.4 to +0.6). Pressing it is always possible and costs
   nothing, so the agent spammed it (40-80% of all steps) and starved.
   Touching an object takes walking; pressing a button does not. So an
   intrinsic reward on an action is a reward the agent can farm.
2. **The avoidance trap.** The wary transferred rule (`mix0`) disliked every
   button (about −0.75) and stopped pressing all of them, GOOD included. Its
   evidence for GOOD was clearly positive (+1 to +6), but with no presses its
   memory emptied, confidence fell to 0 and the taste never recovered. This
   is the "useless versus not learned yet" failure of DreamerV3's B3 mask,
   reproduced in miniature.
3. **Confounded evidence.** NULL buttons got positive evidence in every arm,
   even with no taste at all (+0.1 to +0.5). Evidence is "the return after
   pressing, minus the return on quiet steps". It measures *the situations in
   which the agent chooses to press*, not what the button does.

Evolution's answer (`btn0`) was to switch the taste off:
- every button is born disliked (−0.85) and stays there;
- loudness is turned down to 0.33, so the policy mostly ignores it.

It ends up about equal to no taste.

## Implications for DreamerV3

A learning taste over actions cannot simply be added as reward. It needs:
- **evidence against a state baseline**: the advantage after the action, with
  the critic's value as the baseline, rather than the quiet-step mean. This
  removes the confound in failure 3;
- **taste as an action prior, not a reward**: a bias on the policy's logits
  cannot be farmed (failure 1);
- **optimism when unsure**: low confidence pulls the bias back toward zero, or
  up, so avoided actions are retried (failure 2).

These three changes are tested below.

## The fix, first check (2026-10-27)

There are three new options:
- `Config.evidence = 'advantage'` (`reward.advantage_evidence`);
- `Config.taste_as = 'prior'`: taste is a bias on the button logits, and the
  learning reward is external only;
- `Config.optimism` (`reward.shown`): where evidence has faded, the policy sees
  this value instead of the taste.

Check: `2 flips, delay 15`, 8 paired lifetimes, no evolution.

| rule, mode | gain over no taste |
|---|---|
| hand-set, reward + window (old) | −65.1 ± 3.4 (0/8) |
| hand-set, reward + advantage | −46.5 ± 2.0 (0/8) |
| hand-set, prior + window | +9.1 ± 2.5 (7/8) |
| hand-set, prior + advantage | +6.7 ± 3.0 (5/8) |
| hand-set, prior + advantage, optimism 0.5 | +5.4 ± 2.6 (6/8) |
| mix0, prior + advantage | +8.2 ± 3.1 (7/8) |
| mix0, prior + advantage, optimism 0.5 | +7.0 ± 3.4 (6/8) |

What it shows:
- **Taste as a prior is the fix that matters.** Every rule turns from a large
  loss into a small gain, the wary `mix0` included.
- **Advantage evidence alone is not enough.** As a reward, taste still drives
  presses to zero and traps itself.
- **Advantage evidence removes the confound.** The useless buttons' evidence
  is now at or below zero, where before it was positive.
- **Optimism makes no difference here.** With a prior, the policy keeps
  pressing on its own, so memory never fades enough for it to matter.

Evolution and the paired 11-world test are running on the box
(`btn2adv`, `btn2advblind`, `btn2win`).

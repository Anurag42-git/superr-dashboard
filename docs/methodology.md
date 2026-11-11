# Methodology — Pressure Performance Rating

**SupeRR Selector Season 2 · Round 1**
*Identify the most reliable players under pressure.*

Data: **707 matches, 168,596 deliveries** — IPL 2022–2026 and T20Is from 2022.

---

## 1. Final rankings

These are the lists entered on Unstop, in rank order. "Top-10 chance" is the share of 300 bootstrap
re-samples of matches in which the player stayed in the top 10.

### Top 10 batters under pressure

| Rank | Player | Team | Score | Rating | Consistency | Pressure balls | Top-10 chance |
|---|---|---|---|---|---|---|---|
| 1 | Tim David | Royal Challengers Bengaluru | 2.45 | +24.9 | 69% | 363 | 98% |
| 2 | Suryakumar Yadav | Mumbai Indians | 1.76 | +19.1 | 70% | 315 | 90% |
| 3 | Andre Russell | Kolkata Knight Riders | 1.70 | +19.2 | 68% | 251 | 75% |
| 4 | Heinrich Klaasen | Sunrisers Hyderabad | 1.50 | +19.4 | 63% | 320 | 87% |
| 5 | Rovman Powell | Kolkata Knight Riders | 1.24 | +16.6 | 65% | 508 | 70% |
| 6 | Jos Buttler | Gujarat Titans | 1.13 | +15.1 | 67% | 186 | 72% |
| 7 | Nicholas Pooran | Lucknow Super Giants | 1.05 | +15.9 | 62% | 322 | 62% |
| 8 | Rinku Singh | Kolkata Knight Riders | 0.89 | +16.1 | 59% | 386 | 52% |
| 9 | Shreyas Iyer | Punjab Kings | 0.85 | +14.8 | 61% | 227 | 56% |
| 10 | Namboori Tilak Varma | Mumbai Indians | 0.61 | +11.5 | 65% | 545 | 30% |

### Top 10 bowlers under pressure

| Rank | Player | Team | Score | Rating | Consistency | Pressure balls | Top-10 chance |
|---|---|---|---|---|---|---|---|
| 1 | Matheesha Pathirana | Kolkata Knight Riders | 3.18 | +17.0 | 75% | 497 | 94% |
| 2 | Jasprit Bumrah | Mumbai Indians | 2.23 | +14.6 | 61% | 443 | 71% |
| 3 | Mustafizur Rahman | Kolkata Knight Riders | 1.99 | +11.8 | 66% | 434 | 63% |
| 4 | Sunil Narine | Kolkata Knight Riders | 1.89 | +11.6 | 64% | 361 | 58% |
| 5 | Varun Chakravarthy | Kolkata Knight Riders | 1.72 | +10.6 | 64% | 593 | 65% |
| 6 | Shaheen Afridi | Pakistan | 1.45 | +8.9 | 64% | 264 | 36% |
| 7 | Abrar Ahmed | Pakistan | 1.45 | +7.0 | 71% | 269 | 40% |
| 8 | Sufiyan Muqeem | Pakistan | 1.38 | +7.2 | 69% | 151 | 45% |
| 9 | Kuldeep Yadav | Delhi Capitals | 1.32 | +9.3 | 59% | 454 | 36% |
| 10 | Noor Ahmad | Chennai Super Kings | 1.29 | +12.0 | 47% | 399 | 38% |

\* Rating is the situation-adjusted value per pressure ball, ×100. +24.9 means +0.249 runs of value
per ball more than an average player in the same situation.

---

## 2. How pressure is defined

Pressure is not one number on the scoreboard. A team needing 12 an over with 8 wickets in hand in
over 8 is relaxed; the same rate with 3 wickets left in over 17 is not. Four signals are combined
into a **Pressure Index (PI)** for every delivery:

- **Stage (S)** — how much of the innings is gone: `S = ((balls bowled + 1) / 120) ^ 0.75`. Early
  balls cannot decide a match; late balls can.
- **Closeness (C)** — how balanced the contest is: `C = 1 − |2·WP − 1|`, where WP is the batting
  side's win probability from a gradient-boosted model trained on match states (balls left, wickets,
  runs, target, required and current rate, competition). Out-of-fold AUC is **0.68** for the first
  innings and **0.89** for chases.
- **Squeeze (Q)** in a chase — `0.65 × required-rate strain + 0.35 × wicket strain`, where
  rate strain `= clip((required rate − 6) / 10, 0, 1)` and wicket strain
  `= clip((wickets lost − 2) / 6, 0, 1)`.
- **Squeeze (Q)** in the first innings — `0.5 × how far the score is behind the typical score at that
  ball (per 35 runs) + 0.5 × wicket strain`. A first innings is under pressure when the team is in
  trouble.

```
PI = 100 × S × (0.5 × C + 0.5 × Q)     chase
PI = 100 × S × Q                       first innings
```

Closeness is uninformative before a target exists, which is why it is omitted in the first innings.

**Cut-offs, fixed before ranking players:** high-pressure ball = **PI ≥ 26.0** (top 25%, 42,149
balls); ordinary ball = **PI ≤ 10.5** (bottom 50%).

### Why this separates pressure from ordinary play

| | High pressure | Ordinary |
|---|---|---|
| Chase median required rate | 12.83 | 7.33 |
| Wickets lost (mean) | 5.29 | 1.28 |
| Balls left (mean) | 30 | 80 |
| Win-probability swing per ball | 2.56 pts | 1.75 pts |
| Wicket falls on | 7.42% of balls | 4.53% of balls |

The index is built from match state only, never from the outcome of the ball itself, yet the average
swing in win probability rises across its deciles. That is the independent check that it measures
real stakes.

---

## 3. How performance is measured

### Run value of a ball

Every delivery gets a run value: runs scored (including extras) minus a wicket cost, minus what an
average player produces in the same situation.

The wicket cost is the number of runs a wicket is worth, estimated by regressing the change in win
probability on runs and wickets:

| | Powerplay | Middle | Death |
|---|---|---|---|
| First innings | 10.9 | 9.3 | 7.2 |
| Chase | 7.3 | 6.1 | 3.4 |

"Same situation" means the same competition type (IPL or T20I), innings, over, wickets lost (0–2,
3–5, 6+) and, for batters, batting-order group (1–3, 4–5, 6–7, 8+) or, for bowlers, pace or spin.
This stops finishers being compared with openers, and death-over bowlers with powerplay bowlers. A
bowler's value is the same number with the sign reversed.

### Player baseline and pressure rating

For every player we compute the average run value on ordinary balls (their usual level) and on
high-pressure balls. Because 120–300 pressure balls is a small sample, we use an **empirical-Bayes**
estimate: the pressure rating starts from what the player's usual level predicts, then moves toward
the observed pressure value in proportion to how many balls we have. Between-player spread was
estimated from the data and equals about **1,196 balls** (batters) and **1,216 balls** (bowlers) of
prior weight. In plain terms: a player needs a long record before their pressure results outweigh
their general quality.

### Consistency

We count each match in which the player faced or bowled at least 6 high-pressure balls and mark it
positive if the net run value was above zero.

```
Consistency = (positives + 3) / (matches + 6)
```

which pulls small samples toward 50%.

### Final score

```
Score = 0.65 × z(pressure rating) + 0.35 × z(consistency)
```

where `z` is standardised among eligible players. The brief asks for players who deliver
*consistently*, which is why consistency carries a third of the weight.

### A clutch-shift test, and why it is not scored

We also tested the gap between each player's pressure level and their usual level. For batters the
estimated between-player variance of this gap is indistinguishable from zero, and for bowlers it is
small. In other words, the data do not show players who are reliably better or worse under pressure
than their general quality predicts; they show players whose general quality is high and whose
pressure results hold up. The shift is therefore reported in the dashboard but given no weight.
Adding it back at 25% keeps **9 of 10 batters and 8 of 10 bowlers** unchanged.

### Supporting metrics (not in the score)

For readability the dashboard also shows classic numbers on pressure balls: strike rate, boundary %,
dot % and dismissals for batters; economy, dot %, boundaries and wickets for bowlers.

---

## 4. Fair comparisons

- **Sample size** — a player must have at least **120 high-pressure balls**, at least **10
  high-pressure appearances** (6+ balls each) and at least **60 ordinary balls** so that a baseline
  exists. This leaves **45 batters and 108 bowlers**. Everyone else is not ranked.
- **IPL and T20I together** — the situation baseline is split by competition type, so a run in the
  higher-scoring IPL is compared with other IPL runs. Players who appear in both are pooled; the
  match-state model also uses competition type as an input.
- **Opposition strength** — balls against associate nations are given half weight; balls against
  full members and IPL teams get full weight.
- **Match context** — expected values depend on over, wickets lost and role, so players are never
  credited for simply batting at the death or bowling in the powerplay.
- **Data hygiene** — only IPL 2022–2026 and T20I matches are used. Asia Cup Rising Stars and the
  Emerging Teams Asia Cup (A-team events) are excluded. Super overs, abandoned or no-result matches
  and rain-revised matches are excluded because the target is not comparable. Wides and no-balls do
  not count as balls faced or bowled; non-striker run-outs are not charged to the striker; run-outs
  are not credited to the bowler.

### Evidence for the top 10

Each bar in the dashboard shows the middle 90% of ranks the player received across 300 re-samples of
matches. Narrow bars (for example Tim David and Matheesha Pathirana) are well supported. Wide bars
mean the player is in the group of contenders but the exact order is uncertain.

### Why the leaders lead

**Batters.** Tim David scored 641 runs off 363 high-pressure balls (strike rate 176.6, 21
dismissals) and remained clear of the field after adjusting for role and situation. The top group is
dominated by middle-order and finishing batters because high pressure is concentrated in the second
half of an innings; openers rarely face enough high-pressure balls to qualify.

**Bowlers.** Matheesha Pathirana conceded 504 runs from 440 high-pressure balls (economy 6.87, 40
wickets, 36.4% dot balls) and has the highest consistency among the top 10 (75% of pressure matches
positive).

---

## 5. Reliability and limitations

### Checks we ran

| Check | Batters | Bowlers |
|---|---|---|
| Top-10 overlap when only the pressure rating is used (no consistency) | 9/10 | 8/10 |
| Top-10 overlap with 50/50 rating and consistency weights | 9/10 | 8/10 |
| Top-10 overlap with 80/20 weights | 9/10 | 10/10 |
| Top-10 overlap when pressure = top 30% of balls | 7/10 | 7/10 |
| Top-10 overlap when pressure = top 20% of balls | 8/10 | 8/10 |
| Split-half correlation of the final score (two random halves of matches, 30 repeats) | 0.15 | 0.07 |

### What the checks say

- The lists are robust to how we set the weights and the pressure threshold: at least 7 of 10 names
  survive every change, and 8–10 of 10 survive most.
- The split-half correlation is low. Two random halves of the matches give only weakly similar
  orderings. This is a known property of pressure statistics: a player sees roughly 150–550
  high-pressure balls in four years, and ball-level outcomes are very noisy. We chose a shrinkage
  method and a minimum sample precisely because of this, but nobody should read the exact order
  within the top 10 as settled.
- Because the rating borrows heavily from each player's usual level, it rewards players who are good
  in general and do not fall away under pressure. It cannot, on this data, identify players who are
  specifically better in the moment.

### Limitations

- **Bye and leg-bye runs are not recorded per ball** in the file, so they are counted as 1 run each.
  First-innings totals match the stated margin exactly in only about 30% of decided-by-runs matches,
  so a small amount of noise remains in the match state.
- **The win-probability model has no team-strength or venue input**, which makes first-innings win
  probability close to 50/50 and is why first-innings pressure is defined through trouble (behind
  par, wickets lost) instead of closeness.
- **Opposition strength is a two-level weight**, not a full rating. T20I players who face weaker
  sides may be slightly flattered; franchise players face a consistently strong pool.
- **Players below the minimum balls** (for example, many openers and part-time bowlers) are not
  ranked, even if their few pressure balls were excellent.
- **Four seasons of data and one hold-out-free sample**: the ranking has not been tested on data the
  model never saw.

---

## 6. Reproducibility

All steps use the supplied CSV files only: filter matches, build ball-level match states, fit the
win-probability model, compute the Pressure Index, compute run values with situation baselines,
apply the shrinkage and consistency, and rank.

```bash
pip install -r requirements.txt
python code/superr_pipeline.py     # writes results/*.csv and results/data.json
```

Ratings for every eligible player are in `results/bat_ratings_all_eligible.csv` and
`results/bowl_ratings_all_eligible.csv`; the two submission lists are in
`results/top10_submission.csv`.

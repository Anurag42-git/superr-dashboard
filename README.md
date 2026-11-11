# Who delivers when it matters?

**Pressure-performance ratings for IPL and T20I batters and bowlers.**

Built for **Rajasthan Royals — SupeRR Selector Season 2, Round 1**: *identify the most reliable
players under pressure.*

**Live dashboard:** https://royals-dashboard.netlify.app  
**Full report (PDF):** [`report/SupeRR_Selector_Methodology_Report.pdf`](report/SupeRR_Selector_Methodology_Report.pdf)  
**Method write-up:** [`docs/methodology.md`](docs/methodology.md)

---

## The answer

### Top 10 batters under pressure

| # | Player | Team | Score | Rating | Consistency | Pressure balls | Top-10 chance |
|---|--------|------|-------|--------|-------------|----------------|---------------|
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

| # | Player | Team | Score | Rating | Consistency | Pressure balls | Top-10 chance |
|---|--------|------|-------|--------|-------------|----------------|---------------|
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

**Reading the columns.** *Score* is the ranking metric. *Rating* is the situation-adjusted value
per pressure ball, ×100 — so +24.9 means +0.249 runs of value per ball more than an average player
in the same situation. *Consistency* is the share of pressure matches in which the player's net
value was positive, shrunk toward 50%. *Top-10 chance* is how often the player stayed in the top 10
across 300 bootstrap re-samples of matches.

```
Score = 0.65 × z(rating) + 0.35 × z(consistency)
```

Both terms are standardised across the eligible pool. The brief asks for players who deliver
*consistently*, so consistency carries a third of the weight.

---

## How pressure is defined

Pressure is a property of the **situation**, never of the outcome, so it is computed before any
runs or wickets from the ball are counted:

```
PI = 100 × Stage × (0.50 × Closeness + 0.50 × Squeeze)   [chase]
PI = 100 × Stage × Squeeze                               [first innings]
```

| Component | Definition |
|---|---|
| **Stage** | `((balls bowled + 1) / 120)^0.75` — late balls can decide a match, early ones cannot |
| **Closeness** | `1 − \|2·WP − 1\|`, where WP is the batting side's win probability from a gradient-boosted model (out-of-fold AUC **0.68** first innings, **0.89** chases) |
| **Squeeze (chase)** | `0.65 × rate strain + 0.35 × wicket strain`; rate strain `= clip((RRR − 6)/10, 0, 1)`, wicket strain `= clip((wkts lost − 2)/6, 0, 1)` |
| **Squeeze (1st inns)** | `0.50 × (runs behind par / 35) + 0.50 × wicket strain` — a first innings is pressured when the team is in trouble |

**Cut-offs, fixed before any player was ranked:** high-pressure = **PI ≥ 26.0** (top 25%,
**42,149 balls**); ordinary = **PI ≤ 10.5** (bottom 50%).

**Independent check that it measures real stakes.** High-pressure balls come with a chase median
required rate of **12.83** versus **7.33**, an average of **5.29** wickets lost versus **1.28**, and
**30** balls left versus **80**. Win probability moves **2.56** points per ball versus **1.75**, and a
wicket falls on **7.42%** of balls versus **4.53%**.

---

## How performance is measured

1. **Run value per ball** — runs scored (including extras) minus the cost of a wicket, minus what an
   average player produces in the same situation. Wicket cost is estimated by regressing the change
   in win probability on runs and wickets: **10.9 / 9.3 / 7.2** (first innings: powerplay / middle /
   death) and **7.3 / 6.1 / 3.4** (chase). A bowler's value is the same number with the sign reversed.
2. **Same situation** means the same competition type, innings, over, wickets-lost band (0–2, 3–5,
   6+) and role group — batting order 1–3 / 4–5 / 6–7 / 8+, or pace versus spin. Finishers are never
   compared with openers, or death bowlers with powerplay bowlers.
3. **Empirical Bayes** — 120–300 pressure balls is a small sample, so the pressure rating starts at
   what the player's usual level predicts and moves toward the observed pressure value in proportion
   to the evidence: prior weight ≈ **1,196 balls** for batters, **1,216** for bowlers.
4. **Consistency** — matches with at least 6 pressure balls, positive if net value > 0:
   `(positives + 3) / (matches + 6)`, which pulls small samples toward 50%.

### Eligibility

**45 batters and 108 bowlers** qualify: at least **120 pressure balls**, at least **10 pressure
appearances**, and at least **60 ordinary balls** so a baseline exists. Everyone else is not ranked.

---

## Validation

| Check | Batters | Bowlers |
|---|---|---|
| Top-10 overlap using the pressure rating only (no consistency) | 9/10 | 8/10 |
| Top-10 overlap with 50/50 rating and consistency weights | 9/10 | 8/10 |
| Top-10 overlap with 80/20 weights | 9/10 | 10/10 |
| Top-10 overlap when pressure = top 30% of balls | 7/10 | 7/10 |
| Top-10 overlap when pressure = top 20% of balls | 8/10 | 8/10 |
| Split-half correlation of the final score (two random halves of matches, 30 repeats) | **0.15** | **0.07** |

The lists are robust to how the weights and the pressure threshold are set — at least 7 of 10 names
survive every change, and 8–10 of 10 survive most. The split-half correlation is low, and that is the
honest headline: four seasons give a player roughly 150–550 pressure balls, and ball-level outcomes
are noisy. The exact order inside the top 10 is not settled by this data.

---

## What this does not claim

- **It cannot identify pressure specialists.** The gap between a player's pressure level and their
  usual level has a between-player variance indistinguishable from zero for batters. The rating
  therefore rewards players who are good in general and do not fall away under pressure; it does not
  find players who are specifically better in the moment. The shift is shown in the dashboard but
  carries no weight in the score (adding it at 25% keeps 9/10 batters and 8/10 bowlers unchanged).
- **Byes and leg-byes are not recorded per ball** in the source, so they are counted as 1 run each.
  First-innings totals match the stated margin exactly in only about 30% of matches decided by runs.
- **No team-strength or venue input** in the win-probability model, which is why first-innings
  pressure is defined through trouble rather than closeness.
- **Opposition strength is a two-level weight**, not a full rating: balls against associate nations
  get half weight.
- **No hold-out sample.** Every check is within the same four seasons.

---

## Repository layout

```
superr-dashboard/
├── index.html                     # the dashboard (single file, no build step)
├── report/
│   └── SupeRR_Selector_Methodology_Report.pdf
├── results/
│   ├── top10_submission.csv       # the two ranked lists, as entered on Unstop
│   ├── bat_ratings_all_eligible.csv
│   └── bowl_ratings_all_eligible.csv
├── code/
│   └── superr_pipeline.py         # full pipeline, stage by stage
└── data/raw/                      # the six supplied CSVs (git-ignored)
```

## Reproduce

```bash
pip install -r requirements.txt
# put ball_by_ball.csv, matches.csv, competitions.csv, teams.csv and
# competition_team_player_mapping.csv in data/raw/
python code/superr_pipeline.py     # writes results/*.csv
```

The pipeline runs in five stages and caches intermediates as `*.pkl` between them. It writes
`results/data.json`, the per-player payload that `index.html` embeds; the dashboard's interactive
pressure meter additionally carries a small coefficient block from the win-probability model.

## Data

707 matches (301 IPL, 406 T20I) and **168,596 deliveries**, IPL 2022–2026 and T20Is from 2022.
Asia Cup Rising Stars and Emerging Teams Asia Cup are excluded, as are super overs, abandoned and
rain-revised matches (their targets are not comparable).

---

*Anurag — B.Tech CSE (AI & Analytics), MIT-ADT University, Pune.*

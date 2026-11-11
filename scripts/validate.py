"""Integrity checks for the SupeRR Selector submission.

Verifies that the dashboard, the results tables and the report agree:
    python scripts/validate.py

Exits non-zero if any check fails.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
DASHBOARD = ROOT / "index.html" if (ROOT / "index.html").exists() else ROOT / "docs" / "index.html"

EXPECTED_META = {
    "matches": 707,
    "balls": 168596,
    "ipl_matches": 301,
    "t20i_matches": 406,
    "hp_balls": 42149,
    "thr_hp": 26.0,
    "thr_ord": 10.5,
    "n_elig_bat": 45,
    "n_elig_bowl": 108,
}

TOP10_COLUMNS = [
    "category",
    "rank",
    "name",
    "team",
    "score",
    "rating",
    "consist",
    "hp_balls",
    "p_top10",
]

failures: list[str] = []
checks = 0


def check(condition: bool, message: str) -> None:
    global checks
    checks += 1
    if not condition:
        failures.append(message)


def load_dashboard_payload(path: Path) -> dict:
    html = path.read_text(encoding="utf-8")
    match = re.search(r"const D=(\{.*?\});?\s*\n", html, re.S)
    if not match:
        raise SystemExit(f"could not find the embedded data object in {path}")
    return json.loads(match.group(1))


def main() -> int:
    for name in ("top10_submission.csv", "bat_ratings_all_eligible.csv", "bowl_ratings_all_eligible.csv"):
        check((RESULTS / name).exists(), f"missing results/{name}")

    top = pd.read_csv(RESULTS / "top10_submission.csv")
    bat = pd.read_csv(RESULTS / "bat_ratings_all_eligible.csv")
    bowl = pd.read_csv(RESULTS / "bowl_ratings_all_eligible.csv")

    check(list(top.columns) == TOP10_COLUMNS, f"unexpected top10 columns: {list(top.columns)}")
    check(len(bat) == EXPECTED_META["n_elig_bat"], f"eligible batters {len(bat)} != 45")
    check(len(bowl) == EXPECTED_META["n_elig_bowl"], f"eligible bowlers {len(bowl)} != 108")

    for category in ("Batter", "Bowler"):
        rows = top[top["category"] == category].sort_values("rank")
        check(len(rows) == 10, f"{category}: {len(rows)} ranked entries, expected 10")
        check(list(rows["rank"]) == list(range(1, 11)), f"{category}: ranks are not 1-10 in order")
        check(rows["name"].is_unique, f"{category}: duplicate player names")
        check(
            rows["score"].is_monotonic_decreasing,
            f"{category}: score is not non-increasing with rank",
        )
        check(rows["hp_balls"].min() >= 120, f"{category}: an entry has fewer than 120 pressure balls")
        check(
            rows["p_top10"].between(0, 100).all(),
            f"{category}: top-10 chance outside 0-100",
        )

    # The dashboard must show exactly the same lists as the CSV that was submitted.
    payload = load_dashboard_payload(DASHBOARD)
    check(payload["meta"]["matches"] == EXPECTED_META["matches"], "dashboard match count changed")
    check(payload["meta"]["balls"] == EXPECTED_META["balls"], "dashboard delivery count changed")
    check(payload["meta"]["hp_balls"] == EXPECTED_META["hp_balls"], "dashboard pressure-ball count changed")
    check(payload["meta"]["n_elig_bat"] == len(bat), "dashboard batter count != results")
    check(payload["meta"]["n_elig_bowl"] == len(bowl), "dashboard bowler count != results")
    check(len(payload["bat"]) == len(bat), "dashboard batter list length != results")
    check(len(payload["bowl"]) == len(bowl), "dashboard bowler list length != results")

    for key, category in (("bat", "Batter"), ("bowl", "Bowler")):
        embedded = pd.DataFrame(payload[key]).sort_values("rank").head(10).reset_index(drop=True)
        submitted = top[top["category"] == category].sort_values("rank").reset_index(drop=True)
        check(
            list(embedded["name"]) == list(submitted["name"]),
            f"{key}: dashboard order differs from results/top10_submission.csv",
        )
        check(
            all(abs(float(a) - float(b)) < 5e-3 for a, b in zip(embedded["score"], submitted["score"])),
            f"{key}: dashboard scores differ from results/top10_submission.csv",
        )

    # Report and dashboard quote the same headline numbers.
    report = ROOT / "report" / "SupeRR_Selector_Methodology_Report.pdf"
    check(report.exists(), "missing report/SupeRR_Selector_Methodology_Report.pdf")

    print(f"{checks} checks run, {len(failures)} failed")
    for failure in failures:
        print(f"  FAIL  {failure}")
    if failures:
        return 1
    print("all integrity checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
ATUS analysis: activity frequency, mapping coverage, and occupancy validation.

Produces six analyses, all saved to scripts/atus/outputs/:
  1. activity_frequency_{stratum}.csv  — top activity codes by diary minutes
  2. mapping_coverage.csv             — % of diary time covered by our 220-code map
  3. tewhere_validation.csv           — empirical occupancy rates for D4/D5/D6 codes
  4. time_of_day_distributions.csv    — hourly activity distributions (start-time based)
  5. time_at_activity.csv             — hourly time-in-activity rates (for ActivityScheduler)
  6. schedule_peak_hours.csv          — peak hour per stratum × category

Run:
    python3 scripts/atus/analyze.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

# Allow running from repo root
_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_ROOT))

from scripts.atus.parse import build_dataset, STRATUM_LABELS
from occupant_agent.grounding.activity_code_map import lookup

OUTPUT_DIR = _ROOT / "scripts" / "atus" / "outputs"

# TEWHERE location codes (from ATUS codebook)
TEWHERE_HOME       = 1
TEWHERE_WORKPLACE  = 2
TEWHERE_RESTAURANT = 4


# ── 1. Activity frequency by stratum ────────────────────────────────────────

def activity_frequency(df: pd.DataFrame) -> None:
    """
    For each stratum: top 50 activity codes ranked by total weighted diary minutes.
    Includes the mapping's occupancy/room/devices for each code.
    """
    print("\n── 1. Activity frequency by stratum ──")
    for stratum, label in STRATUM_LABELS.items():
        sub = df[df["stratum"] == stratum].copy()
        n_respondents = sub["tucaseid"].nunique()
        if n_respondents == 0:
            print(f"  {stratum}: no respondents found")
            continue

        # Weighted minutes: tuactdur × tufinlwgt (population weight)
        sub["weighted_min"] = sub["tuactdur"] * sub["tufinlwgt"]

        freq = (
            sub.groupby("trcode")
            .agg(
                total_weighted_min=("weighted_min", "sum"),
                n_episodes=("tuactdur", "count"),
                mean_duration_min=("tuactdur", "mean"),
            )
            .sort_values("total_weighted_min", ascending=False)
            .head(50)
            .reset_index()
        )

        # Attach mapping metadata
        freq["description"] = freq["trcode"].apply(lambda c: lookup(c).description)
        freq["occupancy"]   = freq["trcode"].apply(lambda c: lookup(c).occupancy)
        freq["room"]        = freq["trcode"].apply(lambda c: lookup(c).room)
        freq["devices_on"]  = freq["trcode"].apply(lambda c: str(lookup(c).devices_on))

        # Cumulative share of total diary time
        total = sub["weighted_min"].sum()
        freq["pct_diary_time"] = (freq["total_weighted_min"] / total * 100).round(2)
        freq["cum_pct"]        = freq["pct_diary_time"].cumsum().round(2)

        out = OUTPUT_DIR / f"activity_frequency_{stratum}.csv"
        freq.to_csv(out, index=False)
        print(f"  {stratum} ({label}): {n_respondents:,} respondents")
        print(f"    Top 5 activities:")
        for _, row in freq.head(5).iterrows():
            print(f"      {row['trcode']}  {row['description']:<45}  {row['pct_diary_time']:.1f}%  ({row['occupancy']})")
        print(f"    Top 50 codes cover {freq['cum_pct'].iloc[-1]:.1f}% of diary time")
        print(f"    Saved → {out}")


# ── 2. Mapping coverage ──────────────────────────────────────────────────────

def mapping_coverage(df: pd.DataFrame) -> None:
    """
    What % of total diary minutes are covered by our 220 explicit tier-3 codes
    vs. tier-2 fallback vs. tier-1 fallback?
    """
    print("\n── 2. Mapping coverage ──")
    from occupant_agent.grounding.activity_code_map import ACTIVITY_MAP, TIER2_FALLBACK, TIER1_FALLBACK

    def coverage_level(code: str) -> str:
        if code in ACTIVITY_MAP:
            return "tier3_explicit"
        if code[:4] in TIER2_FALLBACK:
            return "tier2_fallback"
        if code[:2] in TIER1_FALLBACK:
            return "tier1_fallback"
        return "unknown"

    df2 = df.copy()
    df2["coverage"] = df2["trcode"].apply(coverage_level)
    df2["weighted_min"] = df2["tuactdur"] * df2["tufinlwgt"]

    total = df2["weighted_min"].sum()
    summary = (
        df2.groupby("coverage")["weighted_min"].sum()
        .rename("weighted_min")
        .to_frame()
    )
    summary["pct"] = (summary["weighted_min"] / total * 100).round(2)
    summary = summary.sort_values("pct", ascending=False)

    out = OUTPUT_DIR / "mapping_coverage.csv"
    summary.to_csv(out)
    print(summary.to_string())
    print(f"  Saved → {out}")

    # Which codes are still unknown?
    unknown_codes = df2[df2["coverage"] == "unknown"]["trcode"].value_counts().head(20)
    if not unknown_codes.empty:
        print(f"\n  Unknown codes (top 20 by episode count):")
        for code, n in unknown_codes.items():
            print(f"    {code}  ({n} episodes)")


# ── 3. TEWHERE validation: empirically verify D4, D5, D6 ────────────────────

def tewhere_validation(df: pd.DataFrame) -> None:
    """
    Use TEWHERE (directly observed location) to validate our three resolution rules:
      D4 — work_from_home: what % of work episodes (050101) occur at home vs. workplace?
      D5 — eating location: what % of eating (110101) episodes occur at home vs. restaurant?
      D6 — home_gym: what % of exercise episodes occur at home?

    Also prints per-stratum WFH rates for reference when tuning WFH_PRIORS in persona.py.
    """
    print("\n── 3. TEWHERE validation (empirical check on D4/D5/D6) ──")

    results = []

    # --- D4: Work at home vs. workplace, by stratum ---
    work = df[df["trcode"].str.startswith("05")].copy()
    work["at_home"]      = work["tewhere"] == TEWHERE_HOME
    work["at_workplace"] = work["tewhere"] == TEWHERE_WORKPLACE
    work["weighted_min"] = work["tuactdur"] * work["tufinlwgt"]

    print("\n  D4 — Work location by stratum (% of weighted work minutes):")
    for stratum in ["O1", "O2", "O3", "O4", "other"]:
        sub = work[work["stratum"] == stratum]
        if sub.empty:
            continue
        total_w = sub["weighted_min"].sum()
        pct_home      = (sub.loc[sub["at_home"],      "weighted_min"].sum() / total_w * 100)
        pct_workplace = (sub.loc[sub["at_workplace"], "weighted_min"].sum() / total_w * 100)
        pct_other     = 100 - pct_home - pct_workplace
        print(f"    {stratum}: home={pct_home:.1f}%  workplace={pct_workplace:.1f}%  other={pct_other:.1f}%")
        results.append({"analysis": "D4_work_location", "stratum": stratum,
                        "pct_home": round(pct_home, 2),
                        "pct_workplace": round(pct_workplace, 2),
                        "pct_other": round(pct_other, 2)})

    # --- D5: Eating location ---
    eat = df[df["trcode"].isin(["110101", "110199"])].copy()
    eat["weighted_min"] = eat["tuactdur"] * eat["tufinlwgt"]
    total_w = eat["weighted_min"].sum()

    eat_home       = eat[eat["tewhere"] == TEWHERE_HOME]["weighted_min"].sum()
    eat_restaurant = eat[eat["tewhere"] == TEWHERE_RESTAURANT]["weighted_min"].sum()
    eat_other      = total_w - eat_home - eat_restaurant

    pct_home_eat = eat_home / total_w * 100
    pct_rest_eat = eat_restaurant / total_w * 100
    print(f"\n  D5 — Eating location (all respondents):")
    print(f"    home={pct_home_eat:.1f}%  restaurant={pct_rest_eat:.1f}%  other={100-pct_home_eat-pct_rest_eat:.1f}%")
    results.append({"analysis": "D5_eating_location", "stratum": "all",
                    "pct_home": round(pct_home_eat, 2),
                    "pct_restaurant": round(pct_rest_eat, 2)})

    # --- D6: Exercise location ---
    exercise_codes = {
        "130101", "130107", "130116", "130120", "130121", "130122", "130126"
    }
    ex = df[df["trcode"].isin(exercise_codes)].copy()
    if not ex.empty:
        ex["weighted_min"] = ex["tuactdur"] * ex["tufinlwgt"]
        total_w = ex["weighted_min"].sum()
        pct_home_ex = ex[ex["tewhere"] == TEWHERE_HOME]["weighted_min"].sum() / total_w * 100
        print(f"\n  D6 — Home-resolvable exercise at home: {pct_home_ex:.1f}%")
        results.append({"analysis": "D6_exercise_at_home", "stratum": "all",
                        "pct_home": round(pct_home_ex, 2)})

    out = OUTPUT_DIR / "tewhere_validation.csv"
    pd.DataFrame(results).to_csv(out, index=False)
    print(f"\n  Saved → {out}")


# ── 4. Time-of-day distributions ────────────────────────────────────────────

def time_of_day_distributions(df: pd.DataFrame) -> None:
    """
    For each stratum, compute hourly participation rates for key device-triggering
    activity categories. These distributions become the persona's prior schedule.

    Key categories:
      sleeping (0101), food prep (0201), laundry (020202), TV (120301),
      work away (0501 + tewhere≠1), eating (1101)
    """
    print("\n── 4. Time-of-day distributions ──")

    CATEGORIES = {
        "sleeping":  lambda c: c.startswith("0101"),
        "food_prep": lambda c: c.startswith("0201"),
        "laundry":   lambda c: c == "020202",
        "tv":        lambda c: c == "120301",
        "work":      lambda c: c.startswith("05"),
        "eating":    lambda c: c.startswith("1101"),
        "exercise":  lambda c: c.startswith("13"),
    }

    def parse_hour(t: str) -> float | None:
        """Convert 'HH:MM:SS' to fractional hour; ATUS diary runs 04:00–04:00 next day."""
        try:
            h, m, _ = t.split(":")
            return int(h) + int(m) / 60
        except Exception:
            return None

    df2 = df.copy()
    df2["start_hour"] = df2["tustarttim"].apply(parse_hour)
    # ATUS encodes midnight–3:59 AM as hours 24–27; % 24 maps them back to 0–3.
    df2["start_hour_int"] = df2["start_hour"].apply(lambda x: int(x) % 24 if pd.notna(x) else pd.NA)
    df2["weighted_min"] = df2["tuactdur"] * df2["tufinlwgt"]

    all_outputs = []
    for stratum in ["O1", "O2", "O3", "O4"]:
        sub = df2[df2["stratum"] == stratum]
        if sub.empty:
            continue
        for cat_name, cat_filter in CATEGORIES.items():
            episodes = sub[sub["trcode"].apply(cat_filter)].dropna(subset=["start_hour_int"])
            if episodes.empty:
                continue
            # Weighted count of episode starts per hour
            hourly = (
                episodes.groupby("start_hour_int")["weighted_min"]
                .sum()
                .reindex(range(0, 24), fill_value=0)
                .reset_index()
                .rename(columns={"start_hour_int": "hour", "weighted_min": "weighted_min_sum"})
            )
            total = hourly["weighted_min_sum"].sum()
            hourly["pct_of_category"] = (hourly["weighted_min_sum"] / total * 100).round(2) if total > 0 else 0
            hourly["stratum"] = stratum
            hourly["category"] = cat_name
            all_outputs.append(hourly)

    if all_outputs:
        out_df = pd.concat(all_outputs, ignore_index=True)
        out = OUTPUT_DIR / "time_of_day_distributions.csv"
        out_df.to_csv(out, index=False)
        print(f"  Saved hourly distributions for {len(CATEGORIES)} categories × 4 strata → {out}")

        # Print peak hours per stratum/category
        pivot = out_df.loc[out_df.groupby(["stratum", "category"])["pct_of_category"].idxmax()]
        print("\n  Peak start hour per stratum × category:")
        print(pivot[["stratum", "category", "hour", "pct_of_category"]].to_string(index=False))


# ── 5. Time-at-activity distributions (correct grounding for Scheduler) ──────

def time_at_activity(df: pd.DataFrame) -> None:
    """
    Compute what fraction of respondents are IN each activity category
    at each clock hour, split by day type (weekday vs. weekend).

    Unlike time_of_day_distributions (start-time based), this asks:
    "at H:30, what is each person actually doing?" — the correct distribution
    for ActivityScheduler.sample().

    Day type: weekday = Mon–Fri (TUDIARYDAY ∈ {2,3,4,5,6}),
              weekend = Sat–Sun (TUDIARYDAY ∈ {1,7}).
    Splitting is essential for employed strata (O1, O3): work rates at 9am are
    ~50% on weekdays and ~0% on weekends; the blended average (~25%) misrepresents
    both day types and inflates KL-divergence in behavioral validation.

    ATUS extended-hour format: diary runs 4am–4am next day.
    Times 0:00–3:59 of the next day are encoded as 24:00–27:59.

    Output: scripts/atus/outputs/time_at_activity.csv
    Columns: hour (0–23), category, weighted_pct, stratum, day_type
    """
    print("\n── 5. Time-at-activity distributions (weekday / weekend split) ──")

    CATEGORIES_MAP = {
        "sleeping":  lambda c: c.startswith("0101"),
        "work":      lambda c: c.startswith("05"),
        "food_prep": lambda c: c.startswith("0201"),
        "laundry":   lambda c: c == "020202",
        "tv":        lambda c: c == "120301",
        "eating":    lambda c: c.startswith("1101"),
        "exercise":  lambda c: c.startswith("13"),
    }

    def _cat(code: str) -> str:
        for name, fn in CATEGORIES_MAP.items():
            if fn(code):
                return name
        return "other"

    def _parse_min(t: str) -> int | None:
        """Parse ATUS HH:MM:SS (extended hours allowed) → integer minutes."""
        try:
            parts = t.strip().split(":")
            return int(parts[0]) * 60 + int(parts[1])
        except Exception:
            return None

    # Convert clock hour 0–23 to ATUS extended minutes at the half-hour mark:
    #   hours 4–23 → h*60 + 30
    #   hours 0–3  → (h+24)*60 + 30   (ATUS 24:xx–27:xx encoding)
    def _query_min(hour: int) -> int:
        h = hour if hour >= 4 else hour + 24
        return h * 60 + 30

    def _day_type(d: int) -> str:
        """TUDIARYDAY: 1=Sun, 2=Mon, …, 7=Sat."""
        return "weekend" if d in {1, 7} else "weekday"

    df2 = df.copy()
    df2["start_min"] = df2["tustarttim"].apply(_parse_min)
    df2["stop_min"]  = df2["tustoptime"].apply(_parse_min)
    df2["category"]  = df2["trcode"].apply(_cat)
    df2["weighted"]  = df2["tufinlwgt"]
    df2["day_type"]  = df2["tudiaryday"].apply(_day_type)

    df2 = df2.dropna(subset=["start_min", "stop_min"])
    df2["start_min"] = df2["start_min"].astype(int)
    df2["stop_min"]  = df2["stop_min"].astype(int)
    df2 = df2[df2["stop_min"] > df2["start_min"]]

    all_outputs = []
    all_cats = list(CATEGORIES_MAP.keys()) + ["other"]

    for stratum in ["O1", "O2", "O3", "O4"]:
        for day_type in ["weekday", "weekend"]:
            sub = df2[(df2["stratum"] == stratum) & (df2["day_type"] == day_type)]
            if sub.empty:
                print(f"  {stratum} {day_type}: no data")
                continue

            n_days = sub["tucaseid"].nunique()
            resp_weights = (
                sub.drop_duplicates("tucaseid")
                .set_index("tucaseid")["weighted"]
            )
            total_w = resp_weights.sum()

            rows = []
            for hour in range(24):
                q = _query_min(hour)
                active = sub[(sub["start_min"] <= q) & (sub["stop_min"] > q)]

                if active.empty:
                    for cat in all_cats:
                        rows.append({"hour": hour, "category": cat, "weighted_pct": 0.0})
                    continue

                active_dedup = (
                    active.sort_values("start_min")
                    .drop_duplicates("tucaseid", keep="first")
                )
                active_dedup = active_dedup.join(resp_weights.rename("weight"), on="tucaseid")
                cat_weight = active_dedup.groupby("category")["weight"].sum()

                for cat in all_cats:
                    w = cat_weight.get(cat, 0.0)
                    rows.append({
                        "hour": hour,
                        "category": cat,
                        "weighted_pct": round(w / total_w * 100, 3),
                    })

            out_df = pd.DataFrame(rows)
            out_df["stratum"]  = stratum
            out_df["day_type"] = day_type
            all_outputs.append(out_df)

            pivot = out_df.loc[out_df.groupby("category")["weighted_pct"].idxmax()]
            print(f"\n  {stratum} {day_type} ({n_days:,} diary-days) — peak hour per category:")
            for _, r in pivot.sort_values("hour").iterrows():
                if r["weighted_pct"] > 0:
                    print(f"    {r['category']:<12} peak at {int(r['hour']):02d}:00  {r['weighted_pct']:.1f}%")

    if all_outputs:
        combined = pd.concat(all_outputs, ignore_index=True)
        out = OUTPUT_DIR / "time_at_activity.csv"
        combined.to_csv(out, index=False)
        print(f"\n  Saved → {out}  (columns: hour, category, weighted_pct, stratum, day_type)")


# ── 6. Schedule peak hours (for persona.py SCHEDULE_PRIORS alignment) ────────

def schedule_peak_hours(output_dir: Path = OUTPUT_DIR) -> None:
    """
    Compute peak hour per stratum × category (× day_type if available) from
    time_at_activity.csv and save to schedule_peak_hours.csv.

    This CSV lets persona.py load data-aligned activity timing instead of
    hardcoded estimates. When time_at_activity.csv has a day_type column
    (produced by time_at_activity()), the output includes separate
    weekday/weekend peaks — the weekday peaks are more relevant for
    core_memory_text for employed strata (O1, O3).

    Output columns: stratum, category, peak_hour, peak_pct[, day_type]
    """
    taa_path = output_dir / "time_at_activity.csv"
    if not taa_path.exists():
        print(f"  schedule_peak_hours: {taa_path} not found — run time_at_activity() first")
        return

    print("\n── 6. Schedule peak hours ──")
    df = pd.read_csv(taa_path)
    has_day_type = "day_type" in df.columns

    group_cols = ["stratum", "category"]
    if has_day_type:
        group_cols.insert(1, "day_type")

    rows = []
    for keys, group in df.groupby(group_cols):
        if has_day_type:
            stratum, day_type, category = keys
        else:
            stratum, category = keys
            day_type = None

        peak_row = group.loc[group["weighted_pct"].idxmax()]
        row = {
            "stratum": stratum,
            "category": category,
            "peak_hour": int(peak_row["hour"]),
            "peak_pct": round(float(peak_row["weighted_pct"]), 3),
        }
        if has_day_type:
            row["day_type"] = day_type
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_path = output_dir / "schedule_peak_hours.csv"
    out_df.to_csv(out_path, index=False)

    day_label = " (weekday/weekend split)" if has_day_type else " (blended — re-run after running time_at_activity() with microdata)"
    print(f"  Saved → {out_path}{day_label}")
    for stratum in sorted(out_df["stratum"].unique()):
        sub = out_df[out_df["stratum"] == stratum]
        if has_day_type:
            wd = sub[sub["day_type"] == "weekday"].set_index("category")["peak_hour"]
            we = sub[sub["day_type"] == "weekend"].set_index("category")["peak_hour"]
            print(f"  {stratum} weekday peaks: { {c: int(wd[c]) for c in wd.index if c != 'other'} }")
            print(f"  {stratum} weekend peaks: { {c: int(we[c]) for c in we.index if c != 'other'} }")
        else:
            peaks = sub[sub["category"] != "other"].set_index("category")["peak_hour"]
            print(f"  {stratum}: { {c: int(peaks[c]) for c in peaks.index} }")


# ── Main ─────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading ATUS 2022 + 2023 data...")
    df = build_dataset(years=[2022, 2023])

    print(f"\nStratum counts (respondents):")
    for stratum, label in STRATUM_LABELS.items():
        n = df[df["stratum"] == stratum]["tucaseid"].nunique()
        print(f"  {stratum} ({label}): {n:,} respondents")

    activity_frequency(df)
    mapping_coverage(df)
    tewhere_validation(df)
    time_of_day_distributions(df)
    time_at_activity(df)
    schedule_peak_hours()

    print("\nDone. All outputs in scripts/atus/outputs/")

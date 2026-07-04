"""
ATUS microdata loader and demographic stratum classifier.

Loads activity, respondent, roster, and CPS files for specified years,
merges them into a flat episode-level DataFrame, and classifies each
respondent into a demographic stratum (O1–O4).

Usage:
    from scripts.atus.parse import build_dataset, STRATUM_LABELS
    df = build_dataset(years=[2022, 2023])
"""

from __future__ import annotations

import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "atus"

# ── Column selections ───────────────────────────────────────────────────────

ACT_COLS = {
    "TUCASEID",       # 14-digit case ID (join key)
    "TRCODE",         # 6-digit activity code
    "TUACTDUR",       # duration in minutes (last activity not truncated)
    "TUSTARTTIM",     # start time (HH:MM:SS string)
    "TUSTOPTIME",     # stop time (HH:MM:SS string)
    "TEWHERE",        # location code (1=home, 2=workplace, 4=restaurant, ...)
}

RESP_COLS = {
    "TUCASEID",
    "TUYEAR",
    "TUMONTH",
    "TUDIARYDATE",    # diary date (YYYYMMDD)
    "TUDIARYDAY",     # day of week (1=Sunday … 7=Saturday)
    "TELFS",          # labor force status (1=employed@work, 2=employed not@work, 3=unemployed, 4=not in LF)
    "TERET1",         # retirement (1=retired, 2=not retired, -1=N/A)
    "TRNUMHOU",       # number of persons in household
    "TRSPPRES",       # spouse/partner present (1=married spouse, 2=unmarried partner, 3=neither)
    "TRHHCHILD",      # HH children < 18 (1=yes, 2=no)
    "TRDPFTPT",       # full-time (1) vs part-time (2) for employed
    "TUFINLWGT",      # final person weight for population-level estimates
    "TRHOLIDAY",      # diary day is a holiday (1=yes, 0=no)
}

ROST_COLS = {
    "TUCASEID",
    "TULINENO",       # person line number (1=respondent)
    "TEAGE",          # age
    "TERRP",          # relationship to respondent (18/19=self, 20=spouse, 22=own child, ...)
    "TESEX",          # sex (1=male, 2=female)
}

CPS_COLS = {
    "TUCASEID",
    "GESTFIPS",       # state FIPS code (for climate zone matching to Pecan Street)
    "HEFAMINC",       # family income bracket (1–16 ordinal scale)
    "PEHSPNON",       # Hispanic (1=Hispanic, 2=non-Hispanic)
    "PTDTRACE",       # race (1=White, 2=Black, 3=Native American, 4=Asian, ...)
}

# ── Stratum definitions ─────────────────────────────────────────────────────

STRATUM_LABELS = {
    "O1": "Employed adult, single, 25–44",
    "O2": "Retired/not-in-LF, coupled, 65+",
    "O3": "Employed parent with children <18, 35–54",
    "O4": "Unemployed/not-in-LF (non-retired), 25–44",
}

# ── Loaders ─────────────────────────────────────────────────────────────────

def _load(year: int, stem: str, cols: set[str], str_cols: list[str]) -> pd.DataFrame:
    base = DATA_DIR / str(year) / "extracted" / f"{stem}_{year}"
    # BLS distributes .dat files (uppercase headers); IPUMS exports .csv (lowercase headers).
    path = base.with_suffix(".dat") if base.with_suffix(".dat").exists() else base.with_suffix(".csv")
    if not path.exists():
        raise FileNotFoundError(
            f"ATUS data not found: tried {base}.dat and {base}.csv\n"
            f"Download from https://www.bls.gov/tus/data.htm (BLS) or "
            f"https://www.atusdata.org/ (IPUMS ATUS) and place in {DATA_DIR}/<year>/extracted/\n"
            f"Note: IPUMS exports use different column names than BLS .dat files. "
            f"Rename IPUMS columns to match BLS format before use (e.g. CASEID→TUCASEID, DURATION→TUACTDUR)."
        )
    _cols_upper = {c.upper() for c in cols}
    dtype = {c: str for c in str_cols} | {c.lower(): str for c in str_cols}
    df = pd.read_csv(
        path,
        usecols=lambda c: c.upper() in _cols_upper,
        dtype=dtype,
        low_memory=False,
    )
    df.columns = df.columns.str.lower()
    return df


def load_activity(year: int) -> pd.DataFrame:
    df = _load(year, "atusact", ACT_COLS, str_cols=["TUCASEID", "TRCODE"])
    # Pad trcode to 6 digits (leading zeros sometimes stripped by pandas)
    df["trcode"] = df["trcode"].str.zfill(6)
    return df


def load_respondent(year: int) -> pd.DataFrame:
    return _load(year, "atusresp", RESP_COLS, str_cols=["TUCASEID"])


def load_roster(year: int) -> pd.DataFrame:
    return _load(year, "atusrost", ROST_COLS, str_cols=["TUCASEID"])


def load_cps(year: int) -> pd.DataFrame:
    return _load(year, "atuscps", CPS_COLS, str_cols=["TUCASEID"])


# ── Respondent demographics (merge roster + CPS onto respondent) ─────────────

def _respondent_demographics(year: int) -> pd.DataFrame:
    """
    Return one row per respondent with age, sex, and income from roster + CPS.
    Age and sex come from the roster (respondent's own row: terrp in {18,19}).
    Income and state FIPS come from CPS.
    """
    resp = load_respondent(year)
    rost = load_roster(year)
    cps  = load_cps(year)

    # Respondent's own demographics from roster (terrp 18 or 19 = self)
    self_row = rost[rost["terrp"].isin([18, 19])][["tucaseid", "teage", "tesex"]].copy()
    self_row = self_row.drop_duplicates("tucaseid")

    # CPS: state and income
    cps_slim = cps[["tucaseid", "gestfips", "hefaminc", "pehspnon", "ptdtrace"]].copy()
    cps_slim = cps_slim.drop_duplicates("tucaseid")

    demog = (
        resp
        .merge(self_row, on="tucaseid", how="left")
        .merge(cps_slim,  on="tucaseid", how="left")
    )
    demog["year"] = year
    return demog


# ── Stratum classifier ───────────────────────────────────────────────────────

def classify_stratum(df: pd.DataFrame) -> pd.Series:
    """
    Classify respondents into O1–O4 strata.

    Requires columns (lowercase): teage, telfs, teret1, trsppres, trhhchild.
    Returns a Series of stratum labels ("O1"…"O4" or "other"), same index as df.

    Priority when a respondent matches multiple definitions: O1 > O2 > O3 > O4.
    This is conservative — O1 (employed single 25-44) takes precedence over
    O4 (not-in-LF 25-44) for edge cases like leave-of-absence.

    See docs/methodology_decisions.md D9 for rationale.
    """
    age      = df["teage"]
    employed = df["telfs"].isin([1, 2])
    unemployed = df["telfs"] == 3
    not_in_lf  = df["telfs"] == 4
    retired    = df["teret1"] == 1
    coupled    = df["trsppres"].isin([1, 2])
    single     = df["trsppres"] == 3
    has_kids   = df["trhhchild"] == 1

    p1 = (age >= 25) & (age <= 44) & employed & single
    p2 = (age >= 65) & ~employed & coupled
    p3 = (age >= 35) & (age <= 54) & employed & has_kids
    p4 = (age >= 25) & (age <= 44) & (unemployed | (not_in_lf & ~retired))

    result = pd.Series("other", index=df.index, dtype=str)
    result[p4] = "O4"
    result[p3] = "O3"
    result[p2] = "O2"
    result[p1] = "O1"   # highest priority — applied last
    return result


# ── Main builder ─────────────────────────────────────────────────────────────

def build_dataset(years: list[int] | None = None) -> pd.DataFrame:
    """
    Load, merge, and classify ATUS data for the given years.

    Returns a flat DataFrame of activity episodes with respondent demographics
    and stratum label attached. One row = one ATUS activity episode (variable duration).

    Columns of interest after merge:
      trcode, tuactdur, tustarttim, tustoptime, tewhere,
      teage, tesex, telfs, trsppres, trhhchild, tufinlwgt,
      tudiaryday, trholiday, gestfips, hefaminc, stratum, year
    """
    if years is None:
        years = [2022, 2023]

    parts = []
    for year in years:
        print(f"  Loading {year}...")
        act   = load_activity(year)
        demog = _respondent_demographics(year)
        demog["stratum"] = classify_stratum(demog)

        # Columns to carry from respondent → activity rows
        resp_carry = [
            "tucaseid", "year", "tudiarydate", "tudiaryday", "trholiday",
            "tufinlwgt", "teage", "tesex", "telfs", "teret1",
            "trsppres", "trhhchild", "trnumhou", "trdpftpt",
            "gestfips", "hefaminc", "stratum",
        ]
        merged = act.merge(demog[resp_carry], on="tucaseid", how="left")
        parts.append(merged)
        n_resp = demog["tucaseid"].nunique()
        n_act  = len(act)
        strat_counts = demog["stratum"].value_counts().to_dict()
        print(f"    {n_resp:,} respondents, {n_act:,} activity episodes")
        print(f"    Stratum counts: {strat_counts}")

    df = pd.concat(parts, ignore_index=True)
    print(f"\n  Total: {df['tucaseid'].nunique():,} respondents, {len(df):,} episodes")
    return df

"""
ATUS microdata — download instructions and post-download extraction.

BLS blocks automated downloads. Follow the steps below to download manually.

=== Step 1: Download data files ===

Go to: https://www.bls.gov/tus/data.htm
Click the year link (e.g., "2022 Basic ATUS data files"), then:

  Under "2022 Basic ATUS Data Files" — download these zip files:
    atusact-2022.zip   — Activity file  (diary episodes: code, start, stop, duration)
    atusresp-2022.zip  — Respondent file (demographics: age, employment, HH size)
    atusrost-2022.zip  — Roster file (household member composition)
    atuscps-2022.zip   — CPS supplement (employment, income, children — needed for stratum classification)
    [skip] atussum-2022.zip  — Summary file (pre-aggregated totals; we compute our own)

  Under "2022 Basic ATUS Data Dictionaries (PDF Files)" — download the codebooks:
    atusintcodebk22.pdf    — Interview codebook (activity, respondent, roster variables)
    atuscpscodebk22.pdf    — CPS codebook (demographic variables from CPS supplement)

  Repeat the same steps for 2023 (https://www.bls.gov/tus/data/datafiles-2023.htm).

=== Step 2: Place files in the correct folders ===

  data/atus/2022/raw/        ← 2022 zip files (atusact-2022.zip, etc.)
  data/atus/2022/codebooks/  ← 2022 PDF codebooks
  data/atus/2023/raw/        ← 2023 zip files
  data/atus/2023/codebooks/  ← 2023 PDF codebooks

=== Step 3: Extract ===

  python3 scripts/atus/download.py --year 2022 --extract
  python3 scripts/atus/download.py --year 2023 --extract

  Extracted .dat files go to data/atus/{year}/extracted/

=== Why separate years? ===

  See docs/methodology_decisions.md D2: 2022-2023 are the primary datasets;
  2019 is the pre-COVID reference. Years are kept separate so we can combine
  deliberately and report exact coverage in the Methods section.

=== Alternative: IPUMS ATUS (easier, recommended for variable selection) ===

  https://www.atusdata.org/
  Register free → Create extract → Select variables below → Download CSV

  Key variables to include:
    CASEID, YEAR, MONTH, DAY, DAY_OF_WEEK
    AGE, SEX, EMPSTAT, HH_SIZE, FAMINCOME, RACE, HISPAN
    ACTIVITY (6-digit code), DURATION (minutes), START, STOP
    PWTFINL (person weight for population-level estimates)
    HHTENURE (own vs. rent — relevant for appliance ownership)
"""

import argparse
import zipfile
from pathlib import Path

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "atus"

# BLS uses hyphens in data file names: atusact-2022.zip (not underscores)
# Codebooks are shared across files; BLS names them by survey component, not by data file.
REQUIRED_FILES = {
    "atusact-{year}.zip":  "Activity file (diary episodes — primary)",
    "atusresp-{year}.zip": "Respondent file (demographics)",
    "atusrost-{year}.zip": "Roster file (household members)",
    "atuscps-{year}.zip":  "CPS supplement (detailed employment, income, children — needed for stratum classification)",
}

OPTIONAL_FILES = {
    "atuswho-{year}.zip":    "Who file (who was present during each episode — useful for multi-person HH)",
    "atussum-{year}.zip":    "Summary file (pre-aggregated totals; we compute our own — skip)",
    "atusrostec-{year}.zip": "Eldercare roster supplement — skip for v1",
}

# BLS codebook naming convention (two shared PDFs cover all files)
CODEBOOK_FILES = {
    "atusintcodebk{yy}.pdf": "Interview codebook (activity, respondent, roster variables)",
    "atuscpscodebk{yy}.pdf": "CPS codebook (demographic variables from CPS supplement)",
}


def check(year: int) -> None:
    base = DATA_DIR / str(year)
    yy = str(year)[2:]  # e.g. "22" from 2022
    print(f"\nChecking data/atus/{year}/")

    ok = missing = 0
    for template, desc in REQUIRED_FILES.items():
        fname = template.format(year=year)
        path = base / "raw" / fname
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            print(f"  [OK]      raw/{fname} ({desc}) — {size_mb:.1f} MB")
            ok += 1
        else:
            print(f"  [MISSING] raw/{fname} ({desc})")
            missing += 1

    print()
    for template, desc in CODEBOOK_FILES.items():
        fname = template.format(yy=yy)
        path = base / "codebooks" / fname
        if path.exists():
            size_mb = path.stat().st_size / 1e6
            print(f"  [OK]      codebooks/{fname} ({desc}) — {size_mb:.1f} MB")
            ok += 1
        else:
            print(f"  [MISSING] codebooks/{fname} ({desc})")
            missing += 1

    print()
    for template, desc in OPTIONAL_FILES.items():
        fname = template.format(year=year)
        path = base / "raw" / fname
        status = f"{path.stat().st_size / 1e6:.1f} MB" if path.exists() else "not present"
        print(f"  [OPTIONAL] raw/{fname} ({desc}) — {status}")

    print(f"\n  Summary: {ok} required files OK, {missing} missing")


def extract(year: int) -> None:
    raw_dir = DATA_DIR / str(year) / "raw"
    out_dir = DATA_DIR / str(year) / "extracted"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Extract required files; also extract optional who-file if present
    to_extract = list(REQUIRED_FILES) + ["atuswho-{year}.zip"]
    for template in to_extract:
        fname = template.format(year=year)
        path = raw_dir / fname
        if not path.exists():
            print(f"  [SKIP] {fname} not found")
            continue
        print(f"  Extracting {fname} → extracted/")
        with zipfile.ZipFile(path, "r") as zf:
            zf.extractall(out_dir)
    print(f"  Done. Files in data/atus/{year}/extracted/")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, choices=[2019, 2022, 2023], default=2023,
                        help="ATUS survey year to check or extract")
    parser.add_argument("--extract", action="store_true",
                        help="Extract zip files (run after placing zips in raw/)")
    parser.add_argument("--check-all", action="store_true",
                        help="Check status for all three years")
    args = parser.parse_args()

    if args.check_all:
        for yr in [2019, 2022, 2023]:
            check(yr)
    elif args.extract:
        extract(args.year)
    else:
        check(args.year)

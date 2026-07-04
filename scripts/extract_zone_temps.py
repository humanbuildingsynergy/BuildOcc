"""
Extract zone temperatures from EnergyPlus output and write the target CSV.

Reads examples/data/energyplus/eplusout.csv (EnergyPlus 23.1 run of the
DOE SF prototype, CZ2B, Tucson TMY3), extracts the peak-heat week
Aug 10-16, converts units, and writes examples/data/zone_temps_sample.csv.

EnergyPlus timestamps are end-of-interval (e.g. "08/10  00:10:00" is the
end of the 00:00-00:10 window).  This script shifts each timestamp back by
one timestep (10 min) to produce start-of-interval datetimes.
"""

import csv
from datetime import datetime, timedelta
from pathlib import Path

REPO = Path(__file__).parent.parent
INPUT = REPO / "examples" / "data" / "energyplus" / "eplusout.csv"
OUTPUT = REPO / "examples" / "data" / "zone_temps_sample.csv"

YEAR = 2025
START_DATE = (8, 10)
END_DATE = (8, 16)
TIMESTEP_MIN = 10


def parse_ep_datetime(date_str: str, year: int) -> datetime:
    """Parse EnergyPlus 'MM/DD  HH:MM:SS' end-of-interval timestamp."""
    s = date_str.strip()
    # EnergyPlus uses 24:00:00 for midnight end-of-day; normalize to next day
    if "24:00:00" in s:
        date_part = s.split()[0]
        month, day = map(int, date_part.split("/"))
        dt = datetime(year, month, day) + timedelta(days=1)
    else:
        # Parse as-is, then subtract timestep to convert end→start-of-interval
        dt = datetime.strptime(f"{year}/{s}", "%Y/%m/%d  %H:%M:%S")
    return dt - timedelta(minutes=TIMESTEP_MIN)


def in_range(dt: datetime) -> bool:
    start = datetime(YEAR, START_DATE[0], START_DATE[1])
    end = datetime(YEAR, END_DATE[0], END_DATE[1], 23, 59, 59)
    return start <= dt <= end


def main():
    if not INPUT.exists():
        raise FileNotFoundError(
            f"EnergyPlus output not found: {INPUT}\n"
            "Run EnergyPlus on examples/data/energyplus/ first."
        )
    rows = []
    with open(INPUT, newline="") as f:
        reader = csv.reader(f)
        header = next(reader)
        # Identify columns by keyword matching
        outdoor_col = next(
            (i for i, h in enumerate(header) if "Outdoor Air Drybulb" in h),
            None,
        )
        zone_col = next(
            (i for i, h in enumerate(header) if "Zone Mean Air Temperature" in h),
            None,
        )
        if outdoor_col is None:
            raise ValueError(
                f"'Outdoor Air Drybulb' column not found in {INPUT}. "
                f"Available headers: {header}"
            )
        if zone_col is None:
            raise ValueError(
                f"'Zone Mean Air Temperature' column not found in {INPUT}. "
                f"Available headers: {header}"
            )

        for raw in reader:
            if not raw or not raw[0].strip():
                continue
            try:
                dt = parse_ep_datetime(raw[0], YEAR)
            except ValueError:
                continue
            if not in_range(dt):
                continue
            outdoor_c = round(float(raw[outdoor_col]), 2)
            zone_c = round(float(raw[zone_col]), 2)
            rows.append((dt.strftime("%Y-%m-%d %H:%M:%S"), zone_c, outdoor_c))

    print(f"Extracted {len(rows)} rows for Aug {START_DATE[1]}-{END_DATE[1]}, {YEAR}")
    expected = 7 * 24 * (60 // TIMESTEP_MIN)
    if len(rows) != expected:
        print(f"  Warning: expected {expected} rows (7 days × {60 // TIMESTEP_MIN}/hr)")

    with open(OUTPUT, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["datetime", "zone_temp_c", "outdoor_temp_c"])
        writer.writerows(rows)

    print(f"Wrote {OUTPUT}")
    if rows:
        print(f"  First: {rows[0]}")
        print(f"  Last:  {rows[-1]}")
        temps = [r[1] for r in rows]
        print(f"  Zone temp range: {min(temps):.1f}–{max(temps):.1f} °C")


if __name__ == "__main__":
    main()

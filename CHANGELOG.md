# Changelog

All notable changes to BuildOcc are documented here.
This project adheres to [Semantic Versioning](https://semver.org/).

## [1.0.1] — 2026-09-02

A data-correctness release. Three defects in the ATUS grounding pipeline are fixed.
All of them change the bundled distributions, so **results produced with v1.0.0 are not
comparable to results produced with v1.0.1**. No public API changed.

### Fixed

- **Post-midnight hours were empty (hours 00:00–03:59).**
  `scripts/atus/analyze.py` treated ATUS start/stop times as an extended 24–27 hour
  encoding. ATUS actually records plain clock times, and carries the 04:00→04:00 diary
  ordering through activity sequence instead. The old code therefore matched nothing at
  hours 0–3 and dropped every episode that crossed midnight, which deleted essentially all
  overnight sleep. In v1.0.0, `time_at_activity.csv` summed to 0.00 for hours 0–3 and
  thinned through the evening (82 / 67 / 35 / 6 at hours 20–23). Both endpoints are now
  mapped onto a 04:00-anchored diary axis and midnight-crossing episodes are unwrapped.
  Every `(stratum, day_type, hour)` cell now sums to 100.

- **`ActivityScheduler` no longer fabricates sleep for hours 0–3.**
  When an hour carried no reference mass, the scheduler returned the literal `"sleeping"`
  for hours 0–3 rather than the uniform fallback used at every other hour — masking the
  empty-data bug above. With the corrected tables the branch is unreachable; it is
  retained only for a caller-supplied `outputs_dir` with sparse data, and now draws
  uniformly instead of asserting a category the data does not support.

- **Activity code map was shifted against the ATUS lexicon.**
  Three of the eight categories selected the wrong activities:

  | Category | v1.0.0 filter | What that code actually is | v1.0.1 filter |
  |---|---|---|---|
  | `food_prep` | prefix `0201` | Housework (cleaning, laundry, sewing) | prefix `0202` |
  | `laundry` | `020202` | Food presentation | `020102` |
  | `tv` | `120301` | Relaxing, thinking | `120303`/`120304` |

  Bundled code descriptions were shifted the same way (`020101` was labelled "Food and
  drink preparation" when it is Interior cleaning; `120303` was labelled "Computer use for
  leisure" when it is Television and movies). The map has been rebuilt against the
  official ATUS 2023 lexicon and expanded from 220 to 300 tier-3 codes.

- **O4 stratum selected the wrong respondents.**
  `TELFS` has five values (1 employed-at work, 2 employed-absent, 3 unemployed-on layoff,
  4 unemployed-looking, 5 not in labor force). The classifier documented four and used
  `TELFS == 4` for "not in labor force", so it selected unemployed-looking respondents and
  never included anyone actually out of the labor force. O4 was therefore "unemployed
  only" (n = 107) while its label described it as not employed. The `TERET1` screen has
  also been dropped: the codebook defines `TERET1` as *"do you currently want a job, either
  full or part time?"* with universe `TELFS = 5 AND TEAGE >= 50` — it is not a retirement
  flag and cannot apply to O4's 25–44 age band.

  **O4 is now n = 677** (unemployed or not-in-labor-force, ages 25–44). O1, O2 and O3
  membership is unchanged. The four modeled strata now cover 6,611 respondents.

### Changed

- **Nine activity categories instead of eight.** Travel (ATUS tier-1 code `18`) is now its
  own category rather than being absorbed into `other`, where it was the single largest
  component.
- Holiday diary days are pooled with weekends rather than weekdays, so a day off no longer
  contaminates the weekday work distribution.
- `occupant_agent/data/README.txt` now documents that `time_at_activity.csv`
  (time-in-activity) and `time_of_day_distributions.csv` (start-time) answer different
  questions and are not directly comparable, along with the 04:00 diary axis and the
  `TUFINLWGT` population weight.
- Package metadata declares the license as the SPDX expression `Apache-2.0`, so the
  license field renders as an identifier rather than the full license text.

## [1.0.0] — 2026-07-04

Initial public release: Python agent library, REST API, and MCP server, with ATUS-grounded
activity scheduling across four demographic strata, EIA RECS appliance priors, a
Park et al. (2023) memory stream, a demand-response signal typology, and behavioral and
energy validation metrics.

Bundled ATUS-derived data files shipped with the buildocc package.

These are population-weighted summary statistics derived from the American Time
Use Survey (ATUS) 2022-2023 public-use microdata (BLS). They are NOT the raw
microdata (which may not be redistributed).

Files:
  time_at_activity.csv              — hourly time-in-activity rates per stratum/day_type
  schedule_peak_hours.csv           — peak hour per stratum/category (for persona priors)
  activity_frequency_{O1,O2,O3,O4}.csv — top activity code prevalences per stratum
  time_of_day_distributions.csv     — probability mass per hour per activity category
  mapping_coverage.csv              — ATUS code map coverage diagnostics
  tewhere_validation.csv            — empirical occupancy rates by TEWHERE location

To regenerate with your own ATUS microdata:
  python scripts/atus/analyze.py
  # Then pass outputs_dir= to ActivityScheduler and AgentStore

Source: U.S. Bureau of Labor Statistics, American Time Use Survey, 2022-2023.
https://www.bls.gov/tus/

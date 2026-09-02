Bundled ATUS-derived data files shipped with the buildocc package.

These are population-weighted summary statistics derived from the American Time
Use Survey (ATUS) 2022-2023 public-use microdata (BLS). They are NOT the raw
microdata (which may not be redistributed).

Files:
  time_at_activity.csv        — TIME-IN-ACTIVITY rates per stratum/day_type.
                                P(category | hour): the share of the population
                                engaged in each category AT that clock hour,
                                measured by episode overlap at H:30. Every
                                (stratum, day_type, hour) cell sums to 100.
                                This is what ActivityScheduler samples from.
  time_of_day_distributions.csv — START-TIME distribution per stratum/category.
                                For one category, how its episode STARTS are
                                spread across the 24 hours (sums to 100 per
                                stratum x category). NOT comparable to
                                time_at_activity.csv: a category can have most of
                                its starts in the evening while the probability of
                                being IN it at any evening hour stays low.
  schedule_peak_hours.csv     — peak hour per stratum/category (for persona priors)
  activity_frequency_{O1,O2,O3,O4}.csv — top activity code prevalences per stratum,
                                with n_episodes and mean_duration_min per ATUS code
  occupancy_priors.csv        — P(at home | activity code) per stratum
  mapping_coverage.csv        — ATUS code map coverage diagnostics
  tewhere_validation.csv      — empirical occupancy rates by TEWHERE location

Diary axis: ATUS diaries run 04:00 to 04:00 the next day. Episodes that cross
midnight are mapped onto a 04:00-anchored axis, so hours 00:00-03:59 belong to
the same diary day as the preceding evening. Population weights use TUFINLWGT
(shared by the 2022 and 2023 files, so the two years pool directly).

To regenerate with your own ATUS microdata:
  python scripts/atus/analyze.py
  # Then pass outputs_dir= to ActivityScheduler and AgentStore

Source: U.S. Bureau of Labor Statistics, American Time Use Survey, 2022-2023.
https://www.bls.gov/tus/

# Methodology Decisions Log

This document records every non-obvious design decision made during development,
with rationale and literature grounding. It serves as:
- **Background reading** for the accompanying research paper (see citation in README)
- **User guide**: explains to downstream users why the agent behaves as it does

Decisions are numbered for cross-referencing. Each entry follows the format:
**Decision**, **Rationale**, **Alternative considered**, **Consequence if changed**.

---

## D1 — Temporal Resolution: 15-minute timestep

**Decision:** The simulation advances in 15-minute increments.

**Rationale:** ATUS diary records report activity episodes to the nearest minute,
but the minimum meaningfully resolvable interval given diary rounding and activity
transition latency is 15 minutes. This also matches the standard resolution of
building energy data (Pecan Street Dataport, DOE ecobee) and common BAS control
intervals (ASHRAE Guideline 36 recommends ≥15-minute HVAC control periods).

**Alternative considered:** 1-minute or 5-minute steps to capture short-duration
activities (e.g., microwaving food). Rejected: ATUS data does not reliably encode
sub-15-minute activities; finer steps would produce spurious precision.

**Consequence if changed:** Finer steps require a sub-interval timing model
(stochastic interpolation) that is not grounded in ATUS data. Coarser steps
(e.g., 30 min) lose the ability to distinguish morning vs. evening peaks.

---

## D2 — ATUS Year Selection: 2022 and 2023 as primary datasets

**Decision:** Primary persona grounding uses ATUS 2022 and 2023 microdata.
2019 is retained as a pre-COVID reference for sensitivity analysis.
2020 and 2021 are excluded from primary analysis.

**Rationale:**
- **2020–2021 excluded**: COVID-19 lockdowns produced anomalous occupancy
  patterns (near-universal WFH, near-zero commuting, no social activities
  outside the home). These years do not represent the behavioral equilibrium
  we intend to simulate and would systematically overestimate occupancy and
  underestimate travel-related HVAC setback periods.
- **2022–2023 selected**: Post-COVID normalization has stabilized; these years
  capture the new equilibrium including partial WFH adoption. Largest available
  sample sizes with current device ownership patterns (streaming services,
  smart appliances).
- **Multi-year file (2003–2023) as supplement only**: Used only when a specific
  demographic stratum has fewer than ~100 diary-days in the 2022–2023 combined
  sample. Older years carry outdated device ownership (pre-smart-TV, pre-streaming)
  which would bias appliance-use distributions.

**Alternative considered:** Single most recent year only. Rejected: sample sizes
per stratum are too small for reliable distribution fitting (especially for
less common profiles like O4 unemployed adult).

**Consequence if changed:** Including 2020–2021 would underestimate HVAC setback
frequency for employed profiles by an estimated 15–25% relative to the 2022–2023
equilibrium (based on BLS ATUS WFH supplement data showing return-to-office trends).

---

## D3 — Activity Code Mapping: Tier-3 → tier-2 → tier-1 fallback hierarchy

**Decision:** ATUS 6-digit activity codes are mapped to `(occupancy, room, devices)`
using a three-tier fallback lookup:
1. Exact 6-digit match (220 codes explicitly mapped)
2. First 4-digit prefix fallback (42 tier-2 groups)
3. First 2-digit prefix fallback (18 tier-1 categories)

**Rationale:** The ATUS lexicon contains ~400 tier-3 codes, but the long tail
of rare codes (e.g., "Playing racquetball", "Hunting or fishing") cannot
meaningfully be distinguished for building energy purposes. The tier-2 and
tier-1 fallbacks provide reasonable defaults for these codes without requiring
hand-coding every entry. High-frequency codes (sleeping, food prep, laundry,
TV, commuting) are all explicitly mapped at tier-3.

**Alternative considered:** Map only the top-N codes by diary frequency and
drop the rest. Rejected: missing codes cause silent errors in the simulation
that are hard to detect.

**Consequence if changed:** Removing the fallback hierarchy produces `None`
returns for ~35% of diary entries, requiring additional error handling.

---

## D4 — Occupancy Resolution: Work-from-Home Flag

**Decision:** ATUS code `050101` ("Work, main job") does not record work location.
Resolution rule: if `PersonaFlags.work_from_home = True`, occupancy → `home`
(HVAC holds comfort setpoint); else occupancy → `away` (HVAC enters setback).

**Rationale:** Post-2022, approximately 25–30% of employed US workers work from
home at least one day per week (BLS American Time Use Survey WFH supplement, 2023;
Barrero, Bloom & Davis, 2023, "The Evolution of Work from Home"). Treating all
work episodes as `away` would overestimate setback periods and underestimate HVAC
load for a material share of the employed population. The WFH flag is sampled from
ATUS/BLS demographic priors at persona initialization:
- O1 (employed single, 25–44): **24.3% WFH** (empirical — ATUS 2022–23, TEWHERE=1 during work episodes)
- O2 (retired couple, 65+): not applicable (not employed)
- O3 (employed parent, 35–54): **25.7% WFH** (empirical — ATUS 2022–23, TEWHERE=1 during work episodes)
- O4 (unemployed adult, 25–44): not applicable (no job)

**Alternative considered:** Model WFH as a fractional probability applied at each
work episode. Rejected for v1: adds stochasticity without ATUS-grounded per-episode
data; the binary flag is sufficient for population-level validation.

**Consequence if changed:** Removing the WFH flag and treating all work as `away`
underestimates HVAC load during work hours for WFH-prevalent profiles (O1, O3)
by an estimated 15–20% based on Census Bureau WFH prevalence data.

---

## D5 — Occupancy Resolution: Eating Location Inference

**Decision:** ATUS code `110101` ("Eating and drinking") does not record location.
Resolution rule: if the immediately preceding diary episode was a travel episode
(tier-1 code `18`), occupancy → `away` (eating out); else occupancy → `home`.

**Rationale:** ATUS records activity sequences with timestamps. A travel episode
immediately preceding an eating episode is a reliable proxy for eating at a
restaurant or work cafeteria. In the absence of a preceding travel episode, the
USDA Economic Research Service (2023) estimates ~60% of US meals are consumed at
home; treating the non-travel case as `home` is the statistically dominant choice
and avoids requiring a probabilistic model for a moderate-energy activity
(eating itself does not trigger appliances; the device impact is indirect via HVAC).

**Alternative considered:** Treat all eating as `ambiguous` and sample
probabilistically (60% home, 40% away). Rejected: adds stochasticity that is
not grounded in the ATUS sequence data, and eating episodes have low direct
energy impact (no appliance trigger).

**Consequence if changed:** Treating all eating as `home` would underestimate
away-from-home time by ~40 minutes/day for the average adult, slightly
underestimating HVAC setback duration.

---

## D6 — Occupancy Resolution: Home Gym Flag

**Decision:** ATUS exercise codes that are location-ambiguous (yoga, cardiovascular
equipment, weightlifting, aerobics, exercise machine, gymnastics, dancing) resolve
to `home` if `PersonaFlags.home_gym = True`, else `away`.

**Rationale:** ATUS does not record whether exercise occurred at home or at a gym.
Exercise equipment ownership is measurable from RECS 2020: approximately 12% of
US households own a treadmill or stationary bike (EIA RECS 2020, Table CE4.3).
The `home_gym` flag is sampled from RECS ownership rates by household income and
size stratum at persona initialization. Outdoor sports (running, walking, basketball,
etc.) are unconditionally `away` regardless of this flag.

**Alternative considered:** Default all exercise to `away`. Rejected: underestimates
occupancy for the ~12% of households with home equipment, which is a material
share for higher-income strata.

**Consequence if changed:** Removing the flag and defaulting all ambiguous exercise
to `away` slightly overestimates HVAC setback for the home-gym subpopulation,
with negligible impact on population-level averages.

---

## D7 — Energy Calculation: Power Rating × Runtime (No Physics)

**Decision:** Energy consumption is calculated as:
`energy_kWh = power_rating_w / 1000 × on_duration_h`
using manufacturer/EIA power ratings. No building physics model is used through
Phase 3.

**Rationale:** The primary contribution of Phases 1–2 is behavioral realism
(activity timing, occupancy patterns, device-use sequences), not thermal physics.
Introducing a physics model in Phase 1 would conflate behavioral uncertainty with
thermal model uncertainty, making validation ambiguous. Power rating × runtime is
the standard simplification used in behavioral energy simulation studies (see
Fischer et al. 2017, Energy and Buildings).

**Alternative considered:** EnergyPlus co-simulation from Phase 1. Rejected for
Phase 1–3: requires EnergyPlus IDF model calibrated to each household, which is
not available in the ATUS/Pecan Street dataset pairing. Planned for Phase 5 extension.

**Consequence if changed:** Adding physics increases accuracy for HVAC loads
(which depend on envelope, weather, and occupancy jointly) but requires a
calibrated building model per household — not available at ATUS scale.

---

## D8 — Device Set: Simulation Scope

**Decision:** The v1 implemented device set (appears in `env.devices` via `DEVICE_POWER_W`) is:
`hvac, lighting_living, lighting_bedroom, lighting_kitchen, tv, washer, dishwasher`

Additional device IDs (oven, range, microwave, range_hood, dryer, computer,
water_heater, refrigerator, pool_pump, gaming_console, etc.) appear in
`activity_code_map.py` as forward-looking entries for Phase 2 device expansion.
They never appear in `env.devices` in v1; `_device_activity_hint()` skips them
gracefully when absent.

**Rationale:** The v1 set covers the high-leverage demand-response targets
(HVAC, washer, dishwasher) and the primary lighting zones. Expanding to the
full appliance inventory in Phase 2 requires validated RECS-derived ownership
priors and power ratings per device; the activity_code_map entries are ready
to activate once DEVICE_POWER_W is extended.

**Alternative considered:** Whole-home energy only (no device disaggregation).
Rejected: device-level disaggregation is required to validate against Pecan Street
circuit-level data and to identify demand-response targets (washer, dryer, dishwasher).

**Consequence if changed:** Removing device disaggregation prevents circuit-level
CVRMSE validation and eliminates the demand-response scheduling analysis
(Applications 1 and 2).

---

## D9 — Demographic Profiles: ATUS Strata Selection

**Decision:** Four profiles, stratified from ATUS by employment status,
household composition, and age:

| ID | ATUS Stratum | Implemented | Demographic variation study |
|---|---|---|---|
| O1 | Employed adult, single, 25–44 | v1 | Phase 2 |
| O2 | Retired couple, 65+ | v1 | Phase 2 |
| O3 | Employed parent with children, 35–54 | v1 | Phase 4 |
| O4 | Unemployed adult, 25–44 | v1 | Phase 4 |

All four strata are fully implemented and available via `OccupantAgent.from_stratum()`
in v1. The "Phase" column refers to when the formal cross-stratum demographic
variation study (Phase 4) runs the profiles through the full validation and
intervention loop — not when the code is available.

**Rationale:** O1 and O2 represent the two most distinct behavioral archetypes
for building energy: high-away-time working adult vs. home-most-of-day retired
couple. The contrast between them is the clearest test of whether ATUS grounding
produces demographically differentiated energy profiles. O3 and O4 extend the
comparison to parent and unemployed archetypes for the journal paper (Phase 4).

**Alternative considered:** Use census-representative sampling across all strata
simultaneously. Rejected for Phase 2: too many confounders to interpret validation
results cleanly; targeted profiles allow cleaner novelty claims.

---

## D10 — Building Control Signal Types: Three-Category Taxonomy

**Decision:** The building control agent issues exactly three signal types:
- **Type A — Direct command**: "Turn off the dishwasher until after 9pm"
- **Type B — Competence-building (boost)**: explains the mechanism and rationale
- **Type C — Social norm (nudge)**: compares to similar household behavior

**Rationale:** This taxonomy is drawn from the boost vs. nudge debate in behavioral
economics (Hertwig & Grüne-Yanoff, 2017, Perspectives on Psychological Science)
and is the minimum set required for a factorial analysis of signal effectiveness.
The 26.6% advantage of competence-building over social norm signals reported by
Tiefenbeck et al. (2018, Nature Energy) provides an empirical calibration target
for Phase 3.

**Alternative considered:** Continuous spectrum of signal framing. Rejected:
not reproducible or comparable to prior literature; breaks the Phase 5 factorial
design.

**Consequence if changed:** Collapsing to two types (command vs. informational)
eliminates the boost/nudge distinction that is the primary behavioral economics
contribution of Paper 2.

---

## D11 — Platform Interface: Three-Layer Architecture

**Decision:** The agent is exposed through three layers:
1. Python library (Layer 1) — core logic, no network dependency
2. REST API via FastAPI (Layer 2) — platform-agnostic HTTP
3. MCP server (Layer 3) — wraps REST API for LLM-orchestrated systems

**Rationale:** MCP (Anthropic's Model Context Protocol) is an LLM-application
protocol; building energy platforms (EnergyPlus, VOLTTRON, OpenStudio) are not
MCP clients and cannot call an MCP server directly. The REST API layer is the
universal integration point. The MCP layer adds value specifically for
LLM-orchestrated building management systems (e.g., Home Assistant with MCP
client support). This architecture supports the broader adoption claim in Paper 1:
integrable with any platform via REST, optimized for LLM-orchestrated systems via MCP.

**Alternative considered:** MCP only. Rejected: EnergyPlus and VOLTTRON cannot
call MCP natively; would require bespoke adapter per platform.

---

## D12 — Environment State Schema: Frozen at v1.0

**Decision:** The `environment_state` schema (see `occupant_agent/environment/state.py`)
is frozen as v1.0 before Phase 2 validation begins. No field names or types may
change after Phase 2 starts.

**Rationale:** The validation pipeline (Phase 2) calls the REST API and MCP server
using this schema. Any schema change after validation begins would require
re-running all simulations, breaking reproducibility. The schema is designed to
map cleanly onto what EnergyPlus EMS variables and Home Assistant entity states
can natively provide, so no adapter layer is needed at the platform integration point.

**Fields (v1.0):**
- `schema_version`: str — frozen at `"1.0"`
- `timestep`: ISO 8601 datetime
- `zone_temp_c`: float (°C)
- `outdoor_temp_c`: float (°C)
- `tou_rate`: float ($/kWh)
- `thermostat_setpoint_c`: float | None (°C) — current setpoint; None if unknown
- `devices`: list of `{device_id, state: bool|float, power_w}`
- `rooms`: list of `{room_id, occupied: bool}`

---

## D13 — Validation Metrics: Two-Tier Approach

**Decision:** Validation uses two tiers applied in order:
1. **Behavioral tier** (must pass before energy tier):
   - HAR activity recognition F1 against CASAS Aruba/Milan datasets
   - KL-divergence between agent activity-start-time distributions and ATUS population distributions
   - KS-test on activity duration distributions
2. **Energy tier**:
   - CVRMSE < 30% (ASHRAE Guideline 14 threshold)
   - MBE < 10%
   - Pearson correlation of hourly load curves

**Rationale:** If behavioral validation fails, energy validation results are
uninterpretable — the right energy outcome from wrong behavior is coincidental.
Ordering the tiers enforces this dependency. Both tiers are required for a
publishable novelty claim; behavioral realism alone is insufficient for a building
energy venue (BuildSys, e-Energy).

**Comparison baseline:** AgentSense (Homer dataset, 21 participants) on identical
CASAS benchmarks. This is the direct novelty claim — ATUS grounding vs. Homer
grounding on the same evaluation.

---

*Created: 2026-06-27. Updated as decisions are made during development.*

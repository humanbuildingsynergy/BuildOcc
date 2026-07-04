# Datasets and Empirical Resources for LLM Occupant Agent Grounding

**Purpose:** Reference document for grounding and validating LLM-based occupant behavior agents in a smart home simulation. Covers four dimensions: (1) baseline time-use and activity patterns, (2) real smart home device-state data, (3) behavioral response to smart home suggestions/interventions, and (4) demographic-conditioned behavior variation.

---

## 1. Baseline Time-Use and Activity Patterns

### American Time Use Survey (ATUS)
- **Source:** U.S. Bureau of Labor Statistics — https://www.bls.gov/tus/
- **Access:** Free download via BLS website; also available through IPUMS (ipums.org/timeuse) with harmonized variables
- **Coverage:** Nationally representative sample of Americans ages 15+, continuous since 2003
- **Resolution:** Full 24-hour time diary in 15-minute increments; one diary day per respondent
- **What it captures:** All daily activities (sleeping, cooking, childcare, leisure, travel, household work) with start time, duration, location, and co-presence of other household members
- **Grounding use case:** Parse individual diary records into room-level occupancy probability distributions and activity-linked device-load schedules. Condition LLM agent daily routines on measured population behavior rather than LLM priors. Stratify by employment status, household composition, sex, age, and day-of-week.
- **Confidence:** High (3-0 adversarial vote)

### Multinational Time Use Study (MTUS)
- **Source:** Centre for Time Use Research — https://www.timeuse.org/mtus; also via IPUMS (https://www.ipums.org/timeuse/mtus)
- **Access:** Free registration required; available through IPUMS with harmonized codebook
- **Coverage:** 68 harmonized datasets from 22 countries, spanning surveys since the early 1960s
- **Resolution:** Total minutes per day in each activity category; each record represents a full 24-hour diary (minutes sum to 1,440)
- **What it captures:** 41 harmonized activity categories (original release; newer releases use 25 or 69 categories — verify release version); common person-level and household-level demographic variables across all surveys
- **Grounding use case:** Demographic-conditioned activity distributions beyond the U.S. context. Parameterize LLM agent personas by age, household size, sex, employment status, and country. Enables cross-national generalization of the simulation.
- **Note:** Activity category counts differ across MTUS release versions (41, 25, or 69). Confirm the specific release before use.
- **Confidence:** High (3-0 adversarial vote)

---

## 2. Real Smart Home Device-State Data

### Dataport / Pecan Street Dataset (NILMTK format)
- **Source:** Pecan Street Inc. — https://www.pecanstreet.org/dataport/
- **Access:** Academic/institutional access available via application to Pecan Street; some data released publicly through NILMTK
- **Coverage:** 669 U.S. households (subset of 722 total; filtered to those with ≥8 sub-meters)
- **Resolution:** 1-minute circuit-level and building-level electricity readings for one month
- **What it captures:** Circuit-level appliance data enabling NILM (non-intrusive load monitoring) disaggregation; can reconstruct individual device on/off schedules from aggregate consumption signals
- **Grounding use case:** Real appliance-level device-state ground truth for validating LLM agent appliance interaction models. Use NILM disaggregation to infer device on/off event sequences, then compare against agent-generated sequences.
- **Confidence:** High (3-0 adversarial vote for dataset existence and coverage)

### DOE ecobee Smart Thermostat Dataset (2017)
- **Source:** U.S. Department of Energy — https://www.osti.gov/dataexplorer/biblio/dataset/1854924
- **Access:** Publicly accessible via DOE OSTI Data Explorer
- **Coverage:** 1,000 single-family residences across four states: California, Texas, New York, and Illinois; full calendar year 2017
- **Resolution:** 5-minute intervals; thermostat setpoints, HVAC run states, and occupancy detection events
- **What it captures:** Thermostat setpoint sequences, heating/cooling schedules, occupancy detection from ecobee smart thermostats
- **Grounding use case:** Calibrate LLM agent HVAC interaction models including setpoint preferences stratified by climate zone. Serve as benchmark for testing agent demand-response compliance and thermostat override behavior.
- **Confidence:** High (3-0 adversarial vote)

### CASAS Smart Home Datasets (Aruba, Milan, Kyoto7, Cairo)
- **Source:** CASAS project, Washington State University — https://zenodo.org/communities/casas
- **Access:** Publicly available via Zenodo
- **Coverage:** Multiple single- and multi-resident smart homes; real binary event logs
- **Resolution:** Event-level (sub-second timestamps)
- **What it captures:** Binary motion sensor events, door/cabinet open-close events, device activation events (binary ON/OFF); real occupancy and device-use patterns in actual occupied homes
- **Grounding use case:** Used by AgentSense (AAAI 2026) as validation benchmarks for HAR (Human Activity Recognition). Can calibrate or validate LLM agent device interaction models directly against real occupant behavior sequences.
- **Note:** Orange4Home is a related dataset from Amiqual4Home environment and is similarly publicly available.
- **Confidence:** High (confirmed as public validation benchmarks in AgentSense paper)

---

## 3. Behavioral Response to Smart Home Suggestions and Interventions

### ecobee Thermostat Override Behavior Study
- **Source:** arXiv 1912.06705 — large-scale ecobee thermostat dataset study
- **Coverage:** 27,764 smart thermostats, 5-minute resolution
- **Key empirical finding:** Occupant thermostat override behavior follows an inverse relationship between setpoint deviation magnitude and override latency:
  - 1.1°C automated adjustment → overridden in ~30 minutes on average
  - 4.4°C automated adjustment → overridden in ~15 minutes on average
- **Grounding use case:** Calibrate LLM agent override probability distributions conditioned on setpoint delta. Provides empirically grounded response timing parameters for demand-response simulation scenarios.
- **Caveats:** No confidence intervals reported; one sub-study found 7/20 occupants showed no correlation with temperature; volunteer opt-in population may not be representative.
- **Confidence:** Medium (2-1 adversarial vote)

### Boost vs. Nudge RCT (Cambridge Behavioural Public Policy, 2023)
- **Source:** Cambridge Behavioural Public Policy — https://www.cambridge.org/core/journals/behavioural-public-policy/article/boosting-vs-nudging-sustainable-energy-consumption-a-longterm-comparative-field-test-in-a-residential-context/39875649817B5B6AF64DB77E6FC3EBDE
- **Coverage:** N=65 student dormitory residents; 29-week randomized controlled trial
- **Key empirical findings:**
  - Competence-building "boost" interventions (bi-weekly energy-saving tips via QR code): **6.3% reduction** below baseline electricity use
  - Traditional nudge interventions (social comparison + feedback graphs + goal-setting): **20.4% increase** above baseline
  - Boost group outperformed nudge group by **26.6%** (Kruskal-Wallis: H=9.97, p=0.007)
  - Bi-weekly delivery cadence over 29 weeks
- **Grounding use case:** Provides the only confirmed longitudinal RCT comparing intervention types for occupant energy behavior. Quantitative parameters (6.3% reduction, 29-week horizon, bi-weekly cadence) can calibrate LLM agent responsiveness to smart home suggestions in demand-response scenarios. Informs what suggestion types to model: competence-building vs. social comparison.
- **Caveats:** Small N=65 in a COVID-era dormitory context; authors recommend caution. The 20.4% nudge-group increase may reflect seasonal confounds or reactance at small N.
- **Confidence:** Medium (2-1 adversarial vote)

---

## 4. Demographic-Conditioned Behavior Variation

### EIA Residential Energy Consumption Survey (RECS)
- **Source:** U.S. Energy Information Administration — https://www.eia.gov/consumption/residential/
- **Access:** Public; microdata available for download
- **Coverage:** Nationally representative U.S. residential energy consumption survey, conducted periodically (most recent: 2020)
- **What it captures:** Appliance ownership, heating/cooling equipment, building characteristics, household demographics (income, age, housing type, climate region), and self-reported energy use behaviors
- **Grounding use case:** Demographic conditioning for device ownership probability (does a household of a given type own a smart thermostat, dishwasher, EV charger?). Complements ATUS activity patterns with appliance-level household inventories.

---

## 5. Summary: Dataset-to-Use-Case Mapping

| Dataset | Dimension | Access | Primary Use Case |
|---|---|---|---|
| ATUS (BLS/IPUMS) | Time-use baseline | Free | Generate agent daily schedules conditioned on demographic profile |
| MTUS | Time-use baseline | Free (registration) | Cross-national/demographic conditioning; extends beyond U.S. |
| Dataport/Pecan Street | Device-state | Application required | Validate agent appliance on/off patterns against real households |
| DOE ecobee (2017) | Device-state | Public (DOE OSTI) | Calibrate and validate HVAC/thermostat agent behavior |
| CASAS (Aruba, Milan, etc.) | Device-state | Public (Zenodo) | HAR-style validation of agent room-level activity sequences |
| ecobee override study | Intervention response | Public (arXiv) | Calibrate agent override timing in demand-response scenarios |
| Boost vs. Nudge RCT | Intervention response | Published | Calibrate agent responsiveness to different suggestion types |
| EIA RECS | Demographics | Public (EIA) | Condition agent appliance ownership on household demographics |

---

## 6. Open Questions for Study Design

1. **Linked demographic-to-device dataset:** No publicly available dataset currently combines ATUS-style demographic conditioning with Dataport/ecobee-style device-state resolution in a single linked file. This linkage may need to be constructed synthetically.

2. **Quantifying the Homer-vs-ATUS gap:** No study has yet measured how much behavioral realism is lost by grounding LLM agents on Homer (21 participants) vs. ATUS (tens of thousands of respondents). Demonstrating this gap quantitatively would be a strong contribution.

3. **Device-level intervention response data:** The boost/nudge RCT measures aggregate electricity consumption, not per-device state changes. No confirmed longitudinal dataset with both intervention logging and appliance-level resolution was found. This is a validation gap that may require a controlled data collection component or a proxy approach.

4. **ATUS-to-agent transformation pipeline:** The key methodological question is how to convert population-level time-use diary statistics (distributions of activity start times and durations) into per-agent prior distributions over daily schedule and device interaction probabilities. This transformation is not yet standardized in the literature.

---

## 7. Caveats

- All time-use datasets (ATUS, MTUS) rely on self-reported 24-hour diaries, not sensor-observed occupancy. This introduces recall bias and may not capture exact device interaction timing.
- Dataport requires institutional or commercial access through Pecan Street Inc. and may not be freely available to all researchers without an agreement.
- MTUS activity category counts differ across release versions (41 original, 25 Simple File, 69 newer frames). Verify the specific release.
- The thermostat override finding (ecobee study) lacks reported confidence intervals and is based on a volunteer opt-in population, limiting generalizability.
- The boost/nudge RCT has N=65 in an unusual COVID-era dormitory context; the 20.4% nudge-group increase above baseline may reflect seasonal confounds rather than pure intervention effects.
- Several initially documented claims were not confirmed upon independent review, including: UK TUS 2014-15 being directly EnergyPlus-compatible, MTUS covering 100+ surveys and 30 countries, and SmartBench and DomusFM dataset statistics. These claims should not be cited without independent verification.

---

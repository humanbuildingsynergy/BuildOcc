"""
ATUS activity code → occupancy status + device state + room mapping.

Coverage: the tier-3 codes that carry a distinct occupancy, room, or device
consequence, plus tier-2 fallbacks for every tier-1 category. Any 6-digit code
not listed falls to its tier-2 fallback, then its tier-1 fallback.

ATUS code structure:
  Digits 1-2: Tier 1 (major category, e.g. 01 = Personal Care)
  Digits 3-4: Tier 2 (subcategory, e.g. 0101 = Sleeping)
  Digits 5-6: Tier 3 (specific activity, e.g. 010101 = Sleeping)

Codes and descriptions follow the **official ATUS 2023 Activity Lexicon** (BLS).
Earlier revisions of this file used a pre-2023 numbering in which, e.g., 0201
was Food & Drink Prep and 0202 was Housework; the 2023 lexicon transposes those
(0201 = Housework, 0202 = Food & Drink Prep), shifts the 1203 relaxing/leisure
block (120303 = Television, not 120301), and renumbers the 1301 sports list
(yoga = 130136, not 130116). This module now matches the 2023 lexicon exactly.

Key for building energy:
  occupancy: 'home' | 'away' | 'ambiguous'
    'home'      → occupant is in the building (HVAC holds comfort setpoint)
    'away'      → occupant has left (HVAC enters setback; all home devices idle)
    'ambiguous' → context-dependent (e.g., telecommuting, home gym vs. public gym)

  room: where the occupant most likely is when home
  devices_on / devices_off: triggered appliances at activity start
  energy_note: plain-language note for citable methodology section

Reference: ATUS 2023 Activity Lexicon (BLS)
  https://www.bls.gov/tus/lexiconnoex2023.pdf
"""

from __future__ import annotations

import csv
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Literal

Occupancy = Literal["home", "away", "ambiguous"]

# Bundled data directory (same resolution pattern as scheduler.py).
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"


@dataclass
class PersonaFlags:
    """
    Persona-level flags that resolve ambiguous ATUS occupancy codes.
    Set once per agent at initialization; stored in core memory.

    Decisions documented here are citable methodology choices (Phase 0.3):

    work_from_home:
      ATUS code 050101 ("Work, main job") does not record work location.
      Post-2022 ~25-30% of employed workers work from home ≥1 day/week
      (BLS, 2023). When True, all work-tier codes resolve to 'home';
      HVAC holds comfort setpoint during work hours instead of setback.
      Set from ATUS employment status + demographic stratum prior:
        O1 (employed single, 25-44): 30% WFH probability from population data.
        O2 (retired couple, 65+):    not applicable (not employed).

    home_gym:
      ATUS exercise codes (yoga, cardiovascular equipment, weightlifting, aerobics)
      do not record location. When True, these resolve to 'home'; when False,
      they resolve to 'away' (public gym assumed).
      Set from RECS appliance ownership priors (treadmill, stationary bike ownership
      rates by household type) or from persona demographic sampling.
    """

    work_from_home: bool = False
    home_gym: bool = False
    stratum: str | None = None   # enables per-stratum TEWHERE occupancy priors (Rule 4)


# Codes that flip home↔away depending on PersonaFlags.work_from_home (2023 lexicon:
# 0501 Working, 0502 Work-related, 0503 Other income-generating, 0504 Job search).
_WORK_CODES: frozenset[str] = frozenset({
    "050101", "050102", "050103", "050199",
    "050201", "050202", "050203", "050299",
    "050301", "050302", "050303", "050399",
    "050401", "050403", "050499",
})

_EATING_CODES: frozenset[str] = frozenset({"110101", "110199"})

# Exercise codes (2023 lexicon, tier 1301) that could be a home workout or a
# public gym — resolved by PersonaFlags.home_gym. NOTE: these are the 2023 codes
# (yoga = 130136, cardiovascular equipment = 130128, weightlifting = 130133),
# NOT the pre-2023 numbering used by earlier revisions of this file.
_HOME_GYM_RESOLVABLE: frozenset[str] = frozenset({
    "130101",  # Doing aerobics (home workout video vs. aerobics class)
    "130109",  # Dancing
    "130115",  # Doing gymnastics
    "130128",  # Using cardiovascular equipment (home treadmill vs. gym machine)
    "130133",  # Weightlifting / strength training
    "130134",  # Working out, unspecified
    "130136",  # Doing yoga
})


@dataclass
class ActivityMapping:
    atus_code: str
    description: str
    occupancy: Occupancy
    room: str = "any"                    # bedroom, kitchen, living_room, bathroom, garage, outside, any
    devices_on: list[str] = field(default_factory=list)
    devices_off: list[str] = field(default_factory=list)
    energy_note: str = ""


# ============================================================
# TIER-3 MAPPINGS  (full 6-digit codes)
# ============================================================

ACTIVITY_MAP: dict[str, ActivityMapping] = {

    # ------------------------------------------------------------------ #
    # Tier 1: 01 — Personal Care                                          #
    # ------------------------------------------------------------------ #

    # 0101 Sleeping
    "010101": ActivityMapping("010101", "Sleeping", "home", "bedroom",
                              devices_off=["tv", "computer", "dishwasher", "washer", "dryer"],
                              energy_note="Longest continuous HVAC setback trigger; ~8h daily"),
    "010102": ActivityMapping("010102", "Sleeplessness", "home", "bedroom"),
    "010199": ActivityMapping("010199", "Sleeping, n.e.c.", "home", "bedroom"),

    # 0102 Grooming
    "010201": ActivityMapping("010201", "Washing, dressing and grooming oneself", "home", "bathroom",
                              devices_on=["water_heater"],
                              energy_note="Water heater demand: ~10 min shower ≈ 0.5 kWh"),
    "010299": ActivityMapping("010299", "Grooming, n.e.c.", "home", "bathroom"),

    # 0103 Health-related self care
    "010301": ActivityMapping("010301", "Health-related self care", "home", "bathroom"),
    "010399": ActivityMapping("010399", "Self care, n.e.c.", "home", "any"),

    # 0104 Personal activities
    "010401": ActivityMapping("010401", "Personal/private activities", "home", "any"),
    "010499": ActivityMapping("010499", "Self care, n.e.c.", "home", "any"),

    # 0105 Personal care emergencies
    "010501": ActivityMapping("010501", "Personal emergencies", "ambiguous", "any"),
    "010599": ActivityMapping("010599", "Personal care emergencies, n.e.c.", "ambiguous", "any"),

    # ------------------------------------------------------------------ #
    # Tier 1: 02 — Household Activities                                   #
    # 2023 lexicon: 0201 Housework, 0202 Food & Drink Prep,               #
    #   0203 Interior Maint, 0204 Exterior Maint, 0205 Lawn & Garden,     #
    #   0206 Animals & Pets, 0207 Vehicles, 0208 Appliances/Tools/Toys,   #
    #   0209 Household Management.                                        #
    # ------------------------------------------------------------------ #

    # 0201 Housework
    "020101": ActivityMapping("020101", "Interior cleaning", "home", "living_room",
                              devices_on=["vacuum"],
                              energy_note="Vacuum ≈ 0.5–1.4 kW; short duration, low total energy"),
    "020102": ActivityMapping("020102", "Laundry", "home", "laundry_room",
                              devices_on=["washer"],
                              energy_note="Washer ≈ 0.5 kWh/cycle (cold); dryer ≈ 3–5 kWh — dryer starts ~45 min after washer; model as separate delayed trigger"),
    "020103": ActivityMapping("020103", "Sewing, repairing, and maintaining textiles", "home", "living_room"),
    "020104": ActivityMapping("020104", "Storing interior HH items, including food", "home", "kitchen"),
    "020199": ActivityMapping("020199", "Housework, n.e.c.", "home", "living_room"),

    # 0202 Food & Drink Preparation, Presentation, & Clean-up
    "020201": ActivityMapping("020201", "Food and drink preparation", "home", "kitchen",
                              devices_on=["oven", "range", "range_hood", "microwave"],
                              energy_note="Cooking is ~10% of residential energy; oven ≈ 2.3 kW, range ≈ 1.2 kW per burner"),
    "020202": ActivityMapping("020202", "Food presentation", "home", "kitchen",
                              devices_on=["microwave"]),
    "020203": ActivityMapping("020203", "Kitchen and food clean-up", "home", "kitchen",
                              devices_on=["dishwasher"],
                              energy_note="Dishwasher is a primary demand-response target; typical cycle 1.5 kWh"),
    "020299": ActivityMapping("020299", "Food & drink prep, presentation, & clean-up, n.e.c.", "home", "kitchen",
                              devices_on=["microwave"]),

    # 0203 Interior Maintenance, Repair, & Decoration
    "020301": ActivityMapping("020301", "Interior arrangement, decoration, and repairs", "home", "any"),
    "020302": ActivityMapping("020302", "Building and repairing furniture", "home", "garage"),
    "020303": ActivityMapping("020303", "Heating and cooling", "home", "any",
                              devices_on=["hvac"],
                              energy_note="Occupant actively adjusting HVAC — likely a manual override event; log separately"),
    "020399": ActivityMapping("020399", "Interior maintenance, repair, & decoration, n.e.c.", "home", "any"),

    # 0204 Exterior Maintenance, Repair, & Decoration
    "020401": ActivityMapping("020401", "Exterior cleaning", "home", "outside"),
    "020402": ActivityMapping("020402", "Exterior repair, improvements, and decoration", "home", "outside"),
    "020499": ActivityMapping("020499", "Exterior maintenance, repair & decoration, n.e.c.", "home", "outside"),

    # 0205 Lawn, Garden, and Houseplants
    "020501": ActivityMapping("020501", "Lawn, garden, and houseplant care", "home", "outside",
                              energy_note="Occupant is outside but home; HVAC holds setpoint"),
    "020502": ActivityMapping("020502", "Ponds, pools, and hot tubs", "home", "outside",
                              devices_on=["pool_pump"],
                              energy_note="Pool pump ≈ 1–2 kW continuous; major energy consumer in warm climates"),
    "020599": ActivityMapping("020599", "Lawn and garden, n.e.c.", "home", "outside"),

    # 0206 Animals and Pets
    "020601": ActivityMapping("020601", "Care for animals and pets (not veterinary care)", "home", "any"),
    "020602": ActivityMapping("020602", "Walking / exercising / playing with animals", "home", "outside"),
    "020699": ActivityMapping("020699", "Pet and animal care, n.e.c.", "home", "any"),

    # 0207 Vehicles
    "020701": ActivityMapping("020701", "Vehicle repair and maintenance (by self)", "home", "garage"),
    "020799": ActivityMapping("020799", "Vehicles, n.e.c.", "home", "garage"),

    # 0208 Appliances, Tools, and Toys
    "020801": ActivityMapping("020801", "Appliance, tool, and toy set-up, repair, & maintenance (by self)", "home", "garage"),
    "020899": ActivityMapping("020899", "Appliances and tools, n.e.c.", "home", "garage"),

    # 0209 Household Management
    "020901": ActivityMapping("020901", "Financial management", "home", "any",
                              devices_on=["computer"]),
    "020902": ActivityMapping("020902", "Household & personal organization and planning", "home", "any"),
    "020903": ActivityMapping("020903", "HH & personal mail & messages (except e-mail)", "home", "any"),
    "020904": ActivityMapping("020904", "HH & personal e-mail and messages", "home", "any",
                              devices_on=["computer"]),
    "020905": ActivityMapping("020905", "Home security", "home", "any"),
    "020999": ActivityMapping("020999", "Household management, n.e.c.", "home", "any"),

    # ------------------------------------------------------------------ #
    # Tier 1: 03 — Caring For & Helping Household (HH) Members            #
    # 2023 lexicon: 0301 Children, 0302 Children's Education,             #
    #   0303 Children's Health, 0304 Adults, 0305 Helping Adults.        #
    # ------------------------------------------------------------------ #

    # 0301 Caring for & helping HH children  (note: lexicon skips tier-3 07)
    "030101": ActivityMapping("030101", "Physical care for HH children", "home", "any"),
    "030102": ActivityMapping("030102", "Reading to/with HH children", "home", "living_room"),
    "030103": ActivityMapping("030103", "Playing with HH children, not sports", "home", "living_room"),
    "030104": ActivityMapping("030104", "Arts and crafts with HH children", "home", "living_room"),
    "030105": ActivityMapping("030105", "Playing sports with HH children", "home", "outside"),
    "030106": ActivityMapping("030106", "Talking with/listening to HH children", "home", "any"),
    "030108": ActivityMapping("030108", "Organization and planning for HH children", "home", "any"),
    "030109": ActivityMapping("030109", "Looking after HH children (as a primary activity)", "home", "living_room",
                              devices_on=["tv"],
                              energy_note="TV commonly on during passive childcare supervision"),
    "030110": ActivityMapping("030110", "Attending HH children's events", "away", "outside"),
    "030111": ActivityMapping("030111", "Waiting for/with HH children", "ambiguous", "any"),
    "030112": ActivityMapping("030112", "Picking up/dropping off HH children", "away", "outside"),
    "030199": ActivityMapping("030199", "Caring for & helping HH children, n.e.c.", "home", "any"),

    # 0302 Activities related to HH children's education
    "030201": ActivityMapping("030201", "Homework (HH children)", "home", "any"),
    "030202": ActivityMapping("030202", "Meetings and school conferences (HH children)", "away", "outside"),
    "030203": ActivityMapping("030203", "Home schooling of HH children", "home", "any"),
    "030204": ActivityMapping("030204", "Waiting associated with HH children's education", "ambiguous", "any"),
    "030299": ActivityMapping("030299", "Activities related to HH child's education, n.e.c.", "home", "any"),

    # 0303 Activities related to HH children's health
    "030301": ActivityMapping("030301", "Providing medical care to HH children", "home", "any"),
    "030302": ActivityMapping("030302", "Obtaining medical care for HH children", "away", "outside"),
    "030303": ActivityMapping("030303", "Waiting associated with HH children's health", "ambiguous", "any"),
    "030399": ActivityMapping("030399", "Activities related to HH child's health, n.e.c.", "home", "any"),

    # 0304 Caring for HH adults
    "030401": ActivityMapping("030401", "Physical care for HH adults", "home", "any"),
    "030402": ActivityMapping("030402", "Looking after HH adult (as a primary activity)", "home", "any"),
    "030403": ActivityMapping("030403", "Providing medical care to HH adult", "home", "any"),
    "030404": ActivityMapping("030404", "Obtaining medical and care services for HH adult", "ambiguous", "any"),
    "030405": ActivityMapping("030405", "Waiting associated with caring for HH adults", "ambiguous", "any"),
    "030499": ActivityMapping("030499", "Caring for HH adults, n.e.c.", "home", "any"),

    # 0305 Helping HH adults
    "030501": ActivityMapping("030501", "Helping HH adults", "home", "any"),
    "030502": ActivityMapping("030502", "Organization & planning for HH adults", "home", "any"),
    "030503": ActivityMapping("030503", "Picking up/dropping off HH adult", "away", "outside"),
    "030504": ActivityMapping("030504", "Waiting associated with helping HH adults", "ambiguous", "any"),
    "030599": ActivityMapping("030599", "Helping HH adults, n.e.c.", "home", "any"),

    # ------------------------------------------------------------------ #
    # Tier 1: 04 — Caring For & Helping Nonhousehold (NonHH) Members      #
    # Interpretation: nearly always requires leaving the home             #
    # ------------------------------------------------------------------ #

    # 0401 Caring for & helping nonHH children
    "040101": ActivityMapping("040101", "Physical care for nonHH children", "away", "outside"),
    "040102": ActivityMapping("040102", "Reading to/with nonHH children", "ambiguous", "any"),
    "040103": ActivityMapping("040103", "Playing with nonHH children, not sports", "away", "outside"),
    "040104": ActivityMapping("040104", "Arts and crafts with nonHH children", "away", "outside"),
    "040105": ActivityMapping("040105", "Playing sports with nonHH children", "away", "outside"),
    "040106": ActivityMapping("040106", "Talking with/listening to nonHH children", "ambiguous", "any"),
    "040108": ActivityMapping("040108", "Organization & planning for nonHH children", "ambiguous", "any"),
    "040109": ActivityMapping("040109", "Looking after nonHH children (as primary activity)", "away", "outside"),
    "040110": ActivityMapping("040110", "Attending nonHH children's events", "away", "outside"),
    "040111": ActivityMapping("040111", "Waiting for/with nonHH children", "ambiguous", "any"),
    "040112": ActivityMapping("040112", "Dropping off/picking up nonHH children", "away", "outside"),
    "040199": ActivityMapping("040199", "Caring for and helping nonHH children, n.e.c.", "away", "outside"),

    # 0402 NonHH children's education / 0403 health (mostly away)
    "040201": ActivityMapping("040201", "Homework (nonHH children)", "ambiguous", "any"),
    "040203": ActivityMapping("040203", "Home schooling of nonHH children", "ambiguous", "any"),
    "040299": ActivityMapping("040299", "Activities related to nonHH child's educ., n.e.c.", "away", "outside"),
    "040301": ActivityMapping("040301", "Providing medical care to nonHH children", "away", "outside"),
    "040399": ActivityMapping("040399", "Activities related to nonHH child's health, n.e.c.", "away", "outside"),

    # 0404 Caring for nonHH adults
    "040401": ActivityMapping("040401", "Physical care for nonHH adults", "away", "outside"),
    "040402": ActivityMapping("040402", "Looking after nonHH adult (as a primary activity)", "away", "outside"),
    "040403": ActivityMapping("040403", "Providing medical care to nonHH adult", "away", "outside"),
    "040499": ActivityMapping("040499", "Caring for nonHH adults, n.e.c.", "away", "outside"),

    # 0405 Helping nonHH adults
    "040501": ActivityMapping("040501", "Housework, cooking, & shopping assistance for nonHH adults", "away", "outside"),
    "040502": ActivityMapping("040502", "House & lawn maintenance & repair assistance for nonHH adults", "away", "outside"),
    "040507": ActivityMapping("040507", "Picking up/dropping off nonHH adult", "away", "outside"),
    "040599": ActivityMapping("040599", "Helping nonHH adults, n.e.c.", "away", "outside"),

    # ------------------------------------------------------------------ #
    # Tier 1: 05 — Work & Work-Related Activities                         #
    # ATUS does not distinguish WFH vs. office — treat as 'ambiguous'     #
    # and resolve via PersonaFlags.work_from_home (Rule 1).               #
    # ------------------------------------------------------------------ #

    # 0501 Working
    "050101": ActivityMapping("050101", "Work, main job", "ambiguous", "outside",
                              energy_note="Resolution rule: if PersonaFlags.work_from_home → 'home' "
                                          "(HVAC holds comfort setpoint); else → 'away' (HVAC setback). "
                                          "Post-2022 WFH prevalence ~25-30% for employed adults (BLS, 2023)."),
    "050102": ActivityMapping("050102", "Work, other job(s)", "ambiguous", "outside"),
    "050103": ActivityMapping("050103", "Security procedures related to work", "away", "outside"),
    "050199": ActivityMapping("050199", "Working, n.e.c.", "ambiguous", "outside"),

    # 0502 Work-related activities
    "050201": ActivityMapping("050201", "Socializing, relaxing, and leisure as part of job", "away", "outside"),
    "050202": ActivityMapping("050202", "Eating and drinking as part of job", "away", "outside"),
    "050203": ActivityMapping("050203", "Sports and exercise as part of job", "away", "outside"),
    "050299": ActivityMapping("050299", "Work-related activities, n.e.c.", "away", "outside"),

    # 0503 Other income-generating activities (often at home)
    "050301": ActivityMapping("050301", "Income-generating hobbies, crafts, and food", "ambiguous", "any"),
    "050302": ActivityMapping("050302", "Income-generating performances", "away", "outside"),
    "050399": ActivityMapping("050399", "Other income-generating activities, n.e.c.", "ambiguous", "any"),

    # 0504 Job search and interviewing
    "050401": ActivityMapping("050401", "Job search activities", "ambiguous", "any",
                              devices_on=["computer"]),
    "050403": ActivityMapping("050403", "Job interviewing", "away", "outside"),
    "050499": ActivityMapping("050499", "Job search and interviewing, n.e.c.", "ambiguous", "any"),

    # ------------------------------------------------------------------ #
    # Tier 1: 06 — Education                                              #
    # Interpretation: class = away; homework = home; ambiguous by default #
    # ------------------------------------------------------------------ #

    # 0601 Taking class
    "060101": ActivityMapping("060101", "Taking class for degree, certification, or licensure", "away", "outside"),
    "060102": ActivityMapping("060102", "Taking class for personal interest", "away", "outside"),
    "060103": ActivityMapping("060103", "Waiting associated with taking classes", "ambiguous", "any"),
    "060199": ActivityMapping("060199", "Taking class, n.e.c.", "away", "outside"),
    # 0602 Extracurricular school activities
    "060201": ActivityMapping("060201", "Extracurricular club activities", "away", "outside"),
    "060202": ActivityMapping("060202", "Extracurricular music & performance activities", "away", "outside"),
    "060203": ActivityMapping("060203", "Extracurricular student government activities", "away", "outside"),
    "060299": ActivityMapping("060299", "Education-related extracurricular activities, n.e.c.", "ambiguous", "any"),
    # 0603 Research/homework
    "060301": ActivityMapping("060301", "Research/homework for class for degree, certification, or licensure", "home", "any",
                              devices_on=["computer"],
                              energy_note="Computer on; lighting likely on — concentrated load block"),
    "060302": ActivityMapping("060302", "Research/homework for class for personal interest", "home", "any",
                              devices_on=["computer"]),
    "060399": ActivityMapping("060399", "Research/homework, n.e.c.", "home", "any",
                              devices_on=["computer"]),
    # 0604 Registration/administrative
    "060401": ActivityMapping("060401", "Administrative activities: class for degree, certification, or licensure", "ambiguous", "any"),
    "060499": ActivityMapping("060499", "Administrative for education, n.e.c.", "ambiguous", "any"),

    # ------------------------------------------------------------------ #
    # Tier 1: 07 — Consumer Purchases (shopping)                          #
    # Interpretation: nearly always away from home                        #
    # ------------------------------------------------------------------ #

    # 0701 Shopping (store, telephone, internet)
    "070101": ActivityMapping("070101", "Grocery shopping", "away", "outside"),
    "070102": ActivityMapping("070102", "Purchasing gas", "away", "outside"),
    "070103": ActivityMapping("070103", "Purchasing food (not groceries)", "away", "outside"),
    "070104": ActivityMapping("070104", "Shopping, except groceries, food and gas", "away", "outside"),
    "070105": ActivityMapping("070105", "Waiting associated with shopping", "ambiguous", "any"),
    "070199": ActivityMapping("070199", "Shopping, n.e.c.", "away", "outside"),
    # 0702 Researching purchases
    "070201": ActivityMapping("070201", "Comparison shopping", "ambiguous", "any",
                              devices_on=["computer"],
                              energy_note="Online comparison shopping → home + computer; in-store → away"),
    "070299": ActivityMapping("070299", "Researching purchases, n.e.c.", "ambiguous", "any"),

    # ------------------------------------------------------------------ #
    # Tier 1: 08 — Professional & Personal Care Services                  #
    # Interpretation: visiting service providers = away                   #
    # ------------------------------------------------------------------ #

    "080101": ActivityMapping("080101", "Using paid childcare services", "away", "outside"),
    "080199": ActivityMapping("080199", "Using paid childcare services, n.e.c.", "away", "outside"),
    "080201": ActivityMapping("080201", "Banking", "away", "outside"),
    "080202": ActivityMapping("080202", "Using other financial services", "away", "outside"),
    "080299": ActivityMapping("080299", "Using financial services and banking, n.e.c.", "away", "outside"),
    "080301": ActivityMapping("080301", "Using legal services", "away", "outside"),
    "080399": ActivityMapping("080399", "Using legal services, n.e.c.", "away", "outside"),
    "080401": ActivityMapping("080401", "Using health and care services outside the home", "away", "outside"),
    "080402": ActivityMapping("080402", "Using in-home health and care services", "home", "any",
                              energy_note="Care provider comes to the occupant; occupant is home, HVAC at comfort"),
    "080499": ActivityMapping("080499", "Using medical services, n.e.c.", "away", "outside"),
    "080501": ActivityMapping("080501", "Using personal care services", "away", "outside"),
    "080599": ActivityMapping("080599", "Using personal care services, n.e.c.", "away", "outside"),
    "080601": ActivityMapping("080601", "Activities related to purchasing/selling real estate", "away", "outside"),
    "080699": ActivityMapping("080699", "Using real estate services, n.e.c.", "away", "outside"),
    "080701": ActivityMapping("080701", "Using veterinary services", "away", "outside"),
    "080799": ActivityMapping("080799", "Using veterinary services, n.e.c.", "away", "outside"),

    # ------------------------------------------------------------------ #
    # Tier 1: 09 — Household Services                                     #
    # Interpretation: occupant is HOME waiting for/supervising provider   #
    # (except vehicle service, where the occupant goes to the shop)       #
    # ------------------------------------------------------------------ #

    "090101": ActivityMapping("090101", "Using interior cleaning services", "home", "any",
                              energy_note="Occupant home; HVAC at comfort. Service provider may open doors"),
    "090102": ActivityMapping("090102", "Using meal preparation services", "home", "kitchen"),
    "090103": ActivityMapping("090103", "Using clothing repair and cleaning services", "ambiguous", "any"),
    "090199": ActivityMapping("090199", "Using household services, n.e.c.", "home", "any"),
    "090201": ActivityMapping("090201", "Using home maintenance/repair/décor/construction services", "home", "any"),
    "090299": ActivityMapping("090299", "Using home maint/repair/décor/constr services, n.e.c.", "home", "any"),
    "090301": ActivityMapping("090301", "Using pet services", "ambiguous", "any"),
    "090401": ActivityMapping("090401", "Using lawn and garden services", "home", "any"),
    "090501": ActivityMapping("090501", "Using vehicle maintenance or repair services", "away", "outside"),
    "090599": ActivityMapping("090599", "Using vehicle maint. & repair svcs, n.e.c.", "away", "outside"),

    # ------------------------------------------------------------------ #
    # Tier 1: 10 — Government Services & Civic Obligations                #
    # Interpretation: away (at government office, polling place, etc.)    #
    # ------------------------------------------------------------------ #

    "100101": ActivityMapping("100101", "Using police and fire services", "away", "outside"),
    "100102": ActivityMapping("100102", "Using social services", "away", "outside"),
    "100103": ActivityMapping("100103", "Obtaining licenses & paying fines, fees, taxes", "away", "outside"),
    "100199": ActivityMapping("100199", "Using government services, n.e.c.", "away", "outside"),
    "100201": ActivityMapping("100201", "Civic obligations & participation", "away", "outside"),
    "100299": ActivityMapping("100299", "Civic obligations & participation, n.e.c.", "away", "outside"),

    # ------------------------------------------------------------------ #
    # Tier 1: 11 — Eating & Drinking                                      #
    # ATUS does not distinguish location — resolve via prior travel code  #
    # ------------------------------------------------------------------ #

    "110101": ActivityMapping("110101", "Eating and drinking", "ambiguous", "kitchen",
                              energy_note="Resolution rule: if previous ATUS code is a travel episode "
                                          "(tier-1 = '18') → 'away' (eating out); else → 'home'. "
                                          "~60% of US meals eaten at home (USDA ERS, 2023). "
                                          "No device triggered — food was prepared earlier or is cold."),
    "110199": ActivityMapping("110199", "Eating and drinking, n.e.c.", "ambiguous", "kitchen",
                              energy_note="Same resolution rule as 110101: prior travel code → away."),

    # ------------------------------------------------------------------ #
    # Tier 1: 12 — Socializing, Relaxing & Leisure                        #
    # 2023 lexicon: 1201 Socializing, 1202 Attending/Hosting Social       #
    #   Events, 1203 Relaxing & Leisure, 1204 Arts & Entertainment.      #
    # ------------------------------------------------------------------ #

    # 1201 Socializing and communicating
    "120101": ActivityMapping("120101", "Socializing and communicating with others", "ambiguous", "living_room",
                              energy_note="If at home (host): HVAC comfort + lighting. If away (guest): setback."),
    "120199": ActivityMapping("120199", "Socializing and communicating, n.e.c.", "ambiguous", "living_room"),

    # 1202 Attending or hosting social events
    "120201": ActivityMapping("120201", "Attending or hosting parties/receptions/ceremonies", "ambiguous", "living_room",
                              devices_on=["tv", "lighting_living"]),
    "120202": ActivityMapping("120202", "Attending meetings for personal interest (not volunteering)", "away", "outside"),
    "120299": ActivityMapping("120299", "Attending/hosting social events, n.e.c.", "ambiguous", "living_room"),

    # 1203 Relaxing and leisure
    "120301": ActivityMapping("120301", "Relaxing, thinking", "home", "living_room"),
    "120302": ActivityMapping("120302", "Tobacco and drug use", "home", "any"),
    "120303": ActivityMapping("120303", "Television and movies (not religious)", "home", "living_room",
                              devices_on=["tv"],
                              energy_note="#1 leisure activity by duration (~2.8 h/day avg); drives evening peak. "
                                          "TV ≈ 0.1–0.4 kW depending on size/type"),
    "120304": ActivityMapping("120304", "Television (religious)", "home", "living_room",
                              devices_on=["tv"]),
    "120305": ActivityMapping("120305", "Listening to the radio", "home", "any",
                              devices_on=["radio"]),
    "120306": ActivityMapping("120306", "Listening to/playing music (not radio)", "home", "any",
                              devices_on=["stereo"]),
    "120307": ActivityMapping("120307", "Playing games", "home", "living_room",
                              devices_on=["tv", "gaming_console"],
                              energy_note="Video-game play under this code drives a gaming console ≈ 0.1–0.25 kW; can rival TV energy"),
    "120308": ActivityMapping("120308", "Computer use for leisure (except games)", "home", "living_room",
                              devices_on=["computer"],
                              energy_note="Laptop ≈ 0.05 kW; desktop ≈ 0.15–0.3 kW; monitor ≈ 0.025 kW"),
    "120309": ActivityMapping("120309", "Arts and crafts as a hobby", "home", "living_room"),
    "120310": ActivityMapping("120310", "Collecting as a hobby", "home", "any"),
    "120311": ActivityMapping("120311", "Hobbies, except arts & crafts and collecting", "home", "any"),
    "120312": ActivityMapping("120312", "Reading for personal interest", "home", "living_room"),
    "120313": ActivityMapping("120313", "Writing for personal interest", "home", "any"),
    "120399": ActivityMapping("120399", "Relaxing and leisure, n.e.c.", "home", "living_room"),

    # 1204 Arts and entertainment (other than sports)
    "120401": ActivityMapping("120401", "Attending performing arts", "away", "outside"),
    "120402": ActivityMapping("120402", "Attending museums", "away", "outside"),
    "120403": ActivityMapping("120403", "Attending movies/film", "away", "outside"),
    "120404": ActivityMapping("120404", "Attending gambling establishments", "away", "outside"),
    "120499": ActivityMapping("120499", "Arts and entertainment, n.e.c.", "away", "outside"),

    # ------------------------------------------------------------------ #
    # Tier 1: 13 — Sports, Exercise & Recreation                          #
    # 2023 lexicon numbering (1301 Participating). Outdoor sports = away;  #
    # home-resolvable exercise (aerobics/gymnastics/cardio/weights/yoga/   #
    # working-out/dancing) → PersonaFlags.home_gym (Rule 3).              #
    # ------------------------------------------------------------------ #

    "130101": ActivityMapping("130101", "Doing aerobics", "ambiguous", "any",
                              energy_note="Home aerobics (video) = home + TV/computer; gym = away (PersonaFlags.home_gym)"),
    "130102": ActivityMapping("130102", "Playing baseball", "away", "outside"),
    "130103": ActivityMapping("130103", "Playing basketball", "away", "outside"),
    "130104": ActivityMapping("130104", "Biking", "away", "outside"),
    "130105": ActivityMapping("130105", "Playing billiards", "ambiguous", "any"),
    "130106": ActivityMapping("130106", "Boating", "away", "outside"),
    "130107": ActivityMapping("130107", "Bowling", "away", "outside"),
    "130108": ActivityMapping("130108", "Climbing, spelunking, caving", "away", "outside"),
    "130109": ActivityMapping("130109", "Dancing", "ambiguous", "any"),
    "130110": ActivityMapping("130110", "Participating in equestrian sports", "away", "outside"),
    "130111": ActivityMapping("130111", "Fencing", "away", "outside"),
    "130112": ActivityMapping("130112", "Fishing", "away", "outside"),
    "130113": ActivityMapping("130113", "Playing football", "away", "outside"),
    "130114": ActivityMapping("130114", "Golfing", "away", "outside"),
    "130115": ActivityMapping("130115", "Doing gymnastics", "ambiguous", "any"),
    "130116": ActivityMapping("130116", "Hiking", "away", "outside"),
    "130117": ActivityMapping("130117", "Playing hockey", "away", "outside"),
    "130118": ActivityMapping("130118", "Hunting", "away", "outside"),
    "130119": ActivityMapping("130119", "Participating in martial arts", "away", "outside"),
    "130120": ActivityMapping("130120", "Playing racquet sports", "away", "outside"),
    "130121": ActivityMapping("130121", "Participating in rodeo competitions", "away", "outside"),
    "130122": ActivityMapping("130122", "Rollerblading", "away", "outside"),
    "130123": ActivityMapping("130123", "Playing rugby", "away", "outside"),
    "130124": ActivityMapping("130124", "Running", "away", "outside"),
    "130125": ActivityMapping("130125", "Skiing, ice skating, snowboarding", "away", "outside"),
    "130126": ActivityMapping("130126", "Playing soccer", "away", "outside"),
    "130127": ActivityMapping("130127", "Softball", "away", "outside"),
    "130128": ActivityMapping("130128", "Using cardiovascular equipment", "ambiguous", "any",
                              energy_note="Resolution rule: if PersonaFlags.home_gym → 'home'; else → 'away'. "
                                          "RECS 2020: ~12% of US households own a treadmill; higher for "
                                          "higher-income strata. Sample from RECS ownership rates by stratum."),
    "130129": ActivityMapping("130129", "Vehicle touring/racing", "away", "outside"),
    "130130": ActivityMapping("130130", "Playing volleyball", "away", "outside"),
    "130131": ActivityMapping("130131", "Walking", "away", "outside",
                              energy_note="Occupant away from home even if brief; HVAC setback if walk > 30 min"),
    "130132": ActivityMapping("130132", "Participating in water sports", "away", "outside"),
    "130133": ActivityMapping("130133", "Weightlifting/strength training", "ambiguous", "any",
                              energy_note="Same resolution rule as 130128: PersonaFlags.home_gym."),
    "130134": ActivityMapping("130134", "Working out, unspecified", "ambiguous", "any",
                              energy_note="Same resolution rule as 130128: PersonaFlags.home_gym."),
    "130135": ActivityMapping("130135", "Wrestling", "away", "outside"),
    "130136": ActivityMapping("130136", "Doing yoga", "ambiguous", "any",
                              energy_note="Same resolution rule as 130128: PersonaFlags.home_gym."),
    "130199": ActivityMapping("130199", "Playing sports, n.e.c.", "ambiguous", "outside"),
    # 1302 Attending sporting/recreational events (in person) — away.
    # The lexicon enumerates 130201–130232 per sport (130201 = watching aerobics);
    # 130299 is the catch-all for any watched sport/recreation event.
    "130201": ActivityMapping("130201", "Watching aerobics", "away", "outside"),
    "130299": ActivityMapping("130299", "Attending sporting events, n.e.c.", "away", "outside"),
    # 1303 Waiting associated with sports
    "130301": ActivityMapping("130301", "Waiting related to playing sports or exercising", "ambiguous", "outside"),
    "130302": ActivityMapping("130302", "Waiting related to attending sporting events", "ambiguous", "outside"),
    "130399": ActivityMapping("130399", "Waiting associated with sports, exercise, & recreation, n.e.c.", "ambiguous", "outside"),

    # ------------------------------------------------------------------ #
    # Tier 1: 14 — Religious & Spiritual Activities                        #
    # Interpretation: typically away (place of worship); practice = home  #
    # ------------------------------------------------------------------ #

    "140101": ActivityMapping("140101", "Attending religious services", "away", "outside"),
    "140102": ActivityMapping("140102", "Participation in religious practices", "home", "any"),
    "140103": ActivityMapping("140103", "Waiting associated w/religious & spiritual activities", "ambiguous", "any"),
    "140105": ActivityMapping("140105", "Religious education activities", "away", "outside"),
    "149999": ActivityMapping("149999", "Religious and spiritual activities, n.e.c.", "ambiguous", "any"),

    # ------------------------------------------------------------------ #
    # Tier 1: 15 — Volunteer Activities                                    #
    # Interpretation: typically away (serving others at their location)   #
    # ------------------------------------------------------------------ #

    "150101": ActivityMapping("150101", "Computer use (volunteer)", "away", "outside"),
    "150102": ActivityMapping("150102", "Organizing and preparing (volunteer)", "away", "outside"),
    "150103": ActivityMapping("150103", "Reading (volunteer)", "away", "outside"),
    "150104": ActivityMapping("150104", "Telephone calls (volunteer)", "away", "outside"),
    "150105": ActivityMapping("150105", "Writing (volunteer)", "away", "outside"),
    "150106": ActivityMapping("150106", "Fundraising (volunteer)", "away", "outside"),
    "150199": ActivityMapping("150199", "Administrative & support activities (volunteer), n.e.c.", "away", "outside"),
    "150201": ActivityMapping("150201", "Food preparation, presentation, clean-up (volunteer)", "away", "outside"),
    "150202": ActivityMapping("150202", "Collecting & delivering clothing & other goods (volunteer)", "away", "outside"),
    "150203": ActivityMapping("150203", "Providing care (volunteer)", "away", "outside"),
    "150204": ActivityMapping("150204", "Teaching, leading, counseling, mentoring (volunteer)", "away", "outside"),
    "150301": ActivityMapping("150301", "Building houses, wildlife sites, & other structures (volunteer)", "away", "outside"),
    "150302": ActivityMapping("150302", "Indoor & outdoor maintenance, repair, & clean-up (volunteer)", "away", "outside"),
    "150401": ActivityMapping("150401", "Performing (volunteer)", "away", "outside"),
    "150501": ActivityMapping("150501", "Attending meetings, conferences, & training (volunteer)", "away", "outside"),

    # ------------------------------------------------------------------ #
    # Tier 1: 16 — Telephone Calls                                         #
    # Interpretation: occupant is home unless accompanied by travel code   #
    # ------------------------------------------------------------------ #

    "160101": ActivityMapping("160101", "Telephone calls to/from family members", "home", "any"),
    "160102": ActivityMapping("160102", "Telephone calls to/from friends, neighbors, or acquaintances", "home", "any"),
    "160103": ActivityMapping("160103", "Telephone calls to/from education services providers", "home", "any"),
    "160104": ActivityMapping("160104", "Telephone calls to/from salespeople", "home", "any"),
    "160105": ActivityMapping("160105", "Telephone calls to/from professional or personal care svcs providers", "home", "any"),
    "160106": ActivityMapping("160106", "Telephone calls to/from household services providers", "home", "any"),
    "160107": ActivityMapping("160107", "Telephone calls to/from paid child or adult care providers", "home", "any"),
    "160108": ActivityMapping("160108", "Telephone calls to/from government officials", "home", "any"),
    "160199": ActivityMapping("160199", "Telephone calls (to or from), n.e.c.", "home", "any"),

    # ------------------------------------------------------------------ #
    # Tier 1: 18 — Traveling                                               #
    # ALL travel codes = occupant is away from home                       #
    # HVAC should enter setback immediately when a travel code begins     #
    # ------------------------------------------------------------------ #

    "180101": ActivityMapping("180101", "Travel related to personal care", "away", "outside"),
    "180199": ActivityMapping("180199", "Travel related to personal care, n.e.c.", "away", "outside"),
    "180201": ActivityMapping("180201", "Travel related to housework", "away", "outside"),
    "180202": ActivityMapping("180202", "Travel related to food & drink prep., clean-up, & presentation", "away", "outside"),
    "180203": ActivityMapping("180203", "Travel related to interior maintenance, repair, & decoration", "away", "outside"),
    "180204": ActivityMapping("180204", "Travel related to exterior maintenance, repair, & decoration", "away", "outside"),
    "180205": ActivityMapping("180205", "Travel related to lawn, garden, and houseplant care", "away", "outside"),
    "180206": ActivityMapping("180206", "Travel related to care for animals and pets (not vet care)", "away", "outside"),
    "180207": ActivityMapping("180207", "Travel related to vehicle care & maintenance (by self)", "away", "outside"),
    "180208": ActivityMapping("180208", "Travel related to appliance, tool, and toy set-up/repair/maintenance", "away", "outside"),
    "180209": ActivityMapping("180209", "Travel related to household management", "away", "outside"),
    "180299": ActivityMapping("180299", "Travel related to household activities, n.e.c.", "away", "outside"),
    "180301": ActivityMapping("180301", "Travel related to caring for & helping HH children", "away", "outside"),
    "180302": ActivityMapping("180302", "Travel related to HH children's education", "away", "outside"),
    "180399": ActivityMapping("180399", "Travel rel. to caring for & helping HH members, n.e.c.", "away", "outside"),
    "180401": ActivityMapping("180401", "Travel related to caring for and helping nonHH children", "away", "outside"),
    "180499": ActivityMapping("180499", "Travel rel. to caring for & helping nonHH members, n.e.c.", "away", "outside"),
    "180501": ActivityMapping("180501", "Travel related to working", "away", "outside",
                              energy_note="Commute = occupant away; HVAC setback. Duration distributes "
                                          "by ATUS commute time distributions. This is the single largest "
                                          "away-from-home period for employed profiles (O1)."),
    "180502": ActivityMapping("180502", "Travel related to work-related activities", "away", "outside"),
    "180601": ActivityMapping("180601", "Travel related to taking class", "away", "outside"),
    "180701": ActivityMapping("180701", "Travel related to grocery shopping", "away", "outside"),
    "180801": ActivityMapping("180801", "Travel related to using childcare services", "away", "outside"),
    "180901": ActivityMapping("180901", "Travel related to using household services", "away", "outside"),
    "181001": ActivityMapping("181001", "Travel related to using government services", "away", "outside"),
    "181101": ActivityMapping("181101", "Travel related to eating and drinking", "away", "outside"),
    "181201": ActivityMapping("181201", "Travel related to socializing and communicating", "away", "outside"),
    "181301": ActivityMapping("181301", "Travel related to participating in sports/exercise/recreation", "away", "outside"),
    "181401": ActivityMapping("181401", "Travel related to religious/spiritual practices", "away", "outside"),
    "181501": ActivityMapping("181501", "Travel related to volunteering", "away", "outside"),
    "181601": ActivityMapping("181601", "Travel related to phone calls", "away", "outside"),
    "189999": ActivityMapping("189999", "Traveling, n.e.c.", "away", "outside"),

    # ------------------------------------------------------------------ #
    # Data codes / unable to code                                          #
    # ------------------------------------------------------------------ #
    "500101": ActivityMapping("500101", "Insufficient detail in verbatim", "ambiguous", "any"),
    "500103": ActivityMapping("500103", "Missing travel or destination", "ambiguous", "any"),
    "500105": ActivityMapping("500105", "Respondent refused to provide information", "ambiguous", "any"),
    "500106": ActivityMapping("500106", "Gap/can't remember", "ambiguous", "any"),
    "500107": ActivityMapping("500107", "Unable to code activity at 1st tier", "ambiguous", "any"),
}


# ============================================================
# TIER-2 FALLBACKS (4-digit prefix)
# Used when the full 6-digit code is not in ACTIVITY_MAP.
# ============================================================

TIER2_FALLBACK: dict[str, ActivityMapping] = {
    # Personal care
    "0101": ActivityMapping("0101", "Sleeping (any)", "home", "bedroom",
                            devices_off=["tv", "dishwasher", "washer", "dryer"]),
    "0102": ActivityMapping("0102", "Grooming (any)", "home", "bathroom",
                            devices_on=["water_heater"]),
    "0103": ActivityMapping("0103", "Health self-care (any)", "home", "bathroom"),
    "0104": ActivityMapping("0104", "Personal activities (any)", "home", "any"),
    # Household  (2023 numbering)
    "0201": ActivityMapping("0201", "Housework (any)", "home", "living_room"),
    "0202": ActivityMapping("0202", "Food prep (any)", "home", "kitchen",
                            devices_on=["microwave"]),
    "0203": ActivityMapping("0203", "Interior maintenance (any)", "home", "any"),
    "0204": ActivityMapping("0204", "Exterior maintenance (any)", "home", "outside"),
    "0205": ActivityMapping("0205", "Lawn and garden (any)", "home", "outside"),
    "0206": ActivityMapping("0206", "Animals and pets (any)", "home", "any"),
    "0207": ActivityMapping("0207", "Vehicles (any)", "home", "garage"),
    "0208": ActivityMapping("0208", "Appliances/tools (any)", "home", "garage"),
    "0209": ActivityMapping("0209", "Household management (any)", "home", "any"),
    # Caring HH
    "0301": ActivityMapping("0301", "HH childcare (any)", "home", "any"),
    "0302": ActivityMapping("0302", "HH children's education (any)", "home", "any"),
    "0303": ActivityMapping("0303", "HH children's health (any)", "home", "any"),
    "0304": ActivityMapping("0304", "HH adult care (any)", "home", "any"),
    "0305": ActivityMapping("0305", "Helping HH adults (any)", "home", "any"),
    # Caring non-HH
    "0401": ActivityMapping("0401", "Non-HH childcare (any)", "away", "outside"),
    "0402": ActivityMapping("0402", "Non-HH children's education (any)", "ambiguous", "any"),
    "0403": ActivityMapping("0403", "Non-HH children's health (any)", "away", "outside"),
    "0404": ActivityMapping("0404", "Non-HH adult care (any)", "away", "outside"),
    "0405": ActivityMapping("0405", "Helping non-HH adults (any)", "away", "outside"),
    # Work
    "0501": ActivityMapping("0501", "Working (any)", "ambiguous", "outside"),
    "0502": ActivityMapping("0502", "Work-related (any)", "away", "outside"),
    "0503": ActivityMapping("0503", "Other income-generating (any)", "ambiguous", "any"),
    "0504": ActivityMapping("0504", "Job search and interviewing (any)", "ambiguous", "any"),
    # Education
    "0601": ActivityMapping("0601", "Taking class (any)", "away", "outside"),
    "0602": ActivityMapping("0602", "Extracurricular education (any)", "ambiguous", "outside"),
    "0603": ActivityMapping("0603", "Research/homework (any)", "home", "any",
                            devices_on=["computer"]),
    "0604": ActivityMapping("0604", "Education administration (any)", "ambiguous", "any"),
    # Shopping / services
    "0701": ActivityMapping("0701", "Shopping (any)", "away", "outside"),
    "0702": ActivityMapping("0702", "Researching purchases (any)", "ambiguous", "any"),
    "0801": ActivityMapping("0801", "Childcare services (any)", "away", "outside"),
    "0802": ActivityMapping("0802", "Financial services (any)", "away", "outside"),
    "0803": ActivityMapping("0803", "Legal services (any)", "away", "outside"),
    "0804": ActivityMapping("0804", "Medical & care services (any)", "away", "outside"),
    "0805": ActivityMapping("0805", "Personal care services (any)", "away", "outside"),
    "0806": ActivityMapping("0806", "Real estate services (any)", "away", "outside"),
    "0807": ActivityMapping("0807", "Veterinary services (any)", "away", "outside"),
    "0901": ActivityMapping("0901", "Household services received (any)", "home", "any"),
    "0902": ActivityMapping("0902", "Home maint/repair services received (any)", "home", "any"),
    "0903": ActivityMapping("0903", "Pet services (any)", "ambiguous", "any"),
    "0904": ActivityMapping("0904", "Lawn & garden services received (any)", "home", "any"),
    "0905": ActivityMapping("0905", "Vehicle maint/repair services (any)", "away", "outside"),
    "1001": ActivityMapping("1001", "Government services (any)", "away", "outside"),
    "1002": ActivityMapping("1002", "Civic obligations & participation (any)", "away", "outside"),
    # Eating
    "1101": ActivityMapping("1101", "Eating and drinking (any)", "ambiguous", "kitchen"),
    # Leisure
    "1201": ActivityMapping("1201", "Socializing (any)", "ambiguous", "living_room"),
    "1202": ActivityMapping("1202", "Attending/hosting social events (any)", "ambiguous", "living_room"),
    "1203": ActivityMapping("1203", "Relaxing and leisure (any)", "home", "living_room"),
    "1204": ActivityMapping("1204", "Arts and entertainment (any)", "away", "outside"),
    # Sports / exercise
    "1301": ActivityMapping("1301", "Participating in sports/exercise (any)", "ambiguous", "outside"),
    "1302": ActivityMapping("1302", "Attending sporting events (any)", "away", "outside"),
    "1303": ActivityMapping("1303", "Waiting associated with sports (any)", "ambiguous", "outside"),
    # Religious / volunteer
    "1401": ActivityMapping("1401", "Religious/spiritual activities (any)", "ambiguous", "any"),
    "1501": ActivityMapping("1501", "Volunteer admin & support (any)", "away", "outside"),
    "1502": ActivityMapping("1502", "Volunteer social service & care (any)", "away", "outside"),
    "1503": ActivityMapping("1503", "Volunteer maintenance/building (any)", "away", "outside"),
    "1504": ActivityMapping("1504", "Volunteer performance/cultural (any)", "away", "outside"),
    "1505": ActivityMapping("1505", "Volunteer meetings/training (any)", "away", "outside"),
    # Telephone
    "1601": ActivityMapping("1601", "Telephone calls (any)", "home", "any"),
    # Travel (all tier-2 groups → away)
    "1801": ActivityMapping("1801", "Travel: personal care (any)", "away", "outside"),
    "1802": ActivityMapping("1802", "Travel: household activities (any)", "away", "outside"),
    "1803": ActivityMapping("1803", "Travel: caring for HH members (any)", "away", "outside"),
    "1804": ActivityMapping("1804", "Travel: caring for non-HH members (any)", "away", "outside"),
    "1805": ActivityMapping("1805", "Travel: work (any)", "away", "outside"),
    "1806": ActivityMapping("1806", "Travel: education (any)", "away", "outside"),
    "1807": ActivityMapping("1807", "Travel: consumer purchases (any)", "away", "outside"),
    "1808": ActivityMapping("1808", "Travel: professional/personal services (any)", "away", "outside"),
    "1809": ActivityMapping("1809", "Travel: household services (any)", "away", "outside"),
    "1810": ActivityMapping("1810", "Travel: government services (any)", "away", "outside"),
    "1811": ActivityMapping("1811", "Travel: eating and drinking (any)", "away", "outside"),
    "1812": ActivityMapping("1812", "Travel: socializing/leisure (any)", "away", "outside"),
    "1813": ActivityMapping("1813", "Travel: sports/exercise (any)", "away", "outside"),
    "1814": ActivityMapping("1814", "Travel: religious/spiritual (any)", "away", "outside"),
    "1815": ActivityMapping("1815", "Travel: volunteer (any)", "away", "outside"),
    "1816": ActivityMapping("1816", "Travel: telephone calls (any)", "away", "outside"),
    "1818": ActivityMapping("1818", "Travel: security procedures (any)", "away", "outside"),
}


# ============================================================
# TIER-1 FALLBACK (2-digit prefix)
# Last resort if neither tier-3 nor tier-2 matched.
# ============================================================

TIER1_FALLBACK: dict[str, ActivityMapping] = {
    "01": ActivityMapping("01", "Personal care (any)", "home", "any"),
    "02": ActivityMapping("02", "Household activities (any)", "home", "any"),
    "03": ActivityMapping("03", "Caring for HH members (any)", "home", "any"),
    "04": ActivityMapping("04", "Caring for non-HH members (any)", "away", "outside"),
    "05": ActivityMapping("05", "Work and work-related (any)", "ambiguous", "outside"),
    "06": ActivityMapping("06", "Education (any)", "ambiguous", "outside"),
    "07": ActivityMapping("07", "Consumer purchases (any)", "away", "outside"),
    "08": ActivityMapping("08", "Professional services (any)", "away", "outside"),
    "09": ActivityMapping("09", "Household services received (any)", "home", "any"),
    "10": ActivityMapping("10", "Government services (any)", "away", "outside"),
    "11": ActivityMapping("11", "Eating and drinking (any)", "ambiguous", "kitchen"),
    "12": ActivityMapping("12", "Socializing, relaxing, leisure (any)", "home", "living_room"),
    "13": ActivityMapping("13", "Sports, exercise, recreation (any)", "ambiguous", "outside"),
    "14": ActivityMapping("14", "Religious and spiritual (any)", "ambiguous", "any"),
    "15": ActivityMapping("15", "Volunteer activities (any)", "away", "outside"),
    "16": ActivityMapping("16", "Telephone calls (any)", "home", "any"),
    "18": ActivityMapping("18", "Traveling (any)", "away", "outside"),
    "50": ActivityMapping("50", "Data collection limitation (any)", "ambiguous", "any"),
}


# ============================================================
# LOOKUP API
# ============================================================

def lookup(atus_code: str) -> ActivityMapping:
    """
    Return the best ActivityMapping for a 6-digit ATUS code.

    Lookup priority:
      1. Exact tier-3 match (6-digit)
      2. Tier-2 fallback (first 4 digits)
      3. Tier-1 fallback (first 2 digits)
      4. Unknown — returns ambiguous/any
    """
    if atus_code in ACTIVITY_MAP:
        return ACTIVITY_MAP[atus_code]

    prefix4 = atus_code[:4]
    if prefix4 in TIER2_FALLBACK:
        return TIER2_FALLBACK[prefix4]

    prefix2 = atus_code[:2]
    if prefix2 in TIER1_FALLBACK:
        return TIER1_FALLBACK[prefix2]

    return ActivityMapping(
        atus_code, f"Unknown code {atus_code}", "ambiguous", "any",
        energy_note="Code not in lexicon — treat as home/ambiguous conservatively",
    )


def resolve_occupancy(
    atus_code: str,
    persona: PersonaFlags,
    previous_code: str | None = None,
) -> Occupancy:
    """
    Return the resolved occupancy status for an ATUS activity code.

    Applies three persona-context rules on top of the raw mapping:

      Rule 1 — Work location (PersonaFlags.work_from_home):
        0501–0504 "Work / work-related / income-generating / job search" →
        'home' if WFH, else 'away'. Documented choice: ATUS does not record work
        location; WFH flag is set from demographic stratum priors (BLS ATUS WFH
        supplement).

      Rule 2 — Eating location (previous activity context):
        110101/110199 "Eating and drinking" → 'away' if the immediately preceding
        activity was a travel episode (tier-1 code '18'); else → 'home'.
        Documented choice: ~60% of US meals eaten at home (USDA ERS, 2023);
        a preceding travel episode is a reliable proxy for eating out.

      Rule 3 — Exercise location (PersonaFlags.home_gym):
        Exercise codes that could be home or gym (aerobics, gymnastics, cardio
        equipment, weightlifting, working-out, yoga, dancing) → 'home' if
        PersonaFlags.home_gym, else → 'away'.
        Documented choice: RECS 2020 treadmill ownership ~12% nationally;
        sampled per demographic stratum in persona initialization.

    All other ambiguous codes default to 'home' conservatively (avoids
    underestimating occupancy and over-triggering HVAC setback).
    """
    mapping = lookup(atus_code)

    if mapping.occupancy != "ambiguous":
        return mapping.occupancy

    # Rule 1: work codes
    if atus_code in _WORK_CODES:
        return "home" if persona.work_from_home else "away"

    # Rule 2: eating — infer from previous episode
    if atus_code in _EATING_CODES:
        if previous_code and previous_code[:2] == "18":
            return "away"
        return "home"

    # Rule 3: exercise equipment / at-home workouts
    if atus_code in _HOME_GYM_RESOLVABLE:
        return "home" if persona.home_gym else "away"

    # Rule 4: empirical TEWHERE occupancy prior for the code's tier-1 prefix.
    # Data-driven fallback for residual ambiguous codes (shopping, socializing,
    # services, …) that Rules 1-3 do not cover. Per-stratum where available,
    # else the pooled "ALL" prior. Absent (no bundled CSV) → falls through.
    priors = _load_occupancy_priors()
    if priors:
        prefix = atus_code[:2]
        p_home = priors.get((persona.stratum or "ALL", prefix))
        if p_home is None:
            p_home = priors.get(("ALL", prefix))
        if p_home is not None:
            return "home" if p_home >= 0.5 else "away"

    # Conservative default for any remaining ambiguous codes
    return "home"


@lru_cache(maxsize=1)
def _load_occupancy_priors() -> dict[tuple[str, str], float]:
    """
    Load P(at home | stratum, 2-digit prefix) from the bundled occupancy_priors.csv.

    Returns {(stratum, code_prefix): p_home}. Degrades gracefully to an empty dict
    if the file is missing or unreadable, so importing this module never requires
    the data file to be present.
    """
    path = _DATA_DIR / "occupancy_priors.csv"
    priors: dict[tuple[str, str], float] = {}
    try:
        with path.open(newline="") as f:
            for row in csv.DictReader(f):
                priors[(row["stratum"], row["code_prefix"])] = float(row["p_home"])
    except (OSError, KeyError, ValueError):
        return {}
    return priors


def coverage_stats() -> dict:
    """Return coverage metrics useful for the methodology section."""
    return {
        "tier3_explicit": len(ACTIVITY_MAP),
        "tier2_fallbacks": len(TIER2_FALLBACK),
        "tier1_fallbacks": len(TIER1_FALLBACK),
        "away_codes": sum(1 for m in ACTIVITY_MAP.values() if m.occupancy == "away"),
        "home_codes": sum(1 for m in ACTIVITY_MAP.values() if m.occupancy == "home"),
        "ambiguous_codes": sum(1 for m in ACTIVITY_MAP.values() if m.occupancy == "ambiguous"),
    }


if __name__ == "__main__":
    stats = coverage_stats()
    print("Activity code mapping coverage:")
    for k, v in stats.items():
        print(f"  {k}: {v}")
    print()

    # --- Resolution rule demos ---
    p_office  = PersonaFlags(work_from_home=False, home_gym=False)
    p_wfh_gym = PersonaFlags(work_from_home=True,  home_gym=True)

    print("Resolution rule demos (office worker vs. WFH + home gym):")
    cases = [
        # (code, previous_code, description)
        ("050101", None,     "Work, main job"),
        ("110101", "180501", "Eating after commute travel → eating out"),
        ("110101", "020201", "Eating after food prep → eating at home"),
        ("130128", None,     "Cardiovascular equipment"),
        ("130136", None,     "Yoga"),
        ("010101", None,     "Sleeping (unambiguous)"),
        ("180501", None,     "Travel to work (unambiguous)"),
    ]
    header = f"  {'Code':<8} {'Prev':<8} {'Description':<40} {'office':>8} {'WFH+gym':>8}"
    print(header)
    print("  " + "-" * (len(header) - 2))
    for code, prev, desc in cases:
        r_office  = resolve_occupancy(code, p_office,  prev)
        r_wfh_gym = resolve_occupancy(code, p_wfh_gym, prev)
        print(f"  {code:<8} {str(prev or ''):<8} {desc:<40} {r_office:>8} {r_wfh_gym:>8}")

"""
ATUS activity code → occupancy status + device state + room mapping.

Coverage: ~220 of ~400 tier-3 codes, plus tier-2 fallbacks for all 18 tier-1
categories. The remaining ~80 uncovered codes all fall to their tier-2 fallback.

ATUS code structure:
  Digits 1-2: Tier 1 (major category, e.g. 01 = Personal Care)
  Digits 3-4: Tier 2 (subcategory, e.g. 0101 = Sleeping)
  Digits 5-6: Tier 3 (specific activity, e.g. 010101 = Sleeping)

Key for building energy:
  occupancy: 'home' | 'away' | 'ambiguous'
    'home'      → occupant is in the building (HVAC holds comfort setpoint)
    'away'      → occupant has left (HVAC enters setback; all home devices idle)
    'ambiguous' → context-dependent (e.g., telecommuting, home gym vs. public gym)

  room: where the occupant most likely is when home
  devices_on / devices_off: triggered appliances at activity start
  energy_note: plain-language note for citable methodology section

Reference: ATUS 2023 Activity Lexicon (BLS)
  https://www.bls.gov/tus/lexiconwex2023.pdf
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

Occupancy = Literal["home", "away", "ambiguous"]


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


# Codes that flip home↔away depending on PersonaFlags
_WORK_CODES: frozenset[str] = frozenset({
    "050101", "050102", "050103", "050189",
    "050201", "050202", "050203", "050299",
    # Catch-all and n.e.c. work codes present in bundled frequency CSVs that
    # were previously missing and fell through to ambiguous occupancy → home.
    "050301", "050302", "050399",
    "050401", "050402", "050499",
})

_EATING_CODES: frozenset[str] = frozenset({"110101", "110199"})

_HOME_GYM_RESOLVABLE: frozenset[str] = frozenset({
    "130101",  # aerobics (home workout video vs. aerobics class)
    "130107",  # gymnastics
    "130116",  # yoga
    "130120",  # cardiovascular equipment (home treadmill vs. gym machine)
    "130121",  # weightlifting / strength training
    "130122",  # exercise machine (n.e.c.)
    "130126",  # dancing
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
    # Tier 1: 01 — Personal Care Activities                               #
    # ------------------------------------------------------------------ #

    # 0101 Sleeping
    "010101": ActivityMapping("010101", "Sleeping", "home", "bedroom",
                              devices_off=["tv", "computer", "dishwasher", "washer", "dryer"],
                              energy_note="Longest continuous HVAC setback trigger; ~8h daily"),
    "010102": ActivityMapping("010102", "Sleeplessness", "home", "bedroom"),
    "010199": ActivityMapping("010199", "Sleeping, n.e.c.", "home", "bedroom"),

    # 0102 Grooming
    "010201": ActivityMapping("010201", "Washing, dressing, grooming", "home", "bathroom",
                              devices_on=["water_heater"],
                              energy_note="Water heater demand: ~10 min shower ≈ 0.5 kWh"),
    "010299": ActivityMapping("010299", "Personal care, n.e.c.", "home", "bathroom"),

    # 0103 Health-related
    "010301": ActivityMapping("010301", "Health-related self care", "home", "bathroom"),
    "010399": ActivityMapping("010399", "Personal care services, n.e.c.", "home", "any"),

    # 0104 Personal activities
    "010401": ActivityMapping("010401", "Personal/private activities", "home", "any"),

    # ------------------------------------------------------------------ #
    # Tier 1: 02 — Household Activities                                   #
    # ------------------------------------------------------------------ #

    # 0201 Food and drink preparation / clean-up
    "020101": ActivityMapping("020101", "Food and drink preparation", "home", "kitchen",
                              devices_on=["oven", "range", "range_hood", "microwave"],
                              energy_note="Cooking is ~10% of residential energy; oven ≈ 2.3 kW, range ≈ 1.2 kW per burner"),
    "020102": ActivityMapping("020102", "Food presentation", "home", "kitchen",
                              devices_on=["microwave"]),
    "020103": ActivityMapping("020103", "Kitchen and food clean-up", "home", "kitchen",
                              devices_on=["dishwasher"],
                              energy_note="Dishwasher is a primary demand-response target; typical cycle 1.5 kWh"),
    "020199": ActivityMapping("020199", "Food prep, n.e.c.", "home", "kitchen",
                              devices_on=["microwave"]),

    # 0202 Housework
    "020201": ActivityMapping("020201", "Interior cleaning", "home", "living_room",
                              devices_on=["vacuum"],
                              energy_note="Vacuum ≈ 0.5–1.4 kW; short duration, low total energy"),
    "020202": ActivityMapping("020202", "Laundry", "home", "laundry_room",
                              devices_on=["washer"],
                              energy_note="Washer ≈ 0.5 kWh/cycle (cold); dryer ≈ 3–5 kWh — dryer starts ~45 min after washer; model as separate delayed trigger"),
    "020203": ActivityMapping("020203", "Sewing, repairing textiles", "home", "living_room"),
    "020204": ActivityMapping("020204", "Storing interior HH items including food", "home", "kitchen"),
    "020299": ActivityMapping("020299", "Housework, n.e.c.", "home", "living_room"),

    # 0203 Exterior maintenance
    "020301": ActivityMapping("020301", "Lawn, garden, and houseplants", "home", "outside",
                              energy_note="Occupant is outside but home; HVAC holds setpoint"),
    "020302": ActivityMapping("020302", "Ponds, pools, and hot tubs", "home", "outside",
                              devices_on=["pool_pump"],
                              energy_note="Pool pump ≈ 1–2 kW continuous; major energy consumer in warm climates"),
    "020303": ActivityMapping("020303", "Exterior cleaning", "home", "outside"),
    "020399": ActivityMapping("020399", "Exterior maintenance, n.e.c.", "home", "outside"),

    # 0204 Interior maintenance / repair / decoration
    "020401": ActivityMapping("020401", "Interior arrangement, decoration, repairs", "home", "any"),
    "020402": ActivityMapping("020402", "Building and repairing furniture", "home", "garage"),
    "020403": ActivityMapping("020403", "Heating and cooling", "home", "any",
                              devices_on=["hvac"],
                              energy_note="Occupant actively adjusting HVAC — likely a manual override event; log separately"),
    "020404": ActivityMapping("020404", "Plumbing, home repairs", "home", "any"),
    "020499": ActivityMapping("020499", "Interior maintenance, n.e.c.", "home", "any"),

    # 0205 Vehicle / appliance maintenance
    "020501": ActivityMapping("020501", "Vehicle repair/maintenance", "home", "garage"),
    "020502": ActivityMapping("020502", "Appliances, tools, and toys repair", "home", "garage"),
    "020599": ActivityMapping("020599", "Appliance/vehicle maintenance, n.e.c.", "home", "garage"),

    # 0206 Household management
    "020601": ActivityMapping("020601", "Financial management", "home", "any",
                              devices_on=["computer"]),
    "020602": ActivityMapping("020602", "Household and personal organization", "home", "any"),
    "020603": ActivityMapping("020603", "HH and personal mail/messages (non-email)", "home", "any"),
    "020604": ActivityMapping("020604", "HH and personal email/messages", "home", "any",
                              devices_on=["computer"]),
    "020699": ActivityMapping("020699", "Household management, n.e.c.", "home", "any"),

    # 0207 Other household
    "020701": ActivityMapping("020701", "Seasonal decorating (e.g., Christmas)", "home", "living_room"),
    "020799": ActivityMapping("020799", "Household activities, n.e.c.", "home", "any"),

    # ------------------------------------------------------------------ #
    # Tier 1: 03 — Caring For & Helping Household Members                 #
    # ------------------------------------------------------------------ #

    # 0301 Childcare (physical)
    "030101": ActivityMapping("030101", "Physical care for HH children", "home", "any"),
    "030102": ActivityMapping("030102", "Reading to/with HH children", "home", "living_room"),
    "030103": ActivityMapping("030103", "Playing with HH children, not sports", "home", "living_room"),
    "030104": ActivityMapping("030104", "Arts and crafts with HH children", "home", "living_room"),
    "030105": ActivityMapping("030105", "Playing sports with HH children", "home", "outside"),
    "030106": ActivityMapping("030106", "Talking with/listening to HH children", "home", "any"),
    "030107": ActivityMapping("030107", "Organization and planning for HH children", "home", "any"),
    "030108": ActivityMapping("030108", "Looking after HH children (primary activity)", "home", "living_room",
                              devices_on=["tv"],
                              energy_note="TV commonly on during passive childcare supervision"),
    "030109": ActivityMapping("030109", "Attending HH children's events", "away", "outside"),
    "030110": ActivityMapping("030110", "Waiting for/with HH children", "ambiguous", "any"),
    "030111": ActivityMapping("030111", "Picking up/dropping off HH children", "away", "outside"),
    "030199": ActivityMapping("030199", "Childcare for HH children, n.e.c.", "home", "any"),

    # 0302 Adult care
    "030201": ActivityMapping("030201", "Physical care for HH adults", "home", "any"),
    "030202": ActivityMapping("030202", "Looking after HH adult", "home", "any"),
    "030203": ActivityMapping("030203", "Providing medical care to HH adult", "home", "any"),
    "030204": ActivityMapping("030204", "Obtaining care/services for HH adult", "ambiguous", "any"),
    "030299": ActivityMapping("030299", "Adult care for HH members, n.e.c.", "home", "any"),

    # 0303 Helping HH members
    "030301": ActivityMapping("030301", "Helping HH adults", "home", "any"),
    "030399": ActivityMapping("030399", "Helping HH members, n.e.c.", "home", "any"),

    # ------------------------------------------------------------------ #
    # Tier 1: 04 — Caring For & Helping Non-HH Members                   #
    # Interpretation: nearly always requires leaving the home             #
    # ------------------------------------------------------------------ #

    "040101": ActivityMapping("040101", "Physical care for non-HH children", "away", "outside"),
    "040102": ActivityMapping("040102", "Reading to/with non-HH children", "ambiguous", "outside"),
    "040103": ActivityMapping("040103", "Playing with non-HH children", "away", "outside"),
    "040104": ActivityMapping("040104", "Arts and crafts with non-HH children", "away", "outside"),
    "040105": ActivityMapping("040105", "Playing sports with non-HH children", "away", "outside"),
    "040106": ActivityMapping("040106", "Talking with non-HH children", "ambiguous", "any"),
    "040108": ActivityMapping("040108", "Looking after non-HH children", "away", "outside"),
    "040109": ActivityMapping("040109", "Attending events for non-HH children", "away", "outside"),
    "040111": ActivityMapping("040111", "Picking up/dropping off non-HH children", "away", "outside"),
    "040199": ActivityMapping("040199", "Non-HH childcare, n.e.c.", "away", "outside"),
    "040201": ActivityMapping("040201", "Physical care for non-HH adults", "away", "outside"),
    "040202": ActivityMapping("040202", "Looking after non-HH adult", "away", "outside"),
    "040299": ActivityMapping("040299", "Non-HH adult care, n.e.c.", "away", "outside"),
    "040301": ActivityMapping("040301", "Helping non-HH adults", "away", "outside"),
    "040399": ActivityMapping("040399", "Helping non-HH members, n.e.c.", "away", "outside"),

    # ------------------------------------------------------------------ #
    # Tier 1: 05 — Work & Work-Related Activities                         #
    # Interpretation: occupant leaves home unless explicitly WFH          #
    # ATUS does not distinguish WFH vs. office — treat as 'ambiguous'     #
    # and use employment status + COVID-era year as proxy for WFH         #
    # ------------------------------------------------------------------ #

    "050101": ActivityMapping("050101", "Work, main job", "ambiguous", "outside",
                              energy_note="Resolution rule: if PersonaFlags.work_from_home → 'home' "
                                          "(HVAC holds comfort setpoint); else → 'away' (HVAC setback). "
                                          "Post-2022 WFH prevalence ~25-30% for employed adults (BLS, 2023)."),
    "050102": ActivityMapping("050102", "Work, other job(s)", "ambiguous", "outside"),
    "050103": ActivityMapping("050103", "Security procedures related to work", "away", "outside"),
    "050189": ActivityMapping("050189", "Work, n.e.c.", "ambiguous", "outside"),
    "050201": ActivityMapping("050201", "Socializing at job", "away", "outside"),
    "050202": ActivityMapping("050202", "Eating and drinking as part of job", "away", "outside"),
    "050203": ActivityMapping("050203", "Sports as part of job", "away", "outside"),
    "050299": ActivityMapping("050299", "Work-related activities, n.e.c.", "away", "outside"),

    # ------------------------------------------------------------------ #
    # Tier 1: 06 — Education                                              #
    # Interpretation: class = away; homework = home; ambiguous by default #
    # ------------------------------------------------------------------ #

    "060101": ActivityMapping("060101", "Taking class for degree/certification", "away", "outside"),
    "060102": ActivityMapping("060102", "Taking class for personal interest", "away", "outside"),
    "060103": ActivityMapping("060103", "Taking class for other reason", "away", "outside"),
    "060199": ActivityMapping("060199", "Taking class, n.e.c.", "away", "outside"),
    "060201": ActivityMapping("060201", "Extracurricular club activities", "away", "outside"),
    "060202": ActivityMapping("060202", "Extracurricular music/performance activities", "away", "outside"),
    "060203": ActivityMapping("060203", "Research/homework for extracurricular", "home", "any",
                              devices_on=["computer"]),
    "060289": ActivityMapping("060289", "Education extracurricular activities, n.e.c.", "ambiguous", "any"),
    "060301": ActivityMapping("060301", "Research/homework", "home", "any",
                              devices_on=["computer"],
                              energy_note="Computer on; lighting likely on — concentrated load block"),
    "060302": ActivityMapping("060302", "Homework/research as part of job", "ambiguous", "any",
                              devices_on=["computer"]),
    "060399": ActivityMapping("060399", "Research/homework, n.e.c.", "home", "any",
                              devices_on=["computer"]),
    "060401": ActivityMapping("060401", "Administrative activities as student", "ambiguous", "any"),
    "060499": ActivityMapping("060499", "Administrative activities, n.e.c.", "ambiguous", "any"),

    # ------------------------------------------------------------------ #
    # Tier 1: 07 — Consumer Purchases (shopping)                          #
    # Interpretation: nearly always away from home                        #
    # ------------------------------------------------------------------ #

    "070101": ActivityMapping("070101", "Grocery shopping", "away", "outside"),
    "070102": ActivityMapping("070102", "Purchasing gas", "away", "outside"),
    "070103": ActivityMapping("070103", "Purchasing food (not grocery)", "away", "outside"),
    "070104": ActivityMapping("070104", "Shopping, except groceries/food/gas", "away", "outside"),
    "070105": ActivityMapping("070105", "Comparison shopping", "ambiguous", "any",
                              devices_on=["computer"],
                              energy_note="Online comparison shopping → home + computer; in-store → away"),
    "070199": ActivityMapping("070199", "Consumer purchases, n.e.c.", "away", "outside"),

    # ------------------------------------------------------------------ #
    # Tier 1: 08 — Professional & Personal Care Services                  #
    # Interpretation: visiting service providers = away                   #
    # ------------------------------------------------------------------ #

    "080101": ActivityMapping("080101", "Using financial services and banking", "away", "outside"),
    "080102": ActivityMapping("080102", "Using legal services", "away", "outside"),
    "080103": ActivityMapping("080103", "Using medical services", "away", "outside"),
    "080104": ActivityMapping("080104", "Using personal care services (salon, etc.)", "away", "outside"),
    "080105": ActivityMapping("080105", "Using real estate services", "away", "outside"),
    "080199": ActivityMapping("080199", "Using professional services, n.e.c.", "away", "outside"),

    # ------------------------------------------------------------------ #
    # Tier 1: 09 — Household Services                                     #
    # Interpretation: occupant is HOME waiting for/supervising provider   #
    # ------------------------------------------------------------------ #

    "090101": ActivityMapping("090101", "Using interior cleaning services", "home", "any",
                              energy_note="Occupant home; HVAC at comfort. Service provider may open doors"),
    "090102": ActivityMapping("090102", "Using meal preparation services", "home", "kitchen"),
    "090103": ActivityMapping("090103", "Using gardening/lawn care services", "home", "any"),
    "090104": ActivityMapping("090104", "Using property maintenance/repair services", "home", "any"),
    "090199": ActivityMapping("090199", "Using household services, n.e.c.", "home", "any"),

    # ------------------------------------------------------------------ #
    # Tier 1: 10 — Government Services & Civic Obligations                #
    # Interpretation: away (at government office, polling place, etc.)    #
    # ------------------------------------------------------------------ #

    "100101": ActivityMapping("100101", "Using government services", "away", "outside"),
    "100102": ActivityMapping("100102", "Civic obligations (jury duty, etc.)", "away", "outside"),
    "100199": ActivityMapping("100199", "Government services, n.e.c.", "away", "outside"),

    # ------------------------------------------------------------------ #
    # Tier 1: 11 — Eating & Drinking                                      #
    # Interpretation: home if eating at home; away if eating out          #
    # ATUS does not distinguish location — default home, mark ambiguous   #
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
    # ------------------------------------------------------------------ #

    # 1201 Socializing
    "120101": ActivityMapping("120101", "Socializing and communicating with others", "ambiguous", "living_room",
                              energy_note="If at home (host): HVAC comfort + lighting. If away (guest): setback."),
    "120102": ActivityMapping("120102", "Attending or hosting social functions", "ambiguous", "living_room",
                              devices_on=["tv", "lighting_living"]),
    "120103": ActivityMapping("120103", "Attending meetings/conferences/training", "away", "outside"),
    "120199": ActivityMapping("120199", "Socializing, n.e.c.", "ambiguous", "living_room"),

    # 1202 Attending events (performing arts, movies, sports venues)
    "120201": ActivityMapping("120201", "Attending performing arts", "away", "outside"),
    "120202": ActivityMapping("120202", "Attending museums", "away", "outside"),
    "120203": ActivityMapping("120203", "Attending movies/film", "away", "outside"),
    "120204": ActivityMapping("120204", "Attending gambling establishments", "away", "outside"),
    "120299": ActivityMapping("120299", "Attending events, n.e.c.", "away", "outside"),

    # 1203 TV / screen / leisure computing
    "120301": ActivityMapping("120301", "Watching TV", "home", "living_room",
                              devices_on=["tv"],
                              energy_note="#1 leisure activity by duration (~2.8 h/day avg); drives evening peak. "
                                          "TV ≈ 0.1–0.4 kW depending on size/type"),
    "120302": ActivityMapping("120302", "Watching movies (not on TV)", "home", "living_room",
                              devices_on=["tv", "computer"]),
    "120303": ActivityMapping("120303", "Computer use for leisure (not games)", "home", "living_room",
                              devices_on=["computer"],
                              energy_note="Laptop ≈ 0.05 kW; desktop ≈ 0.15–0.3 kW; monitor ≈ 0.025 kW"),
    "120304": ActivityMapping("120304", "Playing non-computer/video games", "home", "living_room"),
    "120305": ActivityMapping("120305", "Playing video/computer games", "home", "living_room",
                              devices_on=["tv", "gaming_console"],
                              energy_note="Gaming console ≈ 0.1–0.25 kW; can rival TV energy"),
    "120306": ActivityMapping("120306", "Playing games, n.e.c.", "home", "living_room"),
    "120307": ActivityMapping("120307", "Listening to the radio", "home", "any",
                              devices_on=["radio"]),
    "120308": ActivityMapping("120308", "Listening to/playing music (not radio)", "home", "any",
                              devices_on=["stereo"]),
    "120309": ActivityMapping("120309", "Playing musical instruments", "home", "living_room"),
    "120310": ActivityMapping("120310", "Writing for leisure", "home", "any"),
    "120311": ActivityMapping("120311", "Other leisure activities", "home", "any"),
    "120312": ActivityMapping("120312", "Knitting, sewing, or crocheting", "home", "living_room"),
    "120313": ActivityMapping("120313", "Meditating", "home", "bedroom"),
    "120399": ActivityMapping("120399", "Relaxing and leisure, n.e.c.", "home", "living_room"),

    # 1204 Reading
    "120401": ActivityMapping("120401", "Reading for personal interest", "home", "living_room"),
    "120402": ActivityMapping("120402", "Reading as part of job", "ambiguous", "any"),
    "120499": ActivityMapping("120499", "Reading, n.e.c.", "home", "any"),

    # ------------------------------------------------------------------ #
    # Tier 1: 13 — Sports, Exercise & Recreation                          #
    # Interpretation: outdoor sports = away; home gym = home              #
    # ATUS does not distinguish — use 'ambiguous' as default              #
    # ------------------------------------------------------------------ #

    "130101": ActivityMapping("130101", "Doing aerobics", "ambiguous", "outside",
                              energy_note="Home aerobics (YouTube) = home + TV/computer; gym = away"),
    "130102": ActivityMapping("130102", "Playing baseball/softball", "away", "outside"),
    "130103": ActivityMapping("130103", "Playing basketball", "away", "outside"),
    "130104": ActivityMapping("130104", "Biking", "away", "outside"),
    "130105": ActivityMapping("130105", "Playing football", "away", "outside"),
    "130106": ActivityMapping("130106", "Golfing", "away", "outside"),
    "130107": ActivityMapping("130107", "Doing gymnastics", "ambiguous", "outside"),
    "130108": ActivityMapping("130108", "Hiking", "away", "outside"),
    "130109": ActivityMapping("130109", "Playing hockey", "away", "outside"),
    "130110": ActivityMapping("130110", "Hunting or fishing", "away", "outside"),
    "130111": ActivityMapping("130111", "Playing soccer", "away", "outside"),
    "130112": ActivityMapping("130112", "Playing softball", "away", "outside"),
    "130113": ActivityMapping("130113", "Playing tennis", "away", "outside"),
    "130114": ActivityMapping("130114", "Playing volleyball", "away", "outside"),
    "130115": ActivityMapping("130115", "Walking", "away", "outside",
                              energy_note="Occupant away from home even if brief; HVAC setback if walk > 30 min"),
    "130116": ActivityMapping("130116", "Doing yoga", "ambiguous", "any"),
    "130117": ActivityMapping("130117", "Playing racquetball", "away", "outside"),
    "130120": ActivityMapping("130120", "Using cardiovascular equipment", "ambiguous", "any",
                              energy_note="Resolution rule: if PersonaFlags.home_gym → 'home'; else → 'away'. "
                                          "RECS 2020: ~12% of US households own a treadmill; higher for "
                                          "higher-income strata. Sample from RECS ownership rates by stratum."),
    "130121": ActivityMapping("130121", "Doing weightlifting/strength training", "ambiguous", "any",
                              energy_note="Same resolution rule as 130120: PersonaFlags.home_gym."),
    "130122": ActivityMapping("130122", "Using exercise machine", "ambiguous", "any",
                              energy_note="Same resolution rule as 130120: PersonaFlags.home_gym."),
    "130123": ActivityMapping("130123", "Running", "away", "outside"),
    "130124": ActivityMapping("130124", "Swimming", "away", "outside"),
    "130125": ActivityMapping("130125", "Skiing", "away", "outside"),
    "130126": ActivityMapping("130126", "Dancing", "ambiguous", "any"),
    "130199": ActivityMapping("130199", "Sports and exercise, n.e.c.", "ambiguous", "outside"),
    "130201": ActivityMapping("130201", "Watching sports/exercise/recreation events", "away", "outside"),
    "130202": ActivityMapping("130202", "Watching sports on TV", "home", "living_room",
                              devices_on=["tv"]),
    "130301": ActivityMapping("130301", "Waiting associated with sports activities", "ambiguous", "outside"),
    "130399": ActivityMapping("130399", "Sports participation, n.e.c.", "ambiguous", "outside"),

    # ------------------------------------------------------------------ #
    # Tier 1: 14 — Religious & Spiritual Activities                        #
    # Interpretation: typically away (place of worship); prayer = home    #
    # ------------------------------------------------------------------ #

    "140101": ActivityMapping("140101", "Attending religious services", "away", "outside"),
    "140102": ActivityMapping("140102", "Participation in religious practices", "ambiguous", "any"),
    "140103": ActivityMapping("140103", "Praying", "home", "any"),
    "140104": ActivityMapping("140104", "Religious education activities", "away", "outside"),
    "140199": ActivityMapping("140199", "Religious and spiritual activities, n.e.c.", "ambiguous", "any"),

    # ------------------------------------------------------------------ #
    # Tier 1: 15 — Volunteer Activities                                    #
    # Interpretation: typically away (serving others at their location)   #
    # ------------------------------------------------------------------ #

    "150101": ActivityMapping("150101", "Administrative/support activities (volunteer)", "away", "outside"),
    "150102": ActivityMapping("150102", "Teaching/educating (volunteer)", "away", "outside"),
    "150103": ActivityMapping("150103", "Collecting/delivering goods (volunteer)", "away", "outside"),
    "150104": ActivityMapping("150104", "Caring for/helping people (volunteer)", "away", "outside"),
    "150105": ActivityMapping("150105", "Fundraising (volunteer)", "away", "outside"),
    "150106": ActivityMapping("150106", "Building/repairing (volunteer)", "away", "outside"),
    "150107": ActivityMapping("150107", "Outdoor activities (volunteer)", "away", "outside"),
    "150108": ActivityMapping("150108", "Performing/cultural activities (volunteer)", "away", "outside"),
    "150199": ActivityMapping("150199", "Volunteer activities, n.e.c.", "away", "outside"),

    # ------------------------------------------------------------------ #
    # Tier 1: 16 — Telephone Calls                                         #
    # Interpretation: occupant is home unless accompanied by travel code   #
    # ------------------------------------------------------------------ #

    "160101": ActivityMapping("160101", "Telephone calls to/from family members", "home", "any"),
    "160102": ActivityMapping("160102", "Telephone calls to/from friends/neighbors", "home", "any"),
    "160103": ActivityMapping("160103", "Telephone calls to/from education services", "home", "any"),
    "160104": ActivityMapping("160104", "Telephone calls to/from financial services", "home", "any"),
    "160105": ActivityMapping("160105", "Telephone calls to/from healthcare", "home", "any"),
    "160106": ActivityMapping("160106", "Telephone calls to/from household services", "home", "any"),
    "160107": ActivityMapping("160107", "Telephone calls to/from personal care services", "home", "any"),
    "160108": ActivityMapping("160108", "Telephone calls to/from retail stores", "home", "any"),
    "160199": ActivityMapping("160199", "Telephone calls, n.e.c.", "home", "any"),

    # ------------------------------------------------------------------ #
    # Tier 1: 18 — Traveling                                               #
    # ALL travel codes = occupant is away from home                       #
    # HVAC should enter setback immediately when a travel code begins     #
    # ------------------------------------------------------------------ #

    "180101": ActivityMapping("180101", "Travel related to personal care", "away", "outside"),
    "180102": ActivityMapping("180102", "Travel: attending personal care services", "away", "outside"),
    "180199": ActivityMapping("180199", "Travel related to personal care, n.e.c.", "away", "outside"),
    "180201": ActivityMapping("180201", "Travel related to household activities", "away", "outside"),
    "180202": ActivityMapping("180202", "Travel related to household services", "away", "outside"),
    "180299": ActivityMapping("180299", "Travel related to household activities, n.e.c.", "away", "outside"),
    "180301": ActivityMapping("180301", "Travel related to caring for HH children", "away", "outside"),
    "180302": ActivityMapping("180302", "Travel related to children's education", "away", "outside"),
    "180399": ActivityMapping("180399", "Travel related to caring for HH members, n.e.c.", "away", "outside"),
    "180401": ActivityMapping("180401", "Travel related to caring for non-HH adults", "away", "outside"),
    "180499": ActivityMapping("180499", "Travel related to non-HH member care, n.e.c.", "away", "outside"),
    "180501": ActivityMapping("180501", "Travel related to work, main job", "away", "outside",
                              energy_note="Commute = occupant away; HVAC setback. Duration distributes "
                                          "by ATUS commute time distributions. This is the single largest "
                                          "away-from-home period for employed profiles (O1)."),
    "180502": ActivityMapping("180502", "Travel related to work, other job", "away", "outside"),
    "180601": ActivityMapping("180601", "Travel related to education", "away", "outside"),
    "180701": ActivityMapping("180701", "Travel related to consumer purchases", "away", "outside"),
    "180801": ActivityMapping("180801", "Travel related to using professional services", "away", "outside"),
    "180901": ActivityMapping("180901", "Travel related to government services/civic obligations", "away", "outside"),
    "181001": ActivityMapping("181001", "Travel related to eating and drinking", "away", "outside"),
    "181101": ActivityMapping("181101", "Travel related to socializing/leisure", "away", "outside"),
    "181201": ActivityMapping("181201", "Travel related to sports/exercise/recreation", "away", "outside"),
    "181301": ActivityMapping("181301", "Travel related to religious/spiritual activities", "away", "outside"),
    "181401": ActivityMapping("181401", "Travel related to volunteer activities", "away", "outside"),
    "181499": ActivityMapping("181499", "Travel related to volunteer activities, n.e.c.", "away", "outside"),
    "181501": ActivityMapping("181501", "Travel related to other activities", "away", "outside"),
    "181601": ActivityMapping("181601", "Travel related to telephone calls", "away", "outside"),
    "189999": ActivityMapping("189999", "Traveling, n.e.c.", "away", "outside"),

    # ------------------------------------------------------------------ #
    # Data collection limitations / unknown                                #
    # ------------------------------------------------------------------ #
    "500101": ActivityMapping("500101", "Insufficient detail in verbatim", "ambiguous", "any"),
    "500103": ActivityMapping("500103", "Missing travel or destination", "ambiguous", "any"),
    "500105": ActivityMapping("500105", "Watching TV not identified", "home", "living_room",
                              devices_on=["tv"]),
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
    # Household
    "0201": ActivityMapping("0201", "Food prep (any)", "home", "kitchen",
                            devices_on=["microwave"]),
    "0202": ActivityMapping("0202", "Housework (any)", "home", "living_room"),
    "0203": ActivityMapping("0203", "Exterior maintenance (any)", "home", "outside"),
    "0204": ActivityMapping("0204", "Interior maintenance (any)", "home", "any"),
    "0205": ActivityMapping("0205", "Vehicle/appliance maintenance (any)", "home", "garage"),
    "0206": ActivityMapping("0206", "Household management (any)", "home", "any"),
    # Caring HH
    "0301": ActivityMapping("0301", "HH childcare (any)", "home", "any"),
    "0302": ActivityMapping("0302", "HH adult care (any)", "home", "any"),
    "0303": ActivityMapping("0303", "Helping HH members (any)", "home", "any"),
    # Caring non-HH
    "0401": ActivityMapping("0401", "Non-HH childcare (any)", "away", "outside"),
    "0402": ActivityMapping("0402", "Non-HH adult care (any)", "away", "outside"),
    "0403": ActivityMapping("0403", "Helping non-HH (any)", "away", "outside"),
    # Work
    "0501": ActivityMapping("0501", "Working (any)", "ambiguous", "outside"),
    "0502": ActivityMapping("0502", "Work-related (any)", "away", "outside"),
    # Education
    "0601": ActivityMapping("0601", "Taking class (any)", "away", "outside"),
    "0602": ActivityMapping("0602", "Extracurricular education (any)", "ambiguous", "outside"),
    "0603": ActivityMapping("0603", "Homework/research (any)", "home", "any",
                            devices_on=["computer"]),
    # Shopping / services
    "0701": ActivityMapping("0701", "Consumer purchases (any)", "away", "outside"),
    "0801": ActivityMapping("0801", "Professional services (any)", "away", "outside"),
    "0901": ActivityMapping("0901", "Household services received (any)", "home", "any"),
    "1001": ActivityMapping("1001", "Government services (any)", "away", "outside"),
    # Eating
    "1101": ActivityMapping("1101", "Eating and drinking (any)", "ambiguous", "kitchen"),
    # Leisure
    "1201": ActivityMapping("1201", "Socializing (any)", "ambiguous", "living_room"),
    "1202": ActivityMapping("1202", "Attending events (any)", "away", "outside"),
    "1203": ActivityMapping("1203", "Screen leisure (any)", "home", "living_room",
                            devices_on=["tv"]),
    "1204": ActivityMapping("1204", "Reading (any)", "home", "any"),
    # Sports / exercise
    "1301": ActivityMapping("1301", "Sports/exercise (any)", "ambiguous", "outside"),
    "1302": ActivityMapping("1302", "Watching sports (any)", "ambiguous", "any"),
    # Religious / volunteer
    "1401": ActivityMapping("1401", "Religious activities (any)", "away", "outside"),
    "1501": ActivityMapping("1501", "Volunteer activities (any)", "away", "outside"),
    # Telephone
    "1601": ActivityMapping("1601", "Telephone calls (any)", "home", "any"),
    # Travel
    "1801": ActivityMapping("1801", "Traveling (any)", "away", "outside"),
    "1810": ActivityMapping("1810", "Traveling (any)", "away", "outside"),
    "1811": ActivityMapping("1811", "Traveling (any)", "away", "outside"),
    "1812": ActivityMapping("1812", "Traveling (any)", "away", "outside"),
    "1813": ActivityMapping("1813", "Traveling (any)", "away", "outside"),
    "1814": ActivityMapping("1814", "Traveling (any)", "away", "outside"),
    "1815": ActivityMapping("1815", "Traveling (any)", "away", "outside"),
    "1816": ActivityMapping("1816", "Traveling (any)", "away", "outside"),
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
    "14": ActivityMapping("14", "Religious and spiritual (any)", "away", "outside"),
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
        050101–050299 "Work, main job / work-related" → 'home' if WFH, else 'away'.
        Documented choice: ATUS does not record work location; WFH flag is set from
        demographic stratum priors (BLS ATUS 2022-23 WFH supplement).

      Rule 2 — Eating location (previous activity context):
        110101/110199 "Eating and drinking" → 'away' if the immediately preceding
        activity was a travel episode (tier-1 code '18'); else → 'home'.
        Documented choice: ~60% of US meals eaten at home (USDA ERS, 2023);
        a preceding travel episode is a reliable proxy for eating out.

      Rule 3 — Exercise location (PersonaFlags.home_gym):
        Exercise codes that could be home or gym (yoga, treadmill, weights, aerobics)
        → 'home' if PersonaFlags.home_gym, else → 'away'.
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

    # Conservative default for any remaining ambiguous codes
    return "home"


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
        ("110101", "020101", "Eating after food prep → eating at home"),
        ("130120", None,     "Cardiovascular equipment"),
        ("130116", None,     "Yoga"),
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

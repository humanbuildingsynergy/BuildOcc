# Tier 2 validation results — BuildOcc v1.0.1 (run 2026-09-03)

Result files behind Tables 7 and 8 and Figure 4 of the SoftwareX manuscript
(SOFTX-D-26-00798, revision 2). Produced with the released v1.0.1 package,
Anthropic provider (claude-haiku-4-5-20251001 for steps, claude-sonnet-4-6 for
reflection), temperature 0, 512-token cap.

    python scripts/validate_strata.py  --provider anthropic --seeds 3 --days 1 \
        --start-date 2025-08-11 --output-dir scripts/experiments/outputs/tier2_v1.0.1_2026-09-03
    python scripts/validate_signals.py --provider anthropic --seeds 5 \
        --output-dir scripts/experiments/outputs/tier2_v1.0.1_2026-09-03

`strata_*`: 4 strata x 3 seeds x 96 timesteps (Table 7, Figure 4).
`signal_*`: 4 strata x 3 signal types x 5 seeds, 10-step warm-up from 16:00 (Table 8).
The `.log` files are the harness console output.

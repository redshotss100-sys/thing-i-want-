# AGENTS.md

## Project rules
- Headless mode must remain pygame-free.
- Prefer minimal changes.
- Do not remove existing logs.
- Prefer JSONL for runtime logs.
- Reuse existing stat names and save paths where possible.
- Validate changes by running the headless sim.

## For logging tasks
- Periodic tick snapshots go to tick_log.jsonl
- Default interval is 2000 ticks
- Each line must be valid JSON
- Keep per-snapshot logging lightweight
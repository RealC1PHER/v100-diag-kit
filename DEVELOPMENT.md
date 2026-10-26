# Development notes

This page is for people changing the kit. You do not need it to run the tests.

## Checks that need no GPU

```bash
python3 v100_diag.py --selftest      # verdict logic and kernel-log parser
shellcheck run_v100_diag.sh          # launcher lint
```

Both run in CI on every push (`.github/workflows/ci.yml`).

## Proving the memory test works (fault injection)

This run corrupts one value in GPU memory on purpose, so **a memory FAIL is the expected result**. The console output and the report are marked as a fault injection test. Never use this run to judge a GPU.

```bash
.venv/bin/python v100_diag.py --profile quick --inject-fault memtest
```

The launcher also accepts `--inject-fault memtest`.

## Short run for checking changes

```bash
sudo ./run_v100_diag.sh --profile quick --scale 0.15 --yes
```

About 2 minutes. Use it to check file names, report layout and the install flow on a real GPU.

## Layout

| File | Role |
|---|---|
| `run_v100_diag.sh` | launcher: prerequisite check, install with confirmation, starts the engine |
| `v100_diag.py` | test engine: telemetry thread, one worker process per GPU and test, verdicts, report |

The engine runs every test in a separate worker process, so a hung or lost GPU cannot freeze the orchestrator. Results are judged in `judge_test()`; add a test by adding a worker function, a label, a summary line and a call in `orchestrate()`.

## Not covered by automatic tests

The lost-GPU path (evidence collection after "fell off the bus") needs a real failing GPU and has only been checked by reading the code.

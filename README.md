# v100-diag-kit

**Acceptance, burn-in and fault-diagnosis testing for NVIDIA Tesla V100 GPUs.** Also runs on other NVIDIA GPUs.

v100-diag-kit tests a GPU the way it is actually used, then gives you a clear verdict and a documented, reproducible record of the result.

## Who it is for

- **Buyers and owners**: check a new or second-hand card before you rely on it, or find out why a card is unstable.
- **Sellers, refurbishers and resellers**: run a standard acceptance and burn-in test before shipping, and attach the report to the sale.
- **Warranty and RMA claims**: produce evidence a seller or manufacturer can read and verify, including kernel errors, telemetry and a checksum-protected archive.
- **System builders and data centres**: qualify cards, adapters, risers and power supplies in a new build, one GPU at a time or all at once.

## What you get

- **A verdict per GPU**: PASS, WARN or FAIL, with the reason for every finding.
- **Real fault detection**: memory pattern errors, wrong computation results, ECC errors, PCIe link problems, thermal throttling, and GPUs that **drop off the bus (Xid 79) when the load keeps switching between busy and idle**.
- **Evidence you can hand over**: a report, machine-readable results, 4 Hz telemetry, kernel log excerpts and a SHA-256 checksum, packed into one archive.
- **A safe, transparent install**: one command on Debian or Ubuntu. It checks what is installed, lists what is missing, asks before installing anything, never touches the NVIDIA driver, and sends nothing anywhere.

## Run it

```bash
git clone https://github.com/RealC1PHER/v100-diag-kit.git
cd v100-diag-kit
sudo ./run_v100_diag.sh --profile quick       # about 10 minutes
sudo ./run_v100_diag.sh --profile standard    # about 40 minutes per GPU
```

Use `sudo` so the kernel log can be read (that is where GPU errors show up). Without it the tests still run, but the report says kernel errors could not be captured.

When it finishes, open `results/run_<date>_<serial>/REPORT_<serial>.md`. The GPU serial number is part of the folder, report and archive names. A `.tar.gz` of the whole result folder is created next to it.

## What it needs

The NVIDIA driver must already be installed and working (`nvidia-smi` shows your GPU). The kit never installs or changes the driver.

Everything else is checked first, listed, and installed only if you answer `y`:

- apt packages: `python3`, `python3-venv`, `python3-pip`, `pciutils`, `ca-certificates`
- Python libraries `torch` (about 2.5 GB), `numpy` and `nvidia-ml-py`, in a private `.venv` folder inside the kit. Nothing else on the system is touched.

Want to look first? `./run_v100_diag.sh --check-only` changes nothing.

| System | Notes |
|---|---|
| Debian 13 | tested on a real V100 |
| Debian 12, Ubuntu 22.04, Ubuntu 24.04 | package and PyTorch install checked in clean containers, not run on a GPU |
| Newest Ubuntu | uses a newer PyTorch build automatically if the default one does not exist for its Python version; the installer verifies the build supports Volta GPUs and stops with a message if not |

## The tests

| Test | Looks for |
|---|---|
| PCIe link and bandwidth | link running below its maximum width or speed, very slow transfers |
| Compute correctness | wrong results, compared with a CPU reference |
| GPU memory pattern test | any bit that reads back differently from what was written |
| **Power-swing burst test** | **GPU lost (Xid 79), driver errors, wrong results while the load jumps between busy and idle** |
| Optional power-limit steps | failures that only happen at a certain power limit |
| Sustained full-load soak | overheating, throttling, falling performance |

All the while it records, four times per second, power, temperature, clocks, utilisation, memory use, PCIe link and throttle reasons, and it watches the kernel log for NVIDIA errors. ECC counters are compared before and after. If a GPU stops responding, the kit saves the evidence (kernel log, `nvidia-smi -q`, and `nvidia-bug-report.sh` if available) and carries on with any other GPU.

A GPU that has fallen off the bus usually stays gone until the machine is **fully powered off and on again**; a reboot is often not enough.

## Options

```
--profile quick|standard|full   test length (default standard)
--gpus all|0,1                  which GPUs to test (default all)
--concurrent                    test all chosen GPUs at once (loads the power supply like real use)
--scale N                       multiply every duration, e.g. 0.5 or 2
--power-limits 150,200          extra burst runs at these power limits in watts (needs root)
--outdir DIR                    where to put the results
--no-archive                    skip the .tar.gz
--yes                           answer yes to every question
--check-only                    only check prerequisites
--install-only                  install prerequisites, then stop
```

By default the GPUs are tested one at a time. Run `--concurrent` as a second pass to put the whole machine under load.

## Reading the result

Each GPU gets one overall result:

- **PASS**: no errors, normal values.
- **WARN**: finished, but something was unusual (very hot, throttling, PCIe link narrower than the card supports, correctable ECC errors, falling performance).
- **FAIL**: a kernel error, a lost GPU, a wrong result, a memory mismatch, uncorrectable ECC errors, or a test that did not finish.

The exit code is 1 if any GPU failed. A PASS does not prove a card is perfect, but a FAIL is a real problem. If a card fails in one machine, try it in another slot or machine: a card that fails everywhere is the card, one that fails only in one place is the slot, adapter or power supply.

## Sending the result

If someone asked you to run the test (a seller, a buyer, a colleague), send them **both files** printed at the end of the run, always together:

- the **`.tar.gz`** archive, for example `results/run_20261026_142914_<serial>.tar.gz`: the whole result folder in one file
- the matching **`.sha256`** file next to it: the checksum that proves the archive was not changed after the test

Nothing is uploaded automatically; you send the files yourself. The receiver checks them with:

```bash
sha256sum -c run_20261026_142914_<serial>.tar.gz.sha256
```

It must print `OK`. Without the `.sha256` file the receiver cannot prove the result is untouched.

If the run was interrupted or the machine had to be powered off, send the contents of the `results/` folder anyway.

## Result files

| File | Content |
|---|---|
| `REPORT_<serial>.md` | the human-readable report |
| `results_<serial>.json` | the same, machine-readable |
| `run.log` | full timestamped log |
| `telemetry_gpu<N>.csv` | the 4 Hz measurements |
| `worker_gpu<N>_<test>.log` | per-test output |
| `kernel_log_new_lines.txt` | kernel messages that appeared during the run |
| `evidence/` | only if a GPU was lost |

The report contains GPU serial numbers and UUIDs, OS and CPU model, driver and library versions, PCI addresses and the measurements. It does not contain host name, user name, IP or MAC addresses.

## Limits

These are load tests that can find a faulty GPU; they cannot certify a good one. The kit changes no firmware, clocks or fan settings. Only `--power-limits` touches the power limit, temporarily, and restores it.

## Questions and bug reports

Open an issue in the **Issues** tab of this repository. For a test problem, attach `REPORT_<serial>.md` (and `run.log` if the run stopped early), after checking them for anything you do not want to share.

## License

MIT, see [LICENSE](LICENSE). Contributors: see [DEVELOPMENT.md](DEVELOPMENT.md).

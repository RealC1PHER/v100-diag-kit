#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
V100 deep-dive diagnostic kit - test engine.

Runs a set of load, correctness and stability tests against one or more NVIDIA GPUs
(written for Tesla V100, works on other NVIDIA GPUs), records 4 Hz telemetry to log
files, captures kernel (Xid) events, and writes a test report.

Normally started through run_v100_diag.sh, which checks and installs the prerequisites.
"""
import argparse
import atexit
import csv
import datetime as dt
import hashlib
import json
import os
import platform
import random
import re
import shutil
import subprocess
import sys
import tarfile
import threading
import time
import traceback

VERSION = "1.0.0"
HERE = os.path.dirname(os.path.abspath(__file__))

PROFILES = {
    #            mem passes, compute s, pcie s, burst s, steady s, per-limit step s
    "quick":    dict(mem_passes=1, compute_s=30,  pcie_s=8,  burst_s=180,  steady_s=120,  limit_step_s=60),
    "standard": dict(mem_passes=3, compute_s=120, pcie_s=20, burst_s=900,  steady_s=600,  limit_step_s=120),
    "full":     dict(mem_passes=6, compute_s=300, pcie_s=45, burst_s=2400, steady_s=1800, limit_step_s=180),
}

THROTTLE_BITS = {
    0x1: "gpu_idle", 0x2: "app_clocks_setting", 0x4: "sw_power_cap", 0x8: "hw_slowdown",
    0x10: "sync_boost", 0x20: "sw_thermal_slowdown", 0x40: "hw_thermal_slowdown", 0x80: "hw_power_brake",
}
THERMAL_BITS = 0x20 | 0x40
BAD_BITS = 0x8 | 0x80

XID_RE = re.compile(r"NVRM: Xid \(PCI:([0-9a-fA-F:.]+)\):? *(\d+)(.*)")
LOST_RE = re.compile(r"fallen off the bus|GPU is lost|GPU has fallen", re.I)

# --------------------------------------------------------------------------------------
# small helpers
# --------------------------------------------------------------------------------------
LOGF = None


def now_iso():
    return dt.datetime.now().astimezone().isoformat(timespec="milliseconds")


def log(msg="", also_print=True):
    line = "%s  %s" % (dt.datetime.now().strftime("%H:%M:%S"), msg)
    if also_print:
        print(line, flush=True)
    if LOGF:
        LOGF.write(now_iso() + "  " + msg + "\n")
        LOGF.flush()


def decode_throttle(v):
    if not v:
        return "none"
    return "+".join(n for b, n in THROTTLE_BITS.items() if v & b) or hex(v)


def run_cmd(cmd, timeout=30):
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return r.returncode, r.stdout, r.stderr
    except Exception as e:  # noqa
        return 127, "", str(e)


def read_klog():
    """Return (source, lines) of the kernel log, or (None, [])."""
    for cmd in (["dmesg"], ["sudo", "-n", "dmesg"],
                ["journalctl", "-k", "-b", "--no-pager", "-o", "short-iso"]):
        rc, out, _ = run_cmd(cmd, 25)
        if rc == 0 and out.strip():
            return " ".join(cmd[:2]) if cmd[0] == "sudo" else cmd[0], out.splitlines()
    return None, []


def new_lines(before, after):
    bset = set(before[-6000:])
    return [l for l in after[-6000:] if l not in bset]


def parse_events(lines):
    ev = []
    for l in lines:
        m = XID_RE.search(l)
        if m:
            ev.append(dict(kind="xid", pci=m.group(1), xid=int(m.group(2)), text=l.strip()[:300]))
        elif LOST_RE.search(l):
            ev.append(dict(kind="gpu_lost", text=l.strip()[:300]))
    return ev


def human(n, unit="B"):
    for p in ("", "K", "M", "G", "T"):
        if abs(n) < 1024:
            return "%.1f %s%s" % (n, p, unit)
        n /= 1024.0
    return "%.1f P%s" % (n, unit)


# --------------------------------------------------------------------------------------
# NVML wrapper (nvidia-ml-py)
# --------------------------------------------------------------------------------------
def nvml():
    import pynvml  # provided by the pip package nvidia-ml-py
    return pynvml


def nv_call(fn, *a, default=None):
    try:
        return fn(*a)
    except Exception:  # noqa
        return default


def power_instant_w(N, h):
    try:
        fv = N.nvmlDeviceGetFieldValues(h, [N.NVML_FI_DEV_POWER_INSTANT])
        v = fv[0]
        if v.nvmlReturn == 0:
            return v.value.uiVal / 1000.0
    except Exception:  # noqa
        pass
    return N.nvmlDeviceGetPowerUsage(h) / 1000.0


def mem_temp_c(N, h):
    try:
        fv = N.nvmlDeviceGetFieldValues(h, [N.NVML_FI_DEV_MEMORY_TEMP])
        v = fv[0]
        if v.nvmlReturn == 0:
            return int(v.value.uiVal)
    except Exception:  # noqa
        pass
    return None


def throttle_now(N, h):
    fn = getattr(N, "nvmlDeviceGetCurrentClocksEventReasons", None) or \
        getattr(N, "nvmlDeviceGetCurrentClocksThrottleReasons")
    return int(fn(h))


def gpu_info(N, h, idx):
    g = dict(index=idx)
    g["name"] = nv_call(N.nvmlDeviceGetName, h, default="?")
    if isinstance(g["name"], bytes):
        g["name"] = g["name"].decode()
    g["uuid"] = nv_call(N.nvmlDeviceGetUUID, h, default="?")
    g["serial"] = nv_call(N.nvmlDeviceGetSerial, h, default="not reported")
    g["part_number"] = nv_call(N.nvmlDeviceGetBoardPartNumber, h, default="not reported")
    g["vbios"] = nv_call(N.nvmlDeviceGetVbiosVersion, h, default="?")
    pci = nv_call(N.nvmlDeviceGetPciInfo, h)
    g["pci_bus_id"] = (pci.busId.decode() if isinstance(pci.busId, bytes) else str(pci.busId)) if pci else "?"
    mi = nv_call(N.nvmlDeviceGetMemoryInfo, h)
    g["memory_total_mib"] = int(mi.total / 2 ** 20) if mi else None
    pl = nv_call(N.nvmlDeviceGetPowerManagementLimitConstraints, h)
    g["power_limit_min_w"] = pl[0] / 1000.0 if pl else None
    g["power_limit_max_w"] = pl[1] / 1000.0 if pl else None
    d = nv_call(N.nvmlDeviceGetPowerManagementDefaultLimit, h)
    g["power_limit_default_w"] = d / 1000.0 if d else None
    e = nv_call(N.nvmlDeviceGetEnforcedPowerLimit, h)
    g["power_limit_enforced_w"] = e / 1000.0 if e else None
    g["temp_slowdown_c"] = nv_call(N.nvmlDeviceGetTemperatureThreshold, h, N.NVML_TEMPERATURE_THRESHOLD_SLOWDOWN)
    g["temp_shutdown_c"] = nv_call(N.nvmlDeviceGetTemperatureThreshold, h, N.NVML_TEMPERATURE_THRESHOLD_SHUTDOWN)
    g["temp_max_operating_c"] = nv_call(N.nvmlDeviceGetTemperatureThreshold, h, N.NVML_TEMPERATURE_THRESHOLD_GPU_MAX)
    g["pcie_gen_current"] = nv_call(N.nvmlDeviceGetCurrPcieLinkGeneration, h)
    g["pcie_gen_max"] = nv_call(N.nvmlDeviceGetMaxPcieLinkGeneration, h)
    g["pcie_width_current"] = nv_call(N.nvmlDeviceGetCurrPcieLinkWidth, h)
    g["pcie_width_max"] = nv_call(N.nvmlDeviceGetMaxPcieLinkWidth, h)
    ecc = nv_call(N.nvmlDeviceGetEccMode, h)
    g["ecc_mode"] = ("enabled" if ecc and ecc[0] else "disabled") if ecc else "not reported"
    g["persistence_mode"] = nv_call(N.nvmlDeviceGetPersistenceMode, h, default="?")
    g["clock_sm_max_mhz"] = nv_call(N.nvmlDeviceGetMaxClockInfo, h, N.NVML_CLOCK_SM)
    g["clock_mem_max_mhz"] = nv_call(N.nvmlDeviceGetMaxClockInfo, h, N.NVML_CLOCK_MEM)
    return g


def ecc_counts(N, h):
    out = {}
    for label, kind in (("corrected", N.NVML_MEMORY_ERROR_TYPE_CORRECTED),
                        ("uncorrected", N.NVML_MEMORY_ERROR_TYPE_UNCORRECTED)):
        for scope_name, scope in (("volatile", N.NVML_VOLATILE_ECC), ("aggregate", N.NVML_AGGREGATE_ECC)):
            v = nv_call(N.nvmlDeviceGetTotalEccErrors, h, kind, scope)
            out["%s_%s" % (label, scope_name)] = int(v) if v is not None else None
    return out


def gpu_alive(N, h):
    try:
        N.nvmlDeviceGetPowerUsage(h)
        N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU)
        return True
    except Exception:  # noqa
        return False


# --------------------------------------------------------------------------------------
# telemetry thread
# --------------------------------------------------------------------------------------
class Telemetry(threading.Thread):
    COLS = ["time_iso", "t_s", "phase", "gpu", "power_w", "limit_w", "temp_c", "mem_temp_c", "sm_mhz",
            "mem_mhz", "util_gpu", "util_mem", "mem_used_mib", "pcie_gen", "pcie_width", "throttle_hex",
            "throttle_names"]

    def __init__(self, N, handles, outdir, interval=0.25):
        super().__init__(daemon=True)
        self.N, self.handles, self.interval = N, handles, interval
        self.phase = "idle"
        self.stop_flag = threading.Event()
        self.t0 = time.monotonic()
        self.files, self.writers = {}, {}
        self.lost = {}
        self.err_count = {i: 0 for i in handles}
        self.stats = {}
        self.last = {}
        self.lock = threading.Lock()
        for i in handles:
            f = open(os.path.join(outdir, "telemetry_gpu%d.csv" % i), "w", newline="", buffering=1)
            w = csv.writer(f)
            w.writerow(self.COLS)
            self.files[i], self.writers[i] = f, w

    def set_phase(self, name):
        self.phase = name

    def _acc(self, gpu, ph, power, temp, mtemp, sm, thr, prev_power):
        s = self.stats.setdefault((gpu, ph), dict(n=0, p_sum=0.0, p_max=0.0, p_min=1e9, t_max=0, mt_max=0,
                                                  sm_sum=0.0, sm_min=1e9, sm_max=0, thr=0, step_max=0.0,
                                                  swings100=0, swings150=0, first=time.time()))
        s["n"] += 1
        s["p_sum"] += power
        s["p_max"] = max(s["p_max"], power)
        s["p_min"] = min(s["p_min"], power)
        s["t_max"] = max(s["t_max"], temp)
        if mtemp is not None:
            s["mt_max"] = max(s["mt_max"], mtemp)
        s["sm_sum"] += sm
        s["sm_min"] = min(s["sm_min"], sm)
        s["sm_max"] = max(s["sm_max"], sm)
        s["thr"] |= thr
        if prev_power is not None:
            d = abs(power - prev_power)
            s["step_max"] = max(s["step_max"], d)
            if d >= 100:
                s["swings100"] += 1
            if d >= 150:
                s["swings150"] += 1
        s["last"] = time.time()

    def run(self):
        N = self.N
        nxt = time.monotonic()
        while not self.stop_flag.is_set():
            nxt += self.interval
            ph = self.phase
            for i, h in self.handles.items():
                if i in self.lost:
                    continue
                t = time.monotonic() - self.t0
                try:
                    power = power_instant_w(N, h)
                    limit = N.nvmlDeviceGetEnforcedPowerLimit(h) / 1000.0
                    temp = N.nvmlDeviceGetTemperature(h, N.NVML_TEMPERATURE_GPU)
                    mtemp = mem_temp_c(N, h)
                    sm = N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_SM)
                    mc = N.nvmlDeviceGetClockInfo(h, N.NVML_CLOCK_MEM)
                    ut = N.nvmlDeviceGetUtilizationRates(h)
                    mi = N.nvmlDeviceGetMemoryInfo(h)
                    gen = N.nvmlDeviceGetCurrPcieLinkGeneration(h)
                    wid = N.nvmlDeviceGetCurrPcieLinkWidth(h)
                    thr = throttle_now(N, h)
                except Exception as e:  # noqa
                    self.err_count[i] += 1
                    self.writers[i].writerow([now_iso(), "%.2f" % t, ph, i, "ERR", "", "", "", "", "", "", "", "",
                                              "", "", "", str(e)[:80]])
                    if self.err_count[i] >= 3 and not gpu_alive(N, h):
                        self.lost[i] = dict(time=now_iso(), phase=ph, error=str(e)[:200])
                    continue
                self.err_count[i] = 0
                with self.lock:
                    prev = self.last.get(i)
                    self._acc(i, ph, power, temp, mtemp, sm, thr, prev)
                    self.last[i] = power
                    self.last[(i, "snap")] = dict(power=power, temp=temp, sm=sm, util=ut.gpu, mtemp=mtemp, thr=thr)
                self.writers[i].writerow([now_iso(), "%.2f" % t, ph, i, "%.1f" % power, "%.0f" % limit, temp,
                                          "" if mtemp is None else mtemp, sm, mc, ut.gpu, ut.memory,
                                          int(mi.used / 2 ** 20), gen, wid, hex(thr), decode_throttle(thr)])
            delay = nxt - time.monotonic()
            if delay > 0:
                self.stop_flag.wait(delay)
            else:
                nxt = time.monotonic()
        for f in self.files.values():
            f.close()

    def snap(self, i):
        with self.lock:
            return dict(self.last.get((i, "snap"), {}))

    def phase_stats(self, i, ph):
        with self.lock:
            s = dict(self.stats.get((i, ph), {}))
        if not s:
            return None
        s["p_avg"] = s["p_sum"] / s["n"]
        s["sm_avg"] = s["sm_sum"] / s["n"]
        return s

    def stop(self):
        self.stop_flag.set()
        self.join(timeout=5)


# --------------------------------------------------------------------------------------
# WORKERS (run in a separate process so a hung / lost GPU cannot hang the orchestrator)
# --------------------------------------------------------------------------------------
def worker_main(name, gpu, params):
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    fn = {"memtest": w_memtest, "compute": w_compute, "pcie": w_pcie, "burst": w_burst,
          "steady": w_steady}[name]
    res = dict(status="ok")
    try:
        import torch  # noqa
        torch.cuda.init()
        res.update(fn(gpu, params))
    except Exception as e:  # noqa
        res = dict(status="error", error="%s: %s" % (type(e).__name__, str(e)[:400]),
                   trace=traceback.format_exc()[-1200:])
    print("RESULT_JSON:" + json.dumps(res), flush=True)


def _prog(msg):
    sys.stderr.write("%s %s\n" % (dt.datetime.now().strftime("%H:%M:%S"), msg))
    sys.stderr.flush()


def w_memtest(dev, p):
    import torch
    torch.cuda.set_device(dev)
    free, total = torch.cuda.mem_get_info()
    reserve = max(int(1.6 * 2 ** 30), int(0.07 * total))
    usable = max(0, int(free - reserve))
    chunk_bytes = 256 * 2 ** 20
    n_chunks = usable // chunk_bytes
    n = chunk_bytes // 4
    if n_chunks < 2:
        raise RuntimeError("not enough free GPU memory for the memory test (free %s)" % human(free))
    chunks = [torch.empty(n, dtype=torch.int32, device=dev) for _ in range(n_chunks)]
    passes = int(p.get("passes", 1))
    inject = bool(p.get("inject_fault"))
    patterns = [("zeros", 0), ("ones", -1), ("0xAAAAAAAA", -1431655766), ("0x55555555", 1431655765),
                ("address", "addr"), ("address_inverted", "addr_inv"), ("random", "rand")]
    errs_total, per_pat, first_errs = 0, {}, []
    t0 = time.time()
    _prog("memory test: %d chunks x 256 MiB = %s per pass, %d pass(es)" % (n_chunks, human(n_chunks * chunk_bytes), passes))
    for ps in range(passes):
        for pname, pv in patterns:
            gens = None
            for ci, c in enumerate(chunks):
                if pv == "addr":
                    c.copy_(torch.arange(ci * n, (ci + 1) * n, dtype=torch.int64, device=dev).to(torch.int32))
                elif pv == "addr_inv":
                    c.copy_(~torch.arange(ci * n, (ci + 1) * n, dtype=torch.int64, device=dev).to(torch.int32))
                elif pv == "rand":
                    g = torch.Generator(device=dev)
                    g.manual_seed(9000 + ps * 1000 + ci)
                    c.copy_(torch.randint(-2 ** 31, 2 ** 31 - 1, (n,), dtype=torch.int32, device=dev, generator=g))
                else:
                    c.fill_(pv)
            if inject and ps == 0 and pname == "zeros":
                chunks[0][12345] = 1  # deliberate corruption to prove the checker catches it
            torch.cuda.synchronize()
            time.sleep(1.5)  # hold the data a moment before reading it back
            perr = 0
            for ci, c in enumerate(chunks):
                if pv == "addr":
                    exp = torch.arange(ci * n, (ci + 1) * n, dtype=torch.int64, device=dev).to(torch.int32)
                    bad = (c != exp)
                elif pv == "addr_inv":
                    exp = ~torch.arange(ci * n, (ci + 1) * n, dtype=torch.int64, device=dev).to(torch.int32)
                    bad = (c != exp)
                elif pv == "rand":
                    g = torch.Generator(device=dev)
                    g.manual_seed(9000 + ps * 1000 + ci)
                    exp = torch.randint(-2 ** 31, 2 ** 31 - 1, (n,), dtype=torch.int32, device=dev, generator=g)
                    bad = (c != exp)
                else:
                    bad = (c != pv)
                cnt = int(bad.sum().item())
                if cnt:
                    perr += cnt
                    if len(first_errs) < 20:
                        idx = torch.nonzero(bad)[:3].flatten().tolist()
                        first_errs.append(dict(pass_=ps, pattern=pname, chunk=ci, count=cnt, first_offsets=idx))
            per_pat[pname] = per_pat.get(pname, 0) + perr
            errs_total += perr
        _prog("memory test pass %d/%d done, errors so far: %d" % (ps + 1, passes, errs_total))
    return dict(chunks=n_chunks, tested_bytes=n_chunks * chunk_bytes, passes=passes, errors=errs_total,
                errors_by_pattern=per_pat, first_errors=first_errs, elapsed_s=round(time.time() - t0, 1),
                fault_injected=inject)


def w_compute(dev, p):
    import torch
    torch.cuda.set_device(dev)
    dur = float(p.get("seconds", 30))
    torch.manual_seed(1234)
    dd = torch.device("cuda", dev)
    out = dict(checks=[])
    # 1) accuracy against a CPU float64 reference
    n = 1024
    A = torch.randn(n, n, dtype=torch.float32)
    B = torch.randn(n, n, dtype=torch.float32)
    ref = A.double() @ B.double()
    scale = float(ref.abs().max())
    for dtname, dt_ in (("fp32", torch.float32), ("fp16", torch.float16)):
        C = (A.to(dd, dt_) @ B.to(dd, dt_)).float().cpu().double()
        rel = float((C - ref).abs().max()) / scale
        tol = 1e-4 if dt_ == torch.float32 else 2e-2
        out["checks"].append(dict(name="matmul_%s_vs_cpu_float64" % dtname, rel_error=rel, tolerance=tol,
                                  ok=rel < tol))
    # 2) exact integer arithmetic against CPU
    m = 1 << 22
    a = torch.randint(0, 2 ** 31 - 1, (m,), dtype=torch.int64)
    b = torch.randint(0, 2 ** 31 - 1, (m,), dtype=torch.int64)
    c = torch.randint(0, 2 ** 31 - 1, (m,), dtype=torch.int64)
    r_cpu = (a * b + c) ^ (a >> 3)
    r_gpu = ((a.to(dd) * b.to(dd) + c.to(dd)) ^ (a.to(dd) >> 3)).cpu()
    bad = int((r_cpu != r_gpu).sum())
    out["checks"].append(dict(name="int64_arithmetic_vs_cpu", mismatches=bad, ok=bad == 0))
    # 3) elementwise float functions
    x = torch.rand(m, dtype=torch.float32) * 10
    ref_f = torch.sin(x.double()) + torch.exp(-x.double())
    got_f = (torch.sin(x.to(dd)) + torch.exp(-x.to(dd))).double().cpu()
    relf = float((got_f - ref_f).abs().max())
    out["checks"].append(dict(name="sin_exp_vs_cpu_float64", max_abs_error=relf, tolerance=1e-5, ok=relf < 1e-5))
    # 4) repeatability: the same matmul must give bit-identical results every time
    mism, iters = 0, 0
    n2 = 4096
    for dtname, dt_ in (("fp16", torch.float16), ("fp32", torch.float32)):
        A2 = torch.randn(n2, n2, device=dd, dtype=dt_)
        B2 = torch.randn(n2, n2, device=dd, dtype=dt_)
        first = A2 @ B2
        end = time.time() + dur / 2.0
        k = 0
        while time.time() < end:
            for _ in range(4):
                if not torch.equal(A2 @ B2, first):
                    mism += 1
                iters += 1
            k += 1
        torch.cuda.synchronize()
    out["checks"].append(dict(name="repeatability_bit_identical", iterations=iters, mismatches=mism, ok=mism == 0))
    out["ok"] = all(c["ok"] for c in out["checks"])
    return out


def w_pcie(dev, p):
    import torch
    torch.cuda.set_device(dev)
    secs = float(p.get("seconds", 10))
    nbytes = 512 * 2 ** 20
    host = torch.empty(nbytes, dtype=torch.uint8).pin_memory()
    devb = torch.empty(nbytes, dtype=torch.uint8, device="cuda:%d" % dev)
    res = {}
    for name, fn in (("host_to_device", lambda: devb.copy_(host, non_blocking=True)),
                     ("device_to_host", lambda: host.copy_(devb, non_blocking=True))):
        rates = []
        end = time.time() + secs
        while time.time() < end:
            torch.cuda.synchronize()
            t = time.time()
            for _ in range(6):
                fn()
            torch.cuda.synchronize()
            rates.append(6 * nbytes / (time.time() - t) / 1e9)
        rates.sort()
        res[name] = dict(gbps_median=round(rates[len(rates) // 2], 2), gbps_min=round(rates[0], 2),
                         gbps_max=round(rates[-1], 2), samples=len(rates))
    return dict(transfers=res)


def _bursty(dev, seconds, seed, verify, progress_tag):
    import torch
    torch.cuda.set_device(dev)
    dd = torch.device("cuda", dev)
    rnd = random.Random(seed)
    torch.manual_seed(seed)
    A = torch.randn(8192, 8192, device=dd, dtype=torch.float16)
    B = torch.randn(8192, 8192, device=dd, dtype=torch.float16)
    C = torch.empty(8192, 8192, device=dd, dtype=torch.float16)
    As = torch.randn(512, 512, device=dd, dtype=torch.float16)
    Bs = torch.randn(512, 512, device=dd, dtype=torch.float16)
    ref = As @ Bs
    torch.cuda.synchronize()
    patterns = [
        ("prefill_like", (0.5, 3.0), (0.5, 2.0)),   # long compute bursts, then idle (LLM prompt processing)
        ("fast_pulses", (0.05, 0.25), (0.10, 0.40)),  # very short pulses
        ("hard_steps", (5.0, 10.0), (4.0, 8.0)),    # long plateaus
    ]
    end = time.time() + seconds
    bursts, mism, per = 0, 0, {}
    pi = 0
    next_report = time.time() + 60
    while time.time() < end:
        pname, on_r, off_r = patterns[pi % len(patterns)]
        seg_end = min(end, time.time() + max(45.0, seconds / 12.0))
        while time.time() < seg_end and time.time() < end:
            t_on = rnd.uniform(*on_r)
            t_off = rnd.uniform(*off_r)
            t1 = time.time() + t_on
            while time.time() < t1:
                for _ in range(3):
                    torch.mm(A, B, out=C)
                torch.cuda.synchronize()
            bursts += 1
            per[pname] = per.get(pname, 0) + 1
            if verify:
                if not torch.equal(As @ Bs, ref):
                    mism += 1
            time.sleep(t_off)  # idle gap: the GPU drops back to idle power
            if time.time() > next_report:
                _prog("%s: %d bursts so far, verification mismatches %d" % (progress_tag, bursts, mism))
                next_report = time.time() + 60
        pi += 1
    return bursts, mism, per


def w_burst(dev, p):
    seconds = float(p.get("seconds", 60))
    bursts, mism, per = _bursty(dev, seconds, int(p.get("seed", 7)), True, "burst test")
    return dict(bursts=bursts, verify_mismatches=mism, bursts_by_pattern=per, seconds=seconds)


def w_steady(dev, p):
    import torch
    torch.cuda.set_device(dev)
    dd = torch.device("cuda", dev)
    seconds = float(p.get("seconds", 60))
    torch.manual_seed(11)
    A = torch.randn(8192, 8192, device=dd, dtype=torch.float16)
    B = torch.randn(8192, 8192, device=dd, dtype=torch.float16)
    C = torch.empty(8192, 8192, device=dd, dtype=torch.float16)
    mem = torch.empty(256 * 2 ** 20, dtype=torch.float32, device=dd)  # 1 GiB, bandwidth mix
    mem.fill_(1.0)
    As = torch.randn(512, 512, device=dd, dtype=torch.float16)
    Bs = torch.randn(512, 512, device=dd, dtype=torch.float16)
    ref = As @ Bs
    torch.cuda.synchronize()
    end = time.time() + seconds
    win_t, win_flop = time.time(), 0.0
    tflops, mism = [], 0
    last_verify = time.time()
    while time.time() < end:
        for _ in range(4):
            torch.mm(A, B, out=C)
        mem.mul_(1.0000001)
        torch.cuda.synchronize()
        win_flop += 4 * 2 * 8192 ** 3
        now = time.time()
        if now - win_t >= 5.0:
            tflops.append(round(win_flop / (now - win_t) / 1e12, 2))
            win_t, win_flop = now, 0.0
        if now - last_verify >= 15.0:
            if not torch.equal(As @ Bs, ref):
                mism += 1
            last_verify = now
    q = max(1, len(tflops) // 4)
    first_q = sum(tflops[:q]) / q if tflops else 0.0
    last_q = sum(tflops[-q:]) / q if tflops else 0.0
    return dict(seconds=seconds, tflops_windows=tflops, tflops_first_quarter=round(first_q, 2),
                tflops_last_quarter=round(last_q, 2), verify_mismatches=mism)


# --------------------------------------------------------------------------------------
# ORCHESTRATOR
# --------------------------------------------------------------------------------------
class Run:
    pass


def collect_system_info(N, handles):
    info = {}
    osr = {}
    try:
        for l in open("/etc/os-release"):
            if "=" in l:
                k, v = l.strip().split("=", 1)
                osr[k] = v.strip('"')
    except Exception:  # noqa
        pass
    info["os"] = osr.get("PRETTY_NAME", platform.platform())
    info["kernel"] = platform.release()
    info["arch"] = platform.machine()
    cpu = "?"
    try:
        for l in open("/proc/cpuinfo"):
            if l.startswith("model name"):
                cpu = l.split(":", 1)[1].strip()
                break
    except Exception:  # noqa
        pass
    info["cpu"] = cpu
    info["cpu_threads"] = os.cpu_count()
    try:
        for l in open("/proc/meminfo"):
            if l.startswith("MemTotal"):
                info["ram_gib"] = round(int(l.split()[1]) / 2 ** 20, 1)
                break
    except Exception:  # noqa
        pass
    rc, out, _ = run_cmd(["systemd-detect-virt"], 10)
    info["virtualization"] = out.strip() if out.strip() else "none detected"
    info["python"] = platform.python_version()
    info["nvidia_driver"] = nv_call(N.nvmlSystemGetDriverVersion, default="?")
    if isinstance(info["nvidia_driver"], bytes):
        info["nvidia_driver"] = info["nvidia_driver"].decode()
    cv = nv_call(N.nvmlSystemGetCudaDriverVersion)
    info["cuda_driver_api"] = "%d.%d" % (cv // 1000, (cv % 1000) // 10) if cv else "?"
    try:
        import torch
        info["torch"] = torch.__version__
        info["torch_cuda_runtime"] = torch.version.cuda
    except Exception:  # noqa
        info["torch"] = "not importable"
    rc, out, _ = run_cmd(["lspci", "-nn"], 10)
    info["lspci_nvidia"] = [l.split(" ", 1)[1] for l in out.splitlines() if "NVIDIA" in l] if rc == 0 else []
    info["gpus"] = [gpu_info(N, h, i) for i, h in handles.items()]
    return info


def judge_test(r):
    """r: dict with test, status, worker, tel, klog_events, alive, ecc_delta, gpuinfo -> (verdict, notes, metrics)"""
    notes, verdict = [], "PASS"

    def bump(v, why):
        nonlocal verdict
        order = {"PASS": 0, "WARN": 1, "FAIL": 2}
        if order[v] > order[verdict]:
            verdict = v
        notes.append("%s: %s" % (v, why))

    w = r.get("worker") or {}
    gi = r["gpuinfo"]
    if r["status"] != "ok":
        bump("FAIL", r.get("status_text", "test did not finish"))
    if not r["alive"]:
        bump("FAIL", "GPU stopped responding during or after the test (fell off the bus / lost)")
    for ev in r["klog_events"]:
        if ev["kind"] == "xid":
            bump("FAIL", "kernel reported NVIDIA Xid %d" % ev["xid"])
        else:
            bump("FAIL", "kernel reported the GPU lost")
    ed = r.get("ecc_delta") or {}
    if (ed.get("uncorrected_volatile") or 0) > 0 or (ed.get("uncorrected_aggregate") or 0) > 0:
        bump("FAIL", "uncorrectable ECC errors increased")
    if (ed.get("corrected_volatile") or 0) > 0 or (ed.get("corrected_aggregate") or 0) > 0:
        bump("WARN", "correctable ECC errors increased")
    t = r["test"]
    if r["status"] == "ok":
        if t == "memtest" and w.get("errors", 0) > 0:
            bump("FAIL", "%d memory pattern mismatches" % w["errors"])
        if t == "compute":
            for c in w.get("checks", []):
                if not c.get("ok"):
                    bump("FAIL", "compute check failed: %s" % c["name"])
        if t in ("burst", "steady") and w.get("verify_mismatches", 0) > 0:
            bump("FAIL", "%d silent computation errors detected under load" % w["verify_mismatches"])
        if t == "steady":
            f, l = w.get("tflops_first_quarter", 0), w.get("tflops_last_quarter", 0)
            if f and l < 0.85 * f:
                bump("WARN", "throughput fell %.0f%% during the run (%.1f -> %.1f TFLOPS)" % (100 * (1 - l / f), f, l))
        if t == "pcie":
            gen = gi.get("pcie_gen_max") or 0
            wid = gi.get("pcie_width_current") or gi.get("pcie_width_max") or 0
            wmax = gi.get("pcie_width_max") or 0
            if wmax and wid < wmax:
                bump("WARN", "the PCIe link is negotiated at x%d, the GPU supports x%d (slot, riser or adapter "
                             "limit; bandwidth below is judged against x%d)" % (wid, wmax, wid))
            per_lane = {1: 0.25, 2: 0.5, 3: 0.985, 4: 1.969, 5: 3.938}.get(gen, 0)
            theo = per_lane * wid
            for k, v in (w.get("transfers") or {}).items():
                if theo and v["gbps_median"] < 0.6 * theo:
                    bump("WARN", "%s %.1f GB/s is below 60%% of the theoretical %.1f GB/s for Gen%d x%d" %
                         (k, v["gbps_median"], theo, gen, wid))
    tel = r.get("tel") or {}
    if tel:
        slow = gi.get("temp_slowdown_c") or 87
        maxop = gi.get("temp_max_operating_c") or 83
        if tel["t_max"] >= slow:
            bump("FAIL", "GPU temperature reached %d C (slowdown threshold %d C)" % (tel["t_max"], slow))
        elif tel["t_max"] >= maxop:
            bump("WARN", "GPU temperature reached %d C (rated maximum operating %d C)" % (tel["t_max"], maxop))
        if tel["thr"] & THERMAL_BITS:
            bump("WARN", "thermal slowdown was active (%s)" % decode_throttle(tel["thr"] & THERMAL_BITS))
        if tel["thr"] & BAD_BITS:
            bump("WARN", "hardware slowdown / power brake was asserted (%s)" % decode_throttle(tel["thr"] & BAD_BITS))
        if t == "burst" and r["status"] == "ok" and tel.get("swings100", 0) < 5:
            bump("WARN", "the burst test produced only %d power swings of 100 W or more per 0.25 s; the "
                         "scenario may not have been stressful enough on this system" % tel.get("swings100", 0))
    return verdict, notes


def write_report(run, outdir):
    L = []
    A = L.append
    A("# V100 deep-dive test report")
    A("")
    if getattr(run, "injected", False):
        A("> **FAULT INJECTION TEST - NOT A REAL RESULT.** One memory value was corrupted on purpose to prove the "
          "checker works. The memory test FAIL below is expected and says nothing about the GPU.")
        A("")
    A("- Kit version: %s" % VERSION)
    A("- Profile: **%s** (duration scale %.2f)%s" % (run.profile, run.scale,
                                                      "  -  CONCURRENT mode" if run.concurrent else ""))
    A("- Started: %s" % run.started)
    A("- Finished: %s%s" % (run.finished, "  **(run was ABORTED before completion)**" if run.aborted else ""))
    A("- Total duration: %s" % run.duration_text)
    A("")
    A("## Overall result")
    A("")
    A("| GPU | Model | Serial number | Result |")
    A("|---|---|---|---|")
    for g in run.info["gpus"]:
        A("| %d | %s | %s | **%s** |" % (g["index"], g["name"], g["serial"], run.overall.get(g["index"], "n/a")))
    A("")
    if any(v == "FAIL" for v in run.overall.values()):
        A("**At least one GPU FAILED. See the failing tests and the events section below. If a GPU stopped "
          "responding (\"fell off the bus\"), the machine must be fully powered off before the GPU can be used again.**")
        A("")
    A("Meaning: PASS = no errors and all measurements normal. WARN = completed but a value was outside the "
      "normal range (details below). FAIL = a hardware error, kernel error, wrong computation result, GPU "
      "loss, or an unfinished test.")
    A("")
    A("## Test results")
    A("")
    A("| GPU | Test | Result | Duration | Key figures |")
    A("|---|---|---|---|---|")
    for r in run.results:
        A("| %d | %s | **%s** | %s | %s |" % (r["gpu"], r["label"], r["verdict"], r["duration_text"], r["summary"]))
    A("")
    A("### Details and notes per test")
    A("")
    for r in run.results:
        A("**GPU %d - %s: %s**" % (r["gpu"], r["label"], r["verdict"]))
        A("")
        for n in r["notes"] or ["no problems found"]:
            A("- %s" % n)
        t = r.get("tel")
        if t:
            A("- Telemetry during this test: power avg %.0f W (min %.0f, max %.0f), largest change between two "
              "0.25 s samples %.0f W, swings of 100 W or more: %d, temperature max %d C%s, GPU clock avg %.0f MHz "
              "(min %d, max %d), throttle reasons seen: %s" % (
                  t["p_avg"], t["p_min"], t["p_max"], t["step_max"], t["swings100"], t["t_max"],
                  "" if not t["mt_max"] else ", memory temperature max %d C" % t["mt_max"], t["sm_avg"], t["sm_min"],
                  t["sm_max"], decode_throttle(t["thr"])))
        A("")
    A("## Events")
    A("")
    if run.events:
        for e in run.events:
            A("- [%s] %s" % (e.get("when", "?"), e["text"]))
    else:
        A("- No kernel Xid errors and no GPU-lost events were recorded.")
    A("- Kernel log source: %s" % (run.klog_source or "NOT READABLE (run with sudo to capture Xid events)"))
    A("")
    A("## ECC memory counters (before -> after the whole run)")
    A("")
    A("| GPU | Mode | Corrected (volatile) | Uncorrected (volatile) | Corrected (aggregate) | Uncorrected (aggregate) |")
    A("|---|---|---|---|---|---|")
    for g in run.info["gpus"]:
        b, a = run.ecc_before.get(g["index"], {}), run.ecc_after.get(g["index"], {})
        def pair(k):
            return "%s -> %s" % (b.get(k), a.get(k))
        A("| %d | %s | %s | %s | %s | %s |" % (g["index"], g["ecc_mode"], pair("corrected_volatile"),
                                             pair("uncorrected_volatile"), pair("corrected_aggregate"),
                                             pair("uncorrected_aggregate")))
    A("")
    A("## PCIe link (idle at start / seen during load / idle at end)")
    A("")
    A("| GPU | Maximum supported | At start | During load (lowest seen) | At end |")
    A("|---|---|---|---|---|")
    for g in run.info["gpus"]:
        i = g["index"]
        lk = run.links.get(i, {})
        A("| %d | Gen%s x%s | %s | %s | %s |" % (i, g["pcie_gen_max"], g["pcie_width_max"], lk.get("start", "?"),
                                              lk.get("load_min", "?"), lk.get("end", "?")))
    A("")
    A("(A GPU may lower its PCIe speed at idle to save power. A lower **width** during load, or a lower speed "
      "during load, points to a slot, riser or cable problem.)")
    A("")
    A("## System")
    A("")
    for k in ("os", "kernel", "arch", "cpu", "cpu_threads", "ram_gib", "virtualization", "nvidia_driver",
              "cuda_driver_api", "python", "torch", "torch_cuda_runtime"):
        A("- %s: %s" % (k.replace("_", " "), run.info.get(k)))
    for l in run.info.get("lspci_nvidia", []):
        A("- PCI device: %s" % l)
    A("")
    A("## GPU inventory")
    A("")
    for g in run.info["gpus"]:
        A("**GPU %d**" % g["index"])
        A("")
        for k, v in g.items():
            if k != "index":
                A("- %s: %s" % (k.replace("_", " "), v))
        A("")
    A("## Settings changed by this run")
    A("")
    A(run.settings_note)
    A("")
    A("## Files in this result folder")
    A("")
    A("- `REPORT_<serial>.md` - this report")
    A("- `results_<serial>.json` - the same results in machine-readable form")
    A("- `run.log` - the full console log with timestamps")
    A("- `telemetry_gpu<N>.csv` - 4 samples per second per GPU: power, limit, temperature, clocks, utilisation, "
      "memory use, PCIe link, throttle reasons, with the test phase name in every row")
    A("- `worker_gpu<N>_<test>.log` - progress output of each test")
    A("- `kernel_log_new_lines.txt` - kernel log lines that appeared during the run")
    A("- `system_info.json` - system and GPU details")
    A("- `evidence/` - only present if a GPU was lost (bug report and log excerpts)")
    A("")
    A("## What this report contains (privacy)")
    A("")
    A("GPU serial numbers and UUIDs (to identify the unit), operating system and hardware model information, "
      "driver and software versions, PCI bus addresses and the test measurements. No host name, user name, IP "
      "address, MAC address or file path of the machine is collected.")
    open(os.path.join(outdir, "REPORT_%s.md" % run.serial_tag), "w").write("\n".join(L) + "\n")


def build_summary(test, w, gi):
    try:
        if test == "memtest":
            return "%s tested per pass, %d pass(es), %d errors" % (human(w["tested_bytes"]), w["passes"], w["errors"])
        if test == "compute":
            return "%d checks, %d failed" % (len(w["checks"]), sum(1 for c in w["checks"] if not c["ok"]))
        if test == "pcie":
            t = w["transfers"]
            return "H2D %.1f GB/s, D2H %.1f GB/s" % (t["host_to_device"]["gbps_median"], t["device_to_host"]["gbps_median"])
        if test == "burst":
            return "%d load/idle bursts, %d verification mismatches" % (w["bursts"], w["verify_mismatches"])
        if test == "steady":
            return "%.1f -> %.1f TFLOPS, %d mismatches" % (w["tflops_first_quarter"], w["tflops_last_quarter"], w["verify_mismatches"])
    except Exception:  # noqa
        pass
    return "-"


LABELS = {"memtest": "GPU memory pattern test", "compute": "Compute correctness", "pcie": "PCIe link and bandwidth",
          "burst": "Power-swing burst test", "steady": "Sustained full-load soak"}


def fmt_dur(s):
    s = int(s)
    return "%dh %02dm %02ds" % (s // 3600, (s % 3600) // 60, s % 60) if s >= 3600 else "%dm %02ds" % (s // 60, s % 60)


def run_worker_set(run, test, gpus, params_for):
    """Start one worker per GPU (concurrently if several), wait, return {gpu: dict(status, worker...)}"""
    procs = {}
    for g in gpus:
        params = params_for(g)
        logf = open(os.path.join(run.outdir, "worker_gpu%d_%s.log" % (g, test)), "w")
        cmd = [sys.executable, os.path.abspath(__file__), "--worker", test, "--gpu", str(g), "--params", json.dumps(params)]
        p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=logf, text=True)
        procs[g] = (p, logf, time.time(), params)
    limit = max(float(par.get("seconds", 60)) for (_, _, _, par) in procs.values()) + 240 if test != "memtest" else 900 + 240
    if test == "memtest":
        limit = 3600
    res = {g: dict(status="pending") for g in gpus}
    outs = {g: "" for g in gpus}
    readers = {}

    def reader(g, p):
        outs[g] = p.stdout.read()
    for g, (p, lf, t0, par) in procs.items():
        th = threading.Thread(target=reader, args=(g, p), daemon=True)
        th.start()
        readers[g] = th
    t_start = time.time()
    last_status = 0
    while True:
        alive = [g for g, (p, _, _, _) in procs.items() if p.poll() is None]
        if not alive:
            break
        if run.tel.lost:
            lost_g = [g for g in run.tel.lost if g in procs]
            if lost_g and all(procs[g][0].poll() is not None or g in run.tel.lost for g in procs):
                pass
        if time.time() - t_start > limit:
            for g in alive:
                procs[g][0].kill()
                res[g] = dict(status="timeout", status_text="test did not finish within the time limit (GPU hang?)")
            break
        if time.time() - last_status >= 30:
            last_status = time.time()
            parts = []
            for g in gpus:
                s = run.tel.snap(g)
                if s:
                    parts.append("GPU%d %.0f W %d C %d MHz" % (g, s["power"], s["temp"], s["sm"]))
            log("  ... %s running, %s elapsed  |  %s" % (LABELS[test], fmt_dur(time.time() - t_start), "   ".join(parts)))
        time.sleep(1.0)
    for g, (p, lf, t0, par) in procs.items():
        try:
            p.wait(timeout=20)
        except Exception:  # noqa
            p.kill()
        readers[g].join(timeout=5)
        lf.close()
        text = outs[g]
        line = [l for l in text.splitlines() if l.startswith("RESULT_JSON:")]
        if res[g]["status"] == "timeout":
            continue
        if line:
            w = json.loads(line[-1][len("RESULT_JSON:"):])
            if w.get("status") == "ok":
                res[g] = dict(status="ok", worker=w)
            else:
                res[g] = dict(status="error", status_text="test program error: %s" % w.get("error"), worker=w)
        else:
            res[g] = dict(status="crash", status_text="test program ended without a result (exit code %s)" % p.returncode)
    return res


def collect_evidence(run, gpu):
    ed = os.path.join(run.outdir, "evidence")
    os.makedirs(ed, exist_ok=True)
    src, lines = read_klog()
    open(os.path.join(ed, "kernel_log_tail.txt"), "w").write("\n".join(lines[-400:]) + "\n")
    rc, out, err = run_cmd(["nvidia-smi", "-q"], 30)
    open(os.path.join(ed, "nvidia-smi_q_after_loss.txt"), "w").write(out + "\n" + err)
    bug = shutil.which("nvidia-bug-report.sh")
    if bug:
        cmd = [bug, "--output-file", os.path.join(ed, "nvidia-bug-report.log.gz")]
        if os.geteuid() != 0:
            cmd = ["sudo", "-n"] + cmd
        log("  collecting nvidia-bug-report (can take a few minutes) ...")
        run_cmd(cmd, 420)
    else:
        open(os.path.join(ed, "NOTE.txt"), "w").write(
            "nvidia-bug-report.sh was not found; please run it as root and send the output file.\n")


def orchestrate(a):
    global LOGF
    prof = dict(PROFILES[a.profile])
    for k in ("compute_s", "pcie_s", "burst_s", "steady_s", "limit_step_s"):
        prof[k] = max(10, int(prof[k] * a.scale))
    N = nvml()
    N.nvmlInit()
    count = N.nvmlDeviceGetCount()
    all_h = {i: N.nvmlDeviceGetHandleByIndex(i) for i in range(count)}
    if a.gpus == "all":
        sel = list(all_h)
    else:
        sel = [int(x) for x in a.gpus.split(",") if x.strip() != ""]
        for s in sel:
            if s not in all_h:
                print("GPU index %d does not exist (found %d GPU(s))" % (s, count))
                return 2
    handles = {i: all_h[i] for i in sel}
    info = collect_system_info(N, handles)

    # result names carry the serial number(s) of the tested GPU(s)
    tags = [re.sub(r"[^A-Za-z0-9]", "", str(g["serial"])) or "unknown" for g in info["gpus"]]
    tags = [("gpu%d" % g["index"] if t == "notreported" else t) for g, t in zip(info["gpus"], tags)]
    serial_tag = "_".join(tags[:4]) + ("_plus%d" % (len(tags) - 4) if len(tags) > 4 else "")
    stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = a.outdir or os.path.join(HERE, "results", "run_%s_%s" % (stamp, serial_tag))
    os.makedirs(outdir, exist_ok=True)
    LOGF = open(os.path.join(outdir, "run.log"), "w")
    run = Run()
    run.info, run.serial_tag = info, serial_tag
    run.outdir, run.profile, run.scale, run.concurrent = outdir, a.profile, a.scale, a.concurrent
    run.started = now_iso()
    run.results, run.events, run.overall, run.links = [], [], {}, {}
    run.ecc_before, run.ecc_after = {}, {}
    run.aborted = False
    run.settings_note = "No GPU or system settings were changed."
    t_begin = time.time()
    json.dump(run.info, open(os.path.join(outdir, "system_info.json"), "w"), indent=2)

    run.injected = bool(a.inject_fault)
    if run.injected:
        log("*** FAULT INJECTION TEST: one memory value is corrupted on purpose. A memory FAIL in this run is EXPECTED "
            "and does not mean the GPU is faulty. ***")
    log("V100 deep-dive test kit %s  |  profile %s  |  %d GPU(s) selected" % (VERSION, a.profile, len(sel)))
    log("OS: %s | kernel %s | driver %s | virtualization: %s" % (run.info["os"], run.info["kernel"],
                                                                  run.info["nvidia_driver"], run.info["virtualization"]))
    for g in run.info["gpus"]:
        log("GPU %d: %s | serial %s | %s MiB | PCIe Gen%s x%s (max Gen%s x%s) | power limit %s W (default %s W) | "
            "ECC %s | VBIOS %s" % (g["index"], g["name"], g["serial"], g["memory_total_mib"], g["pcie_gen_current"],
                                   g["pcie_width_current"], g["pcie_gen_max"], g["pcie_width_max"],
                                   g["power_limit_enforced_w"], g["power_limit_default_w"], g["ecc_mode"], g["vbios"]))
        if "V100" not in g["name"] and not a.any_gpu:
            log("NOTE: this GPU is not a V100 (%s); the tests still run." % g["name"])
        procs = nv_call(N.nvmlDeviceGetComputeRunningProcesses, handles[g["index"]], default=[])
        if procs:
            log("WARNING: %d other process(es) are using GPU %d; results may be affected." % (len(procs), g["index"]))

    run.klog_source, klog_start = read_klog()
    if run.klog_source is None:
        log("NOTE: the kernel log is not readable as this user, so Xid errors cannot be captured. Run the kit with "
            "sudo to enable this.")
    for i, h in handles.items():
        run.ecc_before[i] = ecc_counts(N, h)
        run.links[i] = dict(start="Gen%s x%s" % (run.info["gpus"][list(handles).index(i)]["pcie_gen_current"],
                                                  run.info["gpus"][list(handles).index(i)]["pcie_width_current"]),
                            load_min=None)

    tel = Telemetry(N, handles, outdir)
    run.tel = tel
    tel.start()
    all_klog_new = []

    # optional power-limit steps need root
    limits = [int(x) for x in a.power_limits.split(",") if x.strip()] if a.power_limits else []
    orig_limits = {}
    if limits:
        if os.geteuid() != 0:
            log("--power-limits needs root (sudo). Skipping the power-limit steps.")
            limits = []
        else:
            for i, h in handles.items():
                orig_limits[i] = nv_call(N.nvmlDeviceGetPowerManagementLimit, h)

            def restore_limits():
                for i, h in handles.items():
                    if orig_limits.get(i):
                        try:
                            N.nvmlDeviceSetPowerManagementLimit(h, orig_limits[i])
                        except Exception:  # noqa
                            pass
            atexit.register(restore_limits)
            run.settings_note = ("The optional power-limit steps temporarily changed the GPU power limit "
                                 "(%s W). The original limit was restored at the end." % ", ".join(map(str, limits)))

    gpu_dead = set()

    def do_test(test, gpus, params_for, label_suffix=""):
        gpus = [g for g in gpus if g not in gpu_dead]
        if not gpus:
            return
        phase = test + label_suffix
        tel.set_phase(phase)
        log("== %s%s  on GPU %s" % (LABELS[test], label_suffix, ",".join(map(str, gpus))))
        before_src, before = read_klog()
        eb = {g: ecc_counts(N, handles[g]) for g in gpus}
        t0 = time.time()
        res = run_worker_set(run, test, gpus, params_for)
        dur = time.time() - t0
        time.sleep(1.0)
        _, after = read_klog()
        nl = new_lines(before, after) if after else []
        all_klog_new.extend(nl)
        evs = parse_events(nl)
        for g in gpus:
            alive = gpu_alive(N, handles[g]) and g not in tel.lost
            ea = ecc_counts(N, handles[g])
            ed = {k: (ea[k] - eb[g][k]) if ea.get(k) is not None and eb[g].get(k) is not None else None for k in ea}
            st = tel.phase_stats(g, phase)
            gi = run.info["gpus"][list(handles).index(g)]
            r = dict(test=test, label=LABELS[test] + label_suffix, gpu=g, status=res[g]["status"],
                     status_text=res[g].get("status_text"), worker=res[g].get("worker"), tel=st, alive=alive,
                     klog_events=evs, ecc_delta=ed, gpuinfo=gi, duration_text=fmt_dur(dur))
            verdict, notes = judge_test(r)
            r.update(verdict=verdict, notes=notes, summary=build_summary(test, res[g].get("worker") or {}, gi))
            snap = tel.snap(g)
            cur = "Gen%s x%s" % (nv_call(N.nvmlDeviceGetCurrPcieLinkGeneration, handles[g]),
                                 nv_call(N.nvmlDeviceGetCurrPcieLinkWidth, handles[g]))
            run.links[g]["end"] = cur
            if test in ("steady", "burst"):
                run.links[g].setdefault("load_seen", set()).add(cur)
            run.results.append(r)
            log("   GPU %d: %s  (%s)  %s" % (g, verdict, fmt_dur(dur), r["summary"]))
            for n in notes:
                log("      - " + n)
            for e in evs:
                e = dict(e, when=now_iso())
                run.events.append(e)
            if not alive:
                gpu_dead.add(g)
                log("   !!! GPU %d STOPPED RESPONDING. Remaining tests for this GPU are skipped. The machine "
                    "must be fully powered off before this GPU works again." % g)
                collect_evidence(run, g)
                run.events.append(dict(kind="gpu_lost", when=now_iso(),
                                       text="GPU %d stopped responding during phase '%s'" % (g, phase)))

    try:
        groups = [sel] if a.concurrent else [[g] for g in sel]
        for grp in groups:
            do_test("pcie", grp, lambda g: dict(seconds=prof["pcie_s"]))
            do_test("compute", grp, lambda g: dict(seconds=prof["compute_s"]))
            do_test("memtest", grp, lambda g: dict(passes=prof["mem_passes"], inject_fault=(a.inject_fault == "memtest")))
            do_test("burst", grp, lambda g: dict(seconds=prof["burst_s"], seed=7 + g))
            for lim in limits:
                for g in grp:
                    if g in gpu_dead:
                        continue
                    nv_call(N.nvmlDeviceSetPowerManagementLimit, handles[g], lim * 1000)
                do_test("burst", grp, lambda g: dict(seconds=prof["limit_step_s"], seed=70 + g),
                        label_suffix=" at %d W limit" % lim)
            if limits:
                for g in grp:
                    if orig_limits.get(g):
                        nv_call(N.nvmlDeviceSetPowerManagementLimit, handles[g], orig_limits[g])
            do_test("steady", grp, lambda g: dict(seconds=prof["steady_s"]))
    except KeyboardInterrupt:
        run.aborted = True
        log("Interrupted by the user (Ctrl+C). Writing the partial report ...")
    except Exception as e:  # noqa
        run.aborted = True
        log("The test run stopped because of an unexpected error: %s" % e)
        log(traceback.format_exc())

    tel.set_phase("end")
    time.sleep(1.5)
    for i, h in handles.items():
        run.ecc_after[i] = ecc_counts(N, h)
        if "end" not in run.links[i]:
            run.links[i]["end"] = "?"
        cur = "Gen%s x%s" % (nv_call(N.nvmlDeviceGetCurrPcieLinkGeneration, h), nv_call(N.nvmlDeviceGetCurrPcieLinkWidth, h))
        run.links[i]["end"] = cur
    tel.stop()
    # PCIe under-load summary from telemetry csv
    for i in handles:
        seen = []
        try:
            for row in csv.DictReader(open(os.path.join(outdir, "telemetry_gpu%d.csv" % i))):
                if row["phase"].startswith(("burst", "steady")) and row["pcie_gen"] and row["pcie_width"]:
                    seen.append((int(row["pcie_gen"]), int(row["pcie_width"])))
        except Exception:  # noqa
            pass
        if seen:
            low = min(seen, key=lambda x: (x[1], x[0]))
            high = max(seen, key=lambda x: (x[1], x[0]))
            run.links[i]["load_min"] = "Gen%d x%d (highest seen Gen%d x%d)" % (low[0], low[1], high[0], high[1])
            gi = run.info["gpus"][list(handles).index(i)]
            if gi["pcie_width_max"] and low[1] < gi["pcie_width_max"]:
                for r in run.results:
                    if r["gpu"] == i and r["test"] in ("burst", "steady") and r["verdict"] == "PASS":
                        r["verdict"] = "WARN"
                        r["notes"].append("WARN: PCIe link width during load was x%d, below the maximum x%d "
                                          "(slot, riser or cable problem?)" % (low[1], gi["pcie_width_max"]))

    for g in sel:
        vs = [r["verdict"] for r in run.results if r["gpu"] == g]
        run.overall[g] = "FAIL" if "FAIL" in vs else ("WARN" if "WARN" in vs else ("PASS" if vs else "n/a"))
        if run.aborted and run.overall[g] == "PASS":
            run.overall[g] = "INCOMPLETE"
    if klog_start is not None:
        open(os.path.join(outdir, "kernel_log_new_lines.txt"), "w").write("\n".join(all_klog_new) + "\n")
    run.finished = now_iso()
    run.duration_text = fmt_dur(time.time() - t_begin)

    ser = []
    for r in run.results:
        x = {k: v for k, v in r.items() if k not in ("gpuinfo",)}
        ser.append(x)
    json.dump(dict(kit_version=VERSION, profile=a.profile, scale=a.scale, started=run.started, finished=run.finished,
                   aborted=run.aborted, overall={str(k): v for k, v in run.overall.items()}, results=ser,
                   events=run.events, links=run.links, ecc_before=run.ecc_before, ecc_after=run.ecc_after),
              open(os.path.join(outdir, "results_%s.json" % run.serial_tag), "w"), indent=2, default=lambda o: sorted(o) if isinstance(o, set) else str(o))
    write_report(run, outdir)
    log("")
    log("RESULT: " + "   ".join("GPU %d: %s" % (g, v) for g, v in run.overall.items()))
    log("Report: %s" % os.path.join(outdir, "REPORT_%s.md" % run.serial_tag))
    LOGF.close()
    LOGF = None
    if not a.no_archive:
        base = os.path.basename(outdir.rstrip("/"))
        arch = os.path.join(os.path.dirname(outdir.rstrip("/")), base + ".tar.gz")
        with tarfile.open(arch, "w:gz") as tf:
            tf.add(outdir, arcname=base)
        h = hashlib.sha256(open(arch, "rb").read()).hexdigest()
        open(arch + ".sha256", "w").write("%s  %s\n" % (h, os.path.basename(arch)))
        print("")
        print("=" * 70)
        print(" WHAT TO DO NEXT")
        print(" Send BOTH files, together, to the person who asked you to run the test:")
        print("")
        print("   1. %s" % arch)
        print("   2. %s.sha256" % arch)
        print("")
        print(" The first holds the report, logs and measurements. The second is the")
        print(" checksum that proves the first was not changed. Nothing is uploaded")
        print(" automatically. SHA-256: %s" % h)
        print(" Read the short summary first in: %s" % os.path.join(outdir, "REPORT_%s.md" % run.serial_tag))
        print("=" * 70)
    return 1 if any(v == "FAIL" for v in run.overall.values()) else 0


# --------------------------------------------------------------------------------------
# self test (no GPU needed): parser, decoder, report writer
# --------------------------------------------------------------------------------------
def selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("PASS  " if cond else "FAIL  ") + name)
        ok = ok and cond
    ev = parse_events(["[  89.8] NVRM: Xid (PCI:0000:01:00): 79, pid='<unknown>', name=<unknown>, GPU has fallen off the bus.",
                       "[  90.0] NVRM: GPU 0000:01:00.0: GPU has fallen off the bus.", "unrelated line"])
    check("Xid 79 line is recognised", any(e["kind"] == "xid" and e["xid"] == 79 for e in ev))
    check("GPU lost line is recognised", any(e["kind"] == "gpu_lost" for e in ev))
    check("unrelated kernel lines are ignored", len(ev) == 2)
    check("throttle decoder", decode_throttle(0x24) == "sw_power_cap+sw_thermal_slowdown" and decode_throttle(0) == "none")
    gi = dict(temp_slowdown_c=87, temp_max_operating_c=83, pcie_gen_max=3, pcie_width_max=16)
    base = dict(test="memtest", status="ok", worker=dict(errors=0), tel=dict(t_max=70, thr=0, swings100=0), alive=True,
                klog_events=[], ecc_delta={}, gpuinfo=gi)
    check("clean memory test passes", judge_test(base)[0] == "PASS")
    check("memory errors fail", judge_test(dict(base, worker=dict(errors=3)))[0] == "FAIL")
    check("lost GPU fails", judge_test(dict(base, alive=False))[0] == "FAIL")
    check("Xid fails", judge_test(dict(base, klog_events=[dict(kind="xid", xid=79)]))[0] == "FAIL")
    check("hot GPU warns", judge_test(dict(base, tel=dict(t_max=84, thr=0, swings100=0)))[0] == "WARN")
    check("very hot GPU fails", judge_test(dict(base, tel=dict(t_max=88, thr=0, swings100=0)))[0] == "FAIL")
    check("unfinished test fails", judge_test(dict(base, status="timeout", status_text="x"))[0] == "FAIL")
    check("weak burst test warns", judge_test(dict(base, test="burst", worker=dict(verify_mismatches=0),
                                                    tel=dict(t_max=60, thr=0, swings100=1)))[0] == "WARN")
    check("uncorrectable ECC increase fails", judge_test(dict(base, ecc_delta=dict(uncorrected_volatile=1)))[0] == "FAIL")
    print("\nSELFTEST " + ("OK" if ok else "FAILED"))
    return 0 if ok else 1


def main():
    ap = argparse.ArgumentParser(description="V100 deep-dive diagnostic engine (use run_v100_diag.sh)")
    ap.add_argument("--profile", default="standard", choices=sorted(PROFILES))
    ap.add_argument("--gpus", default="all", help='"all" or a comma list of GPU indexes, e.g. 0,1')
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--scale", type=float, default=1.0, help="multiply all test durations (e.g. 0.5 or 2)")
    ap.add_argument("--concurrent", action="store_true", help="test all selected GPUs at the same time")
    ap.add_argument("--power-limits", default="", help="optional, needs root: extra burst steps at these limits in W, e.g. 150,200")
    ap.add_argument("--any-gpu", action="store_true")
    ap.add_argument("--no-archive", action="store_true")
    ap.add_argument("--inject-fault", default="", help="validation aid: 'memtest' corrupts one value to prove detection")
    ap.add_argument("--selftest", action="store_true")
    ap.add_argument("--worker")
    ap.add_argument("--gpu", type=int)
    ap.add_argument("--params")
    a = ap.parse_args()
    if a.selftest:
        sys.exit(selftest())
    if a.worker:
        worker_main(a.worker, a.gpu, json.loads(a.params or "{}"))
        return
    sys.exit(orchestrate(a))


if __name__ == "__main__":
    main()

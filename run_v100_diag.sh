#!/usr/bin/env bash
# V100 deep-dive diagnostic kit - launcher.
# Checks the prerequisites, tells you what is missing, asks before installing anything,
# then runs the tests and writes logs plus a test report into ./results/.
#
# Usage: ./run_v100_diag.sh [options]      (see --help)
set -u

KIT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VENV="$KIT_DIR/.venv"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cu124}"
MIN_FREE_GB=8

PROFILE="standard"
ASSUME_YES=0
CHECK_ONLY=0
INSTALL_ONLY=0
ENGINE_ARGS=()

usage() {
cat <<'EOF'
V100 deep-dive diagnostic kit

  ./run_v100_diag.sh [options]

Options:
  --profile quick|standard|full   test length (default standard)
                                    quick    about 10 minutes
                                    standard about 40 minutes per GPU
                                    full     about 2 hours per GPU
  --gpus all|0,1                  which GPUs to test (default all)
  --concurrent                    test all selected GPUs at the same time
  --scale N                       multiply every test duration (e.g. 0.5 or 2)
  --power-limits 150,200          extra burst steps at these power limits in W (needs root)
  --outdir DIR                    result folder (default ./results/run_<date>)
  --no-archive                    do not create the .tar.gz
  --yes                           do not ask; install missing prerequisites and start
  --check-only                    only check prerequisites, change nothing
  --install-only                  check and install prerequisites, then stop without testing
  -h, --help                      this text

The kit never changes the NVIDIA driver. Everything else it installs is listed first and
needs your confirmation (unless --yes is given).
EOF
}

while [ $# -gt 0 ]; do
  case "$1" in
    --profile) PROFILE="$2"; ENGINE_ARGS+=(--profile "$2"); shift 2 ;;
    --gpus|--scale|--power-limits|--outdir|--inject-fault) ENGINE_ARGS+=("$1" "$2"); shift 2 ;;
    --concurrent|--no-archive|--any-gpu) ENGINE_ARGS+=("$1"); shift ;;
    --yes|-y) ASSUME_YES=1; shift ;;
    --check-only) CHECK_ONLY=1; shift ;;
    --install-only) INSTALL_ONLY=1; shift ;;
    -h|--help) usage; exit 0 ;;
    *) echo "Unknown option: $1"; usage; exit 2 ;;
  esac
done
case "$PROFILE" in quick|standard|full) ;; *) echo "Unknown profile: $PROFILE"; exit 2 ;; esac

say()  { printf '%s\n' "$*"; }
ok()   { printf '  [ ok ]   %s\n' "$*"; }
miss() { printf '  [MISSING] %s\n' "$*"; }
warn() { printf '  [ warn ] %s\n' "$*"; }

ask() {  # ask "question"  -> 0 yes, 1 no
  [ "$ASSUME_YES" -eq 1 ] && return 0
  if [ ! -t 0 ]; then
    say "Not an interactive terminal and --yes was not given, so nothing was installed."
    return 1
  fi
  local a
  read -r -p "$1 [y/N] " a
  case "$a" in y|Y|yes|YES) return 0 ;; *) return 1 ;; esac
}

SUDO=""
if [ "$(id -u)" -ne 0 ]; then
  if command -v sudo >/dev/null 2>&1; then SUDO="sudo"; else SUDO="NOSUDO"; fi
fi

say "=============================================================="
say " V100 deep-dive diagnostic kit - prerequisite check"
say "=============================================================="

# ---- operating system -------------------------------------------------------------------
if [ -r /etc/os-release ]; then
  # shellcheck disable=SC1091
  . /etc/os-release
  case " ${ID:-} ${ID_LIKE:-} " in
    *" debian "*|*" ubuntu "*) ok "Operating system: ${PRETTY_NAME:-$ID} (Debian family)" ;;
    *) warn "Operating system ${PRETTY_NAME:-unknown} is not Debian based. The kit was written for Debian/Ubuntu; install the equivalents of the packages below yourself." ;;
  esac
fi
if [ "$(uname -m)" != "x86_64" ]; then warn "CPU architecture is $(uname -m); the PyTorch download used here is for x86_64."; fi

# ---- NVIDIA driver (never installed by this kit) ----------------------------------------
DRIVER_OK=1
if command -v nvidia-smi >/dev/null 2>&1; then
  if nvidia-smi -L >/dev/null 2>&1; then
    NGPU="$(nvidia-smi -L | grep -c '^GPU')"
    DRV="$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"
    ok "NVIDIA driver $DRV, $NGPU GPU(s) visible:"
    nvidia-smi --query-gpu=index,name,memory.total,serial --format=csv,noheader | sed 's/^/             GPU /'
  else
    miss "nvidia-smi is installed but cannot talk to the driver (module not loaded, GPU lost, or a reboot is needed)"
    DRIVER_OK=0
  fi
else
  miss "NVIDIA driver (nvidia-smi not found)"
  DRIVER_OK=0
fi
if [ "$DRIVER_OK" -eq 0 ]; then
  cat <<'EOF'

The NVIDIA driver is not working, and this kit does not install drivers (a driver change on
a machine is too invasive to do silently). On Debian, install it yourself, for example:

  sudo apt update
  sudo apt install linux-headers-$(uname -r) nvidia-driver firmware-misc-nonfree
  sudo reboot

(Debian needs the "non-free" component enabled in the apt sources first; Ubuntu: use
"sudo ubuntu-drivers install".) A V100 works with any current driver from the 535 series up.
Then run this script again.
EOF
  exit 3
fi

# ---- other processes on the GPUs --------------------------------------------------------
BUSY="$(nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader 2>/dev/null)"
if [ -n "$BUSY" ]; then
  warn "Other processes are using the GPU(s); they will distort the results. Stop them first:"
  printf '%s\n' "$BUSY" | sed 's/^/             /'
fi

# ---- apt packages -----------------------------------------------------------------------
MISS_APT=()
need_pkg() {  # need_pkg <package> <why>
  if dpkg -s "$1" >/dev/null 2>&1; then ok "package $1 - $2"; else miss "package $1 - $2"; MISS_APT+=("$1"); fi
}
if command -v dpkg >/dev/null 2>&1; then
  need_pkg python3 "runs the test engine"
  need_pkg python3-venv "isolated Python environment for the kit"
  need_pkg python3-pip "installs the Python libraries"
  need_pkg pciutils "lspci, identifies the GPU on the PCIe bus"
  need_pkg ca-certificates "secure download of the Python libraries"
else
  warn "dpkg not found; skipping the package check"
fi
if ! command -v nvidia-bug-report.sh >/dev/null 2>&1; then
  warn "nvidia-bug-report.sh not found (optional: collects extra evidence if a GPU is lost). Package: nvidia-driver-bin or nvidia-utils-<ver>."
else
  ok "nvidia-bug-report.sh - evidence collection if a GPU is lost"
fi

# ---- python libraries -------------------------------------------------------------------
NEED_PIP=0
if [ -x "$VENV/bin/python" ] && "$VENV/bin/python" -c "import torch, pynvml, numpy" >/dev/null 2>&1; then
  ARCHS="$("$VENV/bin/python" -c 'import torch;print(" ".join(torch.cuda.get_arch_list()))' 2>/dev/null)"
  ok "Python libraries in $VENV (torch, nvidia-ml-py); CUDA architectures: $ARCHS"
else
  miss "Python libraries torch (about 2.5 GB download), numpy and nvidia-ml-py, into $VENV"
  NEED_PIP=1
fi

# ---- disk space -------------------------------------------------------------------------
FREE_GB="$(df -Pk "$KIT_DIR" | awk 'NR==2{printf "%d", $4/1048576}')"
if [ "$NEED_PIP" -eq 1 ] && [ "${FREE_GB:-0}" -lt "$MIN_FREE_GB" ]; then
  warn "Only ${FREE_GB} GB free where the kit lives; about $MIN_FREE_GB GB are needed for the libraries."
fi

# ---- privileges for the kernel log ------------------------------------------------------
if [ "$(id -u)" -eq 0 ]; then
  ok "running as root: kernel log (Xid events) and bug report are available"
elif dmesg >/dev/null 2>&1; then
  ok "kernel log readable as this user"
else
  warn "the kernel log is not readable as this user, so Xid / \"fell off the bus\" events cannot be captured. Run the whole script with sudo for the full test."
fi

# ---- summary and install ----------------------------------------------------------------
if [ "${#MISS_APT[@]}" -gt 0 ] || [ "$NEED_PIP" -eq 1 ]; then
  say ""
  say "To run the tests this kit needs to install:"
  [ "${#MISS_APT[@]}" -gt 0 ] && say "  - system packages (apt): ${MISS_APT[*]}"
  [ "$NEED_PIP" -eq 1 ] && say "  - Python libraries into the private folder $VENV (nothing outside this folder is touched), from $TORCH_INDEX_URL and pypi.org"
  say "The NVIDIA driver and your system settings are not changed."
  if [ "$CHECK_ONLY" -eq 1 ]; then say "(--check-only: nothing installed)"; exit 4; fi
  if ! ask "Install the missing parts now?"; then
    say ""
    say "Nothing was installed. To do it by hand:"
    [ "${#MISS_APT[@]}" -gt 0 ] && say "  sudo apt update && sudo apt install -y ${MISS_APT[*]}"
    [ "$NEED_PIP" -eq 1 ] && say "  python3 -m venv $VENV && $VENV/bin/pip install torch --index-url $TORCH_INDEX_URL && $VENV/bin/pip install nvidia-ml-py"
    exit 4
  fi
  if [ "${#MISS_APT[@]}" -gt 0 ]; then
    if [ "$SUDO" = "NOSUDO" ]; then say "sudo is not available and you are not root; install ${MISS_APT[*]} as root and run again."; exit 5; fi
    $SUDO apt-get update || { say "apt-get update failed"; exit 5; }
    $SUDO apt-get install -y "${MISS_APT[@]}" || { say "apt install failed"; exit 5; }
  fi
  if [ "$NEED_PIP" -eq 1 ]; then
    [ -x "$VENV/bin/python" ] || python3 -m venv "$VENV" || { say "could not create the Python environment"; exit 5; }
    "$VENV/bin/pip" install --upgrade pip >/dev/null 2>&1
    # Try the preferred index first; newer Python versions (e.g. 3.14 on the latest Ubuntu) have no wheels
    # there, so fall back to newer CUDA builds that still contain kernels for Volta (sm_70).
    TORCH_OK=0
    for idx in "$TORCH_INDEX_URL" https://download.pytorch.org/whl/cu126 https://download.pytorch.org/whl/cu128; do
      say "Installing torch from $idx ..."
      if "$VENV/bin/pip" install torch --index-url "$idx" \
         && "$VENV/bin/python" -c "import torch,sys;sys.exit(0 if torch.cuda.is_available() and any(a in torch.cuda.get_arch_list() for a in ('sm_70','compute_70')) else 1)" 2>/dev/null; then
        TORCH_OK=1; break
      fi
      say "That build is not usable here; trying the next one."
      "$VENV/bin/pip" uninstall -y torch >/dev/null 2>&1
    done
    if [ "$TORCH_OK" -ne 1 ]; then
      say "Could not install a PyTorch build that works with this GPU and driver. Check nvidia-smi and the Python version ($(python3 --version)), or set TORCH_INDEX_URL."; exit 5
    fi
    "$VENV/bin/pip" install nvidia-ml-py numpy || { say "nvidia-ml-py installation failed"; exit 5; }
  fi
  say "Prerequisites installed."
fi
if [ "$CHECK_ONLY" -eq 1 ] || [ "$INSTALL_ONLY" -eq 1 ]; then say ""; say "All prerequisites are in place."; exit 0; fi

# ---- GPU architecture support check ------------------------------------------------------
if ! "$VENV/bin/python" - <<'EOF'
import sys, torch
cc = [torch.cuda.get_device_capability(i) for i in range(torch.cuda.device_count())]
archs = torch.cuda.get_arch_list()
bad = [c for c in cc if "sm_%d%d" % c not in archs and "compute_%d%d" % c not in archs]
if bad:
    print("The installed PyTorch has no kernels for GPU capability %s (has %s)." % (bad, archs)); sys.exit(1)
EOF
then
  say "Install a PyTorch build that supports this GPU (TORCH_INDEX_URL=... and remove $VENV), then run again."; exit 6
fi

# ---- final confirmation and run ----------------------------------------------------------
case "$PROFILE" in
  quick) DUR="about 10 minutes" ;; standard) DUR="about 40 minutes per GPU" ;; full) DUR="about 2 hours per GPU" ;;
esac
say ""
say "Ready to test (profile $PROFILE, $DUR). The GPU will run at full load, including rapid"
say "load/idle power swings. Please do not run other GPU work, and do not touch the hardware, during the test."
say "If a GPU stops responding the kit records the evidence and stops testing that GPU;"
say "the machine must then be fully powered off (not just rebooted) before it is used again."
if ! ask "Start the test now?"; then say "Cancelled."; exit 0; fi

PYARGS=("${ENGINE_ARGS[@]}")
"$VENV/bin/python" "$KIT_DIR/v100_diag.py" "${PYARGS[@]}"
RC=$?

if [ -n "${SUDO_USER:-}" ] && [ -d "$KIT_DIR/results" ]; then chown -R "$SUDO_USER" "$KIT_DIR/results" 2>/dev/null; fi
say ""
case "$RC" in
  0) say "Test finished: no failures. Send the .tar.gz file AND its .sha256 file (both shown above) to the person who asked for the test." ;;
  1) say "Test finished: at least one FAILURE was found. Send the .tar.gz file AND its .sha256 file (both shown above) to the person who asked for the test." ;;
  *) say "The test did not run to completion (exit code $RC). Send the whole results folder (zip or tar it) to the person who asked for the test." ;;
esac
exit "$RC"

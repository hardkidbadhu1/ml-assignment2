#!/usr/bin/env bash
#
# One-shot run for the BITS Virtual Lab (AWS VM). Produces the screenshot that is
# worth 1 mark, and leaves behind artifacts trained *in the lab environment*.
#
#   bash scripts/lab_run.sh
#
# Why this exists rather than "just run train_fifa.sh": the mark is for evidence
# that the work was performed in the lab, so the output has to make the machine
# identifiable. The banner below prints the hostname, the EC2 instance identity
# and the library versions alongside the metrics, so a single screenshot carries
# both the result and the proof of where it ran.
#
set -uo pipefail
cd "$(dirname "$0")/.."
REPO="$(pwd)"
DATA="data/fifa_world_cup_2026_player_performance.csv"

bold() { printf '\033[1m%s\033[0m\n' "$*"; }
rule() { printf '%s\n' "────────────────────────────────────────────────────────────────────────"; }
die()  { printf '\033[1;31mERROR:\033[0m %s\n' "$*" >&2; exit 1; }

rule
bold " BITS VIRTUAL LAB — ML ASSIGNMENT 2"
bold " Badhmanaban M / 2025AC05386"
rule

# --------------------------------------------------------------------------- #
# 1. Environment identity. This is the part that makes the screenshot evidence.
# --------------------------------------------------------------------------- #
bold "[1/5] Environment"
echo "  date      : $(date -u '+%Y-%m-%d %H:%M:%S UTC')"
echo "  hostname  : $(hostname)"
echo "  user      : $(whoami)"
echo "  kernel    : $(uname -srm)"
if command -v lsb_release >/dev/null 2>&1; then
    echo "  distro    : $(lsb_release -ds 2>/dev/null)"
elif [ -r /etc/os-release ]; then
    echo "  distro    : $(. /etc/os-release && echo "$PRETTY_NAME")"
fi
echo "  cpus      : $(nproc 2>/dev/null || echo '?')   mem: $(free -h 2>/dev/null | awk '/^Mem:/{print $2}' || echo '?')"

# Interpreter selection. Rocky/RHEL 9 ships Python 3.9 as the system `python3`,
# but scikit-learn 1.7.2 and numpy 2.2.6 both declare requires_python >= 3.10.
# On 3.9 pip does not error loudly — it silently resolves to whatever old
# scikit-learn still supports 3.9, and you get artifacts that do not match the
# pins in requirements.txt. Pick a new-enough interpreter explicitly instead.
PYBIN=""
for cand in python3.12 python3.11 python3.10 python3; do
    command -v "$cand" >/dev/null 2>&1 || continue
    ver="$("$cand" -c 'import sys; print("%d%02d" % sys.version_info[:2])' 2>/dev/null)" || continue
    if [ -n "$ver" ] && [ "$ver" -ge 310 ] 2>/dev/null; then PYBIN="$cand"; break; fi
done
if [ -z "$PYBIN" ]; then
    die "no Python >= 3.10 found (system python3 is $(python3 -V 2>&1 | awk '{print $2}')).

scikit-learn 1.7.2 and numpy 2.2.6 require >= 3.10. On Rocky Linux 9, AppStream
has 3.12 — matching runtime.txt, so the lab and Streamlit Cloud agree:

    sudo dnf install -y python3.12 python3.12-pip
    bash scripts/lab_run.sh

(Do not 'fix' this by loosening requirements.txt: the artifacts must be produced
by the same scikit-learn the deployed app pins.)"
fi
echo "  python    : $("$PYBIN" --version 2>&1)  [$(command -v "$PYBIN")]"

# EC2 Instance Metadata Service. IMDSv2 needs a token first; fall back to v1.
# Short timeouts so this is a no-op off EC2 rather than a 30-second hang.
IMDS="http://169.254.169.254/latest"
TOKEN="$(curl -s -X PUT "$IMDS/api/token" \
    -H 'X-aws-ec2-metadata-token-ttl-seconds: 60' --max-time 2 2>/dev/null || true)"
if [ -n "$TOKEN" ]; then
    AUTH=(-H "X-aws-ec2-metadata-token: $TOKEN")
else
    AUTH=()
fi
IID="$(curl -s "${AUTH[@]}" --max-time 2 "$IMDS/meta-data/instance-id" 2>/dev/null || true)"
# Match the actual instance-id shape rather than merely testing for non-empty: an
# intercepting proxy will happily return a 200 with an error string in the body,
# which would otherwise get printed as if it were the instance id.
if [[ "$IID" =~ ^i-[0-9a-f]+$ ]]; then
    ITYPE="$(curl -s "${AUTH[@]}" --max-time 2 "$IMDS/meta-data/instance-type" 2>/dev/null || echo '?')"
    IAZ="$(curl -s "${AUTH[@]}" --max-time 2 "$IMDS/meta-data/placement/availability-zone" 2>/dev/null || echo '?')"
    echo "  ec2       : $IID  ($ITYPE, $IAZ)"
else
    echo "  ec2       : instance metadata unavailable (not on EC2, or IMDS blocked)"
fi
echo "  repo      : $REPO"
echo "  git       : $(git rev-parse --short HEAD 2>/dev/null || echo 'not a git checkout')"
echo

# --------------------------------------------------------------------------- #
# 2. Dependencies
# --------------------------------------------------------------------------- #
bold "[2/5] Dependencies"
# Install into a virtualenv rather than the system interpreter. On RHEL-family
# distros dnf-managed site-packages are owned by root, so a plain `pip install`
# either needs sudo or half-installs into ~/.local and shadows the system copies.
# A venv sidesteps both, and sidesteps PEP 668 on Debian-family too.
VENV="${VENV_DIR:-.venv}"
PY=""

# Judge the venv by whether it actually works, not by the exit code of `venv`.
# A run that dies partway through ensurepip leaves a directory behind that looks
# plausible and makes the *next* `python -m venv` return 0 without ever producing
# an activate script — so probe for a usable pip instead.
venv_usable() { [ -x "$VENV/bin/python" ] && [ -f "$VENV/bin/activate" ] &&
                "$VENV/bin/python" -m pip --version >/dev/null 2>&1; }

if ! venv_usable; then
    if [ -e "$VENV" ]; then
        echo "  venv      : removing unusable $VENV"
        # Non-fatal: on a read-only or root-owned path this fails, and the
        # venv_usable re-check below then routes us to the --user fallback
        # rather than leaving the run wedged.
        rm -rf "$VENV" 2>/dev/null || true
    fi
    "$PYBIN" -m venv "$VENV" >/tmp/venv.log 2>&1 || true
fi

if venv_usable; then
    # shellcheck disable=SC1091
    source "$VENV/bin/activate"
    PY=python
    echo "  venv      : ${VIRTUAL_ENV:-$VENV}"
else
    # `python -m venv` fails when the distro has split ensurepip into a separate
    # package (python3-venv on Debian-family, occasionally absent on minimal RHEL
    # images). Falling back to a --user install into ~/.local keeps the run going
    # without sudo, rather than stopping the whole thing over packaging trivia.
    echo "  venv      : unavailable, falling back to 'pip install --user'"
    echo "              ($(tail -1 /tmp/venv.log 2>/dev/null | cut -c1-90))"
    echo "              to get a clean venv instead:  sudo dnf install -y ${PYBIN}-devel"
    PY="$PYBIN"
    export PIP_USER=1
fi
echo "  python    : $($PY --version 2>&1)"
$PY -m pip install -q --upgrade pip >/dev/null 2>&1 || true
$PY -m pip install -q -r requirements.txt || die "pip install failed.

If pip itself is missing for this interpreter:
    sudo dnf install -y ${PYBIN}-pip        # Rocky / RHEL 9
    sudo apt install -y ${PYBIN}-venv       # Debian / Ubuntu"
$PY - <<'PY'
import importlib, sys
mods = ["sklearn", "pandas", "numpy", "joblib", "matplotlib", "seaborn", "streamlit"]
for m in mods:
    try:
        v = getattr(importlib.import_module(m), "__version__", "?")
        print(f"  {m:<12} {v}")
    except ImportError:
        print(f"  {m:<12} MISSING")
        sys.exit(1)
PY
[ $? -eq 0 ] || die "a required package is missing"
echo

# --------------------------------------------------------------------------- #
# 3. Dataset. It is gitignored (17 MB), so a fresh clone will not have it.
# --------------------------------------------------------------------------- #
bold "[3/5] Dataset"
if [ ! -f "$DATA" ]; then
    echo "  $DATA not found — trying the Kaggle CLI."
    if command -v kaggle >/dev/null 2>&1 && [ -f "$HOME/.kaggle/kaggle.json" ]; then
        mkdir -p data
        kaggle datasets download -d "${KAGGLE_DATASET:?set KAGGLE_DATASET=owner/dataset-slug}" \
            -p data --unzip || die "kaggle download failed"
    else
        die "dataset missing and the Kaggle CLI is not configured.

Pick whichever is least friction:

  (a) copy it up from your laptop
        scp \"$DATA\" <user>@<lab-host>:$REPO/data/

  (b) configure the Kaggle CLI here, then re-run this script
        pip install kaggle
        mkdir -p ~/.kaggle && nano ~/.kaggle/kaggle.json   # paste your API token
        chmod 600 ~/.kaggle/kaggle.json
        export KAGGLE_DATASET=<owner>/<dataset-slug>

  (c) fetch it with a direct link
        curl -L -o \"$DATA\" '<url>'"
    fi
fi
ROWS=$(($(wc -l < "$DATA") - 1))
COLS=$(head -1 "$DATA" | awk -F, '{print NF}')
echo "  file      : $DATA"
echo "  size      : $(du -h "$DATA" | cut -f1)"
echo "  shape     : ${ROWS} rows x ${COLS} columns   (requires >= 500 rows, >= 12 features)"
echo "  sha256    : $(sha256sum "$DATA" | cut -c1-16)…"
echo

# --------------------------------------------------------------------------- #
# 4. Train. LAB_MAX_ROWS lets you shrink the run if the VM is small — SVC is
#    superlinear in n, so it is the only thing that meaningfully moves runtime.
# --------------------------------------------------------------------------- #
bold "[4/5] Training all six models"
START=$(date +%s)
if [ -n "${LAB_MAX_ROWS:-}" ]; then
    echo "  (LAB_MAX_ROWS=$LAB_MAX_ROWS — overriding the default 15000)"
    sed "s/--max-rows 15000/--max-rows $LAB_MAX_ROWS/" scripts/train_fifa.sh | bash \
        || die "training failed"
else
    bash scripts/train_fifa.sh || die "training failed"
fi
echo "  elapsed   : $(( $(date +%s) - START ))s"
echo

# --------------------------------------------------------------------------- #
# 5. Verify
# --------------------------------------------------------------------------- #
bold "[5/5] Verification"
$PY scripts/verify.py || die "verification failed — do not submit these artifacts"
echo

rule
bold " RESULTS — trained on $(hostname)"
rule
$PY - <<'PY'
import pandas as pd
df = pd.read_csv("reports/metrics.csv", index_col=0)
cols = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]
print(df[cols].round(4).to_string())
print()
best = df["MCC"].idxmax()
print(f"Best by MCC: {best}  (MCC={df.loc[best, 'MCC']:.4f}, Accuracy={df.loc[best, 'Accuracy']:.4f})")
PY
rule
bold " SCREENSHOT THIS WINDOW"
echo " It shows the hostname and EC2 instance id above, the six models with all"
echo " six metrics, and every verification check passing — the evidence and the"
echo " result in one frame. Scroll up if the environment block is off-screen."
rule

#!/usr/bin/env bash
# Camera-ready build for ASCI EGCL paper.
#
# Runs in order:
#   0. Generate/refresh camera_ready_numbers.json from ledgers
#   1. verify_safenum.py: all \AggNum{key}{value} checked against manifest
#   2. Anti-Fabrication Guard against main_camera_ready.tex
#   3. Enforcement-flag status check (all five enforcement layers active)
#   4. LaTeX 4-pass build (main_camera_ready.tex -> main_camera_ready.pdf)
#   5. Page-budget verification (body <= 9 pages)
#   6. Single-blind verification (authors visible)
#
# Usage: bash papers/p5-asci/build_camera_ready.sh
# Exit codes: 0 = built, non-zero = failed.

set -euo pipefail

PAPER_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$PAPER_DIR/../.." && pwd)"
MANUSCRIPT="$PAPER_DIR/main_camera_ready.tex"
MANUSCRIPT_REL="papers/p5-asci/main_camera_ready.tex"   # repo-relative, keeps logs path-free
PDF="$PAPER_DIR/main_camera_ready.pdf"

cd "$REPO_ROOT"

log()  { printf '[build-cr] %s\n' "$*"; }
fail() { printf '[build-cr] FAIL: %s\n' "$*" >&2; exit 1; }

# --------------------------------------------------------- 0. Refresh numbers
log "0/6  Regenerating camera_ready_numbers.json ..."
python3 -I p5/analysis/camera_ready_numbers.py || fail "camera_ready_numbers.py failed"

# --------------------------------------------------------- 1. AggNum check
log "1/6  verify_safenum (record-scoped \\AggNum check) ..."
python3 -I scripts/verify_safenum.py "$MANUSCRIPT" || fail "verify_safenum.py: \\AggNum violations found"

# ---------------------------------------------------------------------- 2. Guard
log "2/6  Anti-Fabrication Guard against manuscript ..."

export SYNTHETIC_DATA_FORBIDDEN="${SYNTHETIC_DATA_FORBIDDEN:-true}"
export USE_GLENS_DATA="${USE_GLENS_DATA:-true}"
export USE_GEOMIP_DATA="${USE_GEOMIP_DATA:-true}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

GUARD_LOG_DIR="$REPO_ROOT/evidence/asci_paper"
mkdir -p "$GUARD_LOG_DIR"
GUARD_LOG="$GUARD_LOG_DIR/guard_log_cr_$(date -u +%Y%m%dT%H%M%SZ).txt"

python3 - <<PY | tee "$GUARD_LOG"
import json, sys
from datetime import datetime, timezone
from p5.antifab_guard.guard import check_manuscript
from p5.antifab_guard.ledger_types import LedgerSnapshot

LEDGERS = [
    'evidence/expansion_500/expansion_500_ledger.jsonl',
    'evidence/pilot_50/pilot_50_ledger.jsonl',
    'evidence/expansion_500/abmil_ledger.jsonl',
    'evidence/expansion_500/clam_sb_ledger.jsonl',
    'evidence/expansion_500/analysis/alt_gates_ledger.jsonl',
    'evidence/expansion_500/analysis/conformal_variants_ledger.jsonl',
    'evidence/expansion_500_phikonv1/expansion_500_phikonv1_ledger.jsonl',
    'evidence/expansion_500_lunit/expansion_500_lunit_ledger.jsonl',
    'evidence/expansion_500_dinov2l/expansion_500_dinov2l_ledger.jsonl',
]
entries = []
for path in LEDGERS:
    try:
        with open(path) as f:
            for line in f:
                row = json.loads(line)
                entries.append({
                    'claim_id': row['claim_id'],
                    'claim_text': row['claim_text'],
                    'evidence_ref': row['evidence_ref'],
                    'verdict': row['verdict'],
                    'justification': json.dumps(row.get('justification', {})),
                    'timestamp': row['timestamp'],
                    'backbone_version': row['backbone_version'],
                    'cohort_id': row['cohort_id'],
                    'gate_rule_id': row['gate_rule_id'],
                })
    except FileNotFoundError:
        continue

snap = LedgerSnapshot(entries=entries)
timestamp = datetime.now(timezone.utc).isoformat()
print(f'guard run at {timestamp}')
print(f'manuscript: $MANUSCRIPT_REL')
print(f'ledgers loaded: {len(entries)} entries')
violations = check_manuscript('$MANUSCRIPT_REL', snap)
if violations:
    print(f'  ! {len(violations)} violation(s):', file=sys.stderr)
    for v in violations[:20]:
        print(f'    line {v.line_number}: {v.reason} token={v.token!r}', file=sys.stderr)
    sys.exit(1)
print(f'  guard OK ({len(entries)} ledger entries loaded, 0 violations)')
PY

# --------------------------------------------------- 3. Enforcement layers
log "3/6  Enforcement-flag status check ..."
python3 - <<'PY'
import sys
from p5.antifab_guard.sakana_status import check_sakana_status
status = check_sakana_status()
if status.any_layer_disabled():
    print(f'  ! disabled enforcement layers: {status.disabled_layers()}', file=sys.stderr)
    sys.exit(1)
print('  all 5 enforcement layers active')
PY

# -------------------------------------------- 4. LaTeX 4-pass build
log "4/6  LaTeX 4-pass build ..."
cd "$PAPER_DIR"
pdflatex -interaction=nonstopmode -halt-on-error main_camera_ready.tex > build_cr.log 2>&1 \
    || { tail -30 build_cr.log; fail "pdflatex pass 1"; }
bibtex main_camera_ready >> build_cr.log 2>&1 \
    || { tail -20 build_cr.log; fail "bibtex"; }
pdflatex -interaction=nonstopmode -halt-on-error main_camera_ready.tex >> build_cr.log 2>&1 \
    || { tail -20 build_cr.log; fail "pdflatex pass 2"; }
pdflatex -interaction=nonstopmode -halt-on-error main_camera_ready.tex >> build_cr.log 2>&1 \
    || { tail -20 build_cr.log; fail "pdflatex pass 3"; }
[ -f main_camera_ready.pdf ] || fail "main_camera_ready.pdf not produced"
log "  built $PDF"

# ----------------------------------------------- 5. Page-budget check
log "5/6  Page budget ..."
TOTAL_PAGES=$(pdfinfo main_camera_ready.pdf | awk '/^Pages:/ {print $2}')
[ -n "$TOTAL_PAGES" ] || fail "could not read page count from pdfinfo"

BODY_SPILL_THRESHOLD=30
BODY_PAGES=""
BODY_SPILL_WORDS=0
if command -v pdftotext >/dev/null 2>&1; then
    read -r REF_PAGE BODY_SPILL_WORDS < <(pdftotext -layout main_camera_ready.pdf - 2>/dev/null \
        | python3 -c '
import sys, re
text = sys.stdin.read()
pages = text.split("\f")
for i, page in enumerate(pages):
    if re.search(r"^\s*(?:\d+\s+)?References\s*$", page, flags=re.MULTILINE):
        before = re.split(r"^\s*(?:\d+\s+)?References\s*$", page, maxsplit=1, flags=re.MULTILINE)[0]
        print(i + 1, len([w for w in before.split() if not w.isdigit()]))
        break
')
    if [ -n "$REF_PAGE" ]; then
        if [ "${BODY_SPILL_WORDS:-0}" -gt "$BODY_SPILL_THRESHOLD" ]; then
            BODY_PAGES="$REF_PAGE"
        else
            BODY_PAGES=$((REF_PAGE - 1))
        fi
    fi
fi
[ -z "$BODY_PAGES" ] && BODY_PAGES="$TOTAL_PAGES"

if [ "$BODY_PAGES" -gt 9 ]; then
    fail "PDF body has $BODY_PAGES pages (total $TOTAL_PAGES), exceeds 9-page ASCI cap"
fi
log "  body is $BODY_PAGES pages (total $TOTAL_PAGES); <= 9 ASCI cap"
PAGES="$BODY_PAGES"

# ----------------------------------------------- 6. Single-blind check
log "6/6  Single-blind check ..."
if command -v pdftotext >/dev/null 2>&1; then
    TEXT=$(pdftotext main_camera_ready.pdf -)
    FIRSTPAGE=$(printf '%s' "$TEXT" | sed -n '1,/^Abstract$/p')
    for NAME in 'Enrique Zueco' 'David Scott Lewis' 'Haley Yi'; do
        if ! grep -qiF "$NAME" <<<"$FIRSTPAGE"; then
            fail "Author '$NAME' not visible on title page"
        fi
    done
    log "  authors visible (single-blind confirmed)"
fi

log "SUCCESS — built $PDF ($PAGES body pages) — BUILT — pending human verify"

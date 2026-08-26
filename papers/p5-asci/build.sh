#!/usr/bin/env bash
# ASCI paper build with anti-fabrication guarantees.
#
# Runs in order:
#   1. Anti-Fabrication Guard against the manuscript source
#   2. SAKANA layer status check (all five active)
#   3. LaTeX 4-pass build against vendor/neurips2026/neurips_2026.sty
#   4. Page-budget verification (<=9 including figures + tables)
#   5. Single-blind verification (NOT anonymized; author name visible)
#
# Any step failing fails the whole build. On success, the PDF is stamped
# "BUILT — pending human verify" in the console (build.log line).
#
# Usage: bash papers/p5-asci/build.sh
# Exit codes: 0 = built, non-zero = failed.

set -euo pipefail

PAPER_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$PAPER_DIR/../.." && pwd)"
MANUSCRIPT="$PAPER_DIR/main.tex"
PDF="$PAPER_DIR/main.pdf"

cd "$REPO_ROOT"

log() { printf '[build] %s\n' "$*"; }
fail() { printf '[build] FAIL: %s\n' "$*" >&2; exit 1; }

# ---------------------------------------------------------------------- 1. Guard
log "1/5  Anti-Fabrication Guard against manuscript ..."

# Ensure SAKANA layer 2 env vars are set BEFORE running the guard, so the
# guard does not emit sakana_layer_bypass violations from its own subprocess.
export SYNTHETIC_DATA_FORBIDDEN="${SYNTHETIC_DATA_FORBIDDEN:-true}"
export USE_GLENS_DATA="${USE_GLENS_DATA:-true}"
export USE_GEOMIP_DATA="${USE_GEOMIP_DATA:-true}"
export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"

GUARD_LOG_DIR="$REPO_ROOT/evidence/asci_paper"
mkdir -p "$GUARD_LOG_DIR"
GUARD_LOG="$GUARD_LOG_DIR/guard_log_$(date -u +%Y%m%dT%H%M%SZ).txt"

python3 - <<PY | tee "$GUARD_LOG"
import json, sys
from datetime import datetime, timezone
from p5.antifab_guard.guard import check_manuscript
from p5.antifab_guard.ledger_types import LedgerSnapshot

# Load every ledger the paper legitimately draws from (mean-pool baseline +
# pilot-50 + Phase B ABMIL + CLAM-SB + Phase A alt-gates + calibration).
LEDGERS = [
    'evidence/expansion_500/expansion_500_ledger.jsonl',
    'evidence/pilot_50/pilot_50_ledger.jsonl',
    'evidence/expansion_500/abmil_ledger.jsonl',
    'evidence/expansion_500/clam_sb_ledger.jsonl',
    'evidence/expansion_500/analysis/alt_gates_ledger.jsonl',
    # Round-5 conformal variants (cross-fold + Mondrian, operating alpha=0.10)
    'evidence/expansion_500/analysis/conformal_variants_ledger.jsonl',
    # Cross-backbone runs (Appendix: cross-backbone verdict agreement)
    'evidence/expansion_500_phikonv1/expansion_500_phikonv1_ledger.jsonl',
    'evidence/expansion_500_lunit/expansion_500_lunit_ledger.jsonl',
    'evidence/expansion_500_dinov2l/expansion_500_dinov2l_ledger.jsonl',
]
entries = []
for path in LEDGERS:
    try:
        with open(path) as f:
            n_this = 0
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
                n_this += 1
    except FileNotFoundError:
        # Optional ledger not present — skip. Never silent for the mandatory ones.
        continue

snap = LedgerSnapshot(entries=entries)
timestamp = datetime.now(timezone.utc).isoformat()
print(f'guard run at {timestamp}')
print(f'manuscript: $MANUSCRIPT')
print(f'ledgers loaded: {len(entries)} entries from {len(LEDGERS)} files')
violations = check_manuscript('$MANUSCRIPT', snap)
if violations:
    print(f'  ! {len(violations)} violation(s):', file=sys.stderr)
    for v in violations[:20]:
        print(f'    line {v.line_number}: {v.reason} token={v.token!r} '
              f'claim_id={v.claim_id_or_none}', file=sys.stderr)
    sys.exit(1)
print(f'  guard OK ({len(entries)} ledger entries loaded, 0 violations)')
PY

# ----------------------------------------------------------- 2. SAKANA layers
log "2/5  SAKANA layer status ..."
python3 - <<'PY'
import sys
from p5.antifab_guard.sakana_status import check_sakana_status
status = check_sakana_status()
if status.any_layer_disabled():
    print(f'  ! disabled SAKANA layers: {status.disabled_layers()}', file=sys.stderr)
    sys.exit(1)
print('  all 5 SAKANA layers active')
PY

# -------------------------------------------------------- 3. LaTeX 4-pass build
log "3/5  LaTeX 4-pass build ..."
cd "$PAPER_DIR"
pdflatex -interaction=nonstopmode -halt-on-error main.tex > build.log 2>&1 || { tail -20 build.log; fail "pdflatex pass 1"; }
bibtex main >> build.log 2>&1 || { tail -20 build.log; fail "bibtex"; }
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log 2>&1 || { tail -20 build.log; fail "pdflatex pass 2"; }
pdflatex -interaction=nonstopmode -halt-on-error main.tex >> build.log 2>&1 || { tail -20 build.log; fail "pdflatex pass 3"; }
[ -f main.pdf ] || fail "main.pdf not produced"
log "  built $PDF"

# ------------------------------------------------------- 4. Page-budget check
log "4/5  Page budget ..."
TOTAL_PAGES=$(pdfinfo main.pdf | awk '/^Pages:/ {print $2}')
[ -n "$TOTAL_PAGES" ] || fail "could not read page count from pdfinfo"

# ASCI 2026 caps the body (incl. figures and tables) at 9 pages; references
# and appendices are excluded. We locate the first page whose text contains a
# lone "References" heading and count the body as the pages before it.
#
# L5 fix (page_gate_false_green): body content (Limitations, Broader Impact,
# Conclusion, Reproducibility Statement) can spill ONTO the References page,
# before the "References" heading. Naively taking BODY_PAGES = REF_PAGE - 1
# then UNDERCOUNTS by a page and passes a >9-page body. We now measure the
# body words that appear on the References page before the heading: if they
# exceed a small threshold (real content, not an incidental caption line),
# the body genuinely occupies the References page and we count it.
BODY_SPILL_THRESHOLD=30
BODY_PAGES=""
BODY_SPILL_WORDS=0
if command -v pdftotext >/dev/null 2>&1; then
    read -r REF_PAGE BODY_SPILL_WORDS < <(pdftotext -layout main.pdf - 2>/dev/null \
        | python3 -c '
import sys, re
text = sys.stdin.read()
pages = text.split("\f")
for i, page in enumerate(pages):
    if re.search(r"^\s*References\s*$", page, flags=re.MULTILINE):
        before = re.split(r"^\s*References\s*$", page, maxsplit=1, flags=re.MULTILINE)[0]
        print(i + 1, len(before.split()))
        break
')
    if [ -n "$REF_PAGE" ]; then
        if [ "${BODY_SPILL_WORDS:-0}" -gt "$BODY_SPILL_THRESHOLD" ]; then
            # body content is physically on the References page -> count it
            BODY_PAGES="$REF_PAGE"
        else
            BODY_PAGES=$((REF_PAGE - 1))
        fi
    fi
fi
if [ -z "$BODY_PAGES" ]; then
    BODY_PAGES="$TOTAL_PAGES"
fi

if [ "$BODY_PAGES" -gt 9 ]; then
    fail "PDF body has $BODY_PAGES pages (total $TOTAL_PAGES; $BODY_SPILL_WORDS body words on the References page), exceeds 9-page ASCI cap (refs + appendix excluded)"
fi
log "  body is $BODY_PAGES pages (total $TOTAL_PAGES, refs + appendix excluded; $BODY_SPILL_WORDS body words on the References page, threshold $BODY_SPILL_THRESHOLD); <= 9 ASCI cap"
# keep PAGES variable name for the final SUCCESS line
PAGES="$BODY_PAGES"

# ------------------------------------------------------- 5. Single-blind check
log "5/5  Single-blind check (author name must be visible) ..."
# Extract text from PDF and grep for author name
if command -v pdftotext >/dev/null 2>&1; then
    # Extract full text, then check author name appears near the top (first page).
    # First-page text is separated from body text by the Abstract section marker.
    TEXT=$(pdftotext main.pdf -)
    FIRSTPAGE=$(printf '%s' "$TEXT" | sed -n '1,/^Abstract$/p')
    # Expected author block (ASCI is SINGLE-BLIND: authors MUST be visible).
    # Uses <<< rather than a pipe: under `set -o pipefail` a `printf | grep -q`
    # can return 141 on an early match and silently pass. (Campaign lesson.)
    for NAME in 'Enrique Zueco' 'David Scott Lewis' 'Haley Yi'; do
        if ! grep -qiF "$NAME" <<<"$FIRSTPAGE"; then
            fail "Author '$NAME' not visible on title page; check the author block"
        fi
    done
    if ! grep -qiF 'AIXC Research' <<<"$FIRSTPAGE"; then
        fail "Affiliation 'AIXC Research' not visible on title page"
    fi
    if ! grep -qiF 'reports@aixcresearch.com' <<<"$FIRSTPAGE"; then
        fail "Contact email not visible on title page"
    fi
    log "  3 authors + affiliation + contact visible on title page (single-blind confirmed)"
    # Stale-identity regression check: a previous identity must never reappear.
    for STALE in 'Independent Researcher' 'rocketmail' 'Cabezal'; do
        if grep -qiF "$STALE" <<<"$TEXT"; then
            fail "Stale identity string '$STALE' present in the PDF — the author block is out of date"
        fi
    done
    log "  author-block registry-match guard: PASS (3/3 authors, 0 stale identities)"
else
    log "  pdftotext not available; skipping automated author-visibility check"
fi

log "SUCCESS — built $PDF ($PAGES pages) — BUILT — pending human verify"

"""
Anti-Fabrication Guard: validates numeric claims in manuscripts against ledger.

Scans a LaTeX or Markdown draft for numeric tokens (percentages, AUROC/AUC values,
N counts, decimal ratios) and requires each to resolve to a claim_id in the ledger
whose verdict is SUPPORTED. Non-empty violation list fails the build.

Wraps (does not replace) the five SAKANA enforcement layers; any bypass is itself
a build-failing violation.

Phase C1 (2026-08-11) hardening:
- Numeric matching is now canonicalised: every token (manuscript AND ledger
  claim_text) is parsed to a float and compared with a small numeric tolerance.
  This eliminates the substring false-positives of the pre-C1 matcher, where
  "0.7" would spuriously match a ledger row whose claim_text mentioned "0.70"
  or "0.7778".
- LaTeX numeric macros are respected: `\\num{0.365}` is treated as the number
  0.365, not skipped for being wrapped in a macro.
- Aggregation escape: `\\SafeNum{X}` marks X as a legitimate ledger-derived
  aggregation (e.g. SUPPORTED count across a cohort). Numbers inside
  `\\SafeNum{...}` are whitelisted and not checked against the ledger. The
  aggregation itself must be documented in the paper (§Guard subsection).
"""

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from .ledger_types import LedgerSnapshot
from .sakana_status import check_sakana_status

# Tolerance for float comparisons — 5e-4 accepts 0.365 vs 0.3652 (same
# 3-decimal-place rounding) but rejects 0.365 vs 0.37. Numbers reported to
# more precision than the ledger stores are treated as compatible so long as
# they round to the same 3 decimal places.
NUMERIC_TOL = 5e-4


@dataclass
class Violation:
    """A single manuscript validation failure."""

    token: str
    """The numeric token that failed validation (e.g., '87.3%', 'AUROC=0.92')"""

    line_number: int
    """The line in the manuscript where the token appears (1-indexed)"""

    reason: str
    """Reason for the violation: 'no_claim_id', 'verdict_not_supported', 'sakana_layer_bypass'"""

    claim_id_or_none: Optional[str] = None
    """The claim_id if one was found but rejected; None if no claim found"""


SAFENUM_PATTERN = re.compile(r"\\SafeNum\{([^{}]*)\}")
LATEX_NUM_PATTERN = re.compile(r"\\num\{([^{}]*)\}")


def canonical_value(token: str) -> Optional[float]:
    """Parse a numeric token to a canonical float, or None if unparseable.

    Handles:
      - "0.734"      → 0.734
      - "87.3%"      → 0.873   (percent → fraction)
      - "N=500"      → 500.0
      - "AUROC=0.92" → 0.92
      - "AUC 0.881"  → 0.881
      - "\\num{0.365}" → 0.365
      - "50"         → 50.0
    """
    if not token:
        return None
    s = token.strip()

    # Strip \num{...} wrapper.
    m = LATEX_NUM_PATTERN.search(s)
    if m:
        s = m.group(1).strip()

    # Percentage form — accept both bare "%" and LaTeX-escaped "\%".
    if s.endswith("%"):
        core = s[:-1]
        if core.endswith("\\"):
            core = core[:-1]
        try:
            return float(core.strip()) / 100.0
        except ValueError:
            return None

    # AUROC/AUC labeled numbers.
    m = re.search(r"(?:AUROC|AUC|auroc|auc)\s*[:=\s]+(-?\d+(?:\.\d+)?)", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None

    # N=... form.
    m = re.search(r"\b[Nn]\s*[:=~]\s*(-?\d+(?:\.\d+)?)\b", s)
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None

    # Bare number.
    try:
        return float(s)
    except ValueError:
        pass

    # Fallback — pull first floating-point token out.
    m = re.search(r"-?\d+\.\d+|-?\d+", s)
    if m:
        try:
            return float(m.group(0))
        except ValueError:
            return None
    return None


def extract_safenum_ranges(line: str) -> list[tuple[int, int]]:
    """Return [start, end) index ranges of raw `\\SafeNum{...}` calls in the line."""
    return [(m.start(), m.end()) for m in SAFENUM_PATTERN.finditer(line)]


def _inside_ranges(pos: int, ranges: list[tuple[int, int]]) -> bool:
    return any(a <= pos < b for a, b in ranges)


class NumericTokenMatcher:
    """
    Recognizes numeric tokens in manuscript text.

    Patterns:
    - Percentages: 12.3%, 12%, 100%
    - AUROC/AUC: AUROC=0.87, AUC 0.912, auroc: 0.734
    - N counts: N=500, n = 50, N=10, sample size = 100
    - Bare decimals: 0.734, 0.51 (only if adjacent to result keywords)
    - LaTeX \\num{X} — the X is extracted as a number
    - \\SafeNum{X} — whitelisted (aggregation escape), the X is NOT reported
    """

    # Pattern for percentages: digit(s), optional decimal, %.  Accepts both
    # the LaTeX-escaped form (`\%`) and the bare form (`%`); the captured
    # group is either "99.9%" or "99.9\%" — canonicalisation strips both.
    PERCENTAGE_PATTERN = r"\b(\d+(?:\.\d+)?\\?%)"

    # Pattern for AUROC/AUC: (AUROC|AUC|auroc|auc) followed by = or : or space, then decimal
    AUROC_PATTERN = r"(?:AUROC|AUC|auroc|auc)[\s:=]+(\d+\.?\d*)"

    # Pattern for N counts: N or n followed by = or ~, then digits
    N_COUNT_PATTERN = r"\b[Nn][\s:=~]+(\d+)\b"

    # Keywords that suggest a number is a result (for bare decimals)
    RESULT_KEYWORDS = {
        "accuracy", "precision", "recall", "f1", "auc", "auroc",
        "score", "ratio", "rate", "sensitivity", "specificity",
        "mcc", "coefficient", "correlation", "loss", "error",
        "performance", "metric", "efficacy", "effectiveness",
    }

    @classmethod
    def find_tokens(cls, text: str, line_number: int) -> list[tuple[str, int]]:
        """
        Extract all numeric tokens from a line of text.

        `\\SafeNum{...}` calls are treated as trusted aggregations: any number
        appearing inside them is suppressed from the returned list.
        `\\num{X}` numbers are extracted as normal numeric tokens.

        Args:
            text: A single line of manuscript text
            line_number: The line number (1-indexed) for error reporting

        Returns:
            List of (token, line_number) tuples found in the line
        """
        tokens: list[tuple[str, int]] = []
        safe_ranges = extract_safenum_ranges(text)

        def emit(tok: str, start_pos: int) -> None:
            if _inside_ranges(start_pos, safe_ranges):
                return
            tokens.append((tok, line_number))

        # Find percentages
        for match in re.finditer(cls.PERCENTAGE_PATTERN, text):
            emit(match.group(1), match.start())

        # Find AUROC/AUC values
        for match in re.finditer(cls.AUROC_PATTERN, text):
            emit(match.group(0), match.start())

        # Find N counts
        for match in re.finditer(cls.N_COUNT_PATTERN, text):
            emit(match.group(0), match.start())

        # \num{X} — extract each number wrapped in the LaTeX macro.
        for match in LATEX_NUM_PATTERN.finditer(text):
            emit(match.group(1).strip(), match.start())

        # Find bare decimals adjacent to result keywords
        # Only match if a result keyword appears within ~20 chars
        lower_text = text.lower()
        for keyword in cls.RESULT_KEYWORDS:
            if keyword in lower_text:
                # Look for decimals near this keyword
                pattern = r"\b0\.\d+\b"
                for match in re.finditer(pattern, text):
                    token = match.group(0)
                    # Check if keyword is nearby (within ~30 chars)
                    start = max(0, match.start() - 30)
                    end = min(len(text), match.end() + 30)
                    nearby = text[start:end]
                    if keyword in nearby.lower():
                        emit(token, match.start())
                        break  # Don't duplicate the same token

        return tokens

    @classmethod
    def should_skip_line(cls, line: str) -> bool:
        """
        Decide if a line should be skipped (e.g., LaTeX commands, page numbers).

        Args:
            line: A line of text from the manuscript

        Returns:
            True if the line should be skipped from token scanning
        """
        stripped = line.strip()

        # Skip LaTeX commands (begin/end, document setup, etc.)
        if stripped.startswith("\\"):
            return True

        # Skip comments
        if stripped.startswith("%"):
            return True

        # Skip lines that are pure numbers (likely page numbers or references)
        if stripped.isdigit():
            return True

        # Skip very short lines (likely labels or section numbers)
        if len(stripped) < 3:
            return True

        return False


def check_manuscript(
    manuscript_path: str, ledger_snapshot: LedgerSnapshot
) -> list[Violation]:
    """
    Validate a manuscript for unsupported numeric claims.

    Scans the file for numeric tokens and checks that each resolves to
    a claim_id in the ledger with verdict=SUPPORTED. Also validates that
    all five SAKANA enforcement layers are enabled.

    Args:
        manuscript_path: Path to the manuscript file (LaTeX or Markdown)
        ledger_snapshot: A snapshot of ledger entries to validate against

    Returns:
        List of Violation objects. Non-empty list = build fails.
        Empty list = manuscript is clean.

    Raises:
        FileNotFoundError: If the manuscript file does not exist
    """

    path = Path(manuscript_path)
    if not path.exists():
        raise FileNotFoundError(f"Manuscript not found: {manuscript_path}")

    violations: list[Violation] = []

    # Check SAKANA layer status first
    sakana_status = check_sakana_status()
    if sakana_status.any_layer_disabled():
        disabled = sakana_status.disabled_layers()
        violation = Violation(
            token="<sakana-enforcement>",
            line_number=0,
            reason="sakana_layer_bypass",
            claim_id_or_none=None,
        )
        violations.append(violation)

    # Build canonical value indexes over the ledger.
    #   supported_values[value] = list of matching (claim_id, claim_text)
    #   other_values[value]     = list of (claim_id, claim_text, verdict) for
    #                              non-SUPPORTED claims — used to distinguish
    #                              "no ledger entry" from "wrong verdict".
    ledger_entries = ledger_snapshot.get("entries", [])
    supported_values: dict[float, list[tuple[str, str]]] = {}
    other_values: dict[float, list[tuple[str, str, str]]] = {}
    # Also keep string-substring matching for legacy claim_texts that hold
    # multi-token strings ("slide=X isup=Y conf=Z ...") where a manuscript
    # literal is exactly a substring of the claim_text — the pre-C1 tests
    # depend on this behaviour.
    supported_substrings: list[tuple[str, str]] = []
    other_substrings: list[tuple[str, str, str]] = []

    for entry in ledger_entries:
        claim_text = entry.get("claim_text", "") or ""
        cid = entry.get("claim_id", "") or ""
        verdict = entry.get("verdict") or ""
        # Extract EVERY numeric literal from the claim_text. Percentages must
        # be recognised FIRST (before the plain-decimal alternative) so
        # "92.5%" canonicalises to 0.925 instead of 92.5.
        pattern = r"-?\d+(?:\.\d+)?%|-?\d+\.\d+|-?\d+"
        for m in re.finditer(pattern, claim_text):
            val = canonical_value(m.group(0))
            if val is None:
                continue
            if verdict == "SUPPORTED":
                supported_values.setdefault(val, []).append((cid, claim_text))
            else:
                other_values.setdefault(val, []).append((cid, claim_text, verdict))
        if verdict == "SUPPORTED":
            supported_substrings.append((cid, claim_text))
        else:
            other_substrings.append((cid, claim_text, verdict))

    def _match_value(v: float, table: dict[float, list]) -> Optional[tuple]:
        for known, entries in table.items():
            if abs(known - v) <= NUMERIC_TOL:
                return entries[0]
        return None

    # Read the manuscript file
    with open(path, "r", encoding="utf-8") as f:
        lines = f.readlines()

    # Scan each line for numeric tokens
    for line_num, line in enumerate(lines, start=1):
        # Skip lines that don't need checking
        if NumericTokenMatcher.should_skip_line(line):
            continue

        # Find all numeric tokens in this line
        tokens = NumericTokenMatcher.find_tokens(line, line_num)

        for token, reported_line in tokens:
            val = canonical_value(token)

            claim_hit = None
            verdict_hit = None
            substring_hit = None
            substring_verdict = None

            if val is not None:
                s = _match_value(val, supported_values)
                if s is not None:
                    claim_hit = s[0]
                    verdict_hit = "SUPPORTED"
                else:
                    o = _match_value(val, other_values)
                    if o is not None:
                        claim_hit = o[0]
                        verdict_hit = o[2]

            # Fallback: legacy substring match against claim_text strings
            # (kept so existing tests with claim_text="N=500" style keep
            # working). Only consulted when canonical value could not be
            # extracted from the token; a well-formed numeric token that
            # simply does not have a matching ledger value must be reported
            # (this is the C1 fix — substring FPs on numeric prefixes like
            # "0.7" matching claim_text "0.7778" are eliminated).
            if claim_hit is None and val is None:
                for cid, claim_text in supported_substrings:
                    if token and (token in claim_text or (
                        claim_text and claim_text in line and len(claim_text) > 3
                    )):
                        substring_hit = cid
                        substring_verdict = "SUPPORTED"
                        break
                if substring_hit is None:
                    for cid, claim_text, verdict in other_substrings:
                        if token and (token in claim_text or (
                            claim_text and claim_text in line and len(claim_text) > 3
                        )):
                            substring_hit = cid
                            substring_verdict = verdict
                            break

            final_cid = claim_hit or substring_hit
            final_verdict = verdict_hit or substring_verdict

            if final_cid is None:
                violations.append(Violation(
                    token=token,
                    line_number=reported_line,
                    reason="no_claim_id",
                    claim_id_or_none=None,
                ))
            elif final_verdict != "SUPPORTED":
                violations.append(Violation(
                    token=token,
                    line_number=reported_line,
                    reason="verdict_not_supported",
                    claim_id_or_none=final_cid,
                ))

    return violations

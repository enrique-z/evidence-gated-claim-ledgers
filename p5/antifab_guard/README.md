# Anti-Fabrication Guard

Validates numeric claims in manuscripts against a claim ledger.

## Purpose

The Anti-Fabrication Guard is part of the P5 (ASCI) evidence-gated pipeline. It scans a LaTeX or Markdown manuscript for numeric tokens (percentages, AUROC values, N counts, decimal ratios) and ensures that each resolves to a `claim_id` in the ledger whose verdict is `SUPPORTED`.

A non-empty violation list fails the manuscript build.

## Public API

### `check_manuscript(manuscript_path: str, ledger_snapshot: LedgerSnapshot) -> list[Violation]`

Validates a manuscript against a ledger snapshot.

**Arguments:**
- `manuscript_path`: Path to a `.tex` or `.md` file
- `ledger_snapshot`: A dict with an `entries` list of ledger rows

**Returns:**
- List of `Violation` objects. Empty list = manuscript is clean.
- Non-empty list = build fails.

**Raises:**
- `FileNotFoundError` if the manuscript does not exist

### `Violation` Type

```python
@dataclass
class Violation:
    token: str                    # The numeric token (e.g., "87.3%")
    line_number: int              # Line in manuscript (1-indexed)
    reason: str                   # "no_claim_id", "verdict_not_supported", "sakana_layer_bypass"
    claim_id_or_none: Optional[str]  # The claim_id if found but rejected
```

### Ledger Types

```python
Verdict = Literal["SUPPORTED", "UNSUPPORTED", "NOT_APPLICABLE", "CONTESTED"]

class LedgerEntry(TypedDict):
    claim_id: str
    claim_text: str
    evidence_ref: str
    verdict: Verdict
    justification: str
    timestamp: str
    backbone_version: str
    cohort_id: str
    gate_rule_id: str

class LedgerSnapshot(TypedDict):
    entries: list[LedgerEntry]
```

## Token Recognition

The guard recognizes four token types:

1. **Percentages**: `12.3%`, `95%`
2. **AUROC/AUC**: `AUROC=0.87`, `AUC 0.912`
3. **N counts**: `N=500`, `n = 50`
4. **Decimal ratios**: `0.734`, `0.51` (only if adjacent to result keywords like accuracy, precision, F1, etc.)

## SAKANA Layer Integration

The guard wraps (not replaces) the five SAKANA enforcement layers:

1. **Code** – numpy RNG hook + template bypass detection
2. **Process** – critical env vars force-set at runtime
3. **Data** – database manager validates authenticity
4. **Prompt** – per-domain "NO SYNTHETIC" hardcoded into agent goals
5. **Audit** – incident logger records violations

If any layer is disabled (environment variables missing or sentinel flags off), the guard reports a `sakana_layer_bypass` violation that fails the build.

### Layer Status Detection

The guard checks these environment variables:

- `SYNTHETIC_DATA_FORBIDDEN=true` (layer 1)
- `USE_GLENS_DATA=true` + `USE_GEOMIP_DATA=true` (layer 2)
- `DISABLE_DATABASE_VALIDATION` not set to "true" (layer 3)
- `DISABLE_PROMPT_HARDENING` not set to "true" (layer 4)
- `DISABLE_INCIDENT_LOGGER` not set to "true" (layer 5)

## Example Usage

```python
from p5.antifab_guard import check_manuscript

# Load a manuscript
manuscript_path = "my_paper.tex"

# Build a ledger snapshot from your database
ledger = {
    "entries": [
        {
            "claim_id": "claim_1",
            "claim_text": "N=500",
            "verdict": "SUPPORTED",
            "backbone_version": "phikon-v2",
        },
        {
            "claim_id": "claim_2",
            "claim_text": "AUROC=0.92",
            "verdict": "SUPPORTED",
            "backbone_version": "phikon-v2",
        },
    ]
}

# Check the manuscript
violations = check_manuscript(manuscript_path, ledger)

if violations:
    for v in violations:
        print(f"Line {v.line_number}: {v.reason} for token '{v.token}'")
    sys.exit(1)  # Build fails
else:
    print("Manuscript is clean.")
```

## Testing

Run pytest:

```bash
pytest p5/antifab_guard/tests -v
```

Fixtures are in `p5/antifab_guard/tests/fixtures/`:

- `all_supported.tex` – every number has a SUPPORTED ledger entry
- `one_fabricated.md` – one number with no ledger entry
- `contested_verdict.tex` – a number with CONTESTED verdict
- `unsupported_verdict.md` – a number with UNSUPPORTED verdict
- `not_applicable_verdict.md` – a number with NOT_APPLICABLE verdict
- `no_numeric_tokens.tex` – pure prose, no numbers
- `all_token_types.md` – one of each token type

## Design Notes

### Numeric Token Matching

The matcher uses regex patterns to find tokens. It explicitly skips:

- LaTeX commands (lines starting with `\`)
- Comments (lines starting with `%`)
- Page numbers (lines that are pure digits)
- Very short lines (likely labels)

This minimizes false positives from layout commands, style definitions, and section numbering.

### Claim Lookup

The current implementation matches tokens by substring search in claim text.
In a production system, this would be replaced with a more sophisticated index
(claim_id → metadata mapping) or a similarity search.

### Layer Bypass vs. Fabrication

A `sakana_layer_bypass` violation is reported as a line-0, token-`<sakana-enforcement>`
entry with no claim_id. This prevents the build even if all individual claims are supported.

### Integration with Pipeline

The orchestrator calls `check_manuscript()` as the final build gate, after the 
evidence gate produces the ledger. The gate log records whether the build passed.

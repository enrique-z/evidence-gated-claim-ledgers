# Cost Meter

An append-only ledger of cloud spend with a hard $300 cap and authorization gate for the P5 (ASCI) pipeline.

## Overview

The Cost Meter module tracks all billable operations (compute hours, storage, egress) and enforces a hard budget cap. It refuses to authorize any step whose projected cost would exceed the budget, preventing accidental overspend.

## Usage

```python
from p5.cost_meter import CostMeter

# Initialize (loads config from config.toml by default)
meter = CostMeter()

# Record spend
meter.record(
    item="compute-hours",
    amount=50.0,
    timestamp="2026-08-05T10:00:00Z",
    cohort_id="pilot-50"
)

# Check totals
print(meter.total())                # 50.0
print(meter.budget_remaining())     # 250.0

# Authorize before dispatching a billable step
if meter.authorize(projected_amount=100.0):
    # Safe to dispatch
    pass
else:
    # Would exceed budget — surface actionable pivot signal
    raise RuntimeError(f"Budget cap ($300) would be exceeded. "
                       f"Remaining: ${meter.budget_remaining()}")
```

## API

### `CostMeter(config_path: Optional[str] = None)`

Initialize the meter. Loads configuration from `config.toml` (default path: `p5/cost_meter/config.toml`).

**Args:**
- `config_path`: Path to a custom config.toml. Raises `FileNotFoundError` if the path does not exist.

### `record(item: str, amount: float, timestamp: str, cohort_id: str) -> None`

Record a spend item to the ledger.

**Args:**
- `item`: Description (e.g., `"compute-hours"`, `"storage"`).
- `amount`: Amount in USD. Must be ≥ 0.
- `timestamp`: ISO 8601 timestamp.
- `cohort_id`: Identifier of the cohort or run.

**Raises:**
- `NegativeAmountError`: If `amount < 0`. The error message explicitly states that spend items must be non-negative.

### `total() -> float`

Return the sum of all recorded amounts.

**Returns:** Total spend in USD.

### `budget_remaining() -> float`

Return the remaining budget.

**Returns:** `cap - total()`, where `cap` is loaded from config.

### `authorize(projected_amount: float) -> bool`

Check if a projected spend is within budget.

**Returns:** `True` if `projected_amount + total() <= cap`, `False` otherwise.

## Configuration

Configuration is loaded from `config.toml` (TOML format):

```toml
[budget]
cap = 300.0  # Hard budget cap in USD

[storage]
ledger_path = "~/.cache/neurips-26/cost_meter.jsonl"  # Append-only JSON-lines ledger
```

The cap is **not hard-coded**; it is loaded from the config file at initialization. To use a custom cap, pass a custom `config.toml` to the constructor.

## Persistence

The ledger is stored as a JSON-lines file on disk. Each `record()` call appends one line (a JSON object) to the file. The ledger survives process restarts:

```python
# Process 1
meter1 = CostMeter()
meter1.record("compute", 50.0, "2026-08-05T10:00:00Z", "pilot-50")
# Process exits

# Process 2 (fresh start)
meter2 = CostMeter()
print(meter2.total())  # Still 50.0 — ledger was persisted
```

## Testing

Run pytest tests with:

```bash
pytest p5/cost_meter/tests -v
```

Test coverage includes:
- **Property tests:** sum of recorded amounts equals `total()`, `budget_remaining() = cap - total()`, `authorize(x)` returns `False` iff `x + total() > cap`.
- **Persistence:** write, close, re-open, read yields the same rows.
- **Negative amounts:** raises `NegativeAmountError` with a clear message.
- **Config loading:** cap is loaded from config, not hard-coded.
- **Integration:** typical workflows with multiple cohorts and authorization decisions.

## Errors

### `CostMeterError` (base class)

Base exception for all Cost Meter errors.

### `NegativeAmountError(CostMeterError)`

Raised when attempting to record a negative amount. The error message explicitly states that spend items must be non-negative.

## Design Notes

- **Append-only ledger:** Every `record()` call appends to the ledger file. Rows are never rewritten or deleted.
- **Cap enforcement:** The cap is loaded from config and cannot be overridden at runtime.
- **No side effects:** The meter only reads and appends to the ledger file. It does not modify any cloud infrastructure or incur charges.
- **Integration point:** The orchestrator wires `authorize()` around every billable step and surfaces a pivot signal on refusal, not a silent failure.

## Files

- `meter.py` — Core Cost Meter implementation.
- `config.py` — Configuration loader.
- `config.toml` — Default configuration (cap=300.0).
- `__init__.py` — Public API exports.
- `tests/test_meter.py` — Comprehensive pytest tests.
- `requirements.txt` — Module dependencies (tomli for Python <3.11, pytest, hypothesis).

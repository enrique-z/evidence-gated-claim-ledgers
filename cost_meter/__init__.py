"""
Cost Meter: append-only ledger of cloud spend with a hard $300 cap and authorization gate.

Public API:
    from p5.cost_meter import CostMeter

    meter = CostMeter()
    meter.record("compute-hours", 50.0, "2026-08-05T10:00:00Z", "pilot-50")
    meter.record("storage", 25.5, "2026-08-05T10:30:00Z", "pilot-50")

    print(meter.total())               # 75.5
    print(meter.budget_remaining())    # 224.5
    print(meter.authorize(100.0))      # True (100 + 75.5 <= 300)
    print(meter.authorize(250.0))      # False (250 + 75.5 > 300)
"""

from .meter import CostMeter, CostMeterError, NegativeAmountError

__all__ = ["CostMeter", "CostMeterError", "NegativeAmountError"]

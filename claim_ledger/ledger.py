"""Append-only claim ledger.

Rows: (claim_id, claim_text, evidence_ref, verdict, justification, timestamp,
       backbone_version, cohort_id, gate_rule_id).

Persistence: newline-delimited JSON on disk. Append-only by construction —
`append(row)` opens the file in 'a' mode. Anti-fab guard (issue #25) scans
manuscript drafts and requires every numeric claim to resolve to a row here
whose verdict is `SUPPORTED`.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional


@dataclass(frozen=True)
class LedgerRow:
    claim_id: str
    claim_text: str
    evidence_ref: str
    verdict: str
    justification: Dict[str, Any] = field(default_factory=dict)
    timestamp: str = ""
    backbone_version: str = ""
    cohort_id: str = ""
    gate_rule_id: str = ""
    extra: Dict[str, Any] = field(default_factory=dict)


class ClaimLedger:
    def __init__(self, path: Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, row: LedgerRow) -> None:
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(row)) + "\n")

    def rows(self) -> Iterator[LedgerRow]:
        if not self.path.exists():
            return
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                yield LedgerRow(**obj)

    def count(self, cohort_id: Optional[str] = None) -> int:
        n = 0
        for r in self.rows():
            if cohort_id is None or r.cohort_id == cohort_id:
                n += 1
        return n

    def verdict_distribution(self, cohort_id: Optional[str] = None) -> Dict[str, int]:
        dist: Dict[str, int] = {}
        for r in self.rows():
            if cohort_id is not None and r.cohort_id != cohort_id:
                continue
            dist[r.verdict] = dist.get(r.verdict, 0) + 1
        return dist

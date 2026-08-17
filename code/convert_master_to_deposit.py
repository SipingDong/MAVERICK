# -*- coding: utf-8 -*-
"""
MAVERICK: Master Dataset → Deposit Format Converter

Reads the V8 master dataset (master_dataset.jsonl, N=5094) and emits the
deposit's dual-format dataset files:

  dataset/MAVERICK_validation_dataset.json   (JSON array of records)
  dataset/MAVERICK_validation_dataset.csv    (same records, flat table)

Field mapping (aligned to the deposit schema):

  deposit.id       <- _new_id          (M0000..M5093, unique across V8)
  deposit.truth    <- ground_truth     (authentic -> genuine, non-genuine -> non_genuine)
  deposit.category <- category         (authentic / fabricated / tampered / wild)
  deposit.subtype  <- subcategory      (empty string when absent)
  deposit.doi      <- extracted from reference ("doi:10.xxxx..." -> 10.xxxx...), else empty
  deposit.reference<- reference        (verbatim, full citation)
  deposit.source   <- notes            (verbatim)

Guarantees:
  * Every one of the 5094 rows is preserved; nothing is dropped or altered.
  * CSV is UTF-8, CRLF line endings, quoted where needed.

Usage:
    python code/convert_master_to_deposit.py [master_dataset.jsonl]
"""

import csv
import io
import json
import re
import sys
from pathlib import Path

# --- Paths ------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE = Path(sys.argv[1]) if len(sys.argv) > 1 else (
    Path("/Coze/Drive/辩溪/所有对话/主对话/MAVERICK_升级v8/dataset/master_dataset.jsonl")
)
JSON_OUT = REPO_ROOT / "dataset" / "MAVERICK_validation_dataset.json"
CSV_OUT = REPO_ROOT / "dataset" / "MAVERICK_validation_dataset.csv"

# --- DOI extraction ----------------------------------------------------------
# Matches "doi:10.xxxx..." (case-insensitive, optional space / colon variants).
DOI_RE = re.compile(r"(?i)(?:doi)\s*[:：]?\s*(10\.\S+)")


def extract_doi(reference: str) -> str:
    """Return the DOI string from a reference, or '' if none present."""
    if not reference:
        return ""
    m = DOI_RE.search(reference)
    return m.group(1) if m else ""


def to_truth(ground_truth: str) -> str:
    """Map ground_truth -> deposit truth label."""
    return "genuine" if ground_truth == "authentic" else "non_genuine"


def to_record(raw: dict) -> dict:
    """Convert one master-dataset row into a deposit record."""
    return {
        "id": raw.get("_new_id", ""),
        "truth": to_truth(raw.get("ground_truth", "")),
        "category": raw.get("category", ""),
        "subtype": raw.get("subcategory") or "",
        "doi": extract_doi(raw.get("reference", "")),
        "reference": raw.get("reference", ""),
        "source": raw.get("notes", ""),
    }


# --- Main ---------------------------------------------------------------------
def main():
    rows = []
    with open(DEFAULT_SOURCE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))

    print(f"Read master dataset: {len(rows)} rows from {DEFAULT_SOURCE}")

    records = [to_record(r) for r in rows]

    # --- Write JSON -----------------------------------------------------------
    JSON_OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)
    print(f"Wrote JSON: {JSON_OUT} ({len(records)} records)")

    # --- Write CSV (UTF-8, CRLF) ----------------------------------------------
    # Quote every field for maximum compatibility with downstream tools.
    buf = io.StringIO()
    writer = csv.writer(buf, quoting=csv.QUOTE_MINIMAL, lineterminator="\r\n")
    writer.writerow(["id", "truth", "category", "subtype", "doi", "reference", "source"])
    for rec in records:
        writer.writerow([rec["id"], rec["truth"], rec["category"], rec["subtype"],
                         rec["doi"], rec["reference"], rec["source"]])
    with open(CSV_OUT, "w", encoding="utf-8", newline="") as f:
        f.write(buf.getvalue())
    print(f"Wrote CSV: {CSV_OUT} ({len(records)} data rows)")

    # --- Summary verification ---------------------------------------------------
    from collections import Counter
    truth_cnt = Counter(rec["truth"] for rec in records)
    cat_cnt = Counter(rec["category"] for rec in records)
    doi_cnt = sum(1 for rec in records if rec["doi"])
    ids = [rec["id"] for rec in records]
    print()
    print("=== Verification ===")
    print(f"Total records      : {len(records)}")
    print(f"Unique ids         : {len(set(ids))} / {len(ids)}")
    print(f"truth distribution : {dict(truth_cnt)}")
    print(f"category distribution: {dict(cat_cnt)}")
    print(f"rows with doi      : {doi_cnt} / {len(records)}")
    return records


if __name__ == "__main__":
    main()

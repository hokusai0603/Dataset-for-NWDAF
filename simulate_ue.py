#!/usr/bin/env python3
"""
simulate_ue.py
==============
Simulates a single UE generating multiple overlapping traffic flows.

Usage:
    python simulate_ue.py scenario_config.json

Input:  JSON config specifying flows (category, start offset, duration).
Output: Single CSV with all packets merged and sorted by adjusted timestamp.

If a session is shorter than the requested duration, additional sessions
are automatically concatenated to fill the time.
"""

import csv
import json
import os
import random
import sys
from pathlib import Path

# ── paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "Combined_Dataset"


# ── helpers ────────────────────────────────────────────────────────────

def load_config(config_path: str) -> dict:
    """Load and validate the JSON scenario config."""
    with open(config_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    assert "total_time" in cfg, "Config must have 'total_time'"
    assert "flows" in cfg and len(cfg["flows"]) > 0, "Config must have at least one flow"

    for i, flow in enumerate(cfg["flows"]):
        assert "category" in flow, f"Flow {i} must have 'category'"
        assert "start" in flow, f"Flow {i} must have 'start'"
        assert "duration" in flow, f"Flow {i} must have 'duration'"
        flow.setdefault("app", None)

    cfg.setdefault("output_file", "simulated_ue_traffic.csv")
    return cfg


def find_session_files(category: str, app_filter: str = None) -> list:
    """
    Return list of CSV paths in Combined_Dataset/<category>/.
    Optionally filter by app_name in the filename.
    """
    cat_dir = DATASET_PATH / category
    if not cat_dir.exists():
        print(f"  [WARN] Category dir not found: {cat_dir}")
        return []

    files = list(cat_dir.glob("*.csv"))

    if app_filter:
        # Filter: filename contains the app name (case-insensitive)
        app_lower = app_filter.lower()
        files = [f for f in files if app_lower in f.name.lower()]

    return files


def read_session_packets(csv_path: Path) -> tuple:
    """
    Read a session CSV and return (header, rows).
    The header is a list of column names.
    Each row is a list of string values.
    """
    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.reader(f)
        header = next(reader)
        rows = list(reader)
    return header, rows


def get_timestamp_column(header: list) -> int:
    """Find the index of the 'relative_time' column."""
    for i, col in enumerate(header):
        if col.strip().lower() == "relative_time":
            return i
    return -1


def parse_timestamp(value: str) -> float:
    """Parse relative_time float string."""
    try:
        return float(value.strip().strip('"'))
    except ValueError:
        return 0.0




def build_flow_packets(
    flow_cfg: dict,
    flow_id: str,
    total_time: float,
) -> tuple:
    """
    Build the packet list for one flow definition.

    Returns (unified_header, packet_rows) where each packet_row has:
      - adjusted_timestamp as first element (float, for sorting)
      - then the CSV string values to write

    Logic:
      1. Find matching sessions.
      2. Load session, rebase timestamps to flow start offset.
      3. If session ends before start+duration, concatenate another.
      4. Trim to [start, min(start+duration, total_time)].
    """
    category = flow_cfg["category"]
    app = flow_cfg.get("app")
    flow_start = float(flow_cfg["start"])
    flow_duration = float(flow_cfg["duration"])
    flow_end = min(flow_start + flow_duration, total_time)

    session_files = find_session_files(category, app)
    if not session_files:
        print(f"  [WARN] No sessions found for {category}"
              f"{f' / {app}' if app else ''}, skipping flow {flow_id}")
        return None, []

    random.shuffle(session_files)

    collected_packets = []
    current_time = flow_start  # where next session should start
    file_idx = 0
    unified_header = None

    while current_time < flow_end:
        # Pick next session file (cycle if exhausted)
        sf = session_files[file_idx % len(session_files)]
        file_idx += 1

        header, rows = read_session_packets(sf)
        if not rows:
            continue

        if unified_header is None:
            unified_header = header

        ts_col = get_timestamp_column(header)
        if ts_col < 0:
            continue

        # Compute delta to rebase session start (relative_time=0) → current_time
        # Since source is already relative_time (starting at 0), 
        # delta is simply the current_time in our simulation timeline.
        delta = current_time

        for i, row in enumerate(rows):
            src_rel_ts = parse_timestamp(row[ts_col])
            adj_ts = src_rel_ts + delta

            # Trim: only keep packets within [flow_start, flow_end]
            if adj_ts < flow_start:
                continue
            if adj_ts > flow_end:
                break

            collected_packets.append((adj_ts, flow_id, header, row))

        if not rows:
            break

        # Advance current_time to end of this session based on relative_time of last packet
        last_src_rel_ts = parse_timestamp(rows[-1][ts_col])
        current_time = last_src_rel_ts + delta + 1e-9 # basically continuous
        
        if current_time >= flow_end:
            break

    return unified_header, collected_packets


# ── main ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python simulate_ue.py <scenario_config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    cfg = load_config(config_path)

    total_time = float(cfg["total_time"])
    output_file = cfg["output_file"]

    print("=" * 60)
    print(f"UE Traffic Simulator")
    print(f"  Total time : {total_time}s")
    print(f"  Flows      : {len(cfg['flows'])}")
    print(f"  Output     : {output_file}")
    print("=" * 60)

    # Collect all packets from all flows
    all_packets = []  # list of (adjusted_ts, flow_id, header, row)
    all_headers = {}  # flow_id → header

    for i, flow_cfg in enumerate(cfg["flows"]):
        flow_id = f"flow_{i}"
        cat = flow_cfg["category"]
        app = flow_cfg.get("app", "any")
        start = flow_cfg["start"]
        dur = flow_cfg["duration"]

        print(f"\n[{flow_id}] {cat}"
              f"{f' ({app})' if app else ''}"
              f"  t={start}s → {start+dur}s")

        header, packets = build_flow_packets(flow_cfg, flow_id, total_time)
        if header:
            all_headers[flow_id] = header
        all_packets.extend(packets)
        print(f"  → {len(packets)} packets collected")

    if not all_packets:
        print("\n[ERR] No packets collected. Check config and Combined_Dataset.")
        sys.exit(1)

    # Sort by adjusted timestamp
    all_packets.sort(key=lambda x: x[0])

    # Build unified output header
    # Use the superset of columns; we pick the first flow's header as base
    # and add flow_id + adjusted_timestamp at the front
    # Since MIRAGE and UTMobileNet have different columns, we keep
    # a generic approach: write the source-specific columns after metadata
    print(f"\nMerging {len(all_packets)} total packets...")

    # Determine if all flows share the same header structure
    # If mixed, we write a generic format
    unique_headers = set()
    for fid, hdr in all_headers.items():
        unique_headers.add(tuple(hdr))

    out_path = SCRIPT_DIR / output_file

    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)

        # The new Combined_Dataset has a standardized prefix (first 14 columns):
        # source_dataset, app_name, category, activity, session_id, session_duration, relative_time,
        # pkt_len, l4_proto, src_port, dst_port, tcp_flags, direction, iat
        
        # We will build a unified column list that keeps this order
        standard_cols = [
            "source_dataset", "app_name", "category", "activity",
            "session_id", "session_duration", "relative_time",
            "pkt_len", "l4_proto", "src_port", "dst_port", 
            "tcp_flags", "direction", "iat"
        ]
        
        # Find all "cascade" (source-specific) columns
        cascade_cols = []
        seen = set(standard_cols)
        for hdr in all_headers.values():
            for col in hdr:
                if col not in seen:
                    cascade_cols.append(col)
                    seen.add(col)

        out_header = ["flow_id", "adjusted_timestamp"] + standard_cols + cascade_cols
        writer.writerow(out_header)

        for adj_ts, flow_id, header, row in all_packets:
            col_map = {col: val for col, val in zip(header, row)}
            
            # Build row in correct order
            out_row = [flow_id, f"{adj_ts:.9f}"] # Microsecond precision (9 digits just to be safe)
            for col in standard_cols + cascade_cols:
                out_row.append(col_map.get(col, ""))
            writer.writerow(out_row)

    print(f"\n{'=' * 60}")
    print(f"Output: {out_path}")
    print(f"Total packets: {len(all_packets)}")

    # Show timeline summary
    if all_packets:
        t_min = all_packets[0][0]
        t_max = all_packets[-1][0]
        print(f"Time range: {t_min:.3f}s → {t_max:.3f}s ({t_max - t_min:.1f}s)")

    # Per-flow summary
    flow_counts = {}
    for _, fid, _, _ in all_packets:
        flow_counts[fid] = flow_counts.get(fid, 0) + 1
    for fid in sorted(flow_counts):
        print(f"  {fid}: {flow_counts[fid]} packets")

    print("=" * 60)


if __name__ == "__main__":
    main()

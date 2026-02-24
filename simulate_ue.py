#!/usr/bin/env python3
"""
simulate_ue.py
==============
Simulates multiple UEs generating overlapping traffic flows at UPF scale.

Usage:
    python simulate_ue.py scenario_config.json

Input:  JSON config specifying UEs, each with flows (action, genre, start, duration).
Output: 1) Single CSV with all UE packets merged and sorted by timestamp.
        2) JSON metadata file recording which session files were used.

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

    assert "UEs" in cfg and len(cfg["UEs"]) > 0, "Config must have at least one UE"

    for u_idx, ue in enumerate(cfg["UEs"]):
        assert "UE_ID" in ue, f"UE {u_idx} must have 'UE_ID'"
        assert "IP_address" in ue, f"UE {u_idx} must have 'IP_address'"
        assert "flows" in ue and len(ue["flows"]) > 0, f"UE {u_idx} must have at least one flow"

        for f_idx, flow in enumerate(ue["flows"]):
            assert "action" in flow, f"UE {u_idx} flow {f_idx} must have 'action'"
            assert "genre" in flow, f"UE {u_idx} flow {f_idx} must have 'genre'"
            assert "start" in flow, f"UE {u_idx} flow {f_idx} must have 'start'"
            assert "duration" in flow, f"UE {u_idx} flow {f_idx} must have 'duration'"
            flow.setdefault("app", None)

    cfg.setdefault("output_file", "simulated_ue_traffic.csv")
    cfg.setdefault("correlation_id", "")
    return cfg


def find_session_files(action: str, genre: str, app_filter: str = None) -> list:
    """
    Return list of CSV paths in Combined_Dataset/<action>/<genre>/.
    Optionally filter by app_name in the filename.
    """
    target_dir = DATASET_PATH / action / genre
    if not target_dir.exists():
        print(f"  [WARN] Dir not found: {target_dir}")
        return []

    files = list(target_dir.glob("*.csv"))

    if app_filter:
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


def get_column_index(header: list, name: str) -> int:
    """Find the index of a column by name (case-insensitive). Returns -1 if not found."""
    name_lower = name.strip().lower()
    for i, col in enumerate(header):
        if col.strip().lower() == name_lower:
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
    ue_id: str,
    ue_ip: str,
) -> tuple:
    """
    Build the packet list for one flow definition.

    Returns (unified_header, packet_rows, used_files) where:
      - unified_header: the CSV header from the source sessions
      - packet_rows: list of (adj_ts, ue_id, ue_ip, flow_id, header, row)
      - used_files: list of session file paths used

    Logic:
      1. Find matching sessions by action/genre.
      2. Load session, rebase timestamps to flow start offset.
      3. If session ends before start+duration, concatenate another.
      4. Trim to [start, start+duration].
      5. Replace UE-side IPs: direction=0 → src_ip = ue_ip,
                               direction=1 → dst_ip = ue_ip.
    """
    action = flow_cfg["action"]
    genre = flow_cfg["genre"]
    app = flow_cfg.get("app")
    flow_start = float(flow_cfg["start"])
    flow_duration = float(flow_cfg["duration"])
    flow_end = flow_start + flow_duration

    # Replay mode: use exact file_path list from a previous meta.json
    pinned_files = flow_cfg.get("file_path")
    if pinned_files:
        session_files = [DATASET_PATH / fp for fp in pinned_files]
        missing = [str(fp) for fp in session_files if not fp.exists()]
        if missing:
            print(f"  [WARN] Replay files missing: {missing}")
            session_files = [fp for fp in session_files if fp.exists()]
        if not session_files:
            print(f"  [WARN] No replay files found, skipping flow {flow_id}")
            return None, [], []
        replay_mode = True
    else:
        session_files = find_session_files(action, genre, app)
        if not session_files:
            print(f"  [WARN] No sessions found for {action}/{genre}"
                  f"{f' / {app}' if app else ''}, skipping flow {flow_id}")
            return None, [], []
        random.shuffle(session_files)
        replay_mode = False

    collected_packets = []
    used_files = []
    current_time = flow_start
    file_idx = 0
    unified_header = None

    while current_time < flow_end:
        sf = session_files[file_idx % len(session_files)]
        file_idx += 1

        header, rows = read_session_packets(sf)
        if not rows:
            continue

        if unified_header is None:
            unified_header = header

        ts_col = get_column_index(header, "relative_time")
        dir_col = get_column_index(header, "direction")
        src_ip_col = get_column_index(header, "src_ip")
        dst_ip_col = get_column_index(header, "dst_ip")

        if ts_col < 0:
            continue

        # Track which file was used
        rel_path = sf.relative_to(DATASET_PATH)
        if str(rel_path) not in used_files:
            used_files.append(str(rel_path))

        delta = current_time
        last_included_rel_ts = None
        session_exhausted = True  # assume we use all rows unless we break

        for row in rows:
            src_rel_ts = parse_timestamp(row[ts_col])
            adj_ts = src_rel_ts + delta

            if adj_ts > flow_end:
                session_exhausted = False  # we broke early, session has more data
                break

            # Replace UE-side IPs
            if dir_col >= 0 and src_ip_col >= 0 and dst_ip_col >= 0:
                direction = row[dir_col].strip()
                if direction == "0":  # Uplink: UE is sender
                    row[src_ip_col] = ue_ip
                elif direction == "1":  # Downlink: UE is receiver
                    row[dst_ip_col] = ue_ip

            last_included_rel_ts = src_rel_ts
            collected_packets.append((adj_ts, ue_id, ue_ip, flow_id, header, row))

        if not rows or last_included_rel_ts is None:
            break

        if session_exhausted:
            # Used all packets in this session — advance past session end, stitch next
            current_time = last_included_rel_ts + delta + 1e-9
        else:
            # Broke early because adj_ts > flow_end — we're done
            break

    return unified_header, collected_packets, used_files


# ── main ───────────────────────────────────────────────────────────────

def main():
    if len(sys.argv) < 2:
        print("Usage: python simulate_ue.py <scenario_config.json>")
        sys.exit(1)

    config_path = sys.argv[1]
    cfg = load_config(config_path)

    output_file = cfg["output_file"]
    correlation_id = cfg.get("correlation_id", "")

    total_ues = len(cfg["UEs"])
    total_flows = sum(len(ue["flows"]) for ue in cfg["UEs"])

    print("=" * 60)
    print(f"Multi-UE Traffic Simulator")
    print(f"  Correlation ID : {correlation_id}")
    print(f"  UEs            : {total_ues}")
    print(f"  Total flows    : {total_flows}")
    print(f"  Output         : {output_file}")
    print("=" * 60)

    # Collect all packets from all UEs/flows
    all_packets = []  # list of (adj_ts, ue_id, ue_ip, flow_id, header, row)
    all_headers = {}  # flow_id → header
    meta_ues = []     # for JSON metadata output

    for u_idx, ue_cfg in enumerate(cfg["UEs"]):
        ue_id = ue_cfg["UE_ID"]
        ue_ip = ue_cfg["IP_address"]

        print(f"\n── UE: {ue_id} (IP: {ue_ip}) ──")

        meta_flows = []

        for f_idx, flow_cfg in enumerate(ue_cfg["flows"]):
            flow_id = f"{ue_id}_flow_{f_idx}"
            action = flow_cfg["action"]
            genre = flow_cfg["genre"]
            app = flow_cfg.get("app", "any")
            start = flow_cfg["start"]
            dur = flow_cfg["duration"]

            has_pinned = "file_path" in flow_cfg
            mode_tag = " [REPLAY]" if has_pinned else ""

            print(f"  [{flow_id}] {action}/{genre}"
                  f"{f' ({app})' if app and app != 'any' else ''}"
                  f"  t={start}s → {start+dur}s{mode_tag}")

            header, packets, used_files = build_flow_packets(
                flow_cfg, flow_id, ue_id, ue_ip
            )
            if header:
                all_headers[flow_id] = header
            all_packets.extend(packets)
            print(f"    → {len(packets)} packets, {len(used_files)} sessions")

            # Record metadata
            meta_flows.append({
                "action": action,
                "genre": genre,
                "app": flow_cfg.get("app"),
                "start": start,
                "duration": dur,
                "file_path": used_files,
            })

        meta_ues.append({
            "UE_ID": ue_id,
            "IP_address": ue_ip,
            "flows": meta_flows,
        })

    if not all_packets:
        print("\n[ERR] No packets collected. Check config and Combined_Dataset.")
        sys.exit(1)

    # Sort all packets by adjusted timestamp
    all_packets.sort(key=lambda x: x[0])

    print(f"\nMerging {len(all_packets)} total packets from {total_ues} UEs...")

    # Build unified output header (no cascade columns — all CSVs are standardized)
    standard_cols = [
        "source_dataset", "app_name", "category", "activity",
        "session_id", "session_duration", "relative_time",
        "pkt_len", "l4_proto", "src_ip", "dst_ip",
        "src_port", "dst_port", "tcp_flags", "direction", "iat"
    ]

    out_header = ["ue_id", "ue_ip", "flow_id", "adjusted_timestamp"] + standard_cols

    out_path = SCRIPT_DIR / output_file

    with open(out_path, "w", newline="", encoding="utf-8") as fout:
        writer = csv.writer(fout)
        writer.writerow(out_header)

        for adj_ts, ue_id, ue_ip, flow_id, header, row in all_packets:
            col_map = {col: val for col, val in zip(header, row)}

            out_row = [ue_id, ue_ip, flow_id, f"{adj_ts:.9f}"]
            for col in standard_cols:
                out_row.append(col_map.get(col, ""))
            writer.writerow(out_row)

    # ── Write JSON metadata ───────────────────────────────────────────
    meta_path = out_path.with_suffix(".meta.json")
    meta = {
        "correlation_id": correlation_id,
        "output_file": output_file,
        "UEs": meta_ues,
    }
    with open(meta_path, "w", encoding="utf-8") as mf:
        json.dump(meta, mf, indent=2, ensure_ascii=False)

    # ── Summary ───────────────────────────────────────────────────────
    print(f"\n{'=' * 60}")
    print(f"Output CSV  : {out_path}")
    print(f"Output Meta : {meta_path}")
    print(f"Total packets: {len(all_packets)}")

    if all_packets:
        t_min = all_packets[0][0]
        t_max = all_packets[-1][0]
        print(f"Time range: {t_min:.3f}s → {t_max:.3f}s ({t_max - t_min:.1f}s)")

    # Per-UE summary
    ue_counts = {}
    for _, uid, _, _, _, _ in all_packets:
        ue_counts[uid] = ue_counts.get(uid, 0) + 1
    for uid in sorted(ue_counts):
        print(f"  {uid}: {ue_counts[uid]} packets")

    print("=" * 60)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
generate_training_data.py
=========================
End-to-end dataset generator producing strictly compliant EES JSON files
alongside a separate labels CSV file for ML training.

Features:
1. Excludes 'upload', 'download', 'background'.
2. Generates exactly 3 UEs engaging in Curriculum Learning per run.
3. Randomizes phase transitions and total duration per run for diversity.
4. Outputs multiple files (runs) to provide varied datasets.

Usage:
    python generate_training_data.py --runs 10 --min-duration 200 --max-duration 400
"""

import argparse
import csv
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = SCRIPT_DIR / "Combined_Dataset"

# ── Config ────────────────────────────────────────────────────────────
EXCLUDED_CATEGORIES = {"upload", "download", "background"}
# We define "heavy" actions and "light" actions to mix
HEAVY_ACTIONS = {"video-streaming", "gaming-online", "videocall", "audiocall"}
LIGHT_ACTIONS = {"chat", "browsing", "search", "open-email", "social-post", "directions"}


def parse_args():
    p = argparse.ArgumentParser(description="Training Dataset Generator")
    p.add_argument("--interval", type=float, default=5.0, help="Reporting interval (seconds)")
    p.add_argument("--min-duration", type=float, default=150.0, help="Minimum total simulation duration (seconds)")
    p.add_argument("--max-duration", type=float, default=450.0, help="Maximum total simulation duration (seconds)")
    p.add_argument("--runs", type=int, default=5, help="Number of distinct output runs to generate")
    p.add_argument("--base-time", type=str, default=None, help="Base time ISO 8601")
    p.add_argument("--output-dir", type=str, default="ees_training_data", help="Output directory")
    return p.parse_args()


def iso_format(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + "Z"


# ── From simulate_ue.py ───────────────────────────────────────────────
def find_session_files(action: str) -> list:
    """Find all Parquet files under Combined_Dataset/<action>/*"""
    files = []
    target_dir = DATASET_PATH / action
    if target_dir.exists():
        # Match any genre under this action
        for genre_dir in target_dir.iterdir():
            if genre_dir.is_dir():
                files.extend(list(genre_dir.glob("*.parquet")))
    return files


def read_session_packets(file_path: Path) -> tuple:
    df = pd.read_parquet(file_path)
    header = df.columns.tolist()
    
    # Fill NA with empty string, convert to str
    df = df.fillna("")
    rows = df.astype(str).values.tolist()
    return header, rows


def parse_timestamp(value: str) -> float:
    try:
        return float(value.strip().strip('"'))
    except ValueError:
        return 0.0


def build_flow_packets(action: str, start: float, duration: float, ue_ip: str) -> list:
    """Read dataset CSVs and build packet rows mapping src/dst IPs."""
    session_files = find_session_files(action)
    if not session_files:
        return []
        
    random.shuffle(session_files)
    flow_end = start + duration
    collected = []
    
    current_time = start
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
            
        try:
            ts_col = header.index("relative_time")
            dir_col = header.index("direction")
            len_col = header.index("pkt_len")
        except ValueError:
            continue
            
        delta = current_time
        last_included_rel_ts = None
        session_exhausted = True
        
        for row in rows:
            src_rel_ts = parse_timestamp(row[ts_col])
            adj_ts = src_rel_ts + delta
            
            if adj_ts > flow_end:
                session_exhausted = False
                break

            direction = row[dir_col].strip()
            pkt_len = int(float(row[len_col])) if row[len_col] else 0

            # Store absolute minimum info required for EES bucketing: timestamp, direction, len, action
            collected.append({
                "ts": adj_ts,
                "direction": direction,
                "len": pkt_len,
                "action": action, # We tag individual packets with their action for grounding
                "ue_ip": ue_ip
            })
            
            last_included_rel_ts = src_rel_ts
            
        if not rows or last_included_rel_ts is None:
            break
            
        if session_exhausted:
            current_time = last_included_rel_ts + delta + 1e-9
        else:
            break
            
    return collected


# ── Scenario Generation & Main Logic ──────────────────────────────────

def get_available_actions():
    """Scan Combined_Dataset excluding forbidden ones."""
    available = []
    if not DATASET_PATH.exists():
        return available
    for act_dir in DATASET_PATH.iterdir():
        if act_dir.is_dir() and act_dir.name not in EXCLUDED_CATEGORIES:
            available.append(act_dir.name)
    return available


def main():
    args = parse_args()
    
    available_actions = get_available_actions()
    if not available_actions:
        print(f"[ERR] No valid datasets found in {DATASET_PATH}")
        sys.exit(1)
        
    heavy_pool = list(HEAVY_ACTIONS.intersection(available_actions))
    light_pool = list(LIGHT_ACTIONS.intersection(available_actions))
    
    # Fallback if pools are empty after intersection
    if not heavy_pool: heavy_pool = available_actions
    if not light_pool: light_pool = available_actions

    print("=" * 60)
    print("Training Dataset Generator (Multi-Run mode, 3 UEs per run)")
    print(f"  Runs requested  : {args.runs}")
    print(f"  Duration Range  : {args.min_duration}s - {args.max_duration}s")
    print(f"  Reporting Intv  : {args.interval}s")
    print("=" * 60)

    # Base Time
    base_time = datetime.fromisoformat(args.base_time.replace("Z", "+00:00")) if args.base_time else datetime.now(timezone.utc)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for run_id in range(1, args.runs + 1):
        # Randomize the total simulation length for this dataset combination
        run_duration = random.uniform(args.min_duration, args.max_duration)
        correlation_id = f"training_{int(base_time.timestamp())}_run{run_id:03d}"

        # Strictly 3 UEs
        ues = [
            {"id": "UE_1", "ip": "10.0.0.1"},
            {"id": "UE_2", "ip": "10.0.0.2"},
            {"id": "UE_3", "ip": "10.0.0.3"}
        ]
        
        all_packets = []
        
        for ue in ues:
            ip = ue["ip"]
            
            # Randomize phase cuts per UE
            cut1 = random.uniform(0.2, 0.45)
            cut2 = random.uniform(0.2, 0.45)
            
            dur_p1 = run_duration * cut1
            dur_p2 = run_duration * cut2
            dur_p3 = run_duration - dur_p1 - dur_p2
            
            t_start_p1 = 0.0
            t_start_p2 = dur_p1
            t_start_p3 = dur_p1 + dur_p2
            
            # Phase 1: Baseline
            p1_h = random.choice(heavy_pool)
            pkts = build_flow_packets(p1_h, t_start_p1, dur_p1, ip)
            all_packets.extend(pkts)
            
            # Phase 2: Mixed
            p2_h = random.choice(heavy_pool)
            p2_l = random.choice(light_pool)
            pkts_h = build_flow_packets(p2_h, t_start_p2, dur_p2, ip)
            pkts_l = build_flow_packets(p2_l, t_start_p2, dur_p2, ip)
            all_packets.extend(pkts_h)
            all_packets.extend(pkts_l)
            
            # Phase 3: Extreme 
            p3_h1 = random.choice(heavy_pool)
            p3_h2 = random.choice([h for h in heavy_pool if h != p3_h1] or heavy_pool)
            pkts_e1 = build_flow_packets(p3_h1, t_start_p3, dur_p3, ip)
            pkts_e2 = build_flow_packets(p3_h2, t_start_p3, dur_p3, ip)
            all_packets.extend(pkts_e1)
            all_packets.extend(pkts_e2)
            
        all_packets.sort(key=lambda x: x["ts"])
        print(f"Run {run_id:02d} ({run_duration:5.1f}s): Generated {len(all_packets)} packets.")
        
        # ── EES Bucketing ─────────────────────────────────────────────────
        buckets = {}
        for pkt in all_packets:
            win_idx = int(pkt["ts"] // args.interval)
            ue_ip = pkt["ue_ip"]
            action = pkt["action"]
            key = (win_idx, ue_ip)
            
            if key not in buckets:
                buckets[key] = {
                    "ue_ip": ue_ip,
                    "ul_vol": 0, "dl_vol": 0,
                    "ul_pkts": 0, "dl_pkts": 0,
                    "labels": set()
                }
                
            b = buckets[key]
            b["labels"].add(action)
            
            if pkt["direction"] == "0":
                b["ul_vol"] += pkt["len"]
                b["ul_pkts"] += 1
            elif pkt["direction"] == "1":
                b["dl_vol"] += pkt["len"]
                b["dl_pkts"] += 1

        if not buckets:
            print(f"[WARN] Run {run_id:02d}: No traffic generated. Skipping.")
            continue

        max_win_idx = max(k[0] for k in buckets.keys())
        windows = {}
        
        for win_idx in range(max_win_idx + 1):
            windows[win_idx] = []
            for ue in ues:
                # Zero-filling for empty intervals
                b = buckets.get((win_idx, ue["ip"]), {
                    "ue_ip": ue["ip"],
                    "ul_vol": 0, "dl_vol": 0,
                    "ul_pkts": 0, "dl_pkts": 0,
                    "labels": {"idle"} # Empty interval = idle
                })
                windows[win_idx].append((ue["ip"], b))

        # ── Output EES JSON & Labels CSV ──────────────────────────────────
        all_notifications = []
        labels_rows = [["window_index", "start_time", "end_time", "ue_ip", "ground_truth_labels"]]

        for win_idx in sorted(windows.keys()):
            win_start = base_time + timedelta(seconds=win_idx * args.interval)
            win_end = base_time + timedelta(seconds=(win_idx + 1) * args.interval)
            notification_items = []

            for ue_ip, b in sorted(windows[win_idx], key=lambda x: x[0]):
                total_vol = b["ul_vol"] + b["dl_vol"]
                total_pkts = b["ul_pkts"] + b["dl_pkts"]

                ul_throughput = (b["ul_vol"] * 8) / args.interval
                dl_throughput = (b["dl_vol"] * 8) / args.interval
                ul_pkt_throughput = b["ul_pkts"] / args.interval
                dl_pkt_throughput = b["dl_pkts"] / args.interval

                item = {
                    "eventType": "USER_DATA_USAGE_MEASURES",
                    "timeStamp": iso_format(win_end),
                    "ueIpv4Addr": ue_ip,
                    "startTime": iso_format(win_start),
                    "userDataUsageMeasurements": [
                        {
                            "volumeMeasurement": {
                                "totalVolume": total_vol,
                                "ulVolume": b["ul_vol"],
                                "dlVolume": b["dl_vol"],
                                "totalNbOfPackets": total_pkts,
                                "ulNbOfPackets": b["ul_pkts"],
                                "dlNbOfPackets": b["dl_pkts"],
                            },
                            "throughputMeasurement": {
                                "ulThroughput": f"{ul_throughput:.0f} bps",
                                "dlThroughput": f"{dl_throughput:.0f} bps",
                                "ulPacketThroughput": f"{ul_pkt_throughput:.2f} pps",
                                "dlPacketThroughput": f"{dl_pkt_throughput:.2f} pps",
                            },
                        }
                    ],
                }
                notification_items.append(item)
                
                # Record label
                labels_str = "|".join(sorted(list(b["labels"])))
                labels_rows.append([
                    win_idx,
                    iso_format(win_start),
                    iso_format(win_end),
                    ue_ip,
                    labels_str
                ])

            notification = {
                "notificationItems": notification_items,
                "correlationId": correlation_id,
            }
            all_notifications.append(notification)

        # Write Combined JSON for this run
        combined_path = out_dir / f"training_notifications_run{run_id:03d}.json"
        with open(combined_path, "w", encoding="utf-8") as fout:
            json.dump(all_notifications, fout, indent=2, ensure_ascii=False)

        # Write Labels CSV for this run
        labels_path = out_dir / f"training_labels_run{run_id:03d}.csv"
        with open(labels_path, "w", newline="", encoding="utf-8") as fout:
            writer = csv.writer(fout)
            writer.writerows(labels_rows)

    print(f"\nSaved {args.runs} separate runs to: {out_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()

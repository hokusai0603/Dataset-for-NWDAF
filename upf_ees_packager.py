#!/usr/bin/env python3
"""
upf_ees_packager.py
===================
Converts simulated UE traffic CSV into UPF-EES notification JSON.

Aggregates packets per UE per time window and outputs USER_DATA_USAGE_MEASURES
notifications with volume and throughput measurements.

Usage:
    python upf_ees_packager.py simulated_traffic.csv --interval 30
    python upf_ees_packager.py simulated_traffic.csv --interval 10 --base-time 2026-01-14T12:00:00Z
"""

import argparse
import csv
import json
import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path


def parse_args():
    p = argparse.ArgumentParser(description="UPF-EES Notification Packager")
    p.add_argument("csv_file", help="Path to simulated_traffic.csv")
    p.add_argument(
        "--interval", type=float, default=30.0,
        help="Reporting interval in seconds (default: 30)"
    )
    p.add_argument(
        "--base-time", type=str, default=None,
        help="Simulation start time as ISO 8601 (default: current UTC)"
    )
    p.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory for JSON files (default: upf_ees_output/)"
    )
    return p.parse_args()


def load_meta(csv_path: Path) -> dict:
    """Try to load metadata from the companion .meta.json file."""
    meta_path = csv_path.with_suffix(".meta.json")
    if meta_path.exists():
        with open(meta_path, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def iso_format(dt: datetime) -> str:
    """Format datetime as ISO 8601 with Z suffix."""
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + "Z"


def main():
    args = parse_args()
    csv_path = Path(args.csv_file)
    interval = args.interval

    if not csv_path.exists():
        print(f"[ERR] File not found: {csv_path}")
        sys.exit(1)

    # Base time
    if args.base_time:
        base_time = datetime.fromisoformat(args.base_time.replace("Z", "+00:00"))
    else:
        base_time = datetime.now(timezone.utc)

    # Output directory
    out_dir = Path(args.output_dir) if args.output_dir else csv_path.parent / "upf_ees_output"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load metadata
    meta = load_meta(csv_path)
    correlation_id = meta.get("correlation_id", "")

    # ── Read CSV and bucket packets ───────────────────────────────────
    # bucket_key = (window_index, ue_ip)
    # bucket_val = {ul_vol, dl_vol, ul_pkts, dl_pkts}
    buckets = {}
    ue_ips = {}  # ue_id → ue_ip
    for ue in meta.get("UEs", []):
        ue_ips[ue.get("UE_ID")] = ue.get("IP_address")

    print("=" * 60)
    print("UPF-EES Notification Packager")
    print(f"  Input       : {csv_path}")
    print(f"  Interval    : {interval}s")
    print(f"  Base time   : {iso_format(base_time)}")
    print(f"  Correlation : {correlation_id}")
    print("=" * 60)

    with open(csv_path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            ts = float(row["adjusted_timestamp"])
            ue_id = row["ue_id"]
            ue_ip = row["ue_ip"]
            pkt_len = int(float(row["pkt_len"])) if row["pkt_len"] else 0
            direction = row["direction"].strip()

            ue_ips[ue_id] = ue_ip

            win_idx = int(ts // interval)
            key = (win_idx, ue_ip)

            if key not in buckets:
                buckets[key] = {
                    "ue_id": ue_id,
                    "ul_vol": 0, "dl_vol": 0,
                    "ul_pkts": 0, "dl_pkts": 0,
                }

            b = buckets[key]
            if direction == "0":  # Uplink
                b["ul_vol"] += pkt_len
                b["ul_pkts"] += 1
            elif direction == "1":  # Downlink
                b["dl_vol"] += pkt_len
                b["dl_pkts"] += 1

    if not buckets:
        print("[ERR] No packets found in CSV.")
        sys.exit(1)

    # ── Group by window index (Include zero-traffic UEs) ─────────────
    max_win_idx = max(k[0] for k in buckets.keys())
    windows = {}
    for win_idx in range(max_win_idx + 1):
        windows[win_idx] = []
        for ue_id, ue_ip in ue_ips.items():
            b = buckets.get((win_idx, ue_ip), {
                "ue_id": ue_id,
                "ul_vol": 0, "dl_vol": 0,
                "ul_pkts": 0, "dl_pkts": 0,
            })
            windows[win_idx].append((ue_ip, b))

    # ── Generate JSON files ──────────────────────────────────────────
    total_notifications = 0
    all_notifications = []

    for win_idx in sorted(windows.keys()):
        win_start = base_time + timedelta(seconds=win_idx * interval)
        win_end = base_time + timedelta(seconds=(win_idx + 1) * interval)

        notification_items = []

        for ue_ip, b in sorted(windows[win_idx], key=lambda x: x[0]):
            total_vol = b["ul_vol"] + b["dl_vol"]
            total_pkts = b["ul_pkts"] + b["dl_pkts"]

            ul_throughput = (b["ul_vol"] * 8) / interval  # bps
            dl_throughput = (b["dl_vol"] * 8) / interval  # bps
            ul_pkt_throughput = b["ul_pkts"] / interval   # pps
            dl_pkt_throughput = b["dl_pkts"] / interval   # pps

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

        notification = {
            "notificationItems": notification_items,
            "correlationId": correlation_id,
        }

        # Write individual window file
        out_file = out_dir / f"notification_{win_idx:04d}.json"
        with open(out_file, "w", encoding="utf-8") as fout:
            json.dump(notification, fout, indent=2, ensure_ascii=False)

        all_notifications.append(notification)
        total_notifications += len(notification_items)

    # Also write a combined file with all windows
    combined_path = out_dir / "all_notifications.json"
    with open(combined_path, "w", encoding="utf-8") as fout:
        json.dump(all_notifications, fout, indent=2, ensure_ascii=False)

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\nGenerated {len(windows)} notification windows:")
    for win_idx in sorted(windows.keys()):
        win_start = base_time + timedelta(seconds=win_idx * interval)
        win_end = base_time + timedelta(seconds=(win_idx + 1) * interval)
        ue_count = len(windows[win_idx])
        print(f"  [{win_idx:04d}] {iso_format(win_start)} → {iso_format(win_end)}  ({ue_count} UEs)")

    print(f"\nOutput directory: {out_dir}")
    print(f"  Per-window files : notification_0000.json .. notification_{max(windows.keys()):04d}.json")
    print(f"  Combined file    : all_notifications.json")
    print(f"  Total items      : {total_notifications}")
    print("=" * 60)


if __name__ == "__main__":
    main()

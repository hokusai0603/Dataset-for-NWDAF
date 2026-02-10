#!/usr/bin/env python3
"""
Dataset Transformation Script for ANLF UEcommunication Inference

Transforms MIRAGE-AppAct2024 and UTMobileNet2021 datasets to CSV format
with 1-second aggregation intervals.

Output CSV columns:
  correlationId, appLabel, duration, flowKey, eventType, timeStamp,
  ueIpv4Addr, startTime, totalVolume, ulVolume, dlVolume,
  totalNbOfPackets, ulNbOfPackets, dlNbOfPackets,
  ulThroughput, dlThroughput, ulPacketThroughput, dlPacketThroughput
"""

import os
import json
import csv
import glob
import argparse
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Any, Optional
from collections import defaultdict
import hashlib
import re

# Configuration
DEFAULT_INTERVAL_SEC = 1  # Aggregation interval in seconds (default: 1s)
SCRIPT_DIR = Path(__file__).resolve().parent
MIRAGE_PATH = str(SCRIPT_DIR / ".." / "MIRAGE-AppAct-2024")
UTMOBILE_PATH = str(SCRIPT_DIR / ".." / "UTMobileNet2021" / "Wild Test Data")
OUTPUT_PATH = str(SCRIPT_DIR / "ees_training_data")


def timestamp_to_iso(epoch: float) -> str:
    """Convert Unix epoch to ISO 8601 format."""
    return datetime.fromtimestamp(epoch, tz=timezone.utc).isoformat().replace("+00:00", "Z")


def generate_correlation_id(unique_string: str) -> str:
    """Generate a unique correlation ID from a unique string."""
    return f"corr-{hashlib.md5(unique_string.encode()).hexdigest()[:12]}"


def compute_throughput(volume_bytes: int, duration_sec: float) -> str:
    """Compute throughput in bps string format."""
    if duration_sec <= 0:
        return "0 bps"
    bps = (volume_bytes * 8) / duration_sec
    return f"{bps:.0f} bps"


def compute_packet_throughput(packets: int, duration_sec: float) -> str:
    """Compute packet throughput in pps string format."""
    if duration_sec <= 0:
        return "0.00 pps"
    pps = packets / duration_sec
    return f"{pps:.2f} pps"


def extract_ue_ip_from_flow_key(flow_key: str) -> str:
    """Extract UE IP from MIRAGE flow key (format: src_ip,src_port,dst_ip,dst_port,proto)."""
    parts = flow_key.split(",")
    if len(parts) >= 1:
        # Assume first IP is UE IP (typically 192.168.x.x pattern)
        src_ip = parts[0]
        if src_ip.startswith("192.168") or src_ip.startswith("10."):
            return src_ip
        # If not private, use destination as UE
        if len(parts) >= 3:
            return parts[2]
        return src_ip
    return "10.60.0.1"


def create_notification_item(
    ue_ip: str,
    start_time: float,
    end_time: float,
    ul_bytes: int,
    dl_bytes: int,
    ul_packets: int,
    dl_packets: int
) -> Dict[str, Any]:
    """Create a single notification item in EES format."""
    duration_sec = end_time - start_time if end_time > start_time else 1.0
    
    return {
        "eventType": "USER_DATA_USAGE_MEASURES",
        "timeStamp": timestamp_to_iso(end_time),
        "ueIpv4Addr": ue_ip,
        "startTime": timestamp_to_iso(start_time),
        "userDataUsageMeasurements": [
            {
                "volumeMeasurement": {
                    "totalVolume": ul_bytes + dl_bytes,
                    "ulVolume": ul_bytes,
                    "dlVolume": dl_bytes,
                    "totalNbOfPackets": ul_packets + dl_packets,
                    "ulNbOfPackets": ul_packets,
                    "dlNbOfPackets": dl_packets
                },
                "throughputMeasurement": {
                    "ulThroughput": compute_throughput(ul_bytes, duration_sec),
                    "dlThroughput": compute_throughput(dl_bytes, duration_sec),
                    "ulPacketThroughput": compute_packet_throughput(ul_packets, duration_sec),
                    "dlPacketThroughput": compute_packet_throughput(dl_packets, duration_sec)
                }
            }
        ]
    }


MAX_GAP_SEC = 600  # 10 minutes - max gap before considering session broken

def transform_mirage_flow(flow_key: str, flow_data: Dict, interval_sec: int) -> List[Dict]:
    """Transform a single MIRAGE biflow into EES notification items.
    
    If gaps between packets are less than MAX_GAP_SEC (10 min), fill intermediate
    intervals with zero traffic to maintain continuous time series.
    """
    packet_data = flow_data.get("packet_data", {})
    
    timestamps = packet_data.get("timestamp", [])
    ip_bytes = packet_data.get("IP_packet_bytes", [])
    directions = packet_data.get("packet_dir", [])
    
    if not timestamps or not ip_bytes:
        return []
    
    ue_ip = extract_ue_ip_from_flow_key(flow_key)
    
    # Group packets by time interval
    intervals = defaultdict(lambda: {"ul_bytes": 0, "dl_bytes": 0, "ul_packets": 0, "dl_packets": 0, 
                                       "start": float('inf'), "end": 0})
    
    for i, ts in enumerate(timestamps):
        interval_idx = int(ts // interval_sec)
        pkt_bytes = ip_bytes[i] if i < len(ip_bytes) else 0
        direction = directions[i] if i < len(directions) else 0
        
        intervals[interval_idx]["start"] = min(intervals[interval_idx]["start"], ts)
        intervals[interval_idx]["end"] = max(intervals[interval_idx]["end"], ts)
        
        if direction == 0:  # Uplink
            intervals[interval_idx]["ul_bytes"] += pkt_bytes
            intervals[interval_idx]["ul_packets"] += 1
        else:  # Downlink
            intervals[interval_idx]["dl_bytes"] += pkt_bytes
            intervals[interval_idx]["dl_packets"] += 1
    
    if not intervals:
        return []
    
    # Fill gaps: if gap between intervals < 10 min, insert zero-traffic intervals
    sorted_indices = sorted(intervals.keys())
    filled_intervals = {}
    
    for i, idx in enumerate(sorted_indices):
        filled_intervals[idx] = intervals[idx]
        
        if i < len(sorted_indices) - 1:
            next_idx = sorted_indices[i + 1]
            current_end = intervals[idx]["end"]
            next_start = intervals[next_idx]["start"]
            gap_sec = next_start - current_end
            
            # Fill gap if less than 10 minutes
            if gap_sec > 0 and gap_sec < MAX_GAP_SEC:
                for fill_idx in range(idx + 1, next_idx):
                    fill_start = fill_idx * interval_sec
                    fill_end = fill_start + interval_sec
                    filled_intervals[fill_idx] = {
                        "ul_bytes": 0, "dl_bytes": 0,
                        "ul_packets": 0, "dl_packets": 0,
                        "start": fill_start, "end": fill_end
                    }
    
    notifications = []
    for interval_idx, data in sorted(filled_intervals.items()):
        # Align to interval boundaries for strict time-series
        interval_start = interval_idx * interval_sec
        interval_end = interval_start + interval_sec
        
        item = create_notification_item(
            ue_ip=ue_ip,
            start_time=interval_start,
            end_time=interval_end,
            ul_bytes=data["ul_bytes"],
            dl_bytes=data["dl_bytes"],
            ul_packets=data["ul_packets"],
            dl_packets=data["dl_packets"]
        )
        notifications.append((item, float(interval_sec)))
    
    return notifications


def transform_mirage_file(json_path: str, app_name: str, interval_sec: int) -> List[Dict]:
    """Transform a MIRAGE JSON file to EES format."""
    print(f"  Processing: {os.path.basename(json_path)}")
    
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"    Error reading file: {e}")
        return []
    
    all_notifications = []
    
    for flow_key, flow_data in data.items():
        notifications = transform_mirage_flow(flow_key, flow_data, interval_sec)
        
        # Consistent correlation ID for the entire flow (session)
        flow_unique_string = f"{app_name}:{flow_key}"
        correlation_id = generate_correlation_id(flow_unique_string)
        
        for i, (item, duration_sec) in enumerate(notifications):
            ees_record = {
                "notificationItems": [item],
                "correlationId": correlation_id,
                "appLabel": app_name,  # For ANLF training
                "duration": round(duration_sec, 3)  # Duration in seconds
            }
            all_notifications.append(ees_record)
    
    return all_notifications


def transform_mirage(interval_sec: int = DEFAULT_INTERVAL_SEC) -> Dict[str, List[Dict]]:
    """Transform all MIRAGE datasets."""
    print("\n=== Transforming MIRAGE-AppAct2024 ===")
    
    results = {}
    
    if not os.path.exists(MIRAGE_PATH):
        print(f"MIRAGE path not found: {MIRAGE_PATH}")
        return results
    
    for app_dir in os.listdir(MIRAGE_PATH):
        app_path = os.path.join(MIRAGE_PATH, app_dir)
        if not os.path.isdir(app_path):
            continue
        
        print(f"\nApplication: {app_dir}")
        app_notifications = []
        
        json_files = glob.glob(os.path.join(app_path, "*.json"))
        for json_file in json_files[:5]:  # Limit to 5 files per app for demo
            notifications = transform_mirage_file(json_file, app_dir, interval_sec)
            app_notifications.extend(notifications)
        
        if app_notifications:
            results[app_dir] = app_notifications
            print(f"  Total notifications: {len(app_notifications)}")
    
    return results


def parse_utmobile_timestamp(time_str: str) -> float:
    """Parse UTMobileNet timestamp string to epoch float.
    Format: 'Mar 16, 2019 10:50:31.004693000 CDT'
    """
    if not time_str or time_str == "":
        return 0.0
    try:
        # Remove timezone suffix and nanoseconds (keep milliseconds)
        # Format: "Mar 16, 2019 10:50:31.004693000 CDT"
        parts = time_str.rsplit(" ", 1)  # Split off timezone
        if len(parts) == 2:
            time_str = parts[0]
        
        # Try parsing with milliseconds
        for fmt in [
            "%b %d, %Y %H:%M:%S.%f",
            "%b %d, %Y %H:%M:%S",
        ]:
            try:
                dt = datetime.strptime(time_str[:26], fmt)  # Limit fractional seconds
                return dt.timestamp()
            except ValueError:
                continue
        return 0.0
    except Exception:
        return 0.0


def transform_utmobile_file(csv_path: str, app_name: str, interval_sec: int) -> List[Dict]:
    """Transform a UTMobileNet CSV file to EES format.
    
    Groups packets by flow (IP pair) first, then by time interval within each flow.
    This ensures different connections are tracked separately.
    """
    print(f"  Processing: {os.path.basename(csv_path)}")
    
    try:
        with open(csv_path, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            rows = list(reader)
    except Exception as e:
        print(f"    Error reading file: {e}")
        return []
    
    if not rows:
        return []
    
    # Group by time interval (aggregating all flows)
    # Using dictionary: interval_idx -> metrics
    intervals = defaultdict(lambda: {
        "ul_bytes": 0, "dl_bytes": 0, "ul_packets": 0, "dl_packets": 0,
        "start": float('inf'), "end": 0, "ue_ip": None
    })
    
    # Track the UE IP seen most effectively (or first)
    # In these datasets, usually one local IP is the UE.
    detected_ue_ip = None

    for row in rows:
        try:
            time_str = row.get("frame.time", "")
            ts = parse_utmobile_timestamp(time_str)
            
            frame_len_str = row.get("frame.len", "0")
            frame_len = int(float(frame_len_str)) if frame_len_str else 0
            
            src_ip = row.get("ip.src", "") or ""
            dst_ip = row.get("ip.dst", "") or ""
        except (ValueError, TypeError):
            continue
        
        if ts == 0 or not src_ip or not dst_ip:
            continue
        
        interval_idx = int(ts // interval_sec)
        
        intervals[interval_idx]["start"] = min(intervals[interval_idx]["start"], ts)
        intervals[interval_idx]["end"] = max(intervals[interval_idx]["end"], ts)
        
        # Determine direction: private IP = UE
        is_uplink = src_ip.startswith("192.168") or src_ip.startswith("10.")
        ue_ip = src_ip if is_uplink else dst_ip
        
        if intervals[interval_idx]["ue_ip"] is None and ue_ip:
            intervals[interval_idx]["ue_ip"] = ue_ip
            if detected_ue_ip is None:
                detected_ue_ip = ue_ip
        
        if is_uplink:
            intervals[interval_idx]["ul_bytes"] += frame_len
            intervals[interval_idx]["ul_packets"] += 1
        else:
            intervals[interval_idx]["dl_bytes"] += frame_len
            intervals[interval_idx]["dl_packets"] += 1
    
    if not intervals:
        return []
    
    # Fill gaps (Global for the file)
    sorted_indices = sorted(intervals.keys())
    filled_intervals = {}
    
    for i, idx in enumerate(sorted_indices):
        filled_intervals[idx] = intervals[idx]
        
        if i < len(sorted_indices) - 1:
            next_idx = sorted_indices[i + 1]
            current_end = intervals[idx]["end"]
            next_start = intervals[next_idx]["start"]
            gap_sec = next_start - current_end
            
            if gap_sec > 0 and gap_sec < MAX_GAP_SEC:
                for fill_idx in range(idx + 1, next_idx):
                    fill_start = fill_idx * interval_sec
                    fill_end = fill_start + interval_sec
                    filled_intervals[fill_idx] = {
                        "ul_bytes": 0, "dl_bytes": 0,
                        "ul_packets": 0, "dl_packets": 0,
                        "start": fill_start, "end": fill_end,
                        "ue_ip": detected_ue_ip
                    }
    
    notifications = []
    # Use empty flow key for aggregated data, similar to MIRAGE
    flow_key = "" 
    
    # Consistent correlation ID for the entire file (session)
    session_unique_string = f"{app_name}:{os.path.basename(csv_path)}"
    correlation_id = generate_correlation_id(session_unique_string)
    
    for interval_idx, data in sorted(filled_intervals.items()):
        # Align to interval boundaries
        interval_start = interval_idx * interval_sec
        interval_end = interval_start + interval_sec
        
        item = create_notification_item(
            ue_ip=data["ue_ip"] or detected_ue_ip or "10.60.0.1",
            start_time=interval_start,
            end_time=interval_end,
            ul_bytes=data["ul_bytes"],
            dl_bytes=data["dl_bytes"],
            ul_packets=data["ul_packets"],
            dl_packets=data["dl_packets"]
        )
        ees_record = {
            "notificationItems": [item],
            "correlationId": correlation_id,
            "appLabel": app_name,
            "duration": round(float(interval_sec), 3),
            "flowKey": flow_key
        }
        notifications.append(ees_record)
    
    return notifications


def extract_app_from_filename(filename: str) -> str:
    """Extract application name from UTMobileNet filename."""
    # Format: appname_action_date_time_hash.csv
    basename = os.path.basename(filename).replace(".csv", "")
    parts = basename.split("_")
    if len(parts) >= 2:
        return f"{parts[0]}_{parts[1]}"  # e.g., "dropbox_download"
    return parts[0] if parts else "unknown"


def transform_utmobile(interval_sec: int = DEFAULT_INTERVAL_SEC) -> Dict[str, List[Dict]]:
    """Transform all UTMobileNet datasets."""
    print("\n=== Transforming UTMobileNet2021 ===")
    
    results = {}
    
    if not os.path.exists(UTMOBILE_PATH):
        print(f"UTMobileNet path not found: {UTMOBILE_PATH}")
        return results
    
    csv_files = glob.glob(os.path.join(UTMOBILE_PATH, "*.csv"))
    
    # Group by application
    app_files = defaultdict(list)
    for csv_file in csv_files:
        app_name = extract_app_from_filename(csv_file)
        app_files[app_name].append(csv_file)
    
    for app_name, files in app_files.items():
        print(f"\nApplication: {app_name}")
        app_notifications = []
        
        for csv_file in files[:3]:  # Limit to 3 files per app for demo
            notifications = transform_utmobile_file(csv_file, app_name, interval_sec)
            app_notifications.extend(notifications)
        
        if app_notifications:
            results[app_name] = app_notifications
            print(f"  Total notifications: {len(app_notifications)}")
    
    return results


CSV_COLUMNS = [
    "correlationId", "appLabel", "duration", "flowKey",
    "eventType", "timeStamp", "ueIpv4Addr", "startTime",
    "totalVolume", "ulVolume", "dlVolume",
    "totalNbOfPackets", "ulNbOfPackets", "dlNbOfPackets",
    "ulThroughput", "dlThroughput", "ulPacketThroughput", "dlPacketThroughput"
]


def flatten_record(record: Dict) -> Dict[str, Any]:
    """Flatten a nested EES record into a flat dict for CSV output."""
    item = record["notificationItems"][0]
    vol = item["userDataUsageMeasurements"][0]["volumeMeasurement"]
    thr = item["userDataUsageMeasurements"][0]["throughputMeasurement"]
    return {
        "correlationId": record.get("correlationId", ""),
        "appLabel": record.get("appLabel", ""),
        "duration": record.get("duration", ""),
        "flowKey": record.get("flowKey", ""),
        "eventType": item.get("eventType", ""),
        "timeStamp": item.get("timeStamp", ""),
        "ueIpv4Addr": item.get("ueIpv4Addr", ""),
        "startTime": item.get("startTime", ""),
        "totalVolume": vol.get("totalVolume", 0),
        "ulVolume": vol.get("ulVolume", 0),
        "dlVolume": vol.get("dlVolume", 0),
        "totalNbOfPackets": vol.get("totalNbOfPackets", 0),
        "ulNbOfPackets": vol.get("ulNbOfPackets", 0),
        "dlNbOfPackets": vol.get("dlNbOfPackets", 0),
        "ulThroughput": thr.get("ulThroughput", ""),
        "dlThroughput": thr.get("dlThroughput", ""),
        "ulPacketThroughput": thr.get("ulPacketThroughput", ""),
        "dlPacketThroughput": thr.get("dlPacketThroughput", ""),
    }


def save_csv(filepath: str, records: List[Dict]):
    """Save a list of EES records to a CSV file."""
    with open(filepath, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for record in records:
            writer.writerow(flatten_record(record))


CATEGORY_MAP = {
    "Communication": [
        "discord", "teams", "zoom", "skype", "slack", "messenger", "whatsapp",
        "signal", "telegram", "webex", "meet", "jitsimeet", "gotomeeting",
        "trueconf", "line", "kakaotalk", "hangout", "gmail", "hangouts"
    ],
    "Streaming": [
        "youtube", "netflix", "hulu", "twitch", "crunchyroll", "spotify"
    ],
    "Social_Media": [
        "facebook", "instagram", "twitter", "reddit", "pinterest", "omlet"
    ],
    "File_Transfer": [
        "google-drive"
    ],
    "Gaming": [
        "clashroyale"
    ],
    "Navigation": [
        "google-maps"
    ]
}

def get_category(app_name: str) -> str:
    """Determine category for a given app name."""
    normalized_name = app_name.lower().replace("_manual", "").replace("_", "")
    for category, apps in CATEGORY_MAP.items():
        for app_keyword in apps:
            if app_keyword in normalized_name:
                return category
    return "Other"

def save_results(mirage_results: Dict, utmobile_results: Dict):
    """Save transformed results to output directory as CSV files."""
    print("\n=== Saving Results ===")
    
    # Create output directories
    os.makedirs(os.path.join(OUTPUT_PATH, "mirage_transformed"), exist_ok=True)
    os.makedirs(os.path.join(OUTPUT_PATH, "utmobile_transformed"), exist_ok=True)
    
    all_data = []
    
    # Save MIRAGE results
    for app_name, notifications in mirage_results.items():
        output_file = os.path.join(OUTPUT_PATH, "mirage_transformed", f"{app_name}_ees.csv")
        save_csv(output_file, notifications)
        print(f"Saved: {output_file} ({len(notifications)} records)")
        all_data.extend(notifications)
    
    # Save UTMobileNet results
    for app_name, notifications in utmobile_results.items():
        output_file = os.path.join(OUTPUT_PATH, "utmobile_transformed", f"{app_name}_ees.csv")
        save_csv(output_file, notifications)
        print(f"Saved: {output_file} ({len(notifications)} records)")
        all_data.extend(notifications)
    
    # Save combined dataset
    if all_data:
        combined_file = os.path.join(OUTPUT_PATH, "combined_training_data.csv")
        save_csv(combined_file, all_data)
        print(f"\nCombined dataset: {combined_file} ({len(all_data)} total records)")


def save_categorized_dataset(mirage_results: Dict, utmobile_results: Dict):
    """Save transformed results to output directory organized by category."""
    print("\n=== Saving Categorized Results ===")
    
    # Base directory for combined dataset
    combined_dir = os.path.join(OUTPUT_PATH, "combined_dataset")
    if os.path.exists(combined_dir):
        # Safety cleanup to avoid stale files
        import shutil
        shutil.rmtree(combined_dir)
    os.makedirs(combined_dir, exist_ok=True)
    
    all_records = []
    
    # Process both result sets
    for source_name, results in [("MIRAGE", mirage_results), ("UTMobileNet", utmobile_results)]:
        for app_name, notifications in results.items():
            category = get_category(app_name)
            
            # Create category directory
            category_dir = os.path.join(combined_dir, category)
            os.makedirs(category_dir, exist_ok=True)
            
            # Save individual app file
            output_file = os.path.join(category_dir, f"{app_name}_ees.csv")
            save_csv(output_file, notifications)
            print(f"Saved [{source_name}] to {category}: {os.path.basename(output_file)}")
            
            all_records.extend(notifications)
    
    # Save combined flat dataset for easy loading
    if all_records:
        combined_file = os.path.join(OUTPUT_PATH, "combined_training_data_v2.csv")
        save_csv(combined_file, all_records)
        print(f"\nSaved Combined Categorized Dataset: {combined_file} ({len(all_records)} total records)")


def main():
    parser = argparse.ArgumentParser(description="Transform datasets to UPF-EES format")
    parser.add_argument("--interval", type=int, default=DEFAULT_INTERVAL_SEC,
                        help=f"Aggregation interval in seconds (default: {DEFAULT_INTERVAL_SEC})")
    parser.add_argument("--mirage-only", action="store_true", help="Process only MIRAGE dataset")
    parser.add_argument("--utmobile-only", action="store_true", help="Process only UTMobileNet dataset")
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("Dataset Transformation for ANLF UEcommunication Inference")
    print("=" * 60)
    print(f"Aggregation interval: {args.interval} seconds")
    print(f"Output directory: {OUTPUT_PATH}")
    
    mirage_results = {}
    utmobile_results = {}
    
    if not args.utmobile_only:
        mirage_results = transform_mirage(args.interval)
    
    if not args.mirage_only:
        utmobile_results = transform_utmobile(args.interval)
    
    save_results(mirage_results, utmobile_results)
    save_categorized_dataset(mirage_results, utmobile_results)
    
    print("\n" + "=" * 60)
    print("Transformation complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()

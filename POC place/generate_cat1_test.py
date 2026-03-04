#!/usr/bin/env python3
import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = (SCRIPT_DIR / ".." / "Combined_Dataset").resolve()

# Only CAT1 Actions (Removed gaming-online due to its bursty nature)
CAT1_ACTIONS = {"video-streaming", "videocall", "audiocall", "music-streaming"}

STANDARD_COLS = [
    "source_dataset", "app_name", "category", "activity",
    "session_id", "session_duration", "relative_time",
    "pkt_len", "l4_proto", "src_ip", "dst_ip",
    "src_port", "dst_port", "tcp_flags", "direction", "iat"
]

def parse_args():
    p = argparse.ArgumentParser(description="Cat1 Pure Training Dataset Generator")
    p.add_argument("--interval", type=float, default=5.0)
    p.add_argument("--min-duration", type=float, default=150.0)
    p.add_argument("--max-duration", type=float, default=450.0)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--base-time", type=str, default=None)
    p.add_argument("--output-dir", type=str, default="test")
    return p.parse_args()

def iso_format(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%dT%H:%M:%S") + "Z"

def find_session_files(action: str) -> list:
    files = []
    target_dir = DATASET_PATH / action
    if target_dir.exists():
        for genre_dir in target_dir.iterdir():
            if genre_dir.is_dir():
                files.extend(list(genre_dir.glob("*.parquet")))
    return files

def read_session_packets(file_path: Path) -> tuple:
    df = pd.read_parquet(file_path)
    header = df.columns.tolist()
    df = df.astype(str)
    df = df.replace({"nan": "", "<NA>": "None", "None": "", "NaN": ""}) 
    rows = df.values.tolist()
    return header, rows

def parse_timestamp(value: str) -> float:
    try:
        return float(value.strip().strip('"'))
    except ValueError:
        return 0.0

def build_flow_packets(action: str, start: float, duration: float, ue_ip: str, ue_id: str, flow_id: str) -> list:
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
        
        try:
            header, rows = read_session_packets(sf)
        except Exception:
            continue

        if not rows:
            continue
            
        if unified_header is None:
            unified_header = header
            
        try:
            ts_col = header.index("relative_time")
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

            col_map = {col: val for col, val in zip(header, row)}
            direction_val = col_map.get("direction", "").strip()
            
            if direction_val == "0":
                col_map["src_ip"] = ue_ip
            elif direction_val == "1":
                col_map["dst_ip"] = ue_ip

            session_file_name = sf.name
            
            # Pack all the fields
            pkt_dict = {
                "ue_id": ue_id,
                "ue_ip": ue_ip,
                "flow_id": flow_id,
                "adjusted_timestamp": f"{adj_ts:.9f}",
                "ts": adj_ts,
                "direction": direction_val,
                "len": int(float(col_map.get("pkt_len", 0))) if col_map.get("pkt_len") else 0,
                "action": action,
                "session_file": session_file_name
            }
            
            for col in STANDARD_COLS:
                pkt_dict[col] = col_map.get(col, "")
                
            collected.append(pkt_dict)
            last_included_rel_ts = src_rel_ts
            
        if not rows or last_included_rel_ts is None:
            break
            
        if session_exhausted:
            current_time = last_included_rel_ts + delta + 1e-9
        else:
            break
            
    return collected

def get_available_actions():
    available = []
    if not DATASET_PATH.exists():
        return available
    for act_dir in DATASET_PATH.iterdir():
        if act_dir.is_dir():
            available.append(act_dir.name)
    return available

def main():
    args = parse_args()
    available_actions = get_available_actions()
    
    cat1_pool = list(CAT1_ACTIONS.intersection(available_actions))
    if not cat1_pool:
        print("[ERR] No Cat1 actions found in dataset.")
        sys.exit(1)

    print("=" * 60)
    print("Cat1 Pure Dataset Generator")
    print(f"  Runs requested  : {args.runs}")
    print(f"  Cat1 Pool (Real-time): {cat1_pool}")
    print("=" * 60)

    base_time = datetime.fromisoformat(args.base_time.replace("Z", "+00:00")) if args.base_time else datetime.now(timezone.utc)
    out_dir = Path(SCRIPT_DIR / args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    for run_id in range(1, args.runs + 1):
        run_duration = random.uniform(args.min_duration, args.max_duration)
        correlation_id = f"training_{int(base_time.timestamp())}_run{run_id:03d}"

        ues = [
            {"id": "UE_1", "ip": "10.0.0.1"},
            {"id": "UE_2", "ip": "10.0.0.2"},
            {"id": "UE_3", "ip": "10.0.0.3"}
        ]
        
        all_packets = []
        
        for ue in ues:
            ip = ue["ip"]
            ue_id = ue["id"]
            
            num_phases = random.choice([2, 3])
            
            if num_phases == 2:
                cut1 = random.uniform(0.3, 0.7)
                dur_p1 = run_duration * cut1
                dur_p2 = run_duration - dur_p1
                t_starts = [0.0, dur_p1]
                durs = [dur_p1, dur_p2]
            else:
                cut1 = random.uniform(0.2, 0.45)
                cut2 = random.uniform(0.2, 0.45)
                dur_p1 = run_duration * cut1
                dur_p2 = run_duration * cut2
                dur_p3 = run_duration - dur_p1 - dur_p2
                t_starts = [0.0, dur_p1, dur_p1 + dur_p2]
                durs = [dur_p1, dur_p2, dur_p3]
            
            for phase_idx in range(num_phases):
                t_start = t_starts[phase_idx]
                p_dur = durs[phase_idx]
                
                # ALL phases pull from cat1_pool
                action = random.choice(cat1_pool)
                # Ensure sequential flow IDs
                flow_id = f"{ue_id}_run{run_id}_phase{phase_idx}"
                
                pkts = build_flow_packets(action, t_start, p_dur, ip, ue_id, flow_id)
                all_packets.extend(pkts)
                
                # 20% background noise, also strictly Cat1
                if random.random() < 0.2:
                    noise_action = random.choice([a for a in cat1_pool if a != action] or cat1_pool)
                    noise_flow_id = f"{ue_id}_run{run_id}_phase{phase_idx}_noise"
                    pkts_noise = build_flow_packets(noise_action, t_start, p_dur, ip, ue_id, noise_flow_id)
                    all_packets.extend(pkts_noise)
            
        all_packets.sort(key=lambda x: x["ts"])
        print(f"Run {run_id:02d} ({run_duration:5.1f}s): {len(all_packets)} packets.")
        
        buckets = {}
        for pkt in all_packets:
            win_idx = int(pkt["ts"] // args.interval)
            key = (win_idx, pkt["ue_ip"])
            
            if key not in buckets:
                buckets[key] = {
                    "ue_ip": pkt["ue_ip"],
                    "ul_vol": 0, "dl_vol": 0,
                    "ul_pkts": 0, "dl_pkts": 0,
                    "labels": set()
                }
                
            b = buckets[key]
            b["labels"].add(pkt["action"])
            if pkt["direction"] == "0":
                b["ul_vol"] += pkt["len"]
                b["ul_pkts"] += 1
            elif pkt["direction"] == "1":
                b["dl_vol"] += pkt["len"]
                b["dl_pkts"] += 1

        if not buckets:
            continue

        max_win_idx = max(k[0] for k in buckets.keys())
        windows = {}
        for win_idx in range(max_win_idx + 1):
            windows[win_idx] = []
            for ue in ues:
                b = buckets.get((win_idx, ue["ip"]), {
                    "ue_ip": ue["ip"],
                    "ul_vol": 0, "dl_vol": 0,
                    "ul_pkts": 0, "dl_pkts": 0,
                    "labels": {"idle"}
                })
                windows[win_idx].append((ue["ip"], b))

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
                    "userDataUsageMeasurements": [{
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
                    }],
                }
                notification_items.append(item)
                
                labels_str = "|".join(sorted(list(b["labels"])))
                labels_rows.append([
                    win_idx, iso_format(win_start), iso_format(win_end), ue_ip, labels_str
                ])

            notification = {
                "notificationItems": notification_items,
                "correlationId": correlation_id,
            }
            all_notifications.append(notification)

        combined_path = out_dir / f"training_notifications_run{run_id:03d}.json"
        with open(combined_path, "w", encoding="utf-8") as fout:
            json.dump(all_notifications, fout, indent=2, ensure_ascii=False)

        labels_path = out_dir / f"training_labels_run{run_id:03d}.parquet"
        df_labels = pd.DataFrame(labels_rows[1:], columns=labels_rows[0])
        df_labels.to_parquet(labels_path, index=False, compression="snappy")

        packets_path = out_dir / f"training_packets_run{run_id:03d}.parquet"
        df_packets = pd.DataFrame(all_packets)
        df_packets.to_parquet(packets_path, index=False, compression="snappy")

if __name__ == "__main__":
    main()

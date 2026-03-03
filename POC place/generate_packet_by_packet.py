#!/usr/bin/env python3
"""
generate_packet_by_packet.py
=========================
Outputs a single Parquet file containing packet-by-packet data
generated from both 'cat123' and 'cat1cat2' scenarios.
"""

import argparse
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
import pandas as pd

# ── Paths ─────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
DATASET_PATH = (SCRIPT_DIR / ".." / "Combined_Dataset").resolve()

# ── Config ────────────────────────────────────────────────────────────
CAT1_ACTIONS = {"video-streaming", "videocall", "gaming-online", "audiocall", "music-streaming"}
CAT2_ACTIONS = {"chat", "browsing", "search", "social-post", "open-email", "directions"}
CAT3_ACTIONS = {"download", "upload"}

def parse_args():
    p = argparse.ArgumentParser(description="Packet-by-Packet Training Dataset Generator (Cat123 & Cat1Cat2)")
    p.add_argument("--min-duration", type=float, default=150.0, help="Minimum total simulation duration (seconds)")
    p.add_argument("--max-duration", type=float, default=450.0, help="Maximum total simulation duration (seconds)")
    p.add_argument("--runs", type=int, default=15, help="Number of distinct output runs per scenario")
    p.add_argument("--output", type=str, default="combined_packetbypacket.parquet", help="Output Parquet file name")
    return p.parse_args()

def find_session_files(action: str) -> list:
    """Find all Parquet files under Combined_Dataset/<action>/*"""
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

def build_flow_packets(action: str, start: float, duration: float, ue_ip: str) -> list:
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

            # Keep track of from which parquet this data came
            session_file_name = sf.name
            
            collected.append({
                "ts": adj_ts,
                "direction": direction,
                "len": pkt_len,
                "action": action,
                "ue_ip": ue_ip,
                "session_file": session_file_name
            })
            
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
    if not available_actions:
        print(f"[ERR] No valid datasets found in {DATASET_PATH}")
        sys.exit(1)
        
    cat1_pool = list(CAT1_ACTIONS.intersection(available_actions))
    cat2_pool = list(CAT2_ACTIONS.intersection(available_actions))
    cat3_pool = list(CAT3_ACTIONS.intersection(available_actions))
    
    if not cat1_pool or not cat2_pool or not cat3_pool:
        print("[ERR] Missing necessary action categories (Cat1/Cat2/Cat3) in dataset.")
        sys.exit(1)

    all_scenarios_packets = []

    # Ensure Strictly 3 UEs
    ues = [
        {"id": "UE_1", "ip": "10.0.0.1"},
        {"id": "UE_2", "ip": "10.0.0.2"},
        {"id": "UE_3", "ip": "10.0.0.3"}
    ]

    # --- Scenario 1: Cat123 ---
    print(f"Generating {args.runs} runs for Cat123 scenario...")
    for run_id in range(1, args.runs + 1):
        run_duration = random.uniform(args.min_duration, args.max_duration)
        
        for ue in ues:
            ip = ue["ip"]
            num_phases = random.choice([2, 3])
            mode_choices = ["Cat1", "Cat2", "Cat3"]

            if num_phases == 2:
                dur_p1 = run_duration * random.uniform(0.3, 0.7)
                durs = [dur_p1, run_duration - dur_p1]
                m1 = random.choice(mode_choices)
                modes = [m1, random.choice([m for m in mode_choices if m != m1])]
                t_starts = [0.0, dur_p1]
            else:
                cut1 = random.uniform(0.2, 0.45)
                cut2 = random.uniform(0.2, 0.45)
                dur_p1, dur_p2 = run_duration * cut1, run_duration * cut2
                durs = [dur_p1, dur_p2, run_duration - dur_p1 - dur_p2]
                m1 = random.choice(mode_choices)
                m2 = random.choice([m for m in mode_choices if m != m1])
                m3 = random.choice([m for m in mode_choices if m != m2])
                modes = [m1, m2, m3]
                t_starts = [0.0, dur_p1, dur_p1 + dur_p2]

            for phase_idx in range(num_phases):
                mode = modes[phase_idx]
                pool = cat1_pool if mode == "Cat1" else (cat2_pool if mode == "Cat2" else cat3_pool)
                pkts = build_flow_packets(random.choice(pool), t_starts[phase_idx], durs[phase_idx], ip)
                
                # Add packets and attach scenario details
                for pkt in pkts:
                    pkt["scenario"] = "cat123"
                    pkt["run_id"] = run_id
                    all_scenarios_packets.append(pkt)
                
                # Background noise (20%)
                if random.random() < 0.2:
                    noise_mode = random.choice([m for m in mode_choices if m != mode])
                    noise_pool = cat1_pool if noise_mode == "Cat1" else (cat2_pool if noise_mode == "Cat2" else cat3_pool)
                    pkts_noise = build_flow_packets(random.choice(noise_pool), t_starts[phase_idx], durs[phase_idx], ip)
                    for pkt in pkts_noise:
                        pkt["scenario"] = "cat123"
                        pkt["run_id"] = run_id
                        all_scenarios_packets.append(pkt)

    # --- Scenario 2: Cat1Cat2 ---
    print(f"Generating {args.runs} runs for Cat1Cat2 scenario...")
    for run_id in range(1, args.runs + 1):
        run_duration = random.uniform(args.min_duration, args.max_duration)
        
        for ue in ues:
            ip = ue["ip"]
            num_phases = random.choice([2, 3])
            
            if num_phases == 2:
                dur_p1 = run_duration * random.uniform(0.3, 0.7)
                durs = [dur_p1, run_duration - dur_p1]
                modes = ["Cat1", "Cat2"] if random.choice([True, False]) else ["Cat2", "Cat1"]
                t_starts = [0.0, dur_p1]
            else:
                cut1 = random.uniform(0.2, 0.45)
                cut2 = random.uniform(0.2, 0.45)
                dur_p1, dur_p2 = run_duration * cut1, run_duration * cut2
                durs = [dur_p1, dur_p2, run_duration - dur_p1 - dur_p2]
                modes = ["Cat1", "Cat2", "Cat1"] if random.choice([True, False]) else ["Cat2", "Cat1", "Cat2"]
                t_starts = [0.0, dur_p1, dur_p1 + dur_p2]

            for phase_idx in range(num_phases):
                mode = modes[phase_idx]
                pool = cat1_pool if mode == "Cat1" else cat2_pool
                pkts = build_flow_packets(random.choice(pool), t_starts[phase_idx], durs[phase_idx], ip)
                
                for pkt in pkts:
                    pkt["scenario"] = "cat1cat2"
                    pkt["run_id"] = run_id
                    all_scenarios_packets.append(pkt)
                
                if random.random() < 0.2:
                    noise_mode = "Cat1" if mode == "Cat2" else "Cat2"
                    noise_pool = cat1_pool if noise_mode == "Cat1" else cat2_pool
                    pkts_noise = build_flow_packets(random.choice(noise_pool), t_starts[phase_idx], durs[phase_idx], ip)
                    for pkt in pkts_noise:
                        pkt["scenario"] = "cat1cat2"
                        pkt["run_id"] = run_id
                        all_scenarios_packets.append(pkt)

    if not all_scenarios_packets:
        print("[WARN] No packets generated for any scenario!")
        sys.exit(1)

    print(f"\nTotal packets generated: {len(all_scenarios_packets)}. Exporting to DataFrame...")
    
    # Sort by scenario, run_id, and timestamp
    all_scenarios_packets.sort(key=lambda x: (x["scenario"], x["run_id"], x["ts"]))

    df = pd.DataFrame(all_scenarios_packets)
    
    out_path = Path(SCRIPT_DIR / args.output)
    df.to_parquet(out_path, index=False, compression="snappy")
    
    print(f"Successfully exported all runs to {out_path.resolve()}")

if __name__ == "__main__":
    main()

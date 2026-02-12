#!/usr/bin/env python3
"""
unify_datasets.py
=================
Consolidates MIRAGE-AppAct2024 (.json) and UTMobileNet2021 (.csv) into
    Combined_Dataset/<Category>/<session_file>.csv

- No aggregation: every raw packet row is preserved.
- Sessions are identified per-flow (MIRAGE) or per-file (UTMobileNet).
- Apps are mapped to traffic categories (VoIP, Video_Streaming, etc.).
"""

import csv
import json
import os
import re
import sys
from pathlib import Path
from datetime import datetime

# ── paths ──────────────────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent
MIRAGE_PATH = SCRIPT_DIR / ".." / "MIRAGE-AppAct-2024"
UTMOBILE_PATH = SCRIPT_DIR / ".." / "UTMobileNet2021" / "Deterministic Automated Data"
OUTPUT_PATH = SCRIPT_DIR / "Combined_Dataset"

# ── MIRAGE packet_data fields to keep (exclude L4_raw_payload, is_clear,
#    heuristic, annotations) ────────────────────────────────────────────
MIRAGE_PACKET_FIELDS = [
    "timestamp", "src_port", "dst_port", "packet_dir",
    "IP_packet_bytes", "IP_header_bytes",
    "L4_payload_bytes", "L4_header_bytes",
    "iat", "TCP_win_size", "TCP_flags",
]

# ── unified columns ────────────────────────────────────────────────────
UNIFIED_METRICS = [
    "pkt_len", "l4_proto", "src_port", "dst_port", 
    "tcp_flags", "direction", "iat"
]

# ── category mapping ──────────────────────────────────────────────────
# MIRAGE app-name → category
MIRAGE_CATEGORY = {
    "ClashRoyale": "Gaming",
    "Crunchyroll": "Video_Streaming",
    "Discord":     "VoIP",
    "GotoMeeting": "VoIP",
    "JitsiMeet":   "VoIP",
    "KakaoTalk":   "VoIP",
    "Line":        "VoIP",
    "Meet":        "VoIP",
    "Messenger":   "VoIP",
    "Omlet":       "VoIP",
    "Signal":      "VoIP",
    "Skype":       "VoIP",
    "Slack":       "VoIP",
    "Teams":       "VoIP",
    "Telegram":    "VoIP",
    "Trueconf":    "VoIP",
    "Twitch":      "Video_Streaming",
    "Webex":       "VoIP",
    "WhatsApp":    "VoIP",
    "Zoom":        "VoIP",
}

# UTMobileNet  (app, action) → category
# For actions that are clearly "watching/playing video", map to Video_Streaming.
UTMOBILE_CATEGORY = {
    "dropbox":      "File_Transfer",
    "facebook":     "Social_Media",
    "gmail":        "Email",
    "google-drive": "File_Transfer",
    "google-maps":  "Navigation",
    "hangout":      "VoIP",
    "hulu":         "Video_Streaming",
    "instagram":    "Social_Media",
    "messenger":    "VoIP",
    "netflix":      "Video_Streaming",
    "pinterest":    "Social_Media",
    "reddit":       "Social_Media",
    "spotify":      "Music_Streaming",
    "twitter":      "Social_Media",
    "youtube":      "Video_Streaming",
}


# ── helpers ────────────────────────────────────────────────────────────
def safe_mkdir(path: Path):
    path.mkdir(parents=True, exist_ok=True)


def sanitize_filename(name: str) -> str:
    """Replace characters that are invalid in Windows filenames."""
    return re.sub(r'[<>:"/\\|?*]', '_', name)


def extract_mirage_app_label(filename: str) -> str:
    """
    Extract app label from MIRAGE filename.
    Example: '1620807276_us.zoom.videomeetings_mirage2020dataset_...'
      → package = 'us.zoom.videomeetings' → label = 'Zoom'
    Falls back to parent directory name if pattern doesn't match.
    """
    # Known package-name → app-label map (subset most common)
    PKG_MAP = {
        "com.supercell.clashroyale": "ClashRoyale",
        "com.crunchyroll.crunchyroid": "Crunchyroll",
        "com.discord": "Discord",
        "com.logmeininc.gotomeeting": "GotoMeeting",
        "org.jitsi.meet": "JitsiMeet",
        "com.kakao.talk": "KakaoTalk",
        "jp.naver.line.android": "Line",
        "com.google.android.apps.meetings": "Meet",
        "com.facebook.orca": "Messenger",
        "com.omlet.arcade": "Omlet",
        "org.thoughtcrime.securesms": "Signal",
        "com.skype.raider": "Skype",
        "com.Slack": "Slack",
        "com.microsoft.teams": "Teams",
        "org.telegram.messenger": "Telegram",
        "com.trueconf.videochat": "Trueconf",
        "tv.twitch.android.app": "Twitch",
        "com.cisco.webex.meetings": "Webex",
        "com.whatsapp": "WhatsApp",
        "us.zoom.videomeetings": "Zoom",
    }
    # Pattern: <epoch>_<package>_mirage...
    m = re.match(r'^\d+_(.+?)_mirage', filename)
    if m:
        pkg = m.group(1)
        if pkg in PKG_MAP:
            return PKG_MAP[pkg]
    return None  # caller should fall back to directory name


def extract_utmobile_app_action(filename: str):
    """
    Parse UTMobileNet filename.
    Example: 'dropbox_download_2019-03-16_10-50-30_4fd1c357.csv'
      → app = 'dropbox', action = 'download'
    """
    base = Path(filename).stem  # remove .csv
    parts = base.split("_")
    if len(parts) >= 2:
        app = parts[0]
        action = parts[1]
        return app, action
    return parts[0] if parts else "unknown", "unknown"


def parse_utmobile_timestamp(ts_str: str) -> float:
    """
    Parse UTMobileNet frame.time string to epoch.
    Example: 'Mar 16, 2019 10:50:31.004693000 CDT'
    """
    try:
        # Remove nanosecond portion beyond microseconds and timezone name
        # e.g. 'Mar 16, 2019 10:50:31.004693000 CDT'
        ts_str = ts_str.strip().strip('"')
        # Extract up to microseconds
        m = re.match(
            r'(\w+ \d+, \d{4} \d+:\d+:\d+)\.(\d+)\s+\w+', ts_str
        )
        if m:
            dt_part = m.group(1)
            frac = m.group(2)[:6].ljust(6, '0')  # take 6 digits (microsec)
            dt = datetime.strptime(dt_part, "%b %d, %Y %H:%M:%S")
            return dt.timestamp() + int(frac) / 1_000_000
    except Exception:
        pass
    return 0.0


def map_utmobile_tcp_flags(row: list, flag_indices: dict) -> str:
    """
    Construct TCP flags string from available columns.
    Priority:
      1. 'tcp.flags' (hex) if present.
      2. Reconstruct from individual bits (fin, syn, rst, etc.).
    """
    # 1. Try hex column
    hex_idx = flag_indices.get("hex")
    if hex_idx is not None:
        val = row[hex_idx]
        if val and val.startswith("0x"):
            try:
                msg = []
                v = int(val, 16)
                # Standard flags
                if v & 0x01: msg.append("F")
                if v & 0x02: msg.append("S")
                if v & 0x04: msg.append("R")
                if v & 0x08: msg.append("P")
                if v & 0x10: msg.append("A")
                if v & 0x20: msg.append("U")
                return "".join(msg)
            except:
                pass
    
    # 2. Reconstruct from bits
    # Check for '1' or 'True' in specific columns
    flags = []
    # Map: name -> Code
    mapping = [
        ("fin", "F"), ("syn", "S"), ("reset", "R"), ("rst", "R"),
        ("push", "P"), ("psh", "P"), ("ack", "A"), ("urg", "U"),
        ("ecn", "E"), ("cwr", "C"), ("ns", "N")
    ]
    
    for name, code in mapping:
        idx = flag_indices.get(name)
        # Check if column exists and index is within bounds
        if idx is not None and idx < len(row):
            val = row[idx].strip()
            # Some CSVs use "1", "1.0", "True"
            curr_val_lower = val.lower()
            if val == "1" or val.startswith("1.") or curr_val_lower == "true":
                flags.append(code)
                
    return "".join(flags)


def identify_local_ue_ip(rows, src_idx, dst_idx):
    """
    Identify the most likely local UE IP in a UTMobileNet trace.
    Heuristic: The IP that appears most frequently across all rows.
    """
    if src_idx is None or dst_idx is None:
        return None
    counts = {}
    for r in rows:
        if len(r) > src_idx:
            src = r[src_idx]
            counts[src] = counts.get(src, 0) + 1
        if len(r) > dst_idx:
            dst = r[dst_idx]
            counts[dst] = counts.get(dst, 0) + 1
    
    if not counts:
        return None
    
    # Return the IP with the highest total count
    return max(counts, key=counts.get)


# ── MIRAGE processing ─────────────────────────────────────────────────
def process_mirage():
    """
    Recursively scan MIRAGE dir for .json files.
    Each flow inside a JSON → one session CSV.
    """
    mirage_dir = MIRAGE_PATH.resolve()
    if not mirage_dir.exists():
        print(f"[WARN] MIRAGE path not found: {mirage_dir}")
        return

    json_files = list(mirage_dir.rglob("*.json"))
    print(f"[MIRAGE] Found {len(json_files)} JSON files")

    total_sessions = 0
    total_packets = 0

    for jf in json_files:
        # Determine app label
        app_label = extract_mirage_app_label(jf.name)
        if app_label is None:
            app_label = jf.parent.name  # fall back to directory name

        category = MIRAGE_CATEGORY.get(app_label, "Other")
        out_dir = OUTPUT_PATH / category
        safe_mkdir(out_dir)

        try:
            with open(jf, "r", encoding="utf-8") as f:
                data = json.load(f)
        except Exception as e:
            print(f"  [ERR] {jf.name}: {e}")
            continue

        # Each top-level key is a flow (session)
        for flow_key, flow in data.items():
            pkt = flow.get("packet_data", {})
            meta = flow.get("flow_metadata", {})

            # Determine number of packets
            ts_list = pkt.get("timestamp", [])
            n_pkts = len(ts_list)
            if n_pkts == 0:
                continue

            # Session info
            session_id = flow_key
            session_start = ts_list[0]
            session_duration = meta.get("BF_duration", ts_list[-1] - ts_list[0])
            activity = meta.get("BF_activity", "")

            # Build output filename
            safe_sid = sanitize_filename(flow_key)
            file_epoch = jf.name.split("_")[0] if "_" in jf.name else "0"
            out_name = f"mirage_{app_label}_{file_epoch}_{safe_sid}.csv"
            out_file = out_dir / out_name

            # Build header with unified columns at the start
            pkt_fields_no_ts = [f for f in MIRAGE_PACKET_FIELDS if f != "timestamp"]
            header = [
                "source_dataset", "app_name", "category", "activity",
                "session_id", "session_duration", "relative_time",
            ] + UNIFIED_METRICS + pkt_fields_no_ts

            first_ts = ts_list[0]

            with open(out_file, "w", newline="", encoding="utf-8") as cf:
                writer = csv.writer(cf)
                writer.writerow(header)
                for i in range(n_pkts):
                    rel_time = ts_list[i] - first_ts
                    
                    # Unified Metrics Mapping
                    # Protocol detection: if TCP_flags is non-empty, it's TCP(6), else UDP(17)
                    tcp_f = pkt.get("TCP_flags", [])
                    curr_flags = tcp_f[i] if i < len(tcp_f) else ""
                    proto = 6 if curr_flags else 17
                    
                    unified_row = [
                        pkt.get("IP_packet_bytes", [0])[i], # pkt_len
                        proto,                             # l4_proto
                        pkt.get("src_port", [0])[i],       # src_port
                        pkt.get("dst_port", [0])[i],       # dst_port
                        curr_flags,                        # tcp_flags
                        pkt.get("packet_dir", [0])[i],     # direction
                        pkt.get("iat", [0])[i],            # iat
                    ]

                    row = [
                        "MIRAGE", app_label, category, activity,
                        session_id, session_duration, rel_time,
                    ] + unified_row
                    
                    for field in pkt_fields_no_ts:
                        arr = pkt.get(field, [])
                        row.append(arr[i] if i < len(arr) else "")
                    writer.writerow(row)

            total_sessions += 1
            total_packets += n_pkts

    print(f"[MIRAGE] Done: {total_sessions} sessions, {total_packets} packets")


# ── UTMobileNet processing ────────────────────────────────────────────
def process_utmobile():
    """
    Each CSV file = one session.  Keep all original columns,
    prepend session metadata columns.
    """
    utm_dir = UTMOBILE_PATH.resolve()
    if not utm_dir.exists():
        print(f"[WARN] UTMobileNet path not found: {utm_dir}")
        return

    csv_files = list(utm_dir.glob("*.csv"))
    print(f"[UTMobileNet] Found {len(csv_files)} CSV files")

    total_sessions = 0
    total_packets = 0

    for cf_path in csv_files:
        app, action = extract_utmobile_app_action(cf_path.name)
        category = UTMOBILE_CATEGORY.get(app, "Other")
        out_dir = OUTPUT_PATH / category
        safe_mkdir(out_dir)

        session_id = cf_path.stem
        out_file = out_dir / f"utmobile_{cf_path.name}"

        try:
            with open(cf_path, "r", encoding="utf-8") as fin:
                reader = csv.reader(fin)
                orig_header = next(reader)

                # Read all rows to compute session start/duration
                rows = list(reader)
                if not rows:
                    continue

                # Parse first & last timestamp from 'frame.time' column
                ft_idx = None
                for idx, col in enumerate(orig_header):
                    if col.strip().lower() == "frame.time":
                        ft_idx = idx
                        break

                session_duration = 0.0
                first_ts = 0.0
                all_timestamps = []

                if ft_idx is not None and len(rows) > 0:
                    # Parse all timestamps once
                    all_timestamps = [
                        parse_utmobile_timestamp(r[ft_idx]) for r in rows
                    ]
                    first_ts = all_timestamps[0]
                    session_duration = all_timestamps[-1] - first_ts

                # Write output with metadata + unified columns + relative_time
                meta_header = [
                    "source_dataset", "app_name", "category", "activity",
                    "session_id", "session_duration", "relative_time",
                ] + UNIFIED_METRICS

                # Identify indices for metadata extraction
                src_ip_idx = next((i for i, c in enumerate(orig_header) if "ip.src" in c.lower()), None)
                dst_ip_idx = next((i for i, c in enumerate(orig_header) if "ip.dst" in c.lower()), None)
                pkt_len_idx = next((i for i, c in enumerate(orig_header) if "frame.len" in c.lower()), None)
                proto_idx = next((i for i, c in enumerate(orig_header) if "ip.proto" in c.lower()), None)
                tcp_src_idx = next((i for i, c in enumerate(orig_header) if "tcp.srcport" in c.lower()), None)
                udp_src_idx = next((i for i, c in enumerate(orig_header) if "udp.srcport" in c.lower()), None)
                tcp_dst_idx = next((i for i, c in enumerate(orig_header) if "tcp.dstport" in c.lower()), None)
                udp_dst_idx = next((i for i, c in enumerate(orig_header) if "udp.dstport" in c.lower()), None)
                
                # Identify all potential flag columns
                flag_indices = {}
                # Hex column
                flag_indices["hex"] = next((i for i, c in enumerate(orig_header) if c.strip().lower() == "tcp.flags"), None)
                
                # Bit columns: look for "tcp.flags.fin", "tcp.flags.syn", etc.
                for i, col in enumerate(orig_header):
                    c = col.strip().lower()
                    if c.startswith("tcp.flags."):
                        # Extract suffix: "fin", "syn", "reset", "ns" etc.
                        part = c.split(".")[-1]
                        flag_indices[part] = i

                local_ip = identify_local_ue_ip(rows, src_ip_idx, dst_ip_idx)

                with open(out_file, "w", newline="", encoding="utf-8") as fout:
                    writer = csv.writer(fout)
                    writer.writerow(meta_header + orig_header)
                    
                    prev_ts = first_ts
                    for idx, row in enumerate(rows):
                        # Compute relative_time
                        curr_ts = all_timestamps[idx] if all_timestamps else 0.0
                        rel_time = curr_ts - first_ts
                        iat = curr_ts - prev_ts
                        prev_ts = curr_ts

                        # Direction
                        direction = 0 # Default UL
                        if src_ip_idx is not None and row[src_ip_idx] != local_ip:
                            direction = 1 # DL
                        
                        # Ports
                        s_port = row[tcp_src_idx] if tcp_src_idx is not None and row[tcp_src_idx] else (row[udp_src_idx] if udp_src_idx is not None else "")
                        d_port = row[tcp_dst_idx] if tcp_dst_idx is not None and row[tcp_dst_idx] else (row[udp_dst_idx] if udp_dst_idx is not None else "")
                        
                        # TCP Flags mapping
                        # Pass the pre-calculated dictionary of indices
                        mapped_flags = map_utmobile_tcp_flags(row, flag_indices)

                        unified_vals = [
                            row[pkt_len_idx] if pkt_len_idx is not None else "", # pkt_len
                            row[proto_idx] if proto_idx is not None else "",     # l4_proto
                            s_port,                                              # src_port
                            d_port,                                              # dst_port
                            mapped_flags,                                        # tcp_flags
                            direction,                                           # direction
                            iat                                                  # iat
                        ]

                        meta_row = [
                            "UTMobileNet", app, category, action,
                            session_id, session_duration, rel_time,
                        ] + unified_vals
                        writer.writerow(meta_row + row)

                total_sessions += 1
                total_packets += len(rows)

        except Exception as e:
            print(f"  [ERR] {cf_path.name}: {e}")
            continue

    print(f"[UTMobileNet] Done: {total_sessions} sessions, {total_packets} packets")


# ── main ───────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("Unify Datasets → Combined_Dataset/<Category>/")
    print("=" * 60)
    safe_mkdir(OUTPUT_PATH)

    process_mirage()
    print()
    process_utmobile()

    # Summary
    print("\n" + "=" * 60)
    print("Output directory:", OUTPUT_PATH)
    categories = sorted(
        d.name for d in OUTPUT_PATH.iterdir() if d.is_dir()
    )
    print(f"Categories created: {len(categories)}")
    for cat in categories:
        cat_dir = OUTPUT_PATH / cat
        n_files = len(list(cat_dir.glob("*.csv")))
        print(f"  {cat}: {n_files} session files")
    print("=" * 60)


if __name__ == "__main__":
    main()

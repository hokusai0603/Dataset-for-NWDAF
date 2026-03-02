#!/usr/bin/env python3
"""
generate_iperf_script.py
========================
Reads simulated_traffic.parquet and generates an iperf3 shell script
to reproduce the simulated UE traffic.

User Specifications:
- Server IP: 140.113.110.82
- Bind UEs: Use `iperf3 -B <UE_IP>` to bind the client to the specific UE IP (10.10.0.1 ~ 10.10.0.3)
- Bandwidth: Calculate average bandwidth per flow (by aggregating bytes over duration).
- Downlink: Use `iperf3 -R` (Reverse mode) for downlink flows (direction == "1").
- Overlap/Timing: Use `sleep <offset> && iperf3 ... &` to execute flows at the correct relative time.
"""

import argparse
from pathlib import Path
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser(description="Generate iperf3 shell script from Parquet traffic data")
    p.add_argument("dataset", help="Path to simulated_traffic.parquet")
    p.add_argument("--server", type=str, default="140.113.110.82", help="iperf3 server IP")
    p.add_argument("--output", type=str, default="run_iperf_traffic.sh", help="Output shell script name")
    return p.parse_args()

def main():
    args = parse_args()
    dataset_path = Path(args.dataset)
    
    if not dataset_path.exists():
        print(f"Error: {dataset_path} does not exist.")
        return

    if dataset_path.suffix == '.parquet':
        df = pd.read_parquet(dataset_path)
    elif dataset_path.suffix == '.csv':
        df = pd.read_csv(dataset_path)
    else:
        # Fallback to try reading it as parquet, then csv
        try:
            df = pd.read_parquet(dataset_path)
        except:
            df = pd.read_csv(dataset_path)
    
    # We need to compute flow-level statistics:
    # 1. Start time (min adjusted_timestamp)
    # 2. Duration (max adjusted_timestamp - min adjusted_timestamp)
    # 3. Total Bytes (sum of pkt_len)
    # 4. UE IP (from ue_ip)
    # 5. Protocol (TCP=6, UDP=17 typically, but here we can just check if l4_proto exists or default to TCP)
    # 6. Direction (0=Uplink, 1=Downlink)

    # Make sure we treat numbers explicitly
    df["adjusted_timestamp"] = df["adjusted_timestamp"].astype(float)
    # Handle empty pct_len which might have been converted to string or NaN
    df["pkt_len"] = pd.to_numeric(df["pkt_len"], errors="coerce").fillna(0)
    
    # Group by flow_id and direction. A single flow_id might have both UL and DL packets.
    # It's better to spawn two iperf3 commands for a bidirectional flow (one reverse, one normal).
    grouped = df.groupby(["flow_id", "direction"])
    
    flows_to_run = []
    
    for (flow_id, direction), group in grouped:
        # 0 = Uplink (Client generates traffic, Server receives) -> Standard iperf3
        # 1 = Downlink (Server generates traffic, Client receives) -> iperf3 -R
        
        start_time = group["adjusted_timestamp"].min()
        end_time = group["adjusted_timestamp"].max()
        duration = end_time - start_time
        
        # If duration is 0 (only 1 packet or simultaneous), give it at least 1 second
        if duration <= 0:
            duration = 1.0

        total_bytes = group["pkt_len"].sum()
        total_packets = len(group)
        
        # Avoid division by zero
        if total_packets == 0:
            avg_pkt_len = 1500
        else:
            avg_pkt_len = max(64, int(total_bytes / total_packets))
        
        # Bandwidth in bits per second: (Total Bytes * 8) / duration
        # Convert to Mbps for iperf3
        bandwidth_bps = (total_bytes * 8) / duration
        bandwidth_mbps = bandwidth_bps / 1_000_000.0
        
        # Ensure a minimum bandwidth so iperf3 doesn't complain
        if bandwidth_mbps < 0.001:
            bandwidth_mbps = 0.001

        # Use the first row's UE IP and protocol info
        first_row = group.iloc[0]
        ue_ip = first_row["ue_ip"]
        
        # Check protocol. l4_proto = 17 is UDP. 6 is TCP.
        # If dataset says 17, we use -u. Otherwise default to TCP.
        is_udp = str(first_row.get("l4_proto", "")).strip() == "17"
        
        flows_to_run.append({
            "flow_id": flow_id,
            "direction": str(direction).strip(),
            "ue_ip": ue_ip,
            "start_time": start_time,
            "duration": duration,
            "bandwidth_mbps": bandwidth_mbps,
            "avg_pkt_len": avg_pkt_len,
            "is_udp": is_udp
        })
    
    # Sort commands by start_time so the script is easier to read
    flows_to_run.sort(key=lambda x: x["start_time"])
    
    # Generate shell script
    with open(args.output, "w", newline="\n") as f:
        f.write("#!/bin/bash\n")
        f.write("# Auto-generated iperf3 traffic script\n")
        f.write(f"# Server IP: {args.server}\n\n")
        
        f.write('echo "Starting traffic simulation..."\n\n')
        
        for flow in flows_to_run:
            duration_int = max(1, int(round(flow["duration"])))
            
            cmd_parts = [
                "iperf3",
                f"-c {args.server}",
                f"-B {flow['ue_ip']}",           # Bind to UE IP
                f"-t {duration_int}",            # Duration in seconds
                f"-b {flow['bandwidth_mbps']:.3f}M", # Target bandwidth
                f"-l {flow['avg_pkt_len']}"      # Packet length payload => inherently fixes packet rate
            ]
            
            if flow["is_udp"]:
                cmd_parts.append("-u")
                
            if flow["direction"] == "1":
                cmd_parts.append("-R")       # Reverse mode for Downlink
                
            command_str = " ".join(cmd_parts)
            
            # Format the output line using sleep to delay execution
            f.write(f"# {flow['flow_id']} (Dir: {'DL' if flow['direction'] == '1' else 'UL'})\n")
            f.write(f"(sleep {flow['start_time']:.3f} && {command_str}) &\n\n")
            
        f.write('echo "All iperf3 commands have been dispatched to background."\n')
        f.write('wait\n')
        f.write('echo "Simulation complete."\n')
        
    print(f"Successfully generated {args.output} with {len(flows_to_run)} iperf3 commands.")

if __name__ == "__main__":
    main()

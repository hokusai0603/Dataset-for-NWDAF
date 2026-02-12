# Network Traffic Dataset & UE Simulator for NWDAF

This project provides a unified, high-precision network traffic dataset and a simulation tool to generate realistic User Equipment (UE) traffic patterns for NWDAF (Network Data Analytics Function) research and training.

## 1. Project Overview

We harmonize two distinct datasets into a single, compatible format:
1.  **MIRAGE-AppAct2024**: ~20 mobile apps (Discord, Zoom, etc.) with rich metadata.
2.  **UTMobileNet2021**: ~27 apps (YouTube, Netflix, etc.) from controlled traces.
### Download link
```lin
https://drive.google.com/file/d/1M5lToHDKnkRKKvDORA8NZUX1IziELbIP/view?usp=sharing
```
### Key Features
-   **Unified Schema**: Both datasets are mapped to a common set of 7 core metrics (see below).
-   **Microsecond Precision**: Time metrics (`relative_time`, `iat`) preserve 6-9 decimal places (microsecond to nanosecond scale).
-   **Packet-Level Granularity**: No aggregation; every single packet is preserved as a row.
-   **Session Alignment**: All sessions are rebased to start at `relative_time = 0.0`.

---

## 2. Dataset Statistics

| Category | Sessions | Duration (h) | Size (MB) |
| :--- | :--- | :--- | :--- |
| Email | 127 | 0.47 | 14.80 |
| File_Transfer | 500 | 3.01 | 271.63 |
| Gaming | 451 | 7.79 | 27.03 |
| Music_Streaming | 253 | 1.97 | 159.85 |
| Navigation | 475 | 4.90 | 4,380.76 |
| Social_Media | 1,054 | 7.53 | 1,349.75 |
| Video_Streaming | 12,524 | 305.39 | 7,960.84 |
| VoIP | 94,126 | 3,967.48 | 45,681.33 |
| **TOTAL** | **109,510** | **4,298.54** | **59,846.00** |

*Note: Statistics exclude sessions with corrupt timestamps.*

---

## 3. Dataset Structure (`Combined_Dataset`)

The `unify_datasets.py` script aggregates source files into `Combined_Dataset/<Category>/<Filename>.csv`.

### Unified Column Format (First 14 Columns)
Every CSV file starts with these standardized columns:

| # | Column | Description |
| :--- | :--- | :--- |
| 1 | `source_dataset` | `MIRAGE` or `UTMobileNet` |
| 2 | `app_name` | e.g. `Discord`, `YouTube` |
| 3 | `category` | e.g. `VoIP`, `Video_Streaming` |
| 4 | `activity` | e.g. `call`, `download` (if available) |
| 5 | `session_id` | Unique identifier for the flow/session |
| 6 | `session_duration`| Total duration of the session (seconds) |
| 7 | `relative_time` | **Timestamp relative to session start (start=0.0)** |
| 8 | `pkt_len` | Packet length in bytes |
| 9 | `l4_proto` | Transport Protocol (6=TCP, 17=UDP) |
| 10 | `src_port` | Source Port |
| 11 | `dst_port` | Destination Port |
| 12 | `tcp_flags` | TCP Flags string (e.g. "PA", "S", "F") |
| 13 | `direction` | `0` = Uplink, `1` = Downlink |
| 14 | `iat` | Inter-arrival time (seconds since prev packet) |

*Note: Source-specific columns (like `ip.id` or `frame.cap_len`) are preserved after these 14 columns.*

---

## 3. Workflow

### Step 1: Generate Unified Dataset
Run this script to process raw MIRAGE JSONs and UTMobileNet CSVs into the `Combined_Dataset` folder.

```bash
python unify_datasets.py
```
*Output: `Combined_Dataset/` folder containing standardized CSVs organized by category.*

### Step 2: Simulate UE Traffic
Use `simulate_ue.py` to mix and match sessions from the unified dataset into a single, continuous user traffic stream based on a scenario config.

```bash
python simulate_ue.py scenario_config.json
```
*Output: `simulated_ue_traffic.csv` (by default)*

---

## 4. Simulation Configuration (`scenario_config.json`)

Define your traffic scenario in a JSON file:

```json
{
    "total_time": 300,              // Total simulation time in seconds
    "output_file": "my_simulation.csv",
    "flows": [
        {
            "category": "VoIP",     // Must match folder name in Combined_Dataset
            "app": "Discord",       // Optional: specific app name filter
            "start": 0.0,           // Start time (seconds)
            "duration": 60.0        // Duration to sustain this flow
        },
        {
            "category": "Video_Streaming",
            "start": 10.5,
            "duration": 120.0
        }
    ]
}
```

**How it works:**
-   The Simulator randomly picks actual session files from `Combined_Dataset` matching the criteria.
-   It **stitches** them together if the requested `duration` is longer than a single session.
-   It **shifts** the timestamps to match the `start` time in your timeline.
-   The final output preserves microsecond precision and the unified column structure.

---

## 5. Known Limitations

### UTMobileNet TCP Flags
The raw UTMobileNet2021 dataset lacks a full hexadecimal TCP flags column and is missing several flag bits (SYN, ACK, PSH, RST).
-   **Current Behavior**: We reconstruct flags from the available `tcp.flags.fin` and `tcp.flags.ns` columns.
-   **Impact**: UTMobileNet packets will mostly have empty `tcp_flags`, except for those with **FIN (F)** or **NS (N)** set.
-   **MIRAGE Data**: Unaffected; contains full TCP flag information (e.g., "SPA", "A").

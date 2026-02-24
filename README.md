# Network Traffic Dataset & UE Simulator for NWDAF

This project provides a unified, high-precision network traffic dataset and a simulation tool to generate realistic User Equipment (UE) traffic patterns for NWDAF (Network Data Analytics Function) research and training.

## 1. Project Overview

We harmonize two distinct datasets into a single, compatible format:
1.  **MIRAGE-AppAct2024**: ~20 mobile apps (Discord, Zoom, etc.) with rich metadata.
2.  **UTMobileNet2021**: ~27 apps (YouTube, Netflix, etc.) from controlled traces.
### Download link
```
https://drive.google.com/file/d/1M5lToHDKnkRKKvDORA8NZUX1IziELbIP/view?usp=sharing
```
### Key Features
-   **Unified Schema**: Both datasets are mapped to a common set of 7 core metrics (see below).
-   **Action-based Organization**: Files are organized by `<action>/<genre>/` for behavior-centric analysis.
-   **Microsecond Precision**: Time metrics (`relative_time`, `iat`) preserve 6-9 decimal places.
-   **Packet-Level Granularity**: No aggregation; every single packet is preserved as a row.
-   **Session Alignment**: All sessions are rebased to start at `relative_time = 0.0`.

---

## 2. Dataset Structure

### Directory Layout

```
Combined_Dataset/
├── videocall/VoIP/               # 31,774 sessions
├── audiocall/VoIP/               # 25,665 sessions
├── video-streaming/
│   ├── Video_Streaming/          # 10,845 sessions
│   └── VoIP/                     #  2,500 sessions
├── chat/VoIP/                    #  6,858 sessions
├── browsing/
│   ├── Social_Media/             #    656 sessions
│   ├── Video_Streaming/          #    233 sessions
│   └── Navigation/               #    209 sessions
├── background/
│   ├── VoIP/                     # 27,343 sessions
│   ├── Video_Streaming/          #  1,339 sessions
│   └── Gaming/                   #    142 sessions
├── gaming-online/Gaming/         #    309 sessions
├── social-post/Social_Media/     #    276 sessions
├── upload/File_Transfer/         #    251 sessions
├── download/
│   ├── File_Transfer/            #    250 sessions
│   └── Navigation/               #    126 sessions
├── directions/Navigation/        #    142 sessions
├── search/
│   ├── Social_Media/             #    132 sessions
│   ├── Music_Streaming/          #    126 sessions
│   └── Video_Streaming/          #    115 sessions
├── open-email/Email/             #    127 sessions
└── music-streaming/Music_Streaming/ # 127 sessions
```

**Total: 109,545 sessions** (MIRAGE: 106,145 / UTMobileNet: 3,400)

### Unified Column Format (First 16 Columns)

Every CSV file starts with these standardized columns:

| # | Column | Description |
| :--- | :--- | :--- |
| 1 | `source_dataset` | `MIRAGE` or `UTMobileNet` |
| 2 | `app_name` | e.g. `Discord`, `youtube` |
| 3 | `category` | Genre — e.g. `VoIP`, `Video_Streaming` |
| 4 | `activity` | **Unified action** — e.g. `videocall`, `browsing` |
| 5 | `session_id` | Unique identifier for the flow/session |
| 6 | `session_duration`| Total duration of the session (seconds) |
| 7 | `relative_time` | Timestamp relative to session start (start=0.0) |
| 8 | `pkt_len` | Packet length in bytes |
| 9 | `l4_proto` | Transport Protocol (6=TCP, 17=UDP) |
| 10 | `src_ip` | Source IP address |
| 11 | `dst_ip` | Destination IP address |
| 12 | `src_port` | Source Port |
| 13 | `dst_port` | Destination Port |
| 14 | `tcp_flags` | TCP Flags string (e.g. "PA", "S", "F") |
| 15 | `direction` | `0` = Uplink, `1` = Downlink |
| 16 | `iat` | Inter-arrival time (seconds since prev packet) |

---

## 3. Action Unification

Both datasets' raw action labels are mapped to **14 unified actions** via `ACTION_UNIFICATION_MAP` in `unify_datasets.py`.

### 3.1 MIRAGE (from `BF_activity` field)

Compound labels use the **primary-first rule**: in `X-Y`, X = primary activity.

| Unified Action | Raw Labels | Justification |
| :--- | :--- | :--- |
| `videocall` | `videocall`, `chat-videocall`, `videocall-chat`, `videocall-audiocall` | Large-packet ratio 11–17% (video frames >1KB) |
| `audiocall` | `audiocall`, `chat-audiocall`, `audiocall-chat`, `audiocall-videocall` | Large-packet ratio 7–10% (voice codecs <500B) |
| `video-streaming` | `video-streaming`, `video on-demand` | UL Byte% ~30%, high large-pkt ratio, IAT <20ms |
| `chat` | `chat` | Bidirectional small packets, high IAT (~0.46s) |
| `gaming-online` | `gaming-online` | UL Byte% ~22%, very large packets |
| `background` | `None`, `Unknown` | No labeled activity |

> **Key distinction**: `videocall-audiocall` → `videocall` (large-pkt rate 11.5% > pure audiocall 10.3%), while `audiocall-videocall` → `audiocall` (large-pkt rate 10.3% = pure audiocall). Similarity scores: 88.8%–97.4%.

### 3.2 UTMobileNet (from filename parsing)

| Unified Action | Raw Labels | Apps |
| :--- | :--- | :--- |
| `browsing` | `scroll-newsfeed`, `browse`, `browse-home`, `scroll-home`, `scroll-feed`, `IgSearchBrowse`, `tap-board`, `explore` | facebook, reddit, hulu, netflix, instagram, pinterest, google-maps |
| `video-streaming` | `watch-video`, `play-video` | hulu, netflix, youtube |
| `chat` | `send-message`, `hangout` | messenger, hangout |
| `search` | `search-page`, `search-music`, `catSearch` | facebook, spotify, youtube |
| `download` | `download`, `download-map` | dropbox, google-drive, google-maps |
| `upload` | `upload` | dropbox, google-drive |
| `social-post` | `post`, `post-tweet` | reddit, twitter |
| `music-streaming` | `play-music` | spotify |
| `open-email` | `open-email` | gmail |
| `directions` | `directions` | google-maps |

### 3.3 Cross-Dataset Validation

Two unified actions contain data from **both** datasets, confirming classification consistency:
- **`video-streaming/Video_Streaming`**: MIRAGE 10,466 + UTM 379 sessions
- **`chat/VoIP`**: MIRAGE 6,607 + UTM 251 sessions

---

## 4. Workflow

### Step 1: Generate Unified Dataset
Run this script to process raw MIRAGE JSONs and UTMobileNet CSVs into the `Combined_Dataset` folder.

```bash
python unify_datasets.py
```
*Output: `Combined_Dataset/<action>/<genre>/` containing standardized CSVs.*

### Step 2: Simulate Multi-UE Traffic
```bash
python simulate_ue.py scenario_config.json
```
*Output: `simulated_traffic.csv` + `simulated_traffic.meta.json`*

---

## 5. Simulation Configuration (`scenario_config.json`)

```json
{
    "correlation_id": "sim_001",
    "output_file": "simulated_traffic.csv",
    "UEs": [
        {
            "UE_ID": "UE_001",
            "IP_address": "10.10.0.1",
            "flows": [
                {
                    "action": "videocall",
                    "genre": "VoIP",
                    "app": "Discord",
                    "start": 0.0,
                    "duration": 60.0
                },
                {
                    "action": "browsing",
                    "genre": "Social_Media",
                    "start": 30.0,
                    "duration": 120.0
                }
            ]
        },
        {
            "UE_ID": "UE_002",
            "IP_address": "10.10.0.2",
            "flows": [
                {
                    "action": "video-streaming",
                    "genre": "Video_Streaming",
                    "start": 10.0,
                    "duration": 200.0
                }
            ]
        },
        {
            "UE_ID": "UE_003",
            "IP_address": "10.10.0.3",
            "flows": [
                {
                    "action": "chat",
                    "genre": "VoIP",
                    "start": 15.0,
                    "duration": 50.0
                }
            ]
        }
    ]
}
```
**How it works:**
-   The Simulator randomly picks actual session files from `Combined_Dataset` matching the criteria.
-   It **stitches** them together if the requested `duration` is longer than a single session.
-   It **shifts** the timestamps to match the `start` time in your timeline.
-   The final output preserves microsecond precision and the unified column structure.

| Field | Required | Description |
| :--- | :---: | :--- |
| `correlation_id` | No | Identifier for this simulation run |
| `output_file` | No | Output CSV filename (default: `simulated_ue_traffic.csv`) |
| `UEs[].UE_ID` | Yes | Unique identifier for the UE |
| `UEs[].IP_address` | Yes | UE IPv4 address (replaces client-side IPs) |
| `flows[].action` | Yes | Must match action folder in `Combined_Dataset/` |
| `flows[].genre` | Yes | Must match genre folder in `Combined_Dataset/` |
| `flows[].app` | No | Filter by specific app name in filename |
| `flows[].start` | Yes | Flow start time (seconds) |
| `flows[].duration` | Yes | Flow duration (seconds) |

### How it works
- Each flow runs for exactly `start + duration` seconds — there is no global time cap.
- Session files are randomly selected from `Combined_Dataset/<action>/<genre>/`.
- If a session is shorter than the requested duration, additional sessions are **stitched** end-to-end automatically.
- The configured `IP_address` **replaces** the UE-side IP in every packet (uplink → `src_ip`, downlink → `dst_ip`).
- All UEs' packets are **merged** into a single time-sorted CSV with `ue_id` and `ue_ip` columns prepended.

### Output Files

**CSV** (`simulated_traffic.csv`): Merged packet-level data.

| Column | Description |
| :--- | :--- |
| `ue_id` | UE identifier from config |
| `ue_ip` | UE IP address from config |
| `flow_id` | e.g. `UE_001_flow_0` |
| `adjusted_timestamp` | Absolute timestamp in simulation timeline |
| *(columns 5–20)* | Same 16 unified columns as Combined_Dataset |

**JSON** (`simulated_traffic.meta.json`): Session provenance — records which source files were used for each flow's `file_path` array.

---

## 6. Known Limitations

### UTMobileNet TCP Flags
The raw UTMobileNet2021 dataset lacks a full hexadecimal TCP flags column and is missing several flag bits (SYN, ACK, PSH, RST).
-   **Current Behavior**: We reconstruct flags from the available `tcp.flags.fin` and `tcp.flags.ns` columns.
-   **Impact**: UTMobileNet packets will mostly have empty `tcp_flags`, except for those with **FIN (F)** or **NS (N)** set.
-   **MIRAGE Data**: Unaffected; contains full TCP flag information (e.g., "SPA", "A").

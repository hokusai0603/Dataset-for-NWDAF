# Dataset for NWDAF

Transformed datasets for ANLF (Analytics Logical Function) UEcommunication inference training.

## Overview

This repository contains network traffic datasets transformed to **UPF-EES notify format** for NWDAF (Network Data Analytics Function) use cases.

## Data Sources

| Dataset | Applications | Description |
|---------|-------------|-------------|
| **MIRAGE-AppAct2024** | 20 apps | Discord, Zoom, Teams, Skype, WhatsApp, etc. |
| **UTMobileNet2021** | 27 apps | Dropbox, Netflix, Spotify, YouTube, etc. |

## Output Format

Each record follows the 3GPP UPF-EES `USER_DATA_USAGE_MEASURES` notify format:

```json
{
  "notificationItems": [{
    "eventType": "USER_DATA_USAGE_MEASURES",
    "timeStamp": "2026-01-14T12:00:00Z",
    "ueIpv4Addr": "10.60.0.1",
    "startTime": "2026-01-14T11:59:30Z",
    "userDataUsageMeasurements": [{
      "volumeMeasurement": {
        "totalVolume": 1572864,
        "ulVolume": 524288,
        "dlVolume": 1048576,
        "totalNbOfPackets": 2000,
        "ulNbOfPackets": 800,
        "dlNbOfPackets": 1200
      },
      "throughputMeasurement": {
        "ulThroughput": "139810 bps",
        "dlThroughput": "279620 bps",
        "ulPacketThroughput": "26.67 pps",
        "dlPacketThroughput": "40.00 pps"
      }
    }]
  }],
  "correlationId": "corr-session-001",
  "appLabel": "Discord",
  "duration": 5.0
}
```

## Usage

### Generate Transformed Data

```bash
python transform_datasets.py
```

### Options

| Option | Description |
|--------|-------------|
| `--interval N` | Time aggregation interval in seconds (default: 5) |
| `--mirage-only` | Process only MIRAGE dataset |
| `--utmobile-only` | Process only UTMobileNet dataset |

## Directory Structure

```
dataset/
├── transform_datasets.py      # Transformation script
├── ees_training_data/
│   ├── mirage_transformed/    # MIRAGE app-specific JSON files
│   ├── utmobile_transformed/  # UTMobileNet app-specific JSON files
│   └── combined_training_data.json
└── README.md
```

## Key Features

- **Per-connection tracking**: Each IP pair (flow) is tracked separately
- **Gap filling**: Maintains continuous time series (gaps < 10 min filled with zero traffic)
- **Bidirectional flow aggregation**: UL/DL properly attributed per connection
- **Application labels**: Ready for supervised ML training

## License

Research use only. Original datasets retain their respective licenses.

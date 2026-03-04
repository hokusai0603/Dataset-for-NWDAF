#!/bin/bash
# Auto-generated iperf3 traffic script
# Server IP: 140.113.110.82

echo "Starting traffic simulation..."

# UE_001_flow_0 (Dir: UL)
(sleep 0.000 && iperf3 -c 140.113.110.82 -p 5201 -B 10.10.0.1 -t 1 -b 0.005M -l 64) &

# UE_001_flow_0 (Dir: DL)
(sleep 0.005 && iperf3 -c 140.113.110.82 -p 5202 -B 10.10.0.1 -t 1 -b 0.006M -l 64 -R) &

# UE_001_flow_1 (Dir: DL)
(sleep 5.252 && iperf3 -c 140.113.110.82 -p 5204 -B 10.10.0.1 -t 20 -b 0.082M -l 383 -R) &

# UE_001_flow_1 (Dir: UL)
(sleep 5.255 && iperf3 -c 140.113.110.82 -p 5203 -B 10.10.0.1 -t 19 -b 0.050M -l 264) &

# UE_002_flow_0 (Dir: UL)
(sleep 10.000 && iperf3 -c 140.113.110.82 -p 5205 -B 10.10.0.2 -t 15 -b 0.008M -l 517) &

# UE_002_flow_0 (Dir: DL)
(sleep 10.012 && iperf3 -c 140.113.110.82 -p 5206 -B 10.10.0.2 -t 15 -b 0.004M -l 393 -R) &

# UE_003_flow_0 (Dir: UL)
(sleep 15.000 && iperf3 -c 140.113.110.82 -p 5207 -B 10.10.0.3 -t 11 -b 0.014M -l 64) &

# UE_003_flow_0 (Dir: DL)
(sleep 15.047 && iperf3 -c 140.113.110.82 -p 5208 -B 10.10.0.3 -t 11 -b 0.629M -l 1448 -R) &

echo "All iperf3 commands have been dispatched to background."
wait
echo "Simulation complete."

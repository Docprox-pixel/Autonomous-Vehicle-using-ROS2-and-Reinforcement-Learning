#!/bin/bash
# usage: tools/speed_sweep.sh "6 8 10 12 13.9" [SECONDS]
# For each cruise speed: restart the simulation (same start every time), healthy driver (vitals),
# drive toward the hospital for SECONDS and measure stability (tools/stability_probe.py).
ROOT=$(cd "$(dirname "$0")/.." && pwd); cd $ROOT
SPEEDS=${1:-"6 8 10 12 13.9"}; DUR=${2:-70}
OUT=$ROOT/tools/sim/logs/speed_sweep.txt; : > $OUT
for V in $SPEEDS; do
  tools/sim/start.sh sweep_$V world:=village.wbt rviz:=false stack:=true > /dev/null 2>&1
  source /opt/ros/jazzy/setup.bash; source install/setup.bash
  ros2 param set /local_planner cruise_speed $V > /dev/null 2>&1
  (timeout $((DUR + 20)) python3 tools/driver_vitals.py normal $((DUR + 15)) > /dev/null 2>&1 &)
  sleep 2
  timeout 10 ros2 topic pub --once /route/go_to std_msgs/msg/String "{data: hospital}" > /dev/null 2>&1
  timeout $((DUR + 10)) python3 tools/stability_probe.py $DUR "cruise $V m/s ($(python3 -c "print(round($V*3.6))") km/h)" | tee -a $OUT
done
echo "results: $OUT"

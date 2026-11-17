#!/bin/bash
# usage: tools/sim/start.sh LOG_NAME [launch args...]
# Restart the simulation cleanly (kills every node of a previous run), launch sim.launch.py in the
# background and wait until the vehicle bridge is ready. Log: tools/sim/logs/LOG_NAME.log
# NOTE: do not run this from a shell command line that contains node names (pgrep -f would match it).
ROOT=$(cd "$(dirname "$0")/../.." && pwd); LOGS=$ROOT/tools/sim/logs; mkdir -p $LOGS
RUN=$1; shift
cd $ROOT
PAT="[r]os2 launch auto_car|[w]ebots-bin|[w]ebots-controller|[w]ebots_ros2_driver/driver|[t]raffic_light|[r]viz2|[e]kf_node|[a]sync_slam|[r]os2_supervisor|[o]ctomap_server|[c]loud_preprocess|[o]bject_tracker|[s]afety_shield|[s]umo|[c]ontrol_node|[p]lanning_node|[p]erception_node|[l]ocal_planner|[r]l_planner|[r]oad_perception|[o]bject_detector|[r]oute_planner|[t]raffic_rules|[d]river_monitor[.]py|[c]rossroads_traffic"
for p in $(pgrep -f "$PAT"); do kill $p 2>/dev/null; done; sleep 3
for p in $(pgrep -f "$PAT"); do kill -9 $p 2>/dev/null; done
export WEBOTS_HOME=${WEBOTS_HOME:-$HOME/apps/webots} SUMO_HOME=${SUMO_HOME:-/usr/share/sumo}
# Hybrid-graphics laptop: render Webots (cameras + lidars, every step) on the NVIDIA GPU, not the
# integrated one (on the iGPU the simulation ran at ~0.2x real time and the car's control loop at ~2 Hz)
export __NV_PRIME_RENDER_OFFLOAD=1 __GLX_VENDOR_LIBRARY_NAME=nvidia __VK_LAYER_NV_optimus=NVIDIA_only
source /opt/ros/jazzy/setup.bash; source install/setup.bash
LOG=$LOGS/$RUN.log
setsid ros2 launch auto_car sim.launch.py "$@" > $LOG 2>&1 < /dev/null &
for i in $(seq 1 120); do grep -q "vehicle_bridge ready" $LOG 2>/dev/null && break; sleep 2; done
sleep 6
grep -E "Traceback|process has died" $LOG | head -3
echo "started $RUN"

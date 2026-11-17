#!/bin/bash
# Record the demo video: the car drives itself to the hospital and parks in a free bay.
#
#   tools/record_demo.sh [OUT_DIR] [SPEEDUP]        (default docs/demo, 3x)
#
# Webots records its own 3-D view (Supervisor.movieStartRecording, started by our SUMO supervisor
# controller when AUTO_CAR_MOVIE is set) - other windows on the screen cannot get into the video.
# The camera follows the car (Viewpoint follow "vehicle"). Steps:
#   1. start the simulation (no RViz) with recording on, keep the simulated driver responsive,
#   2. 'go to hospital': route, roundabout, junctions, traffic lights - then the nearest free
#      parking bay is chosen and the car parks,
#   3. stop recording 6 s after PARKED (or after 15 min), wait until Webots has written the file,
#   4. ffmpeg: a SPEEDUP x faster H.264 1080p copy for LinkedIn (demo.mp4) next to the raw movie.
ROOT=$(cd "$(dirname "$0")/.." && pwd)
OUT_DIR=${1:-$ROOT/docs/demo}
SPEED=${2:-3}
mkdir -p "$OUT_DIR"
RAW=$OUT_DIR/demo_raw.mp4
rm -f "$RAW" "$RAW.stop"
export AUTO_CAR_MOVIE=$RAW AUTO_CAR_MOVIE_SIZE=1920x1080
cd "$ROOT"
tools/sim/start.sh demo world:=village.wbt rviz:=false stack:=true | tail -1
LOG=$ROOT/tools/sim/logs/demo.log
source /opt/ros/jazzy/setup.bash && source install/setup.bash
setsid python3 tools/driver_vitals.py normal 1800 > /dev/null 2>&1 &
sleep 8                                                    # let the map align and traffic start
ros2 topic pub --once /route/go_to std_msgs/msg/String "{data: 'hospital'}" > /dev/null
echo "driving to the hospital (recording) ..."
for i in $(seq 1 180); do
    grep -qE "PARKED in bay|PARKED at" "$LOG" && break
    sleep 5
done
grep -E "PARKED" "$LOG" | tail -1
sleep 6
touch "$RAW.stop"
for i in $(seq 1 120); do grep -q "demo movie saved" "$LOG" && break; sleep 2; done
grep "demo movie saved" "$LOG" | tail -1
ffmpeg -y -loglevel error -i "$RAW" -filter:v "setpts=PTS/$SPEED,scale=1920:-2,fps=30" -an \
       -c:v libx264 -preset slow -crf 22 -pix_fmt yuv420p -movflags +faststart "$OUT_DIR/demo.mp4"
ls -la "$OUT_DIR"

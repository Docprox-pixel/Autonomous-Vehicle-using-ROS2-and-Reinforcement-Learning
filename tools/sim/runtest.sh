#!/bin/bash
# usage: [TMO=seconds] [STACK=false] tools/sim/runtest.sh LOG_NAME WORLD -- test/xyz.py [args]
# Restart the sim with WORLD (no planner unless STACK=true; the tests start their own), run a test.
# The driver monitor is OFF in tests (nobody touches the wheel) unless DRIVER_MONITOR=true.
ROOT=$(cd "$(dirname "$0")/../.." && pwd); RUN=$1; WORLD=$2; shift 3
$ROOT/tools/sim/start.sh $RUN stack:=${STACK:-false} world:=$WORLD driver_monitor:=${DRIVER_MONITOR:-false} ${LAUNCH_ARGS} >/dev/null
cd $ROOT; source /opt/ros/jazzy/setup.bash; source install/setup.bash
timeout ${TMO:-150} python3 "$@"

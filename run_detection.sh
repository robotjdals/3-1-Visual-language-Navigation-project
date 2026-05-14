#!/usr/bin/env bash
set -euo pipefail

source /home/min/DINO_ws/install/setup.bash

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"

exec python3 -m efficientnav_detection.detection_node

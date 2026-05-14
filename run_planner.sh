#!/usr/bin/env bash
set -euo pipefail

source /home/min/DINO_ws/install/setup.bash

export ROS_DOMAIN_ID="${ROS_DOMAIN_ID:-30}"
export EFFICIENTNAV_USE_ROS2_DETECTION="${EFFICIENTNAV_USE_ROS2_DETECTION:-1}"
export EFFICIENTNAV_PLANNER_MODEL_PATH="${EFFICIENTNAV_PLANNER_MODEL_PATH:-/home/min/test/models/SmolVLM-500M-Instruct}"
export EFFICIENTNAV_CLIP_PATH="${EFFICIENTNAV_CLIP_PATH:-/home/min/models/clip-vit-base-patch32}"

cd /home/min/test/EfficientNav
exec python3 efficientnav.py

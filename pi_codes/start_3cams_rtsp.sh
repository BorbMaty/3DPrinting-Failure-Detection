#!/bin/bash
HOST="localhost"
PORT="8554"
declare -A PIDS

start_cam() {
  local name=$1
  local dev=$2
  local flip=$3

  echo "[*] Starting $name: $dev -> rtsp://$HOST:$PORT/$name"
  if [ "$flip" = "1" ]; then
    gst-launch-1.0 -e \
      v4l2src device="$dev" \
      ! image/jpeg,width=1920,height=1080,framerate=30/1 \
      ! jpegdec ! videoconvert \
      ! videoflip method=rotate-180 \
      ! v4l2h264enc extra-controls="controls,repeat_sequence_header=1" \
      ! rtspclientsink location="rtsp://$HOST:$PORT/$name" \
      > ~/${name}.log 2>&1 &
  else
    gst-launch-1.0 -e \
      v4l2src device="$dev" \
      ! image/jpeg,width=1280,height=720,framerate=30/1 \
      ! jpegdec ! videoconvert \
      ! v4l2h264enc extra-controls="controls,repeat_sequence_header=1" \
      ! rtspclientsink location="rtsp://$HOST:$PORT/$name" \
      > ~/${name}.log 2>&1 &
  fi
  PIDS[$name]=$!
  echo "[*] $name PID=${PIDS[$name]}"
}

start_cam "cam1" "/dev/v4l/by-id/usb-SunplusIT_Inc_FHD_Camera_Microphone_01.00.00-video-index0" "0"
start_cam "cam2" "/dev/v4l/by-id/usb-Camera_Vendor_Conference_Camera_00.00.01-video-index0" "0"
start_cam "cam3" "/dev/v4l/by-id/usb-HD_Web_Camera_HD_Web_Camera_Ucamera001-video-index0" "0"

echo "[*] Running with auto-restart. Press Ctrl+C to stop."
while true; do
  sleep 5
  for name in cam1 cam2 cam3; do
    pid=${PIDS[$name]}
    if ! kill -0 "$pid" 2>/dev/null; then
      echo "[!] $name died, restarting..."
      sleep 2
      case $name in
        cam1) start_cam "cam1" "/dev/v4l/by-id/usb-SunplusIT_Inc_FHD_Camera_Microphone_01.00.00-video-index0" "0" ;;
        cam2) start_cam "cam2" "/dev/v4l/by-id/usb-Camera_Vendor_Conference_Camera_00.00.01-video-index0" "0" ;;
        cam3) start_cam "cam3" "/dev/v4l/by-id/usb-HD_Web_Camera_HD_Web_Camera_Ucamera001-video-index0" "0" ;;
      esac
    fi
  done
done

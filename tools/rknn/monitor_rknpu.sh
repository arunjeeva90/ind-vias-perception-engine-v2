#!/usr/bin/env bash
set -euo pipefail

echo "[INFO] Searching possible RKNPU/NPU status files..."
FILES=$(
  {
    find /sys -type f \( -iname "*load*" -o -iname "*busy*" -o -iname "*freq*" -o -iname "*clk*" -o -iname "*util*" \) 2>/dev/null | grep -Ei "rknpu|npu|fdab0000|devfreq" || true
    find /sys/kernel/debug -type f \( -iname "*load*" -o -iname "*busy*" -o -iname "*freq*" -o -iname "*clk*" -o -iname "*util*" \) 2>/dev/null | grep -Ei "rknpu|npu|fdab0000|devfreq" || true
  } | sort -u
)

if [ -z "$FILES" ]; then
  echo "[WARN] No obvious NPU load/freq files found."
  echo "[INFO] Run this to inspect manually:"
  echo "sudo find /sys /proc /sys/kernel/debug -iname '*rknpu*' -o -iname '*npu*' 2>/dev/null"
  exit 0
fi

echo "[INFO] Monitoring files:"
echo "$FILES"
echo

while true; do
  clear
  date
  echo
  for f in $FILES; do
    if [ -r "$f" ]; then
      v=$(cat "$f" 2>/dev/null | head -5 | tr '\n' ' ')
      printf "%-90s %s\n" "$f" "$v"
    fi
  done
  sleep 1
done

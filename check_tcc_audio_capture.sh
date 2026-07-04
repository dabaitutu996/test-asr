#!/bin/sh

set -eu

MINUTES="${1:-10}"

echo "== TCC audio capture logs in last ${MINUTES}m =="
/usr/bin/log show --style compact --last "${MINUTES}m" --predicate 'subsystem == "com.apple.TCC"' \
  | grep -n -C 3 -E 'kTCCServiceAudioCapture|NSAudioCaptureUsageDescription|Refusing authorization request|usage description:|AUTHREQ_PROMPTING|accessing=\{TCCDProcess: identifier=.*diag_devices|accessing=\{TCCDProcess: identifier=.*16_asr_compare_tui' \
  || true

echo
echo "Read it like this:"
echo "  - If you see 'service=kTCCServiceAudioCapture', the log is on the right permission."
echo "  - If you then see 'without NSAudioCaptureUsageDescription key', the terminal host app is being denied Audio Capture by TCC."
echo "  - If you see none of those lines, rerun this script immediately after starting diag_devices or the engine example."

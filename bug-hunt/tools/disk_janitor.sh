#!/usr/bin/env bash
# During a bug-hunt run: every 10 minutes delete repro / hunt test binaries, fingerprints and
# incremental caches older than 90 minutes (a later run of that test just relinks it), and log
# the free space. Each repro is its own test binary and a batch builds hundreds of them, which
# can fill the disk (a full disk got the container of run 2026-11-14-batch7 reclaimed).
# Runs until killed or until $1 seconds have passed (default 7000). Start it in the background
# next to the workflow, e.g. `bash bug-hunt/tools/disk_janitor.sh 9000`.
cd "$(dirname "${BASH_SOURCE[0]}")/../.." || exit 1
END=$(( $(date +%s) + ${1:-7000} ))
while [ "$(date +%s)" -lt "$END" ]; do
    for T in jix/target/debug jix/target/release jix/target/miri/x86_64-unknown-linux-gnu/debug \
             target-i686/i686-unknown-linux-musl/debug target-i686/i686-unknown-linux-musl/release; do
        [ -d "$T" ] || continue
        find "$T/deps" -maxdepth 1 \( -name 'repro_*' -o -name 'hunt_*' \) -mmin +90 -delete 2>/dev/null
        find "$T/.fingerprint" "$T/incremental" -maxdepth 1 -mindepth 1 \( -name 'repro_*' -o -name 'hunt_*' \) -mmin +90 -exec rm -rf {} + 2>/dev/null
    done
    echo "$(date +%H:%M) $(df -h / | tail -1 | awk '{print $4 " free"}') target=$(du -sh jix/target 2>/dev/null | cut -f1)"
    sleep 600
done

#!/bin/bash
# Serialised Met acquisition + benchmark finalisation.
#
# Why serialised: the parallel department sweep earlier tripped a Cloudflare
# challenge on the Open Access API, so every request now goes through
# fetch_strata.py (v1.1 search first, 0.35 s between calls, one department per
# process) and the departments run one after another, never concurrently.
#
# Why it waits for the running fetch: manifest_met.csv is opened in append mode
# by fetch_strata.py. Truncating or replacing that file while the writer holds it
# leaves the writer appending to an orphaned inode, i.e. silently lost rows.
set -u
cd /data/abstrabench || exit 1

while kill -0 3118 2>/dev/null; do sleep 20; done
echo "[chain_fetch] dept19 writer exited $(date)"

for spec in "8 40" "14 25"; do
  set -- $spec
  dept=$1; target=$2
  rows=$(($(wc -l < manifest_met.csv) - 1))
  if [ "$rows" -ge 240 ]; then
    echo "[chain_fetch] $rows Met rows already collected, stopping early $(date)"
    break
  fi
  nice -n 18 python3 -u scripts/fetch_strata.py --dept "$dept" --target "$target"
  echo "[chain_fetch] dept $dept rc=$? $(date)"
done

nice -n 18 python3 -u scripts/build_benchmark.py --verify-only
echo "[chain_fetch] verify rc=$? $(date)"
echo "[chain_fetch] ALL DONE $(date)"

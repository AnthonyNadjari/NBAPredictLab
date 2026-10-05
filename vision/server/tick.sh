#!/usr/bin/env bash
# One tick of the reply bot on the server (systemd timer, every 5 min).
#   1. sync the repo (schedule, code, data the replies are grounded in)
#   2. run a session if a scheduled slot is due or the control panel asked for one
#   3. push docs/vision/runs.json + stats.json back for the panel
set -uo pipefail
BASE=/opt/nba-vision
REPO=$BASE/repo
cd "$REPO" || exit 1
# (the repo is synced by the service's ExecStartPre; this script runs from a copy in /tmp,
#  so the git resets below can't change it under bash's feet)

# The box also runs the TCG drop bot: never compete with it.
LOAD=$(cut -d' ' -f1 /proc/loadavg)
if awk -v l="$LOAD" 'BEGIN{exit !(l > 3.0)}'; then
  echo "server busy (load $LOAD): skipping this tick"; exit 0
fi

# shellcheck disable=SC1091
source "$BASE/venv/bin/activate"
source vision/server/env.sh   # .env + secrets relayed from the control panel
export NBAVISION_STATE_DIR=$BASE/state
export NBAVISION_BROWSER_CHANNEL=chromium   # full Chromium in new headless mode (less detectable than the headless shell)
export PYTHONIOENCODING=utf-8
# Memory-lean browsing on the shared box (2.5 GB cap): one tab, no images/video/fonts
export PARALLEL_TABS=1 BLOCK_MEDIA=1
# off-season (before the regular season starts on Oct 20) there are few fresh tweets
[ "$(date -u +%Y%m%d)" -lt 20261020 ] && export MAX_MINUTES_SINCE_POST=360
export CHROME_EXTRA_ARGS="--disable-gpu --renderer-process-limit=2 --disable-extensions --disable-background-networking"

MODE=$(python vision/tools/server_request.py)   # none | session | dry | login  (+ max replies)
read -r KIND MAX <<<"$MODE"
if [ "$KIND" = "none" ]; then
  OUT=$(python vision/tools/schedule_gate.py)
  echo "$OUT"
  case "$OUT" in *"due=True"*) KIND=session; MAX="";; *) exit 0;; esac
  export NBAVISION_SLOT=$(echo "$OUT" | sed -n 's/.*slot=\([^ ]*\).*/\1/p')
fi

export DRY_RUN=false NBAVISION_LOGIN_CHECK=false MAX_REPLIES="${MAX:-}"
[ "$KIND" = "dry" ] && export DRY_RUN=true
[ "$KIND" = "login" ] && export NBAVISION_LOGIN_CHECK=true
export NBAVISION_RUN_ID="server-$(date -u +%Y%m%dT%H%M%S)"
echo "Running: $KIND (max=${MAX:-default})"

mkdir -p vision/logs
(cd vision && timeout 9000 python main.py) 2>&1 | tee "$BASE/logs/$NBAVISION_RUN_ID.log"

# push the panel data (re-applied on top of the latest remote state if someone pushed meanwhile)
mkdir -p /tmp/nbavision-status && cp docs/vision/stats.json docs/vision/runs.json /tmp/nbavision-status/
for i in 1 2 3; do
  git fetch -q origin main && git reset -q --hard origin/main
  python vision/tools/merge_status.py /tmp/nbavision-status
  git add docs/vision/stats.json docs/vision/runs.json
  git diff --cached --quiet && break
  git -c user.name="nbavision-server" -c user.email="nbavision@users.noreply.github.com" \
      commit -q -m "Vision: $KIND $NBAVISION_RUN_ID"
  git push -q origin HEAD:main && break
  sleep 5
done
# keep the last 200 local logs
ls -1t "$BASE/logs"/*.log 2>/dev/null | tail -n +201 | xargs -r rm -f

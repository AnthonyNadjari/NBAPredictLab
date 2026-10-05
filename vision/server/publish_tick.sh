#!/usr/bin/env bash
# Post queued threads through the logged-in browser (systemd timer nbapublish, every 2 min).
#   control panel "Publier" -> publish_thread.yml queues it in docs/vision/publish_queue.json
#   -> this renders the cards, posts tweet by tweet, marks the game published and pushes back.
# Separate checkout, browser profile and service from the reply bot: a publish never waits
# for a 2-hour reply session, and the two never share a git index or a Chromium profile.
set -uo pipefail
BASE=/opt/nba-vision
REPO=$BASE/publish-repo
cd "$REPO" || exit 1

# The box also runs the TCG drop bot: back off when it's really busy (a thread can wait 2 min)
LOAD=$(cut -d' ' -f1 /proc/loadavg)
if awk -v l="$LOAD" 'BEGIN{exit !(l > 6.0)}'; then
  echo "server busy (load $LOAD): next tick"; exit 0
fi

# shellcheck disable=SC1091
source "$BASE/venv/bin/activate"
export NBAVISION_STATE_DIR=$BASE/publish-state
mkdir -p "$NBAVISION_STATE_DIR"
TEXTS=$NBAVISION_STATE_DIR/texts.json
REQ=$(python vision/tools/publish_queue.py next "$TEXTS")
[ "$REQ" = "none" ] && exit 0
read -r RID GAME DRY <<<"$REQ"
echo "Publishing $GAME (request $RID, dry run: $DRY)"

source vision/server/env.sh   # .env + secrets relayed from the control panel
# Start from the reply bot's latest logged-in session
[ -f "$BASE/state/state.json" ] && cp -u "$BASE/state/state.json" "$NBAVISION_STATE_DIR/state.json"
export NBAVISION_BROWSER_CHANNEL=chromium PYTHONIOENCODING=utf-8 PUBLISH_BACKEND=browser BLOCK_MEDIA=0
export TW_DRY_RUN=$DRY THREAD_RESULT_FILE=$REPO/thread_result.json
THREAD_TEXTS_JSON=$(cat "$TEXTS" 2>/dev/null || true); export THREAD_TEXTS_JSON
export CHROME_EXTRA_ARGS="--disable-gpu --renderer-process-limit=2 --disable-extensions --disable-background-networking"
# Dry run: exercise the reply composer on one of our own past tweets (nothing is sent)
if [ "$DRY" = "true" ]; then
  PUBLISH_DRY_REPLY_TO=$(python -c "import json; print(json.load(open('data/tweet_history.json'))[-1]['tweet_id'])" 2>/dev/null || true)
  export PUBLISH_DRY_REPLY_TO
fi

RESULT=/tmp/nbapublish-result.json
rm -f thread_result.json "$RESULT"
mkdir -p vision/logs
timeout 900 python scripts/publish_single_thread.py "$GAME" 2>&1 | tee "$BASE/logs/publish-$RID.log"
cp -f thread_result.json "$RESULT" 2>/dev/null \
  || echo '{"posted": 0, "error": "publisher crashed before posting (see the server log)"}' > "$RESULT"

# Mark published + log the outcome, re-applied on top of the latest remote state each try
for i in 1 2 3; do
  git fetch -q origin main && git reset -q --hard origin/main
  cp -f "$RESULT" thread_result.json
  if [ "$DRY" != "true" ] && python -c "import json,sys; sys.exit(0 if json.load(open('thread_result.json')).get('posted') else 1)"; then
    python scripts/mark_published.py "$GAME"
  fi
  python vision/tools/publish_queue.py log "$RID" "$GAME" thread_result.json
  git add docs/pending_games.json docs/vision/publish_log.json
  git diff --cached --quiet && break
  git -c user.name="nbavision-server" -c user.email="nbavision@users.noreply.github.com" \
      commit -q -m "Publish: $GAME ($([ "$DRY" = true ] && echo dry run || echo posted))"
  git push -q origin HEAD:main && break
  sleep 5
done
rm -f thread_result.json
ls -1t "$BASE/logs"/publish-*.log 2>/dev/null | tail -n +61 | xargs -r rm -f

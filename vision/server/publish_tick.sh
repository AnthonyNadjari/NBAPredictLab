#!/usr/bin/env bash
# Post queued threads through the logged-in browser on the server.
#   control panel "Publier" -> publish_thread.yml queues it in docs/vision/publish_queue.json
#   -> this renders the cards, posts tweet by tweet, marks the game published and pushes back.
# Own checkout, browser profile and service, separate from the reply bot: a publish never
# waits for a long reply session, and the two never share a git index or a Chromium profile.
set -uo pipefail
BASE=/opt/nba-vision
REPO=$BASE/publish-repo
cd "$REPO" || exit 1
git fetch -q origin main && git reset -q --hard origin/main || { echo "git sync failed"; exit 1; }

# The box also runs the TCG drop bot: back off when it's really busy (a thread can wait 2 min)
LOAD=$(cut -d' ' -f1 /proc/loadavg)
if awk -v l="$LOAD" 'BEGIN{exit !(l > 6.0)}'; then
  echo "server busy (load $LOAD): next tick"; exit 0
fi

# shellcheck disable=SC1091
source "$BASE/venv/bin/activate"
python -c "import pandas" 2>/dev/null || pip install -q -r vision/requirements.txt
export NBAVISION_STATE_DIR=$BASE/publish-state
mkdir -p "$NBAVISION_STATE_DIR"
TEXTS=$NBAVISION_STATE_DIR/texts.json
LEDGER=$NBAVISION_STATE_DIR/attempted_games.txt   # games whose Post button was clicked at least once
# Every 20 min: the tape summary (injuries, alert drafts, tipster picks) for the panel's Veille tab
TAPE_LAST=$NBAVISION_STATE_DIR/tape_export_last
if [ $(( $(date +%s) - $(cat "$TAPE_LAST" 2>/dev/null || echo 0) )) -ge 1200 ]; then
  date +%s > "$TAPE_LAST"
  if [ "$(NBA_TAPE_DIR=$BASE/tape python3 vision/tools/export_tape.py 2>/dev/null)" = "changed" ]; then
    cp docs/vision/tape.json /tmp/nbatape-export.json
    for i in 1 2 3; do
      git fetch -q origin main && git reset -q --hard origin/main
      cp /tmp/nbatape-export.json docs/vision/tape.json
      git add docs/vision/tape.json
      git diff --cached --quiet && break
      git -c user.name="nbavision-server" -c user.email="nbavision@users.noreply.github.com" commit -q -m "Veille: tape summary"
      git push -q origin HEAD:main && break
      sleep $((5 + RANDOM % 10))
    done
  fi
fi

REQ=$(python vision/tools/publish_queue.py next "$TEXTS") || { echo "queue read failed"; exit 1; }
[ "$REQ" = "none" ] && exit 0
read -r RID GAME DRY <<<"$REQ"
[ -n "${RID:-}" ] && [ -n "${GAME:-}" ] && [ -n "${DRY:-}" ] || { echo "malformed queue answer: $REQ"; exit 1; }
echo "Publishing $GAME (request $RID, dry run: $DRY)"

RESULT=/tmp/nbapublish-result.json
rm -f thread_result.json "$RESULT"
if [ "$DRY" != "true" ] && grep -qxF "$GAME" "$LEDGER" 2>/dev/null; then
  # A previous attempt clicked Post for this game (maybe live, maybe not pushed): never blindly again
  echo '{"posted": 0, "error": "already attempted on the server: check the account before posting again", "uncertain": true}' > "$RESULT"
else
  source vision/server/env.sh   # .env + secrets relayed from the control panel
  # Start from the reply bot's latest logged-in session
  [ -f "$BASE/state/state.json" ] && cp -u "$BASE/state/state.json" "$NBAVISION_STATE_DIR/state.json"
  export NBAVISION_BROWSER_CHANNEL=chromium PYTHONIOENCODING=utf-8 PUBLISH_BACKEND=browser BLOCK_MEDIA=0
  export TW_DRY_RUN=$DRY THREAD_RESULT_FILE=$REPO/thread_result.json
  THREAD_TEXTS_JSON=$(cat "$TEXTS" 2>/dev/null || true); export THREAD_TEXTS_JSON
  export CHROME_EXTRA_ARGS="--disable-gpu --renderer-process-limit=2 --disable-extensions --disable-background-networking"
  if [ "$DRY" = "true" ]; then
    # exercise the reply composer on one of our own past tweets (nothing is sent)
    PUBLISH_DRY_REPLY_TO=$(python -c "import json; print(json.load(open('data/tweet_history.json'))[-1]['tweet_id'])" 2>/dev/null || true)
    export PUBLISH_DRY_REPLY_TO
  fi
  mkdir -p vision/logs
  timeout 900 python scripts/publish_single_thread.py "$GAME" 2>&1 | tee "$BASE/logs/publish-$RID.log"
  if [ -f thread_result.json ]; then
    cp -f thread_result.json "$RESULT"
  else
    echo '{"posted": 0, "error": "publisher crashed before posting (see the server log)"}' > "$RESULT"
  fi
  # killed while a tweet was being sent: its outcome is unknown
  python - "$RESULT" <<'PY'
import json, sys
p = sys.argv[1]; r = json.load(open(p))
if r.get("sending") is not None and not r.get("dry_run"):
    r["uncertain"] = True
    r["error"] = r.get("error") or f"stopped while sending tweet {r['sending'] + 1}: check the account before posting again"
    json.dump(r, open(p, "w"))
PY
  if [ "$DRY" != "true" ] && python -c "import json,sys; r=json.load(open('$RESULT')); sys.exit(0 if r.get('posted') or r.get('uncertain') else 1)"; then
    echo "$GAME" >> "$LEDGER"
  fi
fi

# Mark published (also when uncertain: never re-post blindly) + log the outcome for the panel,
# re-applied on top of the latest remote state each try
for i in 1 2 3; do
  git fetch -q origin main && git reset -q --hard origin/main
  cp -f "$RESULT" thread_result.json
  if [ "$DRY" != "true" ] && python -c "import json,sys; r=json.load(open('thread_result.json')); sys.exit(0 if r.get('posted') or r.get('uncertain') else 1)"; then
    python scripts/mark_published.py "$GAME"
  fi
  python vision/tools/publish_queue.py log "$RID" "$GAME" thread_result.json
  git add docs/pending_games.json docs/vision/publish_log.json
  git diff --cached --quiet && break
  git -c user.name="nbavision-server" -c user.email="nbavision@users.noreply.github.com" \
      commit -q -m "Publish: $GAME ($([ "$DRY" = true ] && echo dry run || echo posted))"
  git push -q origin HEAD:main && break
  sleep $((5 + RANDOM % 10))
done
rm -f thread_result.json
ls -1t "$BASE/logs"/publish-*.log 2>/dev/null | tail -n +61 | xargs -r rm -f

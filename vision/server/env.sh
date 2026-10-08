# Environment shared by the server scripts (sourced from the repo root, BASE set by the caller):
# the server's .env (Windows line endings tolerated) + the secrets saved in the control panel,
# relayed sealed for this server by vision_secrets.yml.
set -a; . <(tr -d '\r' < "$BASE/.env"); set +a
if [ -f vision/server/secrets.enc ] && [ -f "$BASE/state/relay.key" ]; then
  RELAY_PASS=$(openssl pkeyutl -decrypt -inkey "$BASE/state/relay.key" -pkeyopt rsa_padding_mode:oaep \
               -pkeyopt rsa_oaep_md:sha256 -in vision/server/secrets.key.enc 2>/dev/null)
  if [ -n "$RELAY_PASS" ]; then
    set -a; . <(openssl enc -d -aes-256-cbc -pbkdf2 -iter 200000 -pass "pass:$RELAY_PASS" \
                -in vision/server/secrets.enc 2>/dev/null); set +a
  else
    echo "could not decrypt relayed secrets"
  fi
  unset RELAY_PASS
fi
# LLM provider: DeepSeek-flash when its key is set (tested 8 Oct on 255 tweets: ~3% factual errors
# vs ~10% for Groq's free model, no rate limit); the Groq key stays as the fallback.
if [ -n "${DEEPSEEK_API_KEY:-}" ]; then
  export GROQ_API_KEY="${LLM_API_KEY:-}"   # kept: takes over past the daily cap or if DeepSeek fails
  export LLM_API_KEY="$DEEPSEEK_API_KEY" LLM_BASE_URL=https://api.deepseek.com/v1 LLM_MODEL=deepseek-flash
  export LLM_TIMEOUT_SECONDS=60 LLM_DAILY_BUDGET_USD="${LLM_DAILY_BUDGET_USD:-0.60}"
  # A/B 8 Oct (80 tweets each): drafting without thinking + fact check with thinking = same quality,
  # 28 vs 33 publishable, -42% cost per posted reply (output tokens are most of the bill)
  export LLM_REASONING="${LLM_REASONING:-none}" VERIFY_REASONING="${VERIFY_REASONING:-high}"
fi

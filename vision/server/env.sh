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

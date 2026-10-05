#!/usr/bin/env bash
# Install / update the NBAVision reply bot on a Linux server (run as root, idempotent).
# Isolated from anything else on the box: own user, own directory, own venv.
#   /opt/nba-vision/{repo,venv,state,logs,.env}
# After the first run, add the printed deploy key to the GitHub repo (write access).
set -euo pipefail
BASE=/opt/nba-vision
REPO_SSH=git@github.com:AnthonyNadjari/NBAPredictLab.git

id nbavision >/dev/null 2>&1 || useradd --system --create-home --home-dir /home/nbavision --shell /usr/sbin/nologin nbavision
mkdir -p $BASE/{state,logs}
chown -R nbavision:nbavision $BASE

# deploy key (repo-scoped) for pushing the panel data
KEY=/home/nbavision/.ssh/nbavision_deploy
if [ ! -f $KEY ]; then
  sudo -u nbavision mkdir -p /home/nbavision/.ssh
  sudo -u nbavision ssh-keygen -q -t ed25519 -N "" -C "nbavision@$(hostname)" -f $KEY
fi
sudo -u nbavision bash -c "cat > /home/nbavision/.ssh/config <<EOF
Host github.com
  IdentityFile $KEY
  IdentitiesOnly yes
  StrictHostKeyChecking accept-new
EOF
chmod 600 /home/nbavision/.ssh/config"

if [ ! -d $BASE/repo/.git ]; then
  sudo -u nbavision git clone -q --depth 50 https://github.com/AnthonyNadjari/NBAPredictLab.git $BASE/repo
fi
sudo -u nbavision git -C $BASE/repo remote set-url origin $REPO_SSH

if [ ! -x $BASE/venv/bin/python ]; then
  python3 -m venv $BASE/venv
  chown -R nbavision:nbavision $BASE/venv
fi
sudo -u nbavision $BASE/venv/bin/pip install -q --upgrade pip
sudo -u nbavision $BASE/venv/bin/pip install -q -r $BASE/repo/vision/requirements.txt
$BASE/venv/bin/python -m playwright install-deps chromium >/dev/null
sudo -u nbavision PLAYWRIGHT_BROWSERS_PATH=$BASE/browsers $BASE/venv/bin/python -m playwright install chromium

[ -f $BASE/.env ] || { install -m 600 -o nbavision -g nbavision /dev/null $BASE/.env; }
grep -q PLAYWRIGHT_BROWSERS_PATH $BASE/.env || echo "PLAYWRIGHT_BROWSERS_PATH=$BASE/browsers" >> $BASE/.env

cat > /etc/systemd/system/nbavision.service <<EOF
[Unit]
Description=NBAVision reply bot tick
After=network-online.target

[Service]
Type=oneshot
User=nbavision
ExecStartPre=/usr/bin/git -C $BASE/repo fetch -q origin main
ExecStartPre=/usr/bin/git -C $BASE/repo reset -q --hard origin/main
ExecStart=/bin/bash -c 'cp $BASE/repo/vision/server/tick.sh /tmp/nbavision-tick.sh && exec bash /tmp/nbavision-tick.sh'
TimeoutStartSec=9600
# Shares the box with the TCG drop bot: low priority, capped resources
Nice=19
CPUWeight=10
CPUQuota=100%
IOSchedulingClass=idle
MemoryHigh=2048M
MemoryMax=2560M
EOF
cat > /etc/systemd/system/nbavision.timer <<EOF
[Unit]
Description=NBAVision reply bot every 5 minutes

[Timer]
OnBootSec=2min
OnUnitActiveSec=5min
Persistent=true

[Install]
WantedBy=timers.target
EOF
# Thread publisher: its own checkout, browser profile and service (see publish_tick.sh)
if [ ! -d $BASE/publish-repo/.git ]; then
  sudo -u nbavision git clone -q --depth 50 https://github.com/AnthonyNadjari/NBAPredictLab.git $BASE/publish-repo
fi
sudo -u nbavision git -C $BASE/publish-repo remote set-url origin $REPO_SSH
sudo -u nbavision mkdir -p $BASE/publish-state

cat > /etc/systemd/system/nbapublish.service <<EOF
[Unit]
Description=NBA Predict Lab thread publisher tick
After=network-online.target

[Service]
Type=oneshot
User=nbavision
ExecStartPre=/usr/bin/git -C $BASE/publish-repo fetch -q origin main
ExecStartPre=/usr/bin/git -C $BASE/publish-repo reset -q --hard origin/main
ExecStart=/bin/bash -c 'cp $BASE/publish-repo/vision/server/publish_tick.sh /tmp/nbapublish-tick.sh && exec bash /tmp/nbapublish-tick.sh'
TimeoutStartSec=1200
Nice=10
CPUWeight=20
CPUQuota=100%
IOSchedulingClass=best-effort
MemoryHigh=1024M
MemoryMax=1280M
EOF

cat > /etc/systemd/system/nbapublish.timer <<EOF
[Unit]
Description=NBA Predict Lab thread publisher every 2 minutes

[Timer]
OnBootSec=3min
OnUnitActiveSec=2min
Persistent=true

[Install]
WantedBy=timers.target
EOF

systemctl daemon-reload
systemctl enable --now nbavision.timer >/dev/null
systemctl enable --now nbapublish.timer >/dev/null
echo "== deploy key (add to GitHub with write access):"
cat $KEY.pub

/**
 * Vercel Serverless Function - control panel actions
 *
 * POST { password, action }
 *   action "verify" -> { success }
 *   action "status" -> recent runs of the daily and publish workflows + current cron
 *   action "run"    -> start the daily prediction workflow now (workflow_dispatch)
 *   action "x_check" -> run the read-only X connection check
 *   action "vision_run" { dry_run, max_replies } -> start a reply-bot session
 *   action "vision_schedule" { schedule } -> save docs/vision/schedule.json
 *   action "set_secret" { name, value } -> store TWITTER_COOKIES_JSON or LLM_API_KEY as a repo secret
 *
 * Environment (same as publish.js): PUBLISH_PASSWORD, GITHUB_TOKEN, GITHUB_REPO
 */
const crypto = require('crypto');
const sodium = require('libsodium-wrappers');

// Cookies pasted from the browser: a Cookie-Editor JSON export, or "auth_token=...; ct0=..."
function normalizeCookies(raw) {
  const text = String(raw || '').trim();
  let list;
  if (text.startsWith('[')) {
    list = JSON.parse(text).filter(c => c && c.name && /(^|\.)(x|twitter)\.com$/.test(String(c.domain || '.x.com')));
  } else {
    list = text.split(';').map(p => p.trim()).filter(Boolean).map(p => {
      const i = p.indexOf('=');
      return { name: p.slice(0, i).trim(), value: p.slice(i + 1).trim(), domain: '.x.com', path: '/',
        secure: true, httpOnly: p.startsWith('auth_token'), sameSite: 'None' };
    });
  }
  const names = new Set(list.map(c => c.name));
  if (!names.has('auth_token') || !names.has('ct0')) throw new Error('auth_token and ct0 cookies are required');
  return JSON.stringify(list);
}

const ALLOWED_ORIGINS = [
  'https://anthonynadjari.github.io',
  'http://localhost:8000',
  'http://127.0.0.1:8000',
];

function safeEqual(a, b) {
  const ha = crypto.createHash('sha256').update(a).digest();
  const hb = crypto.createHash('sha256').update(b).digest();
  return crypto.timingSafeEqual(ha, hb);
}

async function gh(path, token, init = {}) {
  const res = await fetch(`https://api.github.com${path}`, {
    ...init,
    headers: {
      Authorization: `Bearer ${token}`,
      Accept: 'application/vnd.github+json',
      'User-Agent': 'NBA-Predictor-Admin',
      ...(init.body ? { 'Content-Type': 'application/json' } : {}),
    },
  });
  return res;
}

function slimRun(r) {
  return {
    id: r.id,
    workflow: r.name,
    event: r.event,
    status: r.status,
    conclusion: r.conclusion,
    created_at: r.created_at,
    updated_at: r.updated_at,
    url: r.html_url,
  };
}

module.exports = async function handler(req, res) {
  const origin = req.headers.origin;
  if (ALLOWED_ORIGINS.includes(origin)) {
    res.setHeader('Access-Control-Allow-Origin', origin);
    res.setHeader('Vary', 'Origin');
  }
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');
  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') return res.status(405).json({ success: false, error: 'Use POST' });

  try {
    const { password, action } = req.body || {};
    const expected = process.env.PUBLISH_PASSWORD;
    if (!expected) return res.status(500).json({ success: false, error: 'Server config error' });
    if (typeof password !== 'string' || !safeEqual(password, expected)) {
      return res.status(401).json({ success: false, error: 'Invalid password' });
    }

    const token = process.env.GITHUB_TOKEN;
    const repo = process.env.GITHUB_REPO;
    if (!token || !repo) return res.status(500).json({ success: false, error: 'Server config error' });

    if (action === 'verify') return res.status(200).json({ success: true });

    if (action === 'status') {
      const files = {
        daily: 'daily_predictions.yml', publish: 'publish_thread.yml',
        vision: 'vision.yml', xcheck: 'x_check.yml',
      };
      const entries = await Promise.all(Object.entries(files).map(async ([key, file]) => {
        const r = await gh(`/repos/${repo}/actions/workflows/${file}/runs?per_page=${key === 'vision' ? 60 : 10}`, token);
        if (!r.ok) return [key, []];
        let runs = (await r.json()).workflow_runs.map(slimRun);
        if (key === 'vision') {
          // 15-min schedule ticks that found no session due finish in seconds: hide them
          runs = runs.filter(x => x.event !== 'schedule' || x.status !== 'completed' || x.conclusion !== 'success'
            || (new Date(x.updated_at) - new Date(x.created_at)) > 180000).slice(0, 12);
        }
        return [key, runs];
      }));
      const byKind = Object.fromEntries(entries);
      const runs = [...byKind.daily, ...byKind.publish, ...byKind.vision, ...byKind.xcheck]
        .sort((a, b) => b.created_at.localeCompare(a.created_at));
      const wfRes = await gh(`/repos/${repo}/contents/.github/workflows/daily_predictions.yml`, token);
      let cron = null;
      if (wfRes.ok) {
        const content = Buffer.from((await wfRes.json()).content, 'base64').toString('utf8');
        const m = content.match(/cron:\s*['"](.*?)['"]/);
        cron = m ? m[1] : null;
      }
      return res.status(200).json({ success: true, runs, cron });
    }

    if (action === 'run') {
      const r = await gh(`/repos/${repo}/actions/workflows/daily_predictions.yml/dispatches`, token, {
        method: 'POST',
        body: JSON.stringify({ ref: 'main' }),
      });
      if (r.status === 204) return res.status(200).json({ success: true });
      return res.status(502).json({ success: false, error: `GitHub API error (${r.status})` });
    }

    if (action === 'x_check') {
      const r = await gh(`/repos/${repo}/actions/workflows/x_check.yml/dispatches`, token, {
        method: 'POST', body: JSON.stringify({ ref: 'main' }),
      });
      if (r.status === 204) return res.status(200).json({ success: true });
      return res.status(502).json({ success: false, error: `GitHub API error (${r.status})` });
    }

    if (action === 'vision_run') {
      const dry = req.body.dry_run === true;
      const max = Number.isInteger(req.body.max_replies) && req.body.max_replies > 0 && req.body.max_replies <= 100
        ? String(req.body.max_replies) : '';
      const r = await gh(`/repos/${repo}/actions/workflows/vision.yml/dispatches`, token, {
        method: 'POST', body: JSON.stringify({ ref: 'main', inputs: { dry_run: dry, max_replies: max,
          login_check: req.body.login_check === true } }),
      });
      if (r.status === 204) return res.status(200).json({ success: true });
      return res.status(502).json({ success: false, error: `GitHub API error (${r.status})` });
    }

    if (action === 'vision_schedule') {
      // body.schedule = { enabled: bool, schedules: [{time:'HH:MM', enabled}], settings: {max_replies} }
      const sc = req.body.schedule || {};
      const okTime = t => typeof t === 'string' && /^([01]\d|2[0-3]):[0-5]\d$/.test(t);
      if (!Array.isArray(sc.schedules) || sc.schedules.length > 24 || !sc.schedules.every(x => okTime(x.time))) {
        return res.status(400).json({ success: false, error: 'Invalid schedule' });
      }
      const max = parseInt((sc.settings || {}).max_replies, 10);
      if (!(max >= 1 && max <= 100)) return res.status(400).json({ success: false, error: 'max_replies 1-100' });
      const clean = {
        enabled: sc.enabled !== false,
        timezone: 'Europe/Paris',
        schedules: sc.schedules.map(x => ({ time: x.time, enabled: x.enabled !== false }))
          .sort((a, b) => a.time.localeCompare(b.time)),
        settings: { max_replies: max },
      };
      const path = `/repos/${repo}/contents/docs/vision/schedule.json`;
      const cur = await gh(path, token);
      if (!cur.ok) return res.status(502).json({ success: false, error: `GitHub API error (${cur.status})` });
      const sha = (await cur.json()).sha;
      const put = await gh(path, token, {
        method: 'PUT',
        body: JSON.stringify({
          message: 'Update reply bot schedule from control panel',
          content: Buffer.from(JSON.stringify(clean, null, 2) + '\n').toString('base64'),
          sha,
        }),
      });
      if (!put.ok) return res.status(502).json({ success: false, error: `GitHub API error (${put.status})` });
      return res.status(200).json({ success: true, schedule: clean });
    }

    if (action === 'set_secret') {
      const name = req.body.name;
      let value = req.body.value;
      if (!['TWITTER_COOKIES_JSON', 'LLM_API_KEY'].includes(name) || typeof value !== 'string' || !value.trim()
          || value.length > 20000) {
        return res.status(400).json({ success: false, error: 'Invalid secret' });
      }
      try {
        value = name === 'TWITTER_COOKIES_JSON' ? normalizeCookies(value) : value.trim();
      } catch (e) {
        return res.status(400).json({ success: false, error: `Cookies invalides : ${e.message}` });
      }
      const keyRes = await gh(`/repos/${repo}/actions/secrets/public-key`, token);
      if (!keyRes.ok) return res.status(502).json({ success: false, error: `GitHub API error (${keyRes.status}): token needs Secrets write access` });
      const { key, key_id } = await keyRes.json();
      await sodium.ready;
      const sealed = sodium.crypto_box_seal(sodium.from_string(value), sodium.from_base64(key, sodium.base64_variants.ORIGINAL));
      const put = await gh(`/repos/${repo}/actions/secrets/${name}`, token, {
        method: 'PUT',
        body: JSON.stringify({ encrypted_value: sodium.to_base64(sealed, sodium.base64_variants.ORIGINAL), key_id }),
      });
      if (put.status === 201 || put.status === 204) return res.status(200).json({ success: true });
      return res.status(502).json({ success: false, error: `GitHub API error (${put.status}): token needs Secrets write access` });
    }

    return res.status(400).json({ success: false, error: 'Unknown action' });
  } catch (err) {
    console.error('admin error', err);
    return res.status(500).json({ success: false, error: 'Internal server error' });
  }
};

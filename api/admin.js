/**
 * Vercel Serverless Function - control panel actions
 *
 * POST { password, action }
 *   action "verify" -> { success }
 *   action "status" -> recent runs of the daily and publish workflows + current cron
 *   action "run"    -> start the daily prediction workflow now (workflow_dispatch)
 *
 * Environment (same as publish.js): PUBLISH_PASSWORD, GITHUB_TOKEN, GITHUB_REPO
 */
const crypto = require('crypto');

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
      const [runsRes, wfRes] = await Promise.all([
        gh(`/repos/${repo}/actions/runs?per_page=30`, token),
        gh(`/repos/${repo}/contents/.github/workflows/daily_predictions.yml`, token),
      ]);
      if (!runsRes.ok) return res.status(502).json({ success: false, error: `GitHub API error (${runsRes.status})` });
      const runs = (await runsRes.json()).workflow_runs
        .filter(r => ['Daily NBA Predictions', 'Publish Twitter Thread'].includes(r.name))
        .map(slimRun);
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

    return res.status(400).json({ success: false, error: 'Unknown action' });
  } catch (err) {
    console.error('admin error', err);
    return res.status(500).json({ success: false, error: 'Internal server error' });
  }
};

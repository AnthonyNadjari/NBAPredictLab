/**
 * Vercel Serverless Function - Update Daily Prediction Schedule
 *
 * Updates the cron schedule in .github/workflows/daily_predictions.yml
 * via the GitHub Contents API.
 *
 * Environment Variables Required (same as publish.js):
 * - PUBLISH_PASSWORD
 * - GITHUB_TOKEN (PAT with repo scope)
 * - GITHUB_REPO (e.g. "AnthonyNadjari/NBAPredictLab")
 */

module.exports = async function handler(req, res) {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  if (req.method === 'OPTIONS') return res.status(200).end();
  if (req.method !== 'POST') {
    return res.status(405).json({ success: false, error: 'Use POST' });
  }

  try {
    const { password, hour, minute } = req.body;

    if (!password || hour === undefined || minute === undefined) {
      return res.status(400).json({
        success: false,
        error: 'Missing fields: password, hour, minute'
      });
    }

    // Validate hour/minute
    const h = parseInt(hour);
    const m = parseInt(minute);
    if (isNaN(h) || h < 0 || h > 23 || isNaN(m) || m < 0 || m > 59) {
      return res.status(400).json({
        success: false,
        error: 'Invalid time. Hour 0-23, minute 0-59.'
      });
    }

    // Auth
    const correctPassword = process.env.PUBLISH_PASSWORD;
    if (!correctPassword || password !== correctPassword) {
      return res.status(401).json({ success: false, error: 'Invalid password' });
    }

    const githubToken = process.env.GITHUB_TOKEN;
    const githubRepo = process.env.GITHUB_REPO;
    if (!githubToken || !githubRepo) {
      return res.status(500).json({ success: false, error: 'Server config error' });
    }

    const filePath = '.github/workflows/daily_predictions.yml';
    const apiBase = `https://api.github.com/repos/${githubRepo}/contents/${filePath}`;
    const headers = {
      'Authorization': `Bearer ${githubToken}`,
      'Accept': 'application/vnd.github.v3+json',
      'Content-Type': 'application/json',
      'User-Agent': 'NBA-Predictor-Schedule'
    };

    // 1. Get current file content + sha
    const getResp = await fetch(apiBase, { headers });
    if (!getResp.ok) {
      const err = await getResp.text();
      return res.status(500).json({
        success: false,
        error: `Failed to read workflow: ${getResp.status}`,
        details: err
      });
    }

    const fileData = await getResp.json();
    const currentContent = Buffer.from(fileData.content, 'base64').toString('utf-8');
    const sha = fileData.sha;

    // 2. Replace cron line
    const newCron = `${m} ${h} * * *`;
    const updatedContent = currentContent.replace(
      /cron:\s*['"].*?['"]/,
      `cron: '${newCron}'`
    );

    if (updatedContent === currentContent) {
      return res.status(400).json({
        success: false,
        error: 'Could not find cron pattern in workflow file'
      });
    }

    // 3. Commit the update
    const putResp = await fetch(apiBase, {
      method: 'PUT',
      headers,
      body: JSON.stringify({
        message: `Update daily prediction schedule to ${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')} UTC`,
        content: Buffer.from(updatedContent).toString('base64'),
        sha: sha
      })
    });

    if (!putResp.ok) {
      const err = await putResp.text();
      return res.status(500).json({
        success: false,
        error: `Failed to update: ${putResp.status}`,
        details: err
      });
    }

    return res.status(200).json({
      success: true,
      message: `Schedule updated to ${String(h).padStart(2, '0')}:${String(m).padStart(2, '0')} UTC`,
      cron: newCron
    });

  } catch (error) {
    console.error('Error:', error);
    return res.status(500).json({
      success: false,
      error: 'Internal server error',
      message: error.message
    });
  }
}

/**
 * Vercel Serverless Function - Publish Twitter Thread
 *
 * This function acts as a secure proxy between the GitHub Pages frontend
 * and the GitHub API. It validates a simple password and triggers the
 * GitHub Actions workflow to publish a Twitter thread.
 *
 * Environment Variables Required (set in Vercel dashboard):
 * - PUBLISH_PASSWORD: Simple password for authentication
 * - GITHUB_TOKEN: Personal Access Token with 'repo' scope
 * - GITHUB_REPO: Repository in format "username/repo"
 */

const crypto = require('crypto');

function safeEqual(a, b) {
  const ha = crypto.createHash('sha256').update(a).digest();
  const hb = crypto.createHash('sha256').update(b).digest();
  return crypto.timingSafeEqual(ha, hb);
}

module.exports = async function handler(req, res) {
  // Enable CORS
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Content-Type');

  // Handle preflight request
  if (req.method === 'OPTIONS') {
    return res.status(200).end();
  }

  // Only allow POST
  if (req.method !== 'POST') {
    return res.status(405).json({
      success: false,
      error: 'Method not allowed. Use POST.'
    });
  }

  try {
    // Parse request body
    const { game_id, password } = req.body;

    // Validate required fields
    if (!game_id || !password) {
      return res.status(400).json({
        success: false,
        error: 'Missing required fields: game_id and password'
      });
    }

    // game_id ends up in a GitHub Actions run; only allow the exporter's format
    // (e.g. "Boston_Celtics_vs_Detroit_Pistons_2026-10-20").
    if (typeof game_id !== 'string' || game_id.length > 120 || !/^[A-Za-z0-9_.-]+$/.test(game_id)) {
      return res.status(400).json({
        success: false,
        error: 'Invalid game_id'
      });
    }

    // Verify password
    const correctPassword = process.env.PUBLISH_PASSWORD;
    if (!correctPassword) {
      console.error('PUBLISH_PASSWORD not configured in environment');
      return res.status(500).json({
        success: false,
        error: 'Server configuration error. Contact administrator.'
      });
    }

    if (!safeEqual(String(password), correctPassword)) {
      return res.status(401).json({
        success: false,
        error: 'Invalid password'
      });
    }

    // Get GitHub configuration
    const githubToken = process.env.GITHUB_TOKEN;
    const githubRepo = process.env.GITHUB_REPO; // Format: "username/repo"

    if (!githubToken || !githubRepo) {
      console.error('GitHub configuration missing');
      return res.status(500).json({
        success: false,
        error: 'Server configuration error. Contact administrator.'
      });
    }

    // Trigger GitHub Actions workflow via repository_dispatch
    const githubApiUrl = `https://api.github.com/repos/${githubRepo}/dispatches`;

    const response = await fetch(githubApiUrl, {
      method: 'POST',
      headers: {
        'Authorization': `Bearer ${githubToken}`,
        'Accept': 'application/vnd.github.v3+json',
        'Content-Type': 'application/json',
        'User-Agent': 'NBA-Predictor-Publisher'
      },
      body: JSON.stringify({
        event_type: 'publish_thread',
        client_payload: {
          game_id: game_id,
          triggered_by: 'web_interface',
          timestamp: new Date().toISOString()
        }
      })
    });

    // GitHub returns 204 No Content on success
    if (response.status === 204 || response.status === 200) {
      return res.status(200).json({
        success: true,
        message: 'Workflow triggered successfully',
        game_id: game_id
      });
    }

    // Handle GitHub API errors
    const errorText = await response.text();
    console.error('GitHub API error:', response.status, errorText);

    return res.status(502).json({
      success: false,
      error: `GitHub API error (${response.status})`
    });

  } catch (error) {
    console.error('Error in publish function:', error);
    return res.status(500).json({
      success: false,
      error: 'Internal server error'
    });
  }
}

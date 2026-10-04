#!/usr/bin/env python3
"""
Trigger the NBAVision Engine workflow via GitHub API (workflow_dispatch).
Requires a Personal Access Token with 'workflow' scope.

Set GITHUB_TOKEN or GH_TOKEN in the environment, then run:
  python scripts/trigger_workflow.py

Optional: GITHUB_REPO=owner/repo (default: AnthonyNadjari/NBAVisionEngine)
"""
import os
import sys
import urllib.request

REPO = os.environ.get("GITHUB_REPO", "AnthonyNadjari/NBAVisionEngine")
WORKFLOW_ID = "nbavision.yml"
URL = f"https://api.github.com/repos/{REPO}/actions/workflows/{WORKFLOW_ID}/dispatches"


def main() -> None:
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if not token:
        print("Set GITHUB_TOKEN or GH_TOKEN (PAT with 'workflow' scope).", file=sys.stderr)
        sys.exit(1)
    req = urllib.request.Request(
        URL,
        data=b'{"ref":"main"}',
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req) as r:
            if r.status in (200, 204):
                print("Workflow triggered.")
            else:
                print(f"Unexpected status: {r.status}", file=sys.stderr)
                sys.exit(1)
    except urllib.error.HTTPError as e:
        print(f"API error: {e.code} {e.reason}", file=sys.stderr)
        if e.fp:
            body = e.fp.read().decode("utf-8", errors="replace")
            print(body, file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

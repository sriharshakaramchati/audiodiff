#!/usr/bin/env bash
# usage: ./publish.sh <your-github-username>
# needs: gh cli, logged in (gh auth login)
set -e
USER=${1:?usage: ./publish.sh <github-username>}
sed -i.bak "s/__GH_USER__/$USER/" docs/index.html && rm docs/index.html.bak
git init -b main 2>/dev/null || true
git add .
git commit -m "audiodiff v0: perceptual diff for audio versions"
gh repo create audiodiff --public --source=. --push \
  --description "Perceptual diff for two versions of an audio file. CLI + browser."
gh api "repos/$USER/audiodiff/pages" -X POST \
  -f "source[branch]=main" -f "source[path]=/docs" >/dev/null
echo ""
echo "repo:      https://github.com/$USER/audiodiff"
echo "live tool: https://$USER.github.io/audiodiff/   (pages takes ~1 min to build)"

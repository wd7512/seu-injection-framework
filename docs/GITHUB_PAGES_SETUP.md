# GitHub Pages Setup Guide

This document explains how to configure GitHub Pages for the SEU Injection Framework documentation.

## Overview

The documentation is automatically built and deployed to GitHub Pages using GitHub Actions. The workflow is defined in `.github/workflows/docs.yml`.

## Initial Setup (One-Time Configuration)

After merging this PR, a repository administrator needs to configure GitHub Pages:

### Step 1: Enable GitHub Pages

1. Go to the repository on GitHub: https://github.com/wd7512/seu-injection-framework
2. Click **Settings** (repository settings, not account settings)
3. In the left sidebar, click **Pages**
4. Under **Source**, select:
   - **Source**: GitHub Actions
5. Click **Save**

That's it! The workflow will handle everything else.

### Step 2: Verify Deployment

After the first push to `main` that triggers the docs workflow:

1. Go to the **Actions** tab
2. Find the "Build and Deploy Sphinx Documentation" workflow
3. Wait for it to complete (should take 2-3 minutes)
4. Visit https://wd7512.github.io/seu-injection-framework/
5. You should see the documentation homepage

## Workflow Details

The documentation workflow (`.github/workflows/docs.yml`) does the following:

1. **Triggers**:
   - Push to `main` branch (for paths: `docs/**`, `src/seu_injection/**`, `.github/workflows/docs.yml`)
   - Pull requests to `main` (build only, no deploy)
   - Manual trigger via workflow dispatch

2. **Build Job**:
   - Sets up Python 3.11
   - Installs Sphinx and dependencies
   - Builds HTML documentation
   - Creates `.nojekyll` file (prevents Jekyll processing)
   - Uploads as Pages artifact

3. **Deploy Job** (only on `main` branch):
   - Deploys the artifact to GitHub Pages
   - Updates the live site at https://wd7512.github.io/seu-injection-framework/

## Troubleshooting

### Documentation not showing up

1. Check the Actions tab for workflow failures
2. Verify GitHub Pages is set to "GitHub Actions" source
3. Check that the workflow has permission to deploy (should be automatic with `pages: write` permission)

### Build failures

1. Review the workflow logs in the Actions tab
2. Common issues:
   - Sphinx warnings (usually non-fatal)
   - Missing dependencies (check if new imports were added)
   - Syntax errors in RST/Markdown files

### Local testing

To test the exact same build process locally:

```bash
# Install dependencies
pip install sphinx sphinx-rtd-theme sphinx-autodoc-typehints myst-parser
pip install torch numpy tqdm --index-url https://download.pytorch.org/whl/cpu

# Build docs
cd docs
make clean
make html

# View locally
python -m http.server 8000 --directory build/html
# Visit http://localhost:8000
```

## Updating Documentation

Documentation updates are automatically deployed when:

1. Changes are made to `docs/**` or `src/seu_injection/**`
2. The changes are merged to the `main` branch
3. The workflow completes successfully

No manual intervention is needed for updates.

## Custom Domain (Optional)

To use a custom domain (e.g., docs.seu-injection.org):

1. Add a `CNAME` file to `docs/source/_static/` with your domain
2. Update the Makefile to copy CNAME to build output
3. Configure DNS records to point to GitHub Pages
4. In repository Settings > Pages, set the custom domain

See [GitHub's custom domain docs](https://docs.github.com/en/pages/configuring-a-custom-domain-for-your-github-pages-site) for details.

## Links

- **Live Documentation**: https://wd7512.github.io/seu-injection-framework/
- **Workflow File**: `.github/workflows/docs.yml`
- **Sphinx Config**: `docs/source/conf.py`
- **GitHub Pages Settings**: https://github.com/wd7512/seu-injection-framework/settings/pages

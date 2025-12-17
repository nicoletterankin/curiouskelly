# Check Vercel GitHub Actions Secrets

## Required Secrets

The workflow `.github/workflows/deploy-vercel.yml` requires these secrets:

1. **VERCEL_TOKEN** - Vercel API token
2. **VERCEL_ORG_ID** - Vercel organization/team ID  
3. **VERCEL_PROJECT_ID** - Vercel project ID

## How to Check if Secrets are Set

### Method 1: GitHub Web Interface

1. Go to: **https://github.com/nicoletterankin/curiouskelly**
2. Click **Settings** tab
3. Click **Secrets and variables** → **Actions**
4. Look for these secrets:
   - `VERCEL_TOKEN`
   - `VERCEL_ORG_ID`
   - `VERCEL_PROJECT_ID`

If they exist, they're set. You won't see their values (they're encrypted).

### Method 2: Check Workflow Runs

1. Go to: **https://github.com/nicoletterankin/curiouskelly/actions**
2. Find the **"Deploy to Vercel"** workflow
3. Click on the latest run
4. Check the job logs:
   - If secrets are missing, you'll see: `Skipping job: conditional check 'secrets.VERCEL_TOKEN != '' && ...' evaluated to false`
   - If secrets are set but wrong, you'll see Vercel API errors

### Method 3: GitHub CLI (if installed)

```bash
gh secret list
```

This will show all repository secrets (but not their values).

## Workflow Behavior

The workflow has this condition:
```yaml
if: ${{ secrets.VERCEL_TOKEN != '' && secrets.VERCEL_ORG_ID != '' && secrets.VERCEL_PROJECT_ID != '' }}
```

**If secrets are NOT set:**
- The workflow will skip silently (no error, just won't run)
- You'll see "This check was skipped" in the workflow run

**If secrets ARE set:**
- The workflow will run
- It will attempt to deploy to Vercel

## Quick Check

Run this to see if the workflow ran after our last commit:

1. Go to: https://github.com/nicoletterankin/curiouskelly/actions
2. Look for workflow runs triggered by commit `9da66c7`
3. Check if "Deploy to Vercel" workflow ran or was skipped

## If Secrets Are Missing

Follow the guide in `docs/deployment/VERCEL_SETUP_GUIDE.md`:
- Step 3: Get Vercel API credentials
- Step 4: Add GitHub Secrets

## Note

Even if GitHub Actions secrets are NOT set, Vercel may still auto-deploy if:
- The Vercel project is connected to GitHub
- Vercel's native integration is enabled
- The project has auto-deploy enabled

Check Vercel dashboard → Project Settings → Git to see if auto-deploy is enabled.
















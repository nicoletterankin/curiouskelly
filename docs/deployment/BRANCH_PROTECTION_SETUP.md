# Branch Protection Setup - Step-by-Step Guide

**Repository:** `nicoletterankin/curiouskelly`  
**Target Branch:** `main`

---

## Quick Setup Instructions

### Step 1: Navigate to Branch Protection Settings

1. Go to: https://github.com/nicoletterankin/curiouskelly/settings/branches
2. You should see the message: "Classic branch protections have not been configured"
3. Click the button: **"Add classic branch protection rule"**

### Step 2: Configure Branch Protection Rule

1. **Branch name pattern:**
   - Enter: `main`
   - This will protect the main branch

2. **Protect matching branches** section - Enable the following:

   ✅ **Require a pull request before merging**
   - Check the box
   - **Required number of approvals before merging:** Set to `1`
   - ✅ **Require review from Code Owners** (if you have a CODEOWNERS file)
   - ✅ **Dismiss stale pull request approvals when new commits are pushed**
   - ✅ **Require approval of the most recent push**

   ✅ **Require status checks to pass before merging**
   - Check the box
   - **Require branches to be up to date before merging:** Check this box
   - **Status checks that are required:**
     - You can add specific checks later (e.g., `build`, `test`, `lint`)
     - For now, leave empty or add checks as you configure CI/CD

   ✅ **Require conversation resolution before merging**
   - Check the box
   - This ensures all comments are addressed

   ✅ **Require signed commits**
   - Optional but recommended for security
   - Check if you want to enforce GPG signing

   ✅ **Require linear history**
   - Optional but recommended
   - Prevents merge commits, requires rebase or squash

   ✅ **Include administrators**
   - **IMPORTANT:** Check this box
   - This ensures even admins must follow the rules

   ✅ **Do not allow bypassing the above settings**
   - Check this box for maximum security
   - Prevents anyone from bypassing protections

   ✅ **Restrict who can push to matching branches**
   - Optional: Only if you want to restrict pushes to specific users/teams
   - Leave unchecked for now

   ✅ **Allow force pushes**
   - **Leave UNCHECKED** (this is a security risk)

   ✅ **Allow deletions**
   - **Leave UNCHECKED** (prevents accidental branch deletion)

3. **Rules applied to everyone including administrators**
   - Make sure this is checked

4. Click **"Create"** button at the bottom

### Step 3: Verify Protection

After creating the rule, you should see:
- A rule listed for `main` branch
- All the protections you enabled displayed
- The ability to edit or delete the rule

---

## Recommended Configuration Summary

For `curiouskelly.com` production repository, use these settings:

| Setting | Value |
|---------|-------|
| Branch pattern | `main` |
| Require pull request | ✅ Yes (1 approval) |
| Require status checks | ✅ Yes (branches up to date) |
| Require conversation resolution | ✅ Yes |
| Require signed commits | ⚠️ Optional |
| Require linear history | ⚠️ Optional |
| Include administrators | ✅ Yes |
| Allow force pushes | ❌ No |
| Allow deletions | ❌ No |

---

## Alternative: Using Branch Rulesets (Newer Feature)

GitHub also offers "Branch rulesets" which provide more granular control:

1. Click **"Add branch ruleset"** instead
2. **Ruleset name:** `main-branch-protection`
3. **Target branches:** `main`
4. Configure similar protections as above
5. **Ruleset enforcement:** `Active`

---

## Post-Setup Verification

After configuring branch protection:

1. ✅ Try to push directly to `main` (should be blocked or require override)
2. ✅ Create a test branch and pull request
3. ✅ Verify that PR requires approval
4. ✅ Verify that status checks must pass

---

## Troubleshooting

### Issue: "Cannot push to protected branch"
**Solution:** This is expected! You must use pull requests to make changes to `main`.

### Issue: "Status checks not running"
**Solution:** 
- Make sure GitHub Actions workflows are configured
- Add the workflow names to the "Required status checks" list
- Wait for at least one successful workflow run

### Issue: "Need to bypass protection for emergency fix"
**Solution:**
- If you checked "Do not allow bypassing", you'll need to temporarily disable the rule
- Or create a separate `hotfix` branch that merges to `main` via PR

---

## Next Steps

After branch protection is configured:

1. ✅ Continue with Cloudflare Pages setup
2. ✅ Continue with Vercel setup
3. ✅ Add GitHub Secrets for CI/CD
4. ✅ Test deployment workflows

---

**Last Updated:** 2025-01-11  
**Repository:** https://github.com/nicoletterankin/curiouskelly/settings/branches


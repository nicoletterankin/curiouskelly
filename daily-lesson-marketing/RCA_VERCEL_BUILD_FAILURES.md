# Root Cause Analysis: 100% Vercel Build Failures

**Date**: November 14, 2025  
**Status**: CRITICAL - 100% build failure rate  
**Impact**: Complete deployment failure, no successful production builds

---

## Executive Summary

**Primary Root Cause**: Node version mismatch between local development (v24.11.0) and Vercel deployment (v18.20.0), combined with uncommitted `package-lock.json`, causing inconsistent dependency resolution and module path resolution failures.

**Build Success Rate**:
- Local (Windows, Node 24): ✅ 100% success
- Vercel (Linux, Node 18): ❌ 100% failure
- GitHub Actions (Linux, Node 18): Unknown (needs verification)

---

## Root Causes Identified

### 1. **Node Version Mismatch** ⚠️ CRITICAL

**Issue**:
- Local development: Node v24.11.0, npm v11.6.1
- Vercel/CI: Node v18.20.0 (from `.nvmrc`)
- Node 18 and Node 24 have different module resolution algorithms

**Impact**:
- Path alias resolution differs between environments
- Import statement handling differs (especially `.js` extensions for `.ts` files)
- Vite bundler behavior varies by Node version

**Evidence**:
```bash
# Local
$ node --version
v24.11.0

# Vercel (.nvmrc)
18.20.0
```

---

### 2. **Missing Package Lock File in Git** ⚠️ CRITICAL

**Issue**:
- `package-lock.json` exists locally but is **gitignored**
- Vercel regenerates lockfile on every build
- Non-deterministic dependency resolution

**Impact**:
- Different dependency versions between local and Vercel
- Transitive dependencies may resolve differently
- Build behavior is non-reproducible

**Evidence**:
```bash
$ git ls-files package-lock.json
# (empty output - file is gitignored)

$ cat .gitignore | grep package-lock
package-lock.json
```

**Current State**:
```json
// vercel.json
"installCommand": "npm install --legacy-peer-deps"
```

This installs fresh dependencies every time without a lockfile, leading to version drift.

---

### 3. **Path Alias Resolution Issues** ⚠️ HIGH

**Issue**:
- TypeScript path aliases (`@lib/*`, `@components/*`, etc.) configured in both `tsconfig.json` and `astro.config.mjs`
- Vite resolver in Astro may not respect these aliases consistently in Node 18
- API routes (`src/pages/api/lead.ts`) cannot resolve aliases in production build

**Impact**:
- `Cannot resolve "@lib/validation"` errors
- Forced to use relative paths (`../../lib/validation`) in API routes
- Inconsistent import patterns across codebase

**Evidence from Build Logs**:
```
[vite:load-fallback] Could not load /vercel/path0/daily-lesson-marketing/src/lib/validation (imported by src/pages/api/lead.ts)
```

**Current Workarounds Applied**:
- Line 2 of `src/pages/api/lead.ts` changed from `@lib/validation` to `../../lib/validation`
- Added explicit `extensions` array to Vite config
- Still failing on Vercel

---

### 4. **SCSS Import Resolution** ⚠️ MEDIUM

**Issue**:
- Bootstrap SCSS imports using deprecated `~` prefix
- Sass preprocessor configuration may differ between Node versions

**Impact**:
- Warning messages (not blocking locally, but may fail on Vercel)

**Evidence**:
```scss
// src/styles/main.scss
@import 'bootstrap/scss/functions';  // Works on Node 24, may fail on Node 18
```

**Required**:
```scss
@import '~bootstrap/scss/functions';  // Explicit tilde prefix
```

---

### 5. **TypeScript Configuration Drift** ⚠️ MEDIUM

**Issue**:
- `astro check` was removed from build script to bypass errors
- Type checking only happens in separate `build:check` script
- Production builds ship without type validation

**Impact**:
- Type errors hidden until runtime
- Path alias issues not caught pre-build
- Quality gate removed

**Evidence**:
```json
// package.json
"build": "astro build",  // No type check
"build:check": "astro check && astro build"  // Type check exists but not used
```

---

### 6. **Vercel Configuration Misalignment** ⚠️ LOW

**Issue**:
- `vercel.json` specifies `buildCommand: "npm run build"`
- No environment-specific build configuration
- Missing Node version override

**Current Config**:
```json
{
  "buildCommand": "npm run build",
  "installCommand": "npm install --legacy-peer-deps",
  "framework": "astro"
}
```

**Missing**:
- Explicit Node version specification
- Build environment variables
- Cache configuration

---

## Failure Pattern Analysis

### Timeline of Failed Deploys (Recent → Oldest)

1. **311cGAmDD** (4m ago): "resolve Vercel build failures - fix..."
   - Error: `Could not load ../lib/validation.js from src/pages/api/lead.ts`

2. **6ytaVPGrR** (6m ago): "use relative path with .js extension..."
   - Error: Same module resolution failure

3. **7kt3Vzcfp** (12m ago): "add missing validation functions..."
   - Error: Missing exports in validation.ts

4. **8yHL9mhKC** (16m ago): "replace relative imports with @li..."
   - Error: Path alias not resolving

5. **EKAGTBHs** (19m ago): "resolve path alias issues and cre..."
   - Error: Multiple TypeScript and path errors

**Pattern**: Every fix that works locally fails on Vercel due to environment differences.

---

## Why Local Builds Succeed

```bash
$ npm run build
> astro build

✓ Completed in 1.20s
[build] 26 page(s) built in 1.20s
[build] Complete!
```

**Success Factors**:
1. Node v24 has improved ESM module resolution
2. Local `package-lock.json` ensures consistent dependencies
3. Windows filesystem is case-insensitive (masks path issues)
4. Vite in Node 24 handles path aliases better

---

## Impact Assessment

### Business Impact
- **Deployment Velocity**: 0 successful deploys
- **Time to Production**: ∞ (blocked indefinitely)
- **Developer Confidence**: Low (every change expected to fail)

### Technical Debt
- 10+ failed deployment attempts
- Multiple workarounds in codebase
- Inconsistent import patterns
- Type checking bypassed

---

## Recommended Fixes (Priority Order)

### 🔴 P0: Critical Path to Green Build

#### 1. Commit package-lock.json
```bash
# Remove from .gitignore
sed -i '/package-lock.json/d' .gitignore

# Rebuild lockfile with Node 18
nvm use 18
rm package-lock.json
npm install --legacy-peer-deps

# Commit
git add package-lock.json .gitignore
git commit -m "fix: commit package-lock.json for reproducible builds"
```

#### 2. Standardize Node Version
```json
// package.json - add/verify
"engines": {
  "node": "18.20.0",
  "npm": ">=9.0.0"
}
```

Update local development:
```bash
nvm install 18.20.0
nvm use 18.20.0
```

#### 3. Fix Vercel Configuration
```json
// vercel.json
{
  "buildCommand": "npm run build",
  "outputDirectory": "dist",
  "installCommand": "npm ci",  // Changed from "npm install --legacy-peer-deps"
  "framework": "astro",
  "build": {
    "env": {
      "NODE_VERSION": "18.20.0"
    }
  }
}
```

#### 4. Restore Type Checking
```json
// package.json
"build": "astro check && astro build",  // Restore type check
```

Fix any type errors that emerge.

### 🟡 P1: Module Resolution Stability

#### 5. Consolidate Import Strategy
Either:
- **Option A**: Use path aliases everywhere (requires fixing Vite config for Node 18)
- **Option B**: Use relative imports everywhere (more reliable, less elegant)

**Recommendation**: Option B for immediate stability.

```bash
# Replace all @lib imports with relative paths in API routes
# src/pages/api/lead.ts
import { validateLeadForm, sanitizeFormData } from '../../lib/validation';

# src/pages/api/rum.ts  
import { validateRumPayload } from '../../lib/rum-validation';
```

#### 6. Fix SCSS Imports
```scss
// src/styles/main.scss
// Change all Bootstrap imports to use explicit paths
@import '../node_modules/bootstrap/scss/functions';
@import '../node_modules/bootstrap/scss/variables';
// ... etc
```

Or update `astro.config.mjs`:
```js
css: {
  preprocessorOptions: {
    scss: {
      includePaths: [path.resolve(__dirname, './node_modules')]
    }
  }
}
```

### 🟢 P2: Long-term Improvements

#### 7. Add Pre-deployment Validation
```yaml
# .github/workflows/deploy-vercel.yml
- name: Validate build on Node 18
  uses: actions/setup-node@v4
  with:
    node-version: '18.20.0'
    
- name: Test build
  run: |
    cd daily-lesson-marketing
    npm ci
    npm run build
```

#### 8. Add Build Monitoring
```json
// vercel.json
{
  "github": {
    "silent": false,
    "enabled": true
  }
}
```

#### 9. Document Environment Parity
Create `.tool-versions` or enforce via pre-commit hooks:
```bash
# .tool-versions
nodejs 18.20.0
```

---

## Testing Plan

### Phase 1: Local Validation (Node 18)
```bash
# Switch to Node 18
nvm use 18.20.0

# Clean install
rm -rf node_modules package-lock.json
npm install --legacy-peer-deps

# Test build
npm run build

# Expected: Should fail with same errors as Vercel
# Action: Fix errors until build succeeds
```

### Phase 2: Commit Fixes
```bash
git add package-lock.json .gitignore astro.config.mjs vercel.json
git commit -m "fix: align local and Vercel build environments"
git push origin main
```

### Phase 3: Monitor Vercel
- Watch deployment logs
- Verify build succeeds
- Test deployed site

### Phase 4: Validate Production
- Check all routes
- Verify API endpoints
- Test form submissions
- Validate analytics tracking

---

## Prevention Measures

1. **Enforce Node Version**: Add `engines.strict=true` to `.npmrc`
2. **Pre-commit Hooks**: Validate Node version matches .nvmrc
3. **CI/CD Parity**: Ensure GitHub Actions uses identical environment as Vercel
4. **Documentation**: Update README with required Node version
5. **Monitoring**: Add deployment success rate tracking

---

## Success Criteria

- [ ] Local build succeeds on Node 18.20.0
- [ ] Vercel build succeeds
- [ ] All 26 pages deploy successfully
- [ ] API routes functional (`/api/lead`, `/api/rum`)
- [ ] No TypeScript errors
- [ ] No module resolution errors
- [ ] Sitemap generates correctly
- [ ] All assets load properly

---

## Notes

**Why --legacy-peer-deps?**
- Required for Bootstrap 4.x with modern npm versions
- Should continue using, but with lockfile committed

**Why Not Upgrade Bootstrap?**
- Out of scope for this fix
- Would require extensive CSS/component changes
- Risk of breaking changes

**Alternative: Use Vite 6 + Rollup 5**
- May resolve module issues
- Requires Astro upgrade
- Higher risk, should be separate effort

---

## References

- Astro Documentation: https://docs.astro.build/en/guides/troubleshooting/
- Vercel Build Configuration: https://vercel.com/docs/projects/project-configuration
- Node.js Releases: https://nodejs.org/en/about/previous-releases
- npm lockfile spec: https://docs.npmjs.com/cli/v10/configuring-npm/package-lock-json


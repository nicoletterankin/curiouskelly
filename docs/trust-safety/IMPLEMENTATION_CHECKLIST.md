# TRUST & SAFETY IMPLEMENTATION CHECKLIST

## For Deployment to curiouskelly.com

This checklist ensures all Trust & Safety changes are properly deployed and verified.

---

## Pre-Deployment Checklist

### 1. File Changes (All completed in this session)

#### Chat Overlay System (`public/js/chat-overlay.js`)
- [x] Updated file header with Trust & Safety documentation
- [x] Added `getSimulatedContentPref()` and `setSimulatedContentPref()` methods
- [x] Added `toggleSimulated()` method for user control
- [x] Added `showTooltip()` and `hideTooltip()` methods
- [x] Updated `addComment()` to include ✨ indicator
- [x] Updated `addSpecificComment()` to include ✨ indicator
- [x] Updated `updateLiveBadge()` to show "✨ Social" instead of fake "LIVE" metrics
- [x] Added disclosure tooltip HTML/CSS
- [x] Added CSS for `.simulated-indicator`, `.simulated-hidden`, `#simulated-tooltip`
- [x] Added localStorage preference persistence
- [x] Comments now respect `simulatedEnabled` preference

#### learn.html
- [x] Changed "LIVE" badge to "✨ Social" badge
- [x] Added onclick handler to show tooltip
- [x] Updated badge dot color to amber (#f59e0b)

#### Homepage Updates
- [x] `index.html` - Removed "millions of learners" claim
- [x] `index-final.html` - Removed "millions of learners" claim
- [x] `index-production.html` - Removed "millions of learners" claim
- [x] `kelly.html` - Updated meta description

#### Company Name Updates (PBC)
- [x] `app.html` - Changed to "Lesson of the Day PBC"
- [x] `about.html` - Changed to "Lesson of the Day PBC"
- [x] `careers.html` - Changed to "Lesson of the Day PBC"
- [x] `diversity.html` - Changed to "Lesson of the Day PBC"
- [x] `enterprise.html` - Changed to "Lesson of the Day PBC"
- [x] `gifts.html` - Changed to "Lesson of the Day PBC"
- [x] `missions.html` - Changed to "Lesson of the Day PBC"
- [x] `terms.html` - Changed to "Lesson of the Day PBC"
- [x] `social.html` - Changed to "Lesson of the Day PBC"
- [x] `privacy.html` - Changed to "Lesson of the Day PBC"

#### New Pages
- [x] `public/trust.html` - Trust & Safety landing page

### 2. Documentation Created

- [x] `docs/trust-safety/TRUST_AND_SAFETY_INDEX.md`
- [x] `docs/trust-safety/SIMULATED_SOCIAL_CONTENT.md`
- [x] `docs/trust-safety/USER_CONTROLS.md`
- [x] `docs/trust-safety/TRUST_SAFETY_PRINCIPLES.md`
- [x] `docs/trust-safety/DISCLOSURE_STANDARDS.md`
- [x] `docs/trust-safety/SAFETY_TEAM_CHARTER.md`
- [x] `docs/trust-safety/IMPLEMENTATION_CHECKLIST.md` (this file)

### 3. Configuration Updates

- [x] `CLAUDE.md` - Added Trust & Safety section

---

## Deployment Steps

### Step 1: Verify Local Changes
```bash
# In project root
git status
git diff public/js/chat-overlay.js
git diff public/learn.html
git diff public/trust.html
```

### Step 2: Test Locally
1. Start local dev server
2. Navigate to `/learn.html`
3. Verify:
   - ✨ Social badge appears (not "LIVE")
   - Comments show ✨ indicator after username
   - Clicking badge shows disclosure tooltip
   - "Turn off" button in tooltip works
   - Settings persist after page refresh

### Step 3: Test Trust Page
1. Navigate to `/trust`
2. Verify all sections render correctly
3. Verify navigation links work
4. Verify footer shows "Lesson of the Day PBC"

### Step 4: Deploy
```bash
# Commit changes
git add -A
git commit -m "feat: Trust & Safety disclosure for simulated social content

- Add ✨ indicator to all simulated comments
- Replace fake LIVE badge with honest ✨ Social badge
- Add disclosure tooltip with turn-off option
- Create /trust page explaining our approach
- Update company name to Lesson of the Day PBC across all pages
- Remove misleading 'millions of learners' claims
- Add comprehensive T&S documentation

Closes Trust & Safety initiative"

# Push to deploy branch
git push origin main
```

### Step 5: Post-Deploy Verification
1. Visit https://curiouskelly.com/learn
2. Verify ✨ Social badge appears
3. Verify comments show ✨ indicator
4. Verify tooltip appears on click
5. Verify preference persists

6. Visit https://curiouskelly.com/trust
7. Verify page loads correctly

8. Visit https://curiouskelly.com/
9. Verify no fake metrics visible

---

## User Acceptance Criteria

### Must Pass
- [ ] ✨ appears on every simulated comment
- [ ] Badge says "✨ Social" not "LIVE"
- [ ] Clicking badge shows explanation
- [ ] User can turn off simulated content
- [ ] Preference persists across sessions
- [ ] /trust page is accessible
- [ ] No fake viewer counts shown
- [ ] No "millions of learners" claims (unless true)
- [ ] Footer says "Lesson of the Day PBC"

### Nice to Have
- [ ] Tooltip is accessible via keyboard
- [ ] Screen reader announces simulated content
- [ ] Works on mobile
- [ ] Works offline (PWA)

---

## Rollback Plan

If issues arise:

```bash
# Revert to previous version
git revert HEAD
git push origin main
```

Or selectively revert chat overlay:
```bash
git checkout HEAD~1 -- public/js/chat-overlay.js
git commit -m "revert: chat overlay changes"
git push origin main
```

---

## Monitoring After Launch

### Week 1
- Monitor for user confusion reports
- Check analytics for /trust page visits
- Review support tickets about simulated content

### Week 2+
- Survey users about understanding of simulation
- Measure % who customize settings
- Track any psychological harm reports

### Red Flags (Immediate Action)
- Users reporting they didn't know content was simulated
- Users describing simulated users as "friends"
- Psychological distress reports
- Media coverage questioning transparency

---

## Future Work

### Phase 2: Enhanced Controls
- [ ] Granular type controls (peer comments, age responses, etc.)
- [ ] Disclosure mode options (standard/enhanced/maximum)
- [ ] Parent/guardian controls for child accounts

### Phase 3: Real Community
- [ ] Real user comments system
- [ ] Clear marking: real (✓) vs simulated (✨)
- [ ] Option to see only real community

### Phase 4: External Audit
- [ ] Third-party ethics audit
- [ ] Academic research partnership
- [ ] Public transparency report

---

*Last updated: December 2025*
*Owner: Trust & Safety*


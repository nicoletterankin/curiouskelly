# Kelly OS Migration Plan - Vercel-Only Architecture

**Date:** December 2025  
**Purpose:** Step-by-step migration from hybrid (Vercel + Cloudflare) to Vercel-only  
**Status:** Ready for implementation

---

## Executive Summary

**Current State:** Hybrid (Vercel + Cloudflare Workers + Supabase)  
**Target State:** Vercel-only (Vercel + Supabase)  
**Timeline:** 2-4 weeks  
**Risk Level:** Low (phased migration, can rollback)

---

## Phase 1: Assessment & Preparation (Week 1)

### 1.1 Current State Audit

**Tasks:**
- [ ] Document all Cloudflare Workers and their functions
- [ ] List all Cloudflare R2 buckets and contents
- [ ] Identify dependencies on Cloudflare services
- [ ] Map Cloudflare routes to Vercel equivalents

**Deliverables:**
- Cloudflare services inventory
- Migration priority list
- Risk assessment

### 1.2 Vercel Setup

**Tasks:**
- [ ] Verify Vercel project configuration
- [ ] Set up Vercel Blob Storage (if needed)
- [ ] Configure Edge Config (for dynamic config)
- [ ] Test Vercel Edge Functions

**Deliverables:**
- Vercel project ready
- Blob storage configured
- Edge Config set up

---

## Phase 2: Static Assets Migration (Week 1-2)

### 2.1 Unity Assets (Legacy)

**Current:** Cloudflare R2 (`curious-kelly-media` bucket)  
**Target:** Vercel Blob Storage or Supabase Storage

**Steps:**
1. [ ] List all Unity assets in R2
2. [ ] Download assets locally
3. [ ] Upload to Vercel Blob Storage (or Supabase Storage)
4. [ ] Update asset URLs in code
5. [ ] Test asset loading
6. [ ] Decommission R2 bucket (after verification)

**Files to Update:**
- `public/unity/*` (if still referenced)
- Any Unity WebGL integration code

**Risk:** Low (Unity not in production)

### 2.2 Video Assets

**Current:** Supabase Storage (`kelly-videos` bucket)  
**Target:** Keep on Supabase Storage (no change needed)

**Steps:**
- [ ] Verify Supabase Storage bucket exists
- [ ] Test video URLs
- [ ] No migration needed (already optimal)

**Risk:** None (already correct)

---

## Phase 3: API Migration (Week 2)

### 3.1 Lessons API Worker

**Current:** Cloudflare Worker (`infrastructure/cloudflare/lessons-api-worker/`)  
**Target:** Vercel Edge Function (`api/lessons/[dayNumber]-edge.ts`)

**Status:** ✅ **Already exists!**

**Steps:**
- [ ] Verify Edge Function works correctly
- [ ] Update DNS to point to Vercel (if using custom domain)
- [ ] Decommission Cloudflare Worker
- [ ] Remove `wrangler.toml` config

**Risk:** Low (Edge Function already implemented)

### 3.2 Unity CDN Worker

**Current:** Cloudflare Worker (`infrastructure/cloudflare/unity-cdn-worker/`)  
**Target:** Vercel Edge Function + Blob Storage

**Steps:**
1. [ ] Create Vercel Edge Function: `api/unity-assets/[...path].ts`
2. [ ] Migrate assets to Vercel Blob Storage
3. [ ] Implement CDN logic in Edge Function
4. [ ] Update Unity WebGL to use new endpoint
5. [ ] Test asset loading
6. [ ] Decommission Cloudflare Worker

**Risk:** Low (Unity not in production, can be deprecated)

---

## Phase 4: Database Migration (Week 2-3)

### 4.1 Cloudflare D1 Mirror

**Current:** Cloudflare D1 (`lessons-db`) mirroring Supabase  
**Target:** Remove D1, use Supabase directly with Edge caching

**Steps:**
1. [ ] Verify Supabase performance (should be sufficient)
2. [ ] Implement Edge caching in Vercel Edge Functions
3. [ ] Update lesson endpoints to use Supabase + cache
4. [ ] Test performance
5. [ ] Decommission D1 database

**Risk:** Low (D1 was redundant, Supabase is primary)

### 4.2 Edge Caching Strategy

**Implementation:**
- Use Vercel Edge Config for static config
- Use Vercel KV for dynamic caching (if needed)
- Cache lesson data at edge (TTL: 1 hour)
- Invalidate on lesson update

**Files:**
- `api/lessons/[dayNumber]-edge.ts` (already implements caching)

---

## Phase 5: DNS & Routing (Week 3)

### 5.1 DNS Configuration

**Current:** Domain may point to Cloudflare  
**Target:** Point to Vercel

**Steps:**
1. [ ] Verify current DNS setup
2. [ ] Update DNS records to point to Vercel
3. [ ] Configure SSL in Vercel (automatic)
4. [ ] Test domain resolution
5. [ ] Update Cloudflare DNS (if using Cloudflare DNS)

**Risk:** Medium (DNS changes require propagation time)

### 5.2 Route Updates

**Current Routes:**
- `api.curiouskelly.com/lessons/*` → Cloudflare Worker
- `unity.curiouskelly.com/*` → Cloudflare Worker

**Target Routes:**
- `curiouskelly.com/api/lessons/*` → Vercel Edge Function
- `curiouskelly.com/api/unity-assets/*` → Vercel Edge Function

**Steps:**
- [ ] Update API client URLs
- [ ] Update Unity WebGL asset URLs
- [ ] Test all routes
- [ ] Remove Cloudflare subdomains

---

## Phase 6: Testing & Validation (Week 3-4)

### 6.1 Functional Testing

**Test Cases:**
- [ ] Lesson loading works
- [ ] Video playback works
- [ ] Asset loading works
- [ ] API endpoints respond correctly
- [ ] Edge caching works
- [ ] Performance is acceptable

### 6.2 Performance Testing

**Metrics:**
- [ ] TTFB < 200ms (edge)
- [ ] API response time < 500ms
- [ ] Asset loading < 2s
- [ ] No regressions from current state

### 6.3 Rollback Plan

**If Issues:**
1. Revert DNS to Cloudflare
2. Re-enable Cloudflare Workers
3. Investigate issues
4. Fix and retry migration

---

## Phase 7: Cleanup (Week 4)

### 7.1 Decommission Cloudflare Services

**Steps:**
- [ ] Remove Cloudflare Workers
- [ ] Delete Cloudflare R2 buckets (after verification)
- [ ] Delete Cloudflare D1 database
- [ ] Remove `wrangler.toml` config
- [ ] Archive Cloudflare infrastructure code

### 7.2 Code Cleanup

**Files to Remove:**
- `infrastructure/cloudflare/` (archive, don't delete)
- `wrangler.toml` (or mark as deprecated)
- Cloudflare-specific code

**Files to Update:**
- `README.md` (update deployment instructions)
- `docs/deployment/` (update guides)

---

## Migration Checklist

### Pre-Migration

- [ ] Complete current state audit
- [ ] Set up Vercel Blob Storage
- [ ] Test Vercel Edge Functions
- [ ] Create rollback plan
- [ ] Notify team of migration

### During Migration

- [ ] Migrate static assets
- [ ] Migrate API endpoints
- [ ] Update DNS
- [ ] Test functionality
- [ ] Monitor performance

### Post-Migration

- [ ] Verify all functionality works
- [ ] Decommission Cloudflare services
- [ ] Update documentation
- [ ] Celebrate success! 🎉

---

## Risk Mitigation

### High-Risk Items

**DNS Changes:**
- Risk: Downtime during propagation
- Mitigation: Use low TTL, change during low-traffic period

**API Migration:**
- Risk: Breaking changes
- Mitigation: Test thoroughly, keep Cloudflare as backup initially

### Low-Risk Items

**Unity Assets:**
- Risk: Low (not in production)
- Mitigation: Can be deprecated if needed

**D1 Database:**
- Risk: Low (redundant)
- Mitigation: Supabase is primary, D1 was mirror

---

## Success Criteria

**Technical:**
- ✅ All APIs migrated to Vercel
- ✅ All assets migrated to Vercel/Supabase
- ✅ Performance maintained or improved
- ✅ Zero downtime migration

**Business:**
- ✅ No user-facing issues
- ✅ Cost reduction (no Cloudflare Workers)
- ✅ Simplified architecture
- ✅ Faster deployments

---

## Timeline Summary

| Phase | Duration | Key Deliverables |
|-------|----------|------------------|
| **Phase 1: Assessment** | Week 1 | Audit, Vercel setup |
| **Phase 2: Assets** | Week 1-2 | Assets migrated |
| **Phase 3: APIs** | Week 2 | APIs migrated |
| **Phase 4: Database** | Week 2-3 | D1 removed |
| **Phase 5: DNS** | Week 3 | DNS updated |
| **Phase 6: Testing** | Week 3-4 | Validation complete |
| **Phase 7: Cleanup** | Week 4 | Cloudflare decommissioned |

**Total: 4 weeks**

---

## Post-Migration Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    TARGET STATE                         │
├─────────────────────────────────────────────────────────┤
│ Frontend:     Vercel (Static HTML/JS from public/)     │
│ API:          Vercel Edge Functions + Serverless        │
│ Database:     Supabase (PostgreSQL)                    │
│ Storage:      Supabase Storage + Vercel Blob (optional)│
│ CDN:          Vercel Edge Network                      │
│ Caching:      Vercel Edge Config + KV                  │
└─────────────────────────────────────────────────────────┘
```

**Benefits:**
- ✅ Unified platform (Vercel)
- ✅ Simpler architecture
- ✅ Better developer experience
- ✅ Lower costs
- ✅ Faster deployments

---

**Status:** ✅ Ready for implementation  
**Next Step:** Begin Phase 1 - Assessment & Preparation  
**Owner:** Engineering Team  
**Review:** Weekly during migration



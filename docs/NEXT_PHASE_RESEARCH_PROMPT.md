# Next Phase Research Prompt

## Context for Next Session

You are implementing the Learner Lifecycle system for Curious Kelly. This is a **zero-trust audit trail** that tracks every interaction between learners and Kelly, in both directions.

### Documents to Read First
1. `docs/LEARNER_LIFECYCLE_ARCHITECTURE.md` - Complete system design
2. `docs/BUSINESS_MODEL_ARCHITECTURE.md` - Access tiers and pricing
3. `CLAUDE.md` - Operating rules

---

## Research Tasks

### 1. Supabase Schema Investigation
**Question**: What tables exist in the current Supabase schema?

**Actions**:
- Query Supabase to list all tables
- Check the `users` table structure
- Check existing indexes
- Identify any existing event/tracking tables

**Why**: We need to extend users and add new tables without breaking existing functionality.

---

### 2. Stripe Configuration Audit
**Question**: What Stripe products/prices exist? Do we have single-lesson purchase capability?

**Actions**:
- List all Stripe products in the account
- Check existing Price IDs (monthly, annual, lifetime, family, gifts)
- Determine if we need to create a new "single lesson" product
- Check if Stripe supports dynamic pricing (pay what you want, regional)

**Why**: Pay-per-lesson requires a Stripe product. Regional pricing requires multiple Price IDs.

---

### 3. Live Class Infrastructure
**Question**: What technology should power hourly live classes?

**Research**:
- Can we use YouTube Live with scheduled premieres?
- Daily.co / Whereby / Zoom SDK options?
- How do other ed-tech platforms do live classes at scale?
- What's the minimum viable live class experience?

**Constraints**:
- Must work on web, mobile, Roku
- Must handle 1000+ concurrent viewers per session
- Must be affordable at scale
- Recording for replay is required

---

### 4. Offline Download System
**Question**: How should lesson downloads work?

**Research**:
- What format for offline lessons? (Video + JSON manifest?)
- How to handle DRM/access verification?
- Progressive Web App vs native app for offline?
- How do other apps (Duolingo, Netflix) handle downloads?

**Constraints**:
- Must work on iOS/Android
- Must verify subscription status periodically
- Must not allow piracy/sharing of downloaded content

---

### 5. Comment System Design
**Question**: What's the best architecture for lesson comments?

**Research**:
- Real-time comments (Supabase Realtime, Pusher, etc.)
- Moderation workflow (pre-moderation vs post-moderation)
- Threading depth (flat, 1-level, full tree)
- AI moderation for spam/abuse

**Constraints**:
- Must be safe for kids (Curious Kelly serves all ages)
- Must not require manual moderation for every comment (doesn't scale)
- Must allow Kelly to respond to comments (AI-generated?)

---

### 6. Artwork Submission Flow
**Question**: How should user-submitted artwork work?

**Research**:
- File upload to Supabase Storage
- Image size/format requirements
- Moderation workflow
- How to display user art in lessons
- Attribution and licensing

**Constraints**:
- Must have clear rights assignment (user grants license)
- Must be safe (no inappropriate content)
- Must credit the artist
- Must be optional (not required to participate)

---

## Implementation Priority

Based on research, recommend which to build first:

| Feature | Complexity | User Value | Revenue Impact |
|---------|------------|------------|----------------|
| Event logging | Medium | Low (invisible) | High (enables everything) |
| Comments | Medium | High | Medium (engagement) |
| Pay-per-lesson | Low | Medium | High (new revenue) |
| Live classes | High | Very High | Medium |
| Downloads | High | High | Medium (retention) |
| Artwork | Medium | Medium | Low |

**Recommended Order**:
1. Event logging (foundation)
2. Pay-per-lesson (revenue)
3. Comments (engagement)
4. Artwork (community)
5. Downloads (retention)
6. Live classes (differentiation)

---

## Deliverables Expected

After research, produce:

1. **SQL migration file** - Create new tables in Supabase
2. **API endpoint stubs** - `/api/events`, `/api/comments`, etc.
3. **Frontend integration plan** - Where to add event tracking calls
4. **Stripe setup instructions** - New products needed
5. **Live class technology recommendation** - With cost estimate

---

## Questions for User

Before implementing, confirm:

1. **Live class frequency**: Every hour on the hour, 24/7? Or just certain hours?
2. **Comment moderation**: Pre-moderate all, or post-moderate with AI filter?
3. **Artwork licensing**: Does Kelly own submitted art, or just license it?
4. **Download limits**: How many lessons can be downloaded at once?
5. **Event retention**: How long to keep detailed event logs? (7 years?)

---

## Session Start Checklist

When starting next session:
- [ ] Read the three documents above
- [ ] Query Supabase schema
- [ ] Query Stripe products
- [ ] Confirm priorities with user
- [ ] Begin with user_events table creation

---

*This prompt saved: December 15, 2025*

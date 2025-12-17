# HeyGen Broken Avatar IDs - NEEDS FIX

**Date:** 2025-12-17

## Summary

6 out of 36 avatar IDs are returning 500 errors from HeyGen. These need to be re-uploaded or the IDs need to be updated once processing completes.

## Broken IDs (Check in HeyGen Dashboard)

| Archetype | Motion | Broken ID | Status |
|-----------|--------|-----------|--------|
| **architect** | C (Filler) | `4fdb0efd1d8f4cada563f3d0b1e6d7ce` | ❌ |
| **diplomat** | A (Warm Welcoming) | `0c110d20a92d47e68340baef8b2816b3` | ❌ |
| **macgyver** | A (Warm Welcoming) | `c5aab6ab13d940f8ae4700d546bd6b6b` | ❌ |
| **strategist** | A (Warm Welcoming) | `08d53d1b065041bda2e5b6bc32962a8a` | ❌ |
| **survivor** | B (Talk Talk Talk) | `3788e55c134e411586cc3e0b5a5786b0` | ❌ |
| **survivor** | C (Filler) | `5c2f5a75a0314a33bec25521d05d85b3` | ❌ |

## Working Archetypes (Full videos can be generated)

✅ scientist - all 3 motions work  
✅ explorer - all 3 motions work  
✅ rebel - all 3 motions work  
✅ empath - all 3 motions work  
✅ mystic - all 3 motions work  
✅ provider - all 3 motions work  
✅ storyteller - all 3 motions work  

## How to Fix

1. Go to HeyGen Dashboard → Talking Photos
2. Find each broken avatar by ID
3. Check if it's still processing:
   - **If processing:** Wait and try again later
   - **If failed:** Re-upload the source image with the motion prompt
4. Get the new avatar ID once complete
5. Update `generated-images/kelly-motion-library.json` with new IDs
6. Use the HTML tool: `public/admin/motion-library.html`

## Once Fixed

Run the remaining archetypes:

```bash
# After fixing architect C:
npx tsx scripts/heygen-video-generator.ts --day 351 --archetype architect

# After fixing diplomat A:
npx tsx scripts/heygen-video-generator.ts --day 351 --archetype diplomat

# After fixing macgyver A:
npx tsx scripts/heygen-video-generator.ts --day 351 --archetype macgyver

# After fixing strategist A:
npx tsx scripts/heygen-video-generator.ts --day 351 --archetype strategist

# After fixing survivor B and C:
npx tsx scripts/heygen-video-generator.ts --day 351 --archetype survivor
```

## Current Queue Status

Check with: `npx tsx scripts/heygen-check-status.ts --day 351`

Or poll until complete: `npx tsx scripts/heygen-check-status.ts --day 351 --poll`

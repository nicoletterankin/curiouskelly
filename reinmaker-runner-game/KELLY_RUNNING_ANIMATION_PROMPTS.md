# Kelly's Running Animation - Reference-Based Prompts

**Purpose:** Create 3-frame running animation cycle based on real reference photos
**Reference:** Real person in casual clothing (light blue sweatshirt, jeans, white sneakers)
**Game Style:** Stylized fantasy game sprite (light linen tunic with brass/circuit accents)

---

## Animation Cycle Overview

**Frame 0 (Contact):** Left foot down, right leg back, weight on left foot
**Frame 1 (Passing):** Right leg down, left leg crossing mid-stride  
**Frame 2 (Air):** Left leg forward extended, right leg back extended, both feet off ground

---

## Frame 0: Contact Frame
**File:** `player_run_0.png` (rename from `player.png` after transparency fix)
**Reference Pose:** Left foot forward flat on ground, right foot on ball pushing off

### Detailed Prompt:
```
Stylized game sprite of Kelly, the Rein Maker's Daughter, full-body running pose (CONTACT FRAME - LEFT FOOT DOWN), left leg extended forward with foot flat on ground bearing weight, right leg extended backward with heel lifted and only ball of foot touching ground for push-off, body slightly angled forward showing forward momentum, arms swinging naturally in opposition (left arm forward bent at elbow, right arm back bent at elbow), light blue crew-neck sweatshirt, blue jeans cuffed at ankles, white sneakers, long wavy brown hair flowing, determined expression looking forward-right, light linen tunic fused with delicate brass/circuit accents (adapt reference clothing to fantasy game style while maintaining exact pose), small glowing circuit-rein brooch on chest, readable silhouette, facing right, orthographic, cel-shaded, painterly texture pass, soft forge key light, cool rim light, transparent background, master size 1024x1280 for downscaling, no background, no watermark. 

Style: Game sprite, side-scrolling runner character, clean edges, consistent art style. Based on reference photo of woman in casual clothing mid-stride - left foot down, right foot pushing off. Adapt the pose exactly from reference but stylize the clothing to match game's fantasy aesthetic (light linen tunic with brass/circuit accents instead of sweatshirt, keep jeans/sneakers aesthetic but make them game-appropriate). The pose should feel like she's just planted her left foot and is pushing off with her right.

Negative: photo, blurry, watermark, static pose, facing left, complex background, realistic photo style, both feet flat on ground.
```

---

## Frame 1: Passing/Mid-Stride Frame
**File:** `player_run_1.png`
**Reference Pose:** Right leg planted, left leg crossing mid-stride, body upright

### Detailed Prompt:
```
Stylized game sprite of Kelly, the Rein Maker's Daughter, full-body running pose (PASSING / MID-STRIDE FRAME), right leg planted firmly on ground with knee slightly bent bearing weight, left leg bent at knee and lifted crossing past the standing leg mid-stride with foot off ground, body more upright and facing forward, torso rotated slightly to show dynamic motion, arms swinging in opposition (right arm forward bent at elbow, left arm back bent at elbow), light blue crew-neck sweatshirt, blue jeans cuffed at ankles, white sneakers, long wavy brown hair flowing, determined expression looking forward-right, light linen tunic fused with delicate brass/circuit accents (adapt reference clothing to fantasy game style while maintaining exact pose), small glowing circuit-rein brooch on chest, readable silhouette, facing right, orthographic, cel-shaded, painterly texture pass, soft forge key light, cool rim light, transparent background, master size 1024x1280 for downscaling, no background, no watermark.

Style: Game sprite, side-scrolling runner character, clean edges, consistent art style. Based on reference photo of woman in mid-stride passing position - right leg down, left leg crossing. This is the "passing" frame where legs are crossing mid-stride. The pose should feel like she's mid-transition with her left leg crossing past her right. Adapt the pose exactly from reference but stylize clothing to match game's fantasy aesthetic.

Negative: photo, blurry, watermark, static pose, both feet on ground, no motion, facing left, complex background, realistic photo style, legs spread wide.
```

---

## Frame 2: Air/Maximum Extension Frame
**File:** `player_run_2.png`
**Reference Pose:** Left leg forward extended, right leg back extended, both feet off ground

### Detailed Prompt:
```
Stylized game sprite of Kelly, the Rein Maker's Daughter, full-body running pose (AIR / MAXIMUM EXTENSION FRAME), left leg forward fully extended with foot slightly off ground preparing to land, right leg backward fully extended with foot completely off ground, both feet clearly off the ground at maximum extension showing she's airborne, body stretched forward in full running stride, arms at maximum swing (left arm forward fully extended, right arm back fully extended), light blue crew-neck sweatshirt, blue jeans cuffed at ankles, white sneakers, long wavy brown hair flowing dramatically with motion, determined expression looking forward-right, light linen tunic fused with delicate brass/circuit accents flowing dramatically with motion (adapt reference clothing to fantasy game style while maintaining exact pose), small glowing circuit-rein brooch on chest, maximum forward momentum and speed, readable silhouette, facing right, orthographic, cel-shaded, painterly texture pass, soft forge key light, cool rim light, transparent background, master size 1024x1280 for downscaling, no background, no watermark.

Style: Game sprite, side-scrolling runner character, clean edges, consistent art style. Based on reference photo of woman at maximum extension - left leg forward, right leg back, both feet off ground. This is the "peak" frame where she's fully extended and airborne. The pose should feel like maximum speed and extension - both legs spread wide, arms at maximum swing, body stretched forward. Adapt the pose exactly from reference but stylize clothing to match game's fantasy aesthetic.

Negative: photo, blurry, watermark, static pose, feet on ground, no motion, facing left, complex background, legs together, compact pose, realistic photo style, both feet on ground.
```

---

## Pose Reference Details

### Frame 0 (Contact) - Based on Reference:
- **Left Foot:** Flat on ground, bearing all weight
- **Right Foot:** Ball of foot touching, heel lifted (push-off position)
- **Body:** Slightly angled forward, head looking forward-right
- **Arms:** Left arm forward, right arm back (natural opposition)
- **Key:** Moment of weight transfer - left foot just planted

### Frame 1 (Passing) - Based on Reference:
- **Right Leg:** Firmly planted, knee bent, bearing weight
- **Left Leg:** Bent at knee, crossing past right leg, foot off ground
- **Body:** More upright, torso rotated slightly forward
- **Arms:** Right arm forward, left arm back (natural opposition)
- **Key:** Mid-stride transition - legs crossing each other

### Frame 2 (Air/Maximum Extension) - Based on Reference:
- **Left Leg:** Forward fully extended, foot off ground (preparing to land)
- **Right Leg:** Back fully extended, foot off ground (maximum extension)
- **Body:** Fully stretched forward, maximum extension
- **Arms:** Left arm forward extended, right arm back extended
- **Key:** Peak of stride - both feet off ground, maximum speed

---

## Art Style Translation Guide

**Reference Elements → Game Elements:**

| Reference | Game Adaptation |
|-----------|----------------|
| Light blue sweatshirt | Light linen tunic with brass/circuit decorative accents |
| Blue jeans cuffed at ankles | Game-appropriate pants/legwear (keep cuffed aesthetic) |
| White sneakers | Practical boots/shoes (keep clean, simple aesthetic) |
| Long wavy brown hair | Same (long wavy brown hair) |
| Casual, practical clothing | Fantasy-casual with circuit/brass accents |
| Clean white background | Transparent background for sprite |

**Maintain:**
- Exact poses from reference
- Body proportions
- Hair style and color
- Expression (determined, focused)
- Arm/leg positions
- Forward momentum feel

**Adapt:**
- Clothing style (fantasy game aesthetic)
- Add circuit-rein brooch
- Add brass/circuit accents
- Stylize to cel-shaded game art
- Remove photo-realistic details

---

## Animation Timing Notes

**Frame Rate:** 10 FPS (as per code: `frameRate: 10`)
**Frame Order:** 0 → 1 → 2 → 0 (loop)
**Duration:** Each frame displayed for 100ms

**Code Implementation:**
```typescript
// Load all 3 frames in MenuScene.ts
this.load.image('player_run_0', 'player_run_0.png');
this.load.image('player_run_1', 'player_run_1.png');
this.load.image('player_run_2', 'player_run_2.png');

// Create animation in GameScene.ts
this.anims.create({
  key: 'run',
  frames: [
    { key: 'player_run_0' },
    { key: 'player_run_1' },
    { key: 'player_run_2' }
  ],
  frameRate: 10,
  repeat: -1
});

// Play animation when on ground
if (this.player.body?.touching.down) {
  this.player.play('run', true);
} else {
  this.player.stop();
}
```

---

## Consistency Checklist

When generating these frames, ensure:
- ✅ Same character design as reference (hair, build, proportions)
- ✅ Exact poses from reference images
- ✅ Consistent fantasy clothing adaptation across all frames
- ✅ Same color palette (light linen, brass accents, teal glow)
- ✅ Same art style (cel-shaded, painterly texture)
- ✅ Same lighting (soft forge key light, cool rim light)
- ✅ Same size (1024x1280px master)
- ✅ Same facing direction (right)
- ✅ Transparent background (CRITICAL - no opaque backgrounds)
- ✅ Bottom-center pivot point (for ground alignment)
- ✅ Readable silhouette at small size
- ✅ Smooth motion progression across all 3 frames

---

## Next Steps After Generation

1. **Fix transparency** on all frames (remove backgrounds)
2. **Save as PNG-24** with alpha channel
3. **Rename current file:**
   - `player.png` → `player_run_0.png` (after transparency fix)
4. **Verify animation cycle:**
   - Frame 0 → Frame 1 → Frame 2 → Frame 0 feels smooth
   - No jarring jumps between frames
   - Consistent character size across all frames
5. **Update code** in `MenuScene.ts` to load all 3 frames
6. **Update code** in `GameScene.ts` to create animation and play it
7. **Test in game** - animation should loop smoothly when running

---

## Alternative: 2-Frame Cycle (Simpler)

If you prefer a simpler 2-frame cycle:
- **Frame 0:** Contact (existing)
- **Frame 1:** Air/Maximum Extension (new)

This creates a simpler "bounce" animation but still looks good. Use Frame 2 prompt for the single additional frame needed.

# Kelly Image Mapping - First Pass

## Overview
Clean zoom system with parallel image sets for full-screen and panel-open states. **Only 16:9 images used.**

## Image Mapping

### Full Screen (Panel Closed)

| Zoom Level | Image | Description |
|------------|-------|-------------|
| **0 - Close-up** | `soft smile lips closed.png` | Face only, close-up headshot |
| **1 - Head & Shoulders** | `curious kelly.PNG` | Head and shoulders view (default) |
| **2 - Upper Body** | `Curious Kelly in final pose in Chair - UI elements will go on the side rails.png` | Upper body with director's chair |
| **3 - Full Body** | `kelly standing in jeans.png` | Full body standing |

### Panel Open (Kelly Looks at Panel)

| Zoom Level | Image | Description |
|------------|-------|-------------|
| **0 - Close-up** | `soft smile lips closed.png` | Same as full screen (no panel-specific close-up) |
| **1 - Head & Shoulders** | `kelly-looking at calendar and settings menu.png` | **Looking directly at panel** |
| **2 - Upper Body** | `Kelly pointing to middle of calendar and settings.png` | **Pointing at panel** |
| **3 - Full Body** | `t position.png` | Full body T-pose (oriented toward panel) |

### Fallback Images (Panel Open)
- **Head & Shoulders fallback**: `facing to the left.png` (if interaction image not available)
- **Upper Body fallback**: `stage with chair.png` (if interaction image not available)

## 16:9 Verified Images ✅

All images used are confirmed 16:9 aspect ratio:
- ✅ `soft smile lips closed.png` (1344:768, 1.75)
- ✅ `curious kelly.PNG` (2383:1340, 1.78)
- ✅ `facing to the left.png` (1392:752, 1.85)
- ✅ `Curious Kelly in final pose in Chair...` (7680:4320, 1.78)
- ✅ `stage with chair.png` (1344:768, 1.75)
- ✅ `kelly standing in jeans.png` (1920:1080, 1.78)
- ✅ `t position.png` (1344:768, 1.75)
- ✅ `Kelly pointing to middle of calendar and settings.png` (1920:1080, 1.78)
- ✅ `kelly-looking at calendar and settings menu.png` (1920:1080, 1.78)

## Excluded Images ❌

- ❌ `kelly square.jpg` - Square format (0.99 ratio)
- ❌ `neutral face with hair.png` - Square format (1.00 ratio)

## Current Behavior

1. **Default**: Head & Shoulders view with `curious kelly.PNG`
2. **Zoom In (+ button)**: Closer views (Close-up → Head & Shoulders)
3. **Zoom Out (- button)**: Wider views (Head & Shoulders → Upper Body → Full Body)
4. **Panel Opens**: Automatically switches to panel-aware images
5. **Panel Closes**: Returns to full-screen images

## Review & Adjustments Needed

Please review this mapping and let me know:
- Are the zoom level assignments appropriate?
- Should any images be swapped between zoom levels?
- Are the panel-open images the right choices?
- Should we use different images for specific zoom levels?

## Next Steps

Once approved, we can:
- Fine-tune image selection
- Add smooth transitions
- Optimize image loading
- Prepare for Unity 3D model integration


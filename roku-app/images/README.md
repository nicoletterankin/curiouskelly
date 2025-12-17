# Roku Channel Image Assets

## Required Images

These images are referenced in the `manifest` file and MUST be created before channel submission.

### Channel Icons

| File | Dimensions | Format | Description |
|------|------------|--------|-------------|
| `icon_focus_hd.png` | 540×405 | PNG | HD channel icon (main store listing) |
| `icon_focus_sd.png` | 290×218 | PNG | SD channel icon (older Roku devices) |

### Splash Screens

| File | Dimensions | Format | Description |
|------|------------|--------|-------------|
| `splash_hd.jpg` | 1280×720 | JPEG | HD splash screen |
| `splash_sd.jpg` | 720×480 | JPEG | SD splash screen |

## Design Guidelines

### Icon Design
- Use Kelly's face/avatar prominently
- Include "✨ Curious Kelly" text or just the K mark
- Kelly Blue background (#2563eb)
- High contrast for visibility on TV screens
- Leave padding from edges (safe zone: 10% from edges)

### Splash Screen Design
- Full Kelly visual or branded background
- Can include "Loading..." indicator
- Kelly Blue gradient background
- App name centered

## Source Assets

Generate from:
- `assets/kelly-brand-final/images/brand/kelly-logo-square.png`
- `assets/kelly-brand-final/images/expressions/curious-main.jpeg`

## Generation Commands

```bash
# Using ImageMagick (if installed)
# Icon HD (540x405)
magick convert kelly-logo-square.png -resize 540x405 -gravity center -background "#2563eb" -extent 540x405 icon_focus_hd.png

# Icon SD (290x218)
magick convert kelly-logo-square.png -resize 290x218 -gravity center -background "#2563eb" -extent 290x218 icon_focus_sd.png

# Splash HD (1280x720)
magick convert curious-main.jpeg -resize 1280x720^ -gravity center -extent 1280x720 splash_hd.jpg

# Splash SD (720x480)
magick convert curious-main.jpeg -resize 720x480^ -gravity center -extent 720x480 splash_sd.jpg
```

## Validation

Before submission, verify:
- [ ] All 4 files exist in this folder
- [ ] Dimensions exactly match requirements
- [ ] File formats correct (PNG for icons, JPEG for splash)
- [ ] Colors display correctly on TV (test on actual Roku)











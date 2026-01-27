# Ziggurat LED Visualization

Professional before/after visualization project for the Chet Holifield Federal Building.

## Project Structure

```
ziggurat/
├── index.html          # Presentation gallery with before/after sliders
├── compositor.html     # Professional compositor tool for creating "after" images
├── before/             # Original source photographs (untouched)
│   ├── hero-01-quarter-view.jpg
│   ├── hero-02-frontal.jpg
│   ├── hero-03-corner.jpg
│   └── hero-04-detail.jpg
├── after/              # Composited "after" images (your work)
│   └── (export from compositor)
└── README.md
```

## Workflow

1. **Open Compositor** (`compositor.html`)
2. **Select a hero image** from the dropdown
3. **Draw LED panels** using the polygon tool:
   - Click to add points
   - Press `Enter` to close the polygon
   - Press `Escape` to cancel
4. **Adjust style** — glow, color, opacity
5. **Toggle view** — press `Space` to switch between Before/After
6. **Export** — click "Export After" to save the composite

## Image Sources

All source images are from the Library of Congress Carol M. Highsmith Archive (2006):
- [LOC Resource pplot.13819](https://www.loc.gov/resource/pplot.13819/)

No known restrictions on publication per LOC terms.

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `P` | Polygon tool |
| `R` | Rectangle tool |
| `V` | Select tool |
| `Space` | Toggle Before/After view |
| `Enter` | Close current polygon |
| `Escape` | Cancel current drawing |
| `Delete` | Delete selected panel |

## Output

Export creates high-resolution JPEG files at original image dimensions.
Files are saved to your downloads folder as: `{image-name}-after-{time-preset}.jpg`

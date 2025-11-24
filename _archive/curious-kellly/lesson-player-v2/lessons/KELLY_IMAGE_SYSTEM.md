# Kelly Image Placeholder System

## Overview
Robust image loading system for Kelly placeholder images with zoom controls, ready for Unity 3D model integration.

## Image Files
The system uses these actual Kelly images found in the `lessons/` directory:

1. **Close-up** (`soft smile lips closed.png`)
   - Level 0 - Face only, close-up headshot
   - Use for: Intimate moments, emotional responses

2. **Head & Shoulders** (`kelly square.jpg`)
   - Level 1 - Default view, head and shoulders
   - Use for: Standard lesson delivery

3. **Upper Body** (`Curious Kelly in final pose in Chair - UI elements will go on the side rails.png`)
   - Level 2 - Upper body with director's chair
   - Use for: Contextual lessons, showing environment

4. **Full Body** (`t position.png`)
   - Level 3 - Full body T-pose
   - Use for: Wide shots, full scene context

## Features

### Zoom Controls
- **Zoom In Button** (+): Closer view (decreases level)
- **Zoom Out Button** (-): Wider view (increases level)
- **Zoom Indicator**: Shows current zoom level name
- **Keyboard Shortcuts**: `+`/`=` to zoom in, `-`/`_` to zoom out

### Image Loading
- Preloads images before displaying
- Smooth fade transitions between zoom levels
- Automatic fallback if image fails to load
- Loading status indicator
- Error handling with graceful degradation

### Integration Points
- Images are loaded from the `lessons/` directory
- System automatically detects available images
- Fallback text display if images are missing
- Ready for Unity WebGL canvas replacement

## Usage

### Default Behavior
- Starts at Level 1 (Head & Shoulders)
- Automatically loads `kelly square.jpg`
- Shows zoom controls in top-left corner

### Zooming
```javascript
// Programmatic zoom control
calendarApp.zoomIn();      // Closer view
calendarApp.zoomOut();     // Wider view
calendarApp.setKellyZoomLevel(2); // Direct level set
```

### Image Paths
All images are loaded relative to the HTML file location:
- `soft smile lips closed.png`
- `kelly square.jpg`
- `Curious Kelly in final pose in Chair - UI elements will go on the side rails.png`
- `t position.png`

## Future Unity Integration

When Unity 3D model is ready:
1. Replace `<img>` tag with Unity WebGL canvas
2. Keep zoom level system (map to Unity camera positions)
3. Maintain same zoom control interface
4. Unity will handle lipsyncing and animations

## Troubleshooting

### Images Not Showing
1. Check file names match exactly (case-sensitive)
2. Verify files are in `lessons/` directory
3. Check browser console for loading errors
4. Fallback text will display if images fail

### Zoom Controls Not Working
1. Ensure JavaScript is loaded
2. Check browser console for errors
3. Verify zoom buttons are visible (top-left corner)

## Next Steps
- [ ] Test all 4 zoom levels
- [ ] Add smooth camera transitions
- [ ] Integrate with lesson phases (auto-zoom for different phases)
- [ ] Add zoom level persistence (remember user preference)
- [ ] Prepare for Unity 3D model replacement


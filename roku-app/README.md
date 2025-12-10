# Curious Kelly - Roku Channel

Native Roku channel application.

## Features

- ✅ WebView wrapper for curiouskelly.com
- ✅ Loading indicator
- ✅ Remote control navigation
- ✅ HD/FHD support
- ✅ Cookie & localStorage support

## Development

### Prerequisites
- Roku device or Roku TV
- Roku Developer Mode enabled
- Network access to Roku device

### Enable Developer Mode on Roku
1. Press Home 3x, Up 2x, Right, Left, Right, Left, Right
2. Set a developer password
3. Note the IP address

### Deploy to Roku

```bash
# Package the channel
zip -r curious-kelly.zip manifest source components images

# Upload via web interface
# Go to http://[ROKU_IP] in browser
# Upload curious-kelly.zip
```

### Testing
1. Enable developer mode on Roku device
2. Package app as .zip
3. Upload via Roku developer portal
4. Test on device

## Building for Submission

```bash
# Create channel package
zip -r curious-kelly.zip manifest source components images

# Submit to Roku Channel Store
# 1. Log in to https://developer.roku.com
# 2. Go to "Manage My Channels"
# 3. Upload curious-kelly.zip
# 4. Fill in all metadata
# 5. Submit for certification
```

## Required Assets

### Channel Icons
- **Poster Art**: 540x405 PNG/JPEG
- **Channel Icon**: 290x218 PNG/JPEG

### Screenshots
- **HD**: 1280x720 PNG/JPEG (3-4 required)
- **FHD**: 1920x1080 PNG/JPEG (optional)

### Splash Screens
- **HD**: 1280x720 JPEG
- **SD**: 720x480 JPEG

## Roku Channel Store

### Submission Process
1. Create developer account at https://developer.roku.com
2. Go to "Manage My Channels" → "Add Channel"
3. Upload channel package (.zip)
4. Fill in channel properties
5. Upload all required assets
6. Complete content rating questionnaire
7. Submit for certification

### Review Time
- **Initial**: 2-4 weeks
- **Resubmission**: 1-2 weeks
- **No expedited review available**

### Certification Requirements
- No crashes or errors
- Proper remote control navigation
- Acceptable content rating
- All required assets provided
- Privacy policy accessible

## Architecture

```
roku-app/
├── manifest              # Channel metadata
├── source/
│   └── main.brs         # Main entry point
├── components/
│   ├── MainScene.xml    # Main scene UI
│   └── MainScene.brs    # Main scene logic
└── images/              # Channel assets
    ├── icon_focus_hd.png
    ├── icon_focus_sd.png
    ├── splash_hd.jpg
    └── splash_sd.jpg
```

## Tech Stack

- **BrightScript** - Roku programming language
- **SceneGraph** - Roku UI framework
- **WebView** - For displaying curiouskelly.com

## URLs

- **Production**: https://curiouskelly.com
- **Channel Store**: https://channelstore.roku.com/details/[CHANNEL_ID]/curious-kelly

## License

MIT © 2025 Lesson of the Day PBC










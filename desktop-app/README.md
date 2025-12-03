# Curious Kelly - Desktop App

Native desktop application for Windows, macOS, and Linux.

## Features

- ✅ Native window controls
- ✅ Auto-updates
- ✅ Offline support (coming soon)
- ✅ System tray integration (coming soon)
- ✅ Native notifications
- ✅ Keyboard shortcuts
- ✅ Deep linking

## Development

```bash
# Install dependencies
npm install

# Run in development mode
npm run dev

# Build for current platform
npm run build

# Build for specific platform
npm run build:win    # Windows
npm run build:mac    # macOS
npm run build:linux  # Linux

# Build for all platforms
npm run build:all
```

## Building

### Prerequisites

**Windows**:
- Node.js 18+
- Windows 10/11

**macOS**:
- Node.js 18+
- Xcode Command Line Tools
- macOS 11+

**Linux**:
- Node.js 18+
- Required packages: `libgtk-3-0 libnotify4 libnss3 libxss1 libxtst6 xdg-utils libatspi2.0-0 libdrm2 libgbm1 libxcb-dri3-0`

### Build Commands

```bash
# Install dependencies
npm install

# Development
npm start              # Run app
npm run dev           # Run with DevTools

# Production builds
npm run build:win     # Windows installer + portable
npm run build:mac     # macOS DMG (Intel + Apple Silicon)
npm run build:linux   # AppImage, Snap, Deb

# Build all platforms (requires macOS for Mac builds)
npm run build:all
```

## Distribution

Built apps will be in `dist/` directory:

- **Windows**: `.exe` installer + portable `.exe`
- **macOS**: `.dmg` installer + `.zip` archive
- **Linux**: `.AppImage`, `.snap`, `.deb`

## Auto-Updates

The app checks for updates on startup and notifies users when new versions are available.

Updates are distributed via GitHub Releases.

## Architecture

```
desktop-app/
├── src/
│   ├── main.js       # Main process (Electron)
│   └── preload.js    # Preload script (security bridge)
├── build/            # Build resources (icons, etc.)
├── dist/             # Built applications
└── package.json      # Dependencies & build config
```

## Tech Stack

- **Electron 28** - Cross-platform desktop framework
- **electron-builder** - Build & packaging
- **electron-updater** - Auto-update functionality

## URLs

- **Production**: https://curiouskelly.com
- **Development**: http://localhost:4321

## Security

- Context isolation enabled
- Node integration disabled
- Remote module disabled
- Secure preload script

## License

MIT © 2025 Lesson of the Day PBC







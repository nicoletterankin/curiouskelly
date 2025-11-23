# The Rein Maker's Daughter - Runner Game

A story-driven endless runner game built with Phaser 3 and TypeScript.

## 🎮 Game Features

- **Endless Runner Gameplay**: Jump, dodge obstacles, collect Knowledge Stones
- **Seven Tribes Lore**: Collect stones from all seven tribes (Light, Stone, Metal, Code, Air, Water, Fire)
- **Progressive Difficulty**: Game speed increases over time
- **Score System**: Earn points for survival and collecting stones

## 🚀 Quick Start

### Development

```bash
# Install dependencies
npm install

# Start dev server (opens at http://localhost:3000)
npm run dev
```

### Build for Production

```bash
# Build optimized version
npm run build

# Preview production build
npm run preview
```

## 🎯 Controls

- **SPACE** or **UP ARROW** or **Click/Tap**: Jump
- **R**: Restart (after game over)

## 📦 Deployment to Itch.io

### Step 1: Build

```bash
npm run build
```

This creates a `dist/` folder with your game.

### Step 2: Prepare for Itch.io

1. Zip the contents of the `dist/` folder (not the folder itself)
2. Go to [itch.io](https://itch.io)
3. Create a new project → "HTML"
4. Upload the zip file
5. Check "This file will be played in the browser"
6. Set viewport dimensions: 800 x 600
7. Enable fullscreen button (optional)
8. Save & Publish!

### Step 3: Settings

Recommended Itch.io settings:
- **Kind of project**: HTML
- **Viewport dimensions**: 800 x 600
- **Fullscreen button**: Yes
- **Mobile friendly**: Yes (touch controls work)
- **Frame options**: None

## 🛠️ Tech Stack

- **Phaser 3.85**: Game framework
- **TypeScript**: Type-safe development
- **Vite**: Fast build tool and dev server
- **Arcade Physics**: Simple 2D physics

## 📁 Project Structure

```
reinmaker-runner-game/
├── src/
│   ├── main.ts              # Game initialization
│   └── scenes/
│       ├── MenuScene.ts     # Title screen & menu
│       └── GameScene.ts     # Main gameplay
├── public/
│   └── assets/              # Game assets (sprites, sounds)
├── index.html               # Entry point
├── vite.config.ts           # Vite configuration
└── tsconfig.json            # TypeScript configuration
```

## 🎨 Assets

All assets are located in `public/assets/`:
- Player sprite (Kelly character)
- Obstacles (Knowledge Shards)
- Collectibles (7 tribe stones)
- Backgrounds and ground textures
- UI elements

## 🔧 Development Notes

### Adding New Features

1. Create new scene in `src/scenes/`
2. Register scene in `src/main.ts`
3. Add assets to `public/assets/`
4. Load assets in scene's `preload()` method

### Performance

- Target: 60 FPS
- Physics: Arcade (lightweight)
- Assets: Pre-loaded on menu screen
- Garbage collection: Old sprites auto-destroyed

## 📝 Todo / Future Enhancements

- [ ] Connect to quest API (backend integration)
- [ ] Add power-ups
- [ ] Sound effects and music
- [ ] Multiple character skins
- [ ] Leaderboards
- [ ] Mobile optimization

## 📄 License

Part of the UI-TARS ecosystem.

---

**Built in 1 day to ship something!** 🚀
















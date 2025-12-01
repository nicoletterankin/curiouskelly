# Curious Kelly - Mobile App

Native mobile application for iOS and Android.

## Features

- ✅ Native WebView wrapper
- ✅ Offline data caching
- ✅ Native splash screen
- ✅ Deep linking support
- ✅ Push notifications (coming soon)
- ✅ Biometric authentication (coming soon)
- ✅ Native sharing

## Development

### Prerequisites

**iOS**:
- macOS with Xcode 14+
- CocoaPods (`sudo gem install cocoapods`)
- iOS Simulator or physical device

**Android**:
- Android Studio
- Android SDK (API 23+)
- Java JDK 11+
- Android Emulator or physical device

### Setup

```bash
# Install dependencies
npm install

# iOS only - install pods
cd ios && pod install && cd ..

# Run on iOS
npm run ios

# Run on Android
npm run android

# Start Metro bundler
npm start
```

## Building

### iOS

```bash
# Development build
npm run ios

# Production build
npm run build:ios

# Or use Xcode:
# 1. Open ios/CuriousKelly.xcworkspace
# 2. Select Product > Archive
# 3. Follow App Store submission process
```

### Android

```bash
# Development build
npm run android

# Production build
npm run build:android

# Output: android/app/build/outputs/apk/release/app-release.apk
```

## App Store Submission

### iOS App Store

1. Build archive in Xcode
2. Upload to App Store Connect
3. Fill in app metadata
4. Submit for review

**Required**:
- Apple Developer Account ($99/year)
- App Store screenshots (6.5", 5.5")
- App icon (1024x1024)
- Privacy policy URL
- Support URL

### Google Play Store

1. Build signed APK/AAB
2. Upload to Google Play Console
3. Fill in store listing
4. Submit for review

**Required**:
- Google Play Developer Account ($25 one-time)
- Feature graphic (1024x500)
- Screenshots (phone + tablet)
- App icon (512x512)
- Privacy policy URL

## Architecture

```
mobile-app/
├── android/          # Android native code
├── ios/              # iOS native code
├── App.js            # Main React Native component
├── index.js          # Entry point
└── package.json      # Dependencies
```

## Tech Stack

- **React Native 0.73** - Cross-platform mobile framework
- **react-native-webview** - WebView component
- **react-native-splash-screen** - Native splash screen
- **@react-native-async-storage/async-storage** - Local storage
- **react-native-device-info** - Device information

## WebView Communication

The app uses `postMessage` to communicate between React Native and the web app:

```javascript
// From web app to native
window.sendToNative('SAVE_DATA', { key: 'user', value: userData });

// From native to web app
window.addEventListener('message', (event) => {
  const data = JSON.parse(event.data);
  // Handle data
});
```

## URLs

- **Production**: https://curiouskelly.com
- **Staging**: https://staging.curiouskelly.com (if needed)

## Security

- HTTPS only
- Certificate pinning (coming soon)
- Secure storage for sensitive data
- Biometric authentication (coming soon)

## License

MIT © 2025 Lesson of the Day PBC




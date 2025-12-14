import type { CapacitorConfig } from '@capacitor/cli';

const config: CapacitorConfig = {
  appId: 'com.curiouskelly.app',
  appName: 'Curious Kelly',
  // This wrapper intentionally serves the existing static web app:
  // `../public/learn.html` is the app.
  webDir: '../public',
  server: {
    androidScheme: 'https'
  },
  plugins: {
    SplashScreen: {
      launchShowDuration: 2000,
      backgroundColor: '#0f0f1a',
      showSpinner: false
    },
    StatusBar: {
      style: 'dark',
      backgroundColor: '#0f0f1a'
    },
    Keyboard: {
      resize: 'body',
      resizeOnFullScreen: true
    }
  },
  ios: {
    contentInset: 'always'
  }
};

export default config;




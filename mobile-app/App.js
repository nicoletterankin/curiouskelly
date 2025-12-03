import React, { useEffect, useRef, useState } from 'react';
import {
  SafeAreaView,
  StatusBar,
  StyleSheet,
  View,
  ActivityIndicator,
  Alert,
  Platform
} from 'react-native';
import { WebView } from 'react-native-webview';
import SplashScreen from 'react-native-splash-screen';
import AsyncStorage from '@react-native-async-storage/async-storage';
import DeviceInfo from 'react-native-device-info';

const APP_URL = 'https://curiouskelly.com';
const KELLY_BLUE = '#2563eb';

function App() {
  const webViewRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [canGoBack, setCanGoBack] = useState(false);

  useEffect(() => {
    // Hide splash screen after component mounts
    setTimeout(() => {
      SplashScreen.hide();
    }, 1000);

    // Log app info
    console.log('App Version:', DeviceInfo.getVersion());
    console.log('Platform:', Platform.OS);
  }, []);

  const handleNavigationStateChange = (navState) => {
    setCanGoBack(navState.canGoBack);
  };

  const handleMessage = (event) => {
    try {
      const message = JSON.parse(event.nativeEvent.data);
      console.log('Message from web:', message);

      // Handle messages from web app
      switch (message.type) {
        case 'SAVE_DATA':
          AsyncStorage.setItem(message.key, JSON.stringify(message.value));
          break;
        case 'GET_DATA':
          AsyncStorage.getItem(message.key).then(value => {
            webViewRef.current?.postMessage(JSON.stringify({
              type: 'DATA_RESPONSE',
              key: message.key,
              value: value ? JSON.parse(value) : null
            }));
          });
          break;
        default:
          console.log('Unknown message type:', message.type);
      }
    } catch (error) {
      console.error('Error handling message:', error);
    }
  };

  const handleError = (syntheticEvent) => {
    const { nativeEvent } = syntheticEvent;
    console.error('WebView error:', nativeEvent);
    Alert.alert(
      'Connection Error',
      'Unable to load Curious Kelly. Please check your internet connection.',
      [{ text: 'Retry', onPress: () => webViewRef.current?.reload() }]
    );
  };

  const injectedJavaScript = `
    // Inject native app detection
    window.isNativeApp = true;
    window.nativeAppPlatform = '${Platform.OS}';
    window.nativeAppVersion = '${DeviceInfo.getVersion()}';

    // Send message to React Native
    window.sendToNative = function(type, data) {
      window.ReactNativeWebView.postMessage(JSON.stringify({ type, ...data }));
    };

    // Override console for debugging
    const originalLog = console.log;
    console.log = function(...args) {
      originalLog.apply(console, args);
      window.ReactNativeWebView.postMessage(JSON.stringify({ 
        type: 'CONSOLE_LOG', 
        message: args.join(' ') 
      }));
    };

    true; // Required for injected JavaScript
  `;

  return (
    <SafeAreaView style={styles.container}>
      <StatusBar barStyle="light-content" backgroundColor={KELLY_BLUE} />
      
      <WebView
        ref={webViewRef}
        source={{ uri: APP_URL }}
        style={styles.webview}
        onNavigationStateChange={handleNavigationStateChange}
        onMessage={handleMessage}
        onError={handleError}
        onLoadStart={() => setLoading(true)}
        onLoadEnd={() => setLoading(false)}
        injectedJavaScript={injectedJavaScript}
        javaScriptEnabled={true}
        domStorageEnabled={true}
        startInLoadingState={true}
        scalesPageToFit={true}
        allowsBackForwardNavigationGestures={true}
        cacheEnabled={true}
        cacheMode="LOAD_CACHE_ELSE_NETWORK"
        renderLoading={() => (
          <View style={styles.loadingContainer}>
            <ActivityIndicator size="large" color={KELLY_BLUE} />
          </View>
        )}
      />

      {loading && (
        <View style={styles.loadingOverlay}>
          <ActivityIndicator size="large" color={KELLY_BLUE} />
        </View>
      )}
    </SafeAreaView>
  );
}

const styles = StyleSheet.create({
  container: {
    flex: 1,
    backgroundColor: '#0a0a0b',
  },
  webview: {
    flex: 1,
  },
  loadingContainer: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: '#0a0a0b',
  },
  loadingOverlay: {
    position: 'absolute',
    top: 0,
    left: 0,
    right: 0,
    bottom: 0,
    justifyContent: 'center',
    alignItems: 'center',
    backgroundColor: 'rgba(10, 10, 11, 0.8)',
  },
});

export default App;







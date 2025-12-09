import React, { useEffect, useRef, useState } from 'react';
import {
  SafeAreaView,
  StatusBar,
  StyleSheet,
  View,
  ActivityIndicator,
  Alert,
  Platform,
  AppState
} from 'react-native';
import { WebView } from 'react-native-webview';
import SplashScreen from 'react-native-splash-screen';
import AsyncStorage from '@react-native-async-storage/async-storage';
import DeviceInfo from 'react-native-device-info';
import messaging from '@react-native-firebase/messaging';
import notifee, { AndroidImportance, EventType } from '@notifee/react-native';

const APP_URL = 'https://curiouskelly.com';
const KELLY_BLUE = '#2563eb';
const API_BASE = 'https://curiouskelly.com/api';

function App() {
  const webViewRef = useRef(null);
  const [loading, setLoading] = useState(true);
  const [canGoBack, setCanGoBack] = useState(false);
  const appState = useRef(AppState.currentState);

  useEffect(() => {
    // Hide splash screen after component mounts
    setTimeout(() => {
      SplashScreen.hide();
    }, 1000);

    // Log app info
    console.log('App Version:', DeviceInfo.getVersion());
    console.log('Platform:', Platform.OS);

    // Initialize push notifications
    initializePushNotifications();

    // Handle app state changes
    const subscription = AppState.addEventListener('change', handleAppStateChange);

    // Handle notification taps when app is in background
    const unsubscribe = messaging().onNotificationOpenedApp(remoteMessage => {
      console.log('Notification opened app:', remoteMessage);
      handleNotificationTap(remoteMessage);
    });

    // Check if app was opened from a notification (when killed)
    messaging()
      .getInitialNotification()
      .then(remoteMessage => {
        if (remoteMessage) {
          console.log('App opened from quit state:', remoteMessage);
          handleNotificationTap(remoteMessage);
        }
      });

    // Handle foreground notifications
    const unsubscribeForeground = messaging().onMessage(async remoteMessage => {
      console.log('Foreground notification:', remoteMessage);
      displayLocalNotification(remoteMessage);
    });

    // Handle notification events from notifee
    const unsubscribeNotifee = notifee.onForegroundEvent(({ type, detail }) => {
      switch (type) {
        case EventType.PRESS:
          console.log('Notification pressed:', detail.notification);
          if (detail.notification?.data?.url) {
            webViewRef.current?.injectJavaScript(
              `window.location.href = '${detail.notification.data.url}';`
            );
          }
          break;
      }
    });

    return () => {
      subscription.remove();
      unsubscribe();
      unsubscribeForeground();
      unsubscribeNotifee();
    };
  }, []);

  const handleAppStateChange = (nextAppState) => {
    if (appState.current.match(/inactive|background/) && nextAppState === 'active') {
      // App has come to foreground - refresh token if needed
      refreshPushToken();
    }
    appState.current = nextAppState;
  };

  const initializePushNotifications = async () => {
    try {
      // Request permission
      const authStatus = await messaging().requestPermission();
      const enabled =
        authStatus === messaging.AuthorizationStatus.AUTHORIZED ||
        authStatus === messaging.AuthorizationStatus.PROVISIONAL;

      if (!enabled) {
        console.log('Push notification permission not granted');
        return;
      }

      console.log('Push notification permission granted:', authStatus);

      // Create notification channel for Android
      if (Platform.OS === 'android') {
        await notifee.createChannel({
          id: 'kelly_daily',
          name: 'Kelly Daily Lessons',
          description: 'Daily lesson reminders from Kelly',
          importance: AndroidImportance.HIGH,
          vibration: true,
          sound: 'default'
        });

        await notifee.createChannel({
          id: 'kelly_streaks',
          name: 'Streak Notifications',
          description: 'Streak saves and celebrations',
          importance: AndroidImportance.DEFAULT
        });
      }

      // Get FCM token
      const token = await messaging().getToken();
      console.log('FCM Token:', token);

      // Register token with our server
      await registerPushToken(token);

      // Listen for token refresh
      messaging().onTokenRefresh(async newToken => {
        console.log('FCM Token refreshed:', newToken);
        await registerPushToken(newToken);
      });
    } catch (error) {
      console.error('Error initializing push notifications:', error);
    }
  };

  const refreshPushToken = async () => {
    try {
      const token = await messaging().getToken();
      await registerPushToken(token);
    } catch (error) {
      console.error('Error refreshing push token:', error);
    }
  };

  const registerPushToken = async (token) => {
    try {
      // Get user_id from AsyncStorage if logged in
      const userData = await AsyncStorage.getItem('kelly-user');
      const user = userData ? JSON.parse(userData) : null;

      // Get timezone
      const timezone = Intl.DateTimeFormat().resolvedOptions().timeZone;

      const response = await fetch(`${API_BASE}/notifications/subscribe-device`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
          // Include auth token if user is logged in
          ...(user?.accessToken && { 'Authorization': `Bearer ${user.accessToken}` })
        },
        body: JSON.stringify({
          device_token: token,
          platform: Platform.OS,
          device_name: await DeviceInfo.getDeviceName(),
          device_model: DeviceInfo.getModel(),
          app_version: DeviceInfo.getVersion(),
          os_version: DeviceInfo.getSystemVersion(),
          timezone
        })
      });

      const result = await response.json();
      console.log('Push token registered:', result);

      // Store token locally
      await AsyncStorage.setItem('kelly-push-token', token);
    } catch (error) {
      console.error('Error registering push token:', error);
    }
  };

  const displayLocalNotification = async (remoteMessage) => {
    try {
      await notifee.displayNotification({
        title: remoteMessage.notification?.title || "✨ Kelly's going live!",
        body: remoteMessage.notification?.body || "Today's lesson is starting. Join millions learning together.",
        android: {
          channelId: 'kelly_daily',
          smallIcon: 'ic_notification',
          color: KELLY_BLUE,
          pressAction: {
            id: 'default'
          }
        },
        ios: {
          foregroundPresentationOptions: {
            alert: true,
            badge: true,
            sound: true
          }
        },
        data: remoteMessage.data
      });
    } catch (error) {
      console.error('Error displaying notification:', error);
    }
  };

  const handleNotificationTap = (remoteMessage) => {
    // Navigate to specific URL if provided
    const url = remoteMessage.data?.url;
    if (url && webViewRef.current) {
      webViewRef.current.injectJavaScript(
        `window.location.href = '${url}';`
      );
    }
  };

  const handleNavigationStateChange = (navState) => {
    setCanGoBack(navState.canGoBack);
  };

  const handleMessage = async (event) => {
    try {
      const message = JSON.parse(event.nativeEvent.data);
      console.log('Message from web:', message);

      // Handle messages from web app
      switch (message.type) {
        case 'SAVE_DATA':
          await AsyncStorage.setItem(message.key, JSON.stringify(message.value));
          break;
        case 'GET_DATA':
          const value = await AsyncStorage.getItem(message.key);
          webViewRef.current?.postMessage(JSON.stringify({
            type: 'DATA_RESPONSE',
            key: message.key,
            value: value ? JSON.parse(value) : null
          }));
          break;
        case 'USER_LOGGED_IN':
          // User logged in - update push token registration
          await AsyncStorage.setItem('kelly-user', JSON.stringify(message.user));
          refreshPushToken();
          break;
        case 'USER_LOGGED_OUT':
          // User logged out - clear user data
          await AsyncStorage.removeItem('kelly-user');
          break;
        case 'REQUEST_NOTIFICATION_PERMISSION':
          // Web is requesting native notification permission
          initializePushNotifications();
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
    window.pushNotificationsSupported = true;

    // Send message to React Native
    window.sendToNative = function(type, data) {
      window.ReactNativeWebView.postMessage(JSON.stringify({ type, ...data }));
    };

    // Request native notification permission
    window.requestNativeNotifications = function() {
      window.sendToNative('REQUEST_NOTIFICATION_PERMISSION', {});
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

    // Notify web app that native is ready
    window.dispatchEvent(new CustomEvent('nativeAppReady', { 
      detail: { platform: '${Platform.OS}', version: '${DeviceInfo.getVersion()}' }
    }));

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

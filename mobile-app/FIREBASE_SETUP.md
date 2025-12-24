# Firebase Configuration

## Where to Get Real Keys

1. Go to [Firebase Console](https://console.firebase.google.com)
2. Create or select project: `curious-kelly`
3. Add apps:
   - **Android**: Package name `com.curiouskelly.mobile`
   - **iOS**: Bundle ID `com.curiouskelly.mobile`

## Where to Drop Real Keys

### Android

Replace `android/app/google-services.json` with the downloaded file from Firebase Console.

### iOS

Replace `ios/CuriousKellyMobile/GoogleService-Info.plist` with the downloaded file from Firebase Console.

## Required Keys

### google-services.json (Android)

| Key                                      | Description                        |
| ---------------------------------------- | ---------------------------------- |
| `project_info.project_number`            | Firebase project number            |
| `project_info.project_id`                | Firebase project ID                |
| `project_info.storage_bucket`            | Storage bucket URL                 |
| `client[0].client_info.mobilesdk_app_id` | Android app ID (1:xxx:android:xxx) |
| `client[0].api_key[0].current_key`       | API key                            |

### GoogleService-Info.plist (iOS)

| Key                  | Description                        |
| -------------------- | ---------------------------------- |
| `CLIENT_ID`          | OAuth 2.0 client ID                |
| `REVERSED_CLIENT_ID` | Reversed client ID for URL schemes |
| `API_KEY`            | API key                            |
| `GCM_SENDER_ID`      | Cloud Messaging sender ID          |
| `PROJECT_ID`         | Firebase project ID                |
| `STORAGE_BUCKET`     | Storage bucket URL                 |
| `GOOGLE_APP_ID`      | iOS app ID (1:xxx:ios:xxx)         |

## Enabling Push Notifications

1. In Firebase Console → Project Settings → Cloud Messaging
2. Note the **Server Key** for backend API
3. For iOS: Upload APNs authentication key or certificate

## Security Note

Never commit real Firebase keys to public repositories. Use environment variables or secure vaults in CI/CD.



















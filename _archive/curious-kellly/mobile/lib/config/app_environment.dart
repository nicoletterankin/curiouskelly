import 'dart:developer' as developer;

/// Centralized environment configuration for the Curious Kellly mobile app.
///
/// Values are read from compile-time `--dart-define` flags when available and
/// fall back to sane local defaults so the app can run out of the box during
/// development.
class AppEnvironment {
  AppEnvironment({
    required this.backendBaseUrl,
    required this.openAiApiKey,
    required this.defaultLearnerAge,
  });

  /// Loads configuration from compile-time environment with fallbacks.
  factory AppEnvironment.fromPlatform() {
    final backend = const String.fromEnvironment(
      'CK_BACKEND_BASE_URL',
      defaultValue: 'http://localhost:3000',
    );

    final apiKey = const String.fromEnvironment(
      'CK_OPENAI_API_KEY',
      defaultValue: '',
    );

    final ageString = const String.fromEnvironment(
      'CK_DEFAULT_LEARNER_AGE',
      defaultValue: '',
    );

    final age = int.tryParse(ageString);
    if (age == null && ageString.isNotEmpty) {
      developer.log(
        'Invalid CK_DEFAULT_LEARNER_AGE="$ageString". Falling back to $defaultLearnerAgeFallback.',
        name: 'AppEnvironment',
        level: 900, // warning
      );
    }

    return AppEnvironment(
      backendBaseUrl: backend,
      openAiApiKey: apiKey,
      defaultLearnerAge: age ?? defaultLearnerAgeFallback,
    );
  }

  /// Base URL for the Curious Kellly backend service.
  final String backendBaseUrl;

  /// API key passed to the realtime voice service (empty string for local dev).
  final String openAiApiKey;

  /// Default learner age used when one is not supplied via navigation.
  final int defaultLearnerAge;

  /// Default fallback age when no environment override is supplied.
  static const int defaultLearnerAgeFallback = 35;

  /// Convenience helper for constructing websocket URLs.
  String get realtimeSocketUrl => '$backendBaseUrl/api/realtime';
}























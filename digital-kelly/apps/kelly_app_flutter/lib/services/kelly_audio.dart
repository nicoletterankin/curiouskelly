import 'dart:io';

/// Audio service for Kelly
/// 
/// NOTE: Primary audio playback is handled by Unity for authoritative sync.
/// Flutter's audioplayers is retained for fallback tests only.
/// 
/// The Unity engine receives both the AudioClip (WAV) and A2F frames,
/// and plays them together with frame-accurate synchronization using
/// AudioSettings.dspTime. This ensures facial animation stays in sync
/// with speech audio (±1 frame at 30fps).
class KellyAudio {
  /// Play audio using Flutter (fallback only)
  /// For production, use Unity's audio playback
  static Future<void> playTest() async {
    // Placeholder for fallback audio test
    print('⚠️  KellyAudio: Use Unity playback for production');
  }

  /// Get platform-safe path for WAV file
  static Future<String?> getWavPath(String filename) async {
    // Check user's test directory first
    final testPath = await getTestPath(filename);
    if (testPath != null && File(testPath).existsSync()) {
      return testPath;
    }
    return null;
  }

  static Future<String?> getTestPath(String filename) async {
    final home = Platform.environment['HOME'] ?? 
                  Platform.environment['USERPROFILE'];
    if (home == null) return null;
    
    final testDir = Platform.isWindows 
        ? '$home\\DigitalKellyTest\\audio'
        : '$home/DigitalKellyTest/audio';
    
    return Platform.isWindows 
        ? '$testDir\\$filename'
        : '$testDir/$filename';
  }
}



























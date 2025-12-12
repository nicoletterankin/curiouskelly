import 'dart:io';
import 'package:path_provider/path_provider.dart';

/// Platform-safe path resolution helpers
class LocalPaths {
  /// Get the platform's documents directory
  static Future<String> getDocumentsDir() async {
    final dir = await getApplicationDocumentsDirectory();
    return dir.path;
  }

  /// Get the Kelly test directory (user-created)
  static Future<String?> getKellyTestDir() async {
    final home = Platform.environment['HOME'] ?? 
                  Platform.environment['USERPROFILE'];
    
    if (home == null) return null;
    
    return Platform.isWindows 
        ? '$home\\DigitalKellyTest'
        : '$home/DigitalKellyTest';
  }

  /// Get audio file path in test directory
  static Future<String?> getAudioPath(String filename) async {
    final testDir = await getKellyTestDir();
    if (testDir == null) return null;
    
    final audioDir = Platform.isWindows 
        ? '$testDir\\audio'
        : '$testDir/audio';
    
    return Platform.isWindows 
        ? '$audioDir\\$filename'
        : '$audioDir/$filename';
  }

  /// Get A2F JSON file path in test directory
  static Future<String?> getA2fPath(String filename) async {
    final testDir = await getKellyTestDir();
    if (testDir == null) return null;
    
    final a2fDir = Platform.isWindows 
        ? '$testDir\\a2f'
        : '$testDir/a2f';
    
    return Platform.isWindows 
        ? '$a2fDir\\$filename'
        : '$a2fDir/$filename';
  }

  /// Check if test directory exists
  static Future<bool> testDirExists() async {
    final testDir = await getKellyTestDir();
    if (testDir == null) return false;
    return Directory(testDir).existsSync();
  }

  /// Copy asset to documents directory
  static Future<String?> copyAssetToDocuments(
    String assetPath,
    String filename,
  ) async {
    final docsDir = await getDocumentsDir();
    final targetDir = Directory('$docsDir/kelly_assets');
    if (!targetDir.existsSync()) {
      targetDir.createSync(recursive: true);
    }
    
    final targetPath = Platform.isWindows
        ? '$docsDir\\kelly_assets\\$filename'
        : '$docsDir/kelly_assets/$filename';
    
    // Note: Asset copying needs platform-specific implementation
    // For now, return the expected path
    return targetPath;
  }
}



























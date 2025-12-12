import 'dart:io';
import 'package:just_audio/just_audio.dart';
import 'package:path_provider/path_provider.dart';
import 'package:http/http.dart' as http;
import 'package:logger/logger.dart';

/// Lesson Audio Player
/// Plays lesson audio files for specific age groups
class LessonAudioPlayer {
  final Logger _logger = Logger();
  final AudioPlayer _player = AudioPlayer();
  
  // Audio cache directory
  Directory? _cacheDir;
  
  // Current state
  String? _currentLessonId;
  String? _currentAgeGroup;
  bool _isPlaying = false;
  
  // Callbacks
  Function()? onPlaybackStarted;
  Function()? onPlaybackComplete;
  Function(Duration position, Duration duration)? onProgress;
  
  LessonAudioPlayer() {
    _initializeCache();
    _setupPlayerListeners();
  }
  
  Future<void> _initializeCache() async {
    final appDir = await getApplicationDocumentsDirectory();
    _cacheDir = Directory('${appDir.path}/lesson_audio');
    
    if (!_cacheDir!.existsSync()) {
      _cacheDir!.createSync(recursive: true);
    }
    
    _logger.i('[LessonAudioPlayer] Cache initialized: ${_cacheDir!.path}');
  }
  
  void _setupPlayerListeners() {
    _player.playerStateStream.listen((state) {
      if (state.playing && !_isPlaying) {
        _isPlaying = true;
        onPlaybackStarted?.call();
        _logger.i('[LessonAudioPlayer] Playback started');
      } else if (!state.playing && _isPlaying) {
        _isPlaying = false;
        if (state.processingState == ProcessingState.completed) {
          onPlaybackComplete?.call();
          _logger.i('[LessonAudioPlayer] Playback completed');
        }
      }
    });
    
    _player.positionStream.listen((position) {
      final duration = _player.duration;
      if (duration != null) {
        onProgress?.call(position, duration);
      }
    });
  }
  
  /// Play lesson audio for specific section and age
  /// 
  /// [lessonId]: e.g., "water-cycle"
  /// [ageGroup]: e.g., "18-35"
  /// [section]: "welcome", "mainContent", or "wisdomMoment"
  /// [backendUrl]: URL to backend API (optional, uses local if cached)
  Future<bool> playLessonAudio({
    required String lessonId,
    required String ageGroup,
    required String section,
    String? backendUrl,
  }) async {
    try {
      _currentLessonId = lessonId;
      _currentAgeGroup = ageGroup;
      
      _logger.i('[LessonAudioPlayer] Playing: $lessonId / $ageGroup / $section');
      
      // Check if cached
      final cachedFile = await _getCachedAudioFile(lessonId, ageGroup, section);
      
      if (cachedFile != null && cachedFile.existsSync()) {
        // Play from cache
        _logger.d('[LessonAudioPlayer] Playing from cache');
        await _player.setFilePath(cachedFile.path);
        await _player.play();
        return true;
      }
      
      // Download from backend if URL provided
      if (backendUrl != null) {
        final success = await _downloadAndCacheAudio(
          backendUrl,
          lessonId,
          ageGroup,
          section,
        );
        
        if (success) {
          final file = await _getCachedAudioFile(lessonId, ageGroup, section);
          if (file != null) {
            await _player.setFilePath(file.path);
            await _player.play();
            return true;
          }
        }
      }
      
      _logger.e('[LessonAudioPlayer] Audio file not found');
      return false;
      
    } catch (e) {
      _logger.e('[LessonAudioPlayer] Error playing audio: $e');
      return false;
    }
  }
  
  /// Play complete lesson (all sections in sequence)
  Future<void> playCompleteLesson({
    required String lessonId,
    required String ageGroup,
    String? backendUrl,
  }) async {
    _logger.i('[LessonAudioPlayer] Playing complete lesson: $lessonId');
    
    final sections = ['welcome', 'mainContent', 'wisdomMoment'];
    
    for (final section in sections) {
      final success = await playLessonAudio(
        lessonId: lessonId,
        ageGroup: ageGroup,
        section: section,
        backendUrl: backendUrl,
      );
      
      if (success) {
        // Wait for section to complete before playing next
        await _waitForPlaybackComplete();
      } else {
        _logger.w('[LessonAudioPlayer] Failed to play section: $section');
      }
    }
    
    _logger.i('[LessonAudioPlayer] Complete lesson finished');
  }
  
  Future<void> _waitForPlaybackComplete() async {
    while (_isPlaying) {
      await Future.delayed(const Duration(milliseconds: 100));
    }
  }
  
  /// Download audio from backend and cache locally
  Future<bool> _downloadAndCacheAudio(
    String backendUrl,
    String lessonId,
    String ageGroup,
    String section,
  ) async {
    try {
      final url = '$backendUrl/api/lessons/$lessonId/audio/$ageGroup/$section.mp3';
      _logger.d('[LessonAudioPlayer] Downloading: $url');
      
      final response = await http.get(Uri.parse(url));
      
      if (response.statusCode == 200) {
        final file = File('${_cacheDir!.path}/${lessonId}_${ageGroup}_$section.mp3');
        await file.writeAsBytes(response.bodyBytes);
        
        _logger.i('[LessonAudioPlayer] Downloaded and cached: ${file.path}');
        return true;
      } else {
        _logger.e('[LessonAudioPlayer] Download failed: ${response.statusCode}');
        return false;
      }
    } catch (e) {
      _logger.e('[LessonAudioPlayer] Download error: $e');
      return false;
    }
  }
  
  /// Get cached audio file
  Future<File?> _getCachedAudioFile(
    String lessonId,
    String ageGroup,
    String section,
  ) async {
    if (_cacheDir == null) {
      await _initializeCache();
    }
    
    final file = File('${_cacheDir!.path}/${lessonId}_${ageGroup}_$section.mp3');
    return file.existsSync() ? file : null;
  }
  
  /// Load audio from local file path (for testing)
  Future<bool> playLocalFile(String filePath) async {
    try {
      _logger.i('[LessonAudioPlayer] Playing local file: $filePath');
      await _player.setFilePath(filePath);
      await _player.play();
      return true;
    } catch (e) {
      _logger.e('[LessonAudioPlayer] Error playing local file: $e');
      return false;
    }
  }
  
  /// Pause playback
  Future<void> pause() async {
    await _player.pause();
    _logger.i('[LessonAudioPlayer] Paused');
  }
  
  /// Resume playback
  Future<void> resume() async {
    await _player.play();
    _logger.i('[LessonAudioPlayer] Resumed');
  }
  
  /// Stop playback
  Future<void> stop() async {
    await _player.stop();
    _isPlaying = false;
    _logger.i('[LessonAudioPlayer] Stopped');
  }
  
  /// Seek to position
  Future<void> seek(Duration position) async {
    await _player.seek(position);
  }
  
  /// Set volume (0.0 to 1.0)
  Future<void> setVolume(double volume) async {
    await _player.setVolume(volume.clamp(0.0, 1.0));
  }
  
  /// Get current position
  Duration? get position => _player.position;
  
  /// Get duration
  Duration? get duration => _player.duration;
  
  /// Check if playing
  bool get isPlaying => _isPlaying;
  
  /// Get current lesson info
  Map<String, String?> get currentLesson => {
    'lessonId': _currentLessonId,
    'ageGroup': _currentAgeGroup,
  };
  
  /// Clear cache
  Future<void> clearCache() async {
    if (_cacheDir != null && _cacheDir!.existsSync()) {
      await _cacheDir!.delete(recursive: true);
      await _initializeCache();
      _logger.i('[LessonAudioPlayer] Cache cleared');
    }
  }
  
  /// Get cache size
  Future<int> getCacheSize() async {
    if (_cacheDir == null || !_cacheDir!.existsSync()) {
      return 0;
    }
    
    int totalSize = 0;
    await for (final file in _cacheDir!.list(recursive: true)) {
      if (file is File) {
        totalSize += await file.length();
      }
    }
    
    return totalSize;
  }
  
  /// Dispose
  void dispose() {
    _player.dispose();
  }
}























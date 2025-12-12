import 'dart:async';
import 'dart:typed_data';
import 'package:just_audio/just_audio.dart';
import 'package:logger/logger.dart';
import '../flutter_unity_bridge.dart';

/// Audio Player Service for Kelly's voice
/// Handles real-time audio playback with low latency
/// Integrates with Unity for viseme-driven lip-sync
class AudioPlayerService {
  final Logger _logger = Logger();
  final AudioPlayer _player = AudioPlayer();
  
  // Unity bridge for viseme integration
  FlutterUnityBridge? _unityBridge;
  
  // State
  bool _isPlaying = false;
  final StreamController<PlayerState> _stateController = StreamController.broadcast();
  
  // Callbacks
  Function()? onPlaybackStarted;
  Function()? onPlaybackComplete;
  Function(Duration position)? onPositionUpdate;
  
  AudioPlayerService({FlutterUnityBridge? unityBridge}) {
    _unityBridge = unityBridge;
    _setupPlayer();
  }
  
  /// Set Unity bridge for viseme integration
  void setUnityBridge(FlutterUnityBridge bridge) {
    _unityBridge = bridge;
  }
  
  void _setupPlayer() {
    // Listen to player state
    _player.playerStateStream.listen((state) {
      _stateController.add(state);
      
      if (state.playing) {
        if (!_isPlaying) {
          _isPlaying = true;
          onPlaybackStarted?.call();
          _logger.i('[AudioPlayer] Playback started');
        }
      } else {
        if (_isPlaying) {
          _isPlaying = false;
          if (state.processingState == ProcessingState.completed) {
            onPlaybackComplete?.call();
            _logger.i('[AudioPlayer] Playback completed');
          }
        }
      }
    });
    
    // Listen to position
    _player.positionStream.listen((position) {
      onPositionUpdate?.call(position);
    });
  }
  
  /// Play audio from bytes
  /// If Unity bridge is available, sends audio to Unity for viseme-driven playback
  Future<void> playAudioBytes(Uint8List audioBytes) async {
    try {
      _logger.d('[AudioPlayer] Playing ${audioBytes.length} bytes');
      
      // If Unity bridge is available, send audio to Unity for viseme integration
      if (_unityBridge != null) {
        _unityBridge!.playAudio(audioBytes);
        _logger.i('[AudioPlayer] Audio sent to Unity for viseme playback');
        return;
      }
      
      // Fallback: Use just_audio for playback
      // Note: just_audio requires a proper audio format (MP3, WAV, etc.)
      // For raw PCM, you may need to use audioplayers package instead
      
      // For now, assume audioBytes is a complete audio file (MP3/WAV)
      await _player.setAudioSource(
        MemoryAudioSource(audioBytes),
      );
      
      await _player.play();
      
    } catch (e) {
      _logger.e('[AudioPlayer] Error playing audio: $e');
    }
  }
  
  /// Update visemes for Unity lip-sync
  void updateVisemes(Map<String, dynamic> visemes) {
    if (_unityBridge != null) {
      _unityBridge!.updateVisemes(visemes);
    }
  }
  
  /// Play audio from URL
  Future<void> playAudioUrl(String url) async {
    try {
      _logger.i('[AudioPlayer] Playing from URL: $url');
      await _player.setUrl(url);
      await _player.play();
    } catch (e) {
      _logger.e('[AudioPlayer] Error playing URL: $e');
    }
  }
  
  /// Stop playback
  Future<void> stop() async {
    await _player.stop();
    _isPlaying = false;
    _logger.i('[AudioPlayer] Stopped');
  }
  
  /// Pause playback
  Future<void> pause() async {
    await _player.pause();
    _logger.i('[AudioPlayer] Paused');
  }
  
  /// Resume playback
  Future<void> resume() async {
    await _player.play();
    _logger.i('[AudioPlayer] Resumed');
  }
  
  /// Set volume (0.0 to 1.0)
  Future<void> setVolume(double volume) async {
    await _player.setVolume(volume.clamp(0.0, 1.0));
  }
  
  /// Seek to position
  Future<void> seek(Duration position) async {
    await _player.seek(position);
  }
  
  /// Get current position
  Duration? get position => _player.position;
  
  /// Get duration
  Duration? get duration => _player.duration;
  
  /// Check if playing
  bool get isPlaying => _isPlaying;
  
  /// Get state stream
  Stream<PlayerState> get stateStream => _stateController.stream;
  
  /// Dispose
  void dispose() {
    _player.dispose();
    _stateController.close();
  }
}

/// Custom audio source from memory (bytes)
class MemoryAudioSource extends StreamAudioSource {
  final Uint8List _audioBytes;
  
  MemoryAudioSource(this._audioBytes);
  
  @override
  Future<StreamAudioResponse> request([int? start, int? end]) async {
    start ??= 0;
    end ??= _audioBytes.length;
    
    return StreamAudioResponse(
      sourceLength: _audioBytes.length,
      contentLength: end - start,
      offset: start,
      stream: Stream.value(_audioBytes.sublist(start, end)),
      contentType: 'audio/wav', // Adjust based on your audio format
    );
  }
}




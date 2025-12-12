import 'dart:async';
import 'dart:math';
import 'dart:typed_data';
import 'package:logger/logger.dart';

/// Voice Activity Detection (VAD)
/// Detects when user starts/stops speaking
class VoiceActivityDetector {
  final Logger _logger = Logger();
  
  // VAD configuration
  final double silenceThreshold;
  final Duration silenceDuration;
  final Duration speechDuration;
  
  // State
  bool _isSpeaking = false;
  DateTime? _lastSpeechTime;
  DateTime? _speechStartTime;
  List<double> _energyHistory = [];
  
  // Callbacks
  Function()? onSpeechStart;
  Function()? onSpeechEnd;
  Function(double energy)? onEnergyUpdate;
  
  VoiceActivityDetector({
    this.silenceThreshold = 0.02, // Adjust based on testing
    this.silenceDuration = const Duration(milliseconds: 500),
    this.speechDuration = const Duration(milliseconds: 300),
  });
  
  /// Process audio buffer and detect speech
  void processAudio(Uint8List audioData) {
    final energy = _calculateEnergy(audioData);
    _energyHistory.add(energy);
    
    // Keep history limited (last 10 samples)
    if (_energyHistory.length > 10) {
      _energyHistory.removeAt(0);
    }
    
    onEnergyUpdate?.call(energy);
    
    final now = DateTime.now();
    
    if (energy > silenceThreshold) {
      // Speech detected
      _lastSpeechTime = now;
      
      if (!_isSpeaking) {
        // Check if speech duration threshold met
        if (_speechStartTime == null) {
          _speechStartTime = now;
        } else if (now.difference(_speechStartTime!) >= speechDuration) {
          _isSpeaking = true;
          _speechStartTime = null;
          onSpeechStart?.call();
          _logger.i('[VAD] Speech started (energy: ${energy.toStringAsFixed(4)})');
        }
      }
    } else {
      // Silence detected
      if (_isSpeaking && _lastSpeechTime != null) {
        if (now.difference(_lastSpeechTime!) >= silenceDuration) {
          _isSpeaking = false;
          _speechStartTime = null;
          onSpeechEnd?.call();
          _logger.i('[VAD] Speech ended');
        }
      } else {
        // Reset speech start if silence before threshold
        _speechStartTime = null;
      }
    }
  }
  
  /// Calculate audio energy (RMS)
  double _calculateEnergy(Uint8List audioData) {
    if (audioData.isEmpty) return 0.0;
    
    double sum = 0.0;
    
    // Convert bytes to 16-bit samples
    for (int i = 0; i < audioData.length - 1; i += 2) {
      int sample = audioData[i] | (audioData[i + 1] << 8);
      if (sample > 32767) sample -= 65536; // Convert to signed
      sum += sample * sample;
    }
    
    final rms = sqrt(sum / (audioData.length / 2));
    final normalized = rms / 32768.0; // Normalize to 0-1
    
    return normalized;
  }
  
  /// Get current speech status
  bool get isSpeaking => _isSpeaking;
  
  /// Get average energy (last 10 samples)
  double get averageEnergy {
    if (_energyHistory.isEmpty) return 0.0;
    return _energyHistory.reduce((a, b) => a + b) / _energyHistory.length;
  }
  
  /// Reset VAD state
  void reset() {
    _isSpeaking = false;
    _lastSpeechTime = null;
    _speechStartTime = null;
    _energyHistory.clear();
    _logger.i('[VAD] Reset');
  }
}























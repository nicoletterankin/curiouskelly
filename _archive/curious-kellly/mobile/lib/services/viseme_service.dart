import 'dart:async';
import 'package:logger/logger.dart';

/// Viseme Service
/// Converts audio output to viseme data for Unity lip-sync
class VisemeService {
  final Logger _logger = Logger();
  
  // Viseme mapping (OpenAI Realtime API visemes → Unity blendshapes)
  final Map<String, String> _visemeMap = {
    'sil': 'viseme_sil',  // Silence
    'PP': 'viseme_PP',    // P, B, M
    'FF': 'viseme_FF',    // F, V
    'TH': 'viseme_TH',    // TH
    'DD': 'viseme_DD',    // D, T, N, L
    'kk': 'viseme_kk',    // K, G, NG
    'CH': 'viseme_CH',    // CH, J, SH, ZH
    'SS': 'viseme_SS',    // S, Z
    'nn': 'viseme_nn',    // N
    'RR': 'viseme_RR',    // R
    'aa': 'viseme_aa',    // A
    'E': 'viseme_E',      // E
    'ih': 'viseme_ih',    // I
    'oh': 'viseme_oh',    // O
    'ou': 'viseme_ou',    // U
  };
  
  // Current viseme state
  String _currentViseme = 'sil';
  final StreamController<Map<String, double>> _visemeController = StreamController.broadcast();
  
  // Callbacks
  Function(Map<String, double> visemes)? onVisemesUpdated;
  
  /// Process viseme data from OpenAI Realtime API
  void processVisemes(Map<String, dynamic> visemeData) {
    try {
      final visemes = <String, double>{};
      
      // Extract viseme values from API response
      if (visemeData['visemes'] != null) {
        final apiVisemes = visemeData['visemes'] as Map<String, dynamic>;
        
        // Map OpenAI visemes to Unity blendshapes
        for (final entry in apiVisemes.entries) {
          final openAIViseme = entry.key;
          final value = (entry.value as num).toDouble();
          
          final unityViseme = _visemeMap[openAIViseme] ?? 'viseme_sil';
          visemes[unityViseme] = value.clamp(0.0, 1.0);
        }
      }
      
      // Ensure all visemes have values (fill missing with 0)
      for (final viseme in _visemeMap.values) {
        visemes.putIfAbsent(viseme, () => 0.0);
      }
      
      // Update current viseme (highest value)
      if (visemes.isNotEmpty) {
        final maxEntry = visemes.entries.reduce(
          (a, b) => a.value > b.value ? a : b,
        );
        _currentViseme = maxEntry.key;
      }
      
      // Emit viseme update
      _visemeController.add(visemes);
      onVisemesUpdated?.call(visemes);
      
      _logger.d('[Viseme] Updated: $_currentViseme (${visemes.length} visemes)');
    } catch (e) {
      _logger.e('[Viseme] Error processing visemes: $e');
    }
  }
  
  /// Get current viseme values
  Map<String, double> getCurrentVisemes() {
    final visemes = <String, double>{};
    for (final viseme in _visemeMap.values) {
      visemes[viseme] = viseme == _currentViseme ? 1.0 : 0.0;
    }
    return visemes;
  }
  
  /// Get viseme stream for Unity integration
  Stream<Map<String, double>> get visemeStream => _visemeController.stream;
  
  /// Get current active viseme
  String get currentViseme => _currentViseme;
  
  /// Reset visemes to silence
  void reset() {
    _currentViseme = 'sil';
    final silence = <String, double>{};
    for (final viseme in _visemeMap.values) {
      silence[viseme] = viseme == 'viseme_sil' ? 1.0 : 0.0;
    }
    _visemeController.add(silence);
    onVisemesUpdated?.call(silence);
  }
  
  /// Dispose
  void dispose() {
    _visemeController.close();
  }
}





















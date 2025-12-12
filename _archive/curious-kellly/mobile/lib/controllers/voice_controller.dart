import 'package:flutter/foundation.dart';
import 'package:logger/logger.dart';
import '../services/openai_realtime_service.dart';
import '../services/voice_activity_detector.dart';
import '../services/audio_player_service.dart';
import '../services/permission_service.dart';
import '../services/viseme_service.dart';
import '../flutter_unity_bridge.dart';

/// Main Voice Controller
/// Coordinates voice input, OpenAI Realtime, and audio output
class VoiceController extends ChangeNotifier {
  final Logger _logger = Logger();
  
  // Services
  late OpenAIRealtimeService _realtimeService;
  late VoiceActivityDetector _vad;
  late AudioPlayerService _audioPlayer;
  late VisemeService _visemeService;
  final PermissionService _permissionService = PermissionService();
  
  // Unity bridge for viseme integration
  FlutterUnityBridge? _unityBridge;
  
  // State
  VoiceState _state = VoiceState.disconnected;
  int _learnerAge = 35;
  String? _lastUserText;
  String? _lastKellyText;
  int _latencyMs = 0;
  double _audioEnergy = 0.0;
  
  // Configuration
  final String backendUrl;
  
  VoiceController({
    required this.backendUrl,
  }) {
    _initializeServices();
  }
  
  void _initializeServices() {
    // Initialize OpenAI Realtime Service (no API key needed - uses ephemeral key)
    _realtimeService = OpenAIRealtimeService(
      backendUrl: backendUrl,
    );
    
    // Initialize Voice Activity Detector
    _vad = VoiceActivityDetector(
      silenceThreshold: 0.02,
      silenceDuration: const Duration(milliseconds: 500),
      speechDuration: const Duration(milliseconds: 300),
    );
    
    // Initialize Audio Player (with Unity bridge if available)
    _audioPlayer = AudioPlayerService(unityBridge: _unityBridge);
    
    _setupCallbacks();
  }
  
  /// Set Unity bridge for viseme integration
  void setUnityBridge(FlutterUnityBridge bridge) {
    _unityBridge = bridge;
    _audioPlayer.setUnityBridge(bridge);
  }
  
  void _setupCallbacks() {
    // Realtime service callbacks
    _realtimeService.onEvent = (event) {
      _logger.d('[VoiceController] Event: ${event.type}');
      _handleRealtimeEvent(event);
    };
    
    _realtimeService.onTranscriptReceived = (text) {
      _lastUserText = text;
      _logger.i('[VoiceController] User: "$text"');
      notifyListeners();
    };
    
    _realtimeService.onKellyResponse = (text) {
      _lastKellyText = text;
      _logger.i('[VoiceController] Kelly: "$text"');
      notifyListeners();
    };
    
    _realtimeService.onAudioReceived = (audioBytes) {
      _logger.d('[VoiceController] Audio received: ${audioBytes.length} bytes');
      _audioPlayer.playAudioBytes(audioBytes);
    };
    
    _realtimeService.onLatencyUpdate = (latencyMs) {
      _latencyMs = latencyMs;
      notifyListeners();
    };
    
    _realtimeService.onVisemesReceived = (visemes) {
      // Send visemes to Unity for lip-sync
      if (_unityBridge != null) {
        _audioPlayer.updateVisemes(visemes);
      }
    };
    
    // VAD callbacks
    _vad.onSpeechStart = () {
      _logger.i('[VoiceController] User started speaking');
      setState(VoiceState.userSpeaking);
    };
    
    _vad.onSpeechEnd = () {
      _logger.i('[VoiceController] User stopped speaking');
      setState(VoiceState.processing);
    };
    
    _vad.onEnergyUpdate = (energy) {
      _audioEnergy = energy;
      notifyListeners();
    };
    
    // Audio player callbacks
    _audioPlayer.onPlaybackStarted = () {
      _logger.i('[VoiceController] Kelly started speaking');
      setState(VoiceState.kellySpeaking);
    };
    
    _audioPlayer.onPlaybackComplete = () {
      _logger.i('[VoiceController] Kelly finished speaking');
      setState(VoiceState.listening);
    };
  }
  
  void _handleRealtimeEvent(RealtimeEvent event) {
    switch (event.type) {
      case RealtimeEventType.connected:
        setState(VoiceState.connected);
        break;
      case RealtimeEventType.disconnected:
        setState(VoiceState.disconnected);
        break;
      case RealtimeEventType.listeningStarted:
        setState(VoiceState.listening);
        break;
      case RealtimeEventType.listeningStopped:
        setState(VoiceState.idle);
        break;
      case RealtimeEventType.userSpeech:
        setState(VoiceState.processing);
        break;
      case RealtimeEventType.kellyResponse:
        setState(VoiceState.kellySpeaking);
        break;
      case RealtimeEventType.bargeIn:
        setState(VoiceState.userSpeaking);
        break;
      case RealtimeEventType.error:
        setState(VoiceState.error);
        _logger.e('[VoiceController] Error: ${event.data}');
        break;
      default:
        break;
    }
  }
  
  /// Connect to OpenAI Realtime API
  Future<bool> connect({required int learnerAge, String? sessionId}) async {
    _learnerAge = learnerAge;
    setState(VoiceState.connecting);
    
    // Check microphone permission
    final hasPermission = await _permissionService.hasMicrophonePermission();
    if (!hasPermission) {
      final granted = await _permissionService.requestMicrophonePermission();
      if (!granted) {
        _logger.e('[VoiceController] Microphone permission denied');
        setState(VoiceState.error);
        return false;
      }
    }
    
    // Connect to realtime service (fetches ephemeral key internally)
    final success = await _realtimeService.connect(learnerAge: learnerAge, sessionId: sessionId);
    
    if (success) {
      setState(VoiceState.connected);
      _logger.i('[VoiceController] Connected successfully');
    } else {
      setState(VoiceState.error);
      _logger.e('[VoiceController] Connection failed');
    }
    
    return success;
  }
  
  /// Disconnect from service
  Future<void> disconnect() async {
    await _realtimeService.disconnect();
    setState(VoiceState.disconnected);
    _logger.i('[VoiceController] Disconnected');
  }
  
  /// Start listening to user
  void startListening() {
    if (_state != VoiceState.connected && _state != VoiceState.idle) {
      _logger.w('[VoiceController] Cannot start listening in current state: $_state');
      return;
    }
    
    _realtimeService.startListening();
    setState(VoiceState.listening);
    _vad.reset();
  }
  
  /// Stop listening
  void stopListening() {
    _realtimeService.stopListening();
    setState(VoiceState.idle);
    _vad.reset();
  }
  
  /// Barge-in: Interrupt Kelly
  void bargeIn() {
    if (_state != VoiceState.kellySpeaking) {
      _logger.w('[VoiceController] Cannot barge in, Kelly not speaking');
      return;
    }
    
    _audioPlayer.stop();
    _realtimeService.bargeIn();
    _vad.reset();
    setState(VoiceState.listening);
    _logger.i('[VoiceController] Barge-in successful');
  }
  
  /// Send text message (for testing)
  void sendMessage(String text) {
    _realtimeService.sendMessage(text);
    _lastUserText = text;
    notifyListeners();
  }
  
  /// Set learner age (updates Kelly's persona)
  void setLearnerAge(int age) {
    _learnerAge = age;
    notifyListeners();
  }
  
  /// Set voice state
  void setState(VoiceState newState) {
    if (_state != newState) {
      _logger.i('[VoiceController] State: $_state -> $newState');
      _state = newState;
      notifyListeners();
    }
  }
  
  // Getters
  VoiceState get state => _state;
  int get learnerAge => _learnerAge;
  String? get lastUserText => _lastUserText;
  String? get lastKellyText => _lastKellyText;
  int get latencyMs => _latencyMs;
  double get audioEnergy => _audioEnergy;
  double get averageLatencyMs => _realtimeService.averageLatencyMs;
  
  // Viseme service
  VisemeService get visemeService => _visemeService;
  Stream<Map<String, double>> get visemeStream => _visemeService.visemeStream;
  Map<String, double> get currentVisemes => _visemeService.getCurrentVisemes();
  
  bool get isConnected => _state != VoiceState.disconnected && _state != VoiceState.error;
  bool get isListening => _state == VoiceState.listening || _state == VoiceState.userSpeaking;
  bool get isKellySpeaking => _state == VoiceState.kellySpeaking;
  bool get canBargeIn => _state == VoiceState.kellySpeaking;
  
  @override
  void dispose() {
    _realtimeService.dispose();
    _audioPlayer.dispose();
    _visemeService.dispose();
    super.dispose();
  }
}

/// Voice interaction states
enum VoiceState {
  disconnected,   // Not connected
  connecting,     // Connecting to service
  connected,      // Connected but idle
  idle,           // Connected, not listening
  listening,      // Listening for user speech
  userSpeaking,   // User is speaking
  processing,     // Processing user speech
  kellySpeaking,  // Kelly is responding
  error,          // Error state
}

/// Extension for state descriptions
extension VoiceStateExtension on VoiceState {
  String get description {
    switch (this) {
      case VoiceState.disconnected:
        return 'Disconnected';
      case VoiceState.connecting:
        return 'Connecting...';
      case VoiceState.connected:
        return 'Ready';
      case VoiceState.idle:
        return 'Idle';
      case VoiceState.listening:
        return 'Listening...';
      case VoiceState.userSpeaking:
        return 'You are speaking';
      case VoiceState.processing:
        return 'Processing...';
      case VoiceState.kellySpeaking:
        return 'Kelly is speaking';
      case VoiceState.error:
        return 'Error';
    }
  }
  
  bool get isActive {
    return this == VoiceState.listening ||
           this == VoiceState.userSpeaking ||
           this == VoiceState.processing ||
           this == VoiceState.kellySpeaking;
  }
}




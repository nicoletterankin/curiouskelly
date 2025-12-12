import 'dart:async';
import 'dart:convert';
import 'dart:typed_data';
import 'package:flutter_webrtc/flutter_webrtc.dart';
import 'package:web_socket_channel/web_socket_channel.dart';
import 'package:http/http.dart' as http;
import 'package:logger/logger.dart';

/// OpenAI Realtime API Service
/// Handles WebRTC voice streaming with barge-in support
class OpenAIRealtimeService {
  final String backendUrl;
  final Logger _logger = Logger();
  
  // Ephemeral key
  Map<String, dynamic>? _ephemeralKey;
  String? _sessionId;
  
  // WebSocket connection
  WebSocketChannel? _wsChannel;
  bool _isConnected = false;
  Timer? _reconnectTimer;
  Timer? _pingTimer;
  int _reconnectAttempts = 0;
  static const int _maxReconnectAttempts = 3;
  static const Duration _reconnectDelay = Duration(seconds: 2);
  
  // WebRTC components
  RTCPeerConnection? _peerConnection;
  MediaStream? _localStream;
  RTCDataChannel? _dataChannel;
  
  // Audio processing
  final StreamController<Uint8List> _audioInputController = StreamController.broadcast();
  final StreamController<Uint8List> _audioOutputController = StreamController.broadcast();
  final StreamController<String> _transcriptController = StreamController.broadcast();
  final StreamController<RealtimeEvent> _eventController = StreamController.broadcast();
  
  // State
  bool _isSpeaking = false;
  bool _isListening = false;
  int _conversationId = 0;
  int? _learnerAge;
  
  // Performance metrics
  DateTime? _lastRequestTime;
  int _totalLatencyMs = 0;
  int _requestCount = 0;
  final List<int> _latencyHistory = [];
  static const int _maxLatencyHistory = 10;
  
  // Callbacks
  Function(String text)? onTranscriptReceived;
  Function(Uint8List audio)? onAudioReceived;
  Function(String kellyText)? onKellyResponse;
  Function(Map<String, dynamic> visemes)? onVisemesReceived;
  Function(RealtimeEvent event)? onEvent;
  Function(int latencyMs)? onLatencyUpdate;

  OpenAIRealtimeService({
    required this.backendUrl,
  });
  
  /// Fetch ephemeral key from backend
  Future<Map<String, dynamic>?> fetchEphemeralKey({required int learnerAge, String? sessionId}) async {
    try {
      final response = await http.post(
        Uri.parse('$backendUrl/api/realtime/ephemeral-key'),
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'learnerAge': learnerAge,
          'sessionId': sessionId,
        }),
      );
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        if (data['status'] == 'ok') {
          _ephemeralKey = data['data'];
          _sessionId = _ephemeralKey?['sessionId'];
          _learnerAge = learnerAge;
          _logger.i('[OpenAI Realtime] Ephemeral key fetched: ${_ephemeralKey?['sessionId']}');
          return _ephemeralKey;
        }
      }
      
      _logger.e('[OpenAI Realtime] Failed to fetch ephemeral key: ${response.statusCode}');
      return null;
    } catch (e) {
      _logger.e('[OpenAI Realtime] Error fetching ephemeral key: $e');
      return null;
    }
  }

  /// Get audio input stream (for monitoring)
  Stream<Uint8List> get audioInputStream => _audioInputController.stream;
  
  /// Get audio output stream (for playback)
  Stream<Uint8List> get audioOutputStream => _audioOutputController.stream;
  
  /// Get transcript stream (user speech-to-text)
  Stream<String> get transcriptStream => _transcriptController.stream;
  
  /// Get event stream (all realtime events)
  Stream<RealtimeEvent> get eventStream => _eventController.stream;
  
  /// Check if connected
  bool get isConnected => _isConnected;
  
  /// Check if Kelly is speaking
  bool get isSpeaking => _isSpeaking;
  
  /// Check if listening to user
  bool get isListening => _isListening;
  
  /// Get average latency
  double get averageLatencyMs => _requestCount > 0 ? _totalLatencyMs / _requestCount : 0;

  /// Initialize connection to OpenAI Realtime API
  Future<bool> connect({required int learnerAge, String? sessionId}) async {
    try {
      _logger.i('[OpenAI Realtime] Connecting... Age: $learnerAge');
      
      // 1. Fetch ephemeral key from backend
      final ephemeralKey = await fetchEphemeralKey(learnerAge: learnerAge, sessionId: sessionId);
      if (ephemeralKey == null) {
        _logger.e('[OpenAI Realtime] Failed to fetch ephemeral key');
        return false;
      }
      
      // 2. Setup WebRTC peer connection
      await _setupPeerConnection();
      
      // 3. Setup local audio stream (microphone)
      await _setupLocalStream();
      
      // 4. Connect to WebSocket (signaling)
      await _connectWebSocket(learnerAge: learnerAge, sessionId: sessionId);
      
      // 5. Create offer and send to backend
      await _createAndSendOffer(learnerAge);
      
      _isConnected = true;
      _reconnectAttempts = 0;
      _logger.i('[OpenAI Realtime] Connected successfully!');
      
      // Start ping timer for keepalive
      _startPingTimer();
      
      _emitEvent(RealtimeEvent(
        type: RealtimeEventType.connected,
        timestamp: DateTime.now(),
        data: {'learnerAge': learnerAge, 'sessionId': _sessionId},
      ));
      
      return true;
    } catch (e) {
      _logger.e('[OpenAI Realtime] Connection failed: $e');
      _emitEvent(RealtimeEvent(
        type: RealtimeEventType.error,
        timestamp: DateTime.now(),
        data: {'error': e.toString()},
      ));
      return false;
    }
  }

  /// Setup WebRTC peer connection
  Future<void> _setupPeerConnection() async {
    final configuration = {
      'iceServers': [
        {'urls': 'stun:stun.l.google.com:19302'},
      ],
    };
    
    final constraints = {
      'mandatory': {},
      'optional': [
        {'DtlsSrtpKeyAgreement': true},
      ],
    };
    
    _peerConnection = await createPeerConnection(configuration, constraints);
    
    // Handle ICE candidates
    _peerConnection!.onIceCandidate = (RTCIceCandidate candidate) {
      _logger.d('[WebRTC] ICE Candidate: ${candidate.candidate}');
      _sendWebSocketMessage({
        'type': 'ice_candidate',
        'candidate': {
          'candidate': candidate.candidate,
          'sdpMLineIndex': candidate.sdpMLineIndex,
          'sdpMid': candidate.sdpMid,
        },
      });
    };
    
    // Handle connection state changes
    _peerConnection!.onConnectionState = (RTCPeerConnectionState state) {
      _logger.i('[WebRTC] Connection state: $state');
      if (state == RTCPeerConnectionState.RTCPeerConnectionStateConnected) {
        _emitEvent(RealtimeEvent(
          type: RealtimeEventType.webrtcConnected,
          timestamp: DateTime.now(),
        ));
      }
    };
    
    // Handle incoming audio tracks
    _peerConnection!.onTrack = (RTCTrackEvent event) {
      _logger.i('[WebRTC] Remote track received: ${event.track.kind}');
      if (event.track.kind == 'audio') {
        _handleRemoteAudioTrack(event.streams[0]);
      }
    };
  }

  /// Setup local audio stream (microphone)
  Future<void> _setupLocalStream() async {
    try {
      // For mobile, we'll use the record package instead of WebRTC for audio capture
      // WebRTC is more complex on mobile and requires different setup
      
      // TODO: Integrate with record package for mobile audio capture
      // For now, this is a placeholder
      _logger.i('[Audio] Microphone access will be handled by record package');
      
      // When WebRTC is fully implemented:
      /*
      final mediaConstraints = {
        'audio': {
          'echoCancellation': true,
          'noiseSuppression': true,
          'autoGainControl': true,
          'sampleRate': 24000, // 24kHz for OpenAI Realtime
        },
        'video': false,
      };
      
      _localStream = await navigator.mediaDevices.getUserMedia(mediaConstraints);
      
      // Add audio track to peer connection
      if (_peerConnection != null) {
        _localStream!.getAudioTracks().forEach((track) {
          _peerConnection!.addTrack(track, _localStream!);
          _logger.i('[WebRTC] Added local audio track');
        });
      }
      */
    } catch (e) {
      _logger.e('[Audio] Failed to setup local stream: $e');
      // Don't throw - we can still use text input
    }
  }

  /// Connect to WebSocket for signaling
  Future<void> _connectWebSocket({required int learnerAge, String? sessionId}) async {
    final uri = Uri.parse('$backendUrl/api/realtime/ws');
    final queryParams = <String, String>{
      'learnerAge': learnerAge.toString(),
    };
    if (sessionId != null || _sessionId != null) {
      queryParams['sessionId'] = sessionId ?? _sessionId!;
    }
    
    final wsUrl = queryParams.isEmpty 
        ? uri 
        : uri.replace(queryParameters: queryParams);
    
    _wsChannel = WebSocketChannel.connect(wsUrl);
    
    // Listen to WebSocket messages
    _wsChannel!.stream.listen(
      (message) => _handleWebSocketMessage(message),
      onError: (error) {
        _logger.e('[WebSocket] Error: $error');
        _isConnected = false;
        _handleDisconnection();
        _emitEvent(RealtimeEvent(
          type: RealtimeEventType.error,
          timestamp: DateTime.now(),
          data: {'error': 'WebSocket error: $error'},
        ));
      },
      onDone: () {
        _logger.w('[WebSocket] Connection closed');
        _isConnected = false;
        _handleDisconnection();
        _emitEvent(RealtimeEvent(
          type: RealtimeEventType.disconnected,
          timestamp: DateTime.now(),
        ));
      },
    );
  }
  
  /// Handle disconnection with automatic reconnection
  void _handleDisconnection() {
    if (_reconnectAttempts < _maxReconnectAttempts && _learnerAge != null) {
      _reconnectAttempts++;
      _logger.i('[OpenAI Realtime] Scheduling reconnect attempt $_reconnectAttempts/$_maxReconnectAttempts');
      
      _reconnectTimer?.cancel();
      _reconnectTimer = Timer(_reconnectDelay, () async {
        _logger.i('[OpenAI Realtime] Attempting reconnect...');
        final success = await connect(learnerAge: _learnerAge!, sessionId: _sessionId);
        if (!success) {
          _handleDisconnection();
        }
      });
    } else {
      _logger.e('[OpenAI Realtime] Max reconnection attempts reached');
    }
  }
  
  /// Start ping timer for keepalive
  void _startPingTimer() {
    _pingTimer?.cancel();
    _pingTimer = Timer.periodic(const Duration(seconds: 30), (timer) {
      if (_isConnected && _wsChannel != null) {
        _sendWebSocketMessage({'type': 'ping'});
      } else {
        timer.cancel();
      }
    });
  }

  /// Create WebRTC offer and send to backend
  Future<void> _createAndSendOffer(int learnerAge) async {
    final offer = await _peerConnection!.createOffer({
      'offerToReceiveAudio': true,
      'offerToReceiveVideo': false,
    });
    
    await _peerConnection!.setLocalDescription(offer);
    
    _sendWebSocketMessage({
      'type': 'offer',
      'sdp': offer.sdp,
      'learnerAge': learnerAge,
      'sessionId': _sessionId,
    });
    
    _logger.i('[WebRTC] Offer sent');
  }

  /// Handle WebSocket messages (signaling)
  void _handleWebSocketMessage(dynamic message) {
    try {
      final data = jsonDecode(message);
      final type = data['type'] as String?;
      
      _logger.d('[WebSocket] Message received: $type');
      
      switch (type) {
        case 'connected':
          _handleConnected(data);
          break;
        case 'config':
          _handleConfig(data);
          break;
        case 'answer':
          _handleAnswer(data['sdp'] as String);
          break;
        case 'ice_candidate':
          _handleIceCandidate(data['candidate']);
          break;
        case 'transcript':
          _handleTranscript(data);
          break;
        case 'kelly_response':
          _handleKellyResponse(data);
          break;
        case 'visemes':
          _handleVisemes(data);
          break;
        case 'barge_in_confirmed':
          _handleBargeInConfirmed();
          break;
        case 'reconnected':
          _handleReconnected(data);
          break;
        case 'pong':
          // Keepalive response
          break;
        case 'error':
          _handleError(data);
          break;
        default:
          _logger.w('[WebSocket] Unknown message type: $type');
      }
    } catch (e) {
      _logger.e('[WebSocket] Error parsing message: $e');
    }
  }

  /// Handle WebRTC answer from server
  Future<void> _handleAnswer(String sdp) async {
    final answer = RTCSessionDescription(sdp, 'answer');
    await _peerConnection!.setRemoteDescription(answer);
    _logger.i('[WebRTC] Answer set');
  }

  /// Handle ICE candidate from server
  Future<void> _handleIceCandidate(Map<String, dynamic> candidateData) async {
    final candidate = RTCIceCandidate(
      candidateData['candidate'],
      candidateData['sdpMid'],
      candidateData['sdpMLineIndex'],
    );
    await _peerConnection!.addCandidate(candidate);
    _logger.d('[WebRTC] ICE candidate added');
  }

  /// Handle user transcript (speech-to-text)
  void _handleTranscript(Map<String, dynamic> data) {
    final text = data['text'] as String;
    final isFinal = data['isFinal'] as bool? ?? false;
    
    _logger.i('[Transcript] User: "$text" (final: $isFinal)');
    
    if (isFinal) {
      _transcriptController.add(text);
      onTranscriptReceived?.call(text);
      
      _emitEvent(RealtimeEvent(
        type: RealtimeEventType.userSpeech,
        timestamp: DateTime.now(),
        data: {'text': text, 'isFinal': true},
      ));
    }
  }

  /// Handle connection confirmation
  void _handleConnected(Map<String, dynamic> data) {
    final connectionId = data['connectionId'] as String?;
    _logger.i('[WebSocket] Connection confirmed: $connectionId');
  }
  
  /// Handle configuration from backend
  void _handleConfig(Map<String, dynamic> data) {
    final config = data['config'] as Map<String, dynamic>?;
    final kellyAge = data['kellyAge'] as int?;
    _logger.i('[WebSocket] Received configuration (Kelly age: $kellyAge)');
    
    _emitEvent(RealtimeEvent(
      type: RealtimeEventType.webrtcConnected,
      timestamp: DateTime.now(),
      data: {'config': config, 'kellyAge': kellyAge},
    ));
  }
  
  /// Handle Kelly's response
  void _handleKellyResponse(Map<String, dynamic> data) {
    final text = data['text'] as String;
    final audioBase64 = data['audio'] as String?;
    final latencyMs = data['latencyMs'] as int?;
    
    _logger.i('[Kelly] Response: "$text"');
    
    _isSpeaking = true;
    onKellyResponse?.call(text);
    
    if (audioBase64 != null) {
      final audioBytes = base64Decode(audioBase64);
      _audioOutputController.add(audioBytes);
      onAudioReceived?.call(audioBytes);
    }
    
    // Calculate latency
    if (latencyMs != null) {
      _latencyHistory.add(latencyMs);
      if (_latencyHistory.length > _maxLatencyHistory) {
        _latencyHistory.removeAt(0);
      }
      _totalLatencyMs += latencyMs;
      _requestCount++;
      onLatencyUpdate?.call(latencyMs);
      
      _logger.i('[Performance] RTT: ${latencyMs}ms (avg: ${averageLatencyMs.toStringAsFixed(0)}ms)');
    } else if (_lastRequestTime != null) {
      final latency = DateTime.now().difference(_lastRequestTime!).inMilliseconds;
      _latencyHistory.add(latency);
      if (_latencyHistory.length > _maxLatencyHistory) {
        _latencyHistory.removeAt(0);
      }
      _totalLatencyMs += latency;
      _requestCount++;
      onLatencyUpdate?.call(latency);
      
      _logger.i('[Performance] RTT: ${latency}ms (avg: ${averageLatencyMs.toStringAsFixed(0)}ms)');
    }
    
    _emitEvent(RealtimeEvent(
      type: RealtimeEventType.kellyResponse,
      timestamp: DateTime.now(),
      data: {'text': text, 'latencyMs': latencyMs ?? _totalLatencyMs / _requestCount},
    ));
  }
  
  /// Handle barge-in confirmation
  void _handleBargeInConfirmed() {
    _isSpeaking = false;
    _isListening = true;
    _logger.i('[OpenAI Realtime] Barge-in confirmed');
    
    _emitEvent(RealtimeEvent(
      type: RealtimeEventType.bargeIn,
      timestamp: DateTime.now(),
    ));
  }
  
  /// Handle reconnection confirmation
  void _handleReconnected(Map<String, dynamic> data) {
    _sessionId = data['sessionId'] as String?;
    _reconnectAttempts = 0;
    _isConnected = true;
    _logger.i('[OpenAI Realtime] Reconnected: $_sessionId');
    
    _emitEvent(RealtimeEvent(
      type: RealtimeEventType.connected,
      timestamp: DateTime.now(),
      data: {'reconnected': true, 'sessionId': _sessionId},
    ));
  }

  /// Handle visemes for lip-sync
  void _handleVisemes(Map<String, dynamic> data) {
    try {
      final visemes = data['visemes'] as Map<String, dynamic>?;
      if (visemes != null) {
        _logger.d('[Visemes] Received ${visemes.length} visemes');
        onVisemesReceived?.call(visemes as Map<String, dynamic>);
        
        _emitEvent(RealtimeEvent(
          type: RealtimeEventType.visemes,
          timestamp: DateTime.now(),
          data: {'visemes': visemes},
        ));
      }
    } catch (e) {
      _logger.e('[Visemes] Error handling visemes: $e');
    }
  }

  /// Handle errors
  void _handleError(Map<String, dynamic> data) {
    final error = data['message'] as String;
    _logger.e('[Error] $error');
    
    _emitEvent(RealtimeEvent(
      type: RealtimeEventType.error,
      timestamp: DateTime.now(),
      data: {'error': error},
    ));
  }

  /// Handle remote audio track (Kelly's voice)
  void _handleRemoteAudioTrack(MediaStream stream) {
    _logger.i('[Audio] Remote audio track active');
    _isSpeaking = true;
    
    _emitEvent(RealtimeEvent(
      type: RealtimeEventType.kellySpeaking,
      timestamp: DateTime.now(),
      data: {'speaking': true},
    ));
  }

  /// Start listening to user
  void startListening() {
    if (!_isConnected) {
      _logger.w('[OpenAI Realtime] Not connected, cannot start listening');
      return;
    }
    
    _isListening = true;
    _lastRequestTime = DateTime.now();
    
    _sendWebSocketMessage({'type': 'start_listening'});
    
    _logger.i('[OpenAI Realtime] Started listening');
    
    _emitEvent(RealtimeEvent(
      type: RealtimeEventType.listeningStarted,
      timestamp: DateTime.now(),
    ));
  }

  /// Stop listening to user
  void stopListening() {
    _isListening = false;
    
    _sendWebSocketMessage({'type': 'stop_listening'});
    
    _logger.i('[OpenAI Realtime] Stopped listening');
    
    _emitEvent(RealtimeEvent(
      type: RealtimeEventType.listeningStopped,
      timestamp: DateTime.now(),
    ));
  }

  /// Barge-in: Interrupt Kelly mid-speech
  void bargeIn() {
    if (!_isSpeaking) {
      _logger.w('[OpenAI Realtime] Kelly not speaking, cannot barge in');
      return;
    }
    
    _sendWebSocketMessage({'type': 'barge_in'});
    
    _isSpeaking = false;
    _isListening = true;
    
    _logger.i('[OpenAI Realtime] Barge-in triggered');
    
    _emitEvent(RealtimeEvent(
      type: RealtimeEventType.bargeIn,
      timestamp: DateTime.now(),
    ));
  }

  /// Send text message to Kelly (for testing or text-based input)
  void sendMessage(String text) {
    if (!_isConnected) {
      _logger.w('[OpenAI Realtime] Not connected, cannot send message');
      return;
    }
    
    _lastRequestTime = DateTime.now();
    
    _sendWebSocketMessage({
      'type': 'user_message',
      'text': text,
    });
    
    _logger.i('[OpenAI Realtime] Sent message: "$text"');
  }

  /// Send WebSocket message
  void _sendWebSocketMessage(Map<String, dynamic> message) {
    if (_wsChannel == null) return;
    
    final json = jsonEncode(message);
    _wsChannel!.sink.add(json);
  }

  /// Emit event to listeners
  void _emitEvent(RealtimeEvent event) {
    _eventController.add(event);
    onEvent?.call(event);
  }

  /// Disconnect and cleanup
  Future<void> disconnect() async {
    _logger.i('[OpenAI Realtime] Disconnecting...');
    
    _isConnected = false;
    _isListening = false;
    _isSpeaking = false;
    
    // Cancel timers
    _reconnectTimer?.cancel();
    _pingTimer?.cancel();
    
    // Close streams
    _localStream?.getTracks().forEach((track) => track.stop());
    await _localStream?.dispose();
    
    // Close peer connection
    await _peerConnection?.close();
    
    // Close WebSocket
    await _wsChannel?.sink.close();
    
    _emitEvent(RealtimeEvent(
      type: RealtimeEventType.disconnected,
      timestamp: DateTime.now(),
    ));
    
    _logger.i('[OpenAI Realtime] Disconnected');
  }
  
  /// Get current latency percentile (for performance monitoring)
  int getLatencyPercentile(double percentile) {
    if (_latencyHistory.isEmpty) return 0;
    final sorted = List<int>.from(_latencyHistory)..sort();
    final index = ((sorted.length - 1) * percentile).round();
    return sorted[index];
  }
  
  /// Check if latency is within target (<600ms)
  bool get isLatencyWithinTarget {
    if (_latencyHistory.isEmpty) return true;
    final p95 = getLatencyPercentile(0.95);
    return p95 < 600;
  }

  /// Dispose resources
  void dispose() {
    disconnect();
    _audioInputController.close();
    _audioOutputController.close();
    _transcriptController.close();
    _eventController.close();
  }
}

/// Realtime event types
enum RealtimeEventType {
  connected,
  disconnected,
  webrtcConnected,
  listeningStarted,
  listeningStopped,
  userSpeech,
  kellyResponse,
  kellySpeaking,
  bargeIn,
  visemes,
  error,
}

/// Realtime event model
class RealtimeEvent {
  final RealtimeEventType type;
  final DateTime timestamp;
  final Map<String, dynamic>? data;
  
  RealtimeEvent({
    required this.type,
    required this.timestamp,
    this.data,
  });
  
  @override
  String toString() => 'RealtimeEvent(type: $type, time: $timestamp, data: $data)';
}




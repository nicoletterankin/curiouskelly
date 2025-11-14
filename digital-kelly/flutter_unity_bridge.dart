import 'dart:convert';
import 'package:flutter/material.dart';
import 'package:flutter_unity_widget/flutter_unity_widget.dart';

/// Bridge for communicating between Flutter and Unity Kelly avatar
class FlutterUnityBridge {
  UnityWidgetController? _unityController;
  
  // Callbacks for Unity events
  Function(int kellyAge, int learnerAge)? onAvatarReady;
  Function(int kellyAge)? onAgeUpdated;
  Function(String lessonId, int kellyAge)? onLessonStarted;
  Function(Map<String, dynamic> stats)? onPerformanceStats;
  Function()? onPlaybackStopped;
  
  /// Initialize Unity widget controller
  void initialize(UnityWidgetController controller) {
    _unityController = controller;
    print('[FlutterUnityBridge] Initialized');
  }
  
  /// Handle messages from Unity
  void onUnityMessage(String message) {
    try {
      final data = jsonDecode(message);
      final type = data['type'] as String?;
      final payload = data['data'];
      
      print('[FlutterUnityBridge] Received from Unity: $type');
      
      switch (type) {
        case 'ready':
          onAvatarReady?.call(
            payload['kellyAge'] as int,
            payload['learnerAge'] as int,
          );
          break;
          
        case 'ageUpdated':
          onAgeUpdated?.call(payload['kellyAge'] as int);
          break;
          
        case 'lessonStarted':
          onLessonStarted?.call(
            payload['lessonId'] as String,
            payload['kellyAge'] as int,
          );
          break;
          
        case 'performanceStats':
          onPerformanceStats?.call(payload as Map<String, dynamic>);
          break;
          
        case 'stopped':
          onPlaybackStopped?.call();
          break;
          
        default:
          print('[FlutterUnityBridge] Unknown message type: $type');
      }
    } catch (e) {
      print('[FlutterUnityBridge] Error parsing message: $e');
    }
  }
  
  /// Send message to Unity
  void _sendToUnity(Map<String, dynamic> message) {
    if (_unityController == null) {
      print('[FlutterUnityBridge] ERROR: Unity controller not initialized');
      return;
    }
    
    try {
      final json = jsonEncode(message);
      _unityController!.postMessage(
        'UnityMessageManager',
        'ReceiveMessageFromFlutter',
        json,
      );
      print('[FlutterUnityBridge] Sent to Unity: ${message['type']}');
    } catch (e) {
      print('[FlutterUnityBridge] Error sending message: $e');
    }
  }
  
  // Public API for Flutter app
  
  /// Set learner's age (updates Kelly's appearance)
  void setLearnerAge(int age) {
    _sendToUnity({
      'type': 'setAge',
      'age': age,
    });
  }
  
  /// Play a lesson for specific age
  void playLesson(String lessonId, int age) {
    _sendToUnity({
      'type': 'playLesson',
      'lessonId': lessonId,
      'age': age,
    });
  }
  
  /// Speak text with TTS and lip-sync
  void speak(String text, int age) {
    _sendToUnity({
      'type': 'speak',
      'text': text,
      'age': age,
    });
  }
  
  /// Stop current playback
  void stop() {
    _sendToUnity({
      'type': 'stop',
    });
  }
  
  /// Request performance statistics
  void requestPerformanceStats() {
    _sendToUnity({
      'type': 'getPerformance',
    });
  }
  
  /// Set gaze target (normalized screen coordinates 0-1)
  void setGazeTarget(double x, double y) {
    _sendToUnity({
      'type': 'setGazeTarget',
      'x': x,
      'y': y,
    });
  }
}

/// Example Flutter widget using Kelly avatar
class KellyAvatarWidget extends StatefulWidget {
  final int learnerAge;
  final Function(FlutterUnityBridge)? onBridgeReady;
  
  const KellyAvatarWidget({
    Key? key,
    required this.learnerAge,
    this.onBridgeReady,
  }) : super(key: key);
  
  @override
  State<KellyAvatarWidget> createState() => _KellyAvatarWidgetState();
}

class _KellyAvatarWidgetState extends State<KellyAvatarWidget> {
  final FlutterUnityBridge _bridge = FlutterUnityBridge();
  bool _isReady = false;
  int _kellyAge = 27;
  String _status = 'Initializing...';
  
  @override
  void initState() {
    super.initState();
    
    // Setup callbacks
    _bridge.onAvatarReady = (kellyAge, learnerAge) {
      setState(() {
        _isReady = true;
        _kellyAge = kellyAge;
        _status = 'Ready';
      });
      print('[KellyAvatarWidget] Avatar ready: Kelly age $kellyAge, learner age $learnerAge');
    };
    
    _bridge.onAgeUpdated = (kellyAge) {
      setState(() {
        _kellyAge = kellyAge;
      });
      print('[KellyAvatarWidget] Age updated: Kelly is now $kellyAge');
    };
    
    _bridge.onPerformanceStats = (stats) {
      final fps = stats['avgFps'] as double;
      final status = stats['status'] as String;
      print('[KellyAvatarWidget] Performance: ${fps.toStringAsFixed(1)}fps - $status');
    };
  }
  
  void _onUnityCreated(UnityWidgetController controller) {
    _bridge.initialize(controller);
    widget.onBridgeReady?.call(_bridge);
    
    // Set initial age
    Future.delayed(const Duration(milliseconds: 500), () {
      _bridge.setLearnerAge(widget.learnerAge);
    });
  }
  
  void _onUnityMessage(String message) {
    _bridge.onUnityMessage(message);
  }
  
  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        // Unity view
        UnityWidget(
          onUnityCreated: _onUnityCreated,
          onUnityMessage: _onUnityMessage,
          fullscreen: false,
        ),
        
        // Status overlay
        Positioned(
          top: 10,
          left: 10,
          child: Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: Colors.black54,
              borderRadius: BorderRadius.circular(8),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Kelly Avatar',
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  'Status: $_status',
                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                ),
                Text(
                  'Kelly Age: $_kellyAge',
                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                ),
                Text(
                  'Learner Age: ${widget.learnerAge}',
                  style: const TextStyle(color: Colors.white70, fontSize: 12),
                ),
              ],
            ),
          ),
        ),
      ],
    );
  }
}

/// Example usage in app
class ExampleUsage extends StatefulWidget {
  const ExampleUsage({Key? key}) : super(key: key);
  
  @override
  State<ExampleUsage> createState() => _ExampleUsageState();
}

class _ExampleUsageState extends State<ExampleUsage> {
  FlutterUnityBridge? _bridge;
  int _learnerAge = 35;
  
  void _onBridgeReady(FlutterUnityBridge bridge) {
    setState(() {
      _bridge = bridge;
    });
    print('[Example] Bridge ready!');
  }
  
  void _playLesson() {
    _bridge?.playLesson('leaves-change-color', _learnerAge);
  }
  
  void _speak() {
    _bridge?.speak('Why do leaves change color in autumn?', _learnerAge);
  }
  
  void _changeAge(int newAge) {
    setState(() {
      _learnerAge = newAge;
    });
    _bridge?.setLearnerAge(newAge);
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Curious Kellly'),
      ),
      body: Column(
        children: [
          // Unity avatar view
          Expanded(
            flex: 2,
            child: KellyAvatarWidget(
              learnerAge: _learnerAge,
              onBridgeReady: _onBridgeReady,
            ),
          ),
          
          // Controls
          Expanded(
            flex: 1,
            child: Padding(
              padding: const EdgeInsets.all(16),
              child: Column(
                children: [
                  // Age slider
                  Row(
                    children: [
                      const Text('Learner Age:'),
                      Expanded(
                        child: Slider(
                          value: _learnerAge.toDouble(),
                          min: 2,
                          max: 102,
                          divisions: 100,
                          label: '$_learnerAge',
                          onChanged: (value) => _changeAge(value.toInt()),
                        ),
                      ),
                      Text('$_learnerAge'),
                    ],
                  ),
                  
                  const SizedBox(height: 16),
                  
                  // Buttons
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceEvenly,
                    children: [
                      ElevatedButton(
                        onPressed: _bridge != null ? _playLesson : null,
                        child: const Text('Play Lesson'),
                      ),
                      ElevatedButton(
                        onPressed: _bridge != null ? _speak : null,
                        child: const Text('Speak'),
                      ),
                      ElevatedButton(
                        onPressed: _bridge?.stop,
                        child: const Text('Stop'),
                      ),
                    ],
                  ),
                  
                  const SizedBox(height: 16),
                  
                  // Quick age presets
                  Wrap(
                    spacing: 8,
                    children: [
                      ElevatedButton(
                        onPressed: () => _changeAge(5),
                        child: const Text('Age 5'),
                      ),
                      ElevatedButton(
                        onPressed: () => _changeAge(10),
                        child: const Text('Age 10'),
                      ),
                      ElevatedButton(
                        onPressed: () => _changeAge(35),
                        child: const Text('Age 35'),
                      ),
                      ElevatedButton(
                        onPressed: () => _changeAge(70),
                        child: const Text('Age 70'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
      
      floatingActionButton: FloatingActionButton(
        onPressed: _bridge?.requestPerformanceStats,
        child: const Icon(Icons.analytics),
        tooltip: 'Performance Stats',
      ),
    );
  }
}
















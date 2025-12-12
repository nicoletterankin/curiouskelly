import 'dart:convert';
import 'dart:typed_data';

import 'package:flutter/material.dart';
import 'package:flutter_unity_widget/flutter_unity_widget.dart';

/// Bridge for sending commands between Flutter and the embedded Unity avatar.
class FlutterUnityBridge {
  UnityWidgetController? _controller;

  void initialize(UnityWidgetController controller) {
    _controller = controller;
  }

  void onUnityMessage(String message) {
    try {
      final payload = jsonDecode(message) as Map<String, dynamic>;
      final type = payload['type'] as String?;
      if (type == null) {
        return;
      }

      switch (type) {
        case 'stopped':
          onPlaybackStopped?.call();
          break;
        case 'performanceStats':
          final data = payload['data'] as Map<String, dynamic>?;
          if (data != null) {
            onPerformanceStats?.call(data);
          }
          break;
        case 'ready':
          final data = payload['data'] as Map<String, dynamic>?;
          if (data != null) {
            onAvatarReady?.call(
              data['kellyAge'] as int? ?? 27,
              data['learnerAge'] as int? ?? 27,
            );
          }
          break;
        case 'ageUpdated':
          final data = payload['data'] as Map<String, dynamic>?;
          if (data != null) {
            onAgeUpdated?.call(data['kellyAge'] as int? ?? 27);
          }
          break;
        default:
          break;
      }
    } catch (error) {
      debugPrint('Failed to parse Unity message: $error');
    }
  }

  void setLearnerAge(int age) {
    _sendMessage({'type': 'setAge', 'age': age});
  }

  void speak(String text, int age) {
    _sendMessage({'type': 'speak', 'text': text, 'age': age});
  }

  void playLesson(String lessonId, int age) {
    _sendMessage({'type': 'playLesson', 'lessonId': lessonId, 'age': age});
  }

  void stop() {
    _sendMessage({'type': 'stop'});
  }

  void requestPerformanceStats() {
    _sendMessage({'type': 'getPerformance'});
  }

  void setGazeTarget(double x, double y) {
    _sendMessage({'type': 'setGazeTarget', 'x': x, 'y': y});
  }

  /// Send visemes for lip-sync
  /// visemes: Map of viseme names to weights (0.0-1.0)
  void updateVisemes(Map<String, dynamic> visemes) {
    _sendMessage({
      'type': 'updateVisemes',
      'visemes': visemes,
    });
  }

  /// Send audio data for Unity to play
  /// audioBytes: PCM16 audio data
  void playAudio(Uint8List audioBytes) {
    // Encode audio bytes as base64 for transmission
    final base64Audio = base64Encode(audioBytes);
    _sendMessage({
      'type': 'playAudio',
      'audio': base64Audio,
      'format': 'pcm16',
    });
  }

  void _sendMessage(Map<String, dynamic> message) {
    final controller = _controller;
    if (controller == null) {
      return;
    }

    try {
      controller.postMessage(
        'UnityMessageManager',
        'ReceiveMessageFromFlutter',
        jsonEncode(message),
      );
    } catch (error) {
      debugPrint('Failed to send Unity message: $error');
    }
  }

  VoidCallback? onPlaybackStopped;
  void Function(Map<String, dynamic> stats)? onPerformanceStats;
  void Function(int kellyAge, int learnerAge)? onAvatarReady;
  void Function(int kellyAge)? onAgeUpdated;
}

/// Widget wrapper that hosts the Unity avatar view and exposes bridge lifecycle.
class KellyAvatarWidget extends StatefulWidget {
  const KellyAvatarWidget({
    super.key,
    required this.learnerAge,
    this.onBridgeReady,
  });

  final int learnerAge;
  final void Function(FlutterUnityBridge bridge)? onBridgeReady;

  @override
  State<KellyAvatarWidget> createState() => _KellyAvatarWidgetState();
}

class _KellyAvatarWidgetState extends State<KellyAvatarWidget> {
  final FlutterUnityBridge _bridge = FlutterUnityBridge();
  bool _isReady = false;
  int _kellyAge = 27;

  @override
  void initState() {
    super.initState();

    _bridge.onAvatarReady = (kellyAge, learnerAge) {
      setState(() {
        _isReady = true;
        _kellyAge = kellyAge;
      });
      debugPrint('Kelly avatar ready (Kelly age: $kellyAge, learner age: $learnerAge)');
    };

    _bridge.onAgeUpdated = (kellyAge) {
      setState(() => _kellyAge = kellyAge);
    };
  }

  void _onUnityCreated(UnityWidgetController controller) {
    _bridge.initialize(controller);
    widget.onBridgeReady?.call(_bridge);

    Future<void>.delayed(const Duration(milliseconds: 400), () {
      _bridge.setLearnerAge(widget.learnerAge);
    });
  }

  void _onUnityMessage(dynamic message) {
    if (message is String) {
      _bridge.onUnityMessage(message);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Stack(
      children: [
        UnityWidget(
          onUnityCreated: _onUnityCreated,
          onUnityMessage: _onUnityMessage,
          fullscreen: false,
        ),
        Positioned(
          top: 12,
          left: 12,
          child: _StatusPill(isReady: _isReady, kellyAge: _kellyAge),
        ),
      ],
    );
  }
}

class _StatusPill extends StatelessWidget {
  const _StatusPill({required this.isReady, required this.kellyAge});

  final bool isReady;
  final int kellyAge;

  @override
  Widget build(BuildContext context) {
    final color = isReady ? Colors.greenAccent : Colors.orangeAccent;
    final label = isReady ? 'Ready' : 'Loading';

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
      decoration: BoxDecoration(
        color: color.withOpacity(0.2),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(color: color, width: 1.5),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(isReady ? Icons.check_circle : Icons.autorenew, size: 16, color: color),
          const SizedBox(width: 8),
          Text(
            '$label · Kelly age $kellyAge',
            style: TextStyle(color: color, fontWeight: FontWeight.w600),
          ),
        ],
      ),
    );
  }
}




import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import 'dart:math' as math;
import '../controllers/voice_controller.dart';

/// Voice Visualizer
/// Shows audio energy as animated waveform
class VoiceVisualizer extends StatelessWidget {
  final double height;
  final Color color;
  
  const VoiceVisualizer({
    Key? key,
    this.height = 60.0,
    this.color = const Color(0xFF4CAF50),
  }) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    return Consumer<VoiceController>(
      builder: (context, voiceController, child) {
        final energy = voiceController.audioEnergy;
        final isActive = voiceController.isListening;
        
        return Container(
          height: height,
          padding: const EdgeInsets.symmetric(horizontal: 16),
          child: CustomPaint(
            painter: WaveformPainter(
              energy: energy,
              color: isActive ? color : Colors.grey,
              isActive: isActive,
            ),
            size: Size.infinite,
          ),
        );
      },
    );
  }
}

class WaveformPainter extends CustomPainter {
  final double energy;
  final Color color;
  final bool isActive;
  
  WaveformPainter({
    required this.energy,
    required this.color,
    required this.isActive,
  });
  
  @override
  void paint(Canvas canvas, Size size) {
    final paint = Paint()
      ..color = color
      ..strokeWidth = 3.0
      ..strokeCap = StrokeCap.round
      ..style = PaintingStyle.stroke;
    
    final centerY = size.height / 2;
    final barCount = 20;
    final barWidth = size.width / barCount;
    
    for (int i = 0; i < barCount; i++) {
      // Calculate bar height based on energy and position
      double barHeight;
      
      if (isActive) {
        // Animated waveform when active
        final phase = (i / barCount) * 2 * math.pi;
        final wave = math.sin(phase + DateTime.now().millisecondsSinceEpoch / 200);
        barHeight = (centerY * 0.6) * energy * (0.5 + wave * 0.5);
      } else {
        // Flat line when inactive
        barHeight = 2.0;
      }
      
      final x = i * barWidth + barWidth / 2;
      
      // Draw bar
      canvas.drawLine(
        Offset(x, centerY - barHeight / 2),
        Offset(x, centerY + barHeight / 2),
        paint,
      );
    }
  }
  
  @override
  bool shouldRepaint(WaveformPainter oldDelegate) {
    return energy != oldDelegate.energy ||
           isActive != oldDelegate.isActive;
  }
}

/// Voice Status Indicator
/// Shows current voice state with color coding
class VoiceStatusIndicator extends StatelessWidget {
  const VoiceStatusIndicator({Key? key}) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    return Consumer<VoiceController>(
      builder: (context, voiceController, child) {
        final state = voiceController.state;
        final color = _getColorForState(state);
        final description = state.description;
        
        return Container(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
          decoration: BoxDecoration(
            color: color.withOpacity(0.1),
            borderRadius: BorderRadius.circular(20),
            border: Border.all(color: color, width: 2),
          ),
          child: Row(
            mainAxisSize: MainAxisSize.min,
            children: [
              Container(
                width: 12,
                height: 12,
                decoration: BoxDecoration(
                  shape: BoxShape.circle,
                  color: color,
                ),
              ),
              const SizedBox(width: 8),
              Text(
                description,
                style: TextStyle(
                  color: color,
                  fontWeight: FontWeight.bold,
                  fontSize: 14,
                ),
              ),
            ],
          ),
        );
      },
    );
  }
  
  Color _getColorForState(VoiceState state) {
    switch (state) {
      case VoiceState.disconnected:
        return Colors.grey;
      case VoiceState.connecting:
        return Colors.orange;
      case VoiceState.connected:
      case VoiceState.idle:
        return Colors.blue;
      case VoiceState.listening:
      case VoiceState.userSpeaking:
        return Colors.green;
      case VoiceState.processing:
        return Colors.amber;
      case VoiceState.kellySpeaking:
        return Colors.purple;
      case VoiceState.error:
        return Colors.red;
    }
  }
}

/// Latency Indicator
/// Shows current and average latency
class LatencyIndicator extends StatelessWidget {
  const LatencyIndicator({Key? key}) : super(key: key);
  
  @override
  Widget build(BuildContext context) {
    return Consumer<VoiceController>(
      builder: (context, voiceController, child) {
        final latency = voiceController.latencyMs;
        final avgLatency = voiceController.averageLatencyMs;
        final color = _getColorForLatency(latency);
        
        return Container(
          padding: const EdgeInsets.all(8),
          decoration: BoxDecoration(
            color: Colors.black54,
            borderRadius: BorderRadius.circular(8),
          ),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                'Latency',
                style: TextStyle(
                  color: Colors.white70,
                  fontSize: 10,
                ),
              ),
              const SizedBox(height: 4),
              Row(
                children: [
                  Container(
                    width: 8,
                    height: 8,
                    decoration: BoxDecoration(
                      shape: BoxShape.circle,
                      color: color,
                    ),
                  ),
                  const SizedBox(width: 4),
                  Text(
                    '${latency}ms',
                    style: TextStyle(
                      color: Colors.white,
                      fontWeight: FontWeight.bold,
                      fontSize: 14,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 2),
              Text(
                'avg: ${avgLatency.toStringAsFixed(0)}ms',
                style: TextStyle(
                  color: Colors.white60,
                  fontSize: 10,
                ),
              ),
            ],
          ),
        );
      },
    );
  }
  
  Color _getColorForLatency(int latencyMs) {
    if (latencyMs < 300) return Colors.green;
    if (latencyMs < 600) return Colors.amber;
    return Colors.red;
  }
}























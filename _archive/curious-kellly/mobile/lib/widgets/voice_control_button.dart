import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../controllers/voice_controller.dart';

/// Voice Control Button
/// Main button for starting/stopping voice interaction
class VoiceControlButton extends StatefulWidget {
  final double size;
  final Color activeColor;
  final Color inactiveColor;
  
  const VoiceControlButton({
    Key? key,
    this.size = 80.0,
    this.activeColor = const Color(0xFF4CAF50),
    this.inactiveColor = const Color(0xFF9E9E9E),
  }) : super(key: key);
  
  @override
  State<VoiceControlButton> createState() => _VoiceControlButtonState();
}

class _VoiceControlButtonState extends State<VoiceControlButton>
    with SingleTickerProviderStateMixin {
  late AnimationController _pulseController;
  late Animation<double> _pulseAnimation;
  
  @override
  void initState() {
    super.initState();
    _pulseController = AnimationController(
      vsync: this,
      duration: const Duration(milliseconds: 1000),
    )..repeat(reverse: true);
    
    _pulseAnimation = Tween<double>(begin: 1.0, end: 1.2).animate(
      CurvedAnimation(parent: _pulseController, curve: Curves.easeInOut),
    );
  }
  
  @override
  void dispose() {
    _pulseController.dispose();
    super.dispose();
  }
  
  @override
  Widget build(BuildContext context) {
    return Consumer<VoiceController>(
      builder: (context, voiceController, child) {
        final state = voiceController.state;
        final isActive = state.isActive;
        final color = isActive ? widget.activeColor : widget.inactiveColor;
        
        return GestureDetector(
          onTap: () => _handleTap(voiceController),
          onLongPress: () => _handleLongPress(voiceController),
          child: AnimatedBuilder(
            animation: _pulseAnimation,
            builder: (context, child) {
              final scale = isActive ? _pulseAnimation.value : 1.0;
              
              return Transform.scale(
                scale: scale,
                child: Container(
                  width: widget.size,
                  height: widget.size,
                  decoration: BoxDecoration(
                    shape: BoxShape.circle,
                    color: color,
                    boxShadow: [
                      BoxShadow(
                        color: color.withOpacity(0.3),
                        blurRadius: 20,
                        spreadRadius: 5,
                      ),
                    ],
                  ),
                  child: Icon(
                    _getIcon(state),
                    size: widget.size * 0.5,
                    color: Colors.white,
                  ),
                ),
              );
            },
          ),
        );
      },
    );
  }
  
  IconData _getIcon(VoiceState state) {
    switch (state) {
      case VoiceState.disconnected:
      case VoiceState.error:
        return Icons.mic_off;
      case VoiceState.connecting:
        return Icons.sync;
      case VoiceState.listening:
      case VoiceState.userSpeaking:
        return Icons.mic;
      case VoiceState.processing:
        return Icons.hourglass_empty;
      case VoiceState.kellySpeaking:
        return Icons.volume_up;
      default:
        return Icons.mic_none;
    }
  }
  
  void _handleTap(VoiceController controller) {
    final state = controller.state;
    
    if (state == VoiceState.disconnected || state == VoiceState.error) {
      // Show connection dialog
      _showConnectDialog(controller);
    } else if (state == VoiceState.connected || state == VoiceState.idle) {
      // Start listening
      controller.startListening();
    } else if (state == VoiceState.listening || state == VoiceState.userSpeaking) {
      // Stop listening
      controller.stopListening();
    } else if (state == VoiceState.kellySpeaking) {
      // Barge-in
      controller.bargeIn();
    }
  }
  
  void _handleLongPress(VoiceController controller) {
    // Long press for manual text input
    _showTextInputDialog(controller);
  }
  
  void _showConnectDialog(VoiceController controller) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Connect to Kelly'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text('Enter your age to start learning with Kelly:'),
            const SizedBox(height: 16),
            TextField(
              keyboardType: TextInputType.number,
              decoration: const InputDecoration(
                labelText: 'Your Age (2-102)',
                border: OutlineInputBorder(),
              ),
              onSubmitted: (value) {
                final age = int.tryParse(value);
                if (age != null && age >= 2 && age <= 102) {
                  Navigator.pop(context);
                  _connectWithAge(controller, age);
                }
              },
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              _connectWithAge(controller, controller.learnerAge);
            },
            child: const Text('Connect'),
          ),
        ],
      ),
    );
  }
  
  Future<void> _connectWithAge(VoiceController controller, int age) async {
    controller.setLearnerAge(age);
    final success = await controller.connect(learnerAge: age);
    
    if (!success && mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        const SnackBar(content: Text('Failed to connect. Please try again.')),
      );
    }
  }
  
  void _showTextInputDialog(VoiceController controller) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Send Text Message'),
        content: TextField(
          autofocus: true,
          decoration: const InputDecoration(
            hintText: 'Type your message to Kelly...',
            border: OutlineInputBorder(),
          ),
          maxLines: 3,
          onSubmitted: (value) {
            if (value.isNotEmpty) {
              controller.sendMessage(value);
              Navigator.pop(context);
            }
          },
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
        ],
      ),
    );
  }
}























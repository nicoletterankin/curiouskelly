import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../controllers/voice_controller.dart';
import '../widgets/voice_control_button.dart';
import '../widgets/voice_visualizer.dart';
import '../flutter_unity_bridge.dart';

/// Main Conversation Screen
/// Full-screen Kelly avatar with voice interaction
class ConversationScreen extends StatefulWidget {
  final int learnerAge;
  
  const ConversationScreen({
    Key? key,
    required this.learnerAge,
  }) : super(key: key);
  
  @override
  State<ConversationScreen> createState() => _ConversationScreenState();
}

class _ConversationScreenState extends State<ConversationScreen> {
  FlutterUnityBridge? _unityBridge;
  List<ConversationMessage> _messages = [];
  
  @override
  void initState() {
    super.initState();
    _setupVoiceListener();
  }
  
  void _setupVoiceListener() {
    // Listen to voice controller updates
    final voiceController = context.read<VoiceController>();
    
    voiceController.addListener(() {
      // Add user messages
      if (voiceController.lastUserText != null &&
          (_messages.isEmpty || _messages.last.text != voiceController.lastUserText)) {
        setState(() {
          _messages.add(ConversationMessage(
            text: voiceController.lastUserText!,
            isUser: true,
            timestamp: DateTime.now(),
          ));
        });
      }
      
      // Add Kelly messages
      if (voiceController.lastKellyText != null &&
          (_messages.isEmpty || _messages.last.text != voiceController.lastKellyText)) {
        setState(() {
          _messages.add(ConversationMessage(
            text: voiceController.lastKellyText!,
            isUser: false,
            timestamp: DateTime.now(),
          ));
        });
        
        // Trigger Kelly to speak in Unity
        _unityBridge?.speak(voiceController.lastKellyText!, widget.learnerAge);
      }
    });
  }
  
  void _onUnityBridgeReady(FlutterUnityBridge bridge) {
    setState(() {
      _unityBridge = bridge;
    });
    
    // Pass bridge to VoiceController for viseme integration
    context.read<VoiceController>().setUnityBridge(bridge);
    
    // Set initial age
    bridge.setLearnerAge(widget.learnerAge);
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: SafeArea(
        child: Stack(
          children: [
            // Kelly Avatar (Unity)
            Positioned.fill(
              child: KellyAvatarWidget(
                learnerAge: widget.learnerAge,
                onBridgeReady: _onUnityBridgeReady,
              ),
            ),
            
            // Top Status Bar
            Positioned(
              top: 0,
              left: 0,
              right: 0,
              child: _buildStatusBar(),
            ),
            
            // Voice Visualizer (Middle)
            Positioned(
              left: 0,
              right: 0,
              bottom: 200,
              child: const VoiceVisualizer(
                height: 80,
              ),
            ),
            
            // Conversation History (Bottom)
            Positioned(
              left: 0,
              right: 0,
              bottom: 80,
              child: _buildConversationHistory(),
            ),
            
            // Voice Control Button (Center Bottom)
            Positioned(
              bottom: 20,
              left: 0,
              right: 0,
              child: Center(
                child: VoiceControlButton(
                  size: 80,
                ),
              ),
            ),
            
            // Barge-in Button (Right)
            Positioned(
              bottom: 40,
              right: 20,
              child: _buildBargeInButton(),
            ),
            
            // Settings Button (Top Right)
            Positioned(
              top: 10,
              right: 10,
              child: _buildSettingsButton(),
            ),
          ],
        ),
      ),
    );
  }
  
  Widget _buildStatusBar() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            Colors.black.withOpacity(0.8),
            Colors.transparent,
          ],
        ),
      ),
      child: Row(
        children: [
          const VoiceStatusIndicator(),
          const Spacer(),
          const LatencyIndicator(),
        ],
      ),
    );
  }
  
  Widget _buildConversationHistory() {
    if (_messages.isEmpty) {
      return const SizedBox.shrink();
    }
    
    return Container(
      height: 100,
      padding: const EdgeInsets.symmetric(horizontal: 16),
      child: ListView.builder(
        reverse: true,
        itemCount: _messages.length,
        itemBuilder: (context, index) {
          final message = _messages[_messages.length - 1 - index];
          return _buildMessageBubble(message);
        },
      ),
    );
  }
  
  Widget _buildMessageBubble(ConversationMessage message) {
    return Align(
      alignment: message.isUser ? Alignment.centerRight : Alignment.centerLeft,
      child: Container(
        margin: const EdgeInsets.only(bottom: 8),
        padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 8),
        decoration: BoxDecoration(
          color: message.isUser
              ? Colors.blue.withOpacity(0.7)
              : Colors.grey.withOpacity(0.7),
          borderRadius: BorderRadius.circular(16),
        ),
        child: Text(
          message.text,
          style: const TextStyle(
            color: Colors.white,
            fontSize: 14,
          ),
        ),
      ),
    );
  }
  
  Widget _buildBargeInButton() {
    return Consumer<VoiceController>(
      builder: (context, voiceController, child) {
        if (!voiceController.canBargeIn) {
          return const SizedBox.shrink();
        }
        
        return FloatingActionButton(
          onPressed: () => voiceController.bargeIn(),
          backgroundColor: Colors.orange,
          child: const Icon(Icons.front_hand),
          tooltip: 'Interrupt Kelly',
        );
      },
    );
  }
  
  Widget _buildSettingsButton() {
    return IconButton(
      icon: const Icon(Icons.settings, color: Colors.white),
      onPressed: () => _showSettings(),
    );
  }
  
  void _showSettings() {
    showModalBottomSheet(
      context: context,
      backgroundColor: Colors.grey[900],
      builder: (context) => Container(
        padding: const EdgeInsets.all(24),
        child: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Settings',
              style: TextStyle(
                color: Colors.white,
                fontSize: 24,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 24),
            _buildAgeSetting(),
            const SizedBox(height: 16),
            _buildConnectionStatus(),
            const SizedBox(height: 16),
            _buildClearButton(),
          ],
        ),
      ),
    );
  }
  
  Widget _buildAgeSetting() {
    return Consumer<VoiceController>(
      builder: (context, voiceController, child) {
        return Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              'Your Age: ${voiceController.learnerAge}',
              style: const TextStyle(color: Colors.white, fontSize: 16),
            ),
            Slider(
              value: voiceController.learnerAge.toDouble(),
              min: 2,
              max: 102,
              divisions: 100,
              label: '${voiceController.learnerAge}',
              onChanged: (value) {
                voiceController.setLearnerAge(value.toInt());
                _unityBridge?.setLearnerAge(value.toInt());
              },
            ),
          ],
        );
      },
    );
  }
  
  Widget _buildConnectionStatus() {
    return Consumer<VoiceController>(
      builder: (context, voiceController, child) {
        final isConnected = voiceController.isConnected;
        
        return Row(
          children: [
            Icon(
              isConnected ? Icons.wifi : Icons.wifi_off,
              color: isConnected ? Colors.green : Colors.red,
            ),
            const SizedBox(width: 8),
            Text(
              isConnected ? 'Connected' : 'Disconnected',
              style: TextStyle(
                color: isConnected ? Colors.green : Colors.red,
                fontSize: 16,
              ),
            ),
            const Spacer(),
            if (isConnected)
              ElevatedButton(
                onPressed: () => voiceController.disconnect(),
                style: ElevatedButton.styleFrom(backgroundColor: Colors.red),
                child: const Text('Disconnect'),
              )
            else
              ElevatedButton(
                onPressed: () => voiceController.connect(
                  learnerAge: voiceController.learnerAge,
                ),
                style: ElevatedButton.styleFrom(backgroundColor: Colors.green),
                child: const Text('Connect'),
              ),
          ],
        );
      },
    );
  }
  
  Widget _buildClearButton() {
    return SizedBox(
      width: double.infinity,
      child: ElevatedButton(
        onPressed: () {
          setState(() {
            _messages.clear();
          });
          Navigator.pop(context);
        },
        style: ElevatedButton.styleFrom(backgroundColor: Colors.grey[700]),
        child: const Text('Clear Conversation'),
      ),
    );
  }
}

/// Conversation message model
class ConversationMessage {
  final String text;
  final bool isUser;
  final DateTime timestamp;
  
  ConversationMessage({
    required this.text,
    required this.isUser,
    required this.timestamp,
  });
}




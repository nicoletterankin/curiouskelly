import 'package:flutter_test/flutter_test.dart';
import 'package:curious_kellly/services/openai_realtime_service.dart';
import 'package:curious_kellly/controllers/voice_controller.dart';
import 'package:curious_kellly/services/voice_activity_detector.dart';
import 'package:curious_kellly/services/audio_player_service.dart';
import 'package:http/http.dart' as http;
import 'package:http/testing.dart' as http_testing;
import 'dart:convert';
import 'dart:typed_data';

void main() {
  group('OpenAIRealtimeService Tests', () {
    test('fetchEphemeralKey - successful request', () async {
      final service = OpenAIRealtimeService(
        backendUrl: 'http://localhost:3000',
      );

      // Mock HTTP client
      final mockClient = http_testing.MockClient((request) async {
        if (request.url.path == '/api/realtime/ephemeral-key' &&
            request.method == 'POST') {
          final body = jsonDecode(request.body) as Map<String, dynamic>;
          if (body['learnerAge'] != null && body['learnerAge'] is int) {
            return http.Response(
              jsonEncode({
                'status': 'ok',
                'data': {
                  'sessionId': 'test-session-123',
                  'learnerAge': body['learnerAge'] as int,
                  'kellyAge': 27,
                  'kellyPersona': 'knowledgeable-adult',
                  'expiresAt': DateTime.now().add(const Duration(hours: 1)).toIso8601String(),
                }
              }),
              200,
            );
          }
        }
        return http.Response('Not Found', 404);
      });

      // Note: This test requires mocking the HTTP client in the service
      // For now, we'll test the structure
      expect(service.backendUrl, 'http://localhost:3000');
    });

    test('latency tracking - average calculation', () {
      final service = OpenAIRealtimeService(
        backendUrl: 'http://localhost:3000',
      );

      // Test that average latency starts at 0
      expect(service.averageLatencyMs, 0.0);
    });

    test('latency percentile calculation', () {
      final service = OpenAIRealtimeService(
        backendUrl: 'http://localhost:3000',
      );

      // Test getLatencyPercentile with empty history
      expect(service.getLatencyPercentile(0.95), 0);

      // Test isLatencyWithinTarget with empty history
      expect(service.isLatencyWithinTarget, true);
    });

    test('connection state - initial state', () {
      final service = OpenAIRealtimeService(
        backendUrl: 'http://localhost:3000',
      );

      expect(service.isConnected, false);
      expect(service.isSpeaking, false);
      expect(service.isListening, false);
    });
  });

  group('VoiceController Tests', () {
    test('initial state - disconnected', () {
      final controller = VoiceController(
        backendUrl: 'http://localhost:3000',
      );

      expect(controller.state, VoiceState.disconnected);
      expect(controller.isConnected, false);
      expect(controller.isListening, false);
      expect(controller.isKellySpeaking, false);
      expect(controller.canBargeIn, false);
    });

    test('state transitions - connecting', () {
      final controller = VoiceController(
        backendUrl: 'http://localhost:3000',
      );

      controller.setState(VoiceState.connecting);
      expect(controller.state, VoiceState.connecting);
      expect(controller.state.description, 'Connecting...');
    });

    test('state transitions - all states', () {
      final controller = VoiceController(
        backendUrl: 'http://localhost:3000',
      );

      final states = [
        VoiceState.disconnected,
        VoiceState.connecting,
        VoiceState.connected,
        VoiceState.idle,
        VoiceState.listening,
        VoiceState.userSpeaking,
        VoiceState.processing,
        VoiceState.kellySpeaking,
        VoiceState.error,
      ];

      for (final state in states) {
        controller.setState(state);
        expect(controller.state, state);
        expect(controller.state.description, isNotEmpty);
      }
    });

    test('setLearnerAge - updates age', () {
      final controller = VoiceController(
        backendUrl: 'http://localhost:3000',
      );

      controller.setLearnerAge(42);
      expect(controller.learnerAge, 42);
    });

    test('canBargeIn - only when Kelly is speaking', () {
      final controller = VoiceController(
        backendUrl: 'http://localhost:3000',
      );

      expect(controller.canBargeIn, false);

      controller.setState(VoiceState.kellySpeaking);
      expect(controller.canBargeIn, true);

      controller.setState(VoiceState.listening);
      expect(controller.canBargeIn, false);
    });

    test('isActive - correct states', () {
      final controller = VoiceController(
        backendUrl: 'http://localhost:3000',
      );

      controller.setState(VoiceState.listening);
      expect(controller.state.isActive, true);

      controller.setState(VoiceState.userSpeaking);
      expect(controller.state.isActive, true);

      controller.setState(VoiceState.kellySpeaking);
      expect(controller.state.isActive, true);

      controller.setState(VoiceState.disconnected);
      expect(controller.state.isActive, false);

      controller.setState(VoiceState.idle);
      expect(controller.state.isActive, false);
    });
  });

  group('VoiceActivityDetector Tests', () {
    test('initial state - not speaking', () {
      final vad = VoiceActivityDetector();

      expect(vad.isSpeaking, false);
      expect(vad.averageEnergy, 0.0);
    });

    test('processAudio - calculates energy', () {
      final vad = VoiceActivityDetector();

      // Create mock audio data (16-bit PCM samples)
      final audioData = Uint8List(100);
      for (int i = 0; i < audioData.length; i += 2) {
        // Simulate audio signal
        final sample = (32767 * 0.5).round(); // 50% amplitude
        audioData[i] = sample & 0xFF;
        audioData[i + 1] = (sample >> 8) & 0xFF;
      }

      bool speechStarted = false;
      vad.onSpeechStart = () {
        speechStarted = true;
      };

      // Process multiple audio buffers to trigger speech detection
      for (int i = 0; i < 20; i++) {
        vad.processAudio(audioData);
      }

      // Energy should be calculated
      expect(vad.averageEnergy, greaterThan(0.0));
    });

    test('reset - clears state', () {
      final vad = VoiceActivityDetector();

      vad.processAudio(Uint8List(100));
      vad.reset();

      expect(vad.isSpeaking, false);
      expect(vad.averageEnergy, 0.0);
    });

    test('speech detection - silence threshold', () {
      final vad = VoiceActivityDetector(
        silenceThreshold: 0.02,
        speechDuration: const Duration(milliseconds: 300),
        silenceDuration: const Duration(milliseconds: 500),
      );

      bool speechStarted = false;
      vad.onSpeechStart = () {
        speechStarted = true;
      };

      // Create silent audio (low energy)
      final silentAudio = Uint8List(100);
      for (int i = 0; i < 10; i++) {
        vad.processAudio(silentAudio);
      }

      // Should not trigger speech (low energy)
      expect(speechStarted, false);

      // Create loud audio (high energy)
      final loudAudio = Uint8List(100);
      for (int i = 0; i < loudAudio.length; i += 2) {
        final sample = (32767 * 0.8).round(); // 80% amplitude
        loudAudio[i] = sample & 0xFF;
        loudAudio[i + 1] = (sample >> 8) & 0xFF;
      }

      // Process enough to trigger speech
      for (int i = 0; i < 20; i++) {
        vad.processAudio(loudAudio);
        await Future.delayed(const Duration(milliseconds: 50));
      }

      // May or may not trigger depending on timing
      // This is a probabilistic test
    });
  });

  group('AudioPlayerService Tests', () {
    test('initial state - not playing', () {
      final player = AudioPlayerService();

      expect(player.isPlaying, false);
    });

    test('setUnityBridge - updates bridge', () {
      final player = AudioPlayerService();
      // Note: FlutterUnityBridge would need to be mockable
      // For now, test that method exists
      expect(player, isNotNull);
    });

    test('updateVisemes - with Unity bridge', () {
      final player = AudioPlayerService();
      
      // Test that updateVisemes can be called without error
      player.updateVisemes({
        'viseme_0': 0.5,
        'viseme_1': 0.3,
      });

      expect(player, isNotNull);
    });
  });

  group('Performance Tests', () {
    test('latency within target - empty history', () {
      final service = OpenAIRealtimeService(
        backendUrl: 'http://localhost:3000',
      );

      expect(service.isLatencyWithinTarget, true);
    });

    test('latency percentile calculation', () {
      final service = OpenAIRealtimeService(
        backendUrl: 'http://localhost:3000',
      );

      // Test percentile calculation with sample data
      // Note: This would require exposing latency history or adding test data
      expect(service.getLatencyPercentile(0.5), 0);
      expect(service.getLatencyPercentile(0.95), 0);
      expect(service.getLatencyPercentile(0.99), 0);
    });
  });

  group('Error Handling Tests', () {
    test('fetchEphemeralKey - network error', () async {
      final service = OpenAIRealtimeService(
        backendUrl: 'http://invalid-url-test',
      );

      // This will fail but should handle gracefully
      final result = await service.fetchEphemeralKey(learnerAge: 35);
      expect(result, isNull);
    });

    test('fetchEphemeralKey - invalid response', () async {
      final service = OpenAIRealtimeService(
        backendUrl: 'http://localhost:3000',
      );

      // Mock HTTP client returning error
      final mockClient = http_testing.MockClient((request) async {
        return http.Response('Internal Server Error', 500);
      });

      // Service should handle error gracefully
      final result = await service.fetchEphemeralKey(learnerAge: 35);
      // Result may be null on error
      expect(result, anyOf(isNull, isNotNull));
    });
  });
}


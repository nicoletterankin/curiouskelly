import 'package:flutter/material.dart';
import 'bridge/unity_view.dart';
import 'lessons/loader.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  runApp(const KellyApp());
}

class KellyApp extends StatelessWidget {
  const KellyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'Kelly OS',
      theme: ThemeData.dark(),
      home: const KellyHomePage(),
    );
  }
}

class KellyHomePage extends StatefulWidget {
  const KellyHomePage({super.key});

  @override
  State<KellyHomePage> createState() => _KellyHomePageState();
}

class _KellyHomePageState extends State<KellyHomePage> {
  UnityViewController? _unityController;

  @override
  void initState() {
    super.initState();
    _loadSampleLesson();
  }

  void _loadSampleLesson() async {
    try {
      final lesson = await LessonLoader.loadSampleLesson();
      print('✅ Kelly OS: Loaded lesson "${lesson.title}"');
    } catch (e) {
      print('⚠️  Kelly OS: Could not load sample lesson: $e');
    }
  }

  void _onUnityCreated(UnityViewController controller) {
    _unityController = controller;
  }

  void _playTest() {
    if (_unityController == null) {
      print('⚠️  Kelly OS: Unity controller not ready');
      return;
    }

    // Send test message to Unity
    final jsonPath = 'assets/kelly_a2f_cache.json';
    final wavPath = '~/DigitalKellyTest/audio/kelly_intro.wav';
    _unityController!.postMessage(
      'KellyController',
      'LoadAndPlay',
      '$jsonPath|$wavPath',
    );
    print('📨 Kelly OS: Sent play message to Unity');
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: Colors.black,
      body: Stack(
        children: [
          // Unity view fills screen
          UnityView(
            onCreated: _onUnityCreated,
          ),
          // Play Test FAB in top-right
          Positioned(
            top: 60,
            right: 24,
            child: FloatingActionButton(
              onPressed: _playTest,
              backgroundColor: Colors.blue.shade800,
              child: const Text(
                'Play\nTest',
                textAlign: TextAlign.center,
                style: TextStyle(fontSize: 12, color: Colors.white),
              ),
            ),
          ),
        ],
      ),
    );
  }
}



























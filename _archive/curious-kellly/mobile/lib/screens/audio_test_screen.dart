import 'package:flutter/material.dart';
import '../services/lesson_audio_player.dart';

/// Audio Test Screen
/// Test lesson audio playback with generated files
class AudioTestScreen extends StatefulWidget {
  const AudioTestScreen({Key? key}) : super(key: key);
  
  @override
  State<AudioTestScreen> createState() => _AudioTestScreenState();
}

class _AudioTestScreenState extends State<AudioTestScreen> {
  final LessonAudioPlayer _audioPlayer = LessonAudioPlayer();
  
  String _selectedLesson = 'water-cycle';
  String _selectedAge = '18-35';
  String _selectedSection = 'welcome';
  
  bool _isPlaying = false;
  Duration _position = Duration.zero;
  Duration _duration = Duration.zero;
  
  final List<String> _lessons = ['water-cycle', 'leaves-change-color'];
  final List<String> _ageGroups = ['2-5', '6-12', '13-17', '18-35', '36-60', '61-102'];
  final List<String> _sections = ['welcome', 'mainContent', 'wisdomMoment'];
  
  @override
  void initState() {
    super.initState();
    _setupCallbacks();
  }
  
  void _setupCallbacks() {
    _audioPlayer.onPlaybackStarted = () {
      setState(() => _isPlaying = true);
    };
    
    _audioPlayer.onPlaybackComplete = () {
      setState(() => _isPlaying = false);
    };
    
    _audioPlayer.onProgress = (position, duration) {
      setState(() {
        _position = position;
        _duration = duration;
      });
    };
  }
  
  void _playAudio() async {
    // For testing, we'll try to play from a local file path
    // In production, this would download from backend
    
    final localPath = 'C:\\Users\\user\\UI-TARS-desktop\\curious-kellly\\backend\\config\\audio\\$_selectedLesson\\${_selectedAge}-${_selectedSection}.mp3';
    
    final success = await _audioPlayer.playLocalFile(localPath);
    
    if (!success && mounted) {
      ScaffoldMessenger.of(context).showSnackBar(
        SnackBar(content: Text('Failed to play audio. File not found: $localPath')),
      );
    }
  }
  
  void _playCompleteLesson() async {
    setState(() => _isPlaying = true);
    
    for (final section in _sections) {
      setState(() => _selectedSection = section);
      
      final localPath = 'C:\\Users\\user\\UI-TARS-desktop\\curious-kellly\\backend\\config\\audio\\$_selectedLesson\\${_selectedAge}-$section.mp3';
      
      final success = await _audioPlayer.playLocalFile(localPath);
      
      if (success) {
        // Wait for section to complete
        while (_audioPlayer.isPlaying) {
          await Future.delayed(const Duration(milliseconds: 100));
        }
      }
    }
    
    setState(() => _isPlaying = false);
  }
  
  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Lesson Audio Test'),
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.stretch,
          children: [
            _buildInfoCard(),
            const SizedBox(height: 24),
            _buildLessonSelector(),
            const SizedBox(height: 16),
            _buildAgeSelector(),
            const SizedBox(height: 16),
            _buildSectionSelector(),
            const SizedBox(height: 24),
            _buildPlaybackControls(),
            const SizedBox(height: 24),
            _buildProgressBar(),
            const SizedBox(height: 24),
            _buildQuickTests(),
          ],
        ),
      ),
    );
  }
  
  Widget _buildInfoCard() {
    return Card(
      color: Colors.blue.shade50,
      child: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              '🎙️ Audio Test',
              style: TextStyle(fontSize: 20, fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            Text(
              'Testing lesson audio from:\n${_audioPlayer.currentLesson['lessonId'] ?? 'None'}',
              style: const TextStyle(fontSize: 14),
            ),
          ],
        ),
      ),
    );
  }
  
  Widget _buildLessonSelector() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Lesson:', style: TextStyle(fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        DropdownButton<String>(
          value: _selectedLesson,
          isExpanded: true,
          items: _lessons.map((lesson) {
            return DropdownMenuItem(value: lesson, child: Text(lesson));
          }).toList(),
          onChanged: (value) {
            if (value != null) {
              setState(() => _selectedLesson = value);
            }
          },
        ),
      ],
    );
  }
  
  Widget _buildAgeSelector() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Age Group:', style: TextStyle(fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        Wrap(
          spacing: 8,
          children: _ageGroups.map((age) {
            final isSelected = age == _selectedAge;
            return ChoiceChip(
              label: Text(age),
              selected: isSelected,
              onSelected: (selected) {
                if (selected) {
                  setState(() => _selectedAge = age);
                }
              },
            );
          }).toList(),
        ),
      ],
    );
  }
  
  Widget _buildSectionSelector() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        const Text('Section:', style: TextStyle(fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        SegmentedButton<String>(
          segments: _sections.map((section) {
            return ButtonSegment(
              value: section,
              label: Text(section),
            );
          }).toList(),
          selected: {_selectedSection},
          onSelectionChanged: (Set<String> selection) {
            setState(() => _selectedSection = selection.first);
          },
        ),
      ],
    );
  }
  
  Widget _buildPlaybackControls() {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceEvenly,
      children: [
        ElevatedButton.icon(
          onPressed: _isPlaying ? null : _playAudio,
          icon: const Icon(Icons.play_arrow),
          label: const Text('Play Section'),
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.green,
            foregroundColor: Colors.white,
          ),
        ),
        ElevatedButton.icon(
          onPressed: _isPlaying ? _audioPlayer.pause : null,
          icon: const Icon(Icons.pause),
          label: const Text('Pause'),
        ),
        ElevatedButton.icon(
          onPressed: _audioPlayer.stop,
          icon: const Icon(Icons.stop),
          label: const Text('Stop'),
          style: ElevatedButton.styleFrom(
            backgroundColor: Colors.red,
            foregroundColor: Colors.white,
          ),
        ),
      ],
    );
  }
  
  Widget _buildProgressBar() {
    final progressPercent = _duration.inMilliseconds > 0
        ? _position.inMilliseconds / _duration.inMilliseconds
        : 0.0;
    
    return Column(
      children: [
        LinearProgressIndicator(
          value: progressPercent,
          backgroundColor: Colors.grey.shade300,
          valueColor: const AlwaysStoppedAnimation<Color>(Colors.blue),
        ),
        const SizedBox(height: 8),
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(_formatDuration(_position)),
            Text(_formatDuration(_duration)),
          ],
        ),
      ],
    );
  }
  
  Widget _buildQuickTests() {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.stretch,
      children: [
        const Text('Quick Tests:', style: TextStyle(fontWeight: FontWeight.bold)),
        const SizedBox(height: 8),
        ElevatedButton(
          onPressed: _isPlaying ? null : _playCompleteLesson,
          child: const Text('Play Complete Lesson (All 3 Sections)'),
        ),
        const SizedBox(height: 8),
        ElevatedButton(
          onPressed: _isPlaying ? null : () {
            setState(() {
              _selectedLesson = 'water-cycle';
              _selectedAge = '2-5';
              _selectedSection = 'welcome';
            });
            _playAudio();
          },
          child: const Text('Test Toddler Kelly (age 3)'),
        ),
        ElevatedButton(
          onPressed: _isPlaying ? null : () {
            setState(() {
              _selectedLesson = 'water-cycle';
              _selectedAge = '61-102';
              _selectedSection = 'welcome';
            });
            _playAudio();
          },
          child: const Text('Test Elder Kelly (age 82)'),
        ),
      ],
    );
  }
  
  String _formatDuration(Duration duration) {
    final minutes = duration.inMinutes;
    final seconds = duration.inSeconds % 60;
    return '${minutes.toString().padLeft(2, '0')}:${seconds.toString().padLeft(2, '0')}';
  }
  
  @override
  void dispose() {
    _audioPlayer.dispose();
    super.dispose();
  }
}






















